"""Case-document corpus and evidence-checked LLM extraction for the demo."""

import hashlib
import json
from pathlib import Path
from threading import Lock

from omegaconf import OmegaConf

from vrscore.documents.docx import DocxReader
from vrscore.feature_extraction.vacancy import FeatureVocabulary
from vrscore.llm import ExtractionPrompts

from .extraction import build_extraction_model, source_reference, valid_entity
from .postprocessing import apply_rules

CASE_FILES = (
    ("frontend", "Вакансия 1.docx"),
    ("accountant", "Вакансия 2.docx"),
    ("curator", "Вакансия 3.docx"),
)
RESUME_FILES = (
    ("r1", "Иванов Алексей_резюме 1.docx", "Иванов Алексей", "Frontend-разработчик"),
    ("r2", "Петров Иван_резюме 2.docx", "Петров Иван", "Frontend-разработчик"),
    ("r3", "Петрова Лилия_резюме 4.docx", "Петрова Лилия", "Бухгалтер"),
    ("r4", "Семенов Алексей _резюме 3.docx", "Семенов Алексей", "Frontend-разработчик"),
)

# Only the exact files attested by the user may be sent to the external API.
APPROVED_HASHES = {
    "Вакансия 1.docx": (
        "005eccea40e3c634b572c4139d4814c6348137deda950af4b652f753413924b4"
    ),
    "Вакансия 2.docx": (
        "c512b0c50ee084d0886a159adc529cddfb98dc6a30514ac51295043a377d890e"
    ),
    "Вакансия 3.docx": (
        "e22f67137b170a7606ba891ac4f90f24ff5073694bca2c554561476895b5bd87"
    ),
    "Иванов Алексей_резюме 1.docx": (
        "cc0246f5f41b0689c65b8753fb1079c4eff966603792d48c3011b15433ec6c6f"
    ),
    "Петров Иван_резюме 2.docx": (
        "598a5c7f1ec69f455de38218ee07793f70e09e4db8c26a954003dc410bf084fc"
    ),
    "Петрова Лилия_резюме 4.docx": (
        "ae210a7f2b59ad6e6f3ba3999bd79d294735654a50b93a77ec9f018bd7b507d9"
    ),
    "Семенов Алексей _резюме 3.docx": (
        "d12ab658081db26f2dca21a71630729dc4df247a25615d802ddebd1c5350d0f2"
    ),
}


def load_case_documents(
    directory: Path, *, documents=None
) -> tuple[list[dict], list[dict]]:
    """Read the seven explicitly approved case DOCX files without truncation."""
    reader = DocxReader()

    def read(filename: str) -> dict:
        path = directory / filename
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != APPROVED_HASHES[filename]:
            raise ValueError("Document differs from approved public version")
        document = reader.read(data, filename=filename)
        if documents is not None:
            documents[document.document_id] = document
        return {
            "text": document.text,
            "document_id": document.document_id,
            "text_version": document.text_version,
            "warnings": list(document.warnings),
        }

    vacancies = []
    for identifier, filename in CASE_FILES:
        document = read(filename)
        title = next(
            line.strip() for line in document["text"].splitlines() if line.strip()
        )
        vacancies.append({"id": identifier, "title": title, **document})
    resumes = [
        {"id": identifier, "name": name, "title": title, **read(filename)}
        for identifier, filename, name, title in RESUME_FILES
    ]
    return vacancies, resumes


class DemoService:
    """Cache validated extractions in memory for the approved public case files."""

    def __init__(
        self,
        provider,
        *,
        prompts=None,
        vocabulary=None,
        labels=None,
        postprocessing=None,
        parsed_dir=None,
    ):
        self.provider = provider
        project = Path(__file__).resolve().parents[2]
        if prompts is None:
            prompts = OmegaConf.to_container(
                OmegaConf.load(project / "configs" / "llm.yaml").prompts.extraction,
                resolve=True,
            )
        if vocabulary is None or labels is None or postprocessing is None:
            defaults = OmegaConf.load(project / "configs" / "feature_extraction.yaml")
            if vocabulary is None:
                vocabulary = OmegaConf.to_container(defaults.vocabulary, resolve=True)
            if labels is None:
                labels = OmegaConf.to_container(defaults.labels, resolve=True)
            if postprocessing is None:
                postprocessing = OmegaConf.to_container(
                    defaults.postprocessing, resolve=True
                )
        self.prompts = ExtractionPrompts.model_validate(prompts)
        self.vocabulary = FeatureVocabulary.model_validate(vocabulary)
        self.labels = dict(labels)
        self.postprocessing_rules = list(postprocessing["rules"])
        self.output_model = build_extraction_model(self.vocabulary)
        self.catalog_json = json.dumps(
            self.vocabulary.model_dump(), ensure_ascii=False, sort_keys=True
        )
        root = project / "public_data" / "raw"
        self.documents = {}
        self.vacancies, self.resumes = load_case_documents(
            root, documents=self.documents
        )
        self.parsed_dir = (
            Path(parsed_dir) if parsed_dir else project / "public_data" / "parsed"
        )
        self.parsed = self._load_parsed()
        self.cache = {}
        self.lock = Lock()

    def _load_parsed(self):
        """Load only well-formed per-document NER outputs from the parsed folder."""
        parsed = {}
        if not self.parsed_dir.is_dir():
            return parsed
        for path in self.parsed_dir.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("id") == path.stem:
                parsed[path.stem] = value
        return parsed

    def catalog(self):
        """Return source metadata and any saved grounded extraction results."""
        return {
            "vacancies": [
                {**item, **self.parsed.get(item["id"], {})} for item in self.vacancies
            ],
            "resumes": self.resumes,
            "cached": len(self.cache),
        }

    def extract_document(self, document, *, kind):
        """Extract, ground, and optionally postprocess one vacancy or resume."""
        if kind not in {"vacancy", "resume"}:
            raise ValueError("Unknown document kind")
        prompt = getattr(self.prompts, kind)
        result = self.provider.structured(
            [
                {"role": "system", "content": self.prompts.system + "\n" + prompt},
                {
                    "role": "user",
                    "content": self.prompts.user_intro
                    + "\n"
                    + self.catalog_json
                    + "\n<source_document>\n"
                    + document["text"]
                    + "\n</source_document>",
                },
            ],
            self.output_model,
            public_data=True,
        )
        result = self.output_model.model_validate(result.model_dump())
        entities = []
        rejected_quotes = 0
        rejected_values = 0
        for entity in result.entities:
            citation = source_reference(
                self.documents[document["document_id"]], entity.quote
            )
            if citation is None:
                rejected_quotes += 1
                continue
            if not valid_entity(entity, self.vocabulary, kind=kind):
                rejected_values += 1
                continue
            category = self.vocabulary.categories[entity.category]
            display_value = (
                category.values.get(entity.value, entity.value)
                if isinstance(entity.value, str)
                else entity.value
            )
            entities.append(
                {
                    "label": self.labels.get(entity.category, entity.category),
                    "category": entity.category,
                    "value": entity.value,
                    "display_value": display_value,
                    "polarity": entity.polarity,
                    "importance": entity.importance,
                    "context": entity.context,
                    "document_id": document["document_id"],
                    "text_version": document["text_version"],
                    **citation,
                }
            )
        rejected = rejected_quotes + rejected_values
        if rejected and not entities:
            raise ValueError("LLM quote or value validation failed")
        if kind == "resume":
            entities = apply_rules(entities, self.postprocessing_rules)
        return {
            "id": document["id"],
            "kind": kind,
            "document_id": document["document_id"],
            "text_version": document["text_version"],
            "entities": entities,
            "rejected": rejected,
            "rejected_quotes": rejected_quotes,
            "rejected_values": rejected_values,
        }

    def extract_all(self):
        """Run actual provider extraction on all approved files and persist JSON."""
        results = {"vacancies": [], "resumes": []}
        for kind, documents in (("vacancy", self.vacancies), ("resume", self.resumes)):
            for document in documents:
                result = self.extract_document(document, kind=kind)
                self.parsed[document["id"]] = result
                self.parsed_dir.mkdir(parents=True, exist_ok=True)
                destination = self.parsed_dir / f"{document['id']}.json"
                temporary = destination.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(destination)
                results["vacancies" if kind == "vacancy" else "resumes"].append(result)
        return results

    def analyze(self, vacancy_id, resume_id=None):
        """Extract uncached resumes and refresh the selected one on every run."""
        if not any(v["id"] == vacancy_id for v in self.vacancies):
            raise ValueError("Unknown vacancy")
        if resume_id is not None and not any(
            r["id"] == resume_id for r in self.resumes
        ):
            raise ValueError("Unknown resume")
        selected = resume_id or self.resumes[0]["id"]
        with self.lock:
            for resume in self.resumes:
                if resume["id"] != selected and resume["id"] in self.cache:
                    continue
                self.cache[resume["id"]] = self.extract_document(resume, kind="resume")
            # The positions are illustrative, not a matching algorithm.
            order = {
                "frontend": ["r2", "r4", "r1", "r3"],
                "accountant": ["r3", "r1", "r2", "r4"],
                "curator": ["r1", "r2", "r3", "r4"],
            }[vacancy_id]
            by_id = {r["id"]: r for r in self.resumes}
            return {
                "mock": True,
                "resumes": [
                    {**by_id[rid], **self.cache[rid], "rank": i + 1}
                    for i, rid in enumerate(order)
                ],
            }
