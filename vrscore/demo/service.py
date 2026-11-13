"""Case-document corpus and evidence-checked LLM extraction for the demo."""

import hashlib
import json
import logging
import re
from pathlib import Path
from threading import Lock

from omegaconf import OmegaConf

from vrscore.documents.docx import DocxReader
from vrscore.feature_extraction.vacancy import FeatureVocabulary
from vrscore.gates import apply_vacancy_defaults, evaluate_first_gate
from vrscore.llm import ExtractionPrompts

from .extraction import (
    Entity,
    build_extraction_model,
    build_vacancy_model,
    source_reference,
    valid_entity,
)
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
    (
        "synthetic_r5",
        "synthetic_r5.docx",
        "Алина Воронцова",
        "Ведущий специалист по закупкам (223-ФЗ)",
    ),
    (
        "synthetic_r6",
        "synthetic_r6.docx",
        "Тимур Беляев",
        "Junior Fullstack-разработчик",
    ),
    (
        "synthetic_r7",
        "synthetic_r7.docx",
        "Марина Сафонова",
        "Биостатистик / биоинформатик",
    ),
    (
        "synthetic_r8",
        "synthetic_r8.docx",
        "Денис Климов",
        "Инженер по робототехнике: автономное движение платформы",
    ),
    (
        "synthetic_r9",
        "synthetic_r9.docx",
        "Екатерина Миронова",
        "Менеджер научных проектов",
    ),
    (
        "synthetic_r10",
        "synthetic_r10.docx",
        "Илья Громов",
        "Младший научный сотрудник (молекулярный биолог)",
    ),
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
    "synthetic_r5.docx": (
        "da1b0634894aefe0217ceb188d3f1cfc8d622f9f095d1281816f310e2b8054f0"
    ),
    "synthetic_r6.docx": (
        "a56e6a6c596e24fd73248da8cc06d71e2616a7f9bc13d497ca428960f7b0d76b"
    ),
    "synthetic_r7.docx": (
        "ed9eee6734b966ede904937a4fef232f039d2fed00a8db36ddc82aa75f13eb71"
    ),
    "synthetic_r8.docx": (
        "0eb56dcce2d4c0f526e55514d9bf44350ca9ca8b9f8811c75fe9753f357d1a5b"
    ),
    "synthetic_r9.docx": (
        "da1d3bba602485b2045e994d65df16418c2e9c0b22dbabaf0d3b73b0493f6977"
    ),
    "synthetic_r10.docx": (
        "2b361727ac1d46eacfa8f79cc431404a05dab6a6046a7bfe670150c52f224acc"
    ),
}


def load_case_documents(
    directory: Path, *, documents=None
) -> tuple[list[dict], list[dict]]:
    """Read the explicitly approved case DOCX files without truncation."""
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
        {
            "id": identifier,
            "name": name,
            "title": title,
            **({"synthetic": True} if identifier.startswith("synthetic_") else {}),
            **read(filename),
        }
        for identifier, filename, name, title in RESUME_FILES
    ]
    return vacancies, resumes


def normalize_gate_evidence(entities):
    """Reject numeric claims lacking cited duration; correct explicit degrees."""
    normalized = []
    for original in entities:
        entity = dict(original)
        quote = entity["evidence"].casefold()
        if entity.get("category") == "experience":
            try:
                value = float(entity["value"])
            except (KeyError, TypeError, ValueError):
                continue
            numbers = re.findall(
                r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:х\s*)?"
                r"(?:[–—-]\s*\d+\s*)?(?:лет|год(?:а|ов)?)\b",
                quote,
            )
            explicit_zero = value == 0 and bool(
                re.search(r"без опыта|опыт (?:работы )?не требуется", quote)
            )
            if not explicit_zero and value not in [
                float(number.replace(",", ".")) for number in numbers
            ]:
                continue
        if entity.get("category") == "education" and re.search(
            r"кандидат[а-яё]*\s+(?:[а-яё-]+\s+){0,3}наук", quote
        ):
            entity["original_value"] = entity.get("value")
            entity["value"] = "candidate_of_sciences"
            entity["display_value"] = "Кандидат наук"
        normalized.append(entity)
    return normalized


def load_extra_vacancies(path):
    """Load preprocessed vacancies and verify their literal source evidence."""
    if not path:
        return []
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Vacancy corpus not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Vacancy corpus must be a list")
    identifiers = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or record.get("kind") != "vacancy"
            or not isinstance(record.get("id"), str)
            or not record["id"]
            or record["id"] in identifiers
            or not isinstance(record.get("title"), str)
            or not isinstance(record.get("text"), str)
            or not isinstance(record.get("entities"), list)
        ):
            raise ValueError("Invalid vacancy corpus record")
        identifiers.add(record["id"])
        for entity in record["entities"]:
            if not isinstance(entity, dict):
                raise ValueError("Invalid vacancy evidence")
            start, end = entity.get("start"), entity.get("end")
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(record["text"])
                or record["text"][start:end] != entity.get("evidence")
            ):
                raise ValueError("Invalid vacancy evidence offsets or text")
        record["entities"] = normalize_gate_evidence(record["entities"])
    return records


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
        vacancy_template=None,
        gate_config=None,
        parsed_dir=None,
        cache_dir=None,
        extra_vacancies_path=None,
        matching_gt_path=None,
    ):
        self.provider = provider
        self.gate_config = gate_config
        project = Path(__file__).resolve().parents[2]
        if prompts is None:
            prompts = OmegaConf.to_container(
                OmegaConf.load(project / "configs" / "llm.yaml").prompts.extraction,
                resolve=True,
            )
        if (
            vocabulary is None
            or labels is None
            or postprocessing is None
            or vacancy_template is None
        ):
            defaults = OmegaConf.load(project / "configs" / "feature_extraction.yaml")
            if vocabulary is None:
                vocabulary = OmegaConf.to_container(defaults.vocabulary, resolve=True)
            if labels is None:
                labels = OmegaConf.to_container(defaults.labels, resolve=True)
            if postprocessing is None:
                postprocessing = OmegaConf.to_container(
                    defaults.postprocessing, resolve=True
                )
            if vacancy_template is None:
                vacancy_template = OmegaConf.to_container(
                    defaults.vacancy_template, resolve=True
                )
        self.prompts = ExtractionPrompts.model_validate(prompts)
        self.vocabulary = FeatureVocabulary.model_validate(vocabulary)
        self.labels = dict(labels)
        self.postprocessing_rules = list(postprocessing["rules"])
        self.vacancy_postprocessing_rules = list(
            postprocessing.get("vacancy_rules", [])
        )
        self.output_model = build_extraction_model(self.vocabulary)
        self.vacancy_template = dict(vacancy_template)
        self.vacancy_model = build_vacancy_model(self.vocabulary, self.vacancy_template)
        self.catalog_json = json.dumps(
            self.vocabulary.model_dump(), ensure_ascii=False, sort_keys=True
        )
        self.vacancy_catalog_json = json.dumps(
            {"categories": self.vocabulary.model_dump()["categories"]},
            ensure_ascii=False,
            sort_keys=True,
        )
        root = project / "public_data" / "raw"
        self.documents = {}
        self.vacancies, self.resumes = load_case_documents(
            root, documents=self.documents
        )
        self.extra_vacancies = load_extra_vacancies(extra_vacancies_path)
        if {v["id"] for v in self.extra_vacancies} & {v["id"] for v in self.vacancies}:
            raise ValueError("Duplicate vacancy ID across corpora")
        self.matching_gt_path = Path(matching_gt_path) if matching_gt_path else None
        self.matching_reference = (
            self._load_matching_reference(self.matching_gt_path)
            if self.matching_gt_path and self.matching_gt_path.exists()
            else None
        )
        self.parsed_dir = (
            Path(parsed_dir) if parsed_dir else project / "public_data" / "parsed"
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.parsed = self._load_parsed()
        self.cache = dict(self.parsed)
        self.vacancy_failures = set()
        self.lock = Lock()

    @staticmethod
    def _load_matching_reference(directory):
        """Load the fixed AI-labeled reference without invoking extraction or gates."""
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Matching reference manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tables = {}
        for name in ("pairs.jsonl", "vacancies.jsonl", "resumes.jsonl"):
            path = directory / name
            raw = path.read_bytes()
            expected_hash = manifest.get("sha256", {}).get(name)
            if not expected_hash or hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError(f"Matching reference checksum mismatch: {name}")
            tables[name] = [
                json.loads(line) for line in raw.decode("utf-8").splitlines()
            ]
        counts = manifest.get("actual", {})
        if (
            len(tables["pairs.jsonl"]) != counts.get("pairs")
            or len(tables["vacancies.jsonl"]) != counts.get("vacancies")
            or len(tables["resumes.jsonl"]) != counts.get("resumes")
        ):
            raise ValueError("Matching reference counts do not match manifest")
        return {
            "manifest": manifest,
            "pairs": tables["pairs.jsonl"],
            "vacancies": {item["id"]: item for item in tables["vacancies.jsonl"]},
            "resumes": {item["id"]: item for item in tables["resumes.jsonl"]},
        }

    def reference_order(self, resume_id):
        """Return AI-annotated ranks separately from vacancies lacking labels."""
        if self.matching_reference is None:
            raise ValueError("Matching reference is not configured")
        if resume_id not in {resume["id"] for resume in self.resumes}:
            raise ValueError("Unknown resume")
        available = {
            item["id"]: item for item in [*self.vacancies, *self.extra_vacancies]
        }
        if resume_id not in self.matching_reference["resumes"]:
            raise ValueError("Resume is not covered by the matching reference")
        resume = next(item for item in self.resumes if item["id"] == resume_id)
        frozen_resume = self.matching_reference["resumes"][resume_id]
        if hashlib.sha256(resume["text"].encode()).hexdigest() != frozen_resume.get(
            "text_sha256"
        ):
            raise ValueError("Resume text changed since reference annotation")
        pairs = [
            pair
            for pair in self.matching_reference["pairs"]
            if pair.get("resume_id") == resume_id
        ]
        pairs.sort(key=lambda pair: pair.get("rank", 0))
        if len(pairs) != 20 or len({pair.get("vacancy_id") for pair in pairs}) != 20:
            raise ValueError(
                "Matching reference must contain 20 unique ranked vacancies per resume"
            )
        ranked_ids = {pair["vacancy_id"] for pair in pairs}
        if not ranked_ids <= available.keys():
            raise ValueError(
                "Matching reference vacancy is missing from the demo catalog"
            )
        ranked = []
        for pair in pairs:
            source = available[pair["vacancy_id"]]
            frozen = self.matching_reference["vacancies"].get(pair["vacancy_id"], {})
            if hashlib.sha256(source["text"].encode()).hexdigest() != frozen.get(
                "text_sha256"
            ):
                raise ValueError("Vacancy text changed since reference annotation")
            ranked.append(
                {
                    "vacancy_id": pair["vacancy_id"],
                    "title": source["title"],
                    "rank": pair["rank"],
                    "relevance": pair["relevance"],
                    "rationale": pair.get("rationale", ""),
                }
            )
        unannotated = [
            {"id": item["id"], "title": item["title"]}
            for item in [*self.vacancies, *self.extra_vacancies]
            if item["id"] not in ranked_ids
        ]
        return {
            "resume_id": resume_id,
            "disclosure": "AI-labeled reference; not an algorithm prediction.",
            "ranked": ranked,
            "unannotated": unannotated,
        }

    def cache_key(self, document, kind):
        """Hash document identity and effective extraction configuration."""
        provider_config = getattr(self.provider, "config", None)
        if provider_config is not None:
            provider_config = provider_config.model_dump(mode="json")
        payload = {
            "version": 1,
            "kind": kind,
            "document_id": document["document_id"],
            "text_version": document["text_version"],
            "text_sha256": hashlib.sha256(document["text"].encode()).hexdigest(),
            "prompts": self.prompts.model_dump(mode="json"),
            "vocabulary": self.vocabulary.model_dump(mode="json"),
            "labels": self.labels,
            "postprocessing": self.postprocessing_rules,
            "llm": provider_config,
            "vacancy_template": self.vacancy_template if kind == "vacancy" else None,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _load_parsed(self):
        """Load only results with matching document and configuration hashes."""
        parsed = {}
        documents = {
            item["id"]: (kind, item)
            for kind, entries in (("vacancy", self.vacancies), ("resume", self.resumes))
            for item in entries
        }
        for directory in (self.parsed_dir, self.cache_dir):
            if directory is None or not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(value, dict) or path.stem not in documents:
                    continue
                kind, document = documents[path.stem]
                if (
                    value.get("id") == path.stem
                    and value.get("kind") == kind
                    and value.get("document_id") == document["document_id"]
                    and value.get("text_version") == document["text_version"]
                    and value.get("cache_key") == self.cache_key(document, kind)
                    and isinstance(value.get("entities"), list)
                ):
                    if kind == "vacancy":
                        value["entities"] = [
                            {
                                **{
                                    k: v
                                    for k, v in entity.items()
                                    if k != "base_importance"
                                },
                                "importance": entity["base_importance"],
                            }
                            if "base_importance" in entity
                            else entity
                            for entity in value["entities"]
                            if not entity.get("defaulted")
                        ]
                        value["entities"] = apply_rules(
                            value["entities"], self.vacancy_postprocessing_rules
                        )
                        value = apply_vacancy_defaults(value, self.gate_config)
                    parsed[path.stem] = value
        return parsed

    def _cached_extraction(self, document, kind):
        key = self.cache_key(document, kind)
        result = self.cache.get(document["id"])
        if result is not None and result.get("cache_key") == key:
            return result
        result = self.extract_document(document, kind=kind)
        self.cache[document["id"]] = result
        if self.cache_dir is not None:
            self._save_result(result, self.cache_dir)
        return result

    def _available_vacancy(self, document):
        """Keep other vacancies available if this one cannot be extracted."""
        if document["id"] not in self.vacancy_failures:
            try:
                return self._cached_extraction(document, "vacancy")
            except Exception as error:
                logging.warning("Vacancy extraction failed (%s)", type(error).__name__)
                self.vacancy_failures.add(document["id"])
        return {"kind": "vacancy", "entities": [], "extraction_error": True}

    @staticmethod
    def _save_result(result, directory):
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{result['id']}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)

    def catalog(self):
        """Return source metadata and any saved grounded extraction results."""
        with self.lock:
            vacancies = [
                {
                    **item,
                    **(
                        self._available_vacancy(item)
                        if self.provider
                        else self.cache.get(item["id"], {})
                    ),
                }
                for item in self.vacancies
            ]
            vacancies.extend(self.extra_vacancies)
        return {
            "vacancies": vacancies,
            "resumes": self.resumes,
            "cached": len(self.cache),
        }

    def extract_document(self, document, *, kind):
        """Extract, ground, and optionally postprocess one vacancy or resume."""
        if kind not in {"vacancy", "resume"}:
            raise ValueError("Unknown document kind")
        prompt = getattr(self.prompts, kind)
        model = self.vacancy_model if kind == "vacancy" else self.output_model
        result = self.provider.structured(
            [
                {"role": "system", "content": self.prompts.system + "\n" + prompt},
                {
                    "role": "user",
                    "content": self.prompts.user_intro
                    + "\n"
                    + (
                        self.vacancy_catalog_json
                        if kind == "vacancy"
                        else self.catalog_json
                    )
                    + "\n<source_document>\n"
                    + document["text"]
                    + "\n</source_document>",
                },
            ],
            model,
            public_data=True,
        )
        result = model.model_validate(result.model_dump())
        if kind == "vacancy":
            extracted = [
                Entity(
                    category=self.vacancy_template[name]["category"],
                    importance=self.vacancy_template[name]["importance"],
                    polarity=None,
                    **fact.model_dump(),
                )
                for name in self.vacancy_template
                for fact in getattr(result, name)
            ]
        else:
            extracted = result.entities
        entities = []
        rejected_quotes = 0
        rejected_values = 0
        for entity in extracted:
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
        else:
            entities = apply_rules(entities, self.vacancy_postprocessing_rules)
        result = {
            "id": document["id"],
            "kind": kind,
            "document_id": document["document_id"],
            "text_version": document["text_version"],
            "cache_key": self.cache_key(document, kind),
            "entities": entities,
            "rejected": rejected,
            "rejected_quotes": rejected_quotes,
            "rejected_values": rejected_values,
        }
        if kind == "vacancy":
            return apply_vacancy_defaults(result, self.gate_config)
        return result

    def extract_all(self):
        """Run actual provider extraction on all approved files and persist JSON."""
        results = {"vacancies": [], "resumes": []}
        for kind, documents in (("vacancy", self.vacancies), ("resume", self.resumes)):
            for document in documents:
                result = self.extract_document(document, kind=kind)
                self.parsed[document["id"]] = result
                self.cache[document["id"]] = result
                self._save_result(result, self.parsed_dir)
                results["vacancies" if kind == "vacancy" else "resumes"].append(result)
        return results

    def gate(self, resume_id):
        """Evaluate each vacancy against one resume without inventing a ranking."""
        resume = next((r for r in self.resumes if r["id"] == resume_id), None)
        if resume is None:
            raise ValueError("Unknown resume")
        with self.lock:
            resume_features = self._cached_extraction(resume, "resume")
            vacancies = []
            for vacancy in self.vacancies:
                features = self._available_vacancy(vacancy)
                if features.get("extraction_error"):
                    decision = {
                        "status": "unknown",
                        "decisions": [
                            {
                                "filter": "experience",
                                "status": "unknown",
                                "explanation": (
                                    "Не удалось извлечь требования вакансии; "
                                    "проверка невозможна."
                                ),
                            }
                        ],
                    }
                else:
                    decision = evaluate_first_gate(
                        features, resume_features, self.gate_config
                    )
                vacancies.append(
                    {
                        "id": vacancy["id"],
                        "title": vacancy["title"],
                        "source_url": vacancy.get("source_url"),
                        "gate": decision,
                        "extraction_error": bool(features.get("extraction_error")),
                    }
                )
            vacancies.extend(
                {
                    "id": vacancy["id"],
                    "title": vacancy["title"],
                    "source_url": vacancy.get("source_url"),
                    "gate": evaluate_first_gate(
                        vacancy, resume_features, self.gate_config
                    ),
                }
                for vacancy in self.extra_vacancies
            )
            return {"resume_id": resume_id, "vacancies": vacancies}

    def analyze(self, vacancy_id, resume_id=None):
        """Reuse results for unchanged documents and extraction configuration."""
        if not any(v["id"] == vacancy_id for v in self.vacancies):
            raise ValueError("Unknown vacancy")
        if resume_id is not None and not any(
            r["id"] == resume_id for r in self.resumes
        ):
            raise ValueError("Unknown resume")
        with self.lock:
            for resume in self.resumes:
                self._cached_extraction(resume, "resume")
            return {
                "mock": False,
                "reference_order": (
                    self.reference_order(resume_id or self.resumes[0]["id"])
                    if self.extra_vacancies and self.matching_reference
                    else None
                ),
                "resumes": [
                    {**resume, **self.cache[resume["id"]]} for resume in self.resumes
                ],
            }
