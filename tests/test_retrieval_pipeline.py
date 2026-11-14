import json

import pytest

from vrscore.retrieval_pipeline import RetrievalPipeline


class DeterministicEncoder:
    def __init__(self):
        self.document_calls = []
        self.query_calls = []
        self.vectors = {
            "python": [1.0, 0.0],
            "sql": [0.0, 1.0],
            "resume": [1.0, 0.0],
        }

    def encode_documents(self, texts, *, titles=None):
        self.document_calls.append((list(texts), list(titles or [])))
        return [self.vectors[t.lower()] for t in texts]

    def encode_query(self, text):
        self.query_calls.append(text)
        return self.vectors["resume"]


def doc(identifier, skill, *, experience=2, title=None):
    result = {
        "id": identifier,
        "kind": "vacancy",
        "text": skill,
        "entities": [
            {"category": "experience", "value": experience},
            {"category": "hard_skill", "value": skill, "evidence": f"src:{identifier}"},
        ],
    }
    if title is not None:
        result["title"] = title
    return result


def resume(text="resume", skill="python", experience=2):
    return {
        "id": "r1",
        "kind": "resume",
        "text": text,
        "entities": [
            {"category": "experience", "value": experience},
            {
                "category": "hard_skill",
                "value": skill,
                "polarity": "present",
                "evidence": "resume evidence",
            },
        ],
    }


def test_real_gate_bm25_and_embedding_return_separate_ranks_scores_and_evidence():
    encoder = DeterministicEncoder()
    pipeline = RetrievalPipeline(
        [doc("p", "python", title="Py"), doc("s", "sql")], encoder
    )
    result = pipeline.search(resume())
    assert [item["id"] for item in result["results"]] == ["p"]
    assert result["results"][0]["lexical_rank"] == 1
    assert result["results"][0]["lexical_score"] > 0
    assert result["results"][0]["embedding_score"] == pytest.approx(1.0)
    assert result["results"][0]["matches"][0]["vacancy_entity"]["evidence"] == "src:p"
    assert set(result["gates"]) == {"p", "s"}
    assert json.loads(json.dumps(result)) == result
    assert encoder.document_calls == [(["python"], ["Py"])]
    assert encoder.query_calls == ["resume"]


def test_gate_excludes_fail_and_includes_unknown_only_when_configured():
    encoder = DeterministicEncoder()
    fail = doc("fail", "python", experience=10)
    unknown = doc("unknown", "python", experience=1)
    unknown["entities"].append({"category": "education", "value": "bachelor"})
    candidate = resume(experience=1)
    candidate["entities"].append(
        {"category": "education", "value": "Неоконченное высшее", "evidence": "source"}
    )
    pipeline = RetrievalPipeline(
        [fail, unknown],
        encoder,
        gate_config={"filters": {"order": ["experience", "education"]}},
        include_unknown=False,
    )
    result = pipeline.search(candidate)
    assert result["gates"]["fail"]["status"] == "fail"
    assert result["gates"]["unknown"]["status"] == "unknown"
    assert result["eligible_ids"] == []
    assert result["accounting"] == {
        "vacancy_count": 2,
        "eligible_count": 0,
        "lexical_hit_count": 0,
        "returned_count": 0,
        "unranked_count": 0,
        "no_hit_ids": [],
    }
    assert encoder.document_calls == []
    assert encoder.query_calls == []

    inclusive = RetrievalPipeline(
        [fail, unknown],
        encoder,
        gate_config={"filters": {"order": ["experience", "education"]}},
        include_unknown=True,
    )
    included = inclusive.search(candidate)
    assert included["eligible_ids"] == ["unknown"]
    assert [item["id"] for item in included["results"]] == ["unknown"]


def test_no_lexical_hits_are_accounted_and_do_not_encode_query():
    encoder = DeterministicEncoder()
    pipeline = RetrievalPipeline([doc("p", "python")], encoder)
    result = pipeline.search(resume(skill="rust"))
    assert result["results"] == []
    assert result["accounting"]["no_hit_ids"] == ["p"]
    assert result["accounting"]["unranked_count"] == 1
    assert encoder.query_calls == []
    assert encoder.document_calls == []


def test_shortlist_is_bounded_and_document_vectors_reused_across_searches():
    encoder = DeterministicEncoder()
    pipeline = RetrievalPipeline(
        [doc("p", "python"), doc("p2", "python"), doc("s", "sql")],
        encoder,
        shortlist_size=1,
    )
    first = pipeline.search(resume())
    second = pipeline.search(resume(text="again"))
    assert len(first["results"]) == len(second["results"]) == 1
    assert len(encoder.document_calls) == 1
    assert encoder.query_calls == ["resume", "again"]
    assert first["results"][0]["lexical_rank"] == 1


def test_duplicate_document_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        RetrievalPipeline(
            [doc("same", "python"), doc("same", "sql")], DeterministicEncoder()
        )


def test_large_shortlist_is_encoded_in_bounded_batches_and_deduplicated():
    class BoundedEncoder(DeterministicEncoder):
        def encode_documents(self, texts, *, titles=None):
            assert len(texts) <= 128
            return super().encode_documents(texts, titles=titles)

    encoder = BoundedEncoder()
    vacancies = [doc(str(i), "python", title=str(i)) for i in range(130)]
    vacancies += [doc("duplicate", "python", title="0")]
    result = RetrievalPipeline(vacancies, encoder, shortlist_size=131).search(resume())
    assert len(result["results"]) == 131
    assert sum(len(texts) for texts, _ in encoder.document_calls) == 130
