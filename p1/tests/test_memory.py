"""Pytest suite for p1/src/memory modules."""

import pytest
from p1.src.memory.vector_store import SimpleVectorStore
from p1.src.memory.data_builder import build_seed_memory, build_test_cases


@pytest.fixture
def populated_store():
    store = SimpleVectorStore()
    seeds = build_seed_memory()
    for s in seeds:
        store.add_record(
            memory_id=s["memory_id"],
            source_problem_id=s["source_problem_id"],
            problem_text=s["problem_text"],
            solution_text=s["solution_text"],
        )
    return store, seeds, build_test_cases()


def test_signatures():
    store = SimpleVectorStore()
    assert hasattr(store, "add_record")
    assert hasattr(store, "retrieve")
    seeds = build_seed_memory()
    assert isinstance(seeds, list)
    test_cases = build_test_cases()
    assert isinstance(test_cases, dict)


def test_empty_store():
    store = SimpleVectorStore()
    res = store.retrieve("hello", top_k=1)
    assert res == []


def test_similar_retrieval_accuracy(populated_store):
    store, _, test_cases = populated_store
    for case in test_cases["similar"]:
        res = store.retrieve(case["problem_text"], top_k=1)
        assert len(res) == 1
        top = res[0]
        assert top["memory_id"] == case["target_memory_id"]
        assert top["similarity"] >= 0.40
        assert "solution_text" in top
        assert "source_problem_id" in top


def test_unrelated_and_simple_low_similarity(populated_store):
    store, _, test_cases = populated_store
    for case in test_cases["unrelated"]:
        res = store.retrieve(case["problem_text"], top_k=1)
        sim = res[0]["similarity"] if res else 0.0
        assert sim <= 0.20

    for case in test_cases["simple"]:
        res = store.retrieve(case["problem_text"], top_k=1)
        sim = res[0]["similarity"] if res else 0.0
        assert sim <= 0.20


def test_top_k_sorting(populated_store):
    store, _, test_cases = populated_store
    query = test_cases["similar"][0]["problem_text"]
    top2 = store.retrieve(query, top_k=2)
    assert len(top2) == 2
    assert top2[0]["similarity"] >= top2[1]["similarity"]
