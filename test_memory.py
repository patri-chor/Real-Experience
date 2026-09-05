"""Independent verification script for Task 01: SimpleVectorStore and data_builder.

Verifies:
1. Class and function signature conformance.
2. Accurate retrieval of 'similar' problems matching target memory seeds.
3. Low similarity scores for 'unrelated' and 'simple' problems.
4. Edge case handling (e.g., top_k, sorting, empty store).
"""

import sys
import inspect
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from p1.src.memory.vector_store import SimpleVectorStore
from p1.src.memory.data_builder import build_seed_memory, build_test_cases


def verify_signatures() -> None:
    """Verify class and function signatures strictly match requirements."""
    print("=== Step 1: Verifying signatures ===")

    # SimpleVectorStore
    assert inspect.isclass(SimpleVectorStore), "SimpleVectorStore must be a class"
    store = SimpleVectorStore()

    # check add_record signature
    add_sig = inspect.signature(store.add_record)
    expected_add_params = ["memory_id", "source_problem_id", "problem_text", "solution_text"]
    for param in expected_add_params:
        assert param in add_sig.parameters, f"Missing parameter '{param}' in add_record"

    # check retrieve signature
    ret_sig = inspect.signature(store.retrieve)
    expected_ret_params = ["query_text", "top_k"]
    for param in expected_ret_params:
        assert param in ret_sig.parameters, f"Missing parameter '{param}' in retrieve"

    # check data_builder functions
    assert callable(build_seed_memory), "build_seed_memory must be callable"
    assert callable(build_test_cases), "build_test_cases must be callable"
    print("[PASS] All class and method signatures match requirements.")


def verify_seed_memory_and_test_cases() -> tuple[list[dict], dict[str, list[dict]]]:
    """Verify data structures produced by data_builder."""
    print("\n=== Step 2: Verifying data builders ===")
    seeds = build_seed_memory()
    assert isinstance(seeds, list), "build_seed_memory must return a list"
    assert len(seeds) >= 3, f"Expected at least 3 seeds, got {len(seeds)}"
    for s in seeds:
        for field in ["memory_id", "source_problem_id", "problem_text", "solution_text"]:
            assert field in s and s[field], f"Seed missing or empty field: {field}"

    test_cases = build_test_cases()
    assert isinstance(test_cases, dict), "build_test_cases must return a dict"
    for cat in ["simple", "similar", "unrelated"]:
        assert cat in test_cases, f"Missing category '{cat}' in test_cases"
        assert len(test_cases[cat]) >= 1, f"Category '{cat}' must have at least 1 item"

    print(f"[PASS] Seeds count: {len(seeds)}, Categories: {list(test_cases.keys())}")
    return seeds, test_cases


def verify_retrieval(seeds: list[dict], test_cases: dict[str, list[dict]]) -> None:
    """Verify vector retrieval accuracy and discrimination."""
    print("\n=== Step 3: Verifying retrieval dynamics ===")
    store = SimpleVectorStore()

    # Empty store edge case
    assert store.retrieve("any query") == [], "Empty store should return empty list"

    # Populate store with seeds
    for s in seeds:
        store.add_record(
            memory_id=s["memory_id"],
            source_problem_id=s["source_problem_id"],
            problem_text=s["problem_text"],
            solution_text=s["solution_text"],
        )

    print(f"Loaded {len(store.records)} seed records into SimpleVectorStore.")

    # 1. Test Similar cases
    print("\n--- Testing 'similar' cases ---")
    similar_scores = []
    for case in test_cases["similar"]:
        results = store.retrieve(case["problem_text"], top_k=1)
        assert len(results) == 1, "Expected 1 retrieved record"
        top = results[0]
        sim = top["similarity"]
        target = case["target_memory_id"]
        matched = top["memory_id"]
        similar_scores.append(sim)
        print(f"Query [{case['id']}]: Target={target}, Matched={matched}, Sim={sim:.4f}")
        assert matched == target, f"Expected {target} but retrieved {matched}"
        assert sim >= 0.35, f"Expected similarity >= 0.35 for similar case, got {sim}"
        assert "solution_text" in top and top["solution_text"], "Missing solution_text in result"

    # 2. Test Unrelated cases
    print("\n--- Testing 'unrelated' cases ---")
    unrelated_scores = []
    for case in test_cases["unrelated"]:
        results = store.retrieve(case["problem_text"], top_k=1)
        sim = results[0]["similarity"] if results else 0.0
        matched = results[0]["memory_id"] if results else "None"
        unrelated_scores.append(sim)
        print(f"Query [{case['id']}]: Matched={matched}, Sim={sim:.4f}")
        assert sim <= 0.20, f"Expected low similarity (<= 0.20) for unrelated case, got {sim}"

    # 3. Test Simple cases
    print("\n--- Testing 'simple' cases ---")
    simple_scores = []
    for case in test_cases["simple"]:
        results = store.retrieve(case["problem_text"], top_k=1)
        sim = results[0]["similarity"] if results else 0.0
        matched = results[0]["memory_id"] if results else "None"
        simple_scores.append(sim)
        print(f"Query [{case['id']}]: Matched={matched}, Sim={sim:.4f}")
        assert sim <= 0.15, f"Expected low similarity (<= 0.15) for simple case, got {sim}"

    # 4. Discrimination verification
    avg_similar = sum(similar_scores) / len(similar_scores)
    max_unrelated = max(unrelated_scores)
    max_simple = max(simple_scores)
    print("\n--- Summary Discrimination Metrics ---")
    print(f"Avg Similar Similarity:   {avg_similar:.4f}")
    print(f"Max Unrelated Similarity: {max_unrelated:.4f}")
    print(f"Max Simple Similarity:    {max_simple:.4f}")
    assert avg_similar > max_unrelated + 0.20, "Discrimination margin between similar and unrelated is insufficient"

    # 5. Top-K and ordering check
    top2 = store.retrieve(test_cases["similar"][0]["problem_text"], top_k=2)
    assert len(top2) == 2, "Expected 2 results for top_k=2"
    assert top2[0]["similarity"] >= top2[1]["similarity"], "Results must be sorted in descending similarity"

    print("\n[PASS] All retrieval verification tests succeeded!")


def main():
    print("Running Task 01 Verification Suite...")
    verify_signatures()
    seeds, test_cases = verify_seed_memory_and_test_cases()
    verify_retrieval(seeds, test_cases)
    print("\n==========================================")
    print("ALL TESTS PASSED: Task 01 successfully completed!")
    print("==========================================")


if __name__ == "__main__":
    main()
