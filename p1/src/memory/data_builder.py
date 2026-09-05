"""Data builder for initial seed memories and evaluation test cases."""

from typing import List, Dict, Any


def build_seed_memory() -> List[Dict[str, Any]]:
    """Build 3-5 seed memory items representing historical solved problems.

    Each record contains:
        memory_id: str
        source_problem_id: str
        problem_text: str
        solution_text: str

    Returns:
        List of seed memory dictionaries.
    """
    seeds = [
        {
            "memory_id": "mem_001",
            "source_problem_id": "seed_speed_round_trip",
            "problem_text": (
                "A train travels from Station A to Station B at a speed of 60 km/h "
                "and returns at a speed of 90 km/h. If the distance between Station A "
                "and Station B is 180 km, what is the average speed of the train for "
                "the entire round trip?"
            ),
            "solution_text": (
                "Step 1: Calculate total round-trip distance: 180 km * 2 = 360 km.\n"
                "Step 2: Calculate outbound travel time: 180 / 60 = 3 hours.\n"
                "Step 3: Calculate return travel time: 180 / 90 = 2 hours.\n"
                "Step 4: Total travel time = 3 + 2 = 5 hours.\n"
                "Step 5: Average speed = Total distance / Total time = 360 / 5 = 72 km/h."
            ),
        },
        {
            "memory_id": "mem_002",
            "source_problem_id": "seed_collaborative_work",
            "problem_text": (
                "Worker Alice can paint a house in 4 hours, and Worker Bob can paint "
                "the same house in 6 hours. How many hours will it take for them to "
                "paint the house working together?"
            ),
            "solution_text": (
                "Step 1: Alice's hourly work rate is 1/4 of the house per hour.\n"
                "Step 2: Bob's hourly work rate is 1/6 of the house per hour.\n"
                "Step 3: Combined work rate = 1/4 + 1/6 = 3/12 + 2/12 = 5/12 per hour.\n"
                "Step 4: Total time required = 1 / (5/12) = 12/5 = 2.4 hours."
            ),
        },
        {
            "memory_id": "mem_003",
            "source_problem_id": "seed_consecutive_sum",
            "problem_text": (
                "The sum of three consecutive integers is 72. What is the value of the "
                "largest integer?"
            ),
            "solution_text": (
                "Step 1: Let the three consecutive integers be n - 1, n, and n + 1.\n"
                "Step 2: Sum equation: (n - 1) + n + (n + 1) = 3n = 72.\n"
                "Step 3: Solve for the middle integer: n = 72 / 3 = 24.\n"
                "Step 4: The largest integer is n + 1 = 24 + 1 = 25."
            ),
        },
        {
            "memory_id": "mem_004",
            "source_problem_id": "seed_sequential_discount",
            "problem_text": (
                "A store originally sells a jacket for $120. During a promotion, it offers "
                "a 25% discount, and then an additional 10% coupon is applied to the discounted "
                "price. What is the final price of the jacket?"
            ),
            "solution_text": (
                "Step 1: Price after the 25% discount = 120 * (1 - 0.25) = 120 * 0.75 = $90.\n"
                "Step 2: Additional 10% coupon on $90 gives 90 * (1 - 0.10) = 90 * 0.90 = $81.\n"
                "Step 3: The final price of the jacket is $81."
            ),
        },
    ]
    return seeds


def build_test_cases() -> Dict[str, List[Dict[str, Any]]]:
    """Construct test cases categorized into simple, similar, and unrelated problems.

    Returns:
        Dict mapping category name ("simple", "similar", "unrelated") to list of test cases.
    """
    test_cases = {
        "simple": [
            {
                "id": "simple_001",
                "category": "simple",
                "problem_text": "What is 1 + 1?",
                "expected_answer": "2",
                "notes": "Trivial arithmetic that requires no historical memory retrieval.",
            },
            {
                "id": "simple_002",
                "category": "simple",
                "problem_text": "If there are 3 red balls and 2 green balls in a box, how many balls are in the box?",
                "expected_answer": "5",
                "notes": "Basic counting problem solvable with direct reasoning.",
            },
        ],
        "similar": [
            {
                "id": "similar_001",
                "category": "similar",
                "target_memory_id": "mem_001",
                "problem_text": (
                    "A car travels from City X to City Y at a speed of 50 km/h "
                    "and returns at a speed of 75 km/h. If the distance between City X "
                    "and City Y is 150 km, what is the average speed of the car for "
                    "the entire round trip?"
                ),
                "notes": "Isomorphic to mem_001 with modified speed and distance parameters.",
            },
            {
                "id": "similar_002",
                "category": "similar",
                "target_memory_id": "mem_002",
                "problem_text": (
                    "Worker Dave can repair a roof in 5 hours, and Worker Eve can repair "
                    "the same roof in 10 hours. How many hours will it take them to repair "
                    "the roof working together?"
                ),
                "notes": "Isomorphic to mem_002 with modified worker rates.",
            },
            {
                "id": "similar_003",
                "category": "similar",
                "target_memory_id": "mem_003",
                "problem_text": (
                    "The sum of three consecutive odd integers is 81. What is the value "
                    "of the largest integer among them?"
                ),
                "notes": "Isomorphic to mem_003 with odd integer constraint.",
            },
        ],
        "unrelated": [
            {
                "id": "unrelated_001",
                "category": "unrelated",
                "problem_text": (
                    "Prove that there are infinitely many prime numbers using Euclid's "
                    "classical proof by contradiction."
                ),
                "notes": "Number theory proof, completely outside the memory database domain.",
            },
            {
                "id": "unrelated_002",
                "category": "unrelated",
                "problem_text": (
                    "Given a binary tree root node, describe how to find the lowest common "
                    "ancestor (LCA) of two given nodes using post-order depth-first search."
                ),
                "notes": "Computer science data structures problem, zero domain overlap.",
            },
        ],
    }
    return test_cases
