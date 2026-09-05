"""Generate synthetic dataset for post-training."""

import json
from pathlib import Path
import random

def generate_sft_dataset(output_path: Path, num_samples: int = 100):
    """Generate mock but structurally correct mathematical reasoning dataset.
    
    Includes two types of problems:
    1. Isomorphic math problems (needs memory recall).
    2. Unrelated/Simple problems (direct reasoning).
    """
    templates_isomorphic = [
        {
            "problem": "A car travels from {A} to {B} at {v1} km/h and returns at {v2} km/h. Distance is {d} km. Average speed?",
            "mem_id": "mem_001",
            "old_desc": "seed_speed_round_trip",
            "logic": "Average speed = Total distance / Total time. Total time = d/v1 + d/v2",
            "calc": lambda v1, v2, d: (2 * d) / (d / v1 + d / v2)
        },
        {
            "problem": "Worker {W1} repairs a {item} in {t1} hours, Worker {W2} in {t2} hours. Working together time?",
            "mem_id": "mem_002",
            "old_desc": "seed_collaborative_work",
            "logic": "Combined rate = 1/t1 + 1/t2. Total time = 1 / Combined rate",
            "calc": lambda t1, t2: 1 / (1 / t1 + 1 / t2)
        }
    ]

    templates_direct = [
        {"problem": "What is {a} + {b}?", "calc": lambda a, b: a + b},
        {"problem": "If there are {a} red balls and {b} green balls, total?", "calc": lambda a, b: a + b}
    ]

    records = []

    # Generate isomorphic
    for _ in range(int(num_samples * 0.7)):
        tmpl = random.choice(templates_isomorphic)
        if "car travels" in tmpl["problem"]:
            v1 = random.choice([40, 50, 60, 80])
            v2 = random.choice([60, 75, 90, 120])
            d = random.choice([120, 150, 180, 240])
            prob = tmpl["problem"].format(A="City A", B="City B", v1=v1, v2=v2, d=d)
            ans = tmpl["calc"](v1, v2, d)
            
            # Format target output exactly as we want RL to enforce
            # Using special tags to make reward function parsing easier
            assistant_response = (
                f"<thought>\n"
                f"我回想起了一道相似的历史题：{tmpl['old_desc']} ({tmpl['mem_id']})。\n"
                f"【分析旧题】：原题为往返行程问题。\n"
                f"【提取逻辑】：{tmpl['logic']}。\n"
                f"【修改方案】：当前 v1={v1}, v2={v2}, d={d}。总路程={2*d}, 总时间={d/v1 + d/v2}。\n"
                f"</thought>\n"
                f"<answer>{ans:.1f}</answer>"
            )
            
        else:
            t1 = random.choice([2, 4, 5, 8])
            t2 = random.choice([3, 6, 10, 12])
            prob = tmpl["problem"].format(W1="Alice", W2="Bob", item="roof", t1=t1, t2=t2)
            ans = tmpl["calc"](t1, t2)
            
            assistant_response = (
                f"<thought>\n"
                f"我回想起了一道相似的历史题：{tmpl['old_desc']} ({tmpl['mem_id']})。\n"
                f"【分析旧题】：原题为合作工效题。\n"
                f"【提取逻辑】：{tmpl['logic']}。\n"
                f"【修改方案】：当前 t1={t1}, t2={t2}。合作效率={1/t1:.3f}+{1/t2:.3f}。\n"
                f"</thought>\n"
                f"<answer>{ans:.2f}</answer>"
            )

        records.append({
            "prompt": [{"role": "user", "content": prob}],
            "messages": [
                {"role": "user", "content": prob},
                {"role": "assistant", "content": assistant_response}
            ]
        })

    # Generate direct
    for _ in range(int(num_samples * 0.3)):
        tmpl = random.choice(templates_direct)
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        prob = tmpl["problem"].format(a=a, b=b)
        ans = tmpl["calc"](a, b)
        
        assistant_response = (
            f"<thought>\n"
            f"这是一个基础直接计算题，无需联想。\n"
            f"计算: {a} + {b} = {ans}。\n"
            f"</thought>\n"
            f"<answer>{ans}</answer>"
        )

        records.append({
            "prompt": [{"role": "user", "content": prob}],
            "messages": [
                {"role": "user", "content": prob},
                {"role": "assistant", "content": assistant_response}
            ]
        })

    random.shuffle(records)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(records)} samples at {output_path}")

if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent.parent / "data" / "post_train.jsonl"
    generate_sft_dataset(out_file, 100)
