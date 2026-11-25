"""Join the frozen train prompts with the frozen 397B teacher ads into SFT examples (runs locally).

    python3 scripts/sft/build_sft_data.py      # defaults: prompts from artifacts/train/prompts/v1 (built by
                                               # scripts/build_training_prompts.py), output to artifacts/train/sft_data/v1

Writes sft_train_v1.jsonl in rank order: {"product_id", "rank", "prompt": [system, user], "completion": [assistant]}.
The prompt messages are exactly those of prompts_train_v1.jsonl (the same as for gold, teacher data and OPD); the
completion is the teacher ad from data/train/v1/teacher_qwen397b_train.jsonl. The training script applies the same
block-shuffled order as OPD. Also writes sft_data_manifest.json with hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEACHER = REPO / "data/train/v1/teacher_qwen397b_train.jsonl"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build SFT examples from frozen prompts and teacher ads.")
    parser.add_argument("--prompts-dir", type=Path, default=REPO / "artifacts/train/prompts/v1")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/train/sft_data/v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prompts_file = args.prompts_dir / "prompts_train_v1.jsonl"
    prompts = [json.loads(l) for l in prompts_file.open(encoding="utf-8")]
    teacher = {}
    for line in TEACHER.open(encoding="utf-8"):
        r = json.loads(line)
        assert r["generation_result"]["status"] == "success" and r["product_id"] not in teacher
        teacher[r["product_id"]] = r
    assert len(prompts) == len(teacher) == 8000 and {p["product_id"] for p in prompts} == teacher.keys()
    out = args.out / "sft_train_v1.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for p in prompts:
            t = teacher[p["product_id"]]
            f.write(json.dumps({"product_id": p["product_id"], "rank": p["rank"], "prompt": p["prompt"],
                                "completion": [{"role": "assistant", "content": t["generation_result"]["text"]}],
                                "teacher_candidate_id": t["candidate_id"]}, ensure_ascii=False) + "\n")
    manifest = {"rows": len(prompts), "sha256": sha(out),
                "prompts": {"file": "prompts_train_v1.jsonl", "sha256": sha(prompts_file)},
                "teacher": {"path": "data/train/v1/teacher_qwen397b_train.jsonl", "sha256": sha(TEACHER),
                            "run_id": "gen_teacher_qwen397b_train_v1"}}
    (args.out / "sft_data_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
