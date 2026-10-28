"""One explicit LLM request. Defaults to mock; live models incur API charges."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from adgenius.shared.llm import LLMClient, ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models.toml")
    parser.add_argument("--model", default="mock", help="Config alias: mock/qwen/deepseek/claude")
    parser.add_argument("--output", type=Path, help="Optional response JSON file (no overwrite)")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    config = ModelConfig.from_toml(args.config, args.model)
    with LLMClient(config) as client:
        response = client.generate([
            {"role": "system", "content": "Write accurate English advertising copy. "
             "Use only the supplied product facts. Return one or two sentences."},
            {"role": "user", "content": "Product: Blue desk stapler. Features: "
             "Metal construction; holds standard staples; staples up to 20 sheets."},
        ], max_output_tokens=256)
    record = json.dumps(response.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(record + "\n")
    print(record)


if __name__ == "__main__":
    main()
