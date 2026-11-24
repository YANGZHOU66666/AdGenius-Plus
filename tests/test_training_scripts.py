"""Offline tests for the training helpers in scripts/ and scripts/opd (no GPU, Colab or API calls)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OPD = SCRIPTS / "opd"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompts = load("build_training_prompts", SCRIPTS / "build_training_prompts.py")
proxy = load("teacher_proxy", OPD / "teacher_proxy.py")
select = load("select_from_pairwise", OPD / "lr_sweep" / "select_from_pairwise.py")
watchdog = load("training_watchdog", OPD / "lr_sweep" / "watchdog.py")
candidates = load("generate_candidates", ROOT / "scripts" / "generate_candidates.py")


class PromptTests(unittest.TestCase):
    def test_prompts_match_the_gold_and_teacher_messages(self):
        from adgenius.shared.product import render_product_information
        products = [json.loads(line) for line in (ROOT / "data/train/v1/dev_products.jsonl").open(encoding="utf-8")][:3]
        rows = prompts.prompt_rows(products)
        self.assertEqual([r["rank"] for r in rows], [0, 1, 2])
        for row, item in zip(rows, products):
            self.assertEqual(row["product_id"], item["product_id"])
            self.assertEqual(row["prompt"], [{"role": "system", "content": candidates.SYSTEM_PROMPT},
                                             {"role": "user", "content": render_product_information(item)}])


class TeacherProxyTests(unittest.TestCase):
    HEADER = [7, 8, 9]

    def test_completion_starts_after_the_first_assistant_header(self):
        prompt = [1, 2, 3, *self.HEADER]
        self.assertEqual(proxy.completion_start(prompt + [20, 21], self.HEADER), 6)
        # A header leaked into the sampled completion must not move the start.
        self.assertEqual(proxy.completion_start(prompt + [20, *self.HEADER, 22], self.HEADER), 6)
        self.assertIsNone(proxy.completion_start([1, 2, 3, 20], self.HEADER))


class SelectionRuleTests(unittest.TestCase):
    def test_denominator_ties_and_order_inconsistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            select.configure(work)
            matchup, rid = "runA__vs__runB", "judge_x"
            products = [f"p{i}" for i in range(200)]
            cand_a = {p: f"a_{p}" for p in products}
            (work / "eval/pairs").mkdir(parents=True)
            (work / "eval/pairwise" / rid).mkdir(parents=True)
            (work / "eval/pairs" / f"{matchup}.excluded.json").write_text(json.dumps({
                "candidate_a": cand_a, "candidate_b": {p: f"b_{p}" for p in products},
                "excluded": [{"product_id": "p0", "rule": "identical_text", "score_a": 0.5}]}))

            def call(order, winner, status="success"):
                return {"prompt_order": order, "status": status, "winner_candidate_id": winner}

            rows = []
            for i, p in enumerate(products[1:], start=1):
                a, b = cand_a[p], f"b_{p}"
                if i <= 100:
                    result = [call("AB", a), call("BA", a)]          # A wins both orders
                elif i <= 150:
                    result = [call("AB", a), call("BA", b)]          # order inconsistent -> 0.5
                elif i <= 151:
                    result = [call("AB", None, "failed"), call("BA", None, "failed")]  # both failed -> 0.5
                else:
                    result = [call("AB", b), call("BA", b)]          # B wins both orders
                rows.append({"candidate_mapping": {"A": a, "B": b}, "pairwise_result": result})
            (work / "eval/pairwise" / rid / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            scores, notes = select.scores(matchup, rid)
        self.assertEqual(len(scores), 200)
        self.assertEqual(sum(scores.values()), 100 + 0.5 * (1 + 50 + 1))
        self.assertEqual(notes, {"order_inconsistent": 50, "both_orders_failed": 1, "settled_by_rule": 1})
        self.assertTrue(select.ties({"ci95": [0.45, 0.55]}))
        self.assertFalse(select.ties({"ci95": [0.51, 0.6]}))


class WatchdogTests(unittest.TestCase):
    def test_unreachable_colab_is_not_a_reclaim(self):
        with patch.object(watchdog, "colab", return_value=(1, "network error")):
            self.assertIsNone(watchdog.alive())
        with patch.object(watchdog, "colab", return_value=(0, "[colab] No active sessions found on server.")):
            self.assertFalse(watchdog.alive())
        with patch.object(watchdog, "colab", return_value=(0, "[sweep] gpu-a100 | Hardware: A100")):
            self.assertTrue(watchdog.alive())


if __name__ == "__main__":
    unittest.main()
