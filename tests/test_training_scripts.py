"""Offline tests for the training helpers in scripts/, scripts/opd and scripts/sft (no GPU, Colab or API calls)."""

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
SFT = SCRIPTS / "sft"
select_sft = load("select_sft", SFT / "lr_sweep" / "select_sft.py")
watchdog_sft = load("watchdog_sft", SFT / "lr_sweep" / "watchdog_sft.py")


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


class SftTests(unittest.TestCase):
    def test_sft_uses_the_opd_data_order(self):
        import re

        def body(path, name):
            source = path.read_text()
            fn = re.search(rf"def {name}\(.*?\n(?=\n\n)", source, re.S).group(0)
            return [line for line in fn.splitlines()[1:] if not line.strip().startswith('"""')]

        self.assertEqual(body(OPD / "opd_train.py", "ordered_prompts"), body(SFT / "sft_train.py", "ordered_rows"))
        self.assertIn('train_sampling_strategy="sequential"', (SFT / "sft_train.py").read_text())

    def test_tie_with_the_runner_up_picks_the_smaller_learning_rate(self):
        lrs = select_sft.LRS
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            products = [f"p{i}" for i in range(200)]
            dev_rows = "".join(json.dumps({"product_id": p, "text": "An ad.", "finish_reason": "stop"}) + "\n"
                               for p in products)
            for lr in lrs:
                run_dir = work / "results" / select_sft.run(lr)
                run_dir.mkdir(parents=True)
                (run_dir / "run_report.json").write_text(json.dumps({"diverged": False}))
                (run_dir / "dev.jsonl").write_text(dev_rows)
            (work / "base.jsonl").write_text(dev_rows)
            # lr[1] beats lr[0] on 104 of 200 products (inside noise); both beat lr[2] on every product.
            wins_for_a = {(lrs[0], lrs[1]): 96, (lrs[0], lrs[2]): 200, (lrs[1], lrs[2]): 200}
            for (a, b), a_wins in wins_for_a.items():
                matchup, rid = f"{select_sft.run(a)}__vs__{select_sft.run(b)}", select_sft.judge_run_id(a, b)
                cand_a = {p: f"{a}_{p}" for p in products}
                (work / "eval/pairs").mkdir(parents=True, exist_ok=True)
                (work / "eval/pairwise" / rid).mkdir(parents=True)
                (work / "eval/pairs" / f"{matchup}.excluded.json").write_text(json.dumps(
                    {"candidate_a": cand_a, "excluded": []}))
                lines = []
                for i, p in enumerate(products):
                    winner = cand_a[p] if i < a_wins else f"{b}_{p}"
                    lines.append(json.dumps({"candidate_mapping": {"A": cand_a[p], "B": f"{b}_{p}"}, "pairwise_result": [
                        {"prompt_order": o, "status": "success", "winner_candidate_id": winner} for o in ("AB", "BA")]}))
                (work / "eval/pairwise" / rid / "results.jsonl").write_text("\n".join(lines) + "\n")
            select_sft.configure(work)
            with patch.object(select_sft, "BASE_DEV", work / "base.jsonl"), patch("builtins.print"):
                select_sft.main()
            summary = json.loads((work / "eval/selection.json").read_text())
        self.assertEqual(summary["best"], select_sft.run(lrs[1]))
        self.assertTrue(summary["best_vs_runner_up_ci_contains_half"])
        self.assertEqual(summary["chosen"], select_sft.run(lrs[0]))
        self.assertTrue(summary["at_grid_edge"])

    def test_unreachable_colab_is_not_a_reclaim(self):
        with patch.object(watchdog_sft, "colab", return_value=(1, "network error")):
            self.assertIsNone(watchdog_sft.alive())
        with patch.object(watchdog_sft, "colab", return_value=(0, "[colab] No active sessions found on server.")):
            self.assertFalse(watchdog_sft.alive())


if __name__ == "__main__":
    unittest.main()
