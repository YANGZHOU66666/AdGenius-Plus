"""Offline tests for the training helpers in scripts/, scripts/opd, scripts/sft and scripts/formal (no GPU, Colab or API calls)."""

import importlib.util
import json
import sys
import tempfile
import types
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
FORMAL = SCRIPTS / "formal"
watchdog_formal = load("watchdog_formal", FORMAL / "watchdog_formal.py")
# drive_sync only needs transformers for the TrainerCallback base class, which is not installed for local tests.
with patch.dict(sys.modules, {"transformers": types.SimpleNamespace(TrainerCallback=object)}):
    drive_sync = load("drive_sync", FORMAL / "drive_sync.py")


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


class FormalRunTests(unittest.TestCase):
    def test_local_checkpoint_needs_state_and_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step, files in ((25, ["trainer_state.json", "optimizer.pt"]), (50, ["trainer_state.json", "optimizer.pt"]),
                                (75, ["trainer_state.json"])):  # step 75 was interrupted while saving
                (root / f"checkpoint-{step}").mkdir()
                for f in files:
                    (root / f"checkpoint-{step}" / f).write_text("x")
            self.assertEqual(drive_sync.local_complete_steps(root), [25, 50])
            self.assertEqual(drive_sync.latest_checkpoint(root, None), root / "checkpoint-50")
            self.assertIsNone(drive_sync.latest_checkpoint(root / "missing", None))

    def test_drive_step_counts_only_with_upload_marker(self):
        listings = {"r/full/": "step_0025/\nstep_0050/\nnotes/\n", "r/full/step_0025/": "UPLOAD_OK\nmodel.safetensors\n",
                    "r/full/step_0050/": "model.safetensors\n"}
        fake = lambda *a, **k: types.SimpleNamespace(stdout=listings.get(a[1], ""), returncode=0)  # noqa: E731
        with patch.object(drive_sync, "rclone", side_effect=fake):
            self.assertEqual(drive_sync.drive_complete_steps("r"), [25])

    def test_unreachable_colab_is_not_a_reclaim(self):
        watchdog_formal.M = dict(watchdog_formal.METHODS["opd"], key="opd")
        with patch.object(watchdog_formal, "colab", return_value=(1, "network error")):
            self.assertIsNone(watchdog_formal.alive())
        with patch.object(watchdog_formal, "colab", return_value=(0, "[colab] No active sessions found on server.")):
            self.assertFalse(watchdog_formal.alive())
        with patch.object(watchdog_formal, "colab", return_value=(0, "[opd] gpu-a100 | Hardware: A100")):
            self.assertTrue(watchdog_formal.alive())

    def test_every_uploaded_file_exists_in_the_repo(self):
        watchdog_formal.PROMPTS, watchdog_formal.SFT_DATA = Path("/prompts"), Path("/sft")
        for method, cfg in watchdog_formal.METHODS.items():
            for f in cfg["files"]:
                path = watchdog_formal.source(f)
                if f.endswith(".jsonl"):
                    self.assertIn(path.parent, (Path("/prompts"), Path("/sft")), f)
                else:
                    self.assertTrue(path.exists(), f"{method}: {path}")
        self.assertTrue((FORMAL / "formal_cell.py").exists() and (FORMAL / "wait_uploads_cell.py").exists())

    def test_trainers_resume_from_checkpoint_weights(self):
        # Trainer's own resume does not map the saved model.language_model.* names (it silently keeps the untrained
        # weights), so the checkpoint must be loaded with from_pretrained.
        for path in (OPD / "opd_train.py", SFT / "sft_train.py"):
            src = path.read_text()
            self.assertIn("from_pretrained(args.resume_from or args.student", src, path.name)
            self.assertIn("trainer.train(resume_from_checkpoint=args.resume_from)", src, path.name)
            self.assertIn('"drive_sync.py"', src, path.name)  # hashed with the code that ran
        self.assertIn('train_sampling_strategy="sequential"', (SFT / "sft_train.py").read_text())

    def test_sweep_watchdogs_upload_drive_sync(self):
        # opd_train.py and sft_train.py import drive_sync, so the sweep VMs need it too.
        self.assertIn("drive_sync.py", watchdog.CODE_FILES)
        self.assertEqual(watchdog.source("drive_sync.py"), FORMAL / "drive_sync.py")
        self.assertIn("drive_sync.py", watchdog_sft.FILES)
        self.assertEqual(watchdog_sft.source("drive_sync.py"), FORMAL / "drive_sync.py")


if __name__ == "__main__":
    unittest.main()
