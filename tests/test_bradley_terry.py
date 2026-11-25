"""Offline tests for the Bradley–Terry design, pair builder, fit and summary."""

from __future__ import annotations

import importlib.util
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import jsonschema

from adgenius.eval import bradley_terry as bt

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("import_generation_outputs",
                                              ROOT / "scripts" / "import_generation_outputs.py")
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)


def write_rows(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def generation(system, product, status="success"):
    text = None if status == "failed" else f"{system} ad for {product}"
    error = {"success": None, "truncated": "output_too_long", "failed": "timeout"}[status]
    return {"candidate_id": f"cand_{system}_{product}", "product_id": product, "run_id": f"gen_{system}_x",
            "sample_index": 0, "generation_result": {"status": status, "text": text, "error": error}, "trace_id": None}


class Fixture:
    def __init__(self, directory: Path, comparisons, statuses=None, extra=""):
        self.directory = directory
        products = [f"p{i}" for i in range(6)]
        write_rows(directory / "products.jsonl", [{"product_id": p} for p in products])
        systems = sorted({name for pair in comparisons for name in pair})
        statuses = statuses or {}
        for system in systems:
            write_rows(directory / f"{system}.jsonl",
                       [generation(system, p, statuses.get((system, p), "success")) for p in products])
        lines = ["[design]", 'name = "t"', f'products = "{directory}/products.jsonl"', 'anchor = "a"',
                 f'pairs = "{directory}/pairs.jsonl"', extra, "[systems]"]
        lines += [f'{s} = "{directory}/{s}.jsonl"' for s in systems]
        for a, b in comparisons:
            lines += ["[[comparisons]]", f'systems = ["{a}", "{b}"]']
        self.path = directory / "design.toml"
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def design(self):
        return bt.load_design(self.path)


def judged(pair, winners, run="r"):
    """PairwiseResult with candidate A/B in storage order; winners: winning candidate per order or None."""
    mapping = dict(zip("AB", (c["candidate_id"] for c in pair["candidates"])))
    orders = []
    for order, winner in zip(("AB", "BA"), winners):
        if winner is None:
            orders.append({"prompt_order": order, "status": "failed", "choice": None, "winner_candidate_id": None,
                           "reason": None, "error": "invalid_output", "trace_id": "t"})
        else:
            choice = "A" if winner == mapping["A"] else "B"
            orders.append({"prompt_order": order, "status": "success", "choice": choice,
                           "winner_candidate_id": winner, "reason": "x", "error": None, "trace_id": "t"})
    return {"result_id": "r_" + pair["pair_id"], "judge_type": "model", "pair_id": pair["pair_id"],
            "judge_run_id": run, "protocol_version": "pairwise-v0.2", "candidate_mapping": mapping,
            "pairwise_result": orders}


class DesignTest(unittest.TestCase):
    def test_rejects_disconnected_unused_and_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            for comparisons, message in (([("a", "b"), ("c", "d")], "do not connect"),
                                         ([("a", "b"), ("b", "a")], "Duplicate")):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    Fixture(Path(temp), comparisons).design()
            fixture = Fixture(Path(temp), [("a", "b")])
            fixture.path.write_text(fixture.path.read_text().replace("[systems]\n", '[systems]\nc = "x.jsonl"\n'))
            with self.assertRaisesRegex(ValueError, "not in any comparison"):
                fixture.design()

    def test_cycle_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(Fixture(Path(temp), [("a", "b"), ("b", "c")]).design()["cycle_rank"], 0)
            self.assertEqual(Fixture(Path(temp), [("a", "b"), ("b", "c"), ("c", "a")]).design()["cycle_rank"], 1)


class BuildPairsTest(unittest.TestCase):
    def test_pairs_interleave_and_failed_generations_forfeit(self):
        with tempfile.TemporaryDirectory() as temp:
            statuses = {("b", "p1"): "truncated", ("c", "p2"): "failed"}
            fixture = Fixture(Path(temp), [("b", "a"), ("c", "b")], statuses)
            pairs, report = bt.build_pairs(fixture.design())
            self.assertEqual([e["pairs"] for e in report["comparisons"]], [6, 5])
            self.assertEqual([e["forfeits"] for e in report["comparisons"]], [0, 1])
            self.assertEqual(bt.plan(fixture.design())["forfeits"], [("p2", 1, 0.0)])  # c failed: c loses
            products = [p["product_id"] for p in pairs]
            self.assertEqual(products[0], products[1])  # both comparisons for one product come together
            self.assertEqual(pairs, bt.build_pairs(fixture.design())[0])
            excluded = Fixture(Path(temp), [("b", "a"), ("c", "b")], statuses, "include_truncated = false")
            planned = bt.plan(excluded.design())
            self.assertEqual([e["pairs"] for e in planned["comparisons"]], [5, 4])
            # b's truncated ad is unusable here: b loses to a (score 0 for b) and to c (score 1 for c).
            self.assertEqual(sorted(planned["forfeits"]), [("p1", 0, 0.0), ("p1", 1, 1.0), ("p2", 1, 0.0)])

    def test_both_failed_is_a_tie_and_missing_record_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp), [("b", "a")], {("a", "p0"): "failed", ("b", "p0"): "failed"})
            self.assertEqual(bt.plan(fixture.design())["forfeits"], [("p0", 0, 0.5)])
            rows = bt.read_jsonl(Path(temp) / "a.jsonl")
            write_rows(Path(temp) / "a.jsonl", rows[1:])
            with self.assertRaisesRegex(ValueError, "No generation record"):
                bt.plan(fixture.design())

    def test_max_products_and_duplicate_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp), [("b", "a")])
            fixture.path.write_text(fixture.path.read_text() + "max_products = 2\n")
            pairs, _ = bt.build_pairs(fixture.design())
            self.assertEqual(len(pairs), 2)
            rows = bt.read_jsonl(Path(temp) / "a.jsonl")
            write_rows(Path(temp) / "b.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "more than one system"):
                bt.build_pairs(fixture.design())


class FitTest(unittest.TestCase):
    def test_recovers_known_strengths(self):
        with tempfile.TemporaryDirectory() as temp:
            design = Fixture(Path(temp), [("b", "a"), ("c", "b"), ("c", "a")]).design()
            truth = {"a": 0.0, "b": 1.0, "c": 1.7}
            totals = {}
            for index, edge in enumerate(design["comparisons"]):
                x, y = edge["systems"]
                totals[index] = [10000, 10000 / (1 + math.exp(-(truth[x] - truth[y])))]
            fitted = bt.fit(totals, design, pseudo=0)
            for name, value in truth.items():
                self.assertAlmostEqual(fitted[name], value, places=6)

    def test_all_wins_is_finite_with_pseudo_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            design = Fixture(Path(temp), [("b", "a")]).design()
            fitted = bt.fit({0: [20, 20.0]}, design)
            self.assertTrue(math.isfinite(fitted["b"]) and fitted["b"] > 3)


class SummaryTest(unittest.TestCase):
    def test_outcomes_orient_score_and_count_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            design = Fixture(Path(temp), [("b", "a"), ("c", "b"), ("c", "a")]).design()
            pairs, _ = bt.build_pairs(design)
            results = []
            for index, pair in enumerate(pairs):
                ids = {c["candidate_id"].split("_")[1]: c["candidate_id"] for c in pair["candidates"]}
                stronger = ids.get("c") or ids["b"]
                weaker = next(c for c in ids.values() if c != stronger)
                if index == 0:
                    winners = (stronger, None)          # incomplete
                elif index == 1:
                    winners = (stronger, weaker)        # orders disagree: tie
                else:
                    winners = (stronger, stronger)
                results.append(judged(pair, winners))
            rows, counts = bt.outcomes(design, pairs, results)
            self.assertEqual(counts["incomplete"], 1)
            self.assertEqual(counts["scored"], len(pairs) - 1)
            forfeits = [("p5", 0, 1.0)]  # b beat a by forfeit on one extra product pair
            counts["forfeits"] = len(forfeits)
            metrics = bt.summarize(design, rows, iterations=100, forfeits=forfeits)
            self.assertEqual(metrics["comparisons"][0]["forfeits"], 1)
            self.assertEqual(metrics["comparisons"][0]["pairs"], len([r for r in rows if r[1] == 0]) + 1)
            self.assertEqual([r["system"] for r in metrics["systems"]], ["c", "b", "a"])
            self.assertEqual(metrics["systems"][-1]["score"], 0.0)
            self.assertEqual(sum(r["ties"] for r in metrics["comparisons"]), 1)
            for row in metrics["comparisons"]:
                self.assertGreater(row["win_rate"], 0.5)  # first-listed system is the stronger one
            report = bt.render_report(metrics, counts)
            self.assertIn("| c |", report)
            metrics["pair_counts"] = counts
            bt.validate_metrics(metrics)
            for change in (lambda m: m["comparisons"][0].update(win_rate=1.5),
                           lambda m: m.pop("pair_counts"),
                           lambda m: m.update(schema_version="bt_metrics_v0")):
                broken = copy.deepcopy(metrics)
                change(broken)
                with self.assertRaises(jsonschema.ValidationError):
                    bt.validate_metrics(broken)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                bt.outcomes(design, pairs, results + results[:1])


class ImportTest(unittest.TestCase):
    def test_finish_reasons_map_to_statuses(self):
        rows = [{"product_id": "p0", "text": "Ad", "finish_reason": "stop"},
                {"product_id": "p1", "text": "Ad cut", "finish_reason": "length"},
                {"product_id": "p2", "text": " ", "finish_reason": "stop"}]
        records = importer.convert(rows, "gen_x_1", {"p0", "p1", "p2"})
        self.assertEqual([r["generation_result"]["status"] for r in records], ["success", "truncated", "failed"])
        self.assertEqual(records, importer.convert(rows, "gen_x_1", {"p0", "p1", "p2"}))
        validator = bt._validator("generation_result.schema.json")
        for record in records:
            validator.validate(record)
        with self.assertRaises(ValueError):
            importer.convert(rows + rows[:1], "gen_x_1", {"p0", "p1", "p2"})
        with self.assertRaises(ValueError):
            importer.convert(rows, "gen_x_1", {"p0"})


if __name__ == "__main__":
    unittest.main()
