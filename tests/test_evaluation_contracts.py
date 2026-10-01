"""CPU-only regression checks for score acceptance; no simulator/model imports."""

import json
from pathlib import Path
import tempfile
import unittest

from examples.libero_plus.protocol import (
    CATEGORIES, CLASSIFICATION_SHA256, REVISION, SUITE_COUNTS,
    canonical_instruction, digest_file, load_manifest, task_instruction, validate_cache, validate_results,
)
from scripts.validate_eval_results import validate_libero


class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        categories = sorted(CATEGORIES)
        self.tasks = [{"suite": suite, "task_id": index, "name": f"{suite}_{index}",
                       "instruction": "put the cup on the table", "category": categories[index % 7]}
                      for suite, count in SUITE_COUNTS.items() for index in range(count)]
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"schema_version": 1, "benchmark_revision": REVISION,
                                            "classification_sha256": CLASSIFICATION_SHA256,
                                            "canonical_task_language": True, "tasks": self.tasks}))
        self.sha = digest_file(self.manifest)

    def write_results(self, tasks):
        for suite in SUITE_COUNTS:
            directory = self.root / suite
            directory.mkdir(exist_ok=True)
            rows = [dict(row, episode_id=0, success=(row["task_id"] % 2 == 0), manifest_sha256=self.sha)
                    for row in tasks if row["suite"] == suite]
            (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_full_score_uses_episode_weighting(self):
        self.write_results(self.tasks)
        result = validate_results(self.root, self.manifest)
        expected = sum(task["task_id"] % 2 == 0 for task in self.tasks)
        self.assertTrue(result["complete"])
        self.assertEqual(result["episodes"], 10030)
        self.assertEqual(result["success_rate"], expected / 10030)
        self.assertEqual(sum(axis["episodes"] for axis in result["categories"].values()), 10030)

    def test_missing_episode_is_diagnostic_only(self):
        self.write_results(self.tasks[:-1])
        with self.assertRaises(ValueError):
            validate_results(self.root, self.manifest)
        result = validate_results(self.root, self.manifest, allow_incomplete=True)
        self.assertFalse(result["complete"])
        self.assertFalse(result["selected_complete"])

    def test_duplicate_is_not_accepted_as_a_partial_result(self):
        self.write_results(self.tasks + [self.tasks[0]])
        with self.assertRaises(ValueError):
            validate_results(self.root, self.manifest, allow_incomplete=True)

    def test_wrong_manifest_or_task_identity_is_rejected(self):
        self.write_results(self.tasks)
        file = self.root / "libero_spatial/episodes.jsonl"
        rows = file.read_text().splitlines()
        for field, value in (("manifest_sha256", "wrong"), ("name", "another_task"), ("success", "error")):
            row = json.loads(rows[0])
            row[field] = value
            file.write_text("\n".join([json.dumps(row), *rows[1:]]) + "\n")
            with self.assertRaises(ValueError):
                validate_results(self.root, self.manifest)

    def test_manifest_requires_unique_full_coverage(self):
        content = json.loads(self.manifest.read_text())
        content["tasks"][-1] = content["tasks"][0]
        self.manifest.write_text(json.dumps(content))
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)

    def test_cache_requires_matching_complete_manifest_and_files(self):
        cache = self.root / "cache"
        cache.mkdir()
        name = "a" * 64 + ".t5_len128.pt"
        (cache / name).write_bytes(b"payload")
        metadata = {"manifest_sha256": self.sha, "context_length": 128, "complete": True,
                    "tasks": 10030, "unique_prompts": 1, "files": [name]}
        record = cache / "libero_plus_cache.json"
        record.write_text(json.dumps(metadata))
        validate_cache(cache, self.manifest)
        (cache / name).unlink()
        with self.assertRaises(ValueError):
            validate_cache(cache, self.manifest)

    def test_instructions_keep_semantics_and_language_perturbations(self):
        name = "pick_up_the_bowl_from_table_center_view_0_0_100_0_0_initstate_3"
        self.assertEqual(canonical_instruction(name), "pick up the bowl from table center")
        self.assertEqual(canonical_instruction("KITCHEN_SCENE10_close_the_drawer_table_1"), "close the drawer")
        self.assertEqual(canonical_instruction("put_the_bowl_on_the_plate_moved_level1"), "put the bowl on the plate")
        self.assertEqual(task_instruction(name, "Language Instructions", "Please lift the bowl."), "Please lift the bowl.")
        self.assertEqual(task_instruction(name, "Sensor Noise", "ignored noise 3"), "pick up the bowl from table center")

    def test_libero_rejects_inconsistent_totals(self):
        result = self.root / "libero.txt"
        content = "Task suite name: libero_spatial\nTotal success rate: 0.5\nTotal episodes: 2\nTotal success: 1\nTask results:\n  [00] 1/2 (0.5000) - test\n"
        result.write_text(content)
        self.assertEqual(validate_libero(result, "libero_spatial", 2, 1), (1, 2))
        result.write_text(content.replace("Total success: 1", "Total success: 2"))
        with self.assertRaises(ValueError):
            validate_libero(result, "libero_spatial", 2, 1)


if __name__ == "__main__":
    unittest.main()
