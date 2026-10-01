"""Execute launchers against a fake training executable; no Python ML stack."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RecipeLaunchers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.capture = self.path / "argv.txt"
        mock = self.path / "capture-argv"
        mock.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
        mock.chmod(0o755)
        self.env = {"PATH": os.environ["PATH"], "HOME": str(self.path), "PYTHON": str(mock),
                    "CAPTURE": str(self.capture), "LIBERO_DATA_ROOT": str(self.path),
                    "VJEPA2_ENCODER_CHECKPOINT": str(self.path / "encoder.pt"),
                    "TEXT_EMBEDDING_CACHE": str(self.path), "OUTDIR": str(self.path / "output")}

    def launch(self, script, **overrides):
        return subprocess.run(["bash", str(ROOT / script)], cwd=ROOT,
                              env=dict(self.env, **overrides), capture_output=True, text=True)

    def arguments(self):
        return self.capture.read_text().splitlines()

    def test_default_libero_and_explicit_hardware_override(self):
        for hardware in ({}, {"NUM_GPUS": "4", "BATCH_SIZE": "32"}):
            result = self.launch("scripts/train_vjepa_policy_fresh_packed48.sh", **hardware)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = self.arguments()
            for flag, value in (("--encoder", "vjepa2_1_vitl"), ("--encoder-checkpoint-key", "ema_encoder"),
                                ("--context-len", "128"), ("--max-state-dim", "48")):
                self.assertEqual(args[args.index(flag) + 1], value)

    def test_batch_mismatch_stops_before_training(self):
        result = self.launch("scripts/train_vjepa_policy_fresh_packed48.sh", NUM_GPUS="4")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_cli_cannot_bypass_global_batch_validation(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/train_vjepa_policy.sh"), "--batch-size", "1"],
                                cwd=ROOT, env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_from_scratch_clears_inherited_predictor(self):
        result = self.launch("scripts/train_vjepa_policy_fresh_packed48.sh", PREDICTOR_INIT="/missing/checkpoint.pt")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--predictor-init", self.arguments())

    def test_pretrained_predictor_requires_and_forwards_weights(self):
        result = self.launch("scripts/train_vjepa_policy_droid_init.sh")
        self.assertNotEqual(result.returncode, 0)
        checkpoint = self.path / "predictor.pt"
        checkpoint.write_bytes(b"fake weights")
        result = self.launch("scripts/train_vjepa_policy_droid_init.sh", PREDICTOR_INIT=str(checkpoint))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        self.assertEqual(args[args.index("--predictor-init") + 1], str(checkpoint))

    def test_gr1_defaults_match_the_paper_batch(self):
        data = self.path / "data"
        for index in range(24):
            meta = data / f"task_{index:02d}" / "meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text("{}")
        artifacts = data / "artifacts"
        artifacts.mkdir()
        (artifacts / "dataset_stats_absolute_minmax_ck16.json").write_text("{}")
        result = self.launch("scripts/gr1/train.sh", DATA_ROOT=str(data), WEIGHTS=str(self.path))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        for flag, value in (("--num_processes", "4"), ("--batch-size", "64"),
                            ("--gradient-accumulation-steps", "1")):
            self.assertEqual(args[args.index(flag) + 1], value)


if __name__ == "__main__":
    unittest.main()
