"""Inspect the selected interpreter without installing packages or loading weights."""

import argparse
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import re
import subprocess
import sys

PROFILES = {
    "source": (),
    "policy": ("vjepa_policy", "torch", "torchvision", "transformers", "sentencepiece",
               "accelerate", "lerobot", "openpi-client", "websockets"),
    "libero": ("libero", "torch", "numpy", "robosuite", "mujoco", "bddl",
               "openpi-client", "tyro", "imageio", "termcolor"),
    "libero-plus": ("libero", "torch", "numpy", "robosuite", "mujoco", "bddl",
                    "openpi-client", "tyro", "imageio", "termcolor", "Wand", "scikit-image"),
    "gr1": ("torch", "gymnasium", "gr00t"),
}


def inspect_environment(profile, require_cuda=False):
    report = {"python": platform.python_version(), "interpreter": sys.executable,
              "platform": platform.platform(), "profile": profile, "packages": {}, "errors": []}
    if sys.version_info < (3, 10):
        report["errors"].append("Python 3.10 or newer is required")
    for package in PROFILES[profile]:
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["errors"].append(f"Missing distribution: {package}")
    if profile == "libero-plus":
        # Read exact pins from the install recipe instead of maintaining a second list.
        recipe = Path(__file__).resolve().parents[1] / "examples/libero_plus/requirements.txt"
        for line in recipe.read_text().splitlines():
            match = re.fullmatch(r"([\w.-]+)==([\d.]+)", line.strip())
            if match is None:
                continue
            package, required = match.groups()
            try:
                installed = metadata.version(package)
                report["packages"][package] = installed
                if installed.split("+", 1)[0] != required:
                    report["errors"].append(f"{package}=={required} required by the Plus recipe; found {installed}")
            except metadata.PackageNotFoundError:
                report["errors"].append(f"Missing pinned distribution: {package}")
    if profile != "source":
        checked = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        report["pip_check"] = (checked.stdout + checked.stderr).strip()
        if checked.returncode:
            report["errors"].append("Installed dependency requirements are inconsistent; see pip_check")
    if require_cuda:
        try:
            import torch
            report["cuda"] = {"available": torch.cuda.is_available(), "build": torch.version.cuda,
                              "devices": torch.cuda.device_count()}
            if not report["cuda"]["available"]:
                report["errors"].append("CUDA is required but unavailable")
        except Exception as error:
            report["errors"].append(f"PyTorch/CUDA import failed: {error}")
    report["ok"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="source")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = inspect_environment(args.profile, args.require_cuda)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
