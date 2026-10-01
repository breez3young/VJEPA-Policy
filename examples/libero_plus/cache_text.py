"""Build the Plus instruction cache in the policy/T5 environment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.libero_plus.protocol import digest_file, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="google/t5-v1_1-xxl")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    from vjepa_policy.datasets.prompts import DEFAULT_PROMPT
    from vjepa_policy.text_embeddings import encode_prompts
    prompts = sorted({DEFAULT_PROMPT.format(task=row["instruction"]) for row in manifest["tasks"]})
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"manifest_sha256": digest_file(args.manifest), "model_name": args.model_name,
                "context_length": args.context_length, "tasks": len(manifest["tasks"]),
                "unique_prompts": len(prompts),
                "files": [f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}.t5_len{args.context_length}.pt"
                          for prompt in prompts]}
    record = args.cache_dir / "libero_plus_cache.json"
    previous = json.loads(record.read_text()) if record.exists() else {}
    previous.pop("complete", None)
    if any(args.cache_dir.iterdir()) and previous != metadata:
        parser.error("Cache directory has different or unknown provenance; choose an empty directory")
    record.write_text(json.dumps(dict(metadata, complete=False), indent=2) + "\n")
    encode_prompts(prompts, args.cache_dir, args.model_name, args.context_length, args.device)
    record.write_text(json.dumps(dict(metadata, complete=True), indent=2) + "\n")
    print(f"Cached {len(prompts)} prompts at context length {args.context_length}")


if __name__ == "__main__":
    main()
