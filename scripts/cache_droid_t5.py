#!/usr/bin/env python
"""Build the packed, non-truncating T5 cache used by DROID pre-training.

The cache is intentionally prompt-indexed (rather than digest-indexed):
``index.json`` maps each complete prompt string to a row in ``embeddings.bin``.
This keeps cache construction and runtime lookup auditable and avoids a second
source of prompt identity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from vjepa_policy.datasets.droid import (
    DROID_BENCH_CACHE_ROOT,
    DROID_INSTRUCTION_KEYS,
    DROID_ROOT,
    DROID_T5_CONTEXT_LENGTH,
    DROID_PROMPT,
    droid_prompt,
    load_droid_info,
)


DEFAULT_T5_MODEL = "google/t5-v1_1-xxl"


def _episode_path(root: Path, template: str, episode_id: int, chunk_size: int) -> Path:
    return root / template.format(
        episode_chunk=episode_id // chunk_size,
        episode_index=episode_id,
    )


def iter_droid_instruction_candidates(
    root: str | os.PathLike[str] = DROID_ROOT,
    *,
    scan_all_rows: bool = False,
):
    """Yield ``(episode_index, candidates)`` from every DROID parquet.

    DROID_v21 stores each instruction column as an episode-constant value.  The
    default reads one row per parquet, avoiding a 17.9M-row text scan while
    still enumerating every episode file.  ``scan_all_rows=True`` is available
    for validating a newly converted release or for small test fixtures.
    """

    import pyarrow.parquet as parquet

    root_path = Path(root)
    info = load_droid_info(root_path)
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunk_size = int(info.get("chunks_size", 1000))
    episode_file = root_path / "meta" / "episodes.jsonl"
    episode_ids: list[int] = []
    with episode_file.open(encoding="utf-8") as handle:
        for line in handle:
            episode_ids.append(int(json.loads(line)["episode_index"]))
    for episode_id in episode_ids:
        path = _episode_path(root_path, template, episode_id, chunk_size)
        if not path.is_file():
            raise FileNotFoundError(f"Missing DROID episode parquet: {path}")
        parquet_file = parquet.ParquetFile(path)
        if scan_all_rows:
            for batch in parquet_file.iter_batches(
                columns=list(DROID_INSTRUCTION_KEYS), batch_size=8192
            ):
                columns = batch.to_pydict()
                for row in range(batch.num_rows):
                    yield (
                        episode_id,
                        tuple(columns[key][row] for key in DROID_INSTRUCTION_KEYS),
                    )
        else:
            batch = parquet_file.read_row_group(
                0,
                columns=list(DROID_INSTRUCTION_KEYS),
            ).slice(0, 1)
            columns = batch.to_pydict()
            yield episode_id, tuple(columns[key][0] for key in DROID_INSTRUCTION_KEYS)


def collect_droid_prompts(
    root: str | os.PathLike[str] = DROID_ROOT,
    *,
    prompt_template: str = DROID_PROMPT,
    scan_all_rows: bool = False,
) -> list[str]:
    """Collect stable, deduplicated canonical prompts for all instruction columns."""

    prompts: set[str] = set()
    for _, candidates in iter_droid_instruction_candidates(
        root, scan_all_rows=scan_all_rows
    ):
        for candidate in candidates:
            if candidate is None:
                continue
            if not isinstance(candidate, str):
                candidate = str(candidate)
            if candidate.strip():
                prompts.add(droid_prompt(candidate, prompt_template))
    if not prompts:
        raise ValueError(f"No non-empty DROID instructions found under {root}")
    return sorted(prompts)


def _input_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer(
        prompt,
        padding=False,
        truncation=False,
        return_tensors=None,
    )
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _token_lengths(tokenizer: Any, prompts: list[str]) -> list[int]:
    return [len(_input_ids(tokenizer, prompt)) for prompt in prompts]


def _hidden_state(output: Any) -> torch.Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        if isinstance(output, (tuple, list)):
            hidden = output[0]
        elif isinstance(output, Mapping):
            hidden = output["last_hidden_state"]
    if hidden is None or not torch.is_tensor(hidden):
        raise TypeError("T5 encoder output has no tensor last_hidden_state")
    return hidden


def build_packed_droid_t5_cache(
    prompts: list[str],
    cache_dir: str | os.PathLike[str],
    tokenizer: Any,
    encoder: Any,
    *,
    context_length: int = DROID_T5_CONTEXT_LENGTH,
    batch_size: int = 1,
    dtype: str | np.dtype = "float16",
    model_dtype: str | None = None,
    model_name: str | None = None,
    prompt_template: str = DROID_PROMPT,
) -> dict[str, Any]:
    """Encode prompts and write ``embeddings.bin``, ``index.json`` and manifest."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not prompts or len(set(prompts)) != len(prompts):
        raise ValueError("prompts must be non-empty and duplicate-free")
    prompts = sorted(prompts)
    lengths = _token_lengths(tokenizer, prompts)
    observed_max = max(lengths)
    if observed_max > context_length:
        raise ValueError(
            f"Prompt requires {observed_max} tokens but context_length={context_length}; "
            "increase the cache length; truncation is disabled."
        )

    # NumPy has no portable bfloat16 memmap dtype; retain bfloat16 encoder
    # output as float32 on disk when a caller requests it.
    dtype_name = str(dtype).replace("torch.", "")
    target_dtype = np.dtype("float32" if dtype_name in {"bfloat16", "bf16"} else dtype_name)
    if target_dtype.kind != "f":
        raise ValueError(f"Embedding dtype must be floating point, got {target_dtype}")
    config = getattr(encoder, "config", None)
    embedding_dim = getattr(config, "d_model", None) or getattr(
        config, "hidden_size", None
    )
    if embedding_dim is None:
        embedding_dim = getattr(encoder, "d_model", None)
    embedding_dim = int(embedding_dim or 0)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    embedding_tmp = cache_path / f".embeddings.bin.tmp.{os.getpid()}"
    index_tmp = cache_path / f".index.json.tmp.{os.getpid()}"
    manifest_tmp = cache_path / f".manifest.json.tmp.{os.getpid()}"
    index: dict[str, dict[str, int]] = {}
    mmap: np.memmap | None = None
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            tokens = tokenizer(
                batch_prompts,
                padding="max_length",
                max_length=context_length,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            if input_ids.ndim != 2 or input_ids.shape[1] != context_length:
                raise ValueError(
                    "Tokenizer returned a sequence length different from the cache contract "
                    f"({tuple(input_ids.shape)} vs (*, {context_length})); truncation is disabled."
                )
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "Tokenizer input_ids and attention_mask shapes disagree"
                )
            try:
                encoder_device = next(encoder.parameters()).device
            except (AttributeError, StopIteration):
                encoder_device = input_ids.device
            with torch.inference_mode():
                hidden = (
                    _hidden_state(
                        encoder(
                            input_ids=input_ids.to(encoder_device),
                            attention_mask=attention_mask.to(encoder_device),
                        )
                    )
                    .detach()
                    .cpu()
                )
            if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
                raise ValueError(f"Unexpected T5 hidden shape {tuple(hidden.shape)}")
            if embedding_dim == 0:
                embedding_dim = int(hidden.shape[-1])
                mmap = np.memmap(
                    embedding_tmp,
                    mode="w+",
                    dtype=target_dtype,
                    shape=(len(prompts), context_length, embedding_dim),
                )
            elif hidden.shape[-1] != embedding_dim:
                raise ValueError(
                    f"T5 hidden dimension changed from {embedding_dim} to {hidden.shape[-1]}"
                )
            if mmap is None:
                mmap = np.memmap(
                    embedding_tmp,
                    mode="w+",
                    dtype=target_dtype,
                    shape=(len(prompts), context_length, embedding_dim),
                )
            mmap[start : start + len(batch_prompts)] = (
                hidden.float().numpy().astype(target_dtype, copy=False)
            )
            for offset, (prompt, mask) in enumerate(
                zip(batch_prompts, attention_mask.tolist(), strict=True)
            ):
                valid_length = int(sum(bool(value) for value in mask))
                index[prompt] = {
                    "row": start + offset,
                    "length": valid_length,
                    "valid_length": valid_length,
                }
        if mmap is None or embedding_dim <= 0:
            raise RuntimeError("Encoder produced no embeddings")
        mmap.flush()
        del mmap
        mmap = None
        os.replace(embedding_tmp, cache_path / "embeddings.bin")
        index_payload = {"version": 1, "prompts": index}
        index_tmp.write_text(
            json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(index_tmp, cache_path / "index.json")
        manifest = {
            "version": 1,
            "num_prompts": len(prompts),
            "num_embeddings": len(prompts),
            "context_length": int(context_length),
            "context_len": int(context_length),
            "embedding_dim": int(embedding_dim),
            "shape": [len(prompts), int(context_length), int(embedding_dim)],
            "dtype": target_dtype.name,
            "model_dtype": model_dtype,
            "embedding_file": "embeddings.bin",
            "index_file": "index.json",
            "model_name": model_name,
            "prompt_template": prompt_template,
            "instruction_keys": list(DROID_INSTRUCTION_KEYS),
            "max_token_length": int(observed_max),
            "max_length": int(context_length),
        }
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(manifest_tmp, cache_path / "manifest.json")
        return manifest
    finally:
        if mmap is not None:
            mmap.flush()
            del mmap
        for temporary in (embedding_tmp, index_tmp, manifest_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def build_droid_t5_cache(
    dataset_root: str | os.PathLike[str] = DROID_ROOT,
    cache_dir: str | os.PathLike[str] = DROID_BENCH_CACHE_ROOT,
    *,
    model_name: str = DEFAULT_T5_MODEL,
    context_length: int = DROID_T5_CONTEXT_LENGTH,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: str = "float16",
    model_dtype: str = "bfloat16",
    scan_all_rows: bool = False,
    prompt_template: str = DROID_PROMPT,
) -> dict[str, Any]:
    """Load T5 and build the complete DROID cache."""

    from transformers import AutoTokenizer, T5EncoderModel

    prompts = collect_droid_prompts(
        dataset_root,
        prompt_template=prompt_template,
        scan_all_rows=scan_all_rows,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    lengths = _token_lengths(tokenizer, prompts)
    observed_max = max(lengths)
    if observed_max > context_length:
        raise ValueError(
            f"Observed DROID prompt length {observed_max} exceeds context_length={context_length}; "
            "refusing to truncate."
        )
    if model_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported T5 model dtype: {model_dtype}")
    encoder = T5EncoderModel.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, model_dtype),
    ).to(device).eval()
    return build_packed_droid_t5_cache(
        prompts,
        cache_dir,
        tokenizer,
        encoder,
        context_length=context_length,
        batch_size=batch_size,
        dtype=dtype,
        model_dtype=model_dtype,
        model_name=model_name,
        prompt_template=prompt_template,
    )


# Concise aliases for callers that use the generic cache terminology.
build_packed_t5_cache = build_packed_droid_t5_cache
cache_droid_t5 = build_droid_t5_cache


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DROID_ROOT)
    parser.add_argument("--cache-dir", default=DROID_BENCH_CACHE_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_T5_MODEL)
    parser.add_argument("--context-length", type=int, default=DROID_T5_CONTEXT_LENGTH)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float16", "float32", "bfloat16"), default="float16"
    )
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Inference dtype for T5; cache dtype is controlled by --dtype.",
    )
    parser.add_argument(
        "--scan-all-rows",
        action="store_true",
        help="Read every parquet row (the verified DROID release is episode-constant).",
    )
    args = parser.parse_args(argv)
    manifest = build_droid_t5_cache(
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        context_length=args.context_length,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        model_dtype=args.model_dtype,
        scan_all_rows=args.scan_all_rows,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
