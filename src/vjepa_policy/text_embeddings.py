"""Precompute T5 instruction embeddings consumed by the policy dataset."""

import argparse
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm

from vjepa_policy.datasets import DEFAULT_PROMPT, read_instructions


def read_prompts(dataset_dirs, instruction_field="task"):
    prompts = set()
    for dataset_dir in dataset_dirs:
        for instruction in read_instructions(dataset_dir, instruction_field).values():
            prompts.add(DEFAULT_PROMPT.format(task=instruction))
    return sorted(prompts)


def precompute_text_embeddings(
    dataset_dirs,
    cache_dir,
    model_name,
    context_length,
    device="cuda",
    instruction_field="task",
):
    from transformers import AutoTokenizer, T5EncoderModel

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    encoder = T5EncoderModel.from_pretrained(model_name).to(device).eval()

    for prompt in tqdm(
        read_prompts(dataset_dirs, instruction_field),
        desc="Encoding task prompts",
    ):
        unpadded_tokens = tokenizer(
            prompt,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        prompt_length = unpadded_tokens.input_ids.shape[1]
        if prompt_length > context_length:
            raise ValueError(
                f"Prompt requires {prompt_length} tokens but context_length="
                f"{context_length}; increase --context-length to avoid truncation: {prompt!r}"
            )
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        output_path = cache_dir / f"{digest}.t5_len{context_length}.pt"
        if output_path.is_file():
            continue
        tokens = tokenizer(
            prompt,
            max_length=context_length,
            padding="max_length",
            truncation=False,
            return_tensors="pt",
        )
        with torch.no_grad():
            context = encoder(
                input_ids=tokens.input_ids.to(device),
                attention_mask=tokens.attention_mask.to(device),
            ).last_hidden_state[0].cpu()
        torch.save(
            {"context": context, "mask": tokens.attention_mask[0].bool()},
            output_path,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dirs", nargs="+", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-name", default="google/t5-v1_1-xxl")
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction-field", choices=("task", "remarks"), default="task")
    args = parser.parse_args(argv)
    precompute_text_embeddings(
        args.dataset_dirs,
        args.cache_dir,
        args.model_name,
        args.context_length,
        args.device,
        args.instruction_field,
    )


if __name__ == "__main__":
    main()
