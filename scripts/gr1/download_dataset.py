"""Download the 24 RoboCasa GR-1 post-training datasets via the Hub API."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

from scripts.gr1.dataset_manifest import (
    DEFAULT_DATA_ROOT,
    REPO_ID,
    REPO_REVISION,
    TASK_NAMES,
)


HF_ENDPOINT = "https://huggingface.co"
LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"
RETRYABLE_DOWNLOAD_ERRORS = (OSError, RuntimeError, requests.RequestException)


def _remove_lfs_pointer(path: Path) -> None:
    """Remove a checked-out Git LFS pointer without touching real data."""
    if not path.is_file() or path.stat().st_size > 1024:
        return
    with path.open("rb") as handle:
        if handle.read(len(LFS_POINTER_HEADER)) == LFS_POINTER_HEADER:
            path.unlink()


def _list_task_files(api: HfApi, task: str, revision: str) -> list[RepoFile]:
    files = [
        entry
        for entry in api.list_repo_tree(
            REPO_ID,
            path_in_repo=task,
            recursive=True,
            revision=revision,
            repo_type="dataset",
        )
        if isinstance(entry, RepoFile)
    ]
    if not files:
        raise RuntimeError(f"Hub API returned no files for {task}")
    prefix = f"{task}/"
    if any(not entry.path.startswith(prefix) for entry in files):
        raise RuntimeError(f"Hub API returned a path outside {task}")
    return files


def _download_file(
    entry: RepoFile,
    *,
    local_dir: Path,
    revision: str,
    endpoint: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> None:
    for attempt in range(max_retries + 1):
        try:
            destination = local_dir / entry.path
            _remove_lfs_pointer(destination)
            downloaded = Path(
                hf_hub_download(
                    REPO_ID,
                    entry.path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=local_dir,
                    endpoint=endpoint,
                    force_download=False,
                )
            )
            actual_size = downloaded.stat().st_size
            if actual_size != entry.size:
                raise RuntimeError(
                    f"Size mismatch for {entry.path}: "
                    f"expected {entry.size}, got {actual_size}"
                )
            return
        except RETRYABLE_DOWNLOAD_ERRORS as error:
            if attempt == max_retries:
                raise
            delay = retry_delay * (2**attempt)
            print(
                f"[{entry.path}] {type(error).__name__}: {error}; "
                f"retry {attempt + 1}/{max_retries} in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)


def _download_task(
    api: HfApi,
    task: str,
    *,
    local_dir: Path,
    revision: str,
    endpoint: str,
    max_workers: int,
    max_retries: int,
    retry_delay: float,
) -> None:
    files = _list_task_files(api, task, revision)
    total_bytes = sum(entry.size for entry in files)
    print(
        f"[{task}] downloading {len(files)} files ({total_bytes / 1e9:.2f} GB)",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _download_file,
                entry,
                local_dir=local_dir,
                revision=revision,
                endpoint=endpoint,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            for entry in files
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 100 == 0 or completed == len(files):
                print(f"[{task}] {completed}/{len(files)} files", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--revision", default=REPO_REVISION)
    parser.add_argument("--endpoint", default=HF_ENDPOINT)
    parser.add_argument(
        "--max-workers",
        "--lfs-concurrency",
        dest="max_workers",
        type=int,
        default=16,
        help="number of concurrent Hugging Face API file downloads",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if args.retry_delay < 0:
        parser.error("--retry-delay must be non-negative")

    local_dir = args.local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=args.endpoint)
    for index, task in enumerate(TASK_NAMES, start=1):
        print(f"Task {index}/{len(TASK_NAMES)}", flush=True)
        _download_task(
            api,
            task,
            local_dir=local_dir,
            revision=args.revision,
            endpoint=args.endpoint,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

    print(f"Downloaded {len(TASK_NAMES)} task directories to {local_dir}")
    print("Run scripts/gr1/prepare.sh to validate all 24,000 episodes.")


if __name__ == "__main__":
    main()
