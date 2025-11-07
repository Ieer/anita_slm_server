#!/usr/bin/env python
"""Utility script to download / prepare models under the local `models/` directory.

Features:
  - Download main chat model (e.g. Qwen2.5-0.5B-Instruct) from HuggingFace.
  - Download embedding models (one or multiple) listed via CLI.
  - Optional metadata-only mode (keeps tokenizer/config/vocab; removes large weight files after download).
  - Optional GGUF model fetch (if a repo id to a quantized variant is provided).
  - Safe re-run: skips existing files unless --force given.
  - Colored, concise logging.

Examples:
  python scripts/download_models.py --chat Qwen/Qwen2.5-0.5B-Instruct \
      --embeddings sentence-transformers/paraphrase-MiniLM-L6-v2 embedding-data/embeddinggemma-300m \
      --metadata-only

  python scripts/download_models.py --chat Qwen/Qwen2.5-0.5B-Instruct \
      --gguf qwen/Qwen2.5-0.5B-Instruct-GGUF --allow-pattern "*.gguf"

  python scripts/download_models.py --embeddings sentence-transformers/paraphrase-MiniLM-L6-v2 --target-dir models

Return codes:
  0 success
  1 invalid arguments
  2 download failure
  3 post-processing failure

Requires: huggingface_hub (already transitively present via transformers / sentence-transformers)
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

try:
    from huggingface_hub import snapshot_download
except ImportError as e:  # pragma: no cover
    print("[ERROR] huggingface_hub not installed. Install transformers or run: pip install huggingface_hub", file=sys.stderr)
    raise

# --------------------------- Constants ---------------------------
DEFAULT_CHAT_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_EMBEDDING_REPOS = [
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
]

METADATA_PATTERNS = [
    "config.json",
    "tokenizer*.json",
    "vocab.*",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
    "sentence_bert_config.json",
    "modules.json",
    "README.md",
    "*.md",
    "*.txt",
]

WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.onnx",
    "*.gguf",
]

COLOR = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "reset": "\033[0m",
}


def c(color: str, msg: str) -> str:
    return f"{COLOR.get(color, '')}{msg}{COLOR['reset']}"


# --------------------------- Helpers ---------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def remove_weight_files(root: Path):
    removed = []
    for pattern in WEIGHT_PATTERNS:
        for p in root.rglob(pattern):
            try:
                p.unlink()
                removed.append(p)
            except Exception:  # pragma: no cover
                print(c("red", f"[WARN] Failed removing {p}"))
    return removed


def list_relative_files(root: Path) -> list[str]:
    return [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]


def match_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def filter_files(files: Iterable[str], allow: Sequence[str] | None, ignore: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for f in files:
        if ignore and match_any(f, ignore):
            continue
        if allow and not match_any(f, allow):
            continue
        out.append(f)
    return out


# --------------------------- Download Logic ---------------------------

def download_repo(repo_id: str, local_dir: Path, allow_patterns: Sequence[str] | None, ignore_patterns: Sequence[str] | None, revision: str | None, force: bool) -> Path:
    ensure_dir(local_dir)
    if any(local_dir.iterdir()) and not force:
        print(c("yellow", f"[SKIP] {repo_id} already exists at {local_dir} (use --force to re-download)"))
        return local_dir
    try:
        print(c("blue", f"[DL ] {repo_id} -> {local_dir}"))
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
            revision=revision,
            local_dir_use_symlinks=False,
        )
        print(c("green", f"[OK ] {repo_id}"))
    except Exception as e:  # pragma: no cover
        print(c("red", f"[FAIL] {repo_id}: {e}"), file=sys.stderr)
        sys.exit(2)
    return local_dir


# --------------------------- Main CLI ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download chat & embedding models into ./models")
    p.add_argument("--chat", help="Chat model repo id (HuggingFace)", default=None)
    p.add_argument("--embeddings", nargs="*", default=[], help="Embedding model repo ids")
    p.add_argument("--gguf", help="Optional GGUF quantized repo id", default=None)
    p.add_argument("--target-dir", default="models", help="Root directory to place models")
    p.add_argument("--revision", default=None, help="Git revision / tag of model repo")
    p.add_argument("--metadata-only", action="store_true", help="Keep only tokenizer/config/vocab (remove weights)")
    p.add_argument("--allow-pattern", action="append", default=[], help="Extra allow pattern (repeatable)")
    p.add_argument("--ignore-pattern", action="append", default=[], help="Extra ignore pattern (repeatable)")
    p.add_argument("--force", action="store_true", help="Force re-download if directory exists")
    p.add_argument("--show", action="store_true", help="Only show planned actions (dry run)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_root = Path(args.target_dir)
    ensure_dir(target_root)

    plan: list[tuple[str, Path, list[str] | None, list[str] | None]] = []

    if args.chat:
        plan.append((args.chat, target_root / "qwen" / Path(args.chat).name, None, None))
    if args.gguf:
        plan.append((args.gguf, target_root / "qwen" / Path(args.gguf).name, ["*.gguf"], None))
    if args.embeddings:
        for rid in args.embeddings:
            plan.append((rid, target_root / "embedding" / Path(rid).name, None, None))

    if not plan:
        print(c("yellow", "[INFO] No repos specified. Use --chat / --embeddings / --gguf."))
        return 1

    if args.show:
        print(c("blue", "Planned downloads:"))
        for rid, loc, ap, ip in plan:
            print(f"  - {rid} -> {loc} allow={ap} ignore={ip}")
        return 0

    for rid, loc, ap, ip in plan:
        allow = ap or []
        # Extend allow patterns if metadata-only mode
        merged_allow = list(allow)
        if args.metadata_only:
            merged_allow.extend(METADATA_PATTERNS)
        if args.allow_pattern:
            merged_allow.extend(args.allow_pattern)
        ignore = list(args.ignore_pattern)
        # If metadata-only: we pre-limit by allow; if not metadata-only but user gave allow, pass it
        download_repo(rid, loc, merged_allow if merged_allow else None, ignore if ignore else None, args.revision, args.force)
        if args.metadata_only and not args.allow_pattern:
            # Safety post-pass removal in case hub delivered extra weight files
            removed = remove_weight_files(loc)
            if removed:
                print(c("yellow", f"[CLEAN] removed {len(removed)} weight files"))

    # Summary
    print(c("green", "All done."))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
