#!/usr/bin/env python3
"""Convert the pinned QwenCleo checkpoint with Hex's transcribe.cpp converter.

Run in scripts/envs/qwen3_asr from the pinned transcribe.cpp checkout; see
docs/plans/qwencleo-asr.md. This tool is not part of the app runtime.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

BACKEND_COMMIT = "7045d1c0fe7cee37747b2792977421d435e75fca"
SOURCE_REPO = "mohammedaly22/QwenCleo-ASR"
SOURCE_COMMIT = "50f6b901a3b882bdf28e71efd91f9f561510fc6f"
WEIGHTS_SHA256 = "d9108368a45f890cb4b65527cfd3e9d26a0481cafe765682bb19142424e502dc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", type=Path, help="Exact transcribe.cpp checkout")
    parser.add_argument("source", type=Path, help="Pinned local HF checkpoint")
    parser.add_argument("output", type=Path, help="Output BF16 GGUF")
    args = parser.parse_args()
    backend = args.backend.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(backend), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != BACKEND_COMMIT:
        parser.error(f"backend must be at {BACKEND_COMMIT}, got {revision}")
    if subprocess.check_output(
        ["git", "-C", str(backend), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        parser.error("backend has tracked changes; use an unmodified checkout")
    if output.exists():
        parser.error(f"output already exists: {output}")

    with (source / "model.safetensors").open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    if digest != WEIGHTS_SHA256:
        parser.error(f"source weights differ from {SOURCE_REPO}@{SOURCE_COMMIT}")
    config = json.loads((source / "config.json").read_text())
    thinker = config["thinker_config"]
    audio, text = thinker["audio_config"], thinker["text_config"]
    expected = {
        "model_type": (config["model_type"], "qwen3_asr"),
        "encoder width": (audio["d_model"], 1024),
        "decoder width": (text["hidden_size"], 2048),
        "decoder layers": (text["num_hidden_layers"], 28),
        "tied embeddings": (text["tie_word_embeddings"], True),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            parser.error(f"unexpected {name}: {actual!r}, expected {wanted!r}")

    # The backend converter drops the tied LM head. Verify this assumption for
    # the fine-tune too, rather than silently dropping independently trained
    # output weights. Some serializers omit the duplicate tied tensor.
    import torch
    from safetensors import safe_open

    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as tensors:
        if "thinker.lm_head.weight" in tensors.keys() and not torch.equal(
            tensors.get_tensor("thinker.lm_head.weight"),
            tensors.get_tensor("thinker.model.embed_tokens.weight"),
        ):
            parser.error("LM head differs from embeddings; converter must preserve it")

    sys.path.insert(0, str(backend / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "convert_qwen3_asr", backend / "scripts/convert-qwen3_asr.py"
    )
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    original_identity = converter.add_general_identity

    def identity(writer, **metadata):
        metadata.update(
            name="QwenCleo-ASR 1.7B",
            basename="qwencleo-asr",
            author="Mohammed Aly",
            organization="Mohammed Aly",
            repo_url=f"https://huggingface.co/{SOURCE_REPO}/tree/{SOURCE_COMMIT}",
        )
        original_identity(writer, **metadata)

    converter.add_general_identity = identity
    output.parent.mkdir(parents=True, exist_ok=True)
    converter.convert(source, output, "qwen3-asr-1.7b", repo_id=SOURCE_REPO)
    print(f"Source: {SOURCE_REPO}@{SOURCE_COMMIT}")
    print(f"Backend: {BACKEND_COMMIT}")
    print("Next: transcribe-quantize INPUT OUTPUT --quant Q8_0")


if __name__ == "__main__":
    main()
