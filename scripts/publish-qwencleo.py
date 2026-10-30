#!/usr/bin/env python3
"""Audit a converted QwenCleo Q8 GGUF and optionally publish it to the Hub."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from gguf import GGMLQuantizationType, GGUFReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--repo", default="reomar15/QwenCleo-ASR-transcribe-cpp-GGUF")
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve()
    if model.name != "QwenCleo-ASR-1.7B-Q8_0.gguf":
        parser.error("expected the QwenCleo Q8_0 artifact filename")
    reader = GGUFReader(model)
    expected = {
        "general.architecture": "qwen3_asr",
        "general.name": "QwenCleo-ASR 1.7B",
        "general.author": "Mohammed Aly",
        "stt.variant": "qwen3-asr-1.7b",
    }
    for key, wanted in expected.items():
        field = reader.get_field(key)
        if field is None or field.contents() != wanted:
            parser.error(f"unexpected or missing {key}")
    allowed = {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.Q8_0,
    }
    types = {tensor.tensor_type for tensor in reader.tensors}
    if not types.issubset(allowed) or GGMLQuantizationType.Q8_0 not in types:
        parser.error(f"tensor types outside Hex's Vulkan subset: {types}")
    with model.open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    manifest = {
        "filename": model.name,
        "bytes": model.stat().st_size,
        "sha256": digest,
        "architecture": "qwen3_asr",
        "variant": "qwen3-asr-1.7b",
        "source_repository": "mohammedaly22/QwenCleo-ASR",
        "source_revision": "50f6b901a3b882bdf28e71efd91f9f561510fc6f",
        "backend_revision": "7045d1c0fe7cee37747b2792977421d435e75fca",
        "tensor_count": len(reader.tensors),
        "tensor_types": sorted(t.name for t in types),
        "license": "apache-2.0",
    }
    # Publication files live beside the generated model, outside Git history.
    folder = model.parent
    (folder / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(args.license, folder / "LICENSE")
    (folder / "README.md").write_text(f"""---
license: apache-2.0
language:
- ar
- en
base_model: mohammedaly22/QwenCleo-ASR
pipeline_tag: automatic-speech-recognition
tags:
- gguf
- transcribe.cpp
- egyptian-arabic
- code-switching
---

# QwenCleo-ASR for transcribe.cpp / Hex

Q8_0 conversion of [Mohammed Aly's QwenCleo-ASR](https://huggingface.co/mohammedaly22/QwenCleo-ASR),
a Qwen3-ASR 1.7B fine-tune for Egyptian Arabic and Arabic/English code-switching.
Base model: [Alibaba Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).
The model and base model are declared Apache-2.0; see LICENSE.
This packaging is for transcribe.cpp, not the author's audio.cpp GGUF runtime.

## Provenance

- Source checkpoint: `{manifest['source_repository']}@{manifest['source_revision']}`.
- transcribe.cpp converter and quantizer: `{manifest['backend_revision']}`.
- File: `{model.name}`; bytes: `{manifest['bytes']}`.
- SHA-256: `{digest}`.
- Architecture: `qwen3_asr`; variant: `qwen3-asr-1.7b`.
- Tensor types: {', '.join(manifest['tensor_types'])}.
- Recipe: [Hex fork](https://github.com/Reomar/hex/tree/feat/qwencleo-asr),
  `scripts/convert-qwencleo.py`, then `transcribe-quantize --quant Q8_0`.
  The wrapper checks the pinned source weight hash and tied embeddings and
  changes attribution metadata; the backend performs tensor conversion.

## Usage and limits

Use transcribe.cpp with 16 kHz mono audio and an explicit `ar` language hint:

```sh
transcribe-cli -m {model.name} --language ar speech.wav
```

The native backend supports offline transcription, cancellation and automatic
long-audio chunking in Hex. It does not provide word timestamps or streaming.
The Hex catalog exposes Arabic only: mixed English terms are the intended use,
not a claim of general English or inherited 30-language quality.

Structural checks and native fixture inference are preparation checks, not an
Egyptian speech benchmark, physical microphone test or installed Hex acceptance.
No independent Egyptian WER or speed is claimed for this converted artifact.
The source author's reported benchmark belongs to that source checkpoint and
evaluation setup; do not treat it as a measurement of this Q8 conversion.
""")
    print(json.dumps(manifest, indent=2))
    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        username = api.whoami()["name"]
        if args.repo.split("/", 1)[0] != username:
            parser.error("publication repository must belong to the signed-in user")
        commit = api.upload_folder(
            repo_id=args.repo,
            repo_type="model",
            folder_path=str(folder),
            allow_patterns=[model.name, "README.md", "LICENSE", "artifact.json"],
            commit_message="Add pinned QwenCleo Q8_0 conversion for transcribe.cpp",
        )
        manifest["repository"] = args.repo
        manifest["revision"] = commit.oid
        (folder / "published.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Published {args.repo}@{commit.oid}")


if __name__ == "__main__":
    main()
