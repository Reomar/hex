# QwenCleo-ASR in Hex

## Goal and scope

Add an optional local QwenCleo-ASR model for Egyptian Arabic and Arabic/English
code-switching, then build and exercise an Apple silicon Hex Test app. Preserve
the current recommended models and first-run defaults until independent speech
evaluation justifies a recommendation change. Do not add Python, CUDA, a model
server, or a second inference scheduler to the distributed app.

Source baseline: Hex `933aa800fdd47928162103440024a257b956efa9`, from `dev`.
Development fork: <https://github.com/Reomar/hex>, branch `feat/qwencleo-asr`.
Use a pull request to the fork's `dev`; do not write to upstream or `production`.

## Verified architectural fit

- Hex's model catalog is `src/transcription_models.rs`: persisted IDs,
  language capabilities, picker order, immutable artifact revisions, byte
  lengths, SHA-256, download progress, cancellation, verification and receipts.
- `src/rune_ui/picker.rs` renders catalog entries and sends `ChooseModel`.
  `src/transcription_preparation.rs` owns preparation and activation admission.
- On macOS, `src/transcription.rs` dispatches GGUF models to the existing
  `Parakeet` adapter in `src/parakeet.rs` (the name includes non-Parakeet models).
  It requires Metal. Linux/Windows use `src/local_transcriber.rs`.
- `src/gguf_session.rs` validates architecture and variant and handles sessions,
  cancellation and long-audio work. Keep that path and the existing engine.
- Hex pins `transcribe.cpp` to
  `7045d1c0fe7cee37747b2792977421d435e75fca`. That revision already includes
  Qwen3-ASR 1.7B inference, conversion and quantization.
- QwenCleo's source checkpoint is `mohammedaly22/QwenCleo-ASR`, a full
  SafeTensors Qwen3-ASR 1.7B fine-tune, declared Apache-2.0. Pin its complete
  source commit before conversion. Its config uses encoder width 1024,
  decoder width 2048, 28 decoder layers and tied embeddings.
- The author's `mohammedaly22/QwenCleo-ASR-GGUF` packages target `audio.cpp`.
  A GGUF extension does not establish compatibility with `transcribe.cpp`;
  use its own converter and verify the output instead of relabeling a file.

## Phase 1: reproducible artifact

1. Check out the exact pinned backend outside the Hex source tree. Use its
   isolated Python 3.12 conversion environment. Install build tools separately;
   they are developer dependencies, not app dependencies.
2. Resolve and record the source checkpoint's full Hugging Face commit. Download
   only SafeTensors and the config/tokenizer/processor inputs. Do not load
   `training_args.bin` or require remote model code.
3. Run `scripts/convert-qwen3_asr.py` with explicit
   `--variant qwen3-asr-1.7b`. Its default derives an unsupported variant from
   the QwenCleo repository name. Preserve the architecture variant; give the
   artifact and the Hex catalog entry a distinct QwenCleo identity. Correct
   converted attribution/source metadata without changing tokenizer or tensors.
4. Verify source embedding/head tying, conversion tensor names, dimensions,
   tensor types, tokenizer/audio IDs, language metadata and variant. Compare
   relevant configuration against the backend's supported 1.7B architecture.
5. Build `transcribe-quantize` from that backend and quantize to Q8_0. Hex's
   Linux/Windows Vulkan shader subset already supports Q8_0; do not introduce
   Q4/Q5 tensors. Hash the final file and record its exact length.
6. Authenticate to Hugging Face locally; never commit or print credentials.
   Publish a dedicated converted-model repository with the model, Apache-2.0
   attribution, source/backend commits, conversion recipe, quantization and
   honest verification limits. Resolve the resulting immutable commit.
7. Compare BF16 and Q8 outputs on representative, explicitly supplied Egyptian
   and mixed-language audio. Distinguish synthetic fixture smokes from human
   speech quality. Do not reproduce the author's benchmark as a Hex result.

Artifact gate: no placeholder repository, fabricated checksum, or nonfunctional
Download button in the shipping catalog. Register only the verified artifact.

## Phase 2: catalog integration

1. Add `QwenCleoAsr17B` with explicit stable wire name `qwencleo_asr_1_7b` in
   serde, `as_str` and `FromStr`. Preserve all previous wire names.
2. Add a `GgufArtifact` with the published repository, immutable revision,
   unique filename ending in `-Q8_0.gguf`, exact size, SHA-256,
   architecture `qwen3_asr`, and variant `qwen3-asr-1.7b`.
3. Add a model definition named `QwenCleo-ASR 1.7B`. Offer Arabic (`ar`), which
   maps to the existing Arabic hint; mixed English speech does not require
   declaring general English support. Do not claim 30-language quality from
   inherited metadata. Start with automatic detection disabled because the
   model author recommends an explicit Arabic hint for quantized weights.
4. Mark streaming, timestamps and recognition hints unsupported. Keep existing
   default/recommendation routing. The picker already lists other compatible
   models after recommendations. A switch from another language must visibly
   fall back to Arabic and remain valid after restart.
5. Do not invent speed or accuracy. Use an unmeasured label and absent WER until
   measurements exist; handle new translated text through the existing i18n
   system. Describe Egyptian Arabic/code-switching in the model card.
6. Add regression coverage for wire names, Arabic-only selection/fallback,
   language hint, truthful capabilities, picker availability, unchanged
   recommendations, pinned artifact identity, and absence of timestamps or
   streaming. Reuse existing downloader integrity/cancellation tests.

## Phase 3: checks, review and distribution

1. Run formatting, catalog/settings regression tests and strict Clippy. Run
   the repository checks on a suitable runner; record failures with their
   boundary rather than treating a queued check as a pass.
2. Update `docs/features/README.md` beside the existing model picker map,
   documenting entry point, Arabic fallback, capabilities and proof limits.
3. Regenerate Linux Rune goldens for the visible picker change. On macOS,
   dispatch the existing `check-linux.yml` on the fork branch, obtain
   `golden-actual`, inspect every changed image, commit the matching goldens
   and rerun. Do not hide unrelated snapshot failures.
4. Push only scoped source, docs and golden changes. Open a PR against the
   fork's `dev`; merge only after its applicable checks pass. Keep the model
   binaries out of Git history.
5. Build an ARM64 app with the repository's existing `test-builds.yml`:
   `HEX_TEST_BUILD=1`, separate `anomaly.hex.test` bundle identity, updates off,
   separate `hex-test` runtime data. This Mac lacks Rust and full Xcode/Metal,
   so prefer the macOS GitHub runner over a stubbed local build. Do not change
   `.github` workflows merely to work around this machine's prerequisites.
6. Download the exact run's macOS artifact, inspect its executable architecture,
   bundle ID, version, signature and dependencies, and install Hex Test beside
   Hex. Preserve any pre-existing test installation before replacement.

## Phase 4: Apple silicon acceptance

1. Run bundled CLI version/model listing and model installation into test data.
   Confirm the downloaded byte count/hash and actual Metal model load.
2. Run an explicit WAV transcription smoke through the bundled executable.
   Exercise silence and a longer clip; record outputs, backend, load time,
   warm inference time and memory without publishing private recordings.
3. Quit regular Hex cleanly before launching Hex Test so shortcuts and the
   microphone do not compete. The user grants Microphone and Accessibility
   to the new test identity; do not automate or bypass macOS consent.
4. Open the picker, select Arabic and QwenCleo, observe progress and successful
   activation, then restart and verify the persisted choice.
5. With the user, dictate Egyptian Arabic, mixed technical English terms,
   numbers, pauses and a longer passage into a disposable text surface.
   Check cancellation, paste, Paste Last and switching back to an existing
   installed model. Compare expected and actual text and perceived latency.
6. Keep build success, fixture inference, physical recording/paste and speech
   accuracy as separate claims. The user supplies/records the evaluation
   speech; no unattended capture.

Rollback: quit Hex Test and reopen Hex. Both model selection and test data stay
isolated. General Arabic recommendation changes, other OS device testing,
streaming and word alignment are future work, not acceptance prerequisites for
this optional Apple silicon test integration.

## Execution record

- Fork created and base branch pushed; implementation not yet complete.
- No checkpoint conversion, inference, checks, golden regeneration, compiled
  app or physical dictation has passed yet. Replace this line with dated
  observed evidence as phases finish.
