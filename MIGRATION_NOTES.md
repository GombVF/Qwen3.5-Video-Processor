# Migration notes: Qwen2.5 v5 to Qwen3.5-4B

## Scope

This project is a clean Qwen3.5 edition. It intentionally removes the old
prompt builders, schemas, public window-analysis contract, compatibility
branches, and v1-v5 benchmark scripts. The retained Qwen2.5 v5 JSON under
`data/baselines/` is benchmark evidence, not executable configuration.

## Qwen3.5-specific decisions

| Decision | Implementation |
|---|---|
| Model loader | `AutoModelForMultimodalLM` + `AutoProcessor` |
| Checkpoint | `Qwen/Qwen3.5-4B` or a local snapshot |
| Precision | BF16 only |
| Device | Entire model on `cuda:0` (ROCm compatibility namespace) |
| Attention | SDPA |
| Thinking | Explicitly disabled in the chat template |
| Generation | Deterministic `do_sample=False` for extraction |
| Concurrency | One process, one model, one `threading.Lock` |
| Video input | Native video tensors decoded by `qwen-vl-utils` |
| Repair | Deterministic forward structural salvage only |

Qwen's model card documents that Qwen3.5 is multimodal, uses thinking by
default, and should be loaded through recent Transformers. The Hugging Face
integration example uses `AutoModelForMultimodalLM`. The project deliberately
does not enable Hub kernels or FlashAttention 2: those paths target environments
that are not the current native Windows/AMD ROCm setup. SDPA is the supported,
portable acceleration path here.

The official non-thinking recommendations use sampling for general chat. This
service keeps greedy decoding because its task is constrained JSON extraction,
where repeatability, faithful validation, and benchmark comparability are more
important than conversational diversity.

References:

- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/docs/transformers/model_doc/qwen3_5
- https://github.com/QwenLM/Qwen3-VL

## Pipeline changes

1. A global video pass produces only summary and visually grounded topics.
2. Twelve-second overlapping windows produce typed observations and objects.
3. OCR samples the full video once at 1 FPS and caps the total frame budget.
4. OCR frames are processed in micro-batches of four with explicit per-image
   indices, reducing inference calls without merging text across images.
5. Consolidation is deterministic and returns one timestamp-free video result.

This directly fixes the v5 OCR contract defect where valid bare JSON arrays were
discarded as “no JSON object found.” The validator now accepts a small set of
structurally equivalent OCR shapes, records the normalization, and never changes
the extracted strings.

## Validation status

- 78 tests pass, 2 skipped (no GPU model needed).
- GPU benchmark executed 2026-09-22 on the target Windows 11/ROCm host.
- All 723 parameters loaded in BF16 on `cuda:0` (AMD Radeon RX 9070 XT).
- Peak VRAM: 9.67 GB across all three test videos.
- All passes (global overview, semantic, OCR) completed with `finishReason: stop`.
- Results written to `data/benchmark-qwen35-4b.json`.

### Setup notes discovered during first run

- `transformers` must be installed from the GitHub `main` branch
  (`pip install https://github.com/huggingface/transformers/archive/refs/heads/main.zip`).
  No released PyPI version (up to 5.17.0) includes `qwen3_5` model support or
  `AutoModelForMultimodalLM`. This will change once 5.18+ ships.
- `scripts/setup-env.ps1` contains an em-dash on line 30 that causes a parse
  error in PowerShell 5.1. Replace it with an ASCII dash or run the steps manually.
- `video_kwargs["fps"]` in `src/visual_service/video/decoder.py` was wrapped in
  a list; the newer `huggingface_hub` strict validator requires a scalar float.
  Fixed to pass the value directly.
- The optimized kernels (`causal_conv1d`, `flash-linear-attention`) are not
  available on Windows/ROCm, so inference uses the reference PyTorch fallbacks.
  This is correct but significantly slower.
