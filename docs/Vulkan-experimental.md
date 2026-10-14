# Experimental Windows CPU/Vulkan adapter

This branch adds an experimental host-backed Vulkan bridge to FreeVideo's
streamed VDN-H3 reference pipeline. **It is not official AMD support, a fully
GPU-native engine, or a demonstrated performance improvement.** Only Windows x64
with an AMD Radeon 890M was tested. Linux, other AMD cards, Intel and NVIDIA Vulkan
devices have not been validated by this branch.

The Legion Go 2 did generate a complete clip, but it did not meet the desired
speed goal: **23m41s for 1.625 seconds at 256×256**. Text encoding used the hybrid
Vulkan bridge; sampling and decoding used the faster CPU reference control.
The strict Vulkan sampler was interrupted before completing its first step.

## Existing AMD work

Checked on 2026-10-13 against upstream FreeVideo commit
`050e7a75bb6ad2ffb810df497c6a739689229940`:

- The official backend factory provides CUDA and MPS, not AMD/ROCm or Vulkan.
- The maintainers say AMD support is in progress in
  [#98](https://github.com/FlashML-org/FreeVideo/issues/98#issuecomment-6041151223)
  and [#76](https://github.com/FlashML-org/FreeVideo/issues/76#issuecomment-6036198093).
- A separate [community VDN ComfyUI port](https://github.com/ALITTLESEEDX/ComfyUI-VDN-H3)
  provides eager PyTorch equivalents and lists AMD/Intel/CPU as untested.
- Repository and web searches did not find a released FreeVideo Vulkan adapter.
  These searches do not prove universal absence or establish a first implementation.

This branch contributes a checked ggml Vulkan bridge, streamed reference adapters,
and reproducible measurements. It does not claim to replace the maintainers' AMD
work or the community VDN projects.

## What executes where

- PyTorch tensors remain CPU-owned. FP8 and NVFP4/AWQ weights are decoded on CPU.
- Supported matrix products and bounded window-attention matrix products execute
  through the ggml Vulkan C API. Window softmax uses NumPy on CPU.
- Normalization, elementwise work, convolutions and unsupported Torch operators
  stay on CPU. Torch dispatch counts and native matrix-call counts are reported
  separately; NumPy helper work is outside the Torch operator counters.
- GPU weights may be cached, bounded to approximately half the configured explicit
  buffer budget. Weak owners and tensor versions prevent reuse after release or
  modification. The remaining budget covers active explicit tensor buffers.
- CPU-reference mode uses the same released model and fixed modulation schedule,
  with PyTorch CPU matrix products and SDPA. It is always labeled separately.

The buffer cap covers explicit ggml buffers, not total CPU allocations, every
driver-internal workspace, or total physical memory. The reference lifecycle also
checks available host RAM. Use a fresh worker process for each stage; ggml feature
selection and the pinned VDN import overlay are process-scoped.

## Setup

Requirements: Windows x64, Python 3.12, Git, a working Vulkan driver, and about
46 GiB for model assets, in addition to dependencies and output space. Model
weights are downloaded separately under their original license. Source code is
Apache-2.0; dependencies retain their licenses in `THIRD_PARTY_NOTICES.md`.

From this checkout in PowerShell:

```powershell
.\setup-vulkan-experimental.ps1 -Python 'C:\path\to\python312.exe'
```

The setup creates/reuses `.venv`, installs the recorded CPU PyTorch stack, clones
the pinned VDN, patched Diffusers and ComfyUI dependencies, downloads the pinned
Windows Vulkan release, verifies its SHA-256, and runs the numerical GPU tests.
It leaves an existing dependency at a different Git revision unchanged and stops.
By default, VDN and ComfyUI are sibling directories; `-DependencyRoot` selects
another directory. Runtime DLLs and path configuration live in ignored
`experimental-data`. Model downloading is a separate action:

```powershell
.\.venv\Scripts\python.exe -m scripts.download_vulkan_assets
```

The downloader bootstraps the pinned FP8 manifest, fetches the released streamed
groups, exact 8-step T2VA tables, original VAEs and NVFP4 encoder, and verifies
published SHA-256 hashes. Large downloads retain completed ranges across retries.
No base-transformer shards, model weights or runtime DLLs are committed here.

## Run

```powershell
# Fastest tested complete pipeline: hybrid encoder, CPU sampler/decoder.
.\test-vulkan.ps1 -Stage all -Compute cpu-reference

# Stages can be repeated separately with existing conditioning/latents.
.\test-vulkan.ps1 -Stage sample -Compute cpu-reference
.\test-vulkan.ps1 -Stage decode -Compute cpu-reference

# Explicit experimental GPU attempt; substantially slower on the tested APU.
.\test-vulkan.ps1 -Stage sample -Compute vulkan
```

The smoke test uses seed 42, eight fixed steps, one pass, 256×256 and 39 frames at
24 fps. Outputs, logs, stage reports and sampled memory history go to ignored
`experimental-output`. The encoded prompt, latents and decoded media retain
separate stages so the text encoder is released before sampling.

For a different prompt, initialize the paths with setup or the launcher, then run:

```powershell
.\.venv\Scripts\python.exe -m scripts.test_vulkan_video encode --prompt 'Your prompt'
```

Direct module entry points default to the sibling dependency layout if no
environment overrides are set. The launcher reads setup's stored path configuration.
No GUI or automatic production-backend selection is added by this branch.

## Validation and measurements

[Machine-readable measurements](../benchmarks/vulkan-legion-go2-2026-10-13/summary.json)
include the hardware, tested versions, scopes and successful/failed checks.
The [preview](../benchmarks/vulkan-legion-go2-2026-10-13/preview.png) shows four
frames of the completed CPU-reference sample. Measurements are single-machine,
single-run observations, excluding downloads and installation.

| Stage | Wall time | Execution |
|---|---:|---|
| Text encoding | 327.99 s | CPU/Vulkan hybrid |
| Eight-step sampling | 990.58 s | CPU reference |
| Video/audio decoding and saving | 101.96 s | CPU reference |
| Total | 1420.53 s / 23m41s | Mixed pipeline |

Encoder process RSS peaked at 14.82 GiB; the minimum available physical RAM was
3.69 GiB. Sampler RSS peaked at 1.99 GiB and decoder RSS at 1.31 GiB. These are
sampled process measurements, not device allocator totals. No BIOS/UMA setting
was changed. Host RAM and shared GPU capacity must not be added as two physical pools.

The Vulkan sampler reached block 9 of step 1 before it was stopped. Later block
intervals were approximately eight seconds; about an hour for eight steps on
this tiny canvas was an **extrapolation**, not a measured completed Vulkan run.
CPU sampling completed all eight steps with finite video/audio latents. The MP4
was independently checked as H.264/AAC, 39 frames, 24 fps, 256×256 and 1.625 s.
The frames show a consistent orange cat with slight head movement; longer clips
and production resolutions remain untested.

Kernel validation:

- Nine real-GPU checks pass: FP32 and batched matrix products, BF16 biased
  projections, attention versus SDPA, a VDN inverse helper, shape/budget rejection,
  empty contractions and cache invalidation after weight modification.
- A real 768-token FF layer compared in FP32 passed its tolerance, with relative
  RMS difference `2.32e-6`. This is not full-model output parity.
- An earlier BF16 real-layer comparison failed its elementwise tolerance on
  3 of 4,128,768 values. That failure is retained in the benchmark summary.
- Native BF16 caching gave 1.90, 1.61 and 1.48 s for three real FF calls, with
  identical cached outputs. Faster device features and FP16 storage did not
  establish a speed advantage; the FP16 experiment also changed arithmetic.

Strict mode disables ggml cooperative matrices and implicit FP16 conversion to
retain the checked FP32 baseline. The optional FP16 probe is isolated from the
default runtime and is not a quality-validated fast backend.

## Development and provenance

Run tests with `FREEVIDEO_VULKAN_LIBRARY_DIR` pointing at the pinned DLL directory:

```powershell
$paths = Get-Content .\experimental-data\experimental-paths.json -Raw | ConvertFrom-Json
$env:FREEVIDEO_VULKAN_LIBRARY_DIR = $paths.runtime
.\.venv\Scripts\python.exe -m pytest tests/test_vulkan_experimental.py -q
```

Without that environment variable, these hardware-specific tests skip. The probe
modules under `scripts` require model assets and fresh workers with the configured
dependency paths. They include a deliberately strict BF16 comparison that can fail;
do not weaken it or relabel it as passing full-model parity.

The adapter was developed with OpenAI Codex assistance and exercised locally.
It reuses FreeVideo's streamed Mac reference lifecycle by adapting owned source
entry points; pinned upstream VDN and ComfyUI files are not rewritten. Source
replacement guards intentionally reject incompatible future reference changes.
FreeVideo's original CUDA and MPS entry points remain separate. Contributions to
upstream should follow `CONTRIBUTING.md`, including prior scope discussion.

Further work should measure a discrete Vulkan GPU, eliminate host/GPU round trips,
retain more compute on-device, and compare full-model outputs and quality before
making acceleration claims. This APU result does not establish performance on
other hardware.
