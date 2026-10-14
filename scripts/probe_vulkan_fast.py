"""Probe native BF16 Vulkan with default device features in a fresh process."""
import json, os, pathlib, time
import torch
from freevideo_engine.vulkan_vdn import activate
from freevideo_engine.backends.vulkan_ggml import VulkanGGML
from freevideo_engine.backends.vulkan_dispatch import VulkanDispatch
from freevideo_engine.backends.mps_weights import load_group
from freevideo_engine.weights import skeleton

activate()
root = pathlib.Path(__file__).resolve().parents[1] / 'experimental-models'
manifest = json.loads((root / 'cache/manifest.json').read_text())
model = skeleton(root / 'h3-base', root / 'stage-dmd-step-250')
block = model.transformer_blocks[0]
load_group(block, root / 'cache/blocks/00.safetensors', manifest['linears'],
           prefix='transformer_blocks.0.', exclude=('adaln_proj.',), device='cpu')
torch.set_num_threads(8)
torch.manual_seed(42)
x = torch.randn(1, 768, 5376, dtype=torch.bfloat16) * .1
gpu = VulkanGGML(os.environ['FREEVIDEO_VULKAN_LIBRARY_DIR'], max_buffer_bytes=4 << 30, strict_fp32=False)
mode = VulkanDispatch(gpu, bf16_fp16=True)
times = []
with torch.no_grad():
    started = time.perf_counter()
    expected = block.ff(x)
    cpu_seconds = time.perf_counter() - started
    for _ in range(3):
        started = time.perf_counter()
        with mode:
            actual = block.ff(x)
        times.append(time.perf_counter() - started)
error = (actual.float() - expected.float()).abs()
report = dict(scope='Real BF16 FF block with FP16 Vulkan projections; not full-model parity',
              cpu_seconds=cpu_seconds, vulkan_seconds=times, finite=actual.isfinite().all().item(),
              max_error=error.max().item(), mean_error=error.mean().item(),
              relative_rms=(error.square().mean().sqrt()/expected.float().square().mean().sqrt()).item(),
              statistics=mode.statistics())
output = root.parent / 'experimental-output' / 'freevideo-vulkan-fast.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert report['finite'] and report['relative_rms'] < .005
gpu.close()
