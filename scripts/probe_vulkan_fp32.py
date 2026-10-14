"""Separate arithmetic parity from BF16 model rounding on real FF weights."""
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
block.ff.float()
torch.manual_seed(42)
value = (torch.randn(1, 768, 5376, dtype=torch.bfloat16) * .1).float()
gpu = VulkanGGML(os.environ['FREEVIDEO_VULKAN_LIBRARY_DIR'])
mode = VulkanDispatch(gpu)
with torch.inference_mode():
    reference = block.ff(value)
    with mode:
        actual = block.ff(value)
error = (actual - reference).abs()
report = dict(scope='Real FF block, FP32 arithmetic; not video quality parity',
              max_error=error.max().item(), mean_error=error.mean().item(),
              relative_rms=(error.square().mean().sqrt() / reference.square().mean().sqrt()).item(),
              finite=actual.isfinite().all().item(), statistics=mode.statistics())
path = root.parent / 'experimental-output' / 'freevideo-vulkan-fp32.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
torch.testing.assert_close(actual, reference, rtol=2e-4, atol=1e-3)
gpu.close()
