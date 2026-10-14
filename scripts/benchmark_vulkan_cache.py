"""Measure first use versus weight reuse on the released FF checkpoint."""
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
torch.manual_seed(42)
x = torch.randn(1, 768, 5376, dtype=torch.bfloat16) * .1
gpu = VulkanGGML(os.environ['FREEVIDEO_VULKAN_LIBRARY_DIR'], max_buffer_bytes=4 << 30)
mode = VulkanDispatch(gpu)
times, outputs = [], []
with torch.no_grad(), mode:
    for _ in range(3):
        started = time.perf_counter()
        outputs.append(block.ff(x))
        times.append(time.perf_counter() - started)
torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
report = dict(scope='Real FF block only; not video generation', seconds=times,
              cached_output_identical=True, statistics=mode.statistics())
output = root.parent / 'experimental-output' / 'freevideo-vulkan-cache.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
gpu.close()
