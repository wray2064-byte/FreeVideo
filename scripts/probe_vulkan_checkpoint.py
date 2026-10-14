"""Compare one real released VDN-H3 feed-forward block on CPU and Vulkan."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from freevideo_engine.vulkan_vdn import activate
from freevideo_engine.backends.vulkan_ggml import VulkanGGML
from freevideo_engine.backends.vulkan_dispatch import VulkanDispatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=8)
    args = parser.parse_args()
    activate()
    from freevideo_engine.weights import skeleton
    from freevideo_engine.backends.mps_weights import load_group
    manifest = json.loads((args.models / 'cache/manifest.json').read_text())
    model = skeleton(args.models / 'h3-base', args.models / 'stage-dmd-step-250')
    block = model.transformer_blocks[0]
    tick = time.perf_counter()
    load_group(block, args.models / 'cache/blocks/00.safetensors', manifest['linears'],
               prefix='transformer_blocks.0.', exclude=('adaln_proj.',), device='cpu')
    load_seconds = time.perf_counter() - tick
    torch.manual_seed(42)
    value = torch.randn(1, args.tokens, model.config.hidden_size, dtype=torch.bfloat16) * 0.1
    gpu = VulkanGGML(os.environ['FREEVIDEO_VULKAN_LIBRARY_DIR'])
    mode = VulkanDispatch(gpu)
    with torch.inference_mode():
        tick = time.perf_counter()
        reference = block.ff(value)
        cpu_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        with mode:
            actual = block.ff(value)
        vulkan_seconds = time.perf_counter() - tick
    difference = (reference.float() - actual.float()).abs()
    report = dict(model='Released VDN-H3 FP8 cache, block 00, feed-forward only',
        input_shape=list(value.shape), load_seconds=load_seconds,
        cpu_seconds=cpu_seconds, vulkan_seconds=vulkan_seconds,
        max_absolute_error=float(difference.max()), mean_absolute_error=float(difference.mean()),
        finite=bool(actual.isfinite().all()), operations=mode.statistics(),
        scope='Not a generated video or full-block parity claim')
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    torch.testing.assert_close(actual, reference, rtol=0.04, atol=0.02)
    gpu.close()


if __name__ == '__main__':
    main()
