"""Experimental portable reference lifecycle, with reported Vulkan matrix work.

Reuse the verified streamed lifecycle while keeping its original Mac entry point
unchanged. CPU tensor ownership and GPU matrix execution are both explicit.
"""
import hashlib
from pathlib import Path
import types
from contextlib import nullcontext


def reference_module(backend='vulkan'):
    if backend not in ('vulkan', 'cpu-reference'):
        raise ValueError('Unknown experimental compute backend')
    path = Path(__file__).with_name('macos_runtime.py')
    text = path.read_text(encoding='utf-8')
    replacements = {
        'from .macos_vdn import activate': 'from .vulkan_vdn import activate',
        "get_backend('mps')": f"get_backend('{backend}')",
        "load_conditioning(conditioning, 'mps'": "load_conditioning(conditioning, 'cpu'",
        "seed, 'mps', step_seconds=durations": "seed, 'cpu', step_seconds=durations",
        "WindowAttention('mps'": f"WindowAttention('{backend}'",
        "device_backend='mps'": f"device_backend='{backend}'",
        ".to('mps')": ".to('cpu')",
        "producer_device='mps'": f"producer_device='{backend}'",
        "device='mps'": "device='cpu'",
        "exclude=('transformer_blocks.', 'token_refiner.refiner_blocks.', 'rope.'),":
            "exclude=('transformer_blocks.', 'token_refiner.refiner_blocks.', 'rope.'), device='cpu',",
        "int8_compute = mps_int8.available()": "int8_compute = False",
    }
    for before, after in replacements.items():
        if before not in text:
            raise ValueError('Portable runtime no longer matches: ' + before)
        text = text.replace(before, after)
    module = types.ModuleType('freevideo_engine._vulkan_reference')
    module.__package__ = 'freevideo_engine'
    module.__file__ = str(path)
    exec(compile(text, str(path), 'exec'), module.__dict__)
    module.source_sha256 = hashlib.sha256(text.encode()).hexdigest()
    return module


class Engine:
    def __init__(self, *args, **kwargs):
        from .backends.vulkan_dispatch import VulkanDispatch
        if kwargs.get('fast_kernels') or kwargs.get('plan_compute') or kwargs.get('attention_impl', 'torch') != 'torch':
            raise ValueError('Experimental Vulkan uses the explicitly bounded reference partitions')
        compute = kwargs.pop('compute', 'vulkan')
        self.reference = reference_module(compute).Engine(*args, **kwargs)
        self.dispatch = VulkanDispatch(self.reference.backend.accelerator) if compute == 'vulkan' else None

    def generate(self, *args, **kwargs):
        with self.dispatch if self.dispatch is not None else nullcontext():
            video, audio, report = self.reference.generate(*args, **kwargs)
        report.update(execution_report=self.dispatch.statistics() if self.dispatch is not None else dict(cpu_reference=True),
                      experimental=True, fully_gpu_native=False)
        return video, audio, report

    def close(self):
        self.reference.close()
        if self.dispatch is not None:
            self.reference.backend.accelerator.close()
