"""Experimental, explicitly hybrid text conditioning and VAE decoding.

Adapt owned reference entry points, leaving upstream CUDA and MPS paths intact.
All tensors remain CPU owned; supported matrix products use the Vulkan bridge.
"""
from pathlib import Path
import types
from contextlib import nullcontext


def reference(name, replacements):
    path = Path(__file__).with_name(name + '.py')
    source = path.read_text(encoding='utf-8')
    for before, after in replacements.items():
        if before not in source:
            raise ValueError('Reference source changed: ' + before)
        source = source.replace(before, after)
    module = types.ModuleType('freevideo_engine._vulkan_' + name)
    module.__package__ = 'freevideo_engine'
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def install_encoder_linears(*args, **kwargs):
    return reference('macos_encoder_linear', {
        "input.device.type != 'mps'": "input.device.type != 'cpu'",
    }).install(*args, **kwargs)


def install_encoder_attention(*args, **kwargs):
    return reference('macos_encoder_attention', {
        "q.device.type != 'mps'": "q.device.type != 'cpu'",
    }).install(*args, **kwargs)


def encode(*args, **kwargs):
    from .backends.vulkan_dispatch import VulkanDispatch
    from .backends.vulkan import VulkanBackend
    backend = VulkanBackend()
    dispatch = VulkanDispatch(backend.accelerator)
    module = reference('macos_encoder', {
        "if sys.platform != 'darwin' or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') not in ('', '0'):":
            'if False:  # Explicit portable experimental entry point',
        "input.device.type != 'mps'": "input.device.type != 'cpu'",
        "get_backend('mps')": "get_backend('vulkan')",
        "'--disable-dynamic-vram',": "'--cpu', '--disable-dynamic-vram',",
        "torch.device('mps')": "torch.device('cpu')",
        'from .macos_encoder_linear import install': 'from .vulkan_stages import install_encoder_linears as install',
        'from .macos_encoder_attention import install': 'from .vulkan_stages import install_encoder_attention as install',
        'from .macos_vdn import activate': 'from .vulkan_vdn import activate',
        "device='mps'": "device='cpu'",
        "device_backend='mps'": "device_backend='vulkan'",
    })
    try:
        with dispatch:
            result = module.encode(*args, **kwargs)
        result.update(experimental=True, fully_gpu_native=False, vulkan_hybrid=dispatch.statistics())
        return result
    finally:
        backend.accelerator.close()


def decode(*args, compute='vulkan', **kwargs):
    from .backends.vulkan_dispatch import VulkanDispatch
    from .backends.vulkan import VulkanBackend
    if compute == 'cpu-reference':
        from .backends.cpu_reference import CPUReferenceBackend
        backend = CPUReferenceBackend()
        dispatch = None
    elif compute == 'vulkan':
        backend = VulkanBackend()
        dispatch = VulkanDispatch(backend.accelerator)
    else:
        raise ValueError('Unknown decoder compute backend')
    module = reference('macos_decode', {
        'from .macos_vdn import activate': 'from .vulkan_vdn import activate',
        "get_backend('mps')": f"get_backend('{compute}')",
        "else 'mps'": "else 'cpu'",
        ".to('mps')": ".to('cpu')",
        "device_backend='mps'": f"device_backend='{compute}'",
        "device='mps'": "device='cpu'",
        "torch.autocast(device_type='mps', dtype=torch.float16, cache_enabled=False)":
            "torch.autocast(device_type='cpu', enabled=False)",
    })
    try:
        with dispatch if dispatch is not None else nullcontext():
            result = module.decode_to_file(*args, cache_weights=False, **kwargs)
        result.update(experimental=True, fully_gpu_native=False,
                      precision='FP32 video and audio; CPU tensor ownership',
                      execution_report=dispatch.statistics() if dispatch is not None else dict(cpu_reference=True))
        return result
    finally:
        if dispatch is not None:
            backend.accelerator.close()
