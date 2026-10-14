"""Experimental host-backed Vulkan backend; CPU work is explicit in its report."""
import gc
import os
from pathlib import Path

import psutil

from .base import BackendCapabilities, DeviceBackend
from .vulkan_ggml import VulkanGGML


class VulkanAttentionKernels:
    def __init__(self, accelerator, global_backend, window_backend, query_chunk, window_varlen):
        if (global_backend, window_backend) != ('vulkan', 'vulkan') or window_varlen:
            raise ValueError('Experimental Vulkan uses bounded non-varlen attention')
        self.accelerator, self.query_chunk = accelerator, query_chunk or 128

    def batched(self, leg, q, k, v, scale):
        if leg != 'vulkan':
            raise ValueError('Unknown Vulkan attention leg')
        from .vulkan_dispatch import attention
        return attention(self.accelerator, q, k, v, scale, query_chunk=self.query_chunk)

    def varlen(self, *args, **kwargs):
        raise NotImplementedError('Vulkan packed varlen attention is not implemented')

    def decomposed(self, *args, **kwargs):
        raise NotImplementedError('Vulkan FA4 decomposition is not implemented')


class VulkanBackend(DeviceBackend):
    device = 'cpu'
    capabilities = BackendCapabilities(name='vulkan', memory_model='host-backed-hybrid',
        linear_policies=('bf16', 'bf16-weight-only'), attention_candidates=('vulkan',),
        pinned_host_weights=False, streamed_weights=True)

    def __init__(self, *, library_dir=None, torch_module=None):
        self._torch = torch_module
        location = library_dir or os.environ.get('FREEVIDEO_VULKAN_LIBRARY_DIR')
        if not location:
            raise ValueError('Set FREEVIDEO_VULKAN_LIBRARY_DIR to matching ggml DLLs')
        self.library_dir = Path(location)
        self._accelerator = None

    @property
    def torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    @property
    def accelerator(self):
        if self._accelerator is None:
            self._accelerator = VulkanGGML(self.library_dir)
        return self._accelerator

    def is_available(self):
        return bool(self.accelerator.backend)

    def synchronize(self):
        self.accelerator.synchronize()

    def empty_cache(self):
        gc.collect()

    def memory_info(self):
        memory = psutil.virtual_memory()
        return int(memory.available), int(memory.total)

    def max_memory_allocated(self):
        return self.accelerator.peak_buffer_bytes

    def max_memory_reserved(self):
        return self.max_memory_allocated()

    def memory_reserved(self):
        return 0

    def reset_peak_memory_stats(self):
        self.accelerator.peak_buffer_bytes = 0

    def configure_budget(self, budget_bytes, allocator_limit_bytes=None, **options):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ValueError('Vulkan budget must be a positive integer')
        self.accelerator.set_budget(min(budget_bytes, allocator_limit_bytes or budget_bytes))
        return dict(enforced=True, effective_allocator_limit_bytes=self.accelerator.max_buffer_bytes,
                    scope='Per-call Vulkan tensor buffers only; host allocations require the runtime guard')

    def arithmetic_identity(self):
        return dict(device_backend='vulkan', execution='host-backed-hybrid',
                    matrix_storage='FP32/BF16', matrix_compute='FP32 accumulation', native_torch_vulkan=False)

    def attention_kernels(self, global_backend, window_backend, **options):
        return VulkanAttentionKernels(self.accelerator, global_backend, window_backend,
                                     options.get('query_chunk', 0), options.get('window_varlen', False))

    def prepare_linears(self, model, manifest, linear_compute, fp8_gemm):
        if linear_compute not in self.capabilities.linear_policies or fp8_gemm != 'auto':
            raise ValueError('Vulkan currently requires explicit host BF16 weight-only decoding')
        if manifest.get('precision') not in ('bf16', 'fp8'):
            raise ValueError('Vulkan prepared storage currently supports BF16 and FP8 only')

    def install_chunked_ff(self, module, chunk, *, recompute=False):
        from .mps import MPSBackend
        return MPSBackend.install_chunked_ff(self, module, chunk, recompute=recompute)

    def make_offloader(self, layers, **options):
        from .mps_weights import LayerResidency
        residency = LayerResidency(layers, backend=self, **options)
        original = residency.stats
        def stats():
            result = original()
            result['policy'] = result['policy'].replace('MPS', 'CPU host-storage')
            result['device_backend'] = self.capabilities.name
            return result
        residency.stats = stats
        return residency

    def pin_layer_weights(self, *args, **kwargs):
        raise NotImplementedError('Host-backed Vulkan does not implement pinned CUDA weight copies')

    def host_memory_stats(self):
        return dict(available_bytes=self.memory_info()[0], pinned_host_weights=False)

    def prepare_streamed_layer(self, *args, **kwargs):
        raise NotImplementedError('Use the explicit one-layer LayerResidency path')

    def unload_streamed_layer(self, *args, **kwargs):
        raise NotImplementedError('Use the explicit one-layer LayerResidency path')

    def memory_stats(self):
        return dict(available_system_bytes=self.memory_info()[0],
                    process_rss_bytes=psutil.Process().memory_info().rss,
                    vulkan=self.accelerator.statistics())
