"""Explicit CPU control for comparing the experimental Vulkan pipeline."""
import psutil
from .base import BackendCapabilities
from .vulkan import VulkanBackend


class CPUAttention:
    def batched(self, leg, q, k, v, scale):
        import torch.nn.functional as F
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
            v.transpose(1, 2), scale=scale).transpose(1, 2)


class CPUReferenceBackend(VulkanBackend):
    capabilities = BackendCapabilities(name='cpu-reference', memory_model='host',
        linear_policies=('bf16', 'bf16-weight-only'), attention_candidates=('cpu-reference',),
        pinned_host_weights=False, streamed_weights=True)

    def __init__(self, *, torch_module=None):
        self._torch = torch_module
        self.peak_rss = 0

    def is_available(self):
        return True

    def synchronize(self):
        pass

    def max_memory_allocated(self):
        memory = psutil.Process().memory_info()
        self.peak_rss = max(self.peak_rss, getattr(memory, 'peak_wset', memory.rss))
        return self.peak_rss

    def reset_peak_memory_stats(self):
        self.peak_rss = psutil.Process().memory_info().rss

    def configure_budget(self, budget_bytes, allocator_limit_bytes=None, **options):
        return dict(enforced=False, effective_allocator_limit_bytes=budget_bytes,
                    scope='CPU reference; runtime host guard, no hard allocator cap')

    def arithmetic_identity(self):
        return dict(device_backend='cpu-reference', execution='CPU only',
                    matrix_storage='FP32/BF16', matrix_compute='PyTorch CPU')

    def attention_kernels(self, global_backend, window_backend, **options):
        if (global_backend, window_backend) != ('cpu-reference', 'cpu-reference'):
            raise ValueError('CPU reference requires explicit CPU attention')
        return CPUAttention()

    def memory_stats(self):
        return dict(process_rss_bytes=psutil.Process().memory_info().rss,
                    available_system_bytes=self.memory_info()[0])
