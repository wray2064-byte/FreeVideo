import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from freevideo_engine.backends.vulkan_ggml import VulkanGGML
from freevideo_engine.backends.vulkan_dispatch import VulkanDispatch, attention
from freevideo_engine.backends.mps_delta import inverse_spd


@pytest.fixture(scope='module')
def gpu():
    location = os.environ.get('FREEVIDEO_VULKAN_LIBRARY_DIR')
    if not location:
        pytest.skip('Set FREEVIDEO_VULKAN_LIBRARY_DIR to the pinned Windows ggml Vulkan DLLs')
    executor = VulkanGGML(location)
    yield executor
    executor.close()


@pytest.mark.parametrize('shape', [(13, 64, 29), (2, 3, 13, 64, 29), (2, 3, 2, 13, 64, 29)])
def test_matrix_products(gpu, shape):
    rng = np.random.default_rng(42)
    batch, m, k, n = shape[:-3], *shape[-3:]
    a = rng.normal(size=batch + (m, k)).astype(np.float32)
    b = rng.normal(size=batch + (k, n)).astype(np.float32)
    np.testing.assert_allclose(gpu.matmul(a, b), a @ b, rtol=3e-5, atol=2e-5)


def test_dispatch_bias_and_bfloat(gpu):
    torch.manual_seed(42)
    a, b, bias = torch.randn(11, 64).bfloat16(), torch.randn(64, 23).bfloat16(), torch.randn(23).bfloat16()
    reference = torch.addmm(bias, a, b, alpha=0.5, beta=0.2)
    mode = VulkanDispatch(gpu)
    with torch.inference_mode(), mode:
        actual = torch.addmm(bias, a, b, alpha=0.5, beta=0.2)
    torch.testing.assert_close(actual, reference, rtol=0.004, atol=0.002)
    assert mode.gpu_ops['aten.addmm.default'] == 1


def test_attention_matches_sdpa(gpu):
    torch.manual_seed(5)
    q, k, v = torch.randn(2, 13, 3, 64), torch.randn(2, 17, 3, 64), torch.randn(2, 17, 3, 64)
    reference = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               scale=0.125).transpose(1, 2)
    actual = attention(gpu, q, k, v, 0.125, query_chunk=5)
    torch.testing.assert_close(actual, reference, rtol=4e-5, atol=2e-5)


def test_vdn_inverse_matches_cholesky(gpu):
    torch.manual_seed(9)
    keys = torch.randn(2, 3, 16, 128) * 0.02
    a = keys.transpose(-1, -2) @ keys
    eye = torch.eye(128).expand_as(a)
    reference = torch.cholesky_inverse(torch.linalg.cholesky(eye + a))
    mode = VulkanDispatch(gpu)
    with torch.inference_mode(), mode:
        actual = inverse_spd(a, max_residual=2e-5, low_precision=False)
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=3e-6)
    assert sum(mode.gpu_ops.values()) > 0


def test_budget_and_shape_rejection(gpu):
    previous = gpu.max_buffer_bytes
    gpu.max_buffer_bytes = 64
    try:
        with pytest.raises(MemoryError):
            gpu.matmul(np.ones((8, 8), np.float32), np.ones((8, 8), np.float32))
        with pytest.raises(ValueError):
            gpu.matmul(np.ones((3, 4), np.float32), np.ones((5, 6), np.float32))
    finally:
        gpu.max_buffer_bytes = previous


def test_empty_contraction(gpu):
    result = gpu.matmul(np.empty((3, 0), np.float32), np.empty((0, 4), np.float32))
    np.testing.assert_array_equal(result, np.zeros((3, 4), np.float32))


def test_linear_cache_reuse_and_weight_mutation(gpu):
    torch.manual_seed(7)
    layer = torch.nn.Linear(64, 32).eval()
    x = torch.randn(2, 3, 64)
    mode = VulkanDispatch(gpu)
    before = gpu.cache_hits
    with torch.no_grad():
        with mode:
            first = layer(x)
            second = layer(x)
        assert gpu.cache_hits > before
        torch.testing.assert_close(first, second)
        layer.weight.add_(0.125)
        expected = layer(x)
        with mode:
            changed = layer(x)
    torch.testing.assert_close(changed, expected, rtol=4e-5, atol=3e-5)
