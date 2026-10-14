"""Explicit hybrid executor: GPU matrix products, reported CPU tensor operators."""
from collections import Counter

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class VulkanDispatch(TorchDispatchMode):
    def __init__(self, accelerator, *, bf16_fp16=False):
        super().__init__()
        self.accelerator = accelerator
        self.bf16_fp16 = bf16_fp16
        self.gpu_ops, self.cpu_ops = Counter(), Counter()

    def product(self, left, right, *, round_output=True, weight=None):
        if left.device.type != 'cpu' or right.device.type != 'cpu':
            raise ValueError('Host-backed Vulkan accepts CPU tensors only')
        if left.dtype != right.dtype or left.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError('Vulkan matrix inputs need matching floating-point dtypes')
        with torch._C._DisableTorchDispatch():
            bf16 = left.dtype == torch.bfloat16 and not self.bf16_fp16
            a, b = [(tensor.detach().view(torch.uint16).numpy() if bf16
                     else tensor.detach().float().numpy()) for tensor in (left, right)]
            left_vector, right_vector = a.ndim == 1, b.ndim == 1
            if left_vector:
                a = a[None, :]
            if right_vector:
                b = b[:, None]
            # Native BF16 preserves its exponent range. The FP16 experiment
            # is an explicit, separately reported arithmetic policy.
            key = owner = None
            if weight is not None:
                owner = weight._base if weight._base is not None else weight
                try:
                    version = owner._version
                except RuntimeError:
                    version = None
                if version is not None:
                    key = (id(owner), version, tuple(weight.shape), tuple(weight.stride()), weight.storage_offset())
            if self.bf16_fp16 and (not np.isfinite(a).all() or not np.isfinite(b).all()
                                  or np.abs(a).max(initial=0) > 65504 or np.abs(b).max(initial=0) > 65504):
                raise ValueError('Experimental FP16 projection exceeds its finite range')
            value = self.accelerator.matmul(a, b, bfloat16=bf16, weight_fp16=self.bf16_fp16,
                                           cache_key=key, cache_owner=owner)
            if left_vector:
                value = np.squeeze(value, axis=-2)
            if right_vector:
                value = np.squeeze(value, axis=-1)
            output = torch.from_numpy(value)
            return output.to(left.dtype) if round_output else output

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if self.bf16_fp16 and name in ('aten.mm.default', 'aten.bmm.default', 'aten.matmul.default',
                                      'aten.linear.default', 'aten.addmm.default'):
            matrix = args[1] if name == 'aten.addmm.default' else args[0]
            if matrix.dtype != torch.bfloat16:
                self.cpu_ops[name] += 1
                return func(*args, **kwargs)
        if name in ('aten.mm.default', 'aten.bmm.default', 'aten.matmul.default'):
            self.gpu_ops[name] += 1
            base = args[1]._base if args[1]._base is not None else args[1]
            return self.product(args[0], args[1], weight=args[1] if isinstance(base, torch.nn.Parameter) else None)
        if name == 'aten.linear.default':
            self.gpu_ops[name] += 1
            with torch._C._DisableTorchDispatch():
                weight = args[1].transpose(-1, -2)
            output = self.product(args[0], weight, round_output=False, weight=args[1])
            with torch._C._DisableTorchDispatch():
                if len(args) > 2 and args[2] is not None:
                    output = output + args[2].float()
                return output.to(args[0].dtype)
        if name == 'aten.addmm.default':
            self.gpu_ops[name] += 1
            base = args[2]._base if args[2]._base is not None else args[2]
            result = self.product(args[1], args[2], round_output=False,
                                  weight=args[2] if isinstance(base, torch.nn.Parameter) else None)
            with torch._C._DisableTorchDispatch():
                alpha, beta = kwargs.get('alpha', 1), kwargs.get('beta', 1)
                value = result * alpha + args[0].float() * beta if beta else result * alpha
                return value.to(args[1].dtype)
        self.cpu_ops[name] += 1
        return func(*args, **kwargs)

    def statistics(self):
        return dict(vulkan=self.accelerator.statistics(), gpu_operators=dict(self.gpu_ops),
                    cpu_operators=dict(self.cpu_ops), hybrid_execution=True,
                    cpu_operator_scope='Intercepted Torch operators; NumPy attention softmax and transfers excluded',
                    projection_policy='experimental BF16-to-FP16' if self.bf16_fp16 else 'native BF16/FP32')


def attention(accelerator, q, k, v, scale, *, query_chunk=128):
    """NHD batched contract, bounded FP32 score matrix and host softmax."""
    if query_chunk < 1:
        raise ValueError('Attention query chunk must be positive')
    with torch._C._DisableTorchDispatch():
        arrays = [x.detach().float().transpose(1, 2).numpy() for x in (q, k, v)]
        qa, ka, va = arrays
        output = np.empty(qa.shape[:-1] + (va.shape[-1],), dtype=np.float32)
        for begin in range(0, qa.shape[-2], query_chunk):
            end = min(begin + query_chunk, qa.shape[-2])
            scores = accelerator.matmul(qa[..., begin:end, :], ka.swapaxes(-1, -2)) * scale
            scores -= scores.max(axis=-1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            output[..., begin:end, :] = accelerator.matmul(probabilities, va)
        return torch.from_numpy(output).transpose(1, 2).to(q.dtype)
