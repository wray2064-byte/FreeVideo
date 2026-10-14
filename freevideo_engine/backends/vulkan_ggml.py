"""Host-backed Vulkan matrix execution through the versioned ggml C API.

Inputs and results cross the host boundary explicitly. This does not claim
PyTorch has a native Vulkan tensor device or that all model operators run on GPU.
"""
import ctypes as C
from collections import OrderedDict
import os
from pathlib import Path
import threading
import time
import weakref

import numpy as np


class InitParams(C.Structure):
    _fields_ = [('mem_size', C.c_size_t), ('mem_buffer', C.c_void_p), ('no_alloc', C.c_bool)]


class VulkanGGML:
    def __init__(self, library_dir, *, device=0, max_buffer_bytes=2 << 30, strict_fp32=True):
        if strict_fp32:
            os.environ['GGML_VK_DISABLE_COOPMAT'] = '1'
            os.environ['GGML_VK_DISABLE_F16'] = '1'
        self.directory = Path(library_dir).resolve()
        self._dll_directory = os.add_dll_directory(str(self.directory))
        self.base = C.CDLL(str(self.directory / 'ggml-base.dll'))
        self.vk = C.CDLL(str(self.directory / 'ggml-vulkan.dll'))
        self.max_buffer_bytes = int(max_buffer_bytes)
        self.lock = threading.RLock()
        self.calls, self.seconds, self.transferred_bytes = 0, 0., 0
        self.peak_buffer_bytes = 0
        self.weights = OrderedDict()
        self.cached_bytes = self.cache_hits = 0
        self._bind()
        self.backend = self.vk.ggml_backend_vk_init(device)
        if not self.backend:
            raise RuntimeError('No Vulkan backend device initialized')
        self.device_name = self.base.ggml_backend_name(self.backend).decode()

    def _bind(self):
        def function(library, name, result, *arguments):
            f = getattr(library, name)
            f.restype, f.argtypes = result, list(arguments)
        p, i64, sz = C.c_void_p, C.c_int64, C.c_size_t
        function(self.vk, 'ggml_backend_vk_init', p, sz)
        function(self.base, 'ggml_backend_name', C.c_char_p, p)
        function(self.base, 'ggml_backend_free', None, p)
        function(self.base, 'ggml_init', p, InitParams)
        function(self.base, 'ggml_free', None, p)
        function(self.base, 'ggml_new_tensor_4d', p, p, C.c_int, i64, i64, i64, i64)
        function(self.base, 'ggml_mul_mat', p, p, p, p)
        function(self.base, 'ggml_prec_set_acc', C.c_bool, p, C.c_int)
        function(self.base, 'ggml_prec_set_src', C.c_bool, p, C.c_int, C.c_int)
        function(self.base, 'ggml_set_input', None, p)
        function(self.base, 'ggml_set_output', None, p)
        function(self.base, 'ggml_new_graph_custom', p, p, sz, C.c_bool)
        function(self.base, 'ggml_build_forward_expand', None, p, p)
        function(self.base, 'ggml_backend_alloc_ctx_tensors', p, p, p)
        function(self.base, 'ggml_backend_buffer_free', None, p)
        function(self.base, 'ggml_backend_buffer_get_size', sz, p)
        function(self.base, 'ggml_backend_tensor_set', None, p, p, sz, sz)
        function(self.base, 'ggml_backend_tensor_get', None, p, p, sz, sz)
        function(self.base, 'ggml_backend_graph_compute', C.c_int, p, p)
        function(self.base, 'ggml_backend_synchronize', None, p)

    def _evict(self, key):
        _, context, buffer, _, size = self.weights.pop(key)
        self.base.ggml_backend_buffer_free(buffer)
        self.base.ggml_free(context)
        self.cached_bytes -= size

    def _weight(self, key, owner, data, k, n, storage_type):
        for old, entry in list(self.weights.items()):
            if entry[0]() is None:
                self._evict(old)
        if key in self.weights:
            entry = self.weights[key]
            if entry[0]() is owner:
                self.weights.move_to_end(key)
                self.cache_hits += 1
                return entry[3]
            self._evict(key)
        limit = self.max_buffer_bytes // 2
        if data.nbytes > limit:
            return None
        while self.weights and self.cached_bytes + data.nbytes + 1024 > limit:
            self._evict(next(iter(self.weights)))
        context = self.base.ggml_init(InitParams(1 << 20, None, True))
        if not context:
            raise MemoryError('Vulkan weight metadata allocation failed')
        buffer = None
        try:
            tensor = self.base.ggml_new_tensor_4d(context, storage_type, k, n, 1, 1)
            self.base.ggml_set_input(tensor)
            buffer = self.base.ggml_backend_alloc_ctx_tensors(context, self.backend)
            if not buffer:
                raise MemoryError('Vulkan cached weight allocation failed')
            size = self.base.ggml_backend_buffer_get_size(buffer)
            self.base.ggml_backend_tensor_set(tensor, data.ctypes.data, 0, data.nbytes)
            self.transferred_bytes += data.nbytes
            self.weights[key] = (weakref.ref(owner), context, buffer, tensor, size)
            self.cached_bytes += size
            return tensor
        except BaseException:
            if buffer:
                self.base.ggml_backend_buffer_free(buffer)
            self.base.ggml_free(context)
            raise

    def matmul(self, left, right, *, weight_fp16=False, bfloat16=False, cache_key=None, cache_owner=None):
        left, right = np.asarray(left), np.asarray(right)
        if left.ndim < 2 or right.ndim < 2:
            raise ValueError('Vulkan matrix inputs must have at least two dimensions')
        if left.shape[-1] != right.shape[-2]:
            raise ValueError('Incompatible matrix contraction dimensions')
        batch = np.broadcast_shapes(left.shape[:-2], right.shape[:-2])
        m, k, n = left.shape[-2], left.shape[-1], right.shape[-1]
        if not all((m, k, n, *batch)):
            return np.zeros(batch + (m, n), dtype=np.float32)
        storage_type = 30 if bfloat16 else 0
        if bfloat16 and (left.dtype != np.uint16 or right.dtype != np.uint16 or weight_fp16):
            raise ValueError('BF16 input storage requires uint16 bits for both matrices')
        x = np.ascontiguousarray(np.broadcast_to(left, batch + (m, k)), dtype=np.uint16 if bfloat16 else np.float32)
        cacheable = cache_key is not None and right.ndim == 2
        weight_type = 1 if weight_fp16 else storage_type
        w = np.ascontiguousarray((right if cacheable else np.broadcast_to(right, batch + (k, n))).swapaxes(-1, -2),
                                 dtype=np.uint16 if bfloat16 else np.float16 if weight_fp16 else np.float32)
        output = np.empty(batch + (m, n), dtype=np.float32)
        required = x.nbytes + w.nbytes + output.nbytes
        if required > self.max_buffer_bytes:
            raise MemoryError(f'Matrix call needs {required} bytes; cap is {self.max_buffer_bytes}')
        gpu_batch = batch if len(batch) <= 2 else (int(np.prod(batch)),)
        b = tuple(reversed(gpu_batch)) + (1,) * (2 - len(gpu_batch))
        with self.lock:
            if not self.backend:
                raise RuntimeError('Vulkan backend is closed')
            tick = time.perf_counter()
            context = self.base.ggml_init(InitParams(4 << 20, None, True))
            if not context:
                raise MemoryError('ggml metadata context allocation failed')
            buffer = None
            try:
                cached = (self._weight((weight_type, cache_key), cache_owner, w, k, n, weight_type)
                          if cacheable else None)
                protected_key = (weight_type, cache_key) if cached else None
                if cacheable and not cached:
                    w = np.ascontiguousarray(np.broadcast_to(w, batch + (n, k)))
                a = cached or self.base.ggml_new_tensor_4d(context, 1 if weight_fp16 else storage_type, k, n, *b)
                c = self.base.ggml_new_tensor_4d(context, storage_type, k, m, *b)
                self.base.ggml_set_input(a)
                self.base.ggml_set_input(c)
                result = self.base.ggml_mul_mat(context, a, c)
                self.base.ggml_prec_set_acc(result, 10)
                if not weight_fp16:
                    self.base.ggml_prec_set_src(result, 15 if bfloat16 else 10, 0)
                    self.base.ggml_prec_set_src(result, 15 if bfloat16 else 10, 1)
                self.base.ggml_set_output(result)
                graph = self.base.ggml_new_graph_custom(context, 128, False)
                self.base.ggml_build_forward_expand(graph, result)
                workspace = x.nbytes + output.nbytes + (0 if cached else w.nbytes) + 1024
                while self.weights and self.cached_bytes + workspace > self.max_buffer_bytes:
                    victim = next((key for key in self.weights if key != protected_key), None)
                    if victim is None:
                        raise MemoryError('Vulkan cache plus active matrices exceeds its cap')
                    self._evict(victim)
                buffer = self.base.ggml_backend_alloc_ctx_tensors(context, self.backend)
                if not buffer:
                    raise MemoryError('Vulkan tensor-buffer allocation failed')
                allocated = self.base.ggml_backend_buffer_get_size(buffer)
                while self.weights and self.cached_bytes + allocated > self.max_buffer_bytes:
                    victim = next((key for key in self.weights if key != protected_key), None)
                    if victim is None:
                        raise MemoryError('Vulkan cache plus active matrices exceeds its cap')
                    self._evict(victim)
                self.peak_buffer_bytes = max(self.peak_buffer_bytes, allocated + self.cached_bytes)
                if not cached:
                    self.base.ggml_backend_tensor_set(a, w.ctypes.data, 0, w.nbytes)
                self.base.ggml_backend_tensor_set(c, x.ctypes.data, 0, x.nbytes)
                status = self.base.ggml_backend_graph_compute(self.backend, graph)
                if status != 0:
                    raise RuntimeError(f'Vulkan graph execution failed: {status}')
                self.base.ggml_backend_tensor_get(result, output.ctypes.data, 0, output.nbytes)
                self.calls += 1
                self.transferred_bytes += x.nbytes + output.nbytes + (0 if cached else w.nbytes)
                return output
            finally:
                self.base.ggml_backend_synchronize(self.backend)
                if buffer:
                    self.base.ggml_backend_buffer_free(buffer)
                self.base.ggml_free(context)
                self.seconds += time.perf_counter() - tick

    def synchronize(self):
        if self.backend:
            self.base.ggml_backend_synchronize(self.backend)

    def set_budget(self, limit):
        if type(limit) is not int or limit <= 0:
            raise ValueError('Vulkan buffer budget must be positive')
        with self.lock:
            self.synchronize()
            self.max_buffer_bytes = limit
            while self.weights and self.cached_bytes > limit // 2:
                self._evict(next(iter(self.weights)))

    def close(self):
        with self.lock:
            if self.backend:
                self.synchronize()
                for key in list(self.weights):
                    self._evict(key)
                self.base.ggml_backend_free(self.backend)
                self.backend = None

    def statistics(self):
        return dict(device=self.device_name, matmul_calls=self.calls,
                    wall_seconds=self.seconds, transfer_bytes=self.transferred_bytes,
                    peak_buffer_bytes=self.peak_buffer_bytes, host_backed=True,
                    cached_weight_bytes=self.cached_bytes, weight_cache_hits=self.cache_hits)
