#ifndef THCUNN_REVERSE_COPY_CUH
#define THCUNN_REVERSE_COPY_CUH

// Custom reverse copy kernel to avoid thrust::reverse_copy which uses
// reverse_iterator (incompatible with CUDA 12/13 device code in CUB/Thrust).
// Compatible with CUDA 10.2, 11.8, 12.9, 13.2.

static __global__ void THCUNN_reverse_copy_index_kernel(long* src, long* dst, ptrdiff_t n) {
  ptrdiff_t i = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[n - 1 - i];
  }
}

static inline void THCUNN_reverse_copy_index(THCState* state, long* src, long* dst, ptrdiff_t n) {
  if (n <= 0) return;
  cudaStream_t stream = THCState_getCurrentStream(state);
  const int blockSize = 256;
  int gridSize = (int)((n + blockSize - 1) / blockSize);
  if (gridSize > 65535) gridSize = 65535;
  THCUNN_reverse_copy_index_kernel<<<gridSize, blockSize, 0, stream>>>(src, dst, n);
}

#endif
