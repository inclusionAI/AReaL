#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template<typename T, int chunk_size>
__global__ void copyDataKernel(T *dst, const T *src, long *dst_offsets, long *src_offsets,
                               long *sizes, long N) {
  long interval_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (interval_id >= N) { return; }
  long chunk_id = blockIdx.y * blockDim.y + threadIdx.y;
  long interval_size = sizes[interval_id];
  long chunk_offset = chunk_id * chunk_size;
  if (chunk_offset >= interval_size) { return; }
  long dst_offset = dst_offsets[interval_id];
  long src_offset = src_offsets[interval_id];
  long _size = interval_size - chunk_offset;
  long size = (chunk_size < _size) ? chunk_size : _size;
  memcpy(dst + dst_offset + chunk_offset, src + src_offset + chunk_offset, size * sizeof(T));
}

template<typename T, int chunk_size>
void set_intervals(const at::Tensor src, at::Tensor dst, const at::Tensor intervals,
                   int max_interval_size) {
  CHECK_DEVICE(src);
  CHECK_DEVICE(dst);
  CHECK_DEVICE(intervals);

  CHECK_CONTIGUOUS(src);
  CHECK_CONTIGUOUS(dst);
  CHECK_CONTIGUOUS(intervals);

  TORCH_CHECK(src.dtype() == dst.dtype(),
              "Source and destination tensors must have the same dtype");

  TORCH_CHECK(intervals.dtype() == torch::kLong, "intervals must be of type long");

  long N = intervals.size(0);
  CHECK_SHAPE(intervals, N, 2);

  at::Tensor interval_sizes = intervals.select(1, 1) - intervals.select(1, 0);
  at::Tensor dst_offsets = intervals.select(1, 0).contiguous();
  at::Tensor src_offsets = interval_sizes.cumsum(0, at::kLong) - interval_sizes;

  // Launch CUDA kernel
  const int threads_per_block_x = 32;
  const int threads_per_block_y = 32;

  const int num_blocks_x = (N + threads_per_block_x - 1) / threads_per_block_x;

  const int n_chunks = (max_interval_size + chunk_size - 1) / chunk_size;
  const int num_blocks_y = (n_chunks + threads_per_block_y - 1) / threads_per_block_y;

  const dim3 numBlocks(num_blocks_x, num_blocks_y);
  const dim3 threadsPerBlock(threads_per_block_x, threads_per_block_y);

  copyDataKernel<T, chunk_size><<<numBlocks, threadsPerBlock>>>(
      dst.data_ptr<T>(), src.data_ptr<T>(), dst_offsets.data_ptr<long>(),
      src_offsets.data_ptr<long>(), interval_sizes.data_ptr<long>(), N);
}

template<typename T, int chunk_size>
at::Tensor slice_intervals(const at::Tensor src, const at::Tensor intervals, long total_size,
                           int max_interval_size) {
  CHECK_DEVICE(src);
  CHECK_DEVICE(intervals);

  CHECK_CONTIGUOUS(src);
  CHECK_CONTIGUOUS(intervals);

  TORCH_CHECK(intervals.dtype() == torch::kLong, "intervals must be of type long");

  long N = intervals.size(0);
  CHECK_SHAPE(intervals, N, 2);

  at::Tensor dst = at::empty({total_size}, src.options());

  at::Tensor interval_sizes = intervals.select(1, 1) - intervals.select(1, 0);
  at::Tensor src_offsets = intervals.select(1, 0).contiguous();
  at::Tensor dst_offsets = interval_sizes.cumsum(0, at::kLong) - interval_sizes;

  // Launch CUDA kernel
  const int threads_per_block_x = 32;
  const int threads_per_block_y = 32;

  const int num_blocks_x = (N + threads_per_block_x - 1) / threads_per_block_x;

  const int n_chunks = (max_interval_size + chunk_size - 1) / chunk_size;
  const int num_blocks_y = (n_chunks + threads_per_block_y - 1) / threads_per_block_y;

  const dim3 numBlocks(num_blocks_x, num_blocks_y);
  const dim3 threadsPerBlock(threads_per_block_x, threads_per_block_y);

  copyDataKernel<T, chunk_size><<<numBlocks, threadsPerBlock>>>(
      dst.data_ptr<T>(), src.data_ptr<T>(), dst_offsets.data_ptr<long>(),
      src_offsets.data_ptr<long>(), interval_sizes.data_ptr<long>(), N);
  return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_intervals_fp32", &set_intervals<float, 2048>, "Set intervals of a 1D tensor");
  m.def("set_intervals_fp16", &set_intervals<at::Half, 2048>, "Set intervals of a 1D tensor");
  m.def("set_intervals_bf16", &set_intervals<at::BFloat16, 2048>, "Set intervals of a 1D tensor");
  m.def("slice_intervals_fp32", &slice_intervals<float, 2048>, "slice intervals of a 1D tensor");
  m.def("slice_intervals_fp16", &slice_intervals<at::Half, 2048>, "slice intervals of a 1D tensor");
  m.def("slice_intervals_bf16", &slice_intervals<at::BFloat16, 2048>,
        "slice intervals of a 1D tensor");
}
