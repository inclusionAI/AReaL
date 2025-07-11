#include <torch/nn/functional.h>
#include <torch/python.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void gae_kernel_1d_nolp_misalign(const float *rewards, const float *values,
                                            const int *cu_seqlens, const bool *bootstrap,
                                            float *adv_out, float *ret_out, int batch_size,
                                            float gamma, float lmbda) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size) { return; }
  // get the idx-th start index from cu_seqlens
  int rs_idx = cu_seqlens[idx];
  int re_idx = cu_seqlens[idx + 1];
  int vs_idx = rs_idx + idx;
  float lastgae = 0.0;
  for (int i = re_idx - rs_idx - 1; i >= 0; i--) {
    float nex_v = i == re_idx - rs_idx - 1 && !bootstrap[idx] ? 0.0 : values[vs_idx + i + 1];
    float delta = rewards[rs_idx + i] + gamma * nex_v - values[vs_idx + i];
    lastgae = delta + gamma * lmbda * lastgae;
    adv_out[rs_idx + i] = lastgae;
    ret_out[rs_idx + i] = lastgae + values[vs_idx + i];
  }
}

template<int num_threads>
std::vector<at::Tensor> gae_1d_nolp_misalign(at::Tensor &rewards, at::Tensor &values,
                                             at::Tensor &cu_seqlens, at::Tensor &bootstrap,
                                             float gamma, float lmbda) {
  int batch_size = cu_seqlens.numel() - 1;
  int total_seqlen = rewards.size(0);
  CHECK_DEVICE(rewards);
  CHECK_DEVICE(values);
  CHECK_DEVICE(cu_seqlens);
  CHECK_DEVICE(bootstrap);
  CHECK_CONTIGUOUS(rewards);
  CHECK_CONTIGUOUS(values);
  CHECK_CONTIGUOUS(cu_seqlens);
  CHECK_CONTIGUOUS(bootstrap);
  CHECK_SHAPE(bootstrap, batch_size);
  CHECK_SHAPE(values, total_seqlen + batch_size);
  TORCH_CHECK(bootstrap.dtype() == torch::kBool, "bootstrap must be bool");
  TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
  TORCH_CHECK(cu_seqlens[0].item<int>() == 0, "cu_seqlens[0] must be 0");
  TORCH_CHECK(cu_seqlens[-1].item<int>() == total_seqlen, "cu_seqlens[-1] must be total_seqlen");
  TORCH_CHECK(rewards.dtype() == values.dtype(), "rewards and values must have the same dtype");

  int num_blocks = (batch_size + num_threads - 1) / num_threads;
  auto adv_out = at::zeros_like(rewards);
  auto ret_out = at::zeros_like(rewards);
  gae_kernel_1d_nolp_misalign<<<num_blocks, num_threads>>>(
      rewards.data_ptr<float>(), values.data_ptr<float>(), cu_seqlens.data_ptr<int>(),
      bootstrap.data_ptr<bool>(), adv_out.data_ptr<float>(), ret_out.data_ptr<float>(), batch_size,
      gamma, lmbda);
  return {adv_out, ret_out};
}

__global__ void gae_kernel_2d_olp(const float *rewards, int r_stride, const float *values,
                                  int v_stride, const bool *done, int d_stride,
                                  const int *done_y_indices, const int *cu_num_dones,
                                  const bool *is_truncate, float *adv_out, int adv_stride,
                                  float *ret_out, int ret_stride, int batch_size, int horizon,
                                  float gamma, float lmbda) {
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int t_breadth = cu_num_dones[batch_idx + 1] - cu_num_dones[batch_idx] + 1;
  if (batch_idx >= batch_size || seq_idx >= t_breadth) { return; }
  int t_start = seq_idx == 0 ? 0 : done_y_indices[cu_num_dones[batch_idx] + seq_idx - 1];
  int t_end =
      seq_idx == t_breadth - 1 ? horizon : done_y_indices[seq_idx + cu_num_dones[batch_idx]];
  bool should_abandon_finalgae =
      seq_idx != t_breadth - 1 && is_truncate[seq_idx + cu_num_dones[batch_idx]];
  float lastgae = 0.0;
  for (int t = t_end - 1; t >= t_start; t--) {
    float nex_v = (t == t_end - 1 && done[batch_idx * d_stride + t + 1])
                      ? 0.0
                      : values[batch_idx * v_stride + t + 1];
    float cur_v = values[batch_idx * v_stride + t];
    float delta = rewards[batch_idx * r_stride + t] + gamma * nex_v - cur_v;
    lastgae = delta + gamma * lmbda * lastgae;
    if (t == t_end - 1 && should_abandon_finalgae) { lastgae = 0.0; }
    adv_out[batch_idx * adv_stride + t] = lastgae;
    ret_out[batch_idx * ret_stride + t] = lastgae + cur_v;
  }
}

template<int num_threads_x, int num_threads_y>
std::vector<at::Tensor> gae_2d_olp(at::Tensor &rewards, at::Tensor &values, at::Tensor &done,
                                   at::Tensor &done_y_indices, at::Tensor &cu_num_dones,
                                   int max_num_dones, at::Tensor &is_truncate, float gamma,
                                   float lmbda) {
  int batch_size = rewards.size(0);
  int horizon = rewards.size(1);
  int num_dones = done_y_indices.numel();
  CHECK_DEVICE(rewards);
  CHECK_DEVICE(values);
  CHECK_DEVICE(done);
  CHECK_DEVICE(done_y_indices);
  CHECK_DEVICE(cu_num_dones);
  CHECK_DEVICE(is_truncate);
  CHECK_CONTIGUOUS(done_y_indices);
  CHECK_CONTIGUOUS(cu_num_dones);
  CHECK_CONTIGUOUS(rewards);
  CHECK_CONTIGUOUS(values);
  CHECK_CONTIGUOUS(done);
  CHECK_CONTIGUOUS(is_truncate);
  CHECK_SHAPE(values, batch_size, horizon + 1);
  CHECK_SHAPE(done, batch_size, horizon + 1);
  CHECK_SHAPE(rewards, batch_size, horizon);
  CHECK_SHAPE(cu_num_dones, batch_size + 1);
  CHECK_SHAPE(is_truncate, num_dones);
  TORCH_CHECK(done_y_indices.dtype() == torch::kInt32, "done_y_indices must be int32");
  TORCH_CHECK(cu_num_dones.dtype() == torch::kInt32, "cu_num_dones must be int32");
  TORCH_CHECK(is_truncate.dtype() == torch::kBool, "is_truncate must be bool");
  TORCH_CHECK(rewards.dtype() == values.dtype(), "rewards and values must have the same dtype");

  auto adv_out = at::zeros_like(rewards);
  auto ret_out = at::zeros_like(rewards);

  dim3 threadsPerBlock(num_threads_x, num_threads_y);
  dim3 numBlocks((batch_size + num_threads_x - 1) / num_threads_x,
                 (max_num_dones + 1 + num_threads_y - 1) / num_threads_y);
  gae_kernel_2d_olp<<<numBlocks, threadsPerBlock>>>(
      rewards.data_ptr<float>(), rewards.stride(0), values.data_ptr<float>(), values.stride(0),
      done.data_ptr<bool>(), done.stride(0), done_y_indices.data_ptr<int>(),
      cu_num_dones.data_ptr<int>(), is_truncate.data_ptr<bool>(), adv_out.data_ptr<float>(),
      adv_out.stride(0), ret_out.data_ptr<float>(), ret_out.stride(0), batch_size, horizon, gamma,
      lmbda);
  return {adv_out, ret_out};
}

__global__ void gae_kernel_2d_nolp(const float *rewards, int r_stride, const float *values,
                                   int v_stride, const bool *on_reset, int or_stride,
                                   const int *on_reset_y_indices, const int *cu_num_resets,
                                   const bool *bootstrap, float *adv_out, int adv_stride,
                                   float *ret_out, int ret_stride, int batch_size, int horizon,
                                   float gamma, float lmbda) {
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int t_breadth = cu_num_resets[batch_idx + 1] - cu_num_resets[batch_idx] + 1;
  if (batch_idx >= batch_size || seq_idx >= t_breadth) { return; }
  int t_start = seq_idx == 0 ? 0 : on_reset_y_indices[cu_num_resets[batch_idx] + seq_idx - 1];
  int t_end =
      seq_idx == t_breadth - 1 ? horizon : on_reset_y_indices[seq_idx + cu_num_resets[batch_idx]];
  bool should_bootstrap = seq_idx == t_breadth - 1 || bootstrap[seq_idx + cu_num_resets[batch_idx]];
  float lastgae = 0.0;
  // t_end - 1 is a `done` step, which should not be included in the calculation
  for (int t = t_end - 2; t >= t_start; t--) {
    float nex_v =
        (t == t_end - 1 && !should_bootstrap) ? 0.0 : values[batch_idx * v_stride + t + 1];
    float cur_v = values[batch_idx * v_stride + t];
    float delta = rewards[batch_idx * r_stride + t] + gamma * nex_v - cur_v;
    lastgae = delta + gamma * lmbda * lastgae;
    adv_out[batch_idx * adv_stride + t] = lastgae;
    ret_out[batch_idx * ret_stride + t] = lastgae + cur_v;
  }
}

template<int num_threads_x, int num_threads_y>
std::vector<at::Tensor> gae_2d_nolp(at::Tensor &rewards, at::Tensor &values, at::Tensor &on_reset,
                                    at::Tensor &on_reset_y_indices, at::Tensor &cu_num_resets,
                                    int max_num_resets, at::Tensor &bootstrap, float gamma,
                                    float lmbda) {
  int batch_size = rewards.size(0);
  int horizon = rewards.size(1);
  int num_resets = on_reset_y_indices.numel();
  CHECK_DEVICE(rewards);
  CHECK_DEVICE(values);
  CHECK_DEVICE(on_reset);
  CHECK_DEVICE(on_reset_y_indices);
  CHECK_DEVICE(cu_num_resets);
  CHECK_DEVICE(bootstrap);
  CHECK_CONTIGUOUS(on_reset_y_indices);
  CHECK_CONTIGUOUS(cu_num_resets);
  CHECK_CONTIGUOUS(rewards);
  CHECK_CONTIGUOUS(values);
  CHECK_CONTIGUOUS(on_reset);
  CHECK_CONTIGUOUS(bootstrap);
  CHECK_SHAPE(values, batch_size, horizon + 1);
  CHECK_SHAPE(on_reset, batch_size, horizon + 1);
  CHECK_SHAPE(rewards, batch_size, horizon);
  CHECK_SHAPE(cu_num_resets, batch_size + 1);
  CHECK_SHAPE(bootstrap, num_resets);
  TORCH_CHECK(on_reset_y_indices.dtype() == torch::kInt32, "on_reset_y_indices must be int32");
  TORCH_CHECK(cu_num_resets.dtype() == torch::kInt32, "cu_num_resets must be int32");
  TORCH_CHECK(bootstrap.dtype() == torch::kBool, "bootstrap must be bool");
  TORCH_CHECK(rewards.dtype() == values.dtype(), "rewards and values must have the same dtype");

  auto adv_out = at::zeros_like(rewards);
  auto ret_out = at::zeros_like(rewards);

  dim3 threadsPerBlock(num_threads_x, num_threads_y);
  dim3 numBlocks((batch_size + num_threads_x - 1) / num_threads_x,
                 (max_num_resets + 1 + num_threads_y - 1) / num_threads_y);
  gae_kernel_2d_olp<<<numBlocks, threadsPerBlock>>>(
      rewards.data_ptr<float>(), rewards.stride(0), values.data_ptr<float>(), values.stride(0),
      on_reset.data_ptr<bool>(), on_reset.stride(0), on_reset_y_indices.data_ptr<int>(),
      cu_num_resets.data_ptr<int>(), bootstrap.data_ptr<bool>(), adv_out.data_ptr<float>(),
      adv_out.stride(0), ret_out.data_ptr<float>(), ret_out.stride(0), batch_size, horizon, gamma,
      lmbda);
  return {adv_out, ret_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Generalized Advantage Estimation (CUDA)";
  m.def("gae_1d_nolp_misalign", &gae_1d_nolp_misalign<256>,
        "1D Generalized Advantage Estimation (CUDA) with no termination overlap and misaligned "
        "rewards/values");
  m.def("gae_2d_olp", &gae_2d_olp<16, 16>,
        "2D Generalized Advantage Estimation (CUDA) with overlapped termination");
  m.def("gae_2d_nolp", &gae_2d_nolp<16, 16>,
        "2D Generalized Advantage Estimation (CUDA) with no termination overlap");
}