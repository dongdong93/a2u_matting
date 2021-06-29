#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int bidownsample_naive_forward_cuda(at::Tensor features, at::Tensor masks,
                              int kernel_size, int group_size, int scale_factor,
                              at::Tensor output);

int bidownsample_naive_backward_cuda(at::Tensor top_grad, at::Tensor features,
                               at::Tensor masks, int kernel_size,
                               int group_size, int scale_factor,
                               at::Tensor bottom_grad, at::Tensor mask_grad);
#endif

int bidownsample_naive_forward(at::Tensor features, at::Tensor masks,
                         int kernel_size, int group_size, int scale_factor,
                         at::Tensor output) {
  if (features.device().is_cuda()) {
#ifdef WITH_CUDA
    return bidownsample_naive_forward_cuda(features, masks, kernel_size,
        group_size, scale_factor, output);
#else
    AT_ERROR("bidownsample naive is not compiled with GPU support");
#endif
  }
  AT_ERROR("bidownsample naive is not implemented on CPU");
}

int bidownsample_naive_backward(at::Tensor top_grad, at::Tensor features,
                               at::Tensor masks, int kernel_size,
                               int group_size, int scale_factor,
                               at::Tensor bottom_grad, at::Tensor mask_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return bidownsample_naive_backward_cuda(top_grad, features, masks, kernel_size,
        group_size, scale_factor, bottom_grad, mask_grad);
#else
    AT_ERROR("bidownsample naive is not compiled with GPU support");
#endif
  }
  AT_ERROR("bidownsample naive is not implemented on CPU");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bidownsample_naive_forward, "bidownsample_naive forward");
  m.def("backward", &bidownsample_naive_backward, "bidownsample_naive backward");
}