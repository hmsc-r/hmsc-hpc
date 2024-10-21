#ifndef MAGMA_CHOLESKY_H_
#define MAGMA_CHOLESKY_H_

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T>
struct MagmaCholeskyFunctor {
  void operator()(const Device& d, int n, const T* in, T* out, int num_matrices);
};

#if TENSORFLOW_USE_ROCM
template <typename T>
struct MagmaCholeskyFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int n, const T* in, T* out, int num_matrices);
 };
#endif

#endif MAGMA_CHOLESKY_H_
