#ifdef TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#include "magma_cholesky.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "magma_v2.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename T>
void MagmaCholeskyFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int n, const T* in, T* out, int num_matrices) {
  magma_init();

  int matrix_size = n * n;
  hipStream_t stream = NULL;
  magma_int_t* d_info;
  magma_device_t device;
  magma_queue_t magma_queue;
  T** dA_array;

  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking); // Data transfer stream
  magma_getdevice(&device);
  magma_queue_create_from_hip(device, stream, NULL, NULL, &magma_queue); // Magma queue

  // Allocate device memory for d_info and dA_array
  hipMalloc((void**)&d_info, sizeof(magma_int_t) * num_matrices);
  hipMalloc((void**)&dA_array, sizeof(T*) * num_matrices);

  // Allocate and copy matrix pointers to dA_array
  T* matrices = out; // Assuming out points to the batched matrix array
  T* hA_array[num_matrices];
  for (int i = 0; i < num_matrices; i++) {
    hA_array[i] = matrices + i * matrix_size;
  }

  // Copy matrix pointers to device
  hipMemcpy(out, in, sizeof(T) * matrix_size * num_matrices, hipMemcpyDeviceToDevice);
  hipMemcpy(dA_array, hA_array, sizeof(T*) * num_matrices, hipMemcpyHostToDevice);
  hipDeviceSynchronize();

  if (num_matrices == 1) {
    // Single matrix case
    magma_dpotrf_expert_gpu(MagmaLower, n, out, n, d_info, 1024, MagmaNative);
    magmablas_dtranspose_inplace(n, out, n, magma_queue);
  } else {
    // Batched case for multiple matrices
    magma_dpotrf_batched(MagmaLower, n, dA_array, n, d_info, num_matrices, magma_queue);
    for (int i = 0; i < num_matrices; i++) {
      magmablas_dtranspose_inplace(n, hA_array[i], n, magma_queue);
    }
  }

  // Copy info back to host
  int* h_info = new int[num_matrices];
  hipMemcpyAsync(h_info, d_info, sizeof(int) * num_matrices, hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);

  for (int i = 0; i < num_matrices; i++) {
    if (0 != h_info[i]) {
      std::stringstream ss;
      ss << -h_info[i] << "-th parameter wrong for matrix " << i;
      delete[] h_info;
      hipFree(d_info);
      hipFree(dA_array);
      hipStreamDestroy(stream);
      magma_queue_destroy(magma_queue);
      throw std::runtime_error(ss.str());
    }
  }

  delete[] h_info;
  hipFree(d_info);
  hipFree(dA_array);
  hipStreamDestroy(stream);
  magma_queue_destroy(magma_queue);
};

// Explicitly instantiate functors for the types of OpKernels registered.
// template struct MagmaCholeskyFunctor<GPUDevice, float>;
template struct MagmaCholeskyFunctor<GPUDevice, double>;

#endif  // TENSORFLOW_USE_ROCM

