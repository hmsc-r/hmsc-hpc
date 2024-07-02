#ifdef TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#include "magma_cholesky.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "magma_v2.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <chrono>
using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename T>
void MagmaCholeskyFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int n, const T* in, T* out, int num_matrices) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  magma_init();
  
  int matrix_size = n*n;
  hipStream_t stream = NULL;
  magma_int_t* d_info = new magma_int_t[num_matrices];
  magma_device_t device;
  magma_queue_t magma_queue;
  

  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking); //Data transfer stream
  magma_getdevice(&device);
  magma_queue_create_from_hip(device,stream, NULL, NULL, &magma_queue); //Magma transpose queue
  

  hipMalloc(reinterpret_cast<void **>(&d_info), sizeof(int) * num_matrices);
  hipMemcpy(out, in, sizeof(T) * matrix_size*num_matrices, hipMemcpyDeviceToDevice);
  hipDeviceSynchronize();
  std::cout << num_matrices << " matrices\n";
  
  
  for (int i = 0; i < num_matrices; i++) {
    // Iterate over matrices, calculate cholesky and transpose
    T* matrix_out = out + i * matrix_size;
    magma_dpotrf_expert_gpu(MagmaLower, n, matrix_out, n, d_info + i, 1024, MagmaNative);
    magmablas_dtranspose_inplace(n, matrix_out, n, magma_queue);
  }
  
  // Copy info
  int* h_info = new int[num_matrices];
  hipMemcpyAsync(h_info, d_info, sizeof(int) * num_matrices, hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);

  for (int i = 0; i < num_matrices; i++) {
    if (0 != h_info[i]) {
      std::stringstream ss;
      ss << -h_info[i] << "-th parameter wrong for matrix " << i;
      delete[] h_info;
      hipFree(&d_info);
      hipStreamDestroy(stream);
      throw std::runtime_error(ss.str());
    }
  }

  delete[] h_info;
  hipFree(&d_info);
  hipStreamDestroy(stream);
  magma_queue_destroy(magma_queue);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

};

// Explicitly instantiate functors for the types of OpKernels registered.
// template struct MagmaCholeskyFunctor<GPUDevice, float>;
template struct MagmaCholeskyFunctor<GPUDevice, double>;

#endif  // TENSORFLOW_USE_ROCM

