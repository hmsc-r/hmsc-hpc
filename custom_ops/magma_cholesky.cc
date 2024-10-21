#include "magma_cholesky.h"
#include <iostream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <Python.h>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("MagmaCholesky")
    .Attr("T: numbertype")
    .Input("input: T")
    .Output("input_times_two: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status();
    });

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class MagmaCholeskyOp : public OpKernel {
 public:
  explicit MagmaCholeskyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    int input_dims = input_tensor.dims();

    OP_REQUIRES(context, input_dims == 3 || input_dims == 2 ,
                errors::InvalidArgument("Input tensor must be 2 or 3-dimensional"));

    int num_matrices = 1;
    if (input_dims==3){
    	num_matrices = input_tensor.dim_size(0);
    }
    int n = input_tensor.dim_size(1);
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    MagmaCholeskyFunctor<Device, T>()(
        context->eigen_device<Device>(),
	n,
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data(),
        num_matrices);
  }
};

// Register the CPU kernels.
//#define REGISTER_CPU(T)                                          \
//  REGISTER_KERNEL_BUILDER(                                       \
//      Name("MagmaCholesky").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
//      MagmaCholeskyOp<CPUDevice, T>);
//REGISTER_CPU(float);
//REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef TENSORFLOW_USE_ROCM
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class MagmaCholeskyFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("MagmaCholesky").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      MagmaCholeskyOp<GPUDevice, T>);
// REGISTER_GPU(float);
REGISTER_GPU(double);

#endif
// Python initialization function
extern "C" PyMODINIT_FUNC PyInit_magma_cholesky(void) {
    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "magma_cholesky",
        nullptr,
        -1,
        nullptr,
    };

    PyObject* module = PyModule_Create(&module_def);
    if (module == nullptr) {
        return nullptr;
    }

    return module;
}

