# Makefile

# Check that MAGMA_ROOT is defined
ifndef MAGMA_ROOT
$(error MAGMA_ROOT is undefined)
endif

# Compile flags for TensorFlow
TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
PYTHON_INCLUDES := $(shell python3-config --includes)

# Directory for magma_cholesky files
SRC_DIR = hmsc/magma_cholesky

# Kernel file and library output in src/magma_cholesky
KERNEL_SRC = $(SRC_DIR)/magma_cholesky.cu.cc
KERNEL_OBJ = $(SRC_DIR)/magma_cholesky.cu.o
LIBRARY = $(SRC_DIR)/magma_cholesky.so

# Use MAGMA_ROOT for Magma library paths
MAGMA_INCLUDE = $(MAGMA_ROOT)/include
MAGMA_LIB = $(MAGMA_ROOT)/lib

# Compile the GPU kernels into the object file
$(KERNEL_OBJ): $(KERNEL_SRC)
	hipcc -std=c++14 -c -o $(KERNEL_OBJ) $(KERNEL_SRC) $(TF_CFLAGS) -D EIGEN_USE_HIP=1 -D TENSORFLOW_USE_ROCM=1 -fPIC --offload-arch=gfx90a -O3 -I$(MAGMA_INCLUDE) -L$(MAGMA_LIB) -I/opt

# Compile the final shared library
$(LIBRARY): $(SRC_DIR)/magma_cholesky.cc $(KERNEL_OBJ)
	gcc -std=c++14 -shared -o $(LIBRARY) $(SRC_DIR)/magma_cholesky.cc $(KERNEL_OBJ) $(TF_CFLAGS) -fPIC $(TF_LFLAGS) -I$(MAGMA_INCLUDE) -L$(MAGMA_LIB) -lmagma -D TENSORFLOW_USE_ROCM=1 $(PYTHON_INCLUDES)

all: $(LIBRARY)

clean:
	rm -f $(KERNEL_OBJ) $(LIBRARY)
