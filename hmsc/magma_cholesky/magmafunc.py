import tensorflow as tf
import os

# Load the custom operator library
lib_path = os.path.join(os.path.dirname(__file__), 'magma_cholesky.so')

#magma_lib = tf.load_op_library('./magma_cholesky.so')
magma_lib = tf.load_op_library(lib_path)
def M_cholesky(input_tensor):
    """
    Applies the custom Cholesky decomposition operator on the input tensor.

    Args:
        input_tensor (tf.Tensor): The input matrix or array of matrices to decompose. With larger matrices arrays are slower.

    Returns:
        tf.Tensor: The Cholesky decomposition of the input tensor.
    """
    return magma_lib.magma_cholesky(input_tensor)

