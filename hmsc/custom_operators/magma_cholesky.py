import tensorflow as tf

# Load the custom operator library
magma_lib = tf.load_op_library('./magma_cholesky.so')

def cholesky(input_tensor):
    """
    Applies the custom Cholesky decomposition operator on the input tensor.

    Args:
        input_tensor (tf.Tensor): The input matrix or array of matrices to decompose. With larger matrices arrays are slower.

    Returns:
        tf.Tensor: The Cholesky decomposition of the input tensor.
    """
    return magma_lib.magma_cholesky(input_tensor)

