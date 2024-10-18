import tensorflow as tf
import os

lib_path = os.environ.get('HMSC_TFOP_LIB')
if lib_path is None:
    raise RuntimeError("HMSC_TFOP_LIB environment variable is not set.")

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

