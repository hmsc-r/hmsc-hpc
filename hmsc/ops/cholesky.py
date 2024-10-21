import tensorflow as tf
import os

lib_path = os.environ.get('HMSC_TFOP_LIB')
if lib_path is None:
    print("Using default TensorFlow operators (HMSC_TFOP_LIB is not set).", flush=True)
    CUSTOM_TFOP_LIB = None
else:
    CUSTOM_TFOP_LIB = tf.load_op_library(lib_path)
    CHOLESKY_MAGMA_SWITCHSIZE = int(os.environ.get("HMSC_CHOLESKY_MAGMA_SWITCHSIZE", 1000))
    print(f"Using custom TensorFlow operators from {lib_path} with "
          f"CHOLESKY_MAGMA_SWITCHSIZE={CHOLESKY_MAGMA_SWITCHSIZE}",
          flush=True)


def cholesky(tensor, *, name=None):
    """Computes the Cholesky decomposition of one or more square matrices.

    Uses the custom operator for large enough matrices.

    Args:
        tensor: Input tensor. Shape [..., M, M].
        name: The name for the operation.

    Returns:
        The Cholesky decomposition of the input tensor.
    """
    print(f'cholesky shape: {tensor.shape} name: {name}', flush=True)
    if CUSTOM_TFOP_LIB is None:
        return tf.linalg.cholesky(tensor, name=name)

    M = tensor.shape[-1]  # static shape
    if M is None:
        M = tf.shape(tensor)[-1]  # dynamic shape
    if M > CHOLESKY_MAGMA_SWITCHSIZE:
        return CUSTOM_TFOP_LIB.magma_cholesky(tensor, name=name)
    return tf.linalg.cholesky(tensor, name=name)

