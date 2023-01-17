import tensorflow as tf

from scipy.linalg import cholesky

def scipy_cholesky(X):
    return cholesky(X)

def kron(A, B):
    tmp1 = A[None, None, :, :] * B[:, :, None, None]
    shape = [tf.shape(A)[0] * tf.shape(B)[0], tf.shape(A)[1] * tf.shape(B)[1]]
    return tf.reshape(tf.transpose(tmp1, [0, 2, 1, 3]), shape)