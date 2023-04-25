import tensorflow as tf

from scipy.linalg import cholesky
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve_triangular

def scipy_cholesky(X):
    return cholesky(X)

def kron(A, B):
    return tf.reshape(tf.einsum("ab,cd->acbd", A, B), shape = [tf.shape(A)[0] * tf.shape(B)[0], tf.shape(A)[1] * tf.shape(B)[1]]) 

def tf_sparse_matmul(A: tf.SparseTensor, B: tf.SparseTensor, dtype=tf.float64):
    A_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(A.indices, A.values, A.dense_shape)
    B_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(B.indices, B.values, B.dense_shape)
    C_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(a=A_sm, b=B_sm, type=dtype)
    C = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(C_sm, dtype)
    return tf.SparseTensor(C.indices, C.values, dense_shape=C.dense_shape)

def tf_sparse_cholesky(A: tf.SparseTensor, dtype=tf.float64):
    A_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(A.indices, A.values, A.dense_shape)
    # Obtain the Sparse Cholesky factor using AMD Ordering for reducing zero
    # fill-in (number of structural non-zeros in the sparse Cholesky factor).
    ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(A_sm)
    cholesky_sparse_matrix = (sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(A_sm, ordering_amd, type=dtype))
    cholesky_sparse_tensor = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(cholesky_sparse_matrix, type=dtype)
    return tf.SparseTensor(cholesky_sparse_tensor.indices, cholesky_sparse_tensor.values, dense_shape=cholesky_sparse_tensor.dense_shape)

def convert_sparse_tensor_to_sparse_matrix(x, i, shape):
    return csr_matrix((x, (i[:,0], i[:,1])), shape=shape)

def convert_sparse_tensor_to_sparse_csc_matrix(x, i, shape):
    return csc_matrix(coo_matrix((x, (i[:,1], i[:,0])), shape=shape))

def scipy_sparse_solve_triangular(A_x, A_i, A_shape, B):
    return spsolve_triangular(convert_sparse_tensor_to_sparse_matrix(A_x, A_i, A_shape), B)
