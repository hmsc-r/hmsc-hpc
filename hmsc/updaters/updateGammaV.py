import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfla, tfr = tf.linalg, tf.random

from hmsc.utils.tflautils import kron


def updateGammaV(params, data, priorHyperparams, dtype=np.float64):
    """Update prior(s) for whole model:
    Gamma - influence of traits on species niches, and
    V - residual covariance of species niches.

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Beta - species niches
        Gamma - influence of traits on species niches
        iV - inverse of residual covariance of species niches
        T -
        mGamma -
        iUGamma -
        V0 -
        f0 -
    """

    Beta = params["Beta"]
    Gamma = params["Gamma"]

    T = data["T"]
    mGamma = priorHyperparams["mGamma"]
    iUGamma = priorHyperparams["iUGamma"]
    V0 = priorHyperparams["V0"]
    f0 = priorHyperparams["f0"]

    nc = tf.shape(Beta)[0]
    ns = tf.shape(Beta)[1]
    nt = Gamma.shape[-1]
    
    Mu = tf.matmul(Gamma, T, transpose_b=True)
    E = Beta - Mu
    A = tf.matmul(E, E, transpose_b=True)
    Vn = tfla.cholesky_solve(tfla.cholesky(A + V0), tf.eye(nc, dtype=dtype))
    LVn = tfla.cholesky(Vn)
    iV = tfd.WishartTriL(tf.cast(f0 + ns, dtype), LVn).sample()

    iSigmaGamma = iUGamma + kron(tf.matmul(T, T, transpose_a=True), iV)
    L = tfla.cholesky(iSigmaGamma)
    mg0 = tf.matmul(iUGamma, mGamma[:,None]) + tf.reshape(tf.matmul(iV, tf.matmul(Beta, T)), [nc*nt,1])
    mg1 = tfla.triangular_solve(L, mg0)
    Gamma = tf.reshape(tfla.triangular_solve(L, mg1 + tfr.normal([nc*nt,1], dtype=dtype), adjoint=True), [nc,nt])
    return {"Gamma": Gamma, "V": tfla.inv(iV)}
