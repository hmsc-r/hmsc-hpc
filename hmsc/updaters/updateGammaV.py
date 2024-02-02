import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hmsc.utils.tfla_utils import kron
from hmsc.utils.tf_named_func import tf_named_func
tfla, tfm, tfr = tf.linalg, tf.math, tf.random
tfd = tfp.distributions

@tf_named_func("gammaV")
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
    rhoInd = params["rhoInd"]
    T = data["T"]
    C, eC, VC = data["C"], data["eC"], data["VC"]
    rhoGroup = data["rhoGroup"]
    mGamma = priorHyperparams["mGamma"]
    iUGamma = priorHyperparams["iUGamma"]
    V0 = priorHyperparams["V0"]
    f0 = priorHyperparams["f0"]
    rhopw = priorHyperparams["rhopw"]

    nc, ns = Beta.shape
    nt = Gamma.shape[-1]
    
    Mu = tf.matmul(Gamma, T, transpose_b=True)
    E = Beta - Mu
    if C is None:
      A = tf.matmul(E, E, transpose_b=True)
    else:
      rhoVec = tf.gather(rhopw[:,0], tf.gather(rhoInd, rhoGroup))
      eQ = rhoVec[:,None]*eC + (1-rhoVec)[:,None]
      eiQ05 = tfm.rsqrt(eQ)
      eiQ05_E_VC = eiQ05 * tf.matmul(E, VC)
      A = tf.matmul(eiQ05_E_VC, eiQ05_E_VC, transpose_b=True)
      
    Vn = tfla.cholesky_solve(tfla.cholesky(A + V0), tf.eye(nc, dtype=dtype))
    LVn = tfla.cholesky(Vn)
    iV = tfd.WishartTriL(tf.cast(f0+ns, dtype), LVn).sample()
    
    if C is None:
      iSigmaGamma = iUGamma + kron(tf.matmul(T, T, transpose_a=True), iV)
      mg0 = tf.matmul(iUGamma, mGamma[:,None]) + tf.reshape(tf.einsum("ck,kj,jt->tc", iV, Beta, T), [nt*nc,1])
    else:
      VCT_T = tf.matmul(VC, T, transpose_a=True)
      eiQ05_VCT_T = eiQ05[:,:,None] * VCT_T
      iSigmaGamma = iUGamma + tf.reshape(tf.einsum("cjt,ck,kjf->tcfk", eiQ05_VCT_T, iV, eiQ05_VCT_T), [nt*nc,nt*nc])
      tmp = tf.reshape(tf.transpose(tf.matmul(eiQ05 * tf.matmul(iV, eiQ05 * tf.matmul(Beta, VC)), VCT_T)), [nt*nc,1])
      mg0 = tf.matmul(iUGamma, mGamma[:,None]) + tmp
      
    L = tfla.cholesky(iSigmaGamma)
    mg1 = tfla.triangular_solve(L, mg0)
    Gamma = tf.transpose(tf.reshape(tfla.triangular_solve(L, mg1 + tfr.normal([nt*nc,1], dtype=dtype), adjoint=True), [nt,nc]))
    return Gamma, iV
