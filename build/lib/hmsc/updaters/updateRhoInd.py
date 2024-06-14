import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hmsc.utils.tf_named_func import tf_named_func
tfla, tfm, tfr = tf.linalg, tf.math, tf.random
tfd = tfp.distributions

@tf_named_func("rho")
def updateRhoInd(params, data, priorHyperparams, dtype=np.float64):
  """Update rho paramters:
  
  
  Parameters
  ----------
  params : dict
  The initial value of the model parameter(s):
  
  """
  
  Beta = params["Beta"]
  Gamma = params["Gamma"]
  iV = params["iV"]
  rhoInd = params["rhoInd"]
  T = data["T"]
  C, eC, VC = data["C"], data["eC"], data["VC"]
  rhoGroup = data["rhoGroup"]
  rhopw = priorHyperparams["rhopw"]
  
  if not (C is None):
    nc, ns = Beta.shape
    gN = rhopw.shape[0]
    rhoN = rhoInd.shape[0]
    Mu = tf.matmul(Gamma, T, transpose_b=True)
    E = Beta - Mu
    E_VC = tf.matmul(E, VC)
    LiV = tfla.cholesky(iV)
    logDetV = -2 * tf.reduce_sum(tfm.log(tfla.diag_part(LiV)))
    
    for k in range(rhoN):
      tmp1 = tf.scatter_nd(tf.stack([tf.range(gN), k*tf.ones([gN],tf.int32)], -1), tf.range(gN), [gN,rhoN])
      rhoIndSt = tmp1 + tf.tensor_scatter_nd_update(rhoInd, [[k]], [0])
      rhoVec = tf.gather(rhopw[:,0], tf.gather(rhoIndSt, rhoGroup, axis=-1))
      eQ = tf.expand_dims(rhoVec, -1)*eC + tf.expand_dims(1-rhoVec, -1)
      E_VC_eiQ05 = E_VC * tfm.rsqrt(eQ)
      qF = tf.einsum("gcj,ck,gkj->g", E_VC_eiQ05, iV, E_VC_eiQ05)
      logDet = ns*logDetV + tf.reduce_sum(tfm.log(eQ), [-1,-2])
      logLike = tfm.log(rhopw[:,1]) - 0.5*logDet - 0.5*qF
      indNew = tf.squeeze(tfr.categorical(logLike[None,:], 1, dtype=tf.int32), -1)
      rhoInd = tf.tensor_scatter_nd_update(rhoInd, [[k]], indNew)

  return rhoInd
