import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hmsc.utils.tf_named_func import tf_named_func
from hmsc.utils.fast_phylo_utils import phyloFastBilinearDet, phyloFastBilinearDetBatched
tfla, tfm, tfr = tf.linalg, tf.math, tf.random
tfd = tfp.distributions

@tf_named_func("rho")
def updateRhoInd(params, data, priorHyperparams, flag_fast_phylo_batched, dtype=np.float64):
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
  phyloFlag, phyloFast = data["phyloFlag"], data["phyloFast"]
  phyloTreeList, phyloTreeRoot = data["phyloTreeList"], data["phyloTreeRoot"]
  C, eC, VC = data["C"], data["eC"], data["VC"]
  covRhoGroup = data["covRhoGroup"]
  rhopw = priorHyperparams["rhopw"]
  nc, ns = Beta.shape
  gN = rhopw.shape[0]
  rhoN = rhoInd.shape[0]
  pfBilinearDet = phyloFastBilinearDetBatched if flag_fast_phylo_batched else phyloFastBilinearDet
  
  if rhoN == 1:
    if phyloFlag == True:
      Mu = tf.matmul(Gamma, T, transpose_b=True)
      E = Beta - Mu
      LiV = tfla.cholesky(iV)
      # logDetV = -2 * tf.reduce_sum(tfm.log(tfla.diag_part(LiV))) # can be ommited as is constant
      if phyloFast == False:
        E_VC = tf.matmul(E, VC)
        for k in range(rhoN): #TODO reuse for vector rho
          tmp1 = tf.scatter_nd(tf.stack([tf.range(gN), k*tf.ones([gN],tf.int32)], -1), tf.range(gN), [gN,rhoN])
          rhoIndSt = tmp1 + tf.tensor_scatter_nd_update(rhoInd, [[k]], [0])
          rhoVec = tf.gather(rhopw[:,0], tf.gather(rhoIndSt, covRhoGroup, axis=-1))
          eQ = tf.expand_dims(rhoVec, -1)*eC + tf.expand_dims(1-rhoVec, -1)
          E_VC_eiQ05 = E_VC * tfm.rsqrt(eQ)
          qf = tf.einsum("gcj,ck,gkj->g", E_VC_eiQ05, iV, E_VC_eiQ05)
          logDet = tf.reduce_sum(tfm.log(eQ), [-1,-2]) # + ns*logDetV
        qf_1, logDet_1 = qf, logDet
      else:
        # rhoInd = tf.squeeze(tfr.categorical(tfm.log(rhopw[tf.newaxis,:,1]), rhoN, dtype=tf.int32), -1)
        print("phyloFast updateRhoInd_vec") #TODO remove after debug
        LiVT_E = tf.matmul(LiV, E, transpose_a=True)
        # LiVT_E_arrx = tf.tile(LiVT_E[None,:,:,None], [gN,1,1,1])
        LiVT_E_arrx = tf.tile(tf.transpose(LiVT_E)[:,None,None,:], [1,gN,1,1])
        tmp1, tmp2 = pfBilinearDet(phyloTreeList, LiVT_E_arrx, LiVT_E_arrx, phyloTreeRoot, tf.ones([1,1],dtype), rhopw[:,None,0], dtype)
        qf = tf.reduce_sum(tfla.diag_part(tmp1), -1)
        logDet = nc*tmp2 # + ns*logDetV
        qf_2, logDet_2 = qf, logDet

      logLike = tfm.log(rhopw[:,1]) - 0.5*logDet - 0.5*qf
      indNew = tf.squeeze(tfr.categorical(logLike[None,:], 1, dtype=tf.int32), -1)
      rhoInd = tf.tensor_scatter_nd_update(rhoInd, [[0]], indNew)
      # tf.print(rhoInd)
      
  return rhoInd
