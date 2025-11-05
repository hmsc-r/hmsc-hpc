import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hmsc.utils.tfla_utils import kron
from hmsc.utils.tf_named_func import tf_named_func
from hmsc.utils.phylo_fast_utils import phyloFastBilinearDet, phyloFastBilinearDetBatched
tfla, tfm, tfr = tf.linalg, tf.math, tf.random
tfd = tfp.distributions

@tf_named_func("rho")
def updateRhoInd(params,
                 data,
                 priorHyperparams,
                 it=None,
                 rhoIndUpdateN=1,
                 flag_fast_phylo_batched=True,
                 dtype=np.float64):
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
  if it == None and (rhoIndUpdateN != 0 or rhoIndUpdateN != rhoN):
    raise ValueError(f"Incompatible rhoIndUpdateN={rhoIndUpdateN} for it=None.")
  if rhoIndUpdateN == 0:
    rhoIndUpdateN = rhoN
  if it == None:
    rhoIndUpdate = tf.range(rhoIndUpdateN)
  else:
    rhoIndUpdate = tfm.mod((it * rhoIndUpdateN) + tf.range(rhoIndUpdateN), rhoN)
  
  if phyloFlag == True:
    Mu = tf.matmul(Gamma, T, transpose_b=True)
    E = Beta - Mu
    LiV = tfla.cholesky(iV)
    if rhoN == 1:
      # logDetV = -2 * tf.reduce_sum(tfm.log(tfla.diag_part(LiV))) # can be ommited as is constant
      if phyloFast == False:
        E_VC = tf.matmul(E, VC)
        # for k in range(rhoN):
        #   tmp1 = tf.scatter_nd(tf.stack([tf.range(gN), k*tf.ones([gN],tf.int32)], -1), tf.range(gN), [gN,rhoN])
        #   rhoIndSt = tmp1 + tf.tensor_scatter_nd_update(rhoInd, [[k]], [0])
        #   rhoVec = tf.gather(rhopw[:,0], tf.gather(rhoIndSt, covRhoGroup, axis=-1))
        #   eQ = tf.expand_dims(rhoVec, -1)*eC + tf.expand_dims(1-rhoVec, -1)
        #   E_VC_eiQ05 = E_VC * tfm.rsqrt(eQ)
        #   qf = tf.einsum("gcj,ck,gkj->g", E_VC_eiQ05, iV, E_VC_eiQ05)
        #   logDet = tf.reduce_sum(tfm.log(eQ), [-1,-2]) # + ns*logDetV
        eQ = rhopw[:,0,None]*eC + (1-rhopw[:,0,None])
        E_VC_eiQ05 = E_VC * tfm.rsqrt(eQ[:,None,:])
        qf = tf.einsum("gcj,ck,gkj->g", E_VC_eiQ05, iV, E_VC_eiQ05)
        logDet = nc*tf.reduce_sum(tfm.log(eQ), -1) # + ns*logDetV
        # qf_1, logDet_1 = qf, logDet
      else:
        # rhoInd = tf.squeeze(tfr.categorical(tfm.log(rhopw[tf.newaxis,:,1]), rhoN, dtype=tf.int32), -1)
        print("phyloFast updateRhoInd") #TODO remove after debug
        LiVT_E = tf.matmul(LiV, E, transpose_a=True)
        # LiVT_E_arrx = tf.tile(LiVT_E[None,:,:,None], [gN,1,1,1])
        # LiVT_E_arrx = tf.tile(tf.transpose(LiVT_E)[:,None,None,:], [1,gN,1,1])
        LiVT_E_arr = tf.transpose(LiVT_E)[:,None,None,:]
        tmp1, tmp2 = pfBilinearDet(phyloTreeList, LiVT_E_arr, LiVT_E_arr, phyloTreeRoot, tf.ones([1,1],dtype), rhopw[:,0,None], dtype)
        qf = tf.reduce_sum(tfla.diag_part(tmp1), -1)
        logDet = nc*tmp2 # + ns*logDetV
        # qf_2, logDet_2 = qf, logDet

      logLike = tfm.log(rhopw[:,1]) - 0.5*logDet - 0.5*qf
      rhoInd = tf.squeeze(tfr.categorical(logLike[None,:], 1, dtype=tf.int32), -1)
      # tf.print(rhoInd)
    else:
      # tf.print("rhoIndUpdate:", rhoIndUpdate)
      for k in rhoIndUpdate:
        # tmp1 = tf.scatter_nd(tf.stack([tf.range(gN), k*tf.ones([gN],tf.int32)], -1), tf.range(gN), [gN,rhoN])
        # rhoIndSt = tmp1 + tf.tensor_scatter_nd_update(rhoInd, [[k]], [0])
        # creates a matrix of variations of rhoInd in k-th position
        ind = tf.stack([tf.range(gN), k*tf.ones([gN], tf.int32)], -1)
        rhoIndSt = tf.tensor_scatter_nd_update(tf.tile(rhoInd[None,:], [gN,1]), ind, tf.range(gN))
        rhoVec = tf.gather(rhopw[:,0], tf.gather(rhoIndSt, covRhoGroup, axis=-1))
        if phyloFast == False:
          V = tfla.cholesky_solve(LiV, tf.eye(nc, dtype=dtype))
          d105 = tfm.sqrt(rhoVec)
          d205 = tfm.sqrt(1 - rhoVec)
          V1 = d105[:,:,None] * V * d105[:,None,:]
          V2 = d205[:,:,None] * V * d205[:,None,:]
          Q = kron(C, V1) + kron(tf.eye(ns, dtype=dtype), V2)
          LQ = tfla.cholesky(Q)
          logDet = 2 * tf.reduce_sum(tfm.log(tfla.diag_part(LQ)), -1)
          iLQe = tfla.triangular_solve(LQ, tf.reshape(tf.transpose(E), [1,ns*nc,1]))
          qf = tf.squeeze(tf.matmul(iLQe, iLQe, transpose_a=True), [-2,-1])
          qf_1, logDet_1 = qf, logDet
        else:
          print(f"vector rho, phyloFast updateRhoInd, iter {k}") #TODO remove after debug
          E_arr = tf.transpose(E)[:,None,:,None]
          tmp1, logDet = pfBilinearDet(phyloTreeList, E_arr, E_arr, phyloTreeRoot, iV, rhoVec, dtype)
          qf = tf.squeeze(tmp1, [-1,-2])
          qf_2, logDet_2 = qf, logDet
          
        logLike = tfm.log(rhopw[:,1]) - 0.5*logDet - 0.5*qf
        indNew = tf.squeeze(tfr.categorical(logLike[None,:], 1, dtype=np.int32), -1)
        rhoInd = tf.tensor_scatter_nd_update(rhoInd, tf.cast([[k]], np.int32), indNew)
      
  return rhoInd
