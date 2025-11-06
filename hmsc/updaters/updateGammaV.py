import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hmsc.utils.tfla_utils import kron
from hmsc.utils.tf_named_func import tf_named_func
from hmsc.utils.phylo_fast_utils import phyloFastBilinearDetBatched as pfBilinearDet
from hmsc.utils.phylo_fast_utils import phyloFastGetPariV
tfla, tfm, tfr = tf.linalg, tf.math, tf.random
tfd, tfb = tfp.distributions, tfp.bijectors

@tf_named_func("gammaV")
def updateGammaV(params,
                 data,
                 priorHyperparams,
                 updateGamma=True,
                 updateiV=True,
                 phyloFastBatched=True,
                 dtype=np.float64):
    """Update prior(s) for whole model:
    Gamma - influence of traits on species niches, and
    V - trait-unexplained residual covariance of species niches.

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Beta - species niches
        Gamma - influence of traits on species niches
        iV - precision (inverse of covariance) of trait-unexplained species niches
        T - species traits
        mGamma - prior mean for Gamma
        iUGamma - prior precision for Gamma
        V0 - prior scale matrix for V
        f0 - prior concentration for V
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
    mGamma = priorHyperparams["mGamma"]
    iUGamma = priorHyperparams["iUGamma"]
    V0 = priorHyperparams["V0"]
    f0 = priorHyperparams["f0"]
    rhopw = priorHyperparams["rhopw"]
    nc, ns = Beta.shape
    nt = Gamma.shape[-1]
    rhoN = rhoInd.shape[0]
    
    Mu = tf.matmul(Gamma, T, transpose_b=True)
    E = Beta - Mu
    if phyloFlag == False:
      fn = f0 + ns
      A = tf.matmul(E, E, transpose_b=True)
    else:
      if rhoN == 1: # tf.size(rhoInd.length) == 1:
        rho = tf.gather(rhopw[:,0], rhoInd)
        if phyloFast == False:
          eQ = rho*eC + (1-rho)
          eiQ05 = tfm.rsqrt(eQ)
          E_VC_eiQ05 = eiQ05 * tf.matmul(E, VC)
          A = tf.matmul(E_VC_eiQ05, E_VC_eiQ05, transpose_b=True)
          A_1 = A #TODO remove
        else:
          print("scalar rho, phyloFast updateGammaV part 1") #TODO
          ET_arr = tf.transpose(E)[:,None,:]
          A, _ = pfBilinearDet(phyloTreeList, ET_arr, ET_arr, phyloTreeRoot, tf.ones([1,1],dtype), rho, dtype)
          A_2 = A #TODO remove
        fn = f0 + ns
      else:
        rhoVec = tf.gather(rhopw[:,0], tf.gather(rhoInd, covRhoGroup))
        if phyloFast == False:
          e = tf.reshape(tf.transpose(E), [ns*nc,1])
          V = tfla.cholesky_solve(tfla.cholesky(iV), tf.eye(nc, dtype=dtype)) 
          d105 = tfm.sqrt(rhoVec)
          d205 = tfm.sqrt(1 - rhoVec)
          D1V = d105[:,None] * V
          D2V = d205[:,None] * V
          V1 = D1V * d105
          V2 = D2V * d205
          Q = kron(C, V1) + kron(tf.eye(ns, dtype=dtype), V2)
          CxV = kron(C, V)
          CxD1V  = kron(C, D1V)
          LQ = tfla.cholesky(Q)
          iLQ_CxD1V = tfla.triangular_solve(LQ, CxD1V)
          S1 = tfla.diag(tf.tile(tf.cast(rhoVec==1, dtype), [ns])) + CxV - tf.matmul(iLQ_CxD1V, iLQ_CxD1V, transpose_a=True) #TODO add explicit zeroing for rho==0?
          m1 = tf.matmul(CxD1V, tfla.cholesky_solve(LQ, e), transpose_a=True)
          z1 = m1 + tf.matmul(tfla.cholesky(S1), tfr.normal([ns*nc, 1], dtype=dtype)) * tf.tile(tf.cast(rhoVec<1, dtype), [ns])[:,None]
          Z1 = tf.reshape(z1, [ns,nc])
          b1 = tf.reshape(Z1 * d105, [ns*nc,1])
          
          B2 = tf.transpose(tf.reshape(e - b1, [ns,nc]))
          W = tfla.diag(tf.cast(rhoVec==1, dtype)) + V2
          LW = tfla.cholesky(W)
          M2 = tf.matmul(D2V, tfla.cholesky_solve(LW, B2), transpose_a=True)
          iLW_D2V = tfla.triangular_solve(LW, D2V)
          S2 = tfla.diag(tf.cast(rhoVec<1, dtype)) + V - tf.matmul(iLW_D2V, iLW_D2V, transpose_a=True) #TODO add explicit zeroing for rho<0?
          Z2 = M2 + tf.matmul(tfla.cholesky(S2), tfr.normal([nc,ns], dtype=dtype)) * tf.cast(rhoVec==1, dtype)[:,None]
          
          fn = f0 + 2 * ns
          LC = tfla.cholesky(C) #TODO maybe rework with eC, VC
          iLCZ1 = tfla.triangular_solve(LC, Z1)
          A = tf.matmul(iLCZ1, iLCZ1, transpose_a=True) + tf.matmul(Z2, Z2, transpose_b=True)
          A1 = A
        else:
          fa, A = phyloFastGetPariV(phyloTreeList, phyloTreeRoot, E, iV, rhoVec, dtype=tf.float64)
          fn = f0 + fa
          A2 = A
    
    if updateiV:
      # Vn = tfla.cholesky_solve(tfla.cholesky(V0 + A), tf.eye(nc, dtype=dtype))
      # LVn = tfla.cholesky(Vn)
      LVn = tfb.CholeskyToInvCholesky().forward(tfla.cholesky(V0 + A))
      iV = tfd.WishartTriL(fn, LVn).sample()
    
    if phyloFlag == False:
      iSigmaGamma = iUGamma + kron(tf.matmul(T, T, transpose_a=True), iV)
      mg02 = tf.reshape(tf.einsum("ck,kj,jt->tc", iV, Beta, T), [nt*nc,1])
    else:
      if rhoN == 1:
        if phyloFast == False:
          VCt_T = tf.matmul(VC, T, transpose_a=True)
          eiQ05_VCt_T = eiQ05[:,None] * VCt_T
          Tt_iQ_T = tf.matmul(eiQ05_VCt_T, eiQ05_VCt_T, transpose_a=True)
          iSigmaGamma = iUGamma + kron(Tt_iQ_T, iV) #tf.reshape(tf.einsum("jt,ck,jf->tcfk", eiQ05_VCt_T, iV, eiQ05_VCt_T), [nt*nc,nt*nc])
          mg02 = tf.reshape(tf.transpose(tf.matmul(iV, tf.matmul(tf.matmul(Beta, VC) * eQ**-1, VCt_T))), [nt*nc,1])
          mg02_1, iSigmaGamma_1 = mg02, iSigmaGamma #TODO remove
        else:
          print("scalar rho, phyloFast updateGammaV part Gamma") #TODO remove
          T_arr = T[:,None,:]
          Tt_iQ_T, _ = pfBilinearDet(phyloTreeList, T_arr, T_arr, phyloTreeRoot, tf.ones([1,1],dtype), rho, dtype)
          iSigmaGamma = iUGamma + kron(Tt_iQ_T, iV)
          iV_Beta_arr = tf.matmul(Beta, iV, transpose_a=True)[:,None,:]
          iVBeta_iQ_Tr, _ = pfBilinearDet(phyloTreeList, iV_Beta_arr, T_arr, phyloTreeRoot, tf.ones([1,1],dtype), rho, dtype)
          mg02 = tf.reshape(tf.transpose(iVBeta_iQ_Tr), [nt*nc,1]) # remove transpose by swapping X and Y in prev line? 
          mg02_2, iSigmaGamma_2 = mg02, iSigmaGamma #TODO remove
      else:
        if phyloFast == False:
          print("vector rho, phyloBase updateGammaV part Gamma") #TODO remove
          V = tfla.cholesky_solve(tfla.cholesky(iV), tf.eye(nc, dtype=dtype)) 
          # d105 = tfm.sqrt(rhoVec)
          # d205 = tfm.sqrt(1 - rhoVec)
          D1V = d105[:,None] * V
          D2V = d205[:,None] * V
          V1 = D1V * d105
          V2 = D2V * d205
          Q = kron(C, V1) + kron(tf.eye(ns, dtype=dtype), V2)
          LQ = tfla.cholesky(Q)
          iLQ_TxI = tfla.triangular_solve(LQ, kron(T, tf.eye(nc, dtype=dtype)))
          iSigmaGamma = iUGamma + tf.matmul(iLQ_TxI, iLQ_TxI, transpose_a=True)
          beta = tf.reshape(tfla.matrix_transpose(Beta), [ns*nc,1])
          mg02 = tf.reshape(tf.matmul(T, tf.reshape(tfla.cholesky_solve(LQ, beta), [ns,nc]), transpose_a=True), [nt*nc,1])
          mg02_1, iSigmaGamma_1 = mg02, iSigmaGamma #TODO remove
        else:
          print("vector rho, phyloFast updateGammaV part Gamma") #TODO remove
          TxI_arr = tf.reshape(kron(T, tf.eye(nc, dtype=dtype)), [ns, nc, nt*nc])
          TxI_iQ_TxI, _ = pfBilinearDet(phyloTreeList, TxI_arr, TxI_arr, phyloTreeRoot, iV, rhoVec, dtype)
          iSigmaGamma = iUGamma + TxI_iQ_TxI
          beta_iQ_T, _ = pfBilinearDet(phyloTreeList, tf.transpose(Beta)[:,:,None], TxI_arr, phyloTreeRoot, iV, rhoVec, dtype)
          mg02 = tf.reshape(tf.transpose(beta_iQ_T), [nt*nc,1])
          mg02_2, iSigmaGamma_2 = mg02, iSigmaGamma #TODO remove
          
    mg0 = tf.matmul(iUGamma, mGamma[:,None]) + mg02 # is order of mGamma/iUGamma (traits-cov) and mg02 (traits-cov) correct? Seems yes.
    L = tfla.cholesky(iSigmaGamma)
    mg1 = tfla.triangular_solve(L, mg0)
    if updateGamma:
      Gamma = tf.transpose(tf.reshape(tfla.triangular_solve(L, mg1 + tfr.normal([nt*nc,1], dtype=dtype), adjoint=True), [nt,nc]))
    return Gamma, iV
