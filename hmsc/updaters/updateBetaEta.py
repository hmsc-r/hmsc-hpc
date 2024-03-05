import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hmsc.utils.tfla_utils import kron
from hmsc.utils.tf_named_func import tf_named_func
tfla, tfm, tfr = tf.linalg, tf.math, tf.random
tfd = tfp.distributions

@tf_named_func("BetaEta")
def updateBetaEta(params, modelDims, data, priorHyperparams, rLHyperparams, dtype=np.float64):
  Z = params["Z"]
  iD = params["iD"]
  Beta = params["Beta"]
  Gamma = params["Gamma"]
  iV = params["iV"]
  rhoInd = params["rhoInd"]
  LambdaList = params["Lambda"]
  EtaList = params["Eta"]
  AlphaIndList = params["AlphaInd"]
  X = params["Xeff"]
  T = data["T"]
  Pi = data["Pi"]
  rhoGroup = data["rhoGroup"]
  Pi = data["Pi"]
  C, eC, VC = data["C"], data["eC"], data["VC"]
  rhopw = priorHyperparams["rhopw"]
  ny = modelDims["ny"]
  ns = modelDims["ns"]
  nc = modelDims["nc"]
  nr = modelDims["nr"]
  npVec = modelDims["np"]
  GammaT = tf.matmul(Gamma, T, transpose_b=True)

  # C, eC, VC = tf.eye(ns,dtype=dtype), tf.ones([ns],dtype), tf.eye(ns,dtype=dtype)


  LRanLevelList = [None] * nr
  for r, (Eta, Lambda) in enumerate(zip(EtaList, LambdaList)):
    LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:,r]), Lambda)

  EtaListNew = [None] * nr
  for r, (Eta, Lambda, AlphaInd, rLPar) in enumerate(zip(EtaList, LambdaList, AlphaIndList, rLHyperparams)):
    randFlag = tf.cast(1, dtype)
    nf = tf.cast(tf.shape(Lambda)[-2], tf.int64)
    # if nf > 0:
    nv = npVec[r]
    S = Z - sum([LRanLevelList[rInd] for rInd in np.setdiff1d(np.arange(nr), r)])
    iD_S = iD*S
    pi = Pi[:,r]
    # if rLPar["sDim"] == 0:
    #   raise Exception('not implemented') # block-inverse calculation
    # else:
    # if rLPar["spatialMethod"] == "Full":
    
    if len(X.shape.as_list()) == 2:
      mu012 = tf.reshape(tf.einsum("ik,ij->jk", X, iD_S, name="mu01.2"), [ns*nc])
      iD_X = tf.einsum("ij,ic->ijc", iD, X)
      X_iD_X = tf.einsum("ic,ij,ik->ckj", X, iD, X)
    else:
      mu012 = tf.reshape(tf.einsum("jik,ij->jk", X, iD_S, name="mu01.2"), [ns*nc])
      iD_X = tf.einsum("ij,jic->ijc", iD, X)
      X_iD_X = tf.einsum("jic,ij,jik->ckj", X, iD, X)
    mu02 = tf.reshape(tfla.matrix_transpose(tf.scatter_nd(pi[:,None], tf.einsum("hj,ij->ih", Lambda, iD_S, name="mu02"), [nv,nf])), [nf*nv])    
    if C is None:
      mu01 = tf.reshape(tfla.matrix_transpose(tf.matmul(iV, GammaT, name="mu01.1")), [ns*nc]) + mu012
    else:
      rhoVec = tf.gather(rhopw[:,0], tf.gather(rhoInd, rhoGroup))
      eQ = rhoVec[:,None]*eC + (1-rhoVec)[:,None]
      eiQ05 = tfm.rsqrt(eQ)
      eiQ_block = tf.expand_dims(eiQ05, 0) * tf.expand_dims(eiQ05, 1)
      PB_stack = tf.einsum("ij,ckj,gj->icgk", VC, eiQ_block*iV[:,:,None], VC, name="PB_stack")
      iK = tf.reshape(PB_stack, [ns*nc]*2)
      iS11 = iK + tf.reshape(tf.transpose(tfla.diag(X_iD_X), [2,0,3,1]), [ns*nc]*2)
      mu01 = tf.reshape(tf.matmul(iK, tf.reshape(tf.transpose(GammaT), [ns*nc,1]), name="mu01.1"), [ns*nc]) + mu012
      
    Pi_iD_X = tf.scatter_nd(pi[:,None], iD_X, shape=[nv,ns,nc])
    Lam_iD_Lam = tf.einsum("hj,ij,fj->ihf", Lambda, iD, Lambda, name="Lam_iD_Lam") # reorder pi multiplication from next line if slow?
    
    if rLPar["sDim"] == 0:
      if C is None:
        print("WARNING - updateBetaEta() may be computationally unjustified for models without phylogeny and non-spatial random level")
        iS11 = tf.reshape(tf.transpose(tfla.diag(iV[:,:,None] + X_iD_X), [2,0,3,1]), [ns*nc]*2)
      
      Pi_Lam_iD_Lam = tf.scatter_nd(pi[:,None], Lam_iD_Lam, shape=[nv,nf,nf])
      A = tf.eye(nf,dtype=dtype) + Pi_Lam_iD_Lam
      LA = tfla.cholesky(A)
      iA = tfla.cholesky_solve(LA, tf.eye(nf,dtype=dtype))
      Lambda_iA_Lambdat = tf.einsum("hj,phg,gl->pjl", Lambda, iA, Lambda, name="Lambda_iA_Lambdat")
      U_2 = tf.reshape(tf.einsum("pjc,pjl,plk->jclk", Pi_iD_X, Lambda_iA_Lambdat, Pi_iD_X, name="U_2"), [ns*nc]*2)
      U = iS11 - U_2
      LU = tfla.cholesky(U)
      
      iLA_mu02 = tfla.triangular_solve(LA, tf.transpose(tf.reshape(mu02,[nf,nv,1]),[1,0,2]))
      iA_mu02 = tf.transpose(tf.squeeze(tfla.triangular_solve(LA, iLA_mu02, adjoint=True), -1))
      Xt_iD_LambdatP_mu02 = tf.reshape(tf.matmul(iD * tf.gather(tf.matmul(iA_mu02, Lambda, transpose_a=True), pi), X, transpose_a=True), [ns*nc])
      mu1 = mu01 - Xt_iD_LambdatP_mu02
      iLU_mu1 = tfla.triangular_solve(LU, mu1[:,None])
      q1 = tf.squeeze(tfla.triangular_solve(LU, iLU_mu1 + randFlag*tfr.normal([ns*nc,1],dtype=dtype), adjoint=True), -1)
      Pt_iD_X_q1 = tf.scatter_nd(pi[:,None], iD * tf.matmul(X, tf.reshape(q1,[ns,nc]), transpose_b=True), [nv,ns])
      LambdaPt_iD_X_q1 = tf.matmul(Lambda, Pt_iD_X_q1, transpose_b=True)
      iA_LambdaPt_iD_X_q1 = tf.reshape(tf.transpose(tfla.cholesky_solve(LA, tf.transpose(LambdaPt_iD_X_q1)[:,:,None]), [1,0,2]), [nf*nv])
      q21 = tf.reshape(tf.transpose(tfla.triangular_solve(LA, iLA_mu02 + randFlag*tfr.normal([nv,nf,1],dtype=dtype), adjoint=True),[1,0,2]), [nf*nv])
      q2 = q21 - iA_LambdaPt_iD_X_q1
      Beta = tf.transpose(tf.reshape(q1, [ns,nc]))
      Eta = tf.transpose(tf.reshape(q2, [nf,nv]))
    
    else:
      iS22_2 = tf.reshape(tf.transpose(tfla.diag(tf.transpose(tf.scatter_nd(pi[:,None], Lam_iD_Lam, shape=[nv,nf,nf]), [1,2,0])), [0,2,1,3]), [nf*nv]*2)
      if rLPar["spatialMethod"] == "Full":
        iWg = rLPar["iWg"]
        iWs = tf.reshape(tf.transpose(tfla.diag(tf.transpose(tf.gather(iWg, AlphaInd), [1,2,0])), [2,0,3,1]), [nf*nv]*2)
        iS22 = iWs + iS22_2
      
        if C is None: # block-inverse calculation
          if len(X.shape.as_list()) == 2:
            B = iV + tf.einsum("ic,ij,ik->jck", X, iD, X)
          else:
            B = iV + tf.einsum("jic,ij,jik->jck", X, iD, X)
          LB = tfla.cholesky(B)
          iB = tfla.cholesky_solve(LB, tf.eye(nc,dtype=dtype))
          tmp1 = tf.einsum("pjc,jck,vjk->jpv", Pi_iD_X, iB, Pi_iD_X)
          W_2 = tf.reshape(tf.einsum("hj,jpv,gj->hpgv", Lambda, tmp1, Lambda), [nf*nv]*2)
          W = iS22 - W_2
          LW = tfla.cholesky(W)
          
          iB_mu01 = tf.squeeze(tfla.cholesky_solve(LB, tf.reshape(mu01,[ns,nc,1])), -1)
          if len(X.shape.as_list()) == 2:
            X_iB_mu01 = tf.matmul(X, iB_mu01, transpose_b=True)
          else:
            X_iB_mu01 = tf.einsum("jik,jk->ij", X, iB_mu01)
          Pt_iD_X_iB_mu01 = tf.scatter_nd(pi[:,None], iD * X_iB_mu01, [nv,ns])
          LambdaPt_iD_X_iB_mu01 = tf.reshape(tf.matmul(Lambda, Pt_iD_X_iB_mu01, transpose_b=True), [nf*nv])
          mu1 = mu02 - LambdaPt_iD_X_iB_mu01
          iLW_mu1 = tfla.triangular_solve(LW, mu1[:,None])
          q2 = tf.squeeze(tfla.triangular_solve(LW, iLW_mu1 + randFlag*tfr.normal([nf*nv,1],dtype=dtype), adjoint=True), -1)
          iD_LambdatP_q2 = iD * tf.gather(tf.matmul(tf.reshape(q2,[nf,nv]), Lambda, transpose_a=True), pi)
          if len(X.shape.as_list()) == 2:
            X_iD_LambdatP_q2 = tf.matmul(iD_LambdatP_q2, X, transpose_a=True)
          else:
            X_iD_LambdatP_q2 = tf.einsum("ij,jik->jk", iD_LambdatP_q2, X)
          iB_X_iD_LambdatP_q2 = tf.reshape(tfla.cholesky_solve(LB, X_iD_LambdatP_q2[:,:,None]), [ns*nc])
          q11 = tf.reshape(tfla.triangular_solve(LB, tfla.triangular_solve(LB, tf.reshape(mu01,[ns,nc,1])) + randFlag*tfr.normal([ns,nc,1],dtype=dtype), adjoint=True), [ns*nc])
          q1 = q11 - iB_X_iD_LambdatP_q2
          Beta = tf.transpose(tf.reshape(q1, [ns,nc]))
          Eta = tf.transpose(tf.reshape(q2, [nf,nv]))
          
        else: # explicit calculation - also marginalize Gamma?
          iS21 = tf.reshape(tf.einsum("hj,pjc->hpjc", Lambda, Pi_iD_X), [nf*nv,ns*nc])
          iS1 = tf.concat([iS11, tfla.matrix_transpose(iS21)], -1)
          iS2 = tf.concat([iS21, iS22], -1)
          iS = tf.concat([iS1, iS2], -2)
          LiS = tfla.cholesky(iS)
          
          mu0 = tf.concat([mu01,mu02], 0)
          mu1 = tfla.triangular_solve(LiS, mu0[:,None], name="mu1")
          v = tf.squeeze(tfla.triangular_solve(LiS, mu1 + randFlag*tfr.normal([ns*nc+nf*nv,1],dtype=dtype), adjoint=True, name="v"), -1)
          Beta = tf.transpose(tf.reshape(v[:ns*nc], [ns,nc]))
          Eta = tf.transpose(tf.reshape(v[ns*nc:], [nf,nv]))
    
      # tf.print(tf.reduce_max(tfm.abs(tf.concat([q1,q2], 0) - v)))
    
    EtaListNew[r] = Eta
    LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:,r]), Lambda)
    # else:
    #   EtaListNew[r] = Eta
    EtaListNew[r] = tf.ensure_shape(EtaListNew[r], [npVec[r],None])

  return Beta, EtaListNew


# from matplotlib import pyplot as plt 
# from scipy.linalg import block_diag
# import time
# import math
# import os

# def implot(data):
#     limit = max(abs(np.min(data)), abs(np.max(data)) )

#     plt.imshow(data, aspect='equal', cmap='RdBu',
#               vmin=-limit, vmax=limit)
#     plt.colorbar()
#     plt.show()

# def kron(A, B):
#     return tf.reshape(tf.einsum("ab,cd->acbd", A, B), shape = [tf.shape(A)[0] * tf.shape(B)[0], tf.shape(A)[1] * tf.shape(B)[1]]) 

# ny = 999
# ns = 133
# nc = 22
# nt = 2
# nf = 5
# nv = math.floor(ny/2)
# dtype = np.float64
# tf.keras.utils.set_random_seed(42)

# Tr = np.random.normal(0, size=[ns,nt]).astype(dtype); Tr[:,0] = 1
# X = np.random.normal(0, 0.2, size=[ny,nc]).astype(dtype); X[:,0] = 1
# Gamma = tfd.Normal(dtype(0), 1).sample([nc,nt])
# VBlockList = [np.array([[1,1],[1,1]])] * int(nc/2)
# if nc % 2: VBlockList = VBlockList + [np.array([[1]])]
# V = (block_diag(*VBlockList).astype(dtype) + np.eye(nc))
# multV = tf.linspace(1/nc, 1, nc)
# V = tf.cast(tf.sqrt(multV)[:,None] * V * tf.sqrt(multV), dtype)
# # V = tf.eye(nc, dtype=dtype)
# Mu = tf.matmul(Gamma, Tr, transpose_b=True)
# Beta = tf.transpose(tfd.MultivariateNormalFullCovariance(tf.transpose(Mu), V).sample())
# LFix = tf.matmul(X, Beta)
# nu, a1, b1, a2, b2 = 3, 1.2, 1, 2, 1
# rLHp = {"nu": nu, "a1": 5, "b1": 1, "a2": 5, "b2": 1, "sDim": 0}
# aDelta = tf.concat([a1 * tf.ones([1, 1], dtype), a2 * tf.ones([nf-1, 1], dtype)], 0)
# bDelta = tf.concat([b1 * tf.ones([1, 1], dtype), b2 * tf.ones([nf-1, 1], dtype)], 0)
# Delta = tfd.Gamma(aDelta, bDelta).sample()
# Tau = tfm.cumprod(Delta, 0)
# Lambda = tf.transpose(tf.squeeze(tfd.StudentT(nu, 0, tfm.rsqrt(Tau)).sample([ns]),-1))
# # tmp = tfd.StudentT(nu, 0, tfm.rsqrt(Tau)).log_prob(Lambda)
# Eta = tfd.Normal(tf.cast(0,dtype), 1).sample([nv,nf])
# Pi = (np.arange(ny) % nv)[:,None]
# np.random.shuffle(Pi)
# LRan = tf.gather(tf.matmul(Eta, Lambda), Pi[:,0])
# rLHyperparams = [rLHp]
# L = LFix + LRan
# sigma = np.random.uniform(size=[ns]).astype(dtype)
# iD = tf.tile(sigma[None,:]**-2, [ny,1])
# pr = tfd.Normal(L,sigma).survival_function(0)
# Y = tfd.Bernoulli(probs=pr).sample()
# Omega = tf.matmul(Lambda,Lambda,transpose_a=True)
# iV = tfla.inv(V)
# x = tfr.normal([ns*nc+nf*nv], dtype=dtype)

# # block1
# timeStart = time.time()
# iS11 = kron(tf.eye(ns,dtype=dtype), iV) + tf.reshape(tf.transpose(tfla.diag(tf.einsum("ic,ij,ik->ckj", X, iD, X)), [2,0,3,1]), [ns*nc]*2)
# iD_X = tf.einsum("ij,ic->ijc", iD, X)
# Pi_iD_X = tf.scatter_nd(Pi[:,0:1], iD_X, shape=[nv,ns,nc])
# iS21 = tf.reshape(tf.einsum("hj,pjc->hpjc", Lambda, Pi_iD_X), [nf*nv,ns*nc])
# # Pit_iD = tf.scatter_nd(Pi[:,0:1], iD, shape=[nv,ns])
# iS22_2 = tf.reshape(tf.transpose(tfla.diag(tf.transpose(tf.scatter_nd(Pi[:,0:1], tf.einsum("hj,ij,fj->ihf",Lambda,iD,Lambda), shape=[nv,nf,nf]), [1,2,0])), [0,2,1,3]), [nf*nv]*2)
# iS22 = kron(tf.eye(nf,dtype=dtype), tf.eye(nv,dtype=dtype)) + iS22_2

# iS1 = tf.concat([iS11, tfla.matrix_transpose(iS21)], -1)
# iS2 = tf.concat([iS21, iS22], -1)
# iS = tf.concat([iS1, iS2], -2)
# LiS = tfla.cholesky(iS)
# Sig_x = tf.squeeze(tfla.cholesky_solve(LiS, x[:,None]), -1)
# time1 = time.time() - timeStart

# # block2
# timeStart = time.time()
# B = iV + tf.einsum("ic,ij,ik->jck", X, iD, X)
# LB = tfla.cholesky(B)
# iB = tfla.cholesky_solve(LB, tf.eye(nc,dtype=dtype))
# tmp1 = tf.einsum("pjc,jck,vjk->jpv", Pi_iD_X, iB, Pi_iD_X)
# iSig22_2 = tf.reshape(tf.einsum("hj,jpv,gj->hpgv", Lambda, tmp1, Lambda), [nf*nv]*2)
# iSig22 = iS22 - iSig22_2
# W = iSig22
# LW = tfla.cholesky(W)

# x1 = x[:ns*nc]
# x2 = x[ns*nc:]
# iB_x1 = tf.squeeze(tfla.cholesky_solve(LB, tf.reshape(x1,[ns,nc,1])), -1)
# LambdaPt_iD_X_iB_x1 = tf.reshape(tf.matmul(Lambda, tf.scatter_nd(Pi[:,0:1], iD * tf.matmul(X, iB_x1, transpose_b=True), [nv,ns]), transpose_b=True), [nf*nv])
# z = x2 - LambdaPt_iD_X_iB_x1
# iW_z = tfla.cholesky_solve(LW, z[:,None])
# q2 = tf.squeeze(iW_z, -1)
# tmp2 = tf.matmul(tf.transpose(iD) * tf.gather(tf.matmul(Lambda, tf.reshape(iW_z,[nf,nv]), transpose_a=True), Pi[:,0], axis=1), X)
# Lambdat_iW_z = tf.reshape(tfla.cholesky_solve(LB, tmp2[:,:,None]), [ns*nc])
# q1 = tf.reshape(tfla.cholesky_solve(LB, tf.reshape(x1,[ns,nc,1])), [ns*nc]) - Lambdat_iW_z
# q = tf.concat([q1,q2], 0)
# time2 = time.time() - timeStart


# print(tf.reduce_max(tfm.abs(Sig_x - q)))
# print("%.3f,%.3f" % (time1, time2))


# Sig = tfla.cholesky_solve(LiS, tf.eye(ns*nc+nf*nv,dtype=dtype))
# Sig22 = tfla.inv(iSig22)
# implot(Sig)
# print(tf.reduce_max(tfm.abs(Sig[ns*nc:,ns*nc:] - Sig22)))
