import numpy as np
import tensorflow as tf
from hmsc.utils.tf_named_func import tf_named_func
from hmsc.utils.phylo_fast_utils import phyloFastSampleBatched as pfSample
tfm, tfla, tfr, tfs = tf.math, tf.linalg, tf.random, tf.sparse

@tf_named_func("betaLambda")
def updateBetaLambda(params, data, priorHyperparams, phyloFastBatched=True, sdMult=1, dtype=np.float64):
    """Update conditional updater(s):
    Beta - species niches, and
    Lambda - species loadings.

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s).
            Z - latent variables
            iD - inverse of residual variance in observation model 
            Gamma - influence of traits on species niches
            iV - inverse residual covariance of species niches
            Eta - latent factors
            Psi - local shrinage species loadings (lambda's prior)
            Delta - delta global shrinage species loadings (lambda's prior)
            X - environmental data
            T - species trait data
            Pi - study design matrix
            rhopw - rho prior grid values and weigths
    """
    Z = params["Z"]
    iD = params["iD"]
    Gamma = params["Gamma"]
    iV = params["iV"]
    rhoInd = params["rhoInd"]
    EtaList = params["Eta"]
    PsiList = params["Psi"]
    DeltaList = params["Delta"]
    X = params["Xeff"]  
    Loff = data["Loff"]
    T = data["T"]
    phyloFlag, phyloFast = data["phyloFlag"], data["phyloFast"]
    phyloTreeList, phyloTreeRoot = data["phyloTreeList"], data["phyloTreeRoot"]
    C, eC, VC = data["C"], data["eC"], data["VC"]
    covRhoGroup = data["covRhoGroup"]
    Pi = data["Pi"]
    rhopw = priorHyperparams["rhopw"]
    ny, nc = X.shape[-2:]
    _, ns = Z.shape
    nr = len(EtaList)
    nfVec = tf.stack([tf.shape(Eta)[-1] for Eta in EtaList])
    nfSum = tf.cast(tf.reduce_sum(nfVec), tf.int32)
    na = nc + nfSum

    S = Z if Loff is None else Z - Loff
    PiEtaList = [None] * nr
    for r, Eta in enumerate(EtaList):
      PiEtaList[r] = tf.gather(Eta, Pi[:,r])
    GammaT = tf.matmul(Gamma, T, transpose_b=True)
    # print(["rank", X.shape, len(X.shape.as_list()), tf.rank(X)])
    if len(X.shape.as_list()) == 2:
      # tf.print(tf.rank(X))
      S -= tf.matmul(X, GammaT)
      XE = tf.concat([X] + PiEtaList, axis=-1)
    else:
      S -= tf.einsum("jik,kj->ik", X, GammaT)
      if nr == 0:
        XE = X
      else:
        XE = tf.concat([X, tf.repeat(tf.expand_dims(tf.concat(PiEtaList, axis=-1), 0), ns, 0)], axis=-1)
    if nr > 0:
      LambdaPriorPrec = tf.concat([Psi * tfm.cumprod(Delta, -2) for Psi, Delta in zip(PsiList, DeltaList)], axis=-2)  
    else:
      LambdaPriorPrec = tf.zeros([0,ns], dtype)
    
    # phyloFlag = False #TODO remove!!!
    if phyloFlag == False:
      if nr > 0:
        iK11_op = tfla.LinearOperatorFullMatrix(iV)
        iK22_op = tfla.LinearOperatorDiag(tf.transpose(LambdaPriorPrec))
        iK = tfla.LinearOperatorBlockDiag([iK11_op, iK22_op]).to_dense()
      else:
        iK = iV
 
      # iD05_XE = tf.multiply(tfla.matrix_transpose(iD)[:,:,None]**0.5, XE, name="iD05_XE")
      # iU = iK + tf.matmul(iD05_XE, iD05_XE, transpose_a=True, name="iU.2")
      if len(XE.shape.as_list()) == 2:
        XE_iD_XET = tf.einsum("ic,ij,ik->jck", XE, iD, XE, name="XE_iD_XET")
        M0 = tf.einsum("ik,ij->jk", XE, iD*S, name="M0")[:,:,None]
      else:
        XE_iD_XET = tf.einsum("jic,ij,jik->jck", XE, iD, XE, name="XE_iD_XET")
        M0 = tf.einsum("jik,ij->jk", XE, iD*S, name="M0")[:,:,None]
      iU = iK + XE_iD_XET
      LiU = tfla.cholesky(iU, name="LiU")
      M = tfla.cholesky_solve(LiU, M0, name="M")
      BetaLambda = tf.transpose(tf.squeeze(M + tfla.triangular_solve(LiU, sdMult*tfr.normal([ns,na,1], dtype=dtype, name="BetaLambda.2"), adjoint=True), -1))
    
    else:
      rhoVec = tf.gather(rhopw[:,0], tf.gather(rhoInd, covRhoGroup))
      if len(XE.shape.as_list()) == 2:
        XE_iD_XET = tf.einsum("ic,ij,ik->ckj", XE, iD, XE, name="XE_iD_XET")
        M0 = tf.matmul(XE, iD*S, transpose_a=True)
      else:
        XE_iD_XET = tf.einsum("jic,ij,jik->ckj", XE, iD, XE, name="XE_iD_XET")
        M0 = tf.einsum("jik,ij->kj", XE, iD*S, name="M0")
        
      if phyloFast == False:
        eQ = rhoVec[:,None]*eC + (1-rhoVec)[:,None]
        eiQ05 = tfm.rsqrt(eQ)
        eiQ_block = tf.expand_dims(eiQ05, 0) * tf.expand_dims(eiQ05, 1)
        PB_stack = tf.einsum("ij,ckj,gj->cikg", VC, eiQ_block*iV[:,:,None], VC, name="PB_stack")
        PB = tf.reshape(PB_stack, [nc*ns,nc*ns])
        iK11_op = tfla.LinearOperatorFullMatrix(PB)
        if nr > 0:
          iK22_op = tfla.LinearOperatorDiag(tf.reshape(LambdaPriorPrec, [nfSum*ns]))
          iK = tfla.LinearOperatorBlockDiag([iK11_op, iK22_op]).to_dense()
        else:
          iK = iK11_op.to_dense()
        iU = iK + tf.reshape(tf.transpose(tfla.diag(XE_iD_XET), [0,2,1,3]), [na*ns]*2)
        LiU = tfla.cholesky(iU)        
        m = tfla.cholesky_solve(LiU, tf.reshape(M0, [na*ns,1]))
        BetaLambda = tf.reshape(m + tfla.triangular_solve(LiU, sdMult*tfr.normal(shape=[na*ns,1], dtype=dtype), adjoint=True), [na,ns])
      else:
        print("phyloFast updateBetaLambda") #TODO remove print
        # BetaLambda = tfr.normal(shape=[na,ns], dtype=dtype)
        lin_op2 = tfla.LinearOperatorDiag(tf.ones([nfSum],dtype))
        iV_e = tfla.LinearOperatorBlockDiag([tfla.LinearOperatorFullMatrix(iV), lin_op2]).to_dense()
        V_e = tfla.LinearOperatorBlockDiag([tfla.LinearOperatorFullMatrix(tfla.inv(iV)), lin_op2]).to_dense()
        rhoVec_e = tf.concat([rhoVec, tf.zeros([nfSum],dtype)], 0)
        rho2Mat_e = tf.concat([tf.tile((1-rhoVec)[:,None], [1,ns]), LambdaPriorPrec**-1], 0)
        BetaLambda = pfSample(phyloTreeList, phyloTreeRoot, V_e, iV_e, rhoVec_e, rho2Mat_e, XE_iD_XET, M0, sdMult=sdMult, dtype=dtype)
        
    if nr > 0:
      BetaLambdaList = tf.split(BetaLambda, tf.concat([tf.constant([nc], tf.int32), nfVec], -1), axis=-2)
      BetaNew = tf.ensure_shape(GammaT + BetaLambdaList[0], [nc,ns])
      LambdaListNew = BetaLambdaList[1:]
    else:
      BetaNew, LambdaListNew = GammaT+BetaLambda, []

    # tf.print(BetaNew)
    return BetaNew, LambdaListNew