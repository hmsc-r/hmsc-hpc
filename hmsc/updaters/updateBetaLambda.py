import numpy as np
import tensorflow as tf

tfm, tfla, tfr = tf.math, tf.linalg, tf.random


def updateBetaLambda(params, data, priorHyperParams, dtype=np.float64):
    """Update conditional updater(s):
    Beta - species niches, and
    Lambda - species loadings.

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s).
            Z - latent variables
            Gamma - influence of traits on species niches
            iV - inverse residual covariance of species niches
            Eta - site loadings
            Psi - local shrinage species loadings (lambda's prior)
            Delta - delta global shrinage species loadings (lambda's prior)
            sigma - residual variance
            X - environmental data
            T - species trait data
            Pi - study design
    """

    Z = params["Z"]
    Gamma = params["Gamma"]
    iV = tfla.inv(params["V"])
    rhoInd = params["rhoInd"]
    EtaList = params["Eta"]
    PsiList = params["Psi"]
    DeltaList = params["Delta"]
    sigma = params["sigma"]
    X = data["X"]
    T = data["T"]
    C, eC, VC = data["C"], data["eC"], data["VC"]
    rhoGroup = data["rhoGroup"]
    Pi = data["Pi"]
    rhopw = priorHyperParams["rhopw"]
    

    ny, nc = X.shape
    _, ns = Z.shape
    nr = len(EtaList)
    nfVec = tf.stack([tf.shape(Eta)[-1] for Eta in EtaList])
    nfSum = tf.cast(tf.reduce_sum(nfVec), tf.int32)

    EtaListFull = [None] * nr
    for r, Eta in enumerate(EtaList):
      EtaListFull[r] = tf.gather(Eta, Pi[:, r])

    XE = tf.concat([X] + EtaListFull, axis=-1)
    GammaT = tf.matmul(Gamma, T, transpose_b=True)
    Mu = tf.concat([GammaT, tf.zeros([nfSum, ns], dtype)], axis=0)
    if nr > 0:
      LambdaPriorPrec = tf.concat([Psi * tfm.cumprod(Delta, -2) for Psi, Delta in zip(PsiList, DeltaList)], axis=-2)
      
    if C is None:
      iK11_op = tfla.LinearOperatorFullMatrix(iV)
      if nr > 0:
        iK22_op = tfla.LinearOperatorDiag(tf.transpose(LambdaPriorPrec))
        iK = tfla.LinearOperatorBlockDiag([iK11_op, iK22_op]).to_dense()
      else:
        iK = iK11_op.to_dense()
      
      iU = iK + tf.matmul(XE, XE, transpose_a=True) / (sigma**2)[:, None, None]
      LiU = tfla.cholesky(iU)
      A = tf.matmul(iK, tf.transpose(Mu)[:,:,None]) + (tf.matmul(Z / sigma**2, XE, transpose_a=True))[:,:,None]
      M = tfla.cholesky_solve(LiU, A)
      BetaLambda = tf.transpose(tf.squeeze(M + tfla.triangular_solve(LiU, tfr.normal(shape=[ns,nc+nfSum,1], dtype=dtype), adjoint=True), -1))
      # tf.transpose(tf.squeeze(M))
    else:
      rhoVec = tf.gather(rhopw[:,0], tf.gather(rhoInd, rhoGroup))
      eQ = rhoVec[:,None]*eC + (1-rhoVec)[:,None]
      eiQ05 = tfm.rsqrt(eQ)
      eiQ_block = tf.expand_dims(eiQ05, 0) * tf.expand_dims(eiQ05, 1)
      PB_stack = tf.einsum("ij,ckj,gj->cikg", VC, eiQ_block*iV[:,:,None], VC, name="updateBetaLambda_P_stack")
      PB = tf.reshape(PB_stack, [nc*ns,nc*ns])
      iK11_op = tfla.LinearOperatorFullMatrix(PB)
      if nr > 0:
        iK22_op = tfla.LinearOperatorDiag(tf.reshape(LambdaPriorPrec, [nfSum*ns]))
        iK = tfla.LinearOperatorBlockDiag([iK11_op, iK22_op]).to_dense()
      else:
        iK = iK11_op.to_dense()
      
      XE_iD_XET = tf.einsum("ic,j,ik->ckj", XE, sigma**-2, XE)
      iU = iK + tf.reshape(tf.transpose(tfla.diag(XE_iD_XET), [0,2,1,3]), [(nc+nfSum)*ns]*2)
      LiU = tfla.cholesky(iU)
      m0 = tf.matmul(iK, tf.reshape(Mu, [(nc+nfSum)*ns,1])) + tf.reshape(tf.matmul(XE, Z / sigma**2, transpose_a=True), [(nc+nfSum)*ns,1])
      m = tfla.cholesky_solve(LiU, m0)
      BetaLambda = tf.reshape(m + tfla.triangular_solve(LiU, tfr.normal(shape=[(nc+nfSum)*ns,1], dtype=dtype), adjoint=True), [nc+nfSum,ns])
      # tf.reshape(m, [nc+nfSum,ns])
    
    if nr > 0:
      BetaLambdaList = tf.split(BetaLambda, tf.concat([tf.constant([nc], tf.int32), nfVec], -1), axis=-2)
      BetaNew, LambdaListNew = tf.ensure_shape(BetaLambdaList[0], [nc,ns]), BetaLambdaList[1:]
    else:
      BetaNew, LambdaListNew = BetaLambda, []

    return BetaNew, LambdaListNew
