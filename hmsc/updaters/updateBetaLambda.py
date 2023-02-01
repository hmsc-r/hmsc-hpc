import numpy as np
import tensorflow as tf

tfm, tfla, tfr = tf.math, tf.linalg, tf.random


def updateBetaLambda(params, data, dtype=np.float64):
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
    EtaList = params["Eta"]
    PsiList = params["Psi"]
    DeltaList = params["Delta"]
    sigma = params["sigma"]
    X = data["X"]
    T = data["T"]
    Pi = data["Pi"]
    

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
    iK11_op = tfla.LinearOperatorFullMatrix(iV)
    if nr > 0:
      LambdaPriorPrec = tf.concat([Psi * tfm.cumprod(Delta, -2) for Psi, Delta in zip(PsiList, DeltaList)], axis=-2)
      iK22_op = tfla.LinearOperatorDiag(tf.transpose(LambdaPriorPrec))
      iK = tfla.LinearOperatorBlockDiag([iK11_op, iK22_op]).to_dense()
    else:
      iK = iK11_op.to_dense()
    
    iU = iK + tf.matmul(XE, XE, transpose_a=True) / (sigma**2)[:, None, None]
    LiU = tfla.cholesky(iU)
    A = tf.matmul(iK, tf.transpose(Mu)[:,:,None]) + (tf.matmul(Z, XE, transpose_a=True) / (sigma**2)[:,None])[:,:,None]
    M = tfla.cholesky_solve(LiU, A)
    BetaLambda = tf.transpose(tf.squeeze(M + tfla.triangular_solve(LiU, tfr.normal(shape=[ns,nc+nfSum,1], dtype=dtype), adjoint=True), -1))
    
    if nr > 0:
      BetaLambdaList = tf.split(BetaLambda, tf.concat([tf.constant([nc], tf.int32), nfVec], -1), axis=-2)
      BetaNew, LambdaListNew = tf.ensure_shape(BetaLambdaList[0], [nc,ns]), BetaLambdaList[1:]
    else:
      BetaNew, LambdaListNew = BetaLambda, []

    return BetaNew, LambdaListNew
