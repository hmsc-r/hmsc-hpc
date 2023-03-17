import numpy as np
import tensorflow as tf
tfm, tfla, tfr, tfs = tf.math, tf.linalg, tf.random, tf.sparse


def updateBetaLambda(params, data, priorHyperparams, dtype=np.float64):
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
    rhopw = priorHyperparams["rhopw"]

    if isinstance(X, list):
      ny, nc = X[0].shape
    else:
      ny, nc = X.shape
    
    _, ns = Z.shape
    nr = len(EtaList)
    nfVec = tf.stack([tf.shape(Eta)[-1] for Eta in EtaList])
    nfSum = tf.cast(tf.reduce_sum(nfVec), tf.int32)
    na = nc + nfSum
    
    iD = tf.ones_like(Z) * sigma**-2


    PiEtaList = [None] * nr
    for r, Eta in enumerate(EtaList):
      PiEtaList[r] = tf.gather(Eta, Pi[:, r])

    if isinstance(X, list):
      # XE = tf.stack([tf.concat([X1] + PiEtaList, axis=-1) for X1 in X])
      if nr == 0:
        XE = tf.stack(X)
      else:
        XE = tf.concat([tf.stack(X), tf.tile(tf.expand_dims(tf.concat(PiEtaList, axis=-1), 0), [ns,1,1])], axis=-1)
    else:
      XE = tf.concat([X] + PiEtaList, axis=-1)

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
      BetaLambda = tf.transpose(tf.squeeze(M + tfla.triangular_solve(LiU, tfr.normal(shape=[ns,na,1], dtype=dtype), adjoint=True), -1))
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
      
      if isinstance(X, list):	
        # XE_iD_XET = tf.reshape(tf.einsum("jic,j,jik->jck", XE, sigma**-2, XE), shape=[n2,n1])
        # ind1 = tf.concat([tf.repeat(tf.range(0,n2,2,dtype=tf.int64),n1),tf.repeat(tf.range(1,n2,2,dtype=tf.int64),n1)], axis=0)
        # ind2 = tf.concat([tf.tile(tf.range(0,n2,2,dtype=tf.int64),[n1]), tf.tile(tf.range(1,n2,2,dtype=tf.int64),[n1])], axis=0)
        # iU = iK + tfs.to_dense(tfs.reorder(tfs.SparseTensor(indices=tf.transpose(tf.stack([ind1,ind2])), values=tf.reshape(XE_iD_XET, [-1]), dense_shape=[n2,n2])))
        XE_iD_XET = tf.einsum("jic,j,jik->ckj", XE, sigma**-2, XE)
        m0 = tf.matmul(iK, tf.reshape(Mu, [na*ns,1])) + tf.reshape(tf.einsum("jik,ij->kj", XE, iD * Z), [na*ns,1])
      else:	
        XE_iD_XET = tf.einsum("ic,j,ik->ckj", XE, sigma**-2, XE)
        m0 = tf.matmul(iK, tf.reshape(Mu, [na*ns,1])) + tf.reshape(tf.matmul(XE, iD * Z, transpose_a=True), [na*ns,1])
      
      iU = iK + tf.reshape(tf.transpose(tfla.diag(XE_iD_XET), [0,2,1,3]), [na*ns]*2)
      LiU = tfla.cholesky(iU)        
      m = tfla.cholesky_solve(LiU, m0)
      BetaLambda = tf.reshape(m + tfla.triangular_solve(LiU, tfr.normal(shape=[na*ns,1], dtype=dtype), adjoint=True), [na,ns])
    
    if nr > 0:
      BetaLambdaList = tf.split(BetaLambda, tf.concat([tf.constant([nc], tf.int32), nfVec], -1), axis=-2)
      BetaNew, LambdaListNew = tf.ensure_shape(BetaLambdaList[0], [nc,ns]), BetaLambdaList[1:]
    else:
      BetaNew, LambdaListNew = BetaLambda, []

    return BetaNew, LambdaListNew
