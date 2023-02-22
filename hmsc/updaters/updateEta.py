import numpy as np
import tensorflow as tf

tfla, tfr, tfs = tf.linalg, tf.random, tf.sparse

from hmsc.utils.tflautils import kron, scipy_cholesky, tf_sparse_matmul, tf_sparse_cholesky, scipy_sparse_solve_triangular


def updateEta(params, data, modelDims, rLHyperparams, dtype=np.float64):
    """Update conditional updater(s):
    Z - site loadings.

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Z - latent variables
        Beta - species niches
        Eta - site loadings
        Lambda - species loadings
        Alpha - scale of site loadings (eta's prior)
        sigma - residual variance
        X - environmental data
        Pi - study design
        iWg - ??
        sDim - ??
    """
    Z = params["Z"]
    Beta = params["Beta"]
    sigma = params["sigma"]
    LambdaList = params["Lambda"]
    EtaList = params["Eta"]
    AlphaIndList = params["AlphaInd"]
    Pi = data["Pi"]
    X = data["X"]
    ny = modelDims["ny"]
    ns = modelDims["ns"]
    nr = modelDims["nr"]
    npVec = modelDims["np"]
    
    iD = tf.ones_like(Z) * sigma**-2

    iSigma = 1 / sigma
    
    if isinstance(X, list):
        for i, X1 in enumerate(X):
            XBeta = tf.matmul(X1, tf.expand_dims(Beta[:,i], -1))
            if i == 0:
                LFix = XBeta
            else:
                LFix = tf.stack([LFix, XBeta], axis=1)
        LFix = tf.squeeze(LFix, axis=-1)
    else:
        LFix = tf.matmul(X, Beta)

    LRanLevelList = [None] * nr
    for r, (Eta, Lambda) in enumerate(zip(EtaList, LambdaList)):
        LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:,r]), Lambda)

    EtaListNew = [None] * nr
    for r, (Eta, Lambda, AlphaInd, rLPar) in enumerate(zip(EtaList, LambdaList, AlphaIndList, rLHyperparams)):
        nf = tf.cast(tf.shape(Lambda)[-2], tf.int64)
        if nf > 0:
            S = Z - LFix - sum([LRanLevelList[rInd] for rInd in np.setdiff1d(np.arange(nr), r)])
            LamInvSigLam = tf.scatter_nd(Pi[:,r,None], tf.einsum("hj,ij,kj->ihk", Lambda, iD, Lambda), [npVec[r],nf,nf])
            mu0 = tf.scatter_nd(Pi[:,r,None], tf.matmul(iD * S, Lambda, transpose_b=True), [npVec[r],nf])
    
            if rLPar["sDim"] > 0:
                if rLPar["spatialMethod"] == "Full":
                    EtaListNew[r] = modelSpatialFull(LamInvSigLam, mu0, AlphaInd, rLPar["iWg"], npVec[r], nf)
                elif rLPar["spatialMethod"] == "GPP":
                    raise NotImplementedError
                elif rLPar["spatialMethod"] == "NNGP":
                    EtaListNew[r] = EtaListNew[r] = modelSpatialNNGP(LamInvSigLam, mu0, AlphaInd, rLPar["iWg"], Pi, S, iSigma, ny, ns, npVec[r], nf)
            else:
                EtaListNew[r] = modelNonSpatial(LamInvSigLam, mu0, npVec[r], nf, dtype)
            
            LRanLevelList[r] = tf.matmul(tf.gather(EtaListNew[r], Pi[:,r]), Lambda)
        else:
            EtaListNew[r] = Eta

    return EtaListNew


def modelSpatialFull(LamInvSigLam, mu0, AlphaInd, iWg, np, nf, dtype=np.float64): 
    #TODO a lot of unnecessary tanspositions - rework if considerably affects perfomance
    iWs = tf.reshape(tf.transpose(tfla.diag(tf.transpose(tf.gather(iWg, AlphaInd), [1,2,0])), [2,0,3,1]), [nf*np,nf*np])
    iUEta = iWs + tf.reshape(tf.transpose(tfla.diag(tf.transpose(LamInvSigLam, [1,2,0])), [0,2,1,3]), [nf*np,nf*np])
    LiUEta = tfla.cholesky(iUEta)
    mu1 = tfla.triangular_solve(LiUEta, tf.reshape(tf.transpose(mu0), [nf*np,1]))
    eta = tfla.triangular_solve(LiUEta, mu1 + tfr.normal([nf*np,1], dtype=dtype), adjoint=True)
    Eta = tf.transpose(tf.reshape(eta, [nf,np]))
    return Eta


def modelNonSpatial(LamInvSigLam, mu0, np, nf, dtype=np.float64):
    iV = tf.eye(nf, dtype=dtype) + LamInvSigLam
    LiV = tfla.cholesky(iV)
    mu1 = tfla.triangular_solve(LiV, tf.expand_dims(mu0, -1))
    Eta = tf.squeeze(tfla.triangular_solve(LiV, mu1 + tfr.normal([np,nf,1], dtype=dtype), adjoint=True), -1)
    return Eta


def modelSpatialNNGP(LamInvSigLam, mu0, AlphaInd, iWg, Pi, S, iSigma, ny, ns, np, nf, dtype=np.float64):
    mask_values = tf.constant([True, False])
    iWs = tfs.from_dense(tf.zeros([np * nf, np * nf], dtype=dtype))
    
    for h in range(nf):
        mask = tf.where(tf.equal(iWg.indices[:, 0], tf.cast(AlphaInd[h], tf.int64)), mask_values[0], mask_values[1])
        masked_iWg = tfs.reduce_sum(tf.sparse.retain(iWg, mask), axis=0, keepdims=True, output_is_sparse=True)
        iWs = tfs.add(iWs, tfs.from_dense(kron(tf.squeeze(tfs.to_dense(masked_iWg)), tf.linalg.diag(tf.cast(tf.one_hot(h, tf.cast(nf, tf.int32)), dtype)))))
            
    P = tfs.SparseTensor(tf.squeeze(tf.stack([tf.expand_dims(tf.range(ny, dtype=tf.int64),1), Pi], axis=1)), tf.ones([ny], dtype=dtype), [ny, np])
    fS = tf_sparse_matmul(tf_sparse_matmul(P, tfs.from_dense(S)), tfs.from_dense(tf.transpose(tf.reshape(tf.tile(iSigma, [nf]), [nf, len(iSigma)]))))

    iUEta = tfs.add(iWs, tfs.from_dense(kron(LamInvSigLam[0, :, :], tf.cast(tfla.diag(tfs.reduce_sum(P, axis=0)), dtype=dtype))))    
    LiUEta = tf_sparse_cholesky(iUEta)
    mu1 = tf.numpy_function(scipy_sparse_solve_triangular, [LiUEta.values, LiUEta.indices, [ny*ns,ny*ns], tf.reshape(tf.transpose(mu0), [nf * np, 1])], dtype)
    Eta = tf.numpy_function(scipy_sparse_solve_triangular, [LiUEta.values, LiUEta.indices, [ny*ns, ny*ns], mu1 + tfr.normal([nf * np, 1], dtype=dtype)], dtype)

    return tf.transpose(tf.reshape(Eta, [nf, np]))
