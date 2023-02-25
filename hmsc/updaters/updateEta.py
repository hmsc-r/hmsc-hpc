import numpy as np
import tensorflow as tf

tfla, tfm, tfr, tfs = tf.linalg, tf.math, tf.random, tf.sparse

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
    Y = data["Y"]
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
    
            if rLPar["sDim"] > 0:
                if rLPar["spatialMethod"] == "Full":
                    EtaListNew[r] = modelSpatialFull(S, iD, Pi[:,r,None], Lambda, AlphaInd, rLPar["iWg"], npVec[r], nf)
                elif rLPar["spatialMethod"] == "GPP":
                    EtaListNew[r] = modelSpatialGPP(Y, S, Pi[:,r,None], Lambda, AlphaInd, iSigma, rLPar["Fg"], rLPar["idDg"], rLPar["idDW12g"], rLPar["xDim"], ny, ns, npVec[r], nf, rLPar["nK"])
                elif rLPar["spatialMethod"] == "NNGP":
                    EtaListNew[r] = modelSpatialNNGP(S, iD, Pi[:,r,None], Lambda, AlphaInd, rLPar["iWg"], iSigma, ny, ns, npVec[r], nf)
            else:
                EtaListNew[r] = modelNonSpatial(S, iD, Pi[:,r,None], Lambda, npVec[r], nf, dtype)
            
            LRanLevelList[r] = tf.matmul(tf.gather(EtaListNew[r], Pi[:,r]), Lambda)
        else:
            EtaListNew[r] = Eta

    return EtaListNew


def modelSpatialFull(S, iD, Pi, Lambda, AlphaInd, iWg, np, nf, dtype=np.float64): 
    #TODO a lot of unnecessary tanspositions - rework if considerably affects perfomance
    LamInvSigLam = tf.scatter_nd(Pi, tf.einsum("hj,ij,kj->ihk", Lambda, iD, Lambda), [np,nf,nf])
    mu0 = tf.scatter_nd(Pi, tf.matmul(iD * S, Lambda, transpose_b=True), [np,nf])
    iWs = tf.reshape(tf.transpose(tfla.diag(tf.transpose(tf.gather(iWg, AlphaInd), [1,2,0])), [2,0,3,1]), [nf*np,nf*np])
    iUEta = iWs + tf.reshape(tf.transpose(tfla.diag(tf.transpose(LamInvSigLam, [1,2,0])), [0,2,1,3]), [nf*np,nf*np])
    LiUEta = tfla.cholesky(iUEta)
    mu1 = tfla.triangular_solve(LiUEta, tf.reshape(tf.transpose(mu0), [nf*np,1]))
    eta = tfla.triangular_solve(LiUEta, mu1 + tfr.normal([nf*np,1], dtype=dtype), adjoint=True)
    Eta = tf.transpose(tf.reshape(eta, [nf,np]))
    return Eta


def modelNonSpatial(S, iD, Pi, Lambda, np, nf, dtype=np.float64):
    LamInvSigLam = tf.scatter_nd(Pi, tf.einsum("hj,ij,kj->ihk", Lambda, iD, Lambda), [np,nf,nf])
    mu0 = tf.scatter_nd(Pi, tf.matmul(iD * S, Lambda, transpose_b=True), [np,nf])
    iV = tf.eye(nf, dtype=dtype) + LamInvSigLam
    LiV = tfla.cholesky(iV)
    mu1 = tfla.triangular_solve(LiV, tf.expand_dims(mu0, -1))
    Eta = tf.squeeze(tfla.triangular_solve(LiV, mu1 + tfr.normal([np,nf,1], dtype=dtype), adjoint=True), -1)
    return Eta


def modelSpatialNNGP(S, iD, Pi, Lambda, AlphaInd, iWg, iSigma, ny, ns, np, nf, dtype=np.float64):
    LamInvSigLam = tf.scatter_nd(Pi, tf.einsum("hj,ij,kj->ihk", Lambda, iD, Lambda), [np,nf,nf])
    mu0 = tf.scatter_nd(Pi, tf.matmul(iD * S, Lambda, transpose_b=True), [np,nf])

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

def modelSpatialGPP(Y, S, Pi, Lambda, AlphaInd, iSigma, Fg, idDg, idDW12g, xDim, ny, ns, np, nf, nK, dtype=tf.float64):

        nyFull = tf.reduce_sum(tf.where(tf.reduce_sum(tf.cast(~tfm.is_nan(Y), tf.uint8), axis=1) == ns, 1, 0))

        if xDim != 0:
            raise NotImplementedError

        if nyFull <= 0:
            pass
        
        if np == ny:
            LamInvSigLam = tf.einsum("ij,j,kj->kj", Lambda, iSigma**-2, Lambda)
            mu0 = tf.einsum("ij,kj->ik", S, tf.einsum("ij,i->ij", Lambda, iSigma))
            iV = tf.eye(nf, dtype=dtype) + LamInvSigLam
            LiV = tfla.cholesky(iV)
            mu1 = tfla.triangular_solve(LiV, tf.expand_dims(mu0, -1))
            Eta = tf.squeeze(tfla.triangular_solve(LiV, mu1 + tfr.normal([np,nf,1], dtype=dtype), adjoint=True), -1)

        else:
            idD = tf.gather(idDg, AlphaInd, axis=1)
    
            def tf_sparse_gather(A, Ind, nf):
                def single_iter(x):
                    return tf.gather(A, x, axis=0)
                return tfs.from_dense(tf.map_fn(single_iter, Ind, fn_output_signature=tf.float64))

            def tf_sparse_update_indices(A, nx, ny):
                x_stride = A.indices[:,0]*nx
                y_stride = A.indices[:,0]*ny
                return tf.stack([A.indices[:,1]+x_stride, A.indices[:,1]+y_stride], axis=1)

            def tf_sparse_diagnal_ops(A, I, nf, nx, ny):
                A1 = tf_sparse_gather(A, I, nf)
                A2 = tf_sparse_update_indices(A1, nx, ny)
                return tfs.SparseTensor(indices=A2, values=A1.values, dense_shape=(nf*nx,nf*ny))

            Fmat = tf_sparse_diagnal_ops(Fg, AlphaInd, nf, nK, nK)
            idD1W12 = tf_sparse_diagnal_ops(idDW12g, AlphaInd, nf, nK, np)
    
            LamSigLamT = tf.einsum("ji,jk,km->im", Lambda, tfla.diag(iSigma), Lambda)

            P = tfs.SparseTensor(tf.squeeze(tf.stack([tf.expand_dims(tf.range(ny, dtype=tf.int64),1), Pi], axis=1)), tf.ones([ny], dtype=dtype), [ny, np])
            fS = tfs.expand_dims(tfs.reshape(tf_sparse_matmul(tf_sparse_matmul(P, tfs.from_dense(S)), tfs.from_dense(tf.transpose(tf.reshape(tf.tile(iSigma, [nf]), [nf, len(iSigma)])))), [-1]), axis=1)

            tmp1 = tfs.from_dense(kron(LamSigLamT, tf.cast(tfla.diag(tfs.reduce_sum(P, axis=1)), dtype=dtype)))
            # tmp1 = tfs.from_dense(kron(LamSigLamT, tfs.to_dense(tfs.SparseTensor(indices=tfs.eye(ny).indices, values=tfs.reduce_sum(P, axis=1), dense_shape=[ny,ny]))))
            tmp2 = tfs.add(tmp1, tfs.SparseTensor(indices=tfs.eye(ny*ns).indices, values= tf.reshape(idD, [-1]), dense_shape=[ny*ns,ny*ns]))

            iA = tfs.from_dense(tfla.solve(tfs.to_dense(tmp2), tf.eye(ny*ns, dtype=dtype)))
            LiA = tf_sparse_cholesky(iA)

            iAidD1W12 = tf_sparse_matmul(idD1W12, iA)
    
            H = tfs.add(Fmat, tfs.map_values(tf.multiply, tf_sparse_matmul(idD1W12, tfs.transpose(iAidD1W12)), -1))
            RH = tf_sparse_cholesky(H)
            iRH = tfla.solve(tfs.to_dense(RH), tf.eye(nK*ns, dtype=dtype))

            mu1 = tf_sparse_matmul(iA, fS)
            tmp1 = tf_sparse_matmul(tfs.transpose(iAidD1W12), tfs.from_dense(iRH))
            mu2 = tf_sparse_matmul(tmp1, tf_sparse_matmul(tfs.transpose(tmp1), fS))

            etaR = tfs.add(tf_sparse_matmul(LiA, tfs.from_dense(tfr.normal([np*nf,1], dtype=dtype))), tf_sparse_matmul(tmp1, tfs.from_dense(tfr.normal([nK*nf,1], dtype=dtype))))
            Eta = tfs.to_dense(tfs.reshape(tfs.add(tfs.add(mu1, mu2), etaR), shape=[np,nf]))

        return Eta

