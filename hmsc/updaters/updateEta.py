import numpy as np
import tensorflow as tf

tfla, tfm, tfr, tfs = tf.linalg, tf.math, tf.random, tf.sparse

from hmsc.utils.tflautils import kron, scipy_cholesky, tf_sparse_matmul, tf_sparse_cholesky, scipy_sparse_solve_triangular, convert_sparse_tensor_to_sparse_csc_matrix

from scipy.sparse.linalg import splu, spsolve_triangular

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


def modelSpatialFull(S, iD, Pi, Lambda, AlphaInd, iWg, nu, nf, dtype=np.float64): 
    #TODO a lot of unnecessary tanspositions - rework if considerably affects perfomance
    LamInvSigLam = tf.scatter_nd(Pi, tf.einsum("hj,ij,kj->ihk", Lambda, iD, Lambda), [nu,nf,nf])
    mu0 = tf.scatter_nd(Pi, tf.matmul(iD * S, Lambda, transpose_b=True), [nu,nf])
    iWs = tf.reshape(tf.transpose(tfla.diag(tf.transpose(tf.gather(iWg, AlphaInd), [1,2,0])), [2,0,3,1]), [nf*nu,nf*nu])
    iUEta = iWs + tf.reshape(tf.transpose(tfla.diag(tf.transpose(LamInvSigLam, [1,2,0])), [0,2,1,3]), [nf*nu,nf*nu])
    LiUEta = tfla.cholesky(iUEta)
    mu1 = tfla.triangular_solve(LiUEta, tf.reshape(tf.transpose(mu0), [nf*nu,1]))
    eta = tfla.triangular_solve(LiUEta, mu1 + tfr.normal([nf*nu,1], dtype=dtype), adjoint=True)
    Eta = tf.transpose(tf.reshape(eta, [nf,nu]))
    return Eta


def modelNonSpatial(S, iD, Pi, Lambda, nu, nf, dtype=np.float64):
    LamInvSigLam = tf.scatter_nd(Pi, tf.einsum("hj,ij,kj->ihk", Lambda, iD, Lambda), [nu,nf,nf])
    mu0 = tf.scatter_nd(Pi, tf.matmul(iD * S, Lambda, transpose_b=True), [nu,nf])
    iV = tf.eye(nf, dtype=dtype) + LamInvSigLam
    LiV = tfla.cholesky(iV)
    mu1 = tfla.triangular_solve(LiV, tf.expand_dims(mu0, -1))
    Eta = tf.squeeze(tfla.triangular_solve(LiV, mu1 + tfr.normal([nu,nf,1], dtype=dtype), adjoint=True), -1)
    return Eta


def modelSpatialNNGP(S, iD, Pi, Lambda, AlphaInd, iWg, iSigma, ny, ns, nu, nf, dtype=np.float64):
    LamInvSigLam = tfs.from_dense(tf.scatter_nd(Pi, tf.einsum("hj,ij,kj->ihk", Lambda, iD, Lambda), [nu,nf,nf]))
    mu0 = tf.scatter_nd(Pi, tf.matmul(iD * S, Lambda, transpose_b=True), [nu,nf])

    mask_values = tf.constant([True, False])
    def tf_sparse_gather(A, Ind):
        def single_iter(x):
            mask = tf.where(tf.equal(A.indices[:, 0], tf.cast(Ind[x], dtype=tf.int64)), mask_values[0], mask_values[1])
            return tfs.to_dense(tfs.reduce_sum(tf.sparse.retain(A, mask), axis=0, keepdims=True, output_is_sparse=True))[-1, :, :]
        return tfs.from_dense(tf.map_fn(single_iter, tf.range(nf), fn_output_signature=dtype))

    def tf_sparse_block_diag(A, n, nx, ny):
        def tf_sparse_update_indices():
            x_stride = A.indices[:,0]*nx
            y_stride = A.indices[:,0]*ny
            return tf.stack([A.indices[:,1]+x_stride, A.indices[:,2]+y_stride], axis=1)
        return tfs.SparseTensor(indices=tf_sparse_update_indices(), values=A.values, dense_shape=(n*nx,n*ny))

    LamInvSigLam_bdiag = tf_sparse_block_diag(LamInvSigLam, nu, nf, nf)
    iWg_bdiag = tf_sparse_block_diag(tf_sparse_gather(iWg, AlphaInd), nf, nu, nu)
    iUEta = tfs.add(tfs.add(iWg_bdiag, LamInvSigLam_bdiag), tfs.eye(nu*nf, dtype=dtype))   

    def modelSpatialNNGP_scipy(Ax, Ai, Ashape, mu0, nu, nf, dtype=np.float64):
        iUEta_csc = convert_sparse_tensor_to_sparse_csc_matrix(Ax, Ai, Ashape)
        LU_factor = splu(iUEta_csc)
        LiUEta = LU_factor.L.multiply(np.sqrt(LU_factor.U.diagonal()))
        mu1 = spsolve_triangular(LiUEta, np.reshape(mu0, [nu*nf]))
        eta = spsolve_triangular(LiUEta.transpose(), mu1 + np.random.normal(dtype(0), dtype(1), size=[nf*nu]), lower=False)
        Eta = np.reshape(eta, [nu,nf])
        return Eta

    #modelSpatialNNGP_local = lambda iUEta_csr, mu0, nu, nf: modelSpatialNNGP_scipy(iUEta_csr, mu0, nu, nf)
    Eta = tf.numpy_function(modelSpatialNNGP_scipy, [iUEta.values, iUEta.indices, [ny*ns, ny*ns], mu0, nu, nf], dtype)
    Eta = tf.ensure_shape(Eta, [nu, None])

    return Eta

    # LiUEta = tf_sparse_cholesky(iUEta)
    # mu1 = tf.numpy_function(scipy_sparse_solve_triangular, [LiUEta.values, LiUEta.indices, [ny*ns, ny*ns], tf.reshape(tf.transpose(mu0), [nf * nu, 1])], dtype)
    # Eta = tf.numpy_function(scipy_sparse_solve_triangular, [LiUEta.values, LiUEta.indices, [ny*ns, ny*ns], mu1 + tfr.normal([nf * nu, 1], dtype=dtype)], dtype)

    #return tf.transpose(tf.reshape(Eta, [nf, nu]))

def modelSpatialGPP(Y, S, Pi, Lambda, AlphaInd, iSigma, Fg, idDg, idDW12g, xDim, ny, ns, nu, nf, nK, dtype=tf.float64):

        nyFull = tf.reduce_sum(tf.where(tf.reduce_sum(tf.cast(~tfm.is_nan(Y), tf.uint8), axis=1) == ns, 1, 0))

        if xDim != 0:
            raise NotImplementedError

        if nyFull <= 0:
            pass
        
        if nu == ny:
            LamInvSigLam = tf.einsum("ij,j,kj->kj", Lambda, iSigma**-2, Lambda)
            mu0 = tf.einsum("ij,kj->ik", S, tf.einsum("ij,i->ij", Lambda, iSigma))
            iV = tf.eye(nf, dtype=dtype) + LamInvSigLam
            LiV = tfla.cholesky(iV)
            mu1 = tfla.triangular_solve(LiV, tf.expand_dims(mu0, -1))
            Eta = tf.squeeze(tfla.triangular_solve(LiV, mu1 + tfr.normal([nu,nf,1], dtype=dtype), adjoint=True), -1)

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

            def tf_sparse_block_diag(A, I, nf, nx, ny):
                A1 = tf_sparse_gather(A, I, nf)
                A2 = tf_sparse_update_indices(A1, nx, ny)
                return tfs.SparseTensor(indices=A2, values=A1.values, dense_shape=(nf*nx,nf*ny))

            Fmat = tf_sparse_block_diag(Fg, AlphaInd, nf, nK, nK)
            idD1W12 = tf_sparse_block_diag(idDW12g, AlphaInd, nf, nK, nu)
    
            LamSigLamT = tf.einsum("ji,jk,km->im", Lambda, tfla.diag(iSigma), Lambda)

            P = tfs.SparseTensor(tf.squeeze(tf.stack([tf.expand_dims(tf.range(ny, dtype=tf.int64),1), Pi], axis=1)), tf.ones([ny], dtype=dtype), [ny, nu])
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

            etaR = tfs.add(tf_sparse_matmul(LiA, tfs.from_dense(tfr.normal([nu*nf,1], dtype=dtype))), tf_sparse_matmul(tmp1, tfs.from_dense(tfr.normal([nK*nf,1], dtype=dtype))))
            Eta = tfs.to_dense(tfs.reshape(tfs.add(tfs.add(mu1, mu2), etaR), shape=[nu,nf]))

        return Eta

