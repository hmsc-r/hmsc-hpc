import numpy as np
import tensorflow as tf
from scipy.sparse import csc_matrix, coo_matrix, block_diag, kron
from scipy.sparse.linalg import splu, spsolve_triangular

tfla, tfr, tfs = tf.linalg, tf.random, tf.sparse

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
    nr = modelDims["nr"]
    npVec = modelDims["np"]
    
    
    
    iD = tf.ones_like(Z) * sigma**-2
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
                    modelSpatialNNGP_local = lambda LamInvSigLam, mu0, Alpha, nf: modelSpatialNNGP_scipy(LamInvSigLam, mu0, Alpha, rLPar["iWList_csc"], npVec[r], nf)
                    # EtaListNew[r] = modelSpatialNNGP_local(LamInvSigLam, mu0, AlphaInd, nf)
                    Eta = tf.numpy_function(modelSpatialNNGP_local, [LamInvSigLam, mu0, AlphaInd, nf], dtype)
                    EtaListNew[r] = tf.ensure_shape(Eta, [npVec[r], None])
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


def modelSpatialNNGP_scipy(LamInvSigLam, mu0, Alpha, iWList, nu, nf, dtype=np.float64):
    LamInvSigLam_bdiag = block_diag([LamInvSigLam[i] for i in range(nu)], dtype=dtype)
    dataList, colList, rowList = [None]*nf, [None]*nf, [None]*nf
    for h, a in enumerate(Alpha):
      iW = coo_matrix(iWList[a])
      dataList[h] = iW.data
      colList[h] = iW.col + h
      rowList[h] = iW.row + h
    dataArray = np.concatenate(dataList)
    colArray = np.concatenate(colList)
    rowArray = np.concatenate(rowList)
    iUEta = csc_matrix(coo_matrix((dataArray,(colArray,rowArray)), [nu*nf,nu*nf])) + LamInvSigLam_bdiag
    # iWs = [kron(iWList[a], csc_matrix(coo_matrix(([1],([h],[h])), [nf,nf]))) for h, a in enumerate(Alpha)] #TODO redo with indices?
    # iUEta = sum(iWs) + LamInvSigLam_bdiag
    LU_factor = splu(iUEta)
    LiUEta = LU_factor.L.multiply(np.sqrt(LU_factor.U.diagonal()))
    mu1 = spsolve_triangular(LiUEta, np.reshape(mu0, [nu*nf]))
    eta = spsolve_triangular(LiUEta.transpose(), mu1 + np.random.normal(dtype(0), dtype(1), size=[nf*nu]), lower=False)
    Eta = np.reshape(eta, [nu,nf])
    return Eta
    