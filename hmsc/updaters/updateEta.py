import numpy as np
import tensorflow as tf
from scipy.sparse.linalg import splu, spsolve_triangular
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, block_diag
from hmsc.utils.tf_named_func import tf_named_func
tfla, tfm, tfr, tfs = tf.linalg, tf.math, tf.random, tf.sparse

@tf_named_func("eta")
def updateEta(params, modelDims, data, rLHyperparams, dtype=np.float64):
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
    iD = params["iD"]
    Beta = params["Beta"]
    LambdaList = params["Lambda"]
    EtaList = params["Eta"]
    AlphaIndList = params["AlphaInd"]
    X = params["Xeff"]
    Pi = data["Pi"]
    ny = modelDims["ny"]
    ns = modelDims["ns"]
    nr = modelDims["nr"]
    npVec = modelDims["np"]
    
    if len(X.shape.as_list()) == 2: #tf.rank(X)
      LFix = tf.matmul(X, Beta)
    else:
      LFix = tf.einsum("jik,kj->ij", X, Beta)
    LRanLevelList = [None] * nr
    for r, (Eta, Lambda) in enumerate(zip(EtaList, LambdaList)):
        LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:,r]), Lambda)

    EtaListNew = [None] * nr
    for r, (Eta, Lambda, AlphaInd, rLPar) in enumerate(zip(EtaList, LambdaList, AlphaIndList, rLHyperparams)):
        nf = tf.cast(tf.shape(Lambda)[-2], tf.int64)
        if nf > 0:
            S = Z - LFix - sum([LRanLevelList[rInd] for rInd in np.setdiff1d(np.arange(nr), r)])
            mu0 = tf.scatter_nd(Pi[:,r,None], tf.matmul(iD * S, Lambda, transpose_b=True), [npVec[r],nf], name="mu0")
            # LamInvSigLam = tf.scatter_nd(Pi[:,r,None], tf.einsum("hj,ij,kj->ihk", Lambda, iD, Lambda), [npVec[r],nf,nf])
            #TODO bottleneck for non-spatial model
            # Pi_iD = tf.scatter_nd(Pi[:,r,None], iD, [npVec[r],ns], name="Pi_iD")
            piMat = tfs.SparseTensor(tf.stack([Pi[:,r], tf.range(ny,dtype=tf.int64)], axis=-1), tf.ones([ny],dtype), [npVec[r],ny])
            Pi_iD = tfs.sparse_dense_matmul(piMat, iD, name="Pi_iD")
            commonFlag = tf.reduce_all(Pi_iD == Pi_iD[0,:])
            if commonFlag:
              LamInvSigLam = tf.einsum("hj,j,kj->hk", Lambda, Pi_iD[0,:], Lambda, name="LamInvSigLam")
            else:
              LamInvSigLam = tf.einsum("hj,ij,kj->ihk", Lambda, Pi_iD, Lambda, name="LamInvSigLam")

            if rLPar["sDim"] == 0:
                if commonFlag:
                    EtaListNew[r] = modelNonSpatialCommon(LamInvSigLam, mu0, npVec[r], nf, dtype)
                else:
                    EtaListNew[r] = modelNonSpatial(LamInvSigLam, mu0, npVec[r], nf, dtype)
            else:
                if commonFlag:
                    LamInvSigLam = tf.tile(LamInvSigLam[None,:,:], [npVec[r],1,1])
                    
                if rLPar["spatialMethod"] == "Full":
                    EtaListNew[r] = modelSpatialFull(LamInvSigLam, mu0, AlphaInd, rLPar["iWg"], npVec[r], nf, dtype)
                elif rLPar["spatialMethod"] == "GPP":
                    EtaListNew[r] = modelSpatialGPP(LamInvSigLam, mu0, AlphaInd, rLPar["Fg"], rLPar["idDg"], rLPar["idDW12g"], rLPar["nK"], npVec[r], nf, dtype)
                elif rLPar["spatialMethod"] == "NNGP":                
                    modelSpatialNNGP_local = lambda LamInvSigLam, mu0, Alpha, nf: modelSpatialNNGP_scipy(LamInvSigLam, mu0, Alpha, rLPar["iWList_csr"], npVec[r], nf, dtype)
                    # EtaListNew[r] = modelSpatialNNGP_local(LamInvSigLam, mu0, AlphaInd, nf)
                    Eta = tf.numpy_function(modelSpatialNNGP_local, [LamInvSigLam, mu0, AlphaInd, nf], dtype)
                    EtaListNew[r] = tf.ensure_shape(Eta, [npVec[r], None])              
            
            LRanLevelList[r] = tf.matmul(tf.gather(EtaListNew[r], Pi[:,r]), Lambda)
        else:
            EtaListNew[r] = Eta
        EtaListNew[r] = tf.ensure_shape(EtaListNew[r], [npVec[r],None])

    return EtaListNew


def modelNonSpatialCommon(LamInvSigLam, mu0, np, nf, dtype=np.float64):
    # tf.print("using common Eta sampler option")
    iV = tf.eye(nf, dtype=dtype) + LamInvSigLam
    LiV = tfla.cholesky(iV, name="LiV")
    mu1 = tfla.triangular_solve(LiV, tfla.matrix_transpose(mu0), name="mu1")
    Eta = tfla.matrix_transpose(tfla.triangular_solve(LiV, mu1 + tfr.normal([nf,np], dtype=dtype), adjoint=True))
    return Eta
  

def modelNonSpatial(LamInvSigLam, mu0, np, nf, dtype=np.float64):
    iV = tf.eye(nf, dtype=dtype) + LamInvSigLam
    LiV = tfla.cholesky(iV, name="LiV")
    # LamInvSigLam_u, LamInvSigLam_id = tf.raw_ops.UniqueV2(x=LamInvSigLam, axis=[0])
    # iV_u = tf.eye(nf, dtype=dtype) + LamInvSigLam_u
    # LiV_u = tfla.cholesky(iV_u)
    # LiV = tf.gather(LiV_u, LamInvSigLam_id)
    mu1 = tfla.triangular_solve(LiV, tf.expand_dims(mu0, -1), name="mu1")
    Eta = tf.squeeze(tfla.triangular_solve(LiV, mu1 + tfr.normal([np,nf,1], dtype=dtype), adjoint=True, name="Eta"), -1)
    return Eta


def modelSpatialFull(LamInvSigLam, mu0, AlphaInd, iWg, np, nf, dtype=np.float64): 
    #TODO a lot of unnecessary tanspositions - rework if considerably affects perfomance
    iWs = tf.reshape(tf.transpose(tfla.diag(tf.transpose(tf.gather(iWg, AlphaInd), [1,2,0])), [2,0,3,1]), [nf*np,nf*np])
    iUEta = iWs + tf.reshape(tf.transpose(tfla.diag(tf.transpose(LamInvSigLam, [1,2,0])), [0,2,1,3]), [nf*np,nf*np])
    LiUEta = tfla.cholesky(iUEta, name="LiUEta")
    mu1 = tfla.triangular_solve(LiUEta, tf.reshape(tf.transpose(mu0), [nf*np,1]), name="mu1")
    eta = tfla.triangular_solve(LiUEta, mu1 + tfr.normal([nf*np,1], dtype=dtype), adjoint=True, name="eta")
    Eta = tf.transpose(tf.reshape(eta, [nf,np]))
    return Eta


def modelSpatialGPP(LamInvSigLam, mu0, AlphaInd, Fg, idDg, idDW12g, nK, nu, nf, dtype=tf.float64):
    idDst = tf.gather(idDg, AlphaInd)
    Fst = tf.gather(Fg, AlphaInd)
    idDW12st = tf.gather(idDW12g, AlphaInd)
    Fmat = tf.reshape(tf.transpose(tfla.diag(tf.transpose(Fst, [1,2,0])), [2,0,3,1]), [nf*nK,nf*nK])
    # idD1W12 = tf.reshape(tf.transpose(tfla.diag(tf.transpose(tf.gather(idDW12g, AlphaInd), [1,2,0])), [2,0,3,1]), [nf*nu,nf*nK])
    
    Ast = LamInvSigLam + tfla.diag(tf.transpose(idDst))
    LAst = tfla.cholesky(Ast, name="LAst")
    iAst = tfla.cholesky_solve(LAst, tf.eye(nf, batch_shape=[nu], dtype=dtype), name="iAst")
    W21idD_iA_idDW12 = tf.reshape(tf.einsum("hia,ihg,gib->hagb", idDW12st, iAst, idDW12st, name="W21idD_iA_idDW12"), [nf*nK]*2)
    H = Fmat - W21idD_iA_idDW12
    LH = tfla.cholesky(H, name="LH")

    # iA_mu0 = tf.squeeze(tfla.triangular_solve(LAst, tfla.triangular_solve(LAst, mu0[:,:,None]), adjoint=True), -1)
    iA_mu0 = tf.einsum("ihg,ih->ig", iAst, mu0, name="iA_mu0")
    W21idD_iA_mu0 = tf.reshape(tf.einsum("hia,ih->ha", idDW12st, iA_mu0, name="W21idD_iA_mu0"), [nf*nK,1])
    iH_W21idD_iA_mu0 = tf.reshape(tfla.cholesky_solve(LH, W21idD_iA_mu0, name="iH_W21idD_iA_mu0"), [nf,nK])
    iA_idDW12_iH_W21idD_iA_mu0 = tf.einsum("ihg,hia,ha->ig", iAst, idDW12st, iH_W21idD_iA_mu0, name="iA_idDW12_iH_W21idD_iA_mu0")
    etaMu = iA_mu0 + iA_idDW12_iH_W21idD_iA_mu0
    
    etaR1 = tf.squeeze(tfla.triangular_solve(LAst, tfr.normal([nu,nf,1], dtype=dtype), adjoint=True, name="etaR1"), -1)
    tmp = tf.reshape(tfla.triangular_solve(LH, tfr.normal([nf*nK,1], dtype=dtype), adjoint=True, name="tmp"), [nf,nK])
    etaR2 = tf.einsum("ihg,hia,ha->ig", iAst, idDW12st, tmp, name="etaR2")
    Eta = etaMu + etaR1 + etaR2
    # print(Eta)
    return tf.ensure_shape(Eta, [nu,None])


def modelSpatialNNGP_scipy(LamInvSigLam, mu0, Alpha, iWList, nu, nf, dtype=np.float64):
    LamInvSigLam_bdiag = block_diag([LamInvSigLam[i] for i in range(nu)], dtype=dtype)
    dataList, colList, rowList = [None]*int(nf), [None]*int(nf), [None]*int(nf)
    for h, a in enumerate(Alpha):
      iW = coo_matrix(iWList[a])
      dataList[h] = iW.data
      colList[h] = iW.col + h*nu
      rowList[h] = iW.row + h*nu
    dataArray = np.concatenate(dataList)
    colArray = np.concatenate(colList)
    rowArray = np.concatenate(rowList)
    iUEta = csc_matrix((dataArray,(colArray,rowArray)), [nu*nf,nu*nf]) + LamInvSigLam_bdiag
    # iWs = [kron(iWList[a], csc_matrix(coo_matrix(([1],([h],[h])), [nf,nf]))) for h, a in enumerate(Alpha)] #replaced with indices
    # iUEta = sum(iWs) + LamInvSigLam_bdiag
    LU_factor = splu(iUEta, "NATURAL", diag_pivot_thresh=0)
    L, U = LU_factor.L, LU_factor.U
    LiUEta = csr_matrix(L.multiply(np.sqrt(U.diagonal())), dtype=dtype)
    mu1 = spsolve_triangular(LiUEta, np.reshape(mu0, [nu*nf]))
    eta = spsolve_triangular(LiUEta.transpose(), mu1 + np.random.normal(dtype(0), dtype(1), size=[nf*nu]), lower=False)
    Eta = np.reshape(eta, [nu,nf]).astype(dtype)
    return Eta