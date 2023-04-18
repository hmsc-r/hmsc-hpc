import numpy as np
import tensorflow as tf

tfla, tfm, tfr = tf.linalg, tf.math, tf.random

from hmsc.utils.tflautils import kron

def updatewRRR(params, modelDims, modelData, rLHyperparams, dtype=tf.float64):
    ny = modelDims["ny"]
    ns = modelDims["ns"]
    nr = modelDims["nr"]
    ncRRR = modelDims["ncRRR"]
    ncNRRR = modelDims["ncNRRR"]
    ncORRR = modelDims["ncORRR"]
    ncsel = modelDims["ncsel"]
    npVec = modelDims["np"]

    Z = params["Z"]
    Beta = params["Beta"]
    sigma = params["sigma"]
    iSigma = 1 / sigma
    EtaList = params["Eta"]
    LambdaList = params["Lambda"]

    PsiRRR = params["PsiRRR"]
    DeltaRRR = params["DeltaRRR"]

    X = modelData["X"]
    X1A = modelData["X1A"]
    XRRR = modelData["XRRR"]
    Pi = modelData["Pi"]
    
    BetaNRRR = Beta[:ncNRRR,:]
    BetaRRR = Beta[ncNRRR:,:]

    if isinstance(X1A, list):
        LFix = tf.einsum("ijk,ki->ji", tf.stack(X1A), BetaNRRR)
    else:
        LFix = tf.matmul(X1A, BetaNRRR)

    LRanLevelList = [None] * nr
    for r, (Eta, Lambda, rLPar) in enumerate(zip(EtaList, LambdaList, rLHyperparams)):
        if(rLPar["xDim"] == 0):
            LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:,r]), Lambda)
        else:
            raise NotImplementedError
        
    if nr > 1:
        S = Z - (LFix + sum(LRanLevelList))
    else:
        S = Z - LFix

    A1 = tf.einsum("ij,jk,lk->il", BetaRRR, tfla.diag(iSigma), BetaRRR)
    A2 = tf.einsum("ji,jk->ik", XRRR, XRRR)
    QtiSigmaQ = kron(A2, A1)
    tauRRR = tfm.cumprod(DeltaRRR, axis=0)
    tauMatRRR = tf.tile(tauRRR, [1,ncORRR])
    iU = tfla.diag(tf.reshape(PsiRRR*tauMatRRR, [-1])) + QtiSigmaQ
    LiU = tfla.cholesky(iU)
    #U = tfla.inv(LiU)
    U = tfla.cholesky_solve(LiU, tf.eye(ncRRR*ncORRR, dtype=dtype))
    mu1 = tf.reshape(tf.einsum("ij,jk,lk,lm->im", BetaRRR, tfla.diag(iSigma), S, XRRR), [-1])[:,None]
    mu = tf.matmul(U, mu1)
    we = mu + tfla.triangular_solve(LiU, tfr.normal([ncRRR*ncORRR,1], dtype=dtype), adjoint=True)
    wRRR = tf.reshape(we, shape=(ncRRR, ncORRR))

    if ncRRR > 0:
        XB = tf.einsum("ij,kj->ik", XRRR, wRRR)
        if isinstance(X1A, list):
            raise NotImplementedError
        else:
            X = tf.concat([X1A, XB], axis=-1)

    return wRRR, X
