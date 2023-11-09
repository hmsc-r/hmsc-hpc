import numpy as np
import tensorflow as tf
from hmsc.utils.tf_named_func import tf_named_func
tfla, tfm, tfr = tf.linalg, tf.math, tf.random

@tf_named_func("wRRR")
def updatewRRR(params, modelDims, modelData, rLHyperparams, dtype=tf.float64):
    ns = modelDims["ns"]
    nr = modelDims["nr"]
    ncRRR = modelDims["ncRRR"]
    ncNRRR = modelDims["ncNRRR"]
    ncORRR = modelDims["ncORRR"]

    Z = params["Z"]
    Beta = params["Beta"]
    iD = params["iD"]
    EtaList = params["Eta"]
    LambdaList = params["Lambda"]
    X = params["Xeff"]
    PsiRRR = params["PsiRRR"]
    DeltaRRR = params["DeltaRRR"]

    XRRR = modelData["XRRR"]
    Pi = modelData["Pi"]
    
    BetaNRRR = Beta[:ncNRRR,:]
    BetaRRR = Beta[ncNRRR:,:]

    XNRRR = tf.gather(X, np.arange(ncNRRR), axis=-1)
    if len(XNRRR.shape.as_list()) == 2: #tf.rank(X)
      LFix = tf.matmul(XNRRR, BetaNRRR)
    else:
      LFix = tf.einsum("jik,kj->ij", XNRRR, BetaNRRR)

    LRanLevelList = [None] * nr
    for r, (Eta, Lambda, rLPar) in enumerate(zip(EtaList, LambdaList, rLHyperparams)):
        if(rLPar["xDim"] == 0):
            LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:,r]), Lambda)
        else:
            raise NotImplementedError   
    S = Z - (LFix + sum(LRanLevelList))
    
    BetaRRR_iD_BetaRRRT = tf.einsum("hj,ij,gj->ihg", BetaRRR, iD, BetaRRR)
    QtiDQ = tf.reshape(tf.einsum("ic,ihg,ik->chkg", XRRR, BetaRRR_iD_BetaRRRT, XRRR), [ncORRR*ncRRR]*2)
    iU = tfla.diag(tf.reshape(tf.einsum("hk,h->kh", PsiRRR, tfm.cumprod(DeltaRRR)), [ncORRR*ncRRR])) + QtiDQ
    LiU = tfla.cholesky(iU)
    mu0 = tf.reshape(tf.einsum("hj,ij,ik->kh", BetaRRR, iD*S, XRRR), [ncORRR*ncRRR,1])
    mu = tfla.cholesky_solve(LiU, mu0)
    w = mu + tfla.triangular_solve(LiU, tfr.normal([ncORRR*ncRRR,1], dtype=dtype), adjoint=True)
    wRRR = tf.transpose(tf.reshape(w, shape=[ncORRR, ncRRR]))

    XeffRRR = tf.einsum("ik,hk->ih", XRRR, wRRR)
    if len(XNRRR.shape.as_list()) == 2: #tf.rank(X)
      Xeff = tf.concat([XNRRR, XeffRRR], axis=-1)
    else:
      Xeff = tf.concat([XNRRR, tf.repeat(tf.expand_dims(XeffRRR,0), ns, 0)], axis=-1)

    return wRRR, Xeff
