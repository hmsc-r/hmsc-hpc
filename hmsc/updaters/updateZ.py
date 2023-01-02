import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfm, tfr = tf.math, tf.random

def updateZ(params, dtype=np.float64):
    """Update conditional updater(s)
    Z - latent variable.
        
    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Beta - species niches
        Eta - site loadings
        Lambda - species loadings
        sigma - residual variance
        Y - community data
        X - environmental data
        Pi - study design
        distr - ??
    """

    Beta = params["BetaLambda"]["Beta"]
    EtaList = params["Eta"]
    LambdaList = params["BetaLambda"]["Lambda"]
    sigma = params["sigma"]

    Y = params["Y"]
    X = params["X"]
    Pi = params["Pi"]
    distr = params["distr"]

    ny, ns = Y.shape
    nr = len(EtaList)
    LFix = tf.matmul(X, Beta)
    LRanLevelList = [None] * nr
    for r, (Eta, Lambda) in enumerate(zip(EtaList, LambdaList)):
        LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:, r]), Lambda)
    L = LFix + sum(LRanLevelList)
    Yo = tfm.logical_not(tfm.is_nan(Y))

    # no data augmentation for normal model in columns with continious unbounded data
    indColNormal = tf.squeeze(tf.where(distr[:, 0] == 1), -1)
    YN = tf.gather(Y, indColNormal, axis=-1)
    YoN = tf.gather(Yo, indColNormal, axis=-1)
    LN = tf.gather(L, indColNormal, axis=-1)
    sigmaN = tf.gather(sigma, indColNormal)
    ZN = tf.cast(YoN, dtype) * YN + (1 - tf.cast(YoN, dtype)) * (
        LN + tfr.normal([ny, tf.size(indColNormal)], dtype=dtype) * sigmaN
    )

    # Albert and Chib (1993) data augemntation for probit model in columns with binary data
    indColProbit = tf.squeeze(tf.where(distr[:, 0] == 2), -1)
    YP = tf.gather(Y, indColProbit, axis=-1)
    YoP = tf.gather(Yo, indColProbit, axis=-1)
    LP = tf.gather(L, indColProbit, axis=-1)
    sigmaP = tf.gather(sigma, indColProbit)
    low = tf.where(
        tfm.logical_or(YP == 0, tfm.logical_not(YoP)),
        tf.cast(-np.inf, dtype),
        tf.zeros_like(YP),
    )
    high = tf.where(
        tfm.logical_or(YP == 1, tfm.logical_not(YoP)),
        tf.cast(np.inf, dtype),
        tf.zeros_like(YP),
    )
    ZP = tfd.TruncatedNormal(
        loc=LP, scale=sigmaP, low=low, high=high, name="TruncatedNormal"
    ).sample()

    ZStack = tf.concat([ZN, ZP], -1)
    indColStack = tf.concat([indColNormal, indColProbit], 0)
    ZNew = tf.transpose(
        tf.scatter_nd(indColStack[:, None], tf.transpose(ZStack), Y.shape[::-1])
    )
    return ZNew
