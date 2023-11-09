import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hmsc.utils.tf_named_func import tf_named_func
tfm = tf.math
tfd = tfp.distributions

@tf_named_func("sigma")
def updateSigma(params, modelDims, data, priorHyperparameters, dtype=np.float64):
    """Update prior(s) for whole model:
    sigma - residual variance.

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Z - latent variables
        Beta - species niches
        Eta - site loadings
        Lambda - species loadings
        sigma - residual variance
        Y - community data
        X - environmental data
        Pi - study design
        distr -
        aSigma -
        bSigma -
    """
    ns = modelDims["ns"]
    nr = modelDims["nr"]

    Z = params["Z"]
    Beta = params["Beta"]
    EtaList = params["Eta"]
    LambdaList = params["Lambda"]
    sigma = params["sigma"]
    X = params["Xeff"]

    Y = data["Y"]
    Pi = data["Pi"]
    distr = data["distr"]
    aSigma = priorHyperparameters["aSigma"]
    bSigma = priorHyperparameters["bSigma"]

    Yo = tf.cast(tfm.logical_not(tfm.is_nan(Y)), dtype)
    indVarSigma = tf.equal(distr[:,1], 1)
    #TODO code below contains redundant calculations for fixed variance columns

    if len(X.shape.as_list()) == 2: #tf.rank(X)
      LFix = tf.matmul(X, Beta)
    else:
      LFix = tf.einsum("jik,kj->ij", X, Beta)
    LRanLevelList = [None] * nr
    for r, (Eta, Lambda) in enumerate(zip(EtaList, LambdaList)):
        LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:, r]), Lambda)

    L = LFix + sum(LRanLevelList)
    Eps = Z - L
    
    alpha = aSigma + tf.reduce_sum(Yo, 0) / 2.0
    beta = bSigma + tf.reduce_sum(Yo*(Eps**2), 0) / 2.0
    isigma2 = tfd.Gamma(concentration=alpha, rate=beta).sample()
    sigmaNew = tf.where(indVarSigma, tfm.rsqrt(isigma2), sigma)

    return tf.ensure_shape(sigmaNew, [ns])

