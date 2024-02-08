import numpy as np
import tensorflow as tf
from hmsc.utils.tf_named_func import tf_named_func
tfla, tfm, tfr = tf.linalg, tf.math, tf.random

@tf_named_func("RRRPriors")
def updatewRRRPriors(params, modelDims, priorHyperparams, dtype=tf.float64):
    Delta = params["DeltaRRR"]
    Lambda = params["wRRR"]
    nu = priorHyperparams["nuRRR"]
    a1 = priorHyperparams["a1RRR"]
    b1 = priorHyperparams["b1RRR"]
    a2 = priorHyperparams["a2RRR"]
    b2 = priorHyperparams["b2RRR"]
    nf = modelDims["ncRRR"]
    ns = modelDims["ns"]
    
    aVec = tf.concat([[a1], tf.repeat(a2, nf-1)], 0)
    bVec = tf.concat([[b1], tf.repeat(b2, nf-1)], 0)

    Tau = tfm.cumprod(Delta)
    Lambda2 = Lambda**2
    aPsi = nu/2 + 0.5
    bPsi = nu/2 + 0.5*Lambda2 * Tau[:,None]
    Psi = tf.squeeze(tfr.gamma([1], aPsi, bPsi, dtype=dtype), 0)

    M = Psi * Lambda2
    rowSumM = tf.reduce_sum(M, -1)
    for h in range(nf):
        Tau = tfm.cumprod(Delta)
        ad = aVec[h] + 0.5 * ns * (nf-h)
        bd = bVec[h] + 0.5 * tf.reduce_sum(Tau[h:] * rowSumM[h:]) / Delta[h]
        Delta = tf.tensor_scatter_nd_update(Delta, [[h]], tfr.gamma([1], ad, bd, dtype=dtype))

    return Psi, Delta