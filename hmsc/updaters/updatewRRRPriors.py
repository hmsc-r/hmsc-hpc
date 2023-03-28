import numpy as np
import tensorflow as tf

tfla, tfm, tfr = tf.linalg, tf.math, tf.random

def updatewRRRPriors(params, modelDims, priorHyperparams, dtype=tf.float64):
    nu = modelDims["nuRRR"]

    a1 = priorHyperparams["a1RRR"]
    b1 = priorHyperparams["b1RRR"]
    a2 = priorHyperparams["a2RRR"]
    b2 = priorHyperparams["b2RRR"]

    Delta = params["DeltaRRR"]
    Lambda = params["wRRR"]

    nf = Lambda.shape[0]
    ns = Lambda.shape[1]
    
    Tau = tfm.cumprod(Delta, axis=1)
    Lambda2 = Lambda**2
    aPsi = nu/2 + 0.5
    bPsi = nu/2 + 0.5*Lambda2 * tf.reshape(tf.tile(Tau, [1,ns]), shape=(nf,ns))
    Psi = tfr.gamma([1], aPsi, bPsi, dtype=dtype)[-1,:,:]

    M = Psi * Lambda2
    ad = a1 + 0.5*ns*nf
    bd = b1 + 0.5 * tf.reduce_sum(Tau * tf.reduce_sum(M, axis=1)) / Delta[0]
    Delta = tf.tensor_scatter_nd_update(Delta, [[0,0]], [tf.squeeze(tfr.gamma([1], ad, bd, dtype=dtype))])

    for h in range(1,nf):
        Tau = tfm.cumprod(Delta, axis=1)
        ad = a2 + 0.5 * ns * (nf-h+1)
        bd = b2 + 0.5 * tf.reduce_sum(Tau[h:] * tf.reduce_sum(M[h:,:], axis=0)) / Delta[h]
        Delta = tf.tensor_scatter_nd_update(Delta, [[h,0]], [tf.squeeze(tfr.gamma([1], ad, bd, dtype=dtype))])

    return Psi, Delta