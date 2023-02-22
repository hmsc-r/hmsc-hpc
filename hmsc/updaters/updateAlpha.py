import numpy as np
import tensorflow as tf

tfm, tfla, tfr, tfs = tf.math, tf.linalg, tf.random, tf.sparse


def updateAlpha(params, rLHyperparams, dtype=np.float64):
    """Update prior(s) for each random level:
    Alpha - scale of site loadings (eta's prior).

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s).
            Eta - site loadings
            sDim - spatial dimension
            alphapw - conditional distribution of each Alpha
            LiWg -
            detWg -
    """

    EtaList = params["Eta"]

    nr = len(EtaList)

    AlphaList = [None] * nr

    for r, (Eta, rLPar) in enumerate(zip(EtaList, rLHyperparams)):
        sDim = rLPar["sDim"]
        nf = tf.cast(tf.shape(Eta)[1], tf.int32)
        if sDim > 0:
            alphapw = rLPar["alphapw"]
            iWg = tfs.to_dense(rLPar["iWg"])
            detWg = rLPar["detWg"]

            # EtaTiWEta = tf.reduce_sum(tf.matmul(LiWg, Eta) ** 2, axis=1)
            EtaTiWEta = tf.einsum("ah,gab,bh->hg", Eta, iWg, Eta)
            logLike = tfm.log(alphapw[:,1]) - 0.5 * detWg - 0.5 * EtaTiWEta
            AlphaList[r] = tf.squeeze(tfr.categorical(logLike, 1, dtype=tf.int32), -1)
        else:
            AlphaList[r] = tf.zeros([nf], dtype=tf.int32)

    return AlphaList
