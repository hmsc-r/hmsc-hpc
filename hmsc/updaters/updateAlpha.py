import numpy as np
import tensorflow as tf
tfm, tfla, tfr = tf.math, tf.linalg, tf.random

def updateAlpha(params, dtype=np.float64):
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

    sDim = params["sDim"]
    alphapwList = params["alphapw"]

    LiWgList = params["LiWg"]
    detWgList = params["detWg"]

    nr = len(EtaList)
    AlphaList = [None] * nr
    for r, (Eta, LiWg, detWg, alphapw) in enumerate(
        zip(EtaList, LiWgList, detWgList, alphapwList)
    ):
        np = Eta.shape[0]
        nf = tf.cast(tf.shape(Eta)[1], tf.int32)
        if sDim[r] > 0:
            EtaTiWEta = tf.reduce_sum(tf.matmul(LiWg, Eta) ** 2, axis=1)
            logLike = (
                tfm.log(alphapw[:, 1])
                - 0.5 * detWg
                - 0.5 * tfla.matrix_transpose(EtaTiWEta)
            )
            like = tfm.exp(
                logLike - tf.math.reduce_logsumexp(logLike, axis=-1, keepdims=True)
            )
            AlphaList[r] = tfr.categorical(like, 1, dtype=tf.int64)
        else:
            AlphaList[r] = tf.zeros([nf, 1], tf.int64)

    return AlphaList
