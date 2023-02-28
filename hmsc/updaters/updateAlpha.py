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
            spatialMethod = rLPar["spatialMethod"]
            alphapw = rLPar["alphapw"]

            if spatialMethod == "GPP":
                detDg = rLPar["detDg"]
                iFg = rLPar["iFg"]
                idDg = rLPar["idDg"]
                idDW12g = rLPar["idDW12g"]

                #tmpMat1 = tf.einsum("ji,lkj->lik", Eta, idDW12g)
                #tmpMat2 = tf.einsum("lij,ljk->lik", tmpMat1, iFg)
                #tmpMat3 = tf.einsum("lij,lkj->lik", tmpMat2, tmpMat1)
                EtaTidDEta = tf.einsum("ij,ik,ij->k", Eta, idDg, Eta)
                logLike = tfm.log(alphapw[:,1]) - 0.5 * detDg - 0.5 * EtaTidDEta
                AlphaList[r] = tf.squeeze(tfr.categorical([logLike], nf, dtype=tf.int32))

            elif spatialMethod == "NNGP":
                detWg = rLPar["detWg"]
                iWg = tfs.to_dense(rLPar["iWg"])
                # EtaTiWEta = tf.reduce_sum(tf.matmul(LiWg, Eta) ** 2, axis=1)
                EtaTiWEta = tf.einsum("ah,gab,bh->hg", Eta, iWg, Eta)
                logLike = tfm.log(alphapw[:,1]) - 0.5 * detWg - 0.5 * EtaTiWEta
                AlphaList[r] = tf.squeeze(tfr.categorical(logLike, nf, dtype=tf.int32))

            else:
                detWg = rLPar["detWg"]
                iWg = rLPar["iWg"]
                # EtaTiWEta = tf.reduce_sum(tf.matmul(LiWg, Eta) ** 2, axis=1)
                EtaTiWEta = tf.einsum("ah,gab,bh->hg", Eta, iWg, Eta)
                logLike = tfm.log(alphapw[:,1]) - 0.5 * detWg - 0.5 * EtaTiWEta
                AlphaList[r] = tf.squeeze(tfr.categorical(logLike, nf, dtype=tf.int32))
        else:
            AlphaList[r] = tf.zeros([nf], dtype=tf.int32)

    return AlphaList
