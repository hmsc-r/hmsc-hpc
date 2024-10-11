import numpy as np
import tensorflow as tf
from hmsc.utils.tf_named_func import tf_named_func
from hmsc.utils.import_utils import calculate_idDW12, set_slice
tfm, tfla, tfr, tfs = tf.math, tf.linalg, tf.random, tf.sparse

@tf_named_func("alpha")
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

            if spatialMethod == "Full":
                detWg = rLPar["detWg"]
                iWg = rLPar["iWg"]
                EtaTiWEta = tf.einsum("ah,gab,bh->hg", Eta, iWg, Eta)
                logLike = tfm.log(alphapw[:,1]) - 0.5 * detWg - 0.5 * EtaTiWEta
            
            elif spatialMethod == "GPP":
                # dD is matrix of added diagonal variance, D is approximated matrix replacing spatial covariance
                detDg = rLPar["detDg"]
                iFg = rLPar["iFg"]
                idDg = rLPar["idDg"]
                EtaT_idD_Eta = tf.einsum("ih,gi,ih->hg", Eta, idDg, Eta, name="EtaT_idD_Eta")

                if "idDW12g" in rLPar:
                    idDW12g = rLPar["idDW12g"]
                    # W21idD_Eta = tf.einsum("gia,ih->gah", idDW12g, Eta, name="W21idD_Eta")
                    W21idD_Eta = tf.matmul(idDW12g, Eta, transpose_a=True, name="W21idD_Eta")
                else: # lowmem
                    var = rLPar["W21idD_Eta_var"]
                    var.assign(tf.zeros(shape=[*var.shape[:-1], tf.shape(Eta)[1]], dtype=dtype))
                    cond = lambda i: tf.less(i, rLPar["alpha"].shape[0])
                    def body(i):
                        idDW12 = calculate_idDW12(rLPar["d12"], rLPar["alpha"][i], idDg[i])
                        tmp = tf.matmul(idDW12, Eta, transpose_a=True, name="W21idD_Eta")
                        set_slice(var, i, tmp)
                        return [i + 1, ]

                    i = tf.constant(0)
                    tf.while_loop(cond, body, [i])

                    W21idD_Eta = var.read_value_no_copy()

                EtaT_idDW21_iF_W21idD_Eta  = tf.einsum("gah,gab,gbh->hg", W21idD_Eta, iFg, W21idD_Eta, name="EtaT_idDW21_iF_W21idD_Eta")
                logLike = tfm.log(alphapw[:,1]) - 0.5 * detDg - 0.5 * (EtaT_idD_Eta - EtaT_idDW21_iF_W21idD_Eta)

            elif spatialMethod == "NNGP":
                detWg = rLPar["detWg"]
                RiWList = rLPar["RiWList"]
                RiWg_Eta = tf.stack([tfs.sparse_dense_matmul(RiW, Eta) for RiW in RiWList], 0)
                EtaTiWEta = tf.transpose(tf.reduce_sum(RiWg_Eta**2, [-2]))
                logLike = tfm.log(alphapw[:,1]) - 0.5 * detWg - 0.5 * EtaTiWEta

            AlphaList[r] = tf.squeeze(tfr.categorical(logLike, 1, dtype=tf.int32), -1)
        else:
            AlphaList[r] = tf.zeros([nf], dtype=tf.int32)
    # print(AlphaList)
    return AlphaList
