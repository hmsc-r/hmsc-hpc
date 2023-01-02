import numpy as np
import tensorflow as tf
tfm, tfr = tf.math, tf.random

def updateNf(params, dtype=np.float64):
    """Update latent factors:
    Eta - site loadings,
    Lambda - species loadings,
    Psi - local shrinage species loadings (lambda's prior), and
    Delta - delta global shrinage species loadings (lambda's prior).
        
    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Eta - site loadings
        Beta - species niches
        Psi - local shrinage species loadings (lambda's prior)
        Delta - delta global shrinage species loadings (lambda's prior)
        nu - 
        a2 - 
        b2 - 
        nfMin -
        nfMax - 
        iter -
    """

    EtaList = params["Eta"]
    LambdaList = params["BetaLambda"]["Lambda"]
    PsiList = params["PsiDelta"]["Psi"]
    DeltaList = params["PsiDelta"]["Delta"]

    # iter ???
    iter = 1

    nu = params["nu"]
    a2 = params["a2"]
    b2 = params["b2"]
    nfMin = params["nfMin"]
    nfMax = params["nfMax"]

    c0 = 1
    c1 = 0.0005
    epsilon = 1e-3  # threshold limit
    prop = 1.00  # proportion of redundant elements within columns
    prob = 1 / tf.exp(c0 + c1 * tf.cast(iter, dtype))  # probability of adapting

    nr = len(LambdaList)
    EtaNew, LambdaNew, PsiNew, DeltaNew = [[None] * nr for i in range(4)]
    for r, (Eta, Lambda, Psi, Delta) in enumerate(
        zip(EtaList, LambdaList, PsiList, DeltaList)
    ):
        if tfr.uniform([], dtype=dtype) < prob:
            nf = tf.shape(Lambda)[0]
            _, ns = Lambda.shape
            np = tf.shape(Eta)[0]
            smallLoadingProp = tf.reduce_mean(
                tf.cast(tfm.abs(Lambda) < epsilon, dtype=dtype), axis=1
            )
            indRedundant = smallLoadingProp >= prop
            numRedundant = tf.reduce_sum(tf.cast(indRedundant, dtype=dtype))

            if (
                nf < nfMax[r] and iter > 20 and numRedundant == 0
            ):  # and tf.reduce_all(smallLoadingProp < 0.995):
                EtaNew[r] = tf.concat([Eta, tfr.normal([np, 1], dtype=dtype)], axis=1)
                LambdaNew[r] = tf.concat(
                    [Lambda, tf.zeros([1, ns], dtype=dtype)], axis=0
                )
                PsiNew[r] = tf.concat(
                    [Psi, tfr.gamma([1, ns], nu[r] / 2, nu[r] / 2, dtype=dtype)], axis=0
                )
                DeltaNew[r] = tf.concat(
                    [Delta, tfr.gamma([1, 1], a2[r], b2[r], dtype=dtype)], axis=0
                )
            elif nf > nfMin[r] and numRedundant > 0:
                indRemain = tf.cast(
                    tf.squeeze(tf.where(tfm.logical_not(indRedundant)), -1), tf.int32
                )
                if tf.shape(indRemain)[0] < nfMin[r]:
                    indRemain = tf.concat(
                        [
                            indRemain,
                            nf - 1 - tf.range(nfMin[r] - tf.shape(indRemain)[0]),
                        ],
                        axis=0,
                    )
                EtaNew[r] = tf.gather(Eta, indRemain, axis=1)
                LambdaNew[r] = tf.gather(Lambda, indRemain, axis=0)
                PsiNew[r] = tf.gather(Psi, indRemain, axis=0)
                DeltaNew[r] = tf.gather(Delta, indRemain, axis=0)
            else:
                EtaNew[r], LambdaNew[r], PsiNew[r], DeltaNew[r] = (
                    Eta,
                    Lambda,
                    Psi,
                    Delta,
                )
        else:
            EtaNew[r], LambdaNew[r], PsiNew[r], DeltaNew[r] = Eta, Lambda, Psi, Delta
    return {"Eta": EtaNew, "Lambda": LambdaNew, "Psi": PsiNew, "Delta": DeltaNew}
