import numpy as np
import tensorflow as tf

tfm, tfr = tf.math, tf.random


def updateNf(params, rLHyperparams, dtype=np.float64):
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
        a2 - prior parameters
        b2 -
        nfMin -
        nfMax -
        iter -
    """

    EtaList = params["Eta"]
    LambdaList = params["Lambda"]
    PsiList = params["Psi"]
    DeltaList = params["Delta"]

    # iter ???
    iter = 1

    c0 = 1
    c1 = 0.0005
    epsilon = 1e-3  # threshold limit
    prop = 1.00  # proportion of redundant elements within columns
    prob = 1 / tf.exp(c0 + c1 * tf.cast(iter, dtype))  # probability of adapting

    nr = len(LambdaList)
    EtaNew, LambdaNew, PsiNew, DeltaNew = [[None] * nr for i in range(4)]
    for r, (Eta, Lambda, Psi, Delta, rLPar) in enumerate(
        zip(EtaList, LambdaList, PsiList, DeltaList, rLHyperparams)
    ):

        nu = rLPar["nu"]
        a2 = rLPar["a2"]
        b2 = rLPar["b2"]
        nfMin = rLPar["nfMin"]
        nfMax = rLPar["nfMax"]

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
                nf < tf.cast(nfMax, tf.int32) and iter > 20 and numRedundant == 0
            ):  # and tf.reduce_all(smallLoadingProp < 0.995):
                EtaNew[r] = tf.concat([Eta, tfr.normal([np, 1], dtype=dtype)], axis=1)
                LambdaNew[r] = tf.concat(
                    [Lambda, tf.zeros([1, ns], dtype=dtype)], axis=0
                )
                PsiNew[r] = tf.concat(
                    [Psi, tfr.gamma([1, ns], nu / 2, nu / 2, dtype=dtype)], axis=0
                )
                DeltaNew[r] = tf.concat(
                    [Delta, tfr.gamma([1, 1], a2, b2, dtype=dtype)], axis=0
                )
            elif nf > tf.cast(nfMin, tf.int32) and numRedundant > 0:
                indRemain = tf.cast(
                    tf.squeeze(tf.where(tfm.logical_not(indRedundant)), -1), tf.int32
                )
                if tf.shape(indRemain)[0] < tf.cast(nfMin, tf.int32):
                    indRemain = tf.concat(
                        [
                            indRemain,
                            nf
                            - 1
                            - tf.range(
                                tf.cast(nfMin, tf.int32) - tf.shape(indRemain)[0]
                            ),
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
