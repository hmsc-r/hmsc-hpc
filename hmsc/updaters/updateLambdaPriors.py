import numpy as np
import tensorflow as tf
tfm, tfr = tf.math, tf.random

def updateLambdaPriors(params, dtype=np.float64):
    """Update prior(s) for each random level:
    Psi - local shrinage species loadings (lambda's prior), and
    Delta - delta global shrinage species loadings (lambda's prior).
       
    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Lambda - species loadings
        Delta - delta global shrinage species loadings (lambda's prior)
        nu - 
        a1 - 
        b1 - 
        a2 - 
        b2 - 
    """

    LambdaList = params["BetaLambda"]["Lambda"]
    DeltaList = params["PsiDelta"]["Delta"]

    nu = params["nu"]
    a1 = params["a1"]
    b1 = params["b1"]
    a2 = params["a2"]
    b2 = params["b2"]

    nr = len(LambdaList)
    PsiNew, DeltaNew = [None] * nr, [None] * nr
    for r, (Lambda, Delta) in enumerate(zip(LambdaList, DeltaList)):
        ns = Lambda.shape[-1]
        nf = tf.shape(Lambda)[0]
        if nf > 0:
            aDelta = tf.concat(
                [a1[r] * tf.ones([1, 1], dtype), a2[r] * tf.ones([nf - 1, 1], dtype)], 0
            )
            bDelta = tf.concat(
                [b1[r] * tf.ones([1, 1], dtype), b2[r] * tf.ones([nf - 1, 1], dtype)], 0
            )
            Lambda2 = Lambda**2
            Tau = tfm.cumprod(Delta, 0)
            aPsi = nu[r] / 2.0 + 0.5
            bPsi = nu[r] / 2.0 + Lambda2 * Tau
            PsiNew[r] = tf.squeeze(tfr.gamma([1], aPsi, bPsi, dtype=dtype), 0)
            M = PsiNew[r] * Lambda2
            rowSumM = tf.reduce_sum(M, 1)
            DeltaNew[r] = Delta
            for h in range(nf):
                Tau = tfm.cumprod(DeltaNew[r], 0)
                ad = aDelta[h, :] + 0.5 * ns * tf.cast(nf - h, dtype)
                bd = (
                    bDelta[h, :]
                    + 0.5
                    * tf.reduce_sum(Tau[h:, :] * rowSumM[h:, None], 0)
                    / DeltaNew[r][h, :]
                )
                DeltaNew[r] = tf.tensor_scatter_nd_update(
                    DeltaNew[r], [[h]], tfr.gamma([1], ad, bd, dtype=dtype)
                )
        else:
            PsiNew[r] = tf.zeros([0, ns], dtype)
            DeltaNew[r] = tf.zeros([0, 1], dtype)
    return {"Psi": PsiNew, "Delta": DeltaNew}
