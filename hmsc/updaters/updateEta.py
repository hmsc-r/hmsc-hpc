import numpy as np
import tensorflow as tf
tfla, tfr = tf.linalg, tf.random

def updateEta(params, dtype=np.float64):
    """Update conditional updater(s):
    Z - site loadings.
        
    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Z - latent variables
        Beta - species niches
        Eta - site loadings
        Lambda - species loadings
        Alpha - scale of site loadings (eta's prior)
        sigma - residual variance
        X - environmental data
        Pi - study design
        iWg - ??
        sDim - ??
    """

    sigma = params["sigma"]

    sDim = params["sDim"]
    Pi = params["Pi"]

    Z = params["Z"]
    Beta = params["BetaLambda"]["Beta"]
    EtaList = params["Eta"]
    LambdaList = params["BetaLambda"]["Lambda"]
    AlphaList = params["Alpha"]
    X = params["X"]
    iWgList = params["iWg"]

    npVec = tf.reduce_max(Pi, 0) + 1

    nr = len(LambdaList)

    LFix = tf.matmul(X, Beta)

    LRanLevelList = [None] * nr
    for r, (Eta, Lambda) in enumerate(zip(EtaList, LambdaList)):
        LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:, r]), Lambda)

    iD = tf.ones_like(Z) * sigma**-2

    EtaListNew = [None] * nr

    for r, (Eta, Lambda, Alpha, iWg) in enumerate(
        zip(EtaList, LambdaList, AlphaList, iWgList)
    ):

        S = (
            Z
            - LFix
            - sum([LRanLevelList[rInd] for rInd in np.setdiff1d(np.arange(nr), r)])
        )

        nf = tf.cast(tf.shape(Lambda)[-2], tf.int64)

        LamInvSigLam = tf.scatter_nd(
            Pi[:, r, None],
            tf.einsum("hj,ij,kj->ihk", Lambda, iD, Lambda),
            tf.stack([npVec[r], nf, nf]),
        )

        mu0 = tf.scatter_nd(
            Pi[:, r, None],
            tf.matmul(iD * S, Lambda, transpose_b=True),
            tf.stack([npVec[r], nf]),
        )

        if sDim[r] > 0:
            Eta = modelSpatialFull(
                Eta, Lambda, LamInvSigLam, mu0, Alpha, iWg, npVec[r], nf
            )
        else:
            Eta = modelNonSpatial(Eta, Lambda, LamInvSigLam, mu0, npVec[r], nf)

        EtaListNew[r] = Eta

        LRanLevelList[r] = tf.matmul(tf.gather(EtaListNew[r], Pi[:, r]), Lambda)

    return EtaListNew


def modelSpatialFull(
    Eta, Lambda, LamInvSigLam, mu0, Alpha, iWg, np, nf, dtype=np.float64
):
    iWs = tf.reshape(
        tf.transpose(
            tfla.diag(tf.transpose(tf.gather(iWg, tf.squeeze(Alpha, -1)), [1, 2, 0])),
            [2, 0, 3, 1],
        ),
        [nf * np, nf * np],
    )
    iUEta = iWs + tf.reshape(
        tf.transpose(tfla.diag(tf.transpose(LamInvSigLam, [1, 2, 0])), [0, 2, 1, 3]),
        [nf * np, nf * np],
    )
    LiUEta = tfla.cholesky(iUEta)
    mu1 = tfla.triangular_solve(LiUEta, tf.reshape(tf.transpose(mu0), [nf * np, 1]))
    eta = tfla.triangular_solve(
        LiUEta, mu1 + tfr.normal([nf * np, 1], dtype=dtype), adjoint=True
    )
    Eta = tf.transpose(tf.reshape(eta, [nf, np]))
    return Eta


def modelNonSpatial(Eta, Lambda, LamInvSigLam, mu0, np, nf, dtype=np.float64):
    iV = tf.eye(nf, dtype=dtype) + LamInvSigLam
    LiV = tfla.cholesky(iV + tf.eye(nf, dtype=dtype))
    mu1 = tfla.triangular_solve(LiV, tf.expand_dims(mu0, -1))
    Eta = tf.squeeze(
        tfla.triangular_solve(
            LiV, mu1 + tfr.normal([np, nf, 1], dtype=dtype), adjoint=True
        ),
        -1,
    )
    return Eta
