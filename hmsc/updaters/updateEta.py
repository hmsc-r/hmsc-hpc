import numpy as np
import tensorflow as tf

tfla, tfr, tfs = tf.linalg, tf.random, tf.sparse

from hmsc.utils.tflautils import kron, scipy_cholesky


def updateEta(params, data, modelDims, rLHyperparams, dtype=np.float64):
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
    Z = params["Z"]
    sigma = params["sigma"]
    Beta = params["Beta"]
    EtaList = params["Eta"]
    LambdaList = params["Lambda"]
    AlphaList = params["Alpha"]

    Pi = data["Pi"]
    X = data["X"]

    npVec = modelDims["np"]

    ny, _ = X.shape
    nr = len(LambdaList)

    iSigma = 1 / sigma

    LFix = tf.matmul(X, Beta)

    LRanLevelList = [None] * nr
    for r, (Eta, Lambda) in enumerate(zip(EtaList, LambdaList)):
        LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:, r]), Lambda)

    iD = tf.ones_like(Z) * sigma**-2

    EtaListNew = [None] * nr

    for r, (Eta, Lambda, Alpha, rLPar) in enumerate(
        zip(EtaList, LambdaList, AlphaList, rLHyperparams)
    ):
        sDim = rLPar["sDim"]

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

        if sDim > 0:

            spatialMethod = rLPar["spatialMethod"]
            iWg = tf.cast(rLPar["iWg"], dtype=dtype)

            if spatialMethod == "NNGP":
                Eta = modelSpatialNNGP(
                    Eta,
                    LamInvSigLam,
                    mu0,
                    Alpha,
                    Pi[:, r],
                    iWg,
                    S,
                    iSigma,
                    npVec[r],
                    nf,
                    ny,
                )
            elif spatialMethod == "GPP":
                raise NotImplementedError
            else:
                Eta = modelSpatialFull(Eta, LamInvSigLam, mu0, Alpha, iWg, npVec[r], nf)

        else:
            Eta = modelNonSpatial(Eta, LamInvSigLam, mu0, npVec[r], nf)

        EtaListNew[r] = Eta

        LRanLevelList[r] = tf.matmul(tf.gather(EtaListNew[r], Pi[:, r]), Lambda)

    return EtaListNew


def modelSpatialFull(Eta, LamInvSigLam, mu0, Alpha, iWg, np, nf, dtype=np.float64):
    iWs = tf.reshape(
        tf.transpose(
            tfla.diag(tf.transpose(tf.gather(iWg, tf.cast(tf.squeeze(Alpha), dtype=tf.int64)), [1, 2, 0])),
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


def modelNonSpatial(Eta, LamInvSigLam, mu0, np, nf, dtype=np.float64):
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


def modelSpatialNNGP(
    Eta, LamInvSigLam, mu0, Alpha, Pi, iWg, S, iSigma, np, nf, ny, dtype=np.float64
):
    iWs = tf.zeros([np * nf, np * nf], dtype=dtype)

    for h in range(nf):
        iWs = iWs + kron(
            tf.gather(iWg, tf.squeeze(Alpha[h], -1)),
            tf.linalg.diag(tf.cast(tf.one_hot(h, tf.cast(nf, tf.int32)), dtype)),
        )

    P = tfs.SparseTensor(
        tf.cast(tf.stack([tf.range(ny), Pi], axis=1), tf.int64),
        tf.ones([ny], dtype=dtype),
        [ny, np],
    )

    fS = tf.matmul(
        tf.matmul(tfs.to_dense(P), S, transpose_a=True),
        tf.reshape(tf.tile(iSigma, [nf]), [nf, len(iSigma)]),
        transpose_b=True,
    )

    iUEta = iWs + kron(
        LamInvSigLam[0, :, :],
        tf.cast(tfla.diag(tfs.reduce_sum(P, axis=0)), dtype=dtype),
    )

    LiUEta = tf.numpy_function(scipy_cholesky, [iUEta], tf.float64)

    mu1 = tfla.triangular_solve(LiUEta, tf.reshape(tf.transpose(mu0), [nf * np, 1]))

    eta = tfla.triangular_solve(
        LiUEta, mu1 + tfr.normal([nf * np, 1], dtype=dtype), adjoint=True
    )

    Eta = tf.transpose(tf.reshape(eta, [nf, np]))

    return Eta
