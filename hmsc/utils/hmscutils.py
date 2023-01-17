import numpy as np
import tensorflow as tf

tfla, tfr = tf.linalg, tf.random

def load_model_data(hmscModel):

    Y = np.asarray(hmscModel.get("Y"))
    X = np.asarray(hmscModel.get("X"))
    T = np.asarray(hmscModel.get("Tr"))
    Pi = np.asarray(hmscModel.get("Pi")).astype(int) - 1
    distr = np.asarray(hmscModel.get("distr")).astype(int)

    modelData = {}
    modelData["Y"] = Y
    modelData["X"] = X
    modelData["T"] = T
    modelData["Pi"] = Pi
    modelData["distr"] = distr

    return modelData

def load_model_data_params(hmscModel):

    ny = int(np.squeeze(hmscModel.get("ny")))
    ns = int(np.squeeze(hmscModel.get("ns")))
    nc = int(np.squeeze(hmscModel.get("nc")))
    nt = int(np.squeeze(hmscModel.get("nt")))
    nr = int(np.squeeze(hmscModel.get("nr")))

    modelDataParams = {}
    modelDataParams["ny"] = ny
    modelDataParams["ns"] = ns
    modelDataParams["nc"] = nc
    modelDataParams["nt"] = nt
    modelDataParams["nr"] = nr

    return modelDataParams


def load_random_level_params(hmscModel):

    nu = np.squeeze(
        [hmscModel.get("rL")[key]["nu"] for key in hmscModel.get("rL").keys()]
    )
    a1 = np.squeeze(
        [hmscModel.get("rL")[key]["a1"] for key in hmscModel.get("rL").keys()]
    )
    b1 = np.squeeze(
        [hmscModel.get("rL")[key]["b1"] for key in hmscModel.get("rL").keys()]
    )
    a2 = np.squeeze(
        [hmscModel.get("rL")[key]["a2"] for key in hmscModel.get("rL").keys()]
    )
    b2 = np.squeeze(
        [hmscModel.get("rL")[key]["b2"] for key in hmscModel.get("rL").keys()]
    )

    nfMin = np.squeeze(
        [hmscModel.get("rL")[key]["nfMin"] for key in hmscModel.get("rL").keys()]
    )
    nfMax = np.squeeze(
        [hmscModel.get("rL")[key]["nfMax"] for key in hmscModel.get("rL").keys()]
    )

    sDim = np.squeeze(
        [hmscModel.get("rL")[key]["sDim"] for key in hmscModel.get("rL").keys()]
    )

    spatialMethod = [
        "".join(hmscModel.get("rL")[key]["spatialMethod"])
        if isinstance(hmscModel.get("rL")[key]["spatialMethod"], list)
        else ""
        for key in hmscModel.get("rL").keys()
    ]

    alphapw = [
        np.abs(np.random.normal(size=[101, 2])),
        np.abs(np.random.normal(size=[101, 2])),
    ]

    rLParams = {}
    rLParams["nu"] = nu
    rLParams["a1"] = a1
    rLParams["b1"] = b1
    rLParams["a2"] = a2
    rLParams["b2"] = b2
    rLParams["nfMin"] = nfMin
    rLParams["nfMax"] = nfMax
    rLParams["sDim"] = sDim
    rLParams["spatialMethod"] = spatialMethod
    rLParams["alphapw"] = alphapw

    return rLParams


def init_random_level_data_params(modelDataParams, modelData, dtype=np.float64):

    nr = modelDataParams["nr"]
    Pi = modelData["Pi"]

    npVec = Pi.max(axis=0) + 1

    WgList = [tfr.normal([101, npVec[r], npVec[r]], dtype=dtype) for r in range(nr)]
    WgList = [
        tf.matmul(WgList[r], WgList[r], transpose_a=True) for r in range(nr)
    ]  # these MUST be SPD matrices!
    iWgList = [tfla.inv(WgList[r]) for r in range(nr)]
    LiWgList = [tfla.cholesky(iWgList[r]) for r in range(nr)]
    detWgList = [tfr.normal([101], dtype=dtype) for r in range(nr)]

    rLDataParams = {}
    rLDataParams["Wg"] = WgList
    rLDataParams["iWg"] = iWgList
    rLDataParams["LiWg"] = LiWgList
    rLDataParams["detWg"] = detWgList

    return rLDataParams


def load_prior_hyper_params(hmscModel):

    mGamma = np.asarray(hmscModel.get("mGamma"))
    iUGamma = np.asarray(hmscModel.get("UGamma"))

    aSigma = np.asarray(hmscModel.get("aSigma"))
    bSigma = np.asarray(hmscModel.get("bSigma"))

    V0 = np.squeeze(hmscModel.get("V0"))
    f0 = int(np.squeeze(hmscModel.get("f0")))

    priorHyperParams = {}
    priorHyperParams["mGamma"] = mGamma
    priorHyperParams["iUGamma"] = iUGamma
    priorHyperParams["f0"] = f0
    priorHyperParams["V0"] = V0
    priorHyperParams["aSigma"] = aSigma
    priorHyperParams["bSigma"] = bSigma

    return priorHyperParams
