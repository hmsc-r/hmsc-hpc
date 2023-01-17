import numpy as np
import tensorflow as tf

tfla, tfr = tf.linalg, tf.random


def load_model_params(hmsc_model):

    ny = int(np.squeeze(hmsc_model.get("ny")))  # 50
    ns = int(np.squeeze(hmsc_model.get("ns")))  # 4
    nc = int(np.squeeze(hmsc_model.get("nc")))  # 3
    nt = int(np.squeeze(hmsc_model.get("nt")))  # 3
    nr = int(np.squeeze(hmsc_model.get("nr")))  # 2

    Y = np.asarray(hmsc_model.get("Y"))
    X = np.asarray(hmsc_model.get("X"))
    T = np.asarray(hmsc_model.get("Tr"))
    Pi = np.asarray(hmsc_model.get("Pi")).astype(int) - 1
    distr = np.asarray(hmsc_model.get("distr")).astype(int)

    nChains = int(np.squeeze(len(hmsc_model["postList"])))

    modelData = {}
    modelData["Y"] = Y
    modelData["X"] = X
    modelData["T"] = T
    modelData["Pi"] = Pi
    modelData["distr"] = distr

    modelDataParams = {}
    modelDataParams["ny"] = ny
    modelDataParams["ns"] = ns
    modelDataParams["nc"] = nc
    modelDataParams["nt"] = nt
    modelDataParams["nr"] = nr

    modelData = {}
    modelData["Y"] = Y
    modelData["X"] = X
    modelData["T"] = T
    modelData["Pi"] = Pi
    modelData["distr"] = distr

    return modelDataParams, modelData, nChains


def load_random_level_params(hmsc_model):

    nu = np.squeeze(
        [hmsc_model.get("rL")[key]["nu"] for key in hmsc_model.get("rL").keys()]
    )
    a1 = np.squeeze(
        [hmsc_model.get("rL")[key]["a1"] for key in hmsc_model.get("rL").keys()]
    )
    b1 = np.squeeze(
        [hmsc_model.get("rL")[key]["b1"] for key in hmsc_model.get("rL").keys()]
    )
    a2 = np.squeeze(
        [hmsc_model.get("rL")[key]["a2"] for key in hmsc_model.get("rL").keys()]
    )
    b2 = np.squeeze(
        [hmsc_model.get("rL")[key]["b2"] for key in hmsc_model.get("rL").keys()]
    )

    nfMin = np.squeeze(
        [hmsc_model.get("rL")[key]["nfMin"] for key in hmsc_model.get("rL").keys()]
    )
    nfMax = np.squeeze(
        [hmsc_model.get("rL")[key]["nfMax"] for key in hmsc_model.get("rL").keys()]
    )

    sDim = np.squeeze(
        [hmsc_model.get("rL")[key]["sDim"] for key in hmsc_model.get("rL").keys()]
    )

    spatialMethod = [
        "".join(hmsc_model.get("rL")[key]["spatialMethod"])
        if isinstance(hmsc_model.get("rL")[key]["spatialMethod"], list)
        else ""
        for key in hmsc_model.get("rL").keys()
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


def load_prior_hyper_params(hmsc_model):

    mGamma = np.asarray(hmsc_model.get("mGamma"))
    iUGamma = np.asarray(hmsc_model.get("UGamma"))

    aSigma = np.asarray(hmsc_model.get("aSigma"))
    bSigma = np.asarray(hmsc_model.get("bSigma"))

    V0 = np.squeeze(hmsc_model.get("V0"))
    f0 = int(np.squeeze(hmsc_model.get("f0")))

    priorHyperParams = {}
    priorHyperParams["mGamma"] = mGamma
    priorHyperParams["iUGamma"] = iUGamma
    priorHyperParams["f0"] = f0
    priorHyperParams["V0"] = V0
    priorHyperParams["aSigma"] = aSigma
    priorHyperParams["bSigma"] = bSigma

    return priorHyperParams
