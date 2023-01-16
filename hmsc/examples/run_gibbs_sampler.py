import numpy as np
import time
import sys
import json

sys.path.append("/Users/anisjyu/Dropbox/hmsc-hpc/hmsc-hpc/")

from random import randint, sample
from datetime import datetime
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

tfd, tfb = tfp.distributions, tfp.bijectors
tfm, tfla, tfr, tfs = tf.math, tf.linalg, tf.random, tf.sparse

from hmsc.gibbs_sampler import GibbsParameter, GibbsSampler

from hmsc.updaters.updateEta import updateEta
from hmsc.updaters.updateAlpha import updateAlpha
from hmsc.updaters.updateBetaLambda import updateBetaLambda
from hmsc.updaters.updateLambdaPriors import updateLambdaPriors
from hmsc.updaters.updateNf import updateNf
from hmsc.updaters.updateGammaV import updateGammaV
from hmsc.updaters.updateSigma import updateSigma
from hmsc.updaters.updateZ import updateZ


def load_params(path):

    with open(path + "obj-complete.json") as json_file:
        obj = json.load(json_file)

    nChains = int(np.squeeze(len(obj["postList"])))

    dtype = np.float64

    ny = int(np.squeeze(obj.get("ny")))  # 50
    ns = int(np.squeeze(obj.get("ns")))  # 4
    nc = int(np.squeeze(obj.get("nc")))  # 3
    nt = int(np.squeeze(obj.get("nt")))  # 3
    nr = int(np.squeeze(obj.get("nr")))  # 2

    nu = np.squeeze([obj.get("rL")[key]["nu"] for key in obj.get("rL").keys()])
    a1 = np.squeeze([obj.get("rL")[key]["a1"] for key in obj.get("rL").keys()])
    b1 = np.squeeze([obj.get("rL")[key]["b1"] for key in obj.get("rL").keys()])
    a2 = np.squeeze([obj.get("rL")[key]["a2"] for key in obj.get("rL").keys()])
    b2 = np.squeeze([obj.get("rL")[key]["b2"] for key in obj.get("rL").keys()])

    nfMin = np.squeeze([obj.get("rL")[key]["nfMin"] for key in obj.get("rL").keys()])
    nfMax = np.squeeze([obj.get("rL")[key]["nfMax"] for key in obj.get("rL").keys()])

    sDim = np.squeeze([obj.get("rL")[key]["sDim"] for key in obj.get("rL").keys()])

    spatialMethod = [
        "".join(obj.get("rL")[key]["spatialMethod"])
        if isinstance(obj.get("rL")[key]["spatialMethod"], list)
        else ""
        for key in obj.get("rL").keys()
    ]

    # alphapw = [obj.get('rL')[key]['alphapw'] for key in obj.get('rL').keys()] # todo
    # alphapw = [None, np.abs(np.random.normal(size=[101, 2]))]
    alphapw = [
        np.abs(np.random.normal(size=[101, 2])),
        np.abs(np.random.normal(size=[101, 2])),
    ]

    distr = np.asarray(obj.get("distr")).astype(int)

    X = np.asarray(obj.get("X"))
    T = np.asarray(obj.get("Tr"))
    Y = np.asarray(obj.get("Y"))

    Pi = np.asarray(obj.get("Pi")).astype(int) - 1
    npVec = Pi.max(axis=0) + 1

    nfVec = 3 + np.arange(nr)

    mGamma = np.asarray(obj.get("mGamma"))
    iUGamma = np.asarray(obj.get("UGamma"))

    aSigma = np.asarray(obj.get("aSigma"))
    bSigma = np.asarray(obj.get("bSigma"))

    V0 = np.squeeze(obj.get("V0"))
    f0 = int(np.squeeze(obj.get("f0")))

    WgList = [tfr.normal([101, npVec[r], npVec[r]], dtype=dtype) for r in range(nr)]
    WgList = [
        tf.matmul(WgList[r], WgList[r], transpose_a=True) for r in range(nr)
    ]  # these MUST be SPD matrices!
    iWgList = [tfla.inv(WgList[r]) for r in range(nr)]
    LiWgList = [tfla.cholesky(iWgList[r]) for r in range(nr)]
    detWgList = [tfr.normal([101], dtype=dtype) for r in range(nr)]

    modelData = {}
    modelData["Y"] = Y
    modelData["X"] = X
    modelData["T"] = T
    modelData["Pi"] = Pi
    modelData["distr"] = distr

    priorHyperParams = {}
    priorHyperParams["mGamma"] = mGamma
    priorHyperParams["iUGamma"] = iUGamma
    priorHyperParams["f0"] = f0
    priorHyperParams["V0"] = V0
    priorHyperParams["aSigma"] = aSigma
    priorHyperParams["bSigma"] = bSigma

    rLDataParams = {}
    rLDataParams["Wg"] = WgList
    rLDataParams["iWg"] = iWgList
    rLDataParams["LiWg"] = LiWgList
    rLDataParams["detWg"] = detWgList

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

    np.random.seed(1)
    tfr.set_seed(1)

    aDeltaList = [
        tf.concat(
            [a1[r] * tf.ones([1, 1], dtype), a2[r] * tf.ones([nfVec[r] - 1, 1], dtype)],
            0,
        )
        for r in range(nr)
    ]
    bDeltaList = [
        tf.concat(
            [b1[r] * tf.ones([1, 1], dtype), b2[r] * tf.ones([nfVec[r] - 1, 1], dtype)],
            0,
        )
        for r in range(nr)
    ]

    Beta = tfr.normal([nc, ns], dtype=dtype)
    Gamma = tfr.normal([nc, nt], dtype=dtype)
    iV = tf.ones([nc, nc], dtype=dtype) + tf.eye(nc, dtype=dtype)
    EtaList = [tfr.normal([npVec[r], nfVec[r]], dtype=dtype) for r in range(nr)]
    PsiList = [1 + tf.abs(tfr.normal([nfVec[r], ns], dtype=dtype)) for r in range(nr)]
    DeltaList = [
        np.random.gamma(aDeltaList[r], bDeltaList[r], size=[nfVec[r], 1])
        for r in range(nr)
    ]
    LambdaList = [tfr.normal([nfVec[r], ns], dtype=dtype) for r in range(nr)]
    AlphaList = [tf.zeros([nfVec[r], 1], dtype=tf.int64) for r in range(nr)]
    Z = tf.zeros_like(Y)

    sigma = tf.abs(tfr.normal([ns], dtype=dtype)) * (distr[:, 1] == 1) + tf.ones(
        [ns], dtype=dtype
    ) * (distr[:, 1] == 0)
    # sigma = tf.ones(ns, dtype=dtype)
    iSigma = 1 / sigma

    model_data = modelData
    prior_params = priorHyperParams
    random_level_data_params = rLDataParams
    random_level_params = rLParams

    sampler_params = {
        "Z": GibbsParameter(Z, updateZ),
        "BetaLambda": GibbsParameter(
            {"Beta": Beta, "Lambda": LambdaList}, updateBetaLambda
        ),
        "GammaV": GibbsParameter({"Gamma": Gamma, "iV": iV}, updateGammaV),
        "PsiDelta": GibbsParameter(
            {"Psi": PsiList, "Delta": DeltaList}, updateLambdaPriors
        ),
        "Eta": GibbsParameter(EtaList, updateEta),
        "sigma": GibbsParameter(sigma, updateSigma),
        "Nf": GibbsParameter(
            {"Eta": EtaList, "Lambda": LambdaList, "Psi": PsiList, "Delta": DeltaList},
            updateNf,
        ),
        "Alpha": GibbsParameter(AlphaList, updateAlpha),
    }

    params = {
        **sampler_params,
        **prior_params,
        **random_level_params,
        **random_level_data_params,
        **model_data,
    }

    return params, nChains


def save_postList(postList, path, nChains):

    json_data = {chain: {} for chain in range(nChains)}

    for chain in range(nChains):
        for i in range(len(postList[chain])):
            sample_data = {}

            sample_data["Beta"] = (
                postList[chain][i]["BetaLambda"]["Beta"].numpy().tolist()
            )
            sample_data["Gamma"] = postList[chain][i]["GammaV"]["iV"].numpy().tolist()

            sample_data["iV"] = postList[chain][i]["GammaV"]["iV"].numpy().tolist()
            sample_data["sigma"] = postList[chain][i]["sigma"].numpy().tolist()
            sample_data["Lambda"] = [
                postList[chain][i]["BetaLambda"]["Lambda"][j].numpy().tolist()
                for j in range(len(postList[chain][i]["BetaLambda"]["Lambda"]))
            ]
            sample_data["Eta"] = [
                postList[chain][i]["Eta"][j].numpy().tolist()
                for j in range(len(postList[chain][i]["Eta"]))
            ]
            sample_data["Psi"] = [
                postList[chain][i]["PsiDelta"]["Psi"][j].numpy().tolist()
                for j in range(len(postList[chain][i]["PsiDelta"]["Psi"]))
            ]
            sample_data["Delta"] = [
                postList[chain][i]["PsiDelta"]["Delta"][j].numpy().tolist()
                for j in range(len(postList[chain][i]["PsiDelta"]["Delta"]))
            ]

            sample_data["Alpha"] = [
                postList[chain][i]["Alpha"][j].numpy().tolist()
                for j in range(len(postList[chain][i]["Alpha"]))
            ]
            postList[chain][i]["BetaLambda"]["Beta"].numpy().tolist()

            sample_data["wRRR"] = sample_data["rho"] = sample_data[
                "PsiRRR"
            ] = sample_data["DeltaRRR"] = None

            json_data[chain][i] = sample_data

    with open(path + "obj-postList.json", "w") as fp:
        json.dump(json_data, fp)


def run_gibbs_sampler(path, save_postList_to_json=False):

    params, nChains = load_params(path)

    gibbs = GibbsSampler(params=params)

    postList = [None] * nChains
    for chain in range(nChains):
        postList[chain] = gibbs.sampling_routine(num_samples=50)

    if save_postList_to_json:
        save_postList(postList, path, nChains)


if __name__ == "__main__":
    print("Whole Gibbs sampler:")

    startTime = time.time()

    # path = "/Users/gtikhono/Downloads/importExport/"
    path = "/users/anisjyu/Documents/demo-import/"

    run_gibbs_sampler(path, True)

    elapsedTime = time.time() - startTime
    print("\nTF decorated whole cycle elapsed %.1f" % elapsedTime)
