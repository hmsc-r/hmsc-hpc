'''
import numpy as np
import time
import sys
from random import randint, sample
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

tfd, tfb = tfp.distributions, tfp.bijectors
tfm, tfla, tfr, tfs = tf.math, tf.linalg, tf.random, tf.sparse

from hmsc.gibbs_sampler import GibbsParameter, GibbsSampler

from updaters.updateEta import updateEta
from updaters.updateAlpha import updateAlpha
from updaters.updateBetaLambda import updateBetaLambda
from updaters.updateLambdaPriors import updateLambdaPriors
from updaters.updateNf import updateNf
from updaters.updateGammaV import updateGammaV
from updaters.updateSigma import updateSigma
from updaters.updateZ import updateZ

import json

def load_params():
    # path = "/Users/gtikhono/Downloads/importExport/"
    # path = "/users/anisjyu/Documents/demo-import/"

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
            [a1[r] * tf.ones([1, 1], dtype), a2[r] * tf.ones([nfVec[r] - 1, 1], dtype)], 0
        )
        for r in range(nr)
    ]
    bDeltaList = [
        tf.concat(
            [b1[r] * tf.ones([1, 1], dtype), b2[r] * tf.ones([nfVec[r] - 1, 1], dtype)], 0
        )
        for r in range(nr)
    ]

    Beta = tfr.normal([nc, ns], dtype=dtype)
    Gamma = tfr.normal([nc, nt], dtype=dtype)
    iV = tf.ones([nc, nc], dtype=dtype) + tf.eye(nc, dtype=dtype)
    EtaList = [tfr.normal([npVec[r], nfVec[r]], dtype=dtype) for r in range(nr)]
    PsiList = [1 + tf.abs(tfr.normal([nfVec[r], ns], dtype=dtype)) for r in range(nr)]
    DeltaList = [
        np.random.gamma(aDeltaList[r], bDeltaList[r], size=[nfVec[r], 1]) for r in range(nr)
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

    params={
        **sampler_params,
        **prior_params,
        **random_level_params,
        **random_level_data_params,
        **model_data,
    }

    return params

def run_sampler():
    params = load_params()

    gibbs = GibbsSampler(params=params)

    postList = [None] * nChains
    for chain in range(nChains):
        postList[chain] = gibbs.sampling_routine(3)

if __name__ == "__main__":
    print("Whole Gibbs sampler:")

    startTime = time.time()

    run_sampler()

    elapsedTime = time.time() - startTime
    print("\nTF decorated whole cycle elapsed %.1f" % elapsedTime)

'''