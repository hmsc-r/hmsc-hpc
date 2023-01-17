import numpy as np
import time
import sys

sys.path.append("/Users/anisjyu/Dropbox/hmsc-hpc/hmsc-hpc/")

from random import randint, sample
from datetime import datetime
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

tfr = tf.random

from hmsc.gibbs_sampler import GibbsParameter, GibbsSampler
from hmsc.updaters.updateEta import updateEta
from hmsc.updaters.updateAlpha import updateAlpha
from hmsc.updaters.updateBetaLambda import updateBetaLambda
from hmsc.updaters.updateLambdaPriors import updateLambdaPriors
from hmsc.updaters.updateNf import updateNf
from hmsc.updaters.updateGammaV import updateGammaV
from hmsc.updaters.updateSigma import updateSigma
from hmsc.updaters.updateZ import updateZ

from hmsc.utils.jsonutils import load_model_from_json, save_postList_to_json
from hmsc.utils.hmscutils import (
    load_model_data_params,
    load_model_data,
    load_prior_hyper_params,
    load_random_level_params,
    init_random_level_data_params,
)


def init_sampler_params(modelDataParams, modelData, rLParams, dtype=np.float64):

    ns = modelDataParams["ns"]
    nc = modelDataParams["nc"]
    nt = modelDataParams["nt"]
    nr = modelDataParams["nr"]

    Y = modelData["Y"]
    Pi = modelData["Pi"]
    distr = modelData["distr"]

    a1 = rLParams["a1"]
    a2 = rLParams["a2"]
    b1 = rLParams["b1"]
    b2 = rLParams["b2"]

    npVec = Pi.max(axis=0) + 1
    nfVec = 3 + np.arange(nr)

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

    return Z, Beta, LambdaList, Gamma, iV, PsiList, DeltaList, EtaList, AlphaList, sigma


def build_sampler(modelDataParams, modelData, rLParams, dtype=np.float64):

    (
        Z,
        Beta,
        LambdaList,
        Gamma,
        iV,
        PsiList,
        DeltaList,
        EtaList,
        AlphaList,
        sigma,
    ) = init_sampler_params(modelDataParams, modelData, rLParams)

    samplerParams = {
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
    return samplerParams


def load_params(file_path, dtype=np.float64):

    hmscModel = load_model_from_json(file_path)

    modelDataParams = load_model_data_params(hmscModel)
    modelData = load_model_data(hmscModel)
    priorHyperParams = load_prior_hyper_params(hmscModel)
    rLParams = load_random_level_params(hmscModel)

    rLDataParams = init_random_level_data_params(modelDataParams, modelData)

    samplerParams = build_sampler(modelDataParams, modelData, rLParams)

    params = {
        **samplerParams,
        **priorHyperParams,
        **rLParams,
        **rLDataParams,
        **modelData,
        **modelDataParams,
    }

    nChains = int(np.squeeze(len(hmscModel["postList"])))

    return params, nChains


def run_gibbs_sampler(file_path, flag_save_postList_to_json=False):

    params, nChains = load_params(file_path)

    gibbs = GibbsSampler(params=params)

    postList = [None] * nChains
    for chain in range(nChains):
        postList[chain] = gibbs.sampling_routine(num_samples=50)

    if flag_save_postList_to_json:
        save_postList_to_json(postList, path, nChains)


if __name__ == "__main__":
    print("Whole Gibbs sampler:")

    startTime = time.time()

    # path = "/Users/gtikhono/Downloads/importExport/"
    path = "/users/anisjyu/Documents/demo-import/"

    filename = "obj-complete.json"

    run_gibbs_sampler(path + filename, flag_save_postList_to_json=True)

    elapsedTime = time.time() - startTime
    print("\nTF decorated whole cycle elapsed %.1f" % elapsedTime)
