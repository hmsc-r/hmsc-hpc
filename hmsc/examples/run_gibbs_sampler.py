import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import time
import sys
import argparse

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

from hmsc.utils.jsonutils import (
    load_model_from_json,
    save_postList_to_json,
    save_chains_postList_to_json,
)
from hmsc.utils.hmscutils import (
    load_model_data_params,
    load_model_data,
    load_prior_hyper_params,
    load_random_level_params,
    init_random_level_data_params,
    init_sampler_params,
)


def build_sampler(postList, dtype=np.float64):

    samplerParams = {
        "Z": GibbsParameter(postList["Z"], updateZ),
        "BetaLambda": GibbsParameter(
            {"Beta": postList["Beta"], "Lambda": postList["Lambda"]}, updateBetaLambda
        ),
        "GammaV": GibbsParameter(
            {"Gamma": postList["Gamma"], "iV": postList["iV"]}, updateGammaV
        ),
        "PsiDelta": GibbsParameter(
            {"Psi": postList["Psi"], "Delta": postList["Delta"]}, updateLambdaPriors
        ),
        "Eta": GibbsParameter(postList["Eta"], updateEta),
        "sigma": GibbsParameter(postList["sigma"], updateSigma),
        "Nf": GibbsParameter(
            {
                "Eta": postList["Eta"],
                "Lambda": postList["Lambda"],
                "Psi": postList["Psi"],
                "Delta": postList["Delta"],
            },
            updateNf,
        ),
        "Alpha": GibbsParameter(postList["Alpha"], updateAlpha),
    }
    return samplerParams


def load_params(file_path, dtype=np.float64):

    hmscModel = load_model_from_json(file_path)

    modelDataParams = load_model_data_params(hmscModel)
    modelData = load_model_data(hmscModel)
    priorHyperParams = load_prior_hyper_params(hmscModel)
    rLParams = load_random_level_params(hmscModel)

    rLDataParams = init_random_level_data_params(modelDataParams, modelData)

    postList = init_sampler_params(hmscModel)

    samplerParams = build_sampler(postList)

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


def run_gibbs_sampler(
    num_samples,
    sample_thining,
    sample_burnin,
    init_obj_file_path,
    postList_file_path,
    flag_save_postList_to_json=True,
):

    params, nChains = load_params(init_obj_file_path)

    gibbs = GibbsSampler(params=params)

    ns = params["ns"]
    nr = params["nr"]

    shape_invariants = [
        (params["Eta"], [tf.TensorShape([None, None])] * nr),
        ((params["BetaLambda"].value)["Beta"], tf.TensorShape([None, ns])),
        ((params["BetaLambda"].value)["Lambda"], [tf.TensorShape([None, ns])] * nr),
        ((params["PsiDelta"].value)["Psi"], [tf.TensorShape([None, ns])] * nr),
        ((params["PsiDelta"].value)["Delta"], [tf.TensorShape([None, 1])] * nr),
        (params["Alpha"], [tf.TensorShape([None, 1])] * nr),
    ]

    postList = [None] * nChains
    for chain in range(nChains):
        print("Computing chain %d" % chain)

        postList[chain] = gibbs.sampling_routine(
            num_samples,
            sample_burnin=sample_burnin,
            sample_thining=sample_thining,
            shape_invariants=shape_invariants,
        )

    if flag_save_postList_to_json:
        save_chains_postList_to_json(postList, postList_file_path, nChains)


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=2,
        help="number of samples obtained per chain",
    )
    argParser.add_argument(
        "-b",
        "--transient",
        type=int,
        default=0,
        help="number of samples discarded before recording posterior samples",
    )
    argParser.add_argument(
        "-t",
        "--thin",
        type=int,
        default=1,
        help="number of samples between each recording of posterior samples",
    )
    argParser.add_argument(
        "-i",
        "--input",
        type=str,
        default="TF-init-obj.json",
        help="input JSON file with parameters for model initialization",
    )
    argParser.add_argument(
        "-o",
        "--output",
        type=str,
        default="TF-postList-obj.json",
        help="output JSON file with recorded posterier samples",
    )
    argParser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=False,
        help="print out information meassages and progress status",
    )

    args = argParser.parse_args()

    print("args=%s" % args)
    # print("args.samples=%s" % args.samples)

    path = "/Users/anisjyu/Dropbox/hmsc-hpc/hmsc-hpc/hmsc"

    init_obj_file_name = args.input
    postList_file_name = args.output

    init_obj_file_path = path + "/examples/data/" + init_obj_file_name
    postList_file_path = path + "/examples/data/" + postList_file_name

    print("Running TF Gibbs sampler:")

    startTime = time.time()

    run_gibbs_sampler(
        num_samples=args.samples,
        sample_thining=args.thin,
        sample_burnin=args.transient,
        init_obj_file_path=init_obj_file_path,
        postList_file_path=postList_file_path,
        flag_save_postList_to_json=True,
    )

    elapsedTime = time.time() - startTime

    print("\nTF decorated whole cycle elapsed %.1f" % elapsedTime)
