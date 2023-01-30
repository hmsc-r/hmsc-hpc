import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import time
import sys
import argparse
import os

sys.path.append("/Users/gtikhono/My Drive/HMSC/2022.06.03 HPC development/hmsc-hpc/hmsc/../")

from random import randint, sample
from datetime import datetime
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

tfr = tf.random
from hmsc.gibbs_sampler import GibbsParameter, GibbsSampler
# from hmsc.updaters.updateEta import updateEta
# from hmsc.updaters.updateAlpha import updateAlpha
# from hmsc.updaters.updateBetaLambda import updateBetaLambda
# from hmsc.updaters.updateLambdaPriors import updateLambdaPriors
# from hmsc.updaters.updateNf import updateNf
# from hmsc.updaters.updateGammaV import updateGammaV
# from hmsc.updaters.updateSigma import updateSigma
# from hmsc.updaters.updateZ import updateZ

from hmsc.utils.jsonutils import (
    load_model_from_json,
    save_postList_to_json,
    save_chains_postList_to_json,
)
from hmsc.utils.hmscutils import (
    load_model_dims,
    load_model_data,
    load_prior_hyperparams,
    load_random_level_hyperparams,
    init_params,
)


# def build_sampler(params, dtype=np.float64):

#     samplerParams = {
#         "Z": GibbsParameter(params["Z"], updateZ),
#         "BetaLambda": GibbsParameter(
#             {"Beta": params["Beta"], "Lambda": params["Lambda"]}, updateBetaLambda
#         ),
#         "GammaV": GibbsParameter(
#             {"Gamma": params["Gamma"], "iV": params["iV"]}, updateGammaV
#         ),
#         "PsiDelta": GibbsParameter(
#             {"Psi": params["Psi"], "Delta": params["Delta"]}, updateLambdaPriors
#         ),
#         "Eta": GibbsParameter(params["Eta"], updateEta),
#         "sigma": GibbsParameter(params["sigma"], updateSigma),
#         "nf": GibbsParameter(
#             {
#                 "Eta": params["Eta"],
#                 "Lambda": params["Lambda"],
#                 "Psi": params["Psi"],
#                 "Delta": params["Delta"],
#             },
#             updateNf,
#         ),
#         "Alpha": GibbsParameter(params["Alpha"], updateAlpha),
#     }
#     return samplerParams


def load_params(file_path, dtype=np.float64):

    hmscImport, hmscModel = load_model_from_json(file_path)
    modelDims = load_model_dims(hmscModel)
    modelData = load_model_data(hmscModel)
    priorHyperparams = load_prior_hyperparams(hmscModel)
    rLHyperparams = load_random_level_hyperparams(hmscModel)
    initParList = init_params(hmscImport.get("initParList"))

    # params = {
    #     **samplerParams,
    #     **priorHyperparams,
    #     **rLHyperParams,
    #     **modelData,
    #     **modelDims,
    # }
    nChains = int(hmscImport.get("nChains")[0])
    return modelDims, modelData, priorHyperparams, rLHyperparams, initParList, nChains


def run_gibbs_sampler(
    num_samples,
    sample_thining,
    sample_burnin,
    init_obj_file_path,
    postList_file_path,
    flag_save_postList_to_json=True,
):

    modelDims, modelData, priorHyperparams, rLHyperparams, initParList, nChains = load_params(init_obj_file_path)
    gibbs = GibbsSampler(modelDims, modelData, priorHyperparams, rLHyperparams)
    # ns = modelDims["ns"]
    # nr = modelDims["nr"]
    # shape_invariants = [
    #     (params["Eta"], [tf.TensorShape([None, None])] * nr),
    #     ((params["BetaLambda"].value)["Beta"], tf.TensorShape([None, ns])),
    #     ((params["BetaLambda"].value)["Lambda"], [tf.TensorShape([None, ns])] * nr),
    #     ((params["PsiDelta"].value)["Psi"], [tf.TensorShape([None, ns])] * nr),
    #     ((params["PsiDelta"].value)["Delta"], [tf.TensorShape([None, 1])] * nr),
    #     (params["Alpha"], [tf.TensorShape([None, 1])] * nr),
    # ]

    postList = [None] * nChains
    for chain in range(nChains):
        print("Computing chain %d" % chain)
        
        parSamples = gibbs.sampling_routine(
            initParList[chain],
            num_samples=num_samples,
            sample_burnin=sample_burnin,
            sample_thining=sample_thining,
        )
        postList[chain] = [None] * num_samples
        for n in range(num_samples):
          parSnapshot = {
            "Beta" : parSamples[0][n],
            "Gamma" : parSamples[1][n],
            "V" : parSamples[2][n],
            "sigma" : parSamples[3][n],
          }
          postList[chain][n] = parSnapshot

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

    path = "/Users/gtikhono/My Drive/HMSC/2022.06.03 HPC development/hmsc-hpc/hmsc/"

    init_obj_file_name = args.input
    postList_file_name = args.output

    postList_file_path = os.path.join(path, "examples/data/", postList_file_name)
    init_obj_file_path = os.path.join(path, "examples/data/", init_obj_file_name)

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

    print("\ndecorated whole cycle elapsed %.1f" % elapsedTime)
