import os
import sys
from contextlib import nullcontext

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# path = os.path.dirname(os.path.dirname(__file__))
# sys.path.append("{}{}".format(path, '/../'))

import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from hmsc.gibbs_sampler import GibbsSampler
tfr = tf.random


from hmsc.utils.export_rds_utils import (
    load_model_from_rds,
    save_chains_postList_to_rds,
)
from hmsc.utils.import_utils import (
    load_model_dims,
    load_model_data,
    load_prior_hyperparams,
    load_random_level_hyperparams,
    load_model_hyperparams,
    init_params,
)

def load_params(file_path, dtype=np.float64):
    hmscImport, hmscModel = load_model_from_rds(file_path)
    modelDims = load_model_dims(hmscModel)
    modelData = load_model_data(hmscModel, hmscImport.get("initParList"))
    priorHyperparams = load_prior_hyperparams(hmscModel)
    # currently not used at all
    # modelHyperparams = load_model_hyperparams(hmscModel, hmscImport.get("dataParList"))
    modelHyperparams = None
    rLHyperparams = load_random_level_hyperparams(hmscModel, hmscImport.get("dataParList"))
    initParList = init_params(hmscImport.get("initParList"), modelData, modelDims, rLHyperparams)
    nChains = int(hmscImport.get("nChains")[0])
  
    return modelDims, modelData, priorHyperparams, modelHyperparams, rLHyperparams, initParList, nChains


def run_gibbs_sampler(
    num_samples,
    sample_thining,
    sample_burnin,
    verbose,
    init_obj_file_path,
    postList_file_path,
    truncated_normal_library=tf,
    flag_save_eta=True,
    flag_save_postList_to_rds=True,
    flag_profile=False,
):
    (
        modelDims,
        modelData,
        priorHyperparams,
        modelHyperparams, #this precomputed one (e.g. Qg) is currently not used and is computed at runtime
        rLHyperparams,
        initParList,
        nChains,
    ) = load_params(init_obj_file_path)
    gibbs = GibbsSampler(modelDims, modelData, priorHyperparams, rLHyperparams)

    print("Running TF Gibbs sampler:")
    
    print("\nInitializing TF graph")
    parSamples = gibbs.sampling_routine(
        initParList[0],
        num_samples=tf.constant(1),
        sample_burnin=tf.constant(1),
        sample_thining=tf.constant(1),
        verbose=verbose,
        truncated_normal_library=truncated_normal_library,
        flag_save_eta=flag_save_eta,
    )
    print("")
    
    # if flag_profile:
    #   tf.profiler.experimental.start('logdir')
    with tf.profiler.experimental.Profile('logdir') if flag_profile else nullcontext():
        startTime = time.time()
        postList = [None] * nChains
        
        for chain in range(nChains):
            print("\nComputing chain %d" % chain)
    
            parSamples = gibbs.sampling_routine(
                initParList[chain],
                num_samples=tf.constant(num_samples),
                sample_burnin=tf.constant(sample_burnin),
                sample_thining=tf.constant(sample_thining),
                verbose=verbose,
                truncated_normal_library=truncated_normal_library,
                flag_save_eta=flag_save_eta,
            )
            postList[chain] = [None] * num_samples
            for n in range(num_samples):
                parSnapshot = {
                    "Beta": parSamples["Beta"][n],
                    "BetaSel": [samples[n] for samples in parSamples["BetaSel"]],
                    "Gamma": parSamples["Gamma"][n],
                    "iV": parSamples["iV"][n],
                    "rhoInd": parSamples["rhoInd"][n],
                    "sigma": parSamples["sigma"][n],
                    "Lambda": [samples[n] for samples in parSamples["Lambda"]],
                    "Psi": [samples[n] for samples in parSamples["Psi"]],
                    "Delta": [samples[n] for samples in parSamples["Delta"]],
                    "Eta": [samples[n] for samples in parSamples["Eta"]] if flag_save_eta else None,
                    "AlphaInd": [samples[n] for samples in parSamples["AlphaInd"]],
                    "wRRR": parSamples["wRRR"][n] if "wRRR" in parSamples else None,
                    "PsiRRR": parSamples["PsiRRR"][n] if "PsiRRR" in parSamples else None,
                    "DeltaRRR": parSamples["DeltaRRR"][n] if "DeltaRRR" in parSamples else None,
                }
                postList[chain][n] = parSnapshot
            
            elapsedTime = time.time() - startTime
            print("\n%d chains completed in %.1f sec\n" % (chain+1, elapsedTime))
    
        elapsedTime = time.time() - startTime
        print("Whole fitting elapsed %.1f" % elapsedTime)
    # if flag_profile:
    #   tf.profiler.experimental.stop()
    if flag_save_postList_to_rds:
        save_chains_postList_to_rds(postList, postList_file_path, nChains, elapsedTime, flag_save_eta)


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
        default="TF-init-obj.rds",
        help="input RDS file with parameters for model initialization",
    )
    argParser.add_argument(
        "-o",
        "--output",
        type=str,
        default="TF-postList-obj.rds",
        help="output RDS file with recorded posterier samples",
    )
    #TODO how an arbitrary path can be used for import?
    # argParser.add_argument(
    #     "-p",
    #     "--path",
    #     type=str,
    #     default="..",
    #     help="path to hmsc-hpc source code",
    # )
    argParser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        help="print out information meassages and progress status",
    )
    argParser.add_argument(
        "--tnlib",
        type=str,
        default="tf",
        help="which library is used for sampling trunacted normal: scipy, tf or tfd",
    )
    argParser.add_argument(
        "--fse",
        type=int,
        default=1,
        help="whether to save Eta posterior",
    )
    argParser.add_argument(
        "--profile",
        type=int,
        default=0,
        help="whether to run profiler alongside sampling",
    )

    args = argParser.parse_args()

    print("args=%s" % args)
    # print("args.samples=%s" % args.samples)

    # path = args.path
    print(os.getcwd())
    init_obj_file_path = args.input
    postList_file_path = args.output

    run_gibbs_sampler(
        num_samples=args.samples,
        sample_thining=args.thin,
        sample_burnin=args.transient,
        verbose=args.verbose,
        init_obj_file_path=init_obj_file_path,
        postList_file_path=postList_file_path,
        truncated_normal_library=args.tnlib,
        flag_save_eta=bool(args.fse),
        flag_save_postList_to_rds=True,
        flag_profile=bool(args.profile),
    )
