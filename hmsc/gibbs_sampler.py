# MIT License

# Copyright (c) 2022 Kit Gallagher

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf
import sys
from hmsc.updaters.updateEta import updateEta
from hmsc.updaters.updateAlpha import updateAlpha
from hmsc.updaters.updateBetaLambda import updateBetaLambda
from hmsc.updaters.updateBetaSel import updateBetaSel
from hmsc.updaters.updateLambdaPriors import updateLambdaPriors
from hmsc.updaters.updateNf import updateNf
from hmsc.updaters.updateGammaV import updateGammaV
from hmsc.updaters.updateRhoInd import updateRhoInd
from hmsc.updaters.updateSigma import updateSigma
from hmsc.updaters.updateZ import updateZ
from hmsc.updaters.updatewRRR import updatewRRR
from hmsc.updaters.updatewRRRPriors import updatewRRRPriors
tfm = tf.math


class GibbsParameter:
    def __init__(self, value, conditional_posterior, posterior_params=None):
        self.__value = value
        self.conditional_posterior = conditional_posterior
        self.posterior_params = posterior_params

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return str(self.__value)

    def get_value(self):
        return self.__value

    def set_value(self, value):
        self.__value = value

    value = property(get_value, set_value)

    def sample(self, sample_params):
        param_values = {}
        for k, v in sample_params.items():
            if isinstance(v, GibbsParameter):
                param_values[k] = v.value
            else:
                param_values[k] = v
        post_params = param_values
        self.__value = self.conditional_posterior(post_params)
        return self.__value


class GibbsSampler(tf.Module):
    def __init__(self, modelDims, modelData, priorHyperparams, rLHyperparams):

        self.modelDims = modelDims
        self.modelData = modelData
        self.priorHyperparams = priorHyperparams
        self.rLHyperparams = rLHyperparams

    @staticmethod
    def printFunction(i, samInd):
        outStr = "iteration " + str(i.numpy() + 1)
        if samInd.numpy() >= 0:
            outStr += " saving " + str(samInd.numpy() + 1)
        else:
            outStr += " transient"
        sys.stdout.write("\r" + outStr)

    @tf.function
    def sampling_routine(
        self,
        paramsInput,
        num_samples=1,
        sample_burnin=0,
        sample_thining=1,
        verbose=1,
        truncated_normal_library="scipy",
        print_retrace_flag=True,
        print_debug_flag= False
    ):
        if print_retrace_flag:
            print("retracing")

        ns = self.modelDims["ns"]
        nr = self.modelDims["nr"]
        nc = self.modelDims["nc"]
        ncsel = self.modelDims["ncsel"]
        ncRRR = self.modelDims["ncRRR"]
        ncNRRR = self.modelDims["ncNRRR"]
        ncORRR = self.modelDims["ncORRR"]
        npVec = self.modelDims["np"]
        params = paramsInput.copy() #TODO due to tf.function requiring not to change its Tensor input
        #TODO potentially move next two lines to somewhere more approriate
        params["iD"] = tf.cast(tfm.logical_not(tfm.is_nan(self.modelData["Y"])), params["Z"].dtype) * tf.ones_like(params["Z"]) * params["sigma"]**-2
        _, _, params["poisson_omega"] = updateZ(params, self.modelData, poisson_preupdate_z=False, poisson_update_omega=True, 
                                                poisson_marginalize_z=False, truncated_normal_library=truncated_normal_library)

        mcmcSamplesBeta = tf.TensorArray(params["Beta"].dtype, size=num_samples)
        mcmcSamplesBetaSel = [tf.TensorArray(tf.bool, size=num_samples) for i in range(ncsel)]
        mcmcSamplesGamma = tf.TensorArray(params["Gamma"].dtype, size=num_samples)
        mcmcSamplesiV = tf.TensorArray(params["iV"].dtype, size=num_samples)
        mcmcSamplesRhoInd = tf.TensorArray(params["rhoInd"].dtype, size=num_samples)
        mcmcSamplesSigma = tf.TensorArray(params["sigma"].dtype, size=num_samples)
        mcmcSamplesLambda = [tf.TensorArray(params["Lambda"][r].dtype, size=num_samples) for r in range(nr)]
        mcmcSamplesPsi = [tf.TensorArray(params["Psi"][r].dtype, size=num_samples) for r in range(nr)]
        mcmcSamplesDelta = [tf.TensorArray(params["Delta"][r].dtype, size=num_samples) for r in range(nr)]
        mcmcSamplesEta = [tf.TensorArray(params["Eta"][r].dtype, size=num_samples) for r in range(nr)]
        mcmcSamplesAlphaInd = [tf.TensorArray(params["AlphaInd"][r].dtype, size=num_samples) for r in range(nr)]
        mcmcSampleswRRR = tf.TensorArray(params["wRRR"].dtype if ncRRR > 0 else tf.float64, size=num_samples)
        mcmcSamplesPsiRRR = tf.TensorArray(params["PsiRRR"].dtype if ncRRR > 0 else tf.float64, size=num_samples)
        mcmcSamplesDeltaRRR = tf.TensorArray(params["DeltaRRR"].dtype if ncRRR > 0 else tf.float64, size=num_samples)
        
        step_num = sample_burnin + num_samples * sample_thining
        print("Iterations %d" % step_num)
        for n in tf.range(step_num):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (params["Eta"], [tf.TensorShape([npVec[r], None]) for r in range(nr)]),
                    (params["Lambda"], [tf.TensorShape([None, ns])] * nr),
                    (params["Psi"], [tf.TensorShape([None, ns])] * nr),
                    (params["Delta"], [tf.TensorShape([None, 1])] * nr),
                    (params["AlphaInd"], [tf.TensorShape(None)] * nr),
                ]
            )
            
            z_marginalize_iter_cond = lambda it: ((it % 2) == 1) & (it >= 0)
            z_marginalize_iter_flag = z_marginalize_iter_cond(n)
            z_marginalize_prev_flag = z_marginalize_iter_cond(n-1)
            
            if z_marginalize_iter_flag == False:
              if z_marginalize_prev_flag == False:
                params["Z"], params["iD"], params["poisson_omega"] = updateZ(params, self.modelData, poisson_preupdate_z=False,
                                                                             poisson_marginalize_z=False, truncated_normal_library=truncated_normal_library)
              else:
                params["Z"], params["iD"], params["poisson_omega"] = updateZ(params, self.modelData, poisson_preupdate_z=True,
                                                                             poisson_marginalize_z=False, truncated_normal_library=truncated_normal_library)
            else:
              if z_marginalize_prev_flag == False:
                params["Z"], params["iD"], params["poisson_omega"] = updateZ(params, self.modelData, poisson_preupdate_z=False,
                                                                             poisson_marginalize_z=True, truncated_normal_library=truncated_normal_library)
              else:
                params["Z"], params["iD"], params["poisson_omega"] = updateZ(params, self.modelData, poisson_preupdate_z=True,
                                                                             poisson_marginalize_z=True, truncated_normal_library=truncated_normal_library)
            if print_debug_flag:
              tf.print("Z", tf.reduce_sum(tf.cast(tfm.is_nan(params["Z"]), tf.int32)))
              tf.print("iD", tf.reduce_sum(tf.cast(tfm.is_nan(params["iD"]), tf.int32)))
            
            params["Beta"], params["Lambda"] = updateBetaLambda(params, self.modelData, self.priorHyperparams)
            if print_debug_flag:
              tf.print("Beta", tf.reduce_sum(tf.cast(tfm.is_nan(params["Beta"]) | (tf.abs(params["Beta"]) > 1e9), tf.int32)))
              tf.print("Lambda", [tf.reduce_sum(tf.cast(tfm.is_nan(par), tf.int32)) for par in params["Lambda"]])
            
            if ncRRR > 0:
              params["wRRR"], params["Xeff"] = updatewRRR(params, self.modelDims, self.modelData, self.rLHyperparams)
              if print_debug_flag:
                tf.print("wRRR", tf.reduce_sum(tf.cast(tfm.is_nan(params["wRRR"]) | (tf.abs(params["wRRR"]) > 1e9), tf.int32)))
                tf.print("Xeff", tf.reduce_sum(tf.cast(tfm.is_nan(params["Xeff"]) | (tf.abs(params["Xeff"]) > 1e9), tf.int32)))
              params["PsiRRR"], params["DeltaRRR"] = updatewRRRPriors(params, self.modelDims, self.priorHyperparams)
            
            if ncsel > 0:
              params["BetaSel"], params["Xeff"] = updateBetaSel(params, self.modelDims, self.modelData, self.rLHyperparams)
              if print_debug_flag:
                # tf.print("BetaSel - not NA", [tf.reduce_sum(tf.cast(par, tf.int32)) for par in params["BetaSel"]])
                tf.print("Xeff", tf.reduce_sum(tf.cast(tfm.is_nan(params["Xeff"]) | (tf.abs(params["Xeff"]) > 1e9), tf.int32)))

            params["Gamma"], params["iV"] = updateGammaV(params, self.modelData, self.priorHyperparams)
            if print_debug_flag:
              tf.print("Gamma", tf.reduce_sum(tf.cast(tfm.is_nan(params["Gamma"]) | (tf.abs(params["Gamma"]) > 1e9), tf.int32)))
              tf.print("iV", tf.reduce_sum(tf.cast(tfm.is_nan(params["iV"]) | (tf.abs(params["iV"]) > 1e9), tf.int32)))
            
            params["rhoInd"] = updateRhoInd(params, self.modelData, self.priorHyperparams)
            
            params["Psi"], params["Delta"] = updateLambdaPriors(params, self.rLHyperparams)
            
            params["Eta"] = updateEta(params, self.modelDims, self.modelData, self.rLHyperparams)
            if print_debug_flag:
              tf.print("Eta", [tf.reduce_sum(tf.cast(tfm.is_nan(par), tf.int32)) for par in params["Eta"]])
            
            params["AlphaInd"] = updateAlpha(params, self.rLHyperparams)
            
            if z_marginalize_iter_flag == False:
              params["sigma"] = updateSigma(params, self.modelDims, self.modelData, self.priorHyperparams)
              if print_debug_flag:
                tf.print("sigma", tf.reduce_sum(tf.cast(tfm.is_nan(params["sigma"]), tf.int32)))

            if n < sample_burnin:
                params["Lambda"], params["Psi"], params["Delta"], params["Eta"], params["AlphaInd"] = updateNf(params, self.rLHyperparams, n)
            # tf.print(tf.shape(params["Lambda"][0])[0])

            samInd = tf.cast((n - sample_burnin + 1) / sample_thining - 1, tf.int32)
            if (n + 1) % verbose == 0:
                tf.py_function(func=GibbsSampler.printFunction, inp=[n, samInd], Tout=[])
                
            if (n >= sample_burnin) & ((n - sample_burnin + 1) % sample_thining == 0):                
                mcmcSamplesBeta = mcmcSamplesBeta.write(samInd, params["Beta"])
                mcmcSamplesBetaSel = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesBetaSel, params["BetaSel"])]
                mcmcSamplesGamma = mcmcSamplesGamma.write(samInd, params["Gamma"])
                mcmcSamplesiV = mcmcSamplesiV.write(samInd, params["iV"])
                mcmcSamplesRhoInd = mcmcSamplesRhoInd.write(samInd, params["rhoInd"])
                mcmcSamplesSigma = mcmcSamplesSigma.write(samInd, params["sigma"])
                mcmcSamplesLambda = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesLambda, params["Lambda"])]
                mcmcSamplesPsi = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesPsi, params["Psi"])]
                mcmcSamplesDelta = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesDelta, params["Delta"])]
                mcmcSamplesEta = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesEta, params["Eta"])]
                mcmcSamplesAlphaInd = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesAlphaInd, params["AlphaInd"])]
                if ncRRR > 0:
                    mcmcSampleswRRR = mcmcSampleswRRR.write(samInd, params["wRRR"])
                    mcmcSamplesPsiRRR = mcmcSamplesPsiRRR.write(samInd, params["PsiRRR"])
                    mcmcSamplesDeltaRRR = mcmcSamplesDeltaRRR.write(samInd, params["DeltaRRR"])

        print("\nCompleted iterations %d" % step_num)
        samples = {}
        samples["Beta"] = mcmcSamplesBeta.stack()
        samples["BetaSel"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesBetaSel]
        samples["Gamma"] = mcmcSamplesGamma.stack()
        samples["iV"] = mcmcSamplesiV.stack()
        samples["rhoInd"] = mcmcSamplesRhoInd.stack()
        samples["sigma"] = mcmcSamplesSigma.stack()
        samples["Lambda"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesLambda]
        samples["Psi"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesPsi]
        samples["Delta"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesDelta]
        samples["Eta"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesEta]
        samples["AlphaInd"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesAlphaInd]
        if ncRRR > 0:
          samples["wRRR"] = mcmcSampleswRRR.stack()
          samples["PsiRRR"] = mcmcSamplesPsiRRR.stack()
          samples["DeltaRRR"] = mcmcSamplesDeltaRRR.stack()
        
        return samples
