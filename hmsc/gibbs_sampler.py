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
from hmsc.updaters.updateLambdaPriors import updateLambdaPriors
from hmsc.updaters.updateNf import updateNf
from hmsc.updaters.updateGammaV import updateGammaV
from hmsc.updaters.updateRhoInd import updateRhoInd
from hmsc.updaters.updateSigma import updateSigma
from hmsc.updaters.updateZ import updateZ


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
        print_retrace_flag=True,
    ):
        if print_retrace_flag:
            print("retracing")

        ns = self.modelDims["ns"]
        nc = self.modelDims["nc"]
        nr = self.modelDims["nr"]
        npVec = self.modelDims["np"]
        params = paramsInput.copy() #TODO due to tf.function requiring not to change its Tensor input
  
        mcmcSamplesBeta = tf.TensorArray(params["Beta"].dtype, size=num_samples)
        mcmcSamplesGamma = tf.TensorArray(params["Gamma"].dtype, size=num_samples)
        mcmcSamplesV = tf.TensorArray(params["V"].dtype, size=num_samples)
        mcmcSamplesRhoInd = tf.TensorArray(params["rhoInd"].dtype, size=num_samples)
        mcmcSamplesSigma = tf.TensorArray(params["sigma"].dtype, size=num_samples)
        mcmcSamplesLambda = [tf.TensorArray(params["Lambda"][r].dtype, size=num_samples) for r in range(nr)]
        mcmcSamplesPsi = [tf.TensorArray(params["Psi"][r].dtype, size=num_samples) for r in range(nr)]
        mcmcSamplesDelta = [tf.TensorArray(params["Delta"][r].dtype, size=num_samples) for r in range(nr)]
        mcmcSamplesEta = [tf.TensorArray(params["Eta"][r].dtype, size=num_samples) for r in range(nr)]
        mcmcSamplesAlphaInd = [tf.TensorArray(params["AlphaInd"][r].dtype, size=num_samples) for r in range(nr)]
        
        step_num = sample_burnin + num_samples * sample_thining
        print("Iterations %d" % step_num)
        for n in tf.range(step_num):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (params["Eta"], [tf.TensorShape([None, None]) for r in range(nr)]),
                    (params["Beta"], tf.TensorShape([nc, ns])),
                    (params["Lambda"], [tf.TensorShape([None, ns])] * nr),
                    (params["Psi"], [tf.TensorShape([None, ns])] * nr),
                    (params["Delta"], [tf.TensorShape([None, 1])] * nr),
                    (params["AlphaInd"], [tf.TensorShape(None)] * nr),
                ]
            )

            params["Z"] = updateZ(params, self.modelData)
            params["Beta"], params["Lambda"] = updateBetaLambda(params, self.modelData, self.priorHyperparams)
            params["Gamma"], params["V"] = updateGammaV(params, self.modelData, self.priorHyperparams)
            params["rhoInd"] = updateRhoInd(params, self.modelData, self.priorHyperparams)
            params["sigma"] = updateSigma(params, self.modelData, self.priorHyperparams)
            params["Psi"], params["Delta"] = updateLambdaPriors(params, self.rLHyperparams)
            params["Eta"] = updateEta(params, self.modelData, self.modelDims, self.rLHyperparams)
            params["AlphaInd"] = updateAlpha(params, self.rLHyperparams)
            
            if n < sample_burnin:
                params["Lambda"], params["Psi"], params["Delta"], params["Eta"], params["AlphaInd"] = updateNf(params, self.rLHyperparams, n)

            samInd = tf.cast((n - sample_burnin + 1) / sample_thining - 1, tf.int32)
            if (n + 1) % verbose == 0:
                tf.py_function(
                    func=GibbsSampler.printFunction, inp=[n, samInd], Tout=[]
                )
            if (n >= sample_burnin) & ((n - sample_burnin + 1) % sample_thining == 0):                
                mcmcSamplesBeta = mcmcSamplesBeta.write(samInd, params["Beta"])
                mcmcSamplesGamma = mcmcSamplesGamma.write(samInd, params["Gamma"])
                mcmcSamplesV = mcmcSamplesV.write(samInd, params["V"])
                mcmcSamplesRhoInd = mcmcSamplesRhoInd.write(samInd, params["rhoInd"])
                mcmcSamplesSigma = mcmcSamplesSigma.write(samInd, params["sigma"])
                mcmcSamplesLambda = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesLambda, params["Lambda"])]
                mcmcSamplesPsi = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesPsi, params["Psi"])]
                mcmcSamplesDelta = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesDelta, params["Delta"])]
                mcmcSamplesEta = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesEta, params["Eta"])]
                mcmcSamplesAlphaInd = [mcmcSamples.write(samInd, par) for mcmcSamples, par in zip(mcmcSamplesAlphaInd, params["AlphaInd"])]

        print("\nCompleted iterations %d" % step_num)
        samples = {}
        samples["Beta"] = mcmcSamplesBeta.stack()
        samples["Gamma"] = mcmcSamplesGamma.stack()
        samples["V"] = mcmcSamplesV.stack()
        samples["rhoInd"] = mcmcSamplesRhoInd.stack()
        samples["sigma"] = mcmcSamplesSigma.stack()
        samples["Lambda"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesLambda]
        samples["Psi"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesPsi]
        samples["Delta"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesDelta]
        samples["Eta"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesEta]
        samples["AlphaInd"] = [mcmcSamples.stack() for mcmcSamples in mcmcSamplesAlphaInd]

        return samples
