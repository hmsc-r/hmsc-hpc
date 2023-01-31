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

    # def single_sample(self, param_name):
    #     value = self.params[param_name].sample(self.params)
    #     self.params[param_name].value = value
    #     return value

    @staticmethod
    def printFunction(i, samInd):
        outStr = "iteration " + str(i.numpy())
        if samInd.numpy() >= 0:
            outStr += " saving " + str(samInd.numpy())
        else:
            outStr += " transient"
        sys.stdout.write("\r" + outStr)

    @tf.function
    def sampling_routine(
        self,
        paramsTmp,
        num_samples=100,
        sample_burnin=0,
        sample_thining=1,
        verbose=1,
        print_retrace_flag=True,
    ):
        if print_retrace_flag:
            print("retracing")

        params = paramsTmp.copy()
        # samplesPars = {}
        # for parName in params.keys:
        #   samplesPars[parName] = tf.TensorArray(params.dtype, size=num_samples)
        mcmcSamples = [tf.TensorArray(tf.float64, size=num_samples) for i in range(9)]
        # samplesGamma = tf.TensorArray(tf.float64, size=num_samples)
        # mcmcSamples = []
        step_num = sample_burnin + num_samples * sample_thining
        print("Iterations %d" % step_num)

        _, ns = self.modelData["Y"].shape
        nr = len(params["Lambda"])

        for n in tf.range(step_num):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (params["Eta"], [tf.TensorShape([None, None])] * nr),
                    (params["Beta"], tf.TensorShape([None, ns])),
                    (params["Lambda"], [tf.TensorShape([None, ns])] * nr),
                    (params["Psi"], [tf.TensorShape([None, ns])] * nr),
                    (params["Delta"], [tf.TensorShape([None, 1])] * nr),
                    (params["Alpha"], [tf.TensorShape([None, 1])] * nr),
                ]
            )

            # tf.autograph.experimental.set_loop_options(
            #     shape_invariants=shape_invariants
            # )
            # print("Iteration %d of %d" % (n, step_num))

            params["Z"] = updateZ(params, self.modelData)
            BetaLambda = updateBetaLambda(params, self.modelData)
            params["Beta"], params["Lambda"] = BetaLambda["Beta"], BetaLambda["Lambda"]
            GammaV = updateGammaV(params, self.modelData, self.priorHyperparams)
            params["Gamma"], params["V"] = GammaV["Gamma"], GammaV["V"]
            params["sigma"] = updateSigma(params, self.modelData, self.priorHyperparams)
            PsiDelta = updateLambdaPriors(params, self.rLHyperparams)
            params["Psi"], params["Delta"] = PsiDelta["Psi"], PsiDelta["Delta"]
            params["Eta"] = updateEta(
                params, self.modelData, self.modelDims, self.rLHyperparams
            )
            params["Alpha"] = updateAlpha(params, self.rLHyperparams)
            
            EtaLambdaPsiDelta = updateNf(params, self.rLHyperparams)
            params["Eta"], params["Lambda"], params["Psi"], params["Delta"] = EtaLambdaPsiDelta["Eta"], EtaLambdaPsiDelta["Lambda"], EtaLambdaPsiDelta["Psi"], EtaLambdaPsiDelta["Delta"]

            samInd = tf.cast((n - sample_burnin + 1) / sample_thining - 1, tf.int32)
            if (n + 1) % verbose == 0:
                tf.py_function(
                    func=GibbsSampler.printFunction, inp=[n, samInd], Tout=[]
                )
            if (n >= sample_burnin) & ((n - sample_burnin + 1) % sample_thining == 0):
                # mcmcSnapshot = {
                #   "Beta" : params["Beta"],
                #   "Gamma" : params["Gamma"],
                #   "V" : params["V"],
                #   "sigma" : params["sigma"],
                # }
                # mcmcSamples.append(mcmcSnapshot)

                npVec = self.modelDims["np"]
                if npVec is None:
                    paddedEtaList = params["Eta"]
                else:
                    paddedEtaList = [tf.pad(Eta, paddings=tf.constant([[0, (np.max(npVec) - Eta.shape[0])], [0, 0]]), mode="CONSTANT") for Eta in params["Eta"]]

                mcmcSamples[0] = mcmcSamples[0].write(samInd, params["Beta"])
                mcmcSamples[1] = mcmcSamples[1].write(samInd, params["Gamma"])
                mcmcSamples[2] = mcmcSamples[2].write(samInd, params["V"])
                mcmcSamples[3] = mcmcSamples[3].write(samInd, params["sigma"])
                mcmcSamples[4] = mcmcSamples[4].write(samInd, params["Alpha"])
                mcmcSamples[5] = mcmcSamples[5].write(samInd, params["Psi"])
                mcmcSamples[6] = mcmcSamples[6].write(samInd, params["Delta"])
                mcmcSamples[7] = mcmcSamples[7].write(samInd, paddedEtaList)
                mcmcSamples[8] = mcmcSamples[8].write(samInd, params["Lambda"])
                

                # for parInd, parName in enumerate(["Beta", "Gamma", "V", "sigma"]): # for unclear reason this cycle does not work...
                #   mcmcSamples[parInd] = mcmcSamples[parInd].write(samInd, params[parName])

        print("Completed iterations %d" % step_num)
        samples = [samples.stack() for samples in mcmcSamples]
        return samples
