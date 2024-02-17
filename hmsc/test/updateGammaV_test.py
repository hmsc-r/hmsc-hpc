import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
tfr, tfla = tf.random, tf.linalg
tfd = tfp.distributions


from hmsc.updaters.updateGammaV import updateGammaV

def _simple_model(has_phylogeny=False, dtype = np.float64):

    ns, nc, nt = 7001, 51, 31
    
    mGamma = tfr.normal([nc*nt], dtype=dtype)
    iUGamma = tf.eye(nc*nt, dtype=dtype)
    rhopw = tfr.normal([101,2], dtype=dtype)
    V0 = tf.eye(nc, dtype=dtype)
    f0 = nc + 1

    T = np.random.normal(size=[ns,nt])
    
    Gamma = tfr.normal([nc,nt], dtype=dtype)
    iV = tfla.inv(tfp.distributions.WishartTriL(tf.cast(f0, dtype), tfla.cholesky(tfla.inv(V0))).sample())
    #iV = tf.ones([nc,nc], dtype=dtype) + tf.eye(nc, dtype=dtype)
    rhoInd = tf.cast(tf.constant([0]), tf.int32)

    if has_phylogeny:
        rhoGroup = np.asarray([0] * nc)
        C = np.eye(ns)
        eC, VC = np.linalg.eigh(C)
    else:
        rhoGroup = np.asarray([0] * nc)
        C, eC, VC = None, None, None

    Beta = tf.matmul(Gamma,T,transpose_b=True) + \
        tf.transpose(tfd.MultivariateNormalFullCovariance(covariance_matrix=tfla.inv(iV)).sample(ns))

    params = {}
    modelData = {}
    modelDims = {}
    priorHyperparams = {}
    rLHyperparams = {}

    params["Beta"] = Beta
    params["Gamma"] = Gamma
    params["iV"] = iV
    params["rhoInd"] = rhoInd

    modelDims["nc"] = nc
    modelDims["nt"] = nt
    modelDims["ns"] = ns

    modelData["T"] = T
    modelData["rhoGroup"] = rhoGroup
    modelData["C"], modelData["eC"], modelData["VC"] = C, eC, VC

    priorHyperparams["mGamma"] = mGamma
    priorHyperparams["iUGamma"] = iUGamma
    priorHyperparams["rhopw"] = rhopw
    priorHyperparams["V0"] = V0
    priorHyperparams["f0"] = f0

    return params, modelDims, modelData, priorHyperparams, rLHyperparams

#@pytest.mark.parametrize("has_phylogeny", [False, True]) # has_phylogeny=True test is slow; uncomment in final version
@pytest.mark.parametrize("has_phylogeny", [False, False])
def test_updateGammaV_sans_phylogeny(has_phylogeny):

    params, modelDims, modelData, priorHyperparams, _ = _simple_model(has_phylogeny)

    GammaTrue, iVTrue = params["Gamma"], params["iV"]

    Gamma, iV = updateGammaV(params, modelData, priorHyperparams)

    assert_allclose(Gamma, GammaTrue, atol=5.0)
    assert_allclose(iV, iVTrue, atol=5.0)

    assert_allclose(tf.reduce_mean(Gamma), tf.reduce_mean(GammaTrue), atol=0.05)
    assert_allclose(tf.reduce_mean(iV), tf.reduce_mean(iVTrue), atol=0.05)

def test_updateGammaV_shape():

    params, modelDims, modelData, priorHyperparams, _ = _simple_model()
    
    Gamma, iV = updateGammaV(params, modelData, priorHyperparams)

    assert tf.shape(Gamma)[0] == modelDims["nc"]
    assert tf.shape(Gamma)[1] == modelDims["nt"]

    assert tf.shape(iV)[0] == modelDims["nc"]
    assert tf.shape(iV)[1] == modelDims["nc"]