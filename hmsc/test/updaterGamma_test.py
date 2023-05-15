import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import tensorflow as tf
tfr = tf.random

from hmsc.updaters.updateGammaV import updateGammaV

def _simple_model(dtype = np.float64):
    # test model - simple integrity

    ny = 123
    nc = 5
    nt = 3
    ns = 101
    
    Beta = tfr.normal([nc,ns], dtype=dtype)
    Gamma = tfr.normal([nc,nt], dtype=dtype)
    iV = tf.ones([nc,nc], dtype=dtype) + tf.eye(nc, dtype=dtype)
    rhoInd = tf.cast(tf.constant([0]), tf.int32)

    X = np.random.normal(size=[ny,nc])
    T = np.random.normal(size=[ns,nt])
    rhoGroup = np.asarray([0] * X.shape[-1]) 
    C, eC, VC = None, None, None

    mGamma = tfr.normal([nc*nt], dtype=dtype)
    iUGamma = tf.eye(nc*nt, dtype=dtype)
    rhopw = tfr.normal([101,2], dtype=dtype)
    V0 = tf.eye(nc, dtype=dtype)
    f0 = 5+1

    params = {}
    modelData = {}
    modelDims = {}
    priorHyperparams = {}

    params["Beta"] = Beta
    params["Gamma"] = Gamma
    params["iV"] = iV
    params["rhoInd"] = rhoInd

    modelDims["ny"] = ny
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

    return params, modelDims, modelData, priorHyperparams

def test_updateGammaV():

    params, modelDims, modelData, priorHyperparams = _simple_model()

    Gamma, _ = updateGammaV(params, modelData, priorHyperparams)

    # assert_allclose(tf.reduce_sum(Gamma), 0.0, atol=1.0)

def test_updateGamma_shape():

    params, modelDims, modelData, priorHyperparams = _simple_model()
    
    Gamma, _ = updateGammaV(params, modelData, priorHyperparams)

    assert tf.shape(Gamma)[0] == modelDims["nc"]
    assert tf.shape(Gamma)[1] == modelDims["nt"]