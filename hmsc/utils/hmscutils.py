import numpy as np
import tensorflow as tf

tfla, tfr = tf.linalg, tf.random


def load_model_data(hmscModel):

    Y = np.asarray(hmscModel.get("YScaled"))
    X = np.asarray(hmscModel.get("XScaled"))
    T = np.asarray(hmscModel.get("TrScaled"))
    C = np.asarray(hmscModel.get("C"))
    Pi = np.asarray(hmscModel.get("Pi")).astype(int) - 1
    distr = np.asarray(hmscModel.get("distr")).astype(int)

    modelData = {}
    modelData["Y"] = Y
    modelData["X"] = X
    modelData["T"] = T
    modelData["C"] = C
    modelData["Pi"] = Pi
    modelData["distr"] = distr

    return modelData


def load_model_dims(hmscModel):

    ny = int(np.squeeze(hmscModel.get("ny")))
    ns = int(np.squeeze(hmscModel.get("ns")))
    nc = int(np.squeeze(hmscModel.get("nc")))
    nt = int(np.squeeze(hmscModel.get("nt")))
    nr = int(np.squeeze(hmscModel.get("nr")))
    npVec = (np.squeeze(hmscModel.get("np"))).astype(int)

    modelDims = {}
    modelDims["ny"] = ny
    modelDims["ns"] = ns
    modelDims["nc"] = nc
    modelDims["nt"] = nt
    modelDims["nr"] = nr
    modelDims["np"] = npVec

    return modelDims


def load_random_level_hyperparams(hmscModel):

    nr = int(np.squeeze(hmscModel.get("nr")))
    rLParams = [None] * nr
    for r in range(nr):
      rLName = hmscModel.get("rL").keys()[r]
      rLPar = {}
      rLPar["nu"] = hmscModel.get("rL")[rLName]["nu"][0]
      rLPar["a1"] = hmscModel.get("rL")[rLName]["a1"][0]
      rLPar["b1"] = hmscModel.get("rL")[rLName]["b1"][0]
      rLPar["a2"] = hmscModel.get("rL")[rLName]["a2"][0]
      rLPar["b2"] = hmscModel.get("rL")[rLName]["b2"][0]
      rLPar["nfMin"] = int(hmscModel.get("rL")[rLName]["nfMin"][0])
      rLPar["nfMax"] = int(hmscModel.get("rL")[rLName]["nfMax"][0])
      rLPar["sDim"] = int(hmscModel.get("rL")[rLName]["sDim"][0])
      rLPar["spatialMethod"] = hmscModel.get("rL")[rLName]["spatialMethod"]
      rLPar["alphapw"] = hmscModel.get("rL")[rLName]["alphapw"]
      if rLPar["sDim"] > 0:
        rLPar["Wg"] = hmscModel.get("rL")[rLName]["Wg"]
        rLPar["iWg"] = hmscModel.get("rL")[rLName]["iWg"]
        rLPar["LiWg"] = hmscModel.get("rL")[rLName]["LiWg"]
        rLPar["detWg"] = hmscModel.get("rL")[rLName]["detWg"]
      rLParams[r] = rLPar

    return rLParams


def load_prior_hyperparams(hmscModel):

    mGamma = np.asarray(hmscModel.get("mGamma"))
    UGamma = np.asarray(hmscModel.get("UGamma"))
    f0 = np.squeeze(hmscModel.get("f0"))
    V0 = np.squeeze(hmscModel.get("V0"))
    aSigma = np.asarray(hmscModel.get("aSigma"))
    bSigma = np.asarray(hmscModel.get("bSigma"))

    priorHyperParams = {}
    priorHyperParams["mGamma"] = mGamma
    priorHyperParams["iUGamma"] = tfla.inv(UGamma)
    priorHyperParams["f0"] = f0
    priorHyperParams["V0"] = V0
    priorHyperParams["aSigma"] = aSigma
    priorHyperParams["bSigma"] = bSigma

    return priorHyperParams


def init_params(importedInitParList, dtype=np.float64):
    
    initParList = [None] * len(importedInitParList)
    for chainInd, importedInitPar in enumerate(importedInitParList):
      Z = tf.constant(importedInitPar["Z"], dtype=dtype)
      Beta = tf.constant(importedInitPar["Beta"], dtype=dtype)
      Gamma = tf.constant(importedInitPar["Gamma"], dtype=dtype)
      V = tf.constant(importedInitPar["V"], dtype=dtype)
      sigma = tf.constant(importedInitPar["sigma"], dtype=dtype)
      LambdaList = [tf.constant(Lambda, dtype=dtype) for Lambda in importedInitPar["Lambda"]]
      PsiList = [tf.constant(Psi, dtype=dtype) for Psi in importedInitPar["Psi"]]
      DeltaList = [tf.constant(Delta, dtype=dtype) for Delta in importedInitPar["Delta"]]
      EtaList = [tf.constant(Eta, dtype=dtype) for Eta in importedInitPar["Eta"]]
      AlphaList = [tf.constant(Alpha, dtype=dtype) for Alpha in importedInitPar["Alpha"]]
      initPar = {}
      initPar["Z"] = Z
      initPar["Beta"] = Beta
      initPar["Gamma"] = Gamma
      initPar["V"] = V
      initPar["sigma"] = sigma
      initPar["Lambda"] = LambdaList
      initPar["Psi"] = PsiList
      initPar["Delta"] = DeltaList
      initPar["Eta"] = EtaList
      initPar["Alpha"] = AlphaList
      initParList[chainInd] = initPar

    return initParList
