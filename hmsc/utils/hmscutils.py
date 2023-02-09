import numpy as np
import tensorflow as tf

tfla, tfr = tf.linalg, tf.random


def load_model_data(hmscModel):

    Y = np.asarray(hmscModel.get("YScaled")).astype(float)
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

    ny = int(hmscModel.get("ny")[0])
    ns = int(hmscModel.get("ns")[0])
    nc = int(hmscModel.get("nc")[0])
    nt = int(hmscModel.get("nt")[0])
    nr = int(hmscModel.get("nr")[0])
    npVec = np.array(hmscModel.get("np"), int)

    modelDims = {}
    modelDims["ny"] = ny
    modelDims["ns"] = ns
    modelDims["nc"] = nc
    modelDims["nt"] = nt
    modelDims["nr"] = nr
    modelDims["np"] = npVec

    return modelDims


def load_random_level_hyperparams(hmscModel, dataParList):

    nr = int(np.squeeze(hmscModel.get("nr")))
    npVec = (np.squeeze(hmscModel.get("np"))).astype(int)
    
    rLParams = [None] * nr
    for r in range(nr):
      rLName = list(hmscModel.get("rL").keys())[r]
      rLPar = {}
      rLPar["nu"] = hmscModel.get("rL")[rLName]["nu"][0]
      rLPar["a1"] = hmscModel.get("rL")[rLName]["a1"][0]
      rLPar["b1"] = hmscModel.get("rL")[rLName]["b1"][0]
      rLPar["a2"] = hmscModel.get("rL")[rLName]["a2"][0]
      rLPar["b2"] = hmscModel.get("rL")[rLName]["b2"][0]
      rLPar["nfMin"] = int(hmscModel.get("rL")[rLName]["nfMin"][0])
      rLPar["nfMax"] = int(hmscModel.get("rL")[rLName]["nfMax"][0])
      rLPar["sDim"] = int(hmscModel.get("rL")[rLName]["sDim"][0])
      rLPar["spatialMethod"] = np.squeeze(hmscModel.get("rL")[rLName]["spatialMethod"]) # squeezed returned string array; assumption that one spatial method per level
      if rLPar["sDim"] > 0 and rLPar["spatialMethod"] != "GPP": # skipping GPP; it has different hyperparams
        rLPar["alphapw"] = np.asarray(hmscModel.get("rL")[rLName]["alphapw"])
        gN = rLPar["alphapw"].shape[0]
        rLPar["Wg"] = np.reshape(dataParList["rLPar"][r]["Wg"], (gN, npVec[r], npVec[r]))
        rLPar["iWg"] = np.reshape(dataParList["rLPar"][r]["iWg"], (gN, npVec[r], npVec[r]))
        rLPar["RiWg"] = np.reshape(dataParList["rLPar"][r]["RiWg"], (gN, npVec[r], npVec[r]))
        rLPar["detWg"] = np.asarray(dataParList["rLPar"][r]["detWg"])
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
      AlphaList = [tf.expand_dims(tf.constant(Alpha, dtype=dtype), 1) for Alpha in importedInitPar["Alpha"]]
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
