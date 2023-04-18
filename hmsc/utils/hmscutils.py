import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix, csc_matrix
tfla, tfr, tfs = tf.linalg, tf.random, tf.sparse


def load_model_data(hmscModel, importedInitParList, dtype=np.float64):

    Y = np.asarray(hmscModel.get("YScaled")).astype(float)
    T = np.asarray(hmscModel.get("TrScaled"))
    C_import = hmscModel.get("C")
    if isinstance(hmscModel.get("XScaled"), dict):
      X = [np.asarray(hmscModel.get("XScaled")[x]) for x in hmscModel.get("XScaled")]
      rhoGroup = np.asarray([0]*X[0].shape[1]) #TODO replace once implemented in R as well
    else:
      X = np.asarray(hmscModel.get("XScaled"))
      rhoGroup = np.asarray([0]*X.shape[1]) #TODO replace once implemented in R as well
    # rhoGroup = np.asarray(hmscModel.get("rhoGroup")).astype(int) - 1
    Pi = np.asarray(hmscModel.get("Pi")).astype(int) - 1
    distr = np.asarray(hmscModel.get("distr")).astype(int)

    modelData = {}
    modelData["Y"] = Y
    modelData["X"] = X
    modelData["T"] = T
    if C_import is None or len(C_import)==0:
      modelData["C"], modelData["eC"], modelData["VC"] = None, None, None
    else:
      C = np.asarray(C_import)
      modelData["C"] = C
      modelData["eC"], modelData["VC"] = np.linalg.eigh(C) #TODO replace once implemented in R as well
    modelData["rhoGroup"] = rhoGroup
    modelData["Pi"] = Pi
    modelData["distr"] = distr

    ncsel = int(hmscModel.get("ncsel")[0])
    if ncsel is not None:
      covGroup = tf.constant(np.squeeze([hmscModel["XSelect"][i]["covGroup"] for i in range(ncsel)]) - 1, dtype=tf.int32) #FIXME these are potentially ragged
      spGroup = tf.constant([np.array(hmscModel["XSelect"][i]["spGroup"]) - 1 for i in range(ncsel)], dtype=tf.int32) #FIXME these are potentially ragged
      q = tf.constant([hmscModel["XSelect"][i]["q"] for i in range(ncsel)], dtype=dtype) #FIXME these are potentially ragged
      
      if ncsel > 0:
        BetaSel = tf.cast(tf.stack([tf.constant(BetaSel, dtype=dtype) for BetaSel in importedInitParList[0]["BetaSel"]]), tf.bool) #FIXME these are potentially ragged
        for i in range(ncsel):
          for spg in range(q.shape[1]):
            if ~BetaSel[i,spg]:
              fsp = np.where(spGroup[i] == spg)[0]
              for j in fsp:
                X[j,covGroup[i]] = 0

    ncRRR = int(hmscModel.get("ncRRR")[0])
    if ncRRR > 0:
      XRRR = np.asarray(hmscModel.get("XRRR")).astype(float)
      wRRR = tf.constant(importedInitParList[0]["wRRR"], dtype=dtype)
    
      X1A = X
      XB = tf.einsum("ij,kj->ik", XRRR, wRRR)
      if isinstance(X, list):
        raise NotImplementedError
      else:
        X = tf.concat([X,XB], axis=-1)
      
      modelData["XRRR"] = XRRR
      modelData["X1A"] = X1A

    modelData["covGroup"] = covGroup #FIXME these three would better be united within XSelect structure, as they are in R
    modelData["spGroup"] = spGroup
    modelData["q"] = q
    modelData["X"] = X #FIXME X is a single entity for the whole model, but BetaSel is chain-specific

    return modelData


def load_model_dims(hmscModel):

    ny = int(hmscModel.get("ny")[0])
    ns = int(hmscModel.get("ns")[0])
    nc = int(hmscModel.get("nc")[0])
    nt = int(hmscModel.get("nt")[0])
    nr = int(hmscModel.get("nr")[0])
    npVec = np.array(hmscModel.get("np"), int)
    ncsel = int(hmscModel.get("ncsel")[0])
    ncRRR = int(hmscModel.get("ncRRR")[0])
    ncNRRR = int(hmscModel.get("ncNRRR")[0])
    ncORRR = int(hmscModel.get("ncORRR")[0])
    nuRRR = int(hmscModel.get("nuRRR")[0])

    modelDims = {}
    modelDims["ny"] = ny
    modelDims["ns"] = ns
    modelDims["nc"] = nc
    modelDims["nt"] = nt
    modelDims["nr"] = nr
    modelDims["np"] = npVec
    modelDims["ncsel"] = ncsel
    modelDims["ncRRR"] = ncRRR
    modelDims["ncNRRR"] = ncNRRR
    modelDims["ncORRR"] = ncORRR
    modelDims["nuRRR"] = nuRRR

    return modelDims

def load_model_hyperparams(hmscModel, dataParList, dtype=np.float64):

    ns = int(np.squeeze(hmscModel.get("ns")))
   
    dataParams = {}
    if len(dataParList["Qg"]) == (ns*ns): # TODO. need to review this condition
      dataParams["Qg"] = np.reshape(dataParList["Qg"], (ns,ns))
      dataParams["iQg"] = np.reshape(dataParList["iQg"], (ns,ns))
      dataParams["RQg"] = np.reshape(dataParList["RQg"], (ns,ns))
    else:
      dataParams["Qg"] = np.reshape(dataParList["Qg"], (101,ns,ns))
      dataParams["iQg"] = np.reshape(dataParList["iQg"], (101,ns,ns))
      dataParams["RQg"] = np.reshape(dataParList["RQg"], (101,ns,ns))

    return dataParams

def load_random_level_hyperparams(hmscModel, dataParList, dtype=np.float64):

    nr = int(np.squeeze(hmscModel.get("nr")))
    npVec = hmscModel.get("np")
    #npVec = (np.array(hmscModel.get("np"))).astype(int)
    
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
      rLPar["xDim"] = int(hmscModel.get("rL")[rLName]["xDim"][0])
      rLPar["spatialMethod"] = np.squeeze(hmscModel.get("rL")[rLName]["spatialMethod"]) # squeezed returned string array; assumption that one spatial method per level
      if rLPar["sDim"] > 0:
        rLPar["alphapw"] = np.array(hmscModel.get("rL")[rLName]["alphapw"])
        gN = rLPar["alphapw"].shape[0]
        if rLPar["spatialMethod"] == "Full":
          rLPar["Wg"] = np.reshape(dataParList["rLPar"][r]["Wg"], (gN, npVec[r], npVec[r]))
          rLPar["iWg"] = np.reshape(dataParList["rLPar"][r]["iWg"], (gN, npVec[r], npVec[r]))
          rLPar["LiWg"] = tfla.matrix_transpose(np.reshape(dataParList["rLPar"][r]["RiWg"], (gN, npVec[r], npVec[r])))
          rLPar["detWg"] = np.array(dataParList["rLPar"][r]["detWg"])
          
        elif rLPar["spatialMethod"] == "GPP":
          nK = int(dataParList["rLPar"][r]["nK"][0])

          rLPar["nK"] = nK
          rLPar["idDg"] = np.transpose(np.asarray(dataParList["rLPar"][r]["idDg"]))
          rLPar["idDW12g"] = np.transpose(np.reshape(dataParList["rLPar"][r]["idDW12g"], (gN, nK, npVec[r])), [0,2,1])
          rLPar["Fg"] = np.reshape(dataParList["rLPar"][r]["Fg"], (gN, nK, nK))
          rLPar["iFg"] = np.reshape(dataParList["rLPar"][r]["iFg"], (gN, nK, nK))
          rLPar["detDg"] = np.asarray(dataParList["rLPar"][r]["detDg"])

        elif rLPar["spatialMethod"] == "NNGP":
          iWList = [
            tfs.reorder(tfs.SparseTensor(
              np.stack([dataParList["rLPar"][r]["iWgi"][g], dataParList["rLPar"][r]["iWgj"][g]], 1), 
              tf.constant(dataParList["rLPar"][r]["iWgx"][g], dtype),
              [npVec[r], npVec[r]],
            ))
            for g in range(gN)
          ]
          iWList_csc = [
            csc_matrix(coo_matrix(
              (np.array(dataParList["rLPar"][r]["iWgx"][g], dtype),
              (dataParList["rLPar"][r]["iWgi"][g], dataParList["rLPar"][r]["iWgj"][g])),
              [npVec[r], npVec[r]],
            ))
            for g in range(gN)
          ]
          RiWList = [ # these are Right factors, but lower triangular, so different from Cholesky
            tfs.reorder(tfs.SparseTensor(
              np.stack([dataParList["rLPar"][r]["RiWgi"][g], dataParList["rLPar"][r]["RiWgj"][g]], 1), 
              tf.constant(dataParList["rLPar"][r]["RiWgx"][g], dtype),
              [npVec[r], npVec[r]],
            ))
            for g in range(gN)
          ]
          # rLPar["iWg"] = tfs.concat(0, [tfs.expand_dims(iW,0) for iW in iWList])
          rLPar["iWList"] = iWList
          rLPar["iWList_csc"] = iWList_csc
          rLPar["RiWList"] = RiWList
          rLPar["detWg"] = np.array(dataParList["rLPar"][r]["detWg"])
          
        elif rLPar["spatialMethod"] == "NNGP":
          iWList = [
            tfs.reorder(tfs.SparseTensor(
              np.stack([dataParList["rLPar"][r]["iWgi"][g], dataParList["rLPar"][r]["iWgj"][g]], 1), 
              tf.constant(dataParList["rLPar"][r]["iWgx"][g], dtype),
              [npVec[r], npVec[r]],
            ))
            for g in range(gN)
          ]
          iWList_csc = [
            csc_matrix(coo_matrix(
              (np.array(dataParList["rLPar"][r]["iWgx"][g], dtype),
              (dataParList["rLPar"][r]["iWgi"][g], dataParList["rLPar"][r]["iWgj"][g])),
              [npVec[r], npVec[r]],
            ))
            for g in range(gN)
          ]
          RiWList = [ # these are Right factors, but lower triangular, so different from Cholesky
            tfs.reorder(tfs.SparseTensor(
              np.stack([dataParList["rLPar"][r]["RiWgi"][g], dataParList["rLPar"][r]["RiWgj"][g]], 1), 
              tf.constant(dataParList["rLPar"][r]["RiWgx"][g], dtype),
              [npVec[r], npVec[r]],
            ))
            for g in range(gN)
          ]
          # rLPar["iWg"] = tfs.concat(0, [tfs.expand_dims(iW,0) for iW in iWList])
          rLPar["iWList"] = iWList
          rLPar["iWList_csc"] = iWList_csc
          rLPar["RiWList"] = RiWList
          rLPar["detWg"] = np.array(dataParList["rLPar"][r]["detWg"])
          
      rLParams[r] = rLPar

    return rLParams


def load_prior_hyperparams(hmscModel):

    mGamma = np.asarray(hmscModel.get("mGamma"))
    UGamma = np.asarray(hmscModel.get("UGamma"))
    f0 = np.squeeze(hmscModel.get("f0"))
    V0 = np.squeeze(hmscModel.get("V0"))
    rhopw = np.asarray(hmscModel.get("rhopw"))
    aSigma = np.asarray(hmscModel.get("aSigma"))
    bSigma = np.asarray(hmscModel.get("bSigma"))
    a1RRR = np.squeeze(hmscModel.get("a1RRR"))
    b1RRR = np.squeeze(hmscModel.get("b1RRR"))
    a2RRR = np.squeeze(hmscModel.get("a2RRR"))
    b2RRR = np.squeeze(hmscModel.get("b2RRR"))

    priorHyperParams = {}
    priorHyperParams["mGamma"] = mGamma
    priorHyperParams["UGamma"] = UGamma
    priorHyperParams["iUGamma"] = tfla.inv(UGamma)
    priorHyperParams["f0"] = f0
    priorHyperParams["V0"] = V0
    priorHyperParams["rhopw"] = rhopw
    priorHyperParams["aSigma"] = aSigma
    priorHyperParams["bSigma"] = bSigma
    priorHyperParams["a1RRR"] = a1RRR
    priorHyperParams["b1RRR"] = b1RRR
    priorHyperParams["a2RRR"] = a2RRR
    priorHyperParams["b2RRR"] = b2RRR

    return priorHyperParams


def init_params(importedInitParList, dtype=np.float64):
    
    initParList = [None] * len(importedInitParList)
    for chainInd, importedInitPar in enumerate(importedInitParList):
      Z = tf.constant(importedInitPar["Z"], dtype=dtype)
      Beta = tf.constant(importedInitPar["Beta"], dtype=dtype)
      Gamma = tf.constant(importedInitPar["Gamma"], dtype=dtype)
      V = tf.constant(importedInitPar["V"], dtype=dtype)
      rhoInd = tf.cast(tf.constant(importedInitPar["rho"]), tf.int32) - 1 #TODO replace once implemented in R as well
      sigma = tf.constant(importedInitPar["sigma"], dtype=dtype)
      LambdaList = [tf.constant(Lambda, dtype=dtype) for Lambda in importedInitPar["Lambda"]]
      PsiList = [tf.constant(Psi, dtype=dtype) for Psi in importedInitPar["Psi"]]
      DeltaList = [tf.constant(Delta, dtype=dtype) for Delta in importedInitPar["Delta"]]
      EtaList = [tf.constant(Eta, dtype=dtype) for Eta in importedInitPar["Eta"]]
      AlphaIndList = [tf.cast(tf.constant(AlphaInd), tf.int32) - 1 for AlphaInd in importedInitPar["Alpha"]]
      BetaSel = tf.cast(tf.stack([tf.constant(BetaSel, dtype=dtype) for BetaSel in importedInitPar["BetaSel"]]), tf.bool) if "BetaSel" in importedInitPar else tf.reshape((), (0, 1))
      PsiRRR = tf.constant(importedInitPar["PsiRRR"], dtype=dtype) if "PsiRRR" in importedInitPar else tf.reshape((), (0, 1))
      DeltaRRR = tf.constant(importedInitPar["DeltaRRR"], dtype=dtype) if "DeltaRRR" in importedInitPar else tf.reshape((), (0, 1))
      wRRR = tf.constant(importedInitPar["wRRR"], dtype=dtype) if "wRRR" in importedInitPar else tf.reshape((), (0, 1))

      initPar = {}
      initPar["Z"] = Z
      initPar["Beta"] = Beta
      initPar["Gamma"] = Gamma
      initPar["V"] = V
      initPar["rhoInd"] = rhoInd
      initPar["sigma"] = sigma
      initPar["Lambda"] = LambdaList
      initPar["Psi"] = PsiList
      initPar["Delta"] = DeltaList
      initPar["Eta"] = EtaList
      initPar["AlphaInd"] = AlphaIndList
      initPar["BetaSel"] = BetaSel
      initPar["PsiRRR"] = PsiRRR
      initPar["DeltaRRR"] = DeltaRRR
      initPar["wRRR"] = wRRR
      initParList[chainInd] = initPar

    return initParList
