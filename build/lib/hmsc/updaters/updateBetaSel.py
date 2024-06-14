import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hmsc.utils.tf_named_func import tf_named_func
tfm, tfla, tfr = tf.math, tf.linalg, tf.random
tfd = tfp.distributions

@tf_named_func("betaSel")
def updateBetaSel(params, modelDims, modelData, rLHyperparams, dtype=tf.float64):
    ny = modelDims["ny"]
    ns = modelDims["ns"]
    nr = modelDims["nr"]
    nc = modelDims["nc"]
    ncsel = modelDims["ncsel"]

    X = modelData["X"]
    XRRR = modelData["XRRR"]
    Pi = modelData["Pi"]
    XSel = modelData["XSel"]

    Z = params["Z"]
    iD = params["iD"]
    Beta = params["Beta"]
    BetaSel = params["BetaSel"]
    wRRR = params["wRRR"]
    LambdaList = params["Lambda"]
    EtaList = params["Eta"]
        
    LRanLevelList = [None] * nr
    for r, (Eta, Lambda, rLPar) in enumerate(zip(EtaList, LambdaList, rLHyperparams)):
        if(rLPar["xDim"] == 0):
            LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:,r]), Lambda)
        else:
            raise NotImplementedError
    S = Z - sum(LRanLevelList)

    XeffRRR = tf.einsum("ik,hk->ih", XRRR, wRRR)
    if X.ndim == 2:
      Xbase = tf.concat([X, XeffRRR], axis=-1)
    else:
      Xbase = tf.concat([X, tf.repeat(tf.expand_dims(XeffRRR, 0), ns, 0)], axis=-1)
    
    bsCovGroupLen = [XSelElem["covGroup"].size for XSelElem in XSel]
    bsInd = tf.concat([XSelElem["covGroup"] for XSelElem in XSel], 0)
    bsActiveList = [tf.gather(BetaSelElem, XSelElem["spGroup"]) for BetaSelElem, XSelElem in zip(BetaSel, XSel)]
    bsActiveMat = tf.stack(bsActiveList, 0)
    XSel_spGroup = tf.stack([XSelElem["spGroup"] for XSelElem in XSel])
    XSel_q = tf.ragged.constant([XSelElem["q"] for XSelElem in XSel], dtype=dtype)
    XSel_spGroup_size = tf.constant([XSelElem["q"].size for XSelElem in XSel], dtype=XSel_spGroup.dtype)
    BetaSel = tf.ragged.stack(BetaSel)
    
    # for i in tf.range(ncsel):
    #   tf.autograph.experimental.set_loop_options(shape_invariants=[(BetaSel, tf.TensorShape([len(XSel), None]))])
    #   bsActive = tf.cast(tf.repeat(bsActiveMat, bsCovGroupLen, 0), dtype)
    #   bsActiveMatAlt = tf.tensor_scatter_nd_update(bsActiveMat, i[None,None], tfm.logical_not(bsActiveMat[None,i,:]))
    #   bsActiveAlt = tf.cast(tf.repeat(bsActiveMatAlt, bsCovGroupLen, 0), dtype)
    #   bsMask = tf.tensor_scatter_nd_min(tf.ones_like(Beta), bsInd[:,None], bsActive)
    #   bsMaskAlt = tf.tensor_scatter_nd_min(tf.ones_like(Beta), bsInd[:,None], bsActiveAlt)
    #   BetaEff = bsMask * Beta
    #   BetaEffAlt = bsMaskAlt * Beta
    #   if X.ndim == 2:
    #     LFix = tf.einsum("ik,kj->ij", Xbase, BetaEff)
    #     LFixAlt = tf.einsum("ik,kj->ij", Xbase, BetaEffAlt)
    #   else:
    #     LFix = tf.einsum("jik,kj->ij", Xbase, BetaEff)
    #     LFixAlt = tf.einsum("jik,kj->ij", Xbase, BetaEffAlt)
      
    #   log_iD05 = tfm.multiply_no_nan(0.5*tfm.log(iD), tf.cast(iD>0, dtype))
    #   ll = log_iD05 - 0.5*iD * (S-LFix)**2 # tfd.Normal(loc=L, scale=iD**-0.5).log_prob(Z)
    #   llAlt = log_iD05 - 0.5*iD * (S-LFixAlt)**2
    #   logLike = tf.scatter_nd(XSel_spGroup[i,:,None], tf.reduce_sum(ll, 0), XSel_spGroup_size[i,None])
    #   logLikeAlt = tf.scatter_nd(XSel_spGroup[i][:,None], tf.reduce_sum(llAlt, 0), XSel_spGroup_size[i,None])
    #   logProb = tf.where(BetaSel[i]==True, tfm.log(XSel_q[i]), tfm.log(tf.constant(1,dtype)-XSel_q[i])) + logLike
    #   logProbAlt = tf.where(BetaSel[i]==False, tfm.log(XSel_q[i]), tfm.log(tf.constant(1,dtype)-XSel_q[i])) + logLikeAlt
    #   BetaSelElemNew = tf.where(tfm.exp(logProbAlt-logProb) > tfr.uniform(XSel_spGroup_size[i,None], dtype=dtype), tfm.logical_not(BetaSel[i]), BetaSel[i])
    #   bsActiveMat = tf.tensor_scatter_nd_update(bsActiveMat, i[None,None], tf.gather(BetaSelElemNew, XSel_spGroup[i,:])[None,:])
    #   BetaSelFlatUpdated = tf.tensor_scatter_nd_update(BetaSel.flat_values, tf.where(BetaSel.value_rowids()==i), BetaSelElemNew)
    #   BetaSel = BetaSel.with_flat_values(BetaSelFlatUpdated)
    
    for i in tf.range(ncsel):
      tf.autograph.experimental.set_loop_options(shape_invariants=[(BetaSel, tf.TensorShape([len(XSel), None]))])
      bsActiveMat0 = tf.tensor_scatter_nd_update(bsActiveMat, i[None,None], tf.zeros([1,ns],bool))
      bsActiveMat1 = tf.tensor_scatter_nd_update(bsActiveMat, i[None,None], tf.ones([1,ns],bool))
      bsActive0 = tf.cast(tf.repeat(bsActiveMat0, bsCovGroupLen, 0), dtype)
      bsActive1 = tf.cast(tf.repeat(bsActiveMat1, bsCovGroupLen, 0), dtype)
      bsMask0 = tf.tensor_scatter_nd_min(tf.ones_like(Beta), bsInd[:,None], bsActive0)
      bsMask1 = tf.tensor_scatter_nd_min(tf.ones_like(Beta), bsInd[:,None], bsActive1)
      BetaEff0 = bsMask0 * Beta
      BetaEff1 = bsMask1 * Beta
      if X.ndim == 2:
        LFix0 = tf.einsum("ik,kj->ij", Xbase, BetaEff0)
        LFix1 = tf.einsum("ik,kj->ij", Xbase, BetaEff1)
      else:
        LFix0 = tf.einsum("jik,kj->ij", Xbase, BetaEff0)
        LFix1 = tf.einsum("jik,kj->ij", Xbase, BetaEff1)
      
      log_iD05 = tfm.multiply_no_nan(0.5*tfm.log(iD), tf.cast(iD>0, dtype))
      ll0 = log_iD05 - 0.5*iD * (S-LFix0)**2 # tfd.Normal(loc=L, scale=iD**-0.5).log_prob(Z)
      ll1 = log_iD05 - 0.5*iD * (S-LFix1)**2
      logLike0 = tf.scatter_nd(XSel_spGroup[i,:,None], tf.reduce_sum(ll0, 0), XSel_spGroup_size[i,None])
      logLike1 = tf.scatter_nd(XSel_spGroup[i,:,None], tf.reduce_sum(ll1, 0), XSel_spGroup_size[i,None])
      logProb0 = tfm.log(tf.constant(1,dtype)-XSel_q[i]) + logLike0
      logProb1 = tfm.log(XSel_q[i]) + logLike1
      prob = tf.nn.softmax(tf.stack([logProb0, logProb1], -1))
      BetaSelElemNew = tf.where(prob[:,0] > tfr.uniform(XSel_spGroup_size[i,None], dtype=dtype), False, True)
      bsActiveMat = tf.tensor_scatter_nd_update(bsActiveMat, i[None,None], tf.gather(BetaSelElemNew, XSel_spGroup[i,:])[None,:])
      BetaSelFlatUpdated = tf.tensor_scatter_nd_update(BetaSel.flat_values, tf.where(BetaSel.value_rowids()==i), BetaSelElemNew)
      BetaSel = BetaSel.with_flat_values(BetaSelFlatUpdated)

    BetaSel = tf.split(BetaSel.flat_values, XSel_spGroup_size)
   
    bsActive = tf.cast(tf.repeat(bsActiveMat, bsCovGroupLen, 0), dtype)
    bsMask = tf.tensor_scatter_nd_min(tf.ones([nc,ns], dtype), bsInd[:,None], bsActive)
    if X.ndim == 2:
      Xeff = tf.einsum("ik,kj->jik", Xbase, bsMask)
    else:
      Xeff = tf.einsum("jik,kj->jik", Xbase, bsMask)
    
    return BetaSel, Xeff
  

