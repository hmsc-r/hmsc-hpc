import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfm, tfla, tfr = tf.math, tf.linalg, tf.random
tfd = tfp.distributions

def updateBetaSel(params, modelDims, modelData, rLHyperparams, dtype=tf.float64):
    ny = modelDims["ny"]
    ns = modelDims["ns"]
    nr = modelDims["nr"]
    nc = modelDims["nc"]
    npVec = modelDims["np"]
    ncsel = modelDims["ncsel"]

    X = modelData["X"]
    Pi = modelData["Pi"]

    covGroup = modelData["covGroup"]    
    spGroup = modelData["spGroup"]
    q = modelData["q"]
    mask = modelData["mask"]

    Z = params["Z"]
    EtaList = params["Eta"]
    Beta = params["Beta"]
    BetaSel = params["BetaSel"]
    LambdaList = params["Lambda"]
    sigma = params["sigma"]

    iSigma = 1 / sigma

    std = iSigma**-0.5

    X1 = tf.stack([X for i in range(ns)])
        
    LRanLevelList = [None] * nr
    for r, (Eta, Lambda, rLPar) in enumerate(zip(EtaList, LambdaList, rLHyperparams)):
        if(int(rLPar["xDim"]) == 0):
            LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:,r]), Lambda)
        else:
            raise NotImplementedError

    pridif = tf.where(BetaSel, tfm.log(q) - tfm.log(1-q), tfm.log(1-q) - tfm.log(q))

    X1 = modelData["X"]
    X1 = tf.stack([X1 for j in range(ns) if not isinstance(X1, dict) and ncsel > 0])

    def get_fsp_all():
        def get_fsp(i,spg):
            indices = tf.where(spGroup[i] == spg)
            return tf.concat([tf.ones_like(indices)*i,indices], axis=1)
        return tf.concat([get_fsp(i,spg) for spg in range(q.shape[1]) for i in range(ncsel)], axis=0)
    fsp_all = get_fsp_all()

    mask_nc = tf.transpose(tf.concat([BetaSel,tf.cast(tf.zeros([nc-ncsel,ns]), tf.bool)],axis=0))
    mask_nc_full = tf.reshape(tf.repeat(mask_nc, tf.ones([nc], dtype=tf.int32)*ny, axis=1), (ns,ny,nc))
    mask_fsp = tfm.logical_and(tf.cast(tf.reduce_max(tf.cast(mask, tf.int8), axis=0), tf.bool), mask_nc_full)

    X = X1 * tf.cast(mask_fsp, dtype)

    LFix = tf.einsum("ijk,ki->ji", X, Beta)
    L = LFix + sum(LRanLevelList)
    LNew = L
    
    ll = tfd.Normal(loc=L, scale=std).cdf(Z)

    indices = fsp_all
    updates = tfm.logical_not(tf.gather_nd(BetaSel, indices=fsp_all))
    BetaSelNew = tf.tensor_scatter_nd_update(BetaSel, indices, updates)
    
    for i in tf.range(ncsel):
        for spg in tf.range(q.shape[1]):
            X2 = tf.gather(X1, [spg])[-1,:,:] * tf.cast(tf.gather_nd(mask, [i,spg]), dtype)
        
            LFix = tf.matmul(X2, Beta)

            if BetaSelNew[i][spg]:
                LNew = L + LFix
            else:
                LNew = L - LFix
            
            fsp = tf.squeeze(tf.where(tf.equal(tf.gather(spGroup, i),spg)))
            fsp = fsp if len(tf.shape(fsp)) > 1 else fsp[None]
       
            updates = tf.reshape(tfd.Normal(loc=tf.gather(LNew, fsp, axis=1), scale=tf.gather(std, fsp)).cdf(tf.gather(Z, fsp, axis=1)), [-1])
            indices = tf.stack([tf.tile(tf.range(ny), [tf.size(fsp)]), tf.cast(tf.repeat(fsp, ny), tf.int32)], axis=1)
            llNew = tf.tensor_scatter_nd_update(ll, indices, updates)
            
            lldif = tf.reduce_sum(tf.gather(llNew, fsp, axis=1)) - tf.reduce_sum(tf.gather(ll, fsp, axis=1))

            # if tfr.uniform(shape=[1]) > tfm.exp(lldif + tf.gather_nd(pridif, [i,spg])): # autograph error for unsupported op
            if True: 
                BetaSel = tf.tensor_scatter_nd_update(BetaSel, [[i,spg]], [BetaSelNew[i,spg]])
                L = LNew
                ll = llNew
            
    return BetaSel