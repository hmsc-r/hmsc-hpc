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

    X1 = modelData["X"]
    Pi = modelData["Pi"]

    covGroup = modelData["covGroup"]
    spGroup = modelData["spGroup"]
    q = modelData["q"]

    Z = params["Z"]
    iD = params["iD"]
    EtaList = params["Eta"]
    Beta = params["Beta"]
    BetaSel = params["BetaSel"]
    LambdaList = params["Lambda"]
    sigma = params["sigma"]

    iSigma = 1 / sigma

    std = iSigma**-0.5

    LRanLevelList = [None] * nr
    for r, (Eta, Lambda) in enumerate(zip(EtaList, LambdaList)):
        LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:,r]), Lambda)
        
    X1 = tf.stack([X1 for i in range(ns)])

    for i in range(ncsel):
    fsp = tf.cast(tf.transpose(tf.stack([tf.where(tf.equal(spGroup, i))[:,1] for i in range(ns)])), dtype=tf.int32)
    fsp_selected = tf.where(tf.equal(BetaSel, False), fsp, -1)
    mx_fsp = tf.reduce_max(fsp_selected, axis=1)

    mask_nc = tf.ones((nc,ns), dtype=tf.bool)
    
    def f1(x):
        return tf.where(tf.less_equal(tf.range(ns), mx_fsp[x]), False, True)
    mask_ncsel = tf.stack(tf.map_fn(f1, tf.range(ncsel), fn_output_signature=tf.bool))
    #mask_ncsel = tf.stack([tf.where(tf.less_equal(tf.range(ns), x), False, True) for x in mx_fsp])

    indices = tf.stack([tf.repeat(covGroup, ns), tf.tile(tf.range(ns), [ncsel])], axis=1)
    updates = tf.reshape(mask_ncsel, [-1])
    mask_nc = tf.tensor_scatter_nd_update(mask_nc, indices, updates)

    mask = tf.transpose(tf.reshape(tf.repeat(mask_nc, tf.ones([ns], dtype=tf.int32)*ny, axis=1), (nc,ny,ns)))

    X = X1*tf.cast(mask, dtype)
    LFix = tf.einsum("ijk,ki->ji", X, Beta)
    L = LFix + sum(LRanLevelList)

    ll = tfd.Normal(loc=L, scale=std).log_prob(Z)

    BetaSelNew = tf.where(BetaSel, False, True)

    fsp = tf.cast(tf.transpose(tf.stack([tf.where(tf.equal(spGroup, i))[:,1] for i in range(ns)])), dtype=tf.int64)

    mx_fsp = tf.cast(tf.reduce_max(fsp, axis=1), dtype=tf.int32)

    mask_nc = tf.zeros((nc,ns), dtype=tf.int32)
    
    def f2(x):
        return tf.where(tf.less_equal(tf.range(ns), mx_fsp[x]), 1, 0)
    mask_ncsel = tf.stack(tf.map_fn(f2, tf.range(ncsel), fn_output_signature=tf.int32))
    #mask_ncsel = tf.stack([tf.where(tf.less_equal(tf.range(ns), x), 1, 0) for x in mx_fsp])
    mask_ncsel = tf.where(BetaSelNew, mask_ncsel, -1 * mask_ncsel)

    indices = tf.stack([tf.repeat(covGroup, ns), tf.tile(tf.range(ns), [ncsel])], axis=1)
    updates = tf.reshape(mask_ncsel, [-1])
    mask_nc = tf.tensor_scatter_nd_update(mask_nc, indices, updates)

    mask = tf.transpose(tf.reshape(tf.repeat(mask_nc, tf.ones([ns], dtype=tf.int32)*ny, axis=1), (nc,ny,ns)))

    X2 = X1*tf.cast(mask, dtype)

    LFix1 = tf.einsum("ijk,ki->ji", X2, Beta)

    pridif = tf.where(BetaSel, tfm.log(q) - tfm.log(1-q), tfm.log(1-q) - tfm.log(q))


        for spg in range(ns):
            if BetaSelNew[i,spg]:
                LNew = L + LFix1
            else:
                LNew = L - LFix1
        
            llNew = ll
            
            j = fsp[i,spg]
            indices = tf.stack([tf.cast(tf.range(ny), tf.int64), tf.repeat(j, ny)], axis=1)
            updates = tf.reshape(tfd.Normal(loc=LNew[:,j], scale=std[j]).cdf(Z[i,j]), [-1])
            llNew = tf.tensor_scatter_nd_update(llNew, indices, updates)
            
            lldif = tf.reduce_sum(llNew[:,spg]) - tf.reduce_sum(ll[:,spg])
            '''
            if tfm.exp(lldif + pridif[i,spg]) > tfr.uniform([1], dtype=dtype): # autograph error for unsupported op
                BetaSel = BetaSelNew
                #BetaSel = tf.tensor_scatter_nd_update(BetaSel, [[i,spg]], [BetaSelNew[i,spg]])
                tf.print(BetaSel)
                L = LNew
                ll = llNew
            '''
            BetaSel = tf.tensor_scatter_nd_update(BetaSel, [[i,spg]], [BetaSelNew[i,spg]])
            L = LNew
            ll = llNew
            
    return BetaSel