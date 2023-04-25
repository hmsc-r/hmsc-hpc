import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.random_ops import parameterized_truncated_normal
from polyagamma import random_polyagamma
tfd = tfp.distributions
tfm, tfr = tf.math, tf.random

def updateZ(params, data, poisson_preupdate_z=True, poisson_update_omega=True, poisson_marginalize_z=False, dtype=np.float64):
    """Update conditional updater(s)
    Z - latent variable.
        
    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Beta - species niches
        Eta - site loadings
        Lambda - species loadings
        sigma - residual variance
        Y - community data
        X - environmental data
        Pi - study design
        distr - matrix regulating observation models per outcome
    """
    INFTY = 1e+12

    ZPrev = params["Z"]
    Beta = params["Beta"]
    EtaList = params["Eta"]
    LambdaList = params["Lambda"]
    sigma = params["sigma"]
    X = params["Xeff"]

    Y = data["Y"]
    Pi = data["Pi"]
    distr = data["distr"]
    ny, ns = Y.shape
    nr = len(EtaList)
    
    if len(X.shape.as_list()) == 2: #tf.rank(X)
      LFix = tf.matmul(X, Beta)
    else:
      LFix = tf.einsum("jik,kj->ij", X, Beta)
    LRanLevelList = [None] * nr
    for r, (Eta, Lambda) in enumerate(zip(EtaList, LambdaList)):
      LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:,r]), Lambda)
    L = LFix + sum(LRanLevelList)
    Yo = tfm.logical_not(tfm.is_nan(Y))

    # no data augmentation for normal model in columns with continious unbounded data
    indColNormal = np.nonzero(distr[:,0] == 1)[0]
    YN = tf.gather(Y, indColNormal, axis=-1)
    YoN = tf.gather(Yo, indColNormal, axis=-1)
    LN = tf.gather(L, indColNormal, axis=-1)
    sigmaN = tf.gather(sigma, indColNormal)
    ZNormal = tf.where(YoN, YN, LN + sigmaN * tfr.normal([ny, tf.size(indColNormal)], dtype=dtype))
    iDNormal = tf.cast(YoN, dtype) * sigmaN**dtype(-2)


    # Albert and Chib (1993) data augemntation for probit model in columns with binary data
    indColProbit = np.nonzero(distr[:,0] == 2)[0]
    YP = tf.gather(Y, indColProbit, axis=-1)
    YoP = tf.gather(Yo, indColProbit, axis=-1)
    YmP = tfm.logical_not(YoP)
    LP = tf.gather(L, indColProbit, axis=-1)
    sigmaP = tf.gather(sigma, indColProbit)
    # low = tf.where(tfm.logical_or(YP == 0, YmP), tf.cast(-np.inf, dtype), tf.zeros_like(YP))
    # high = tf.where(tfm.logical_or(YP == 1, YmP), tf.cast(np.inf, dtype), tf.zeros_like(YP))
    # norm = tfd.Normal(LP, sigmaP)
    # samUnif = tf.random.uniform(YP.shape, norm.cdf(low), norm.cdf(high), dtype=dtype)
    # ZP = norm.quantile(samUnif)
    low = tf.where(tfm.logical_or(YP == 0, YmP), tf.cast(-INFTY, dtype), tf.zeros_like(YP))
    high = tf.where(tfm.logical_or(YP == 1, YmP), tf.cast(INFTY, dtype), tf.zeros_like(YP))
    # ZP = tfd.TruncatedNormal(loc=LP, scale=sigmaP, low=low, high=high).sample()
    nsP = tf.size(indColProbit)
    samTN = parameterized_truncated_normal(shape=[ny*nsP], means=tf.reshape(LP,[ny*nsP]), stddevs=tf.tile(sigmaP, [ny]), 
                                           minvals=tf.reshape(low,[ny*nsP]), maxvals=tf.reshape(high,[ny*nsP]), dtype=dtype)
    ZProbit = tf.reshape(samTN, [ny,nsP])
    iDProbit = tf.cast(YoP, dtype) * sigmaP**dtype(-2)
    
    
    # # Lognormal Poisson with 1D slice sampling
    # indColPoisson = np.nonzero(distr[:,0] == 3)[0]
    # YPo = tf.gather(Y, indColPoisson, axis=-1)
    # YoPo = tf.gather(Yo, indColPoisson, axis=-1)
    # LPo = tf.gather(L, indColPoisson, axis=-1)
    # ZPrevPo = tf.gather(ZPrev, indColPoisson, axis=-1)
    # sigmaPo = tf.gather(sigma, indColPoisson)
    # logLike = lambda a: tfm.multiply_no_nan(tfd.Poisson(log_rate=a).log_prob(YPo), tf.cast(YoPo,dtype)) + tfd.Normal(LPo, sigmaPo).log_prob(a)
    # sampler = tfp.mcmc.SliceSampler(logLike, step_size=dtype(1e0), max_doublings=10)
    # kernel_results = sampler.bootstrap_results(ZPrevPo)
    # ZPoisson, kernel_results_new = sampler.one_step(ZPrevPo, kernel_results)
    
    # # Poisson with external PG sampler
    # indColPoisson = np.nonzero(distr[:,0] == 3)[0]
    # YPo = tf.gather(Y, indColPoisson, axis=-1)
    # YoPo = tf.gather(Yo, indColPoisson, axis=-1)
    # LPo = tf.gather(L, indColPoisson, axis=-1)
    # pg_h = tf.reshape(YPo, [-1]) + r
    # pg_z = tf.reshape(LPo, [-1]) - np.log(r) # sign does not matter
    # draw_pg = lambda h,z: random_polyagamma(h, z, disable_checks=True)
    # omega = tf.numpy_function(draw_pg, [pg_h, pg_z], dtype)
    # iDPoisson = tf.cast(YoPo, dtype) * tf.reshape(omega, YPo.shape)
    # ZPoisson = tf.reshape((0.5*(tf.reshape(YPo, [-1]) - r)) / omega + np.log(r), YPo.shape)

    # Lognormal Poisson with external PG sampler
    r = 1000 #Neg-binomial approximation constant
    indColPoisson = np.nonzero(distr[:,0] == 3)[0]
    YPo = tf.gather(Y, indColPoisson, axis=-1)
    YoPo = tf.gather(Yo, indColPoisson, axis=-1)
    LPo = tf.gather(L, indColPoisson, axis=-1)
    sigmaPo = tf.gather(sigma, indColPoisson)
        
    if poisson_preupdate_z == False:
      ZPo = tf.gather(ZPrev, indColPoisson, axis=-1)
    else:
      omega = params["poisson_omega"]
      sigmaZ2 = (sigmaPo**-2. * tf.ones_like(LPo) + omega)**-1.
      muZ = sigmaZ2*((YPo-r)/2. + omega*np.log(r) + sigmaPo**-2. * LPo)
      ZPo = tfr.normal(YPo.shape, muZ, tf.sqrt(sigmaZ2), dtype=dtype)
    
    omega = draw_polya_gamma(YPo + r, ZPo - np.log(r), dtype=dtype)
    if poisson_marginalize_z == False:
      # sample Z. Required for sigma.
      sigmaZ2 = (sigmaPo**-2. * tf.ones_like(LPo) + omega)**-1.
      muZ = sigmaZ2*((YPo-r)/2. + omega*np.log(r) + sigmaPo**-2. * LPo)
      ZPoisson = tfr.normal(YPo.shape, muZ, tf.sqrt(sigmaZ2), dtype=dtype)
      iDPoisson = tf.cast(YoPo, dtype) * sigmaPo**-2.
    else:
      # marginalize Z for equivalent effect on Beta, Lambda or Eta. Cannot be used for sigma.
      iDPoisson = tf.cast(YoPo, dtype) * (sigmaPo**2. * tf.ones_like(LPo) + omega**-1)**-1
      ZPoisson = (YPo-r)/(2.*omega) + np.log(r)
    poisson_omega = omega
    

    ZStack = tf.concat([ZNormal, ZProbit, ZPoisson], -1)
    iDStack = tf.concat([iDNormal, iDProbit, iDPoisson], -1)
    indColStack = tf.concat([indColNormal, indColProbit, indColPoisson], 0)
    ZNew = tf.gather(ZStack, tf.argsort(indColStack), axis=-1)
    iDNew = tf.gather(iDStack, tf.argsort(indColStack), axis=-1)
    return ZNew, iDNew, poisson_omega



def draw_polya_gamma(h, z, N=10, dtype=np.float64):
  pg_h = tf.reshape(h, [-1])
  pg_z = tf.reshape(z, [-1]) # sign does not matter
  draw_pg = lambda h,z: random_polyagamma(h, z, disable_checks=True)
  omega = tf.reshape(tf.numpy_function(draw_pg, [pg_h, pg_z], dtype), h.shape)
  return omega