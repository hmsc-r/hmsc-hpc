import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.random_ops import parameterized_truncated_normal
from scipy.stats import truncnorm
from hmsc.utils.tf_named_func import tf_named_func
tfm, tfr = tf.math, tf.random
tfd = tfp.distributions


@tf_named_func("z")
def updateZ(params, data, rLHyperparams, *,
            poisson_preupdate_z=True, poisson_marginalize_z=False,
            truncated_normal_library="tf", dtype=np.float64,
            seed=None):
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
    if seed is not None:
        tfr.set_seed(seed)

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

    if X.ndim == 2:
      LFix = tf.matmul(X, Beta)
    else:
      LFix = tf.einsum("jik,kj->ij", X, Beta)
    LRanLevelList = [None] * nr
    for r, (Eta, Lambda, rLPar) in enumerate(zip(EtaList, LambdaList, rLHyperparams)):
      if rLPar["xDim"] == 0:
        LRanLevelList[r] = tf.gather(tf.matmul(Eta, Lambda), Pi[:,r])
      else:
        LRanLevelList[r] = tf.gather(tf.einsum("ih,ik,hjk->ij", Eta, rLPar["xMat"], Lambda), Pi[:,r])
    L = LFix + sum(LRanLevelList)
    Yo = tfm.logical_not(tfm.is_nan(Y))

    indColNormal = np.nonzero(distr[:,0] == 1)[0]
    indColProbit = np.nonzero(distr[:,0] == 2)[0]
    indColPoisson = np.nonzero(distr[:,0] == 3)[0]

    ZNormal, iDNormal = calculate_z_normal(
            indColNormal, Y, Yo, L, sigma,
            dtype=dtype)
    ZProbit, iDProbit = calculate_z_probit(
            indColProbit, Y, Yo, L, sigma,
            truncated_normal_library=truncated_normal_library,
            dtype=dtype)
    ZPoisson, iDPoisson, poisson_omega = calculate_z_poisson(
            indColPoisson, Y, Yo, L, sigma, ZPrev,
            omega=params.get("poisson_omega"),
            poisson_preupdate_z=poisson_preupdate_z,
            poisson_marginalize_z=poisson_marginalize_z,
            dtype=dtype)

    ZStack = tf.concat([ZNormal, ZProbit, ZPoisson], -1)
    iDStack = tf.concat([iDNormal, iDProbit, iDPoisson], -1)
    indColStack = tf.concat([indColNormal, indColProbit, indColPoisson], 0)
    ZNew = tf.gather(ZStack, tf.argsort(indColStack), axis=-1)
    iDNew = tf.gather(iDStack, tf.argsort(indColStack), axis=-1)
    return ZNew, iDNew, poisson_omega


def calculate_z_normal(indColNormal, Y, Yo, L, sigma, *, dtype):
    # no data augmentation for normal model in columns with continious unbounded data
    ny, ns = Y.shape
    YN = tf.gather(Y, indColNormal, axis=-1)
    YoN = tf.gather(Yo, indColNormal, axis=-1)
    LN = tf.gather(L, indColNormal, axis=-1)
    sigmaN = tf.gather(sigma, indColNormal)
    ZNormal = tf.where(YoN, YN, LN + sigmaN * tfr.normal([ny, tf.size(indColNormal)], dtype=dtype))
    iDNormal = tf.cast(YoN, dtype) * sigmaN**-2
    return ZNormal, iDNormal


def calculate_z_probit(indColProbit, Y, Yo, L, sigma, *, truncated_normal_library, dtype):
    # Albert and Chib (1993) data augemntation for probit model in columns with binary data
    INFTY = 1e+3
    ny, ns = Y.shape
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
    nsP = tf.size(indColProbit)

    if truncated_normal_library == "tfd":
      ZProbit = tfd.TruncatedNormal(loc=LP, scale=sigmaP, low=low, high=high).sample(name="z-ZProbit")
    elif truncated_normal_library == "tf":
      if nsP == 0:
        samTN = tf.convert_to_tensor((), dtype=dtype)
      else:
        samTN = parameterized_truncated_normal(shape=[ny*nsP], means=tf.reshape(LP,[ny*nsP]), stddevs=tf.tile(sigmaP,[ny]),
                                               minvals=tf.reshape(low,[ny*nsP]), maxvals=tf.reshape(high,[ny*nsP]), dtype=dtype,
                                               name="z-samTN")
      ZProbit = tf.reshape(samTN, [ny,nsP])
    elif truncated_normal_library == "scipy":
      loc, scale = tf.reshape(LP,[ny*nsP]), tf.tile(sigmaP,[ny])
      a, b = (tf.reshape(low,[ny*nsP]) - loc) / scale, (tf.reshape(high,[ny*nsP]) - loc) / scale
      ZProbit = tf.reshape(tf.numpy_function(truncnorm.rvs, [a, b, loc, scale], dtype), [ny,nsP])

    iDProbit = tf.cast(YoP, dtype) * sigmaP**-2

    return ZProbit, iDProbit


def calculate_z_poisson(indColPoisson, Y, Yo, L, sigma, ZPrev, *,
                        omega,
                        poisson_preupdate_z, poisson_marginalize_z, dtype):
    # Lognormal Poisson with external PG sampler
    r = 1000 #Neg-binomial approximation constant
    YPo = tf.gather(Y, indColPoisson, axis=-1)
    YoPo = tf.gather(Yo, indColPoisson, axis=-1)
    LPo = tf.gather(L, indColPoisson, axis=-1)
    sigmaPo = tf.gather(sigma, indColPoisson)

    if poisson_preupdate_z == False:
      ZPo = tf.gather(ZPrev, indColPoisson, axis=-1)
    else:
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
    return ZPoisson, iDPoisson, poisson_omega


def draw_polya_gamma(h, z, dtype=np.float64):
  # with h > 50 normal approx is used, so we reimplement only that alternative
  # pg_h = tf.reshape(h, [-1])
  # pg_z = tf.reshape(z, [-1]) # sign does not matter
  # draw_pg = lambda h,z: random_polyagamma(h, z, disable_checks=True)
  # omega = tf.reshape(tf.numpy_function(draw_pg, [pg_h, pg_z], dtype), h.shape)
  m0 = 0.25 * h
  s0 = tf.sqrt(h / 24.)
  x1 = tfm.tanh(0.5 * z)
  m1 = 0.5 * h * x1 / z
  s1 = tf.sqrt(0.25 * h * (tfm.sinh(z) - z) * (1. - x1**2) / z**3)
  m = tf.where(z == 0, m0, m1)
  s = tf.where(z == 0, s0, s1)
  # formula in package does not have tf.abs, I added it here to ensure positiveness
  omega = tf.abs(m + s*tfr.normal(h.shape, dtype=dtype))
  return omega
