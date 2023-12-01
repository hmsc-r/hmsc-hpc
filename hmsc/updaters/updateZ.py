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
            truncated_normal_library="tf", dtype=tf.float64,
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

    if X.ndim == 2:
        L = tf.matmul(X, Beta)
    else:
        L = tf.einsum("jik,kj->ij", X, Beta)

    for r, (Eta, Lambda, rLPar) in enumerate(zip(EtaList, LambdaList, rLHyperparams)):
        xMat = rLPar.get("xMat")
        if xMat is None:
            L2 = tf.matmul(Eta, Lambda)
        else:
            L2 = tf.einsum("ih,ik,hjk->ij", Eta, xMat, Lambda)
        L += tf.gather(L2, Pi[:, r])

    Yo = tfm.logical_not(tfm.is_nan(Y))

    indColNormal = tf.where(distr[:,0] == 1)[:, 0]
    indColProbit = tf.where(distr[:,0] == 2)[:, 0]
    indColPoisson = tf.where(distr[:,0] == 3)[:, 0]

    if seed is not None:
        tfr.set_seed(seed + 1)

    ZNormal, iDNormal = calculate_z_normal(
            *gather(Y, Yo, L, sigma, indices=indColNormal),
            dtype=dtype)

    if seed is not None:
        tfr.set_seed(seed + 2)

    ZProbit, iDProbit = calculate_z_probit(
            *gather(Y, Yo, L, sigma, indices=indColProbit),
            truncated_normal_library=truncated_normal_library,
            dtype=dtype)

    if seed is not None:
        tfr.set_seed(seed + 3)

    ZPoisson, iDPoisson, poisson_omega = calculate_z_poisson(
            *gather(Y, Yo, L, sigma, ZPrev, indices=indColPoisson),
            omega=params.get("poisson_omega"),
            preupdate_z=poisson_preupdate_z,
            marginalize_z=poisson_marginalize_z,
            dtype=dtype)

    ZStack = tf.concat([ZNormal, ZProbit, ZPoisson], -1)
    iDStack = tf.concat([iDNormal, iDProbit, iDPoisson], -1)
    indColStack = tf.concat([indColNormal, indColProbit, indColPoisson], 0)
    ZNew = tf.gather(ZStack, tf.argsort(indColStack), axis=-1)
    iDNew = tf.gather(iDStack, tf.argsort(indColStack), axis=-1)
    return ZNew, iDNew, poisson_omega


def gather(*args, indices):
    return (tf.gather(a, indices, axis=-1) for a in args)


def calculate_z_normal(Y, Yo, L, sigma, *, dtype):
    # no data augmentation for normal model in columns with continious unbounded data
    Z = tf.where(Yo, Y, L + sigma * tfr.normal(Y.shape, dtype=dtype))
    iD = tf.cast(Yo, dtype) * sigma**-2
    return Z, iD


def calculate_z_probit(Y, Yo, L, sigma, *, truncated_normal_library, dtype):
    # Albert and Chib (1993) data augemntation for probit model in columns with binary data
    INFTY = 1e+3
    Ym = tfm.logical_not(Yo)
    low = tf.where(tfm.logical_or(Y == 0, Ym), tf.cast(-INFTY, dtype), tf.zeros_like(Y))
    high = tf.where(tfm.logical_or(Y == 1, Ym), tf.cast(INFTY, dtype), tf.zeros_like(Y))
    ny, ns = Y.shape

    if truncated_normal_library == "tfd":
        Z = tfd.TruncatedNormal(loc=L, scale=sigma, low=low, high=high).sample(name="z-ZProbit")
    elif truncated_normal_library == "tf":
        if ns == 0:
            samTN = tf.convert_to_tensor((), dtype=dtype)
        else:
            samTN = parameterized_truncated_normal(shape=[ny*ns], means=tf.reshape(L,[ny*ns]), stddevs=tf.tile(sigma,[ny]),
                                               minvals=tf.reshape(low,[ny*ns]), maxvals=tf.reshape(high,[ny*ns]), dtype=dtype,
                                               name="z-samTN")
        Z = tf.reshape(samTN, [ny,ns])
    elif truncated_normal_library == "scipy":
        loc, scale = tf.reshape(L,[ny*ns]), tf.tile(sigma,[ny])
        a, b = (tf.reshape(low,[ny*ns]) - loc) / scale, (tf.reshape(high,[ny*ns]) - loc) / scale
        Z = tf.reshape(tf.numpy_function(truncnorm.rvs, [a, b, loc, scale], dtype), [ny,ns])

    iD = tf.cast(Yo, dtype) * sigma**-2

    return Z, iD


def calculate_z_poisson(Y, Yo, L, sigma, Z, *,
                        omega,
                        preupdate_z, marginalize_z, dtype):
    # Lognormal Poisson with external PG sampler
    r = tf.constant(1000., dtype=dtype) #Neg-binomial approximation constant

    if preupdate_z:
        Z = sample_z(Y, L, sigma, omega, r, dtype=dtype)

    omega = draw_polya_gamma(Y + r, Z - tfm.log(r), dtype=dtype)
    if marginalize_z:
        # marginalize Z for equivalent effect on Beta, Lambda or Eta. Cannot be used for sigma.
        Z = (Y-r)/(2.*omega) + tfm.log(r)
        iD = tf.cast(Yo, dtype) * (sigma**2. * tf.ones_like(L) + omega**-1)**-1
    else:
        # sample Z. Required for sigma.
        Z = sample_z(Y, L, sigma, omega, r, dtype=dtype)
        iD = tf.cast(Yo, dtype) * sigma**-2.

    return Z, iD, omega


def sample_z(Y, L, sigma, omega, r, dtype):
    sigmaZ2 = (sigma**-2. * tf.ones_like(L) + omega)**-1.
    mu = sigmaZ2*((Y-r)/2. + omega*tfm.log(r) + sigma**-2. * L)
    Z = tfr.normal(Y.shape, mu, tf.sqrt(sigmaZ2), dtype=dtype)
    return Z


def draw_polya_gamma(h, z, dtype):
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
