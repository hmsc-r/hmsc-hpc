import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hmsc.utils.tf_named_func import tf_named_func
tfm, tfla, tfd, tfb = tf.math, tf.linalg, tfp.distributions, tfp.bijectors

@tf.function
def logProb(Beta, Gamma, LiV, sigma, LRan, Y, X, Tr, priorHyperparams, dtype=np.float64):
  Mu = tf.matmul(Gamma, Tr, transpose_b=True)
  # print(iV)
  # logLikeBeta1 = tfd.MultivariateNormalFullCovariance(tfla.matrix_transpose(Mu), tfla.inv(iV)).log_prob(tfla.matrix_transpose(Beta))
  BM = Beta - Mu
  # LiV = tfla.cholesky(iV)
  logDetV = -2*tf.reduce_sum(tfm.log(tfla.diag_part(LiV)))
  # logLikeBeta = -0.5*tf.einsum("cj,ck,kj->j",BM,iV,BM) - 0.5*logDetV - tf.cast(Beta.shape[0],dtype)/2*tfm.log(2*tf.cast(np.pi,dtype))
  qFBM = tf.reduce_sum(tf.matmul(LiV,BM,transpose_a=True)**2, 0)
  logLikeBeta = -0.5*qFBM - 0.5*logDetV - tf.cast(Beta.shape[0],dtype)/2*tfm.log(2*tf.cast(np.pi,dtype))

  # tf.print(logLikeBeta1 - logLikeBeta)

  
  if len(X.shape.as_list()) == 2: #tf.rank(X) X.ndim == 2:
      LFix = tf.matmul(X, Beta)
  else:
      LFix = tf.einsum("jik,kj->ij", X, Beta)
  L = LFix + LRan
  
  #logLikeY = tfd.Normal(L, sigma).log_prob(Z)
  obsDist = tfd.Normal(L,sigma)
  logLikeY0 = tfm.multiply_no_nan(obsDist.log_cdf(0), tf.cast(Y==0,dtype))
  logLikeY1 = tfm.multiply_no_nan(obsDist.log_survival_function(0), tf.cast(Y==1,dtype))
  logLikeY = logLikeY0 + logLikeY1
  
  V0 = priorHyperparams["V0"]
  f0 = priorHyperparams["f0"]
  LiV0 = tfla.cholesky(tfla.inv(V0))
  logPriorV = tfd.WishartTriL(f0, LiV0).log_prob(tf.matmul(LiV,LiV,transpose_b=True))
  mGamma = priorHyperparams["mGamma"]
  iUGamma = priorHyperparams["iUGamma"]
  logPriorGamma = tfd.MultivariateNormalFullCovariance(mGamma, iUGamma).log_prob(tf.reshape(tfla.matrix_transpose(Gamma), [-1]))
  # tf.print(logPriorV, logPriorGamma)
  # tf.print(logLikeY.shape, logLikeBeta.shape)
  log_prob = tf.reduce_sum(logLikeBeta) + tf.reduce_sum(logLikeY) + logPriorV + logPriorGamma
  return(log_prob)

  
  
@tf_named_func("hmc")
def updateHMC(params, data, priorHyperparams, rLHyperparams, sample_burnin, kernel_results, dtype=tf.float64):
    Y = data["Y"]
    X = params["Xeff"]
    Tr = data["T"]
    sigma = params["sigma"]

    EtaList = params["Eta"]
    LambdaList = params["Lambda"]
    Pi = data["Pi"]
    distr = data["distr"]
    LRanList = []
    for r, (Eta, Lambda, rLPar) in enumerate(zip(EtaList, LambdaList, rLHyperparams)):
        xMat = rLPar.get("xMat")
        if xMat is None:
            LRan = tf.matmul(Eta, Lambda)
        else:
            LRan = tf.einsum("ih,ik,hjk->ij", Eta, xMat, Lambda)
        LRanList.append(tf.gather(LRan, Pi[:,r]))
    LRan = tf.add_n([tf.zeros(Y.shape, dtype)] + LRanList)
    # LRan = tf.zeros(Y.shape, dtype)
    
    unnormalized_log_prob = lambda Beta, Gamma, LiV: logProb(Beta, Gamma, LiV, sigma, LRan, Y, X, Tr, priorHyperparams)
    bijectorList = [tfb.Identity(), tfb.Identity(), tfb.FillScaleTriL(diag_shift=tf.constant(1e-6,dtype))]
    hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=unnormalized_log_prob, num_leapfrog_steps=10, step_size=0.01)
    hmc_unconstrained = tfp.mcmc.TransformedTransitionKernel(hmc, bijector=bijectorList)
    hmc_unconstrained_adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(hmc_unconstrained, int(0.8*float(sample_burnin)))

    Beta = params["Beta"]
    Gamma = params["Gamma"]
    LiV = tfla.cholesky(params["iV"])
    log_prob = logProb(Beta, Gamma, LiV, sigma, LRan, Y, X, Tr, priorHyperparams, dtype=tf.float64)
    tf.print(log_prob)
    
    current_state = [Beta, Gamma, LiV]
    tmp = hmc_unconstrained_adaptive.bootstrap_results(current_state)
    inner_tmp = hmc_unconstrained_adaptive.step_size_setter_fn(tmp.inner_results, kernel_results.new_step_size)
    tmp = tmp._replace(step=kernel_results.step, inner_results=inner_tmp)
    tmp = tmp._replace(new_step_size=kernel_results.new_step_size, log_averaging_step=kernel_results.log_averaging_step)
    tmp = tmp._replace(error_sum=kernel_results.error_sum)
    kernel_results = tmp
    inner = kernel_results.inner_results.inner_results
    tf.print(kernel_results.step, inner.accepted_results.step_size)
    next_state, next_kernel_results = hmc_unconstrained_adaptive.one_step(current_state, kernel_results)
    tf.print(next_kernel_results.inner_results.inner_results.is_accepted, next_kernel_results.new_step_size)
    
    Beta, Gamma, LiV = next_state
    iV = tf.matmul(LiV, LiV, transpose_b=True)
    return Beta, Gamma, iV, next_kernel_results
    
def get_hmc_kernel_results(params, data, priorHyperparams, rLHyperparams, sample_burnin, dtype=tf.float64):
    Y = data["Y"]
    X = params["Xeff"]
    Tr = data["T"]
    sigma = params["sigma"]
    LRan = tf.zeros(Y.shape, dtype)
    
    unnormalized_log_prob = lambda Beta, Gamma, LiV: logProb(Beta, Gamma, LiV, sigma, LRan, Y, X, Tr, priorHyperparams)
    bijectorList = [tfb.Identity(), tfb.Identity(), tfb.FillScaleTriL(diag_shift=tf.constant(1e-6,dtype))]
    hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=unnormalized_log_prob, num_leapfrog_steps=2, step_size=0.01)
    hmc_unconstrained = tfp.mcmc.TransformedTransitionKernel(hmc, bijector=bijectorList)
    hmc_unconstrained_adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(hmc_unconstrained, int(0.8*float(sample_burnin)))

    Beta = params["Beta"]
    Gamma = params["Gamma"]
    LiV = tfla.cholesky(params["iV"])
    log_prob = logProb(Beta, Gamma, LiV, sigma, LRan, Y, X, Tr, priorHyperparams, dtype=tf.float64)
    tf.print(log_prob)
    
    current_state = [Beta, Gamma, LiV]
    kernel_results = hmc_unconstrained_adaptive.bootstrap_results(current_state)
    
    return kernel_results
  

# from matplotlib import pyplot as plt 
# from scipy.linalg import block_diag
# import time
# import os

# ny = 99
# ns = 151
# nc = 14
# nt = 7
# dtype = np.float64
# tf.keras.utils.set_random_seed(42)

# Tr = np.random.normal(dtype(0), size=[ns,nt]); Tr[:,0] = 1
# X = np.random.normal(dtype(0), 0.2, size=[ny,nc]); X[:,0] = 1
# Gamma = np.random.normal(dtype(0), size=[nc,nt])
# VBlockList = [np.array([[1,1],[1,1]])] * int(nc/2)
# if nc % 2: VBlockList = VBlockList + [np.array([[1]])]
# V = (block_diag(*VBlockList).astype(dtype) + np.eye(nc)) / 10
# Mu = tf.matmul(Gamma, Tr, transpose_b=True)
# Beta = tf.transpose(tfd.MultivariateNormalFullCovariance(tf.transpose(Mu), V).sample())
# L = tf.matmul(X, Beta)
# sigma = np.ones([ns], dtype)
# pr = tfd.Normal(L,sigma).survival_function(0)
# plt.scatter(L, pr)
# Y = tfd.Bernoulli(probs=pr).sample()


# priorHyperparams = {"f0": nc+1, "V0": np.eye(nc,dtype=dtype), 
#                     "mGamma": np.zeros([nc*nt],dtype), "iUGamma": np.eye(nc*nt,dtype=dtype)}
# params = {"Beta": Beta, "Gamma": Gamma, "iV": tfla.inv(V), "sigma": sigma, "Xeff": tf.constant(X,dtype)}
# data = {"Y": Y, "T": Tr}

# updateHMC(params, data, priorHyperparams, rLHyperparams=None, dtype=tf.float64)

# LRan = tf.zeros(Y.shape, dtype)
# unnormalized_log_prob = lambda Beta, Gamma, LiV: logProb(Beta, Gamma, LiV, sigma, LRan, Y, tf.constant(X,dtype), Tr, priorHyperparams)

# n_sam = 50
# transient = n_sam

# bijectorList = [tfb.Identity(), tfb.Identity(), tfb.FillScaleTriL(diag_shift=dtype(1e-6))]
# hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=unnormalized_log_prob, num_leapfrog_steps=10, step_size=0.01)
# hmc_unconstrained = tfp.mcmc.TransformedTransitionKernel(hmc, bijector=bijectorList)
# hmc_unconstrained_adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(hmc_unconstrained, int(0.8*transient))

# iV = tfla.inv(V)
# LiV = tfla.cholesky(iV)
# initial_state=[Beta, Gamma, LiV]
# initial_state=[tf.zeros_like(Beta), tf.zeros_like(Gamma), tf.eye(nc,dtype=dtype)]
# trace_fn = lambda _, pkr: [pkr.inner_results.inner_results.accepted_results.step_size,
#                            pkr.inner_results.inner_results.log_accept_ratio,
#                            pkr.inner_results.inner_results.is_accepted]


# aaa
# start_time = time.time()
# mcmc_res = tfp.mcmc.sample_chain(transient+n_sam, initial_state, kernel=hmc_unconstrained_adaptive, num_burnin_steps=0, trace_fn=trace_fn)
# stop_time = time.time()
# print(stop_time - start_time)
# mcmc_res[1]

# print(mcmc_res[1])
# plt.subplot(1, 3, 1)
# plt.scatter(Beta, tf.reduce_mean(mcmc_res[0][0][transient:], 0))
# plt.title("Beta")
# plt.subplot(1, 3, 2)
# plt.scatter(Gamma, tf.reduce_mean(mcmc_res[0][1][transient:], 0))
# plt.title("Gamma")
# plt.subplot(1, 3, 3)
# LiVPost = mcmc_res[0][2][transient:]
# plt.scatter(V, tf.reduce_mean(tfla.inv(tf.matmul(LiVPost,LiVPost,transpose_b=True)), 0))
# plt.title("V")
# plt.show()

# beep = lambda x: os.system("echo -n '\a';sleep 0.1;" * x)
# beep(3)


# tf.keras.utils.set_random_seed(42)
# kernel_results = hmc_unconstrained_adaptive.bootstrap_results(initial_state)
# current_state = initial_state
# print(current_state[0][0,0])
# for i in range(n_sam+transient):
#   tmp = hmc_unconstrained_adaptive.bootstrap_results(current_state)
#   inner_tmp = hmc_unconstrained_adaptive.step_size_setter_fn(tmp.inner_results, kernel_results.new_step_size)
#   tmp = tmp._replace(step=kernel_results.step, inner_results=inner_tmp)
#   tmp = tmp._replace(new_step_size=kernel_results.new_step_size, log_averaging_step=kernel_results.log_averaging_step)
#   tmp = tmp._replace(error_sum=kernel_results.error_sum)
#   kernel_results = tmp
#   inner = kernel_results.inner_results.inner_results
#   print(kernel_results.step.numpy(), '%.6f'%inner.accepted_results.step_size.numpy())
#   next_state, next_kernel_results = hmc_unconstrained_adaptive.one_step(current_state, kernel_results)
#   print(next_kernel_results.inner_results.inner_results.is_accepted.numpy(), '%.6f'%next_kernel_results.new_step_size.numpy())
#   current_state = next_state
#   kernel_results = next_kernel_results
#   print(current_state[0].numpy()[0,0])

# #  auto
# # True 0.010153
# # 0.08385276510975985
# # 98 0.010153
# # True 0.010153
# # -0.031031616571357876
# # 99 0.010153
# # True 0.010153
# # 0.12239301308455708