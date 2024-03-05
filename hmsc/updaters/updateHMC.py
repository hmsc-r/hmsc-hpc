import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hmsc.utils.tf_named_func import tf_named_func
tfm, tfla, tfd, tfb = tf.math, tf.linalg, tfp.distributions, tfp.bijectors

# @tf.function    # shall be commented for HMSC-HPC run
def logProb(Beta, Gamma, LiV, sigma, EtaList, LambdaList, DeltaList, Y, X, Tr, Pi, priorHyperparams, rLHyperparams, dtype=np.float64):
  Mu = tf.matmul(Gamma, Tr, transpose_b=True)
  BM = Beta - Mu
  logDetV = -2*tf.reduce_sum(tfm.log(tfla.diag_part(LiV)))
  qFBM = tf.reduce_sum(tf.matmul(LiV,BM,transpose_a=True)**2, 0)
  logLikeBeta = -0.5*qFBM - 0.5*logDetV - tf.cast(Beta.shape[0],dtype)/2*tfm.log(2*tf.cast(np.pi,dtype))

  if len(X.shape.as_list()) == 2: #tf.rank(X) X.ndim == 2:
      LFix = tf.matmul(X, Beta)
  else:
      LFix = tf.einsum("jik,kj->ij", X, Beta)
  LRanList = []
  for r, (Eta, Lambda, rLPar) in enumerate(zip(EtaList, LambdaList, rLHyperparams)):
      LRan = tf.matmul(Eta, Lambda)
      LRanList.append(tf.gather(LRan, Pi[:,r]))
  L = tf.add_n([LFix] + LRanList)
  
  #logLikeY = tfd.Normal(L, sigma).log_prob(Y)
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
  logPriorGamma = tfd.MultivariateNormalTriL(mGamma, tfla.cholesky(iUGamma)).log_prob(tf.reshape(tfla.matrix_transpose(Gamma), [-1]))
  logProbFix = tf.reduce_sum(logLikeBeta) + logPriorV + logPriorGamma
  
  logLikeEtaList = []
  logLambdaEtaList = []
  logDeltaEtaList = []
  for r, (Eta, Lambda, Delta, rLPar) in enumerate(zip(EtaList, LambdaList, DeltaList, rLHyperparams)):
    nf = tf.shape(Lambda)[0]
    nu = rLPar["nu"]
    a1 = rLPar["a1"]
    b1 = rLPar["b1"]
    a2 = rLPar["a2"]
    b2 = rLPar["b2"]
    aDelta = tf.concat([a1 * tf.ones([1, 1], dtype), a2 * tf.ones([nf-1, 1], dtype)], 0)
    bDelta = tf.concat([b1 * tf.ones([1, 1], dtype), b2 * tf.ones([nf-1, 1], dtype)], 0)
    Tau = tfm.cumprod(Delta, 0)
    
    if rLPar["sDim"] == 0:
      llEta = tfd.Normal(tf.cast(0,dtype),tf.cast(1,dtype)).log_prob(Eta)
    else:
      alphapw = rLPar["alphapw"]
      detWg = rLPar["detWg"]
      # iWg = rLPar["iWg"]
      # EtaTiWEta = tf.einsum("ah,gab,bh->hg", Eta, iWg, Eta)
      LiWg = rLPar["LiWg"]
      EtaTiWEta = tf.transpose(tf.reduce_sum(tf.matmul(LiWg, Eta, transpose_a=True)**2, -2))
      logLike = tfm.log(alphapw[:,1]) - 0.5*detWg - 0.5*EtaTiWEta
      llEta = tfm.reduce_logsumexp(logLike, -1)
      
    llLambda = tfd.StudentT(nu,tf.cast(0,dtype),tfm.rsqrt(Tau)).log_prob(Lambda)
    llDelta = tfd.Gamma(aDelta,bDelta).log_prob(Delta)
    logLikeEtaList.append(tf.reduce_sum(llEta))
    logLambdaEtaList.append(tf.reduce_sum(llLambda))
    logDeltaEtaList.append(tf.reduce_sum(llDelta))
    
  logLikeEta = tf.add_n([tf.cast(0,dtype)] + logLikeEtaList)
  logLikeLambda = tf.add_n([tf.cast(0,dtype)] + logLambdaEtaList)
  logLikeDelta = tf.add_n([tf.cast(0,dtype)] + logDeltaEtaList)
  logProbRan = logLikeEta + logLikeLambda + logLikeDelta

  # tf.print(logPriorV, logPriorGamma)
  # tf.print(logLikeY.shape, logLikeBeta.shape)
  log_prob =  tf.reduce_sum(logLikeY) + logProbFix + logProbRan
  
  return log_prob


@tf_named_func("hmc")
# @tf.function    # shall be commented for HMSC-HPC run
def updateHMC(params, data, priorHyperparams, rLHyperparams, num_leapfrog_steps=10, sample_burnin=0,
              step=0, step_size=0.01, log_averaging_step=None, error_sum=None, init=False,
              updateBeta=True, updateGamma=False, updateiV=False,
              updateEta=True, updateLambda=True, updateDelta=False, dtype=tf.float64):
    Y = data["Y"]
    X = params["Xeff"]
    Tr = data["T"]
    Pi = data["Pi"]
    Beta = params["Beta"]
    Gamma = params["Gamma"]
    sigma = params["sigma"]
    EtaList = params["Eta"]
    LambdaList = params["Lambda"]
    DeltaList = params["Delta"]
    distr = data["distr"]
    nr = len(EtaList)
    LiV = tfla.cholesky(params["iV"])
    # log_prob = logProb(Beta, Gamma, LiV, sigma, EtaList, LambdaList, DeltaList, Y, X, Tr, Pi, priorHyperparams, rLHyperparams, dtype=dtype)
    # tf.print(log_prob)
    # aaa
        
    def log_prob_flat(*argv):
      nonlocal Beta, Gamma, LiV, EtaList, LambdaList, DeltaList
      offset = 0
      if updateBeta:
        Beta = argv[offset]
        offset += 1        
      if updateGamma:
        Gamma = argv[offset]
        offset += 1
      if updateiV:
        LiV = argv[offset]
        offset += 1
      if updateEta:
        EtaList = argv[offset:offset+nr]
        offset += nr
      if updateLambda:
        LambdaList = argv[offset:offset+nr]
        offset += nr
      if updateDelta:
        DeltaList = argv[offset:offset+nr]
        offset += nr
      return logProb(Beta, Gamma, LiV, sigma, EtaList, LambdaList, DeltaList, Y, X, Tr, Pi, priorHyperparams, rLHyperparams, dtype)
    
    bijectorList_flat, current_state_flat = [], []
    if updateBeta:
      bijectorList_flat += [tfb.Identity()]
      current_state_flat += [Beta]
    if updateGamma:
      bijectorList_flat += [tfb.Identity()]
      current_state_flat += [Gamma]
    if updateiV:
      bijectorList_flat += [tfb.FillScaleTriL(diag_shift=tf.constant(1e-6,dtype))]
      current_state_flat += [LiV]
    if updateEta:
      bijectorList_flat += [tfb.Identity()]*nr
      current_state_flat += EtaList
    if updateLambda:
      bijectorList_flat += [tfb.Identity()]*nr
      current_state_flat += LambdaList
    if updateDelta:
      bijectorList_flat += [tfb.Softplus()]*nr
      current_state_flat += DeltaList
    
    hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=log_prob_flat, num_leapfrog_steps=num_leapfrog_steps, step_size=tf.constant(0.01,dtype))
    hmc_unconstrained = tfp.mcmc.TransformedTransitionKernel(hmc, bijector=bijectorList_flat)
    hmc_unconstrained_adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(hmc_unconstrained, int(0.8*float(sample_burnin)))
    
    if init == False:
      tmp = hmc_unconstrained_adaptive.bootstrap_results(current_state_flat)
      inner_tmp = hmc_unconstrained_adaptive.step_size_setter_fn(tmp.inner_results, step_size)
      tmp = tmp._replace(step=step, inner_results=inner_tmp, new_step_size=step_size, log_averaging_step=log_averaging_step)
      tmp = tmp._replace(error_sum=error_sum)
      kernel_results = tmp
      inner = kernel_results.inner_results.inner_results
      # tf.print("step", kernel_results.step, "step size", inner.accepted_results.step_size)
      # tf.print(inner.accepted_results.target_log_prob)
      next_state_flat, next_kernel_results = hmc_unconstrained_adaptive.one_step(current_state_flat, kernel_results)
      # tf.print(next_kernel_results.inner_results.inner_results.proposed_results.target_log_prob)
      # tf.print(next_kernel_results.inner_results.inner_results.is_accepted, next_kernel_results.new_step_size, tfm.exp(next_kernel_results.log_averaging_step))
      offset = 0
      if updateBeta:
        Beta = next_state_flat[offset]
        offset += 1
      if updateGamma:
        Gamma = next_state_flat[offset]
        offset += 1
      if updateiV:
        LiV = next_state_flat[offset]
        offset += 1
      if updateEta:
        EtaList = next_state_flat[offset:offset+nr]
        offset += nr
      if updateLambda:
        LambdaList = next_state_flat[offset:offset+nr]
        offset += nr
      if updateDelta:
        DeltaList = next_state_flat[offset:offset+nr]
        offset += nr
    else:
      # print(current_state_flat)
      next_kernel_results = hmc_unconstrained_adaptive.bootstrap_results(current_state_flat)
    
    new_step_size = next_kernel_results.new_step_size
    new_log_averaging_step  = next_kernel_results.log_averaging_step
    new_error_sum = next_kernel_results.error_sum
    iV = tf.matmul(LiV, LiV, transpose_b=True)
    return Beta, Gamma, iV, EtaList, LambdaList, DeltaList, new_step_size, new_log_averaging_step, new_error_sum
    
  

# from matplotlib import pyplot as plt 
# from scipy.linalg import block_diag
# import time
# import os

# ny = 99
# ns = 155
# nc = 14
# nt = 7
# nf = 3
# dtype = np.float32
# tf.keras.utils.set_random_seed(42)

# Tr = np.random.normal(0, size=[ns,nt]).astype(dtype); Tr[:,0] = 1
# X = np.random.normal(0, 0.2, size=[ny,nc]).astype(dtype); X[:,0] = 1
# Gamma = tfd.Normal(dtype(0), 1).sample([nc,nt])
# VBlockList = [np.array([[1,1],[1,1]])] * int(nc/2)
# if nc % 2: VBlockList = VBlockList + [np.array([[1]])]
# V = (block_diag(*VBlockList).astype(dtype) + np.eye(nc))
# multV = tf.linspace(1/nc, 1, nc)
# V = tf.cast(tf.sqrt(multV)[:,None] * V * tf.sqrt(multV), dtype)
# # V = tf.eye(nc, dtype=dtype)
# Mu = tf.matmul(Gamma, Tr, transpose_b=True)
# Beta = tf.transpose(tfd.MultivariateNormalFullCovariance(tf.transpose(Mu), V).sample())
# LFix = tf.matmul(X, Beta)
# nu, a1, b1, a2, b2 = 3, 1.2, 1, 2, 1
# rLHp = {"nu": nu, "a1": 5, "b1": 1, "a2": 5, "b2": 1, "sDim": 0}
# aDelta = tf.concat([a1 * tf.ones([1, 1], dtype), a2 * tf.ones([nf-1, 1], dtype)], 0)
# bDelta = tf.concat([b1 * tf.ones([1, 1], dtype), b2 * tf.ones([nf-1, 1], dtype)], 0)
# Delta = tfd.Gamma(aDelta, bDelta).sample()
# Tau = tfm.cumprod(Delta, 0)
# Lambda = tf.transpose(tf.squeeze(tfd.StudentT(nu, 0, tfm.rsqrt(Tau)).sample([ns]),-1))
# # tmp = tfd.StudentT(nu, 0, tfm.rsqrt(Tau)).log_prob(Lambda)
# Eta = tfd.Normal(tf.cast(0,dtype), 1).sample([ny,nf])
# Pi = np.arange(ny)[:,None]
# LRan = tf.gather(tf.matmul(Eta, Lambda), Pi[:,0])
# rLHyperparams = [rLHp]
# L = LFix + LRan
# sigma = np.ones([ns], dtype)
# pr = tfd.Normal(L,sigma).survival_function(0)
# plt.scatter(L, pr)
# Y = tfd.Bernoulli(probs=pr).sample()
# Omega = tf.matmul(Lambda,Lambda,transpose_a=True)

# n_sam = 5000
# transient = n_sam
# updateBeta = 1
# updateGamma = 0
# updateiV = 0
# updateEta = 1
# updateLambda = 1
# updateDelta = 0
# init = 0

# BetaInit =  init*Beta if updateBeta else Beta
# GammaInit = init*Gamma if updateGamma else Gamma
# iVInit = (init==1)*tfla.inv(V) + (init==0)*tf.eye(nc,dtype=dtype) if updateiV else tfla.inv(V)
# EtaInit = [init*Eta if updateEta else Eta]
# LambdaInit = [init*Lambda if updateLambda else Lambda]
# DeltaInit = [(init==1)*Delta + (init==0)*aDelta if updateDelta else Delta]


# priorHyperparams = {"f0": nc+1, "V0": np.eye(nc,dtype=dtype), 
#                     "mGamma": np.zeros([nc*nt],dtype), "iUGamma": np.eye(nc*nt,dtype=dtype)}
# params = {"Beta": BetaInit, "Gamma": GammaInit, "iV": iVInit, "sigma": sigma, "Xeff": tf.constant(X,dtype), 
#           "Eta": EtaInit, "Lambda": LambdaInit, "Delta": DeltaInit}
# data = {"Y": Y, "T": Tr, "Pi": Pi, "distr": None}


# hmc_res = updateHMC(params, data, priorHyperparams, rLHyperparams, 10, transient, 0, tf.constant(0.01,dtype), init=True,
#                     updateBeta=updateBeta, updateGamma=updateGamma, updateiV=updateiV,
#                     updateEta=updateEta, updateLambda=updateLambda, updateDelta=updateDelta, dtype=dtype)
# ss, las, es = hmc_res[-3:]


# BetaPost = np.zeros([n_sam,nc,ns])
# GammaPost = np.zeros([n_sam,nc,nt])
# VPost = np.zeros([n_sam,nc,nc])
# EtaPost = np.zeros([n_sam,ny,nf])
# LambdaPost = np.zeros([n_sam,nf,ns])
# DeltaPost = np.zeros([n_sam,nf,1])
# start_time = time.time()
# for i in range(n_sam+transient):
#   hmc_res = updateHMC(params, data, priorHyperparams, rLHyperparams, 10, transient, tf.constant(i), ss, las, es, 
#                       updateBeta=updateBeta, updateGamma=updateGamma, updateiV=updateiV,
#                       updateEta=updateEta, updateLambda=updateLambda, updateDelta=updateDelta, dtype=dtype)
#   params["Beta"], params["Gamma"], params["iV"], params["Eta"], params["Lambda"], params["Delta"] = hmc_res[:-3]
#   ss, las, es = hmc_res[-3:]
#   if i > transient:
#     sn = i - transient
#     BetaPost[sn] = params["Beta"].numpy()
#     GammaPost[sn] = params["Gamma"].numpy()
#     VPost[sn] = tfla.inv(params["iV"]).numpy()
#     EtaPost[sn] = params["Eta"][0].numpy()
#     LambdaPost[sn] = params["Lambda"][0].numpy()
#     DeltaPost[sn] = params["Delta"][0].numpy()
    
#   plt.subplot(2, 3, 1)
#   plt.scatter(Beta, params["Beta"]) #[transient:]
#   plt.axline((0, 0), slope=1, color='black')
#   plt.title("Beta")
#   plt.subplot(2, 3, 2)
#   plt.scatter(Gamma, params["Gamma"])
#   plt.axline((0, 0), slope=1, color='black')
#   plt.title("Gamma")
#   plt.subplot(2, 3, 3)
#   plt.scatter(V, tfla.inv(params["iV"]))
#   plt.axline((0, 0), slope=1, color='black')
#   plt.title("V")
#   plt.subplot(2, 3, 4)
#   plt.scatter(Eta, params["Eta"][0]) #[transient:]
#   plt.axline((0, 0), slope=1, color='black')
#   plt.title("Eta")
#   plt.subplot(2, 3, 5)
#   plt.scatter(Lambda, params["Lambda"][0])
#   plt.axline((0, 0), slope=1, color='black')
#   plt.title("Lambda")
#   OmegaSam = tf.matmul(params["Lambda"][0],params["Lambda"][0],transpose_a=True)
#   plt.subplot(2, 3, 6)
#   plt.scatter(Omega, OmegaSam)
#   plt.axline((0, 0), slope=1, color='black')
#   plt.title("Omega")
#   plt.suptitle("iter %d out of %d" % (i, n_sam+transient))
#   plt.show()

  
# stop_time = time.time()
# print(stop_time - start_time)

# plt.subplot(1, 3, 1)
# plt.scatter(Beta, tf.reduce_mean(BetaPost, 0)) #[transient:]
# plt.axline((0, 0), slope=1, color='black')
# plt.title("Beta")
# plt.subplot(1, 3, 2)
# plt.scatter(Gamma, tf.reduce_mean(GammaPost, 0))
# plt.axline((0, 0), slope=1, color='black')
# plt.title("Gamma")
# plt.subplot(1, 3, 3)
# plt.scatter(V, tf.reduce_mean(VPost, 0))
# plt.axline((0, 0), slope=1, color='black')
# plt.title("V")
# plt.show()

# plt.subplot(1, 3, 1)
# plt.scatter(Eta, tf.reduce_mean(EtaPost, 0)) #[transient:]
# plt.axline((0, 0), slope=1, color='black')
# plt.title("Eta")
# plt.subplot(1, 3, 2)
# plt.scatter(Lambda, tf.reduce_mean(LambdaPost, 0))
# plt.axline((0, 0), slope=1, color='black')
# plt.title("Lambda")
# plt.subplot(1, 3, 3)
# plt.scatter(Delta, tf.reduce_mean(DeltaPost, 0))
# plt.axline((0, 0), slope=1, color='black')
# plt.title("Delta")
# plt.show()

# OmegaPost = tf.matmul(LambdaPost,LambdaPost,transpose_a=True)
# plt.scatter(Omega, tf.reduce_mean(OmegaPost, 0))
# plt.axline((0, 0), slope=1, color='black')
# plt.title("Omega")
# plt.show()
