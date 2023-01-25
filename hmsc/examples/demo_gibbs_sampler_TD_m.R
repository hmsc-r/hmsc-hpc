library(Hmsc)
library(jsonify)
library(vioplot)

path = "/Users/anisjyu/Dropbox/hmsc-hpc/hmsc-hpc/hmsc"

m = TD$m

nChains = 4
nSamples = 250
transient = 125
thin = 10

#
# Generate sampled posteriors for TF initialization
#

init_obj = sampleMcmc(m, samples = 1, thin = 1,
                      transient = 0, 
                      nChains = nChains)

init_file_name = "TF-init-obj.json"
init_file_path = file.path(path, "examples/data", init_file_name)

write(to_json(init_obj), file = init_file_path)

#
# Generate sampled posteriors in R
#

# Start the clock!
ptm <- proc.time()

obj.R = sampleMcmc(m, samples = nSamples, 
                   transient = transient, 
                   thin = thin, 
                   nChains = nChains, verbose = 1) #fitted by R

# Stop the clock
proc.time() - ptm

#
# Set RStudio to TF env
#

my_conda_env_name = "tensorflow" # name of my conda TF env

# Start one-time python setup
# INFO. one-time steps to set python for RStudio/reticulate
# INFO. requires a session restart for Sys.setenv() to take effect

library(tidyverse)
py_bin <- reticulate::conda_list() %>% 
  filter(name == my_conda_env_name) %>% 
  pull(python)

Sys.setenv(RETICULATE_PYTHON = py_bin) 

# End one-time python setup

library(reticulate)
use_condaenv(my_conda_env_name, required=TRUE) # activate the TF env
repl_python()

#
# Generate sampled posteriors in TF
#

python_file_name = "run_gibbs_sampler.py"
python_file_path = file.path(path, "examples", python_file_name)

postList_file_name = "TF-postList-obj.json"
postList_file_path = file.path(path, "examples/data", postList_file_name)

python_cmd = paste("python", python_file_path, 
                   "--samples", nSamples,
                   "--transient", transient,
                   "--thin", thin,
                   "--input", init_file_name, 
                   "--output", postList_file_name)

system(paste("chmod a+x", python_file_path)) # set file permissions for shell execution

system(python_cmd, wait=TRUE) # run TF gibbs sampler

system("python --version", wait=TRUE) # run TF gibbs sampler

#
# Import TF-generated sampled posteriors as JSON in R
#

postList.TF <- from_json(postList_file_path)
names(postList.TF) = NULL
for (chain in seq_len(nChains)) {
  names(postList.TF[[chain]]) = NULL
}

obj.TF = init_obj
obj.TF[["postList"]] = postList.TF

obj.TF$samples = nSamples

#
# Rescaling Beta/Gamma; copied from combineParameters.R; need to revisit this section
#

ncNRRR = obj.TF[["ncNRRR"]]
ncRRR = obj.TF[["ncRRR"]]
ncsel = obj.TF[["ncsel"]]

XScalePar = obj.TF[["XScalePar"]]
XInterceptInd = obj.TF[["XInterceptInd"]]
XRRRScalePar = obj.TF[["XRRRScalePar"]]

for (chain in seq_len(nChains)) {
  for (sample in seq_len(nSamples)) {
    Beta = obj.TF[["postList"]][[chain]][[sample]][["Beta"]]
    Gamma = obj.TF[["postList"]][[chain]][[sample]][["Gamma"]]
    V = obj.TF[["postList"]][[chain]][[sample]][["V"]]
    
    for(k in 1:ncNRRR){
      m = XScalePar[1,k]
      s = XScalePar[2,k]
      if(m!=0 || s!=1){
        Beta[k,] = Beta[k,]/s
        Gamma[k,] = Gamma[k,]/s
        if(!is.null(XInterceptInd)){
          Beta[XInterceptInd,] = Beta[XInterceptInd,] - m*Beta[k,]
          Gamma[XInterceptInd,] = Gamma[XInterceptInd,] - m*Gamma[k,]
        }
        V[k,] = V[k,]*s
        V[,k] = V[,k]*s
      }
    }
    
    for(k in seq_len(ncRRR)){
      m = XRRRScalePar[1,k]
      s = XRRRScalePar[2,k]
      if(m!=0 || s!=1){
        Beta[ncNRRR+k,] = Beta[ncNRRR+k,]/s
        Gamma[ncNRRR+k,] = Gamma[ncNRRR+k,]/s
        if(!is.null(XInterceptInd)){
          Beta[XInterceptInd,] = Beta[XInterceptInd,] - m*Beta[ncNRRR+k,]
          Gamma[XInterceptInd,] = Gamma[XInterceptInd,] - m*Gamma[ncNRRR+k,]
        }
        V[ncNRRR+k,] = V[ncNRRR+k,]*s
        V[,ncNRRR+k] = V[,ncNRRR+k]*s
      }
    }
    
    for (i in seq_len(ncsel)){
      XSel = XSelect[[i]]
      for (spg in 1:length(XSel$q)){
        if(!BetaSel[[i]][spg]){
          fsp = which(XSel$spGroup==spg)
          Beta[XSel$covGroup,fsp]=0
        }
      }
    }
    
    obj.TF[["postList"]][[chain]][[sample]][["Beta"]] = Beta
    obj.TF[["postList"]][[chain]][[sample]][["Gamma"]] = Gamma
    obj.TF[["postList"]][[chain]][[sample]][["V"]] = V
  }
}

#
# Plot sampled posteriors summaries from R only and TF
#

obj.R.TF = obj.R
obj.R.TF$postList = c(obj.R$postList,obj.TF$postList)

obj.list = list(R=obj.R,TF=obj.TF,R.TF=obj.R.TF)

#beta and gamma
for(variable in 1:2){
  for(i in 1:3){
    mpost = convertToCodaObject(obj.list[[i]], Rho=FALSE)
    mpost.var = if(variable==1) {mpost$Beta} else {mpost$Gamma} 
    psrf = gelman.diag(mpost.var,multivariate=FALSE)$psrf
    if(i == 1) {ma = psrf[,1]} else {ma = cbind(ma,psrf[,1])}
  }
  par(mfrow=c(2,1))
  vioplot(ma,names=names(obj.list),ylim=c(0,max(ma)),main=c("beta","gamma")[variable])
  vioplot(ma,names=names(obj.list),ylim=c(0.9,1.1),main=c("beta","gamma")[variable])
}

#omega
maxOmega = 100 #number of species pairs to be subsampled
nr = obj.list[[1]]$nr
if(nr>0){
  for(k in 1:nr){
    for(i in 1:3){
      mpost = convertToCodaObject(obj.list[[i]], Rho=FALSE)
      tmp = mpost$Omega[[k]]
      z = dim(tmp[[1]])[2]
      if(z > maxOmega){
        if(i==1) sel = sample(1:z, size = maxOmega)
        for(j in 1:length(tmp)){
          tmp[[j]] = tmp[[j]][,sel]
        }
      }
      psrf = gelman.diag(tmp, multivariate = FALSE)$psrf
      if(i == 1) {ma = psrf[,1]} else {ma = cbind(ma,psrf[,1])}
    }
    par(mfrow=c(2,1))
    vioplot(ma,names=names(obj.list),ylim=c(0,max(ma)),main=paste("omega",names(obj.list[[1]]$ranLevels)[k]))
    vioplot(ma,names=names(obj.list),ylim=c(0.9,1.1),main=paste("omega",names(obj.list[[1]]$ranLevels)[k]))
  }
}
