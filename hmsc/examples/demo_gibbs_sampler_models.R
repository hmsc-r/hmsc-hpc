# library(devtools)
# install_github("hmsc-r/HMSC")
library(Hmsc)
library(jsonify)
library(vioplot)

path = dirname(dirname(rstudioapi::getSourceEditorContext()$path))

load(file = file.path(path, "examples/data", "unfitted_models_2.RData"))
models

Experiments <- function() {
  list(
    M1=list(name="model1",id=1),
    M2=list(name="model2",id=2),
    M3=list(name="model3",id=3),
    M4=list(name="model4",id=4),
    M5=list(name="model5",id=5),
    M6=list(name="model6",id=6),
    M7=list(name="model7",id=7),
    M8=list(name="model8",id=8),
    M9=list(name="model9",id=9)
    )
}
experiments <- Experiments()

selected_experiment = experiments$M9

m = models[[selected_experiment$id]]

if (selected_experiment$name == experiments$M1$name) {
  nChains = 8
  nSamples = 250
  thin = 10
} else if (selected_experiment$name == experiments$M2$name) {
  nChains = 8
  nSamples = 250
  thin = 1
  # rLSite = m$ranLevels$site
  # rLTime = m$ranLevels$time
  # rLSiteNew = HmscRandomLevel(units=rLSite$pi)
  # rLTimeNew = HmscRandomLevel(units=rLTime$pi)
  rLSiteNew = m$ranLevels$site
  rLTimeNew = m$ranLevels$time
  YNew = matrix(colMeans(m$Y, na.rm=TRUE), nrow(m$Y), ncol(m$Y), byrow=TRUE)
  YNew[!is.na(m$Y)] = m$Y[!is.na(m$Y)]
  mNew = Hmsc(Y=YNew, XFormula=m$XFormula, XData=m$XData, TrFormula=m$TrFormula, TrData=m$TrData,
              distr=m$distr, studyDesign=m$studyDesign, ranLevels=list(site=rLSiteNew,time=rLTimeNew))
  m = mNew
} else if (selected_experiment$name == experiments$M3$name) {
  nChains = 8
  nSamples = 250
  thin = 10
} else if (selected_experiment$name == experiments$M4$name) {
  nChains = 8
  nSamples = 250
  thin = 10
} else if (selected_experiment$name == experiments$M5$name) {
  nChains = 8
  nSamples = 250
  thin = 10
  #m$X = 0.5*(m$X[[1]] + m$X[[2]])
  #m$XScaled = 0.5*(m$XScaled[[1]] + m$XScaled[[2]])
  #m$XData = 0.5*(m$XData[[1]] + m$XData[[2]])
  transient = nSamples*thin
} else if (selected_experiment$name == experiments$M6$name) {
  nChains = 8
  nSamples = 250
  thin = 10
  #m$X = 0.5*(m$X[[1]] + m$X[[2]])
  #m$XScaled = 0.5*(m$XScaled[[1]] + m$XScaled[[2]])
  #m$XData = 0.5*(m$XData[[1]] + m$XData[[2]])
} else if (selected_experiment$name == experiments$M7$name) {
  print("Not Implemented: XScaled not found error for engine=pass")
} else if (selected_experiment$name == experiments$M8$name) {
  nChains = 8
  nSamples = 250
  thin = 10
  #m$X = 0.5*(m$X[[1]] + m$X[[2]])
  #m$XScaled = 0.5*(m$XScaled[[1]] + m$XScaled[[2]])
  #m$XData = 0.5*(m$XData[[1]] + m$XData[[2]])
} else if (selected_experiment$name == experiments$M9$name) {
  nChains = 8
  nSamples = 250
  thin = 10
  #m$X = 0.5*(m$X[[1]] + m$X[[2]])
  #m$XScaled = 0.5*(m$XScaled[[1]] + m$XScaled[[2]])
  #m$XData = 0.5*(m$XData[[1]] + m$XData[[2]])
}

transient = nSamples*thin
verbose = thin*10

#
# Generate sampled posteriors for TF initialization
#

init_obj = sampleMcmc(m, samples=nSamples, thin=thin,
                      transient=transient,
                      nChains=nChains, verbose=verbose, engine="pass")

init_file_name = paste("TF-init-obj-", selected_experiment$name, ".json", sep="")
init_file_path = file.path(path, "examples/data", init_file_name)
python_file_name = "run_gibbs_sampler.py"
python_file_path = file.path(path, "examples", python_file_name)
postList_file_name = paste("TF-postList-obj-", selected_experiment$name, ".json", sep="")
postList_file_path = file.path(path, "examples/data", postList_file_name)

nr = init_obj[["hM"]][["nr"]]
rLNames = init_obj[["hM"]][["ranLevelsUsed"]]

for (r in seq_len(nr)) {
  rLName = rLNames[[r]]
  init_obj[["hM"]][["rL"]][[rLName]][["s"]] = NULL
  init_obj[["hM"]][["ranLevels"]][[rLName]][["s"]] = NULL
  
  spatialMethod = init_obj[["hM"]][["rL"]][[r]][["spatialMethod"]]
  
  if (!is.null(spatialMethod)) {
    if (spatialMethod == "NNGP") {
      gN = length(init_obj[["dataParList"]][["rLPar"]][[r]][["iWg"]])
  
      for (i in seq_len(gN)) {
        iWg = as(init_obj[["dataParList"]][["rLPar"]][[r]][["iWg"]][[i]], "dgTMatrix")
        RiWg = as(init_obj[["dataParList"]][["rLPar"]][[r]][["RiWg"]][[i]], "dgTMatrix")

        init_obj[["dataParList"]][["rLPar"]][[r]][["iWgi"]][[i]] = iWg@i
        init_obj[["dataParList"]][["rLPar"]][[r]][["iWgj"]][[i]] = iWg@j
        init_obj[["dataParList"]][["rLPar"]][[r]][["iWgx"]][[i]] = iWg@x
    
        init_obj[["dataParList"]][["rLPar"]][[r]][["RiWgi"]][[i]] = RiWg@i
        init_obj[["dataParList"]][["rLPar"]][[r]][["RiWgj"]][[i]] = RiWg@j
        init_obj[["dataParList"]][["rLPar"]][[r]][["RiWgx"]][[i]] = RiWg@x
      }
      init_obj[["dataParList"]][["rLPar"]][[r]][["iWg"]] = NULL
      init_obj[["dataParList"]][["rLPar"]][[r]][["RiWg"]] = NULL
    }
    else if (spatialMethod == "GPP") {
      init_obj[["dataParList"]][["rLPar"]][[r]][["nK"]] = nrow(init_obj[["dataParList"]][["rLPar"]][[1]][["Fg"]])
    }
  }
  else {}
}

write(to_json(init_obj), file = init_file_path)
python_cmd = paste("python", sprintf("'%s'",python_file_path), 
                   "--samples", nSamples,
                   "--transient", transient,
                   "--thin", thin,
                   "--verbose", verbose,
                   "--input", init_file_name, 
                   "--output", postList_file_name,
                   "--path", sprintf("'%s'",path))
print(python_cmd)

#
# Generate sampled posteriors in R
#
ptm <- proc.time()
obj.R = sampleMcmc(m, samples = nSamples, thin = thin,
                   transient = transient, 
                   nChains = nChains, nParallel=nChains,
                   verbose = verbose, updater=list(Gamma2=FALSE, GammaEta=FALSE)) #fitted by R
print(proc.time() - ptm)

#
# Set RStudio to TF env
#
# my_conda_env_name = "tensorflow" # name of my conda TF env
# my_conda_env_name = "tf_241" # name of my conda TF env

# Start one-time python setup
# INFO. one-time steps to set python for RStudio/reticulate
# INFO. requires a session restart for Sys.setenv() to take effect

# library(tidyverse)
# py_bin <- reticulate::conda_list() %>%
#   filter(name == my_conda_env_name) %>%
#   pull(python)
#
# Sys.setenv(RETICULATE_PYTHON = py_bin)

# End one-time python setup

# library(reticulate)
# use_condaenv(my_conda_env_name, required=TRUE) # activate the TF env
# repl_python()

#
# Generate sampled posteriors in TF
#
system(paste("chmod a+x", sprintf("'%s'",python_file_path))) # set file permissions for shell execution
system("python --version", wait=TRUE)
system(python_cmd, wait=TRUE) # run TF gibbs sampler

#
# Import TF-generated sampled posteriors as JSON in R
#
postList.TF <- from_json(postList_file_path)
# postList_file_str <- paste(readLines(postList_file_path), collapse="\n")
# s = '{"0":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"1":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"2":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"3":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"4":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"5":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"6":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"7":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}}}'
# from_json(s)

names(postList.TF) = NULL
for (chain in seq_len(nChains)) {
  names(postList.TF[[chain]]) = NULL
}

obj.TF = init_obj$hM
obj.TF[["postList"]] = postList.TF

obj.TF$samples = nSamples
obj.TF$thin = thin
obj.TF$transient = transient

#
# Rescaling Beta/Gamma; copied from combineParameters.R; need to revisit this section
#
nt = obj.TF[["nt"]]
TrInterceptInd = obj.TF[["TrInterceptInd"]]
TrScalePar = obj.TF[["TrScalePar"]]
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
    rho = obj.TF$rhopw[obj.TF[["postList"]][[chain]][[sample]][["rhoInd"]], 1]
    
    for(p in 1:nt){
      m = TrScalePar[1,p]
      s = TrScalePar[2,p]
      if(m!=0 || s!=1){
        Gamma[,p] = Gamma[,p]/s
        if(!is.null(TrInterceptInd)){
          Gamma[,TrInterceptInd] = Gamma[,TrInterceptInd] - m*Gamma[,p]
        }
      }
    }
    
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
    obj.TF[["postList"]][[chain]][[sample]][["rho"]] = rho
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
    mpost = convertToCodaObject(obj.list[[i]])
    mpost.var = if(variable==1) {mpost$Beta} else {mpost$Gamma} 
    psrf = gelman.diag(mpost.var,multivariate=FALSE)$psrf
    if(i == 1) {ma = psrf[,1]} else {ma = cbind(ma,psrf[,1])}
  }
  par(mfrow=c(2,1))
  vioplot(ma,names=names(obj.list),ylim=c(0.9,max(ma)),main=c("beta","gamma")[variable])
  vioplot(ma,names=names(obj.list),ylim=c(0.9,1.1),main=c("beta","gamma")[variable])
}

#rho
if(!is.null(obj.R$C)){
  mat = rbind(unlist(getPostEstimate(obj.R, "rho")), unlist(getPostEstimate(obj.TF, "rho")))
  rownames(mat) = c("R", "TF")
  cat("rho posterior summary\n")
  print(mat)
}


#omega
maxOmega = 100 #number of species pairs to be subsampled
nr = obj.list[[1]]$nr
for(k in seq_len(nr)){
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
  vioplot(ma,names=names(obj.list),ylim=c(0.9,max(ma)),main=paste("omega",names(obj.list[[1]]$ranLevels)[k]))
  vioplot(ma,names=names(obj.list),ylim=c(0.9,1.1),main=paste("omega",names(obj.list[[1]]$ranLevels)[k]))
}

