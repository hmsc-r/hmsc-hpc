library(coda)
library(Hmsc)
library(abind)
library(jsonify)

source("/Users/anisjyu/Documents/HMSC/R/getPostEstimate.R")

path = "/Users/anisjyu/Dropbox/hmsc-hpc/hmsc-hpc/hmsc"

nChains = 2

m = Hmsc(TD$m$Y, XFormula=TD$m$XFormula, XData=TD$m$XData, distr=TD$m$distr,
         TrFormula=TD$m$TrFormula, TrData=TD$m$TrData, phyloTree=TD$m$phyloTree,
         studyDesign=TD$m$studyDesign, ranLevels=TD$m$ranLevels)

#
# Generate sampled posteriors for TF initialization
#

init_obj = sampleMcmc(m, samples = 1, thin = 1,
                      transient = 1, 
                      nChains = nChains)

init_file_name = "TF-init-obj.json"
init_file_path = file.path(path, "examples/data", init_file_name)
write(to_json(init_obj), file = init_file_path)

#
# Generate sampled posteriors in R
#

obj_R = sampleMcmc(m, samples = 50, thin = 1,
                   transient = 50,
                   nChains = nChains)

#
# Set RStudio to TF env
#

my_conda_env_name = "tensorflow"

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
use_condaenv(my_conda_env_name) # activate env

#
# Generate sampled posteriors in TF
#

python_file_name = "run_gibbs_sampler.py"
python_file_path = file.path(path, "examples", python_file_name)

postList_file_name = "TF-postList-obj.json"
postList_file_path = file.path(path, "examples/data", postList_file_name)

python_cmd = paste("python", python_file_path, 
                   "--samples", 50, "--thin", 1, 
                   "--transient", 50, 
                   "--input", init_file_name, 
                   "--output", postList_file_name)

system(paste("chmod a+x", python_file_path)) # set permissions

system(python_cmd, wait=TRUE)

#
# Import TF-generated sampled posteriors as JSON in R
#

tmp_json_data <- from_json(postList_file_path)
names(tmp_json_data) = NULL
for (chain in seq_len(nChains)) {
  names(tmp_json_data[[chain]]) = NULL
}

obj_TF = init_obj
obj_TF[["postList"]] = tmp_json_data

#
# Plot sampled posteriors summaries from R only and TF
#

postBeta_R  = getPostEstimate(obj_R,  parName = "Beta")
postBeta_TF = getPostEstimate(obj_TF, parName = "Beta")

par(mar=c(5,11,2.5,0))
plotBeta(m, 
         post = postBeta_R,
         param = "Mean",
         plotTree = F,  
         spNamesNumbers = c(T,F))

par(mar=c(5,11,2.5,0))
plotBeta(m, 
         post = postBeta_TF,
         param = "Mean",
         plotTree = F,  
         spNamesNumbers = c(T,F))

postGamma_R  = getPostEstimate(obj_R,  parName = "Gamma")
postGamma_TF = getPostEstimate(obj_TF, parName = "Gamma")

par(mar=c(5,11,2.5,0))
plotGamma(obj_R, post = postGamma_R, supportLevel = 0.2)

par(mar=c(5,11,2.5,0))
plotGamma(obj_TF, post = postGamma_TF, supportLevel = 0.2)

# Construct an ordination biplot using two chosen latent factors from a previously fitted HMSC model
postEta_R  = getPostEstimate(obj_R,  "Eta")
postEta_TF = getPostEstimate(obj_TF, "Eta")

postLambda_R  = getPostEstimate(obj_R,  "Lambda")
postLambda_TF = getPostEstimate(obj_TF, "Lambda")

biPlot(obj_R,  etaPost=postEta_R,  lambdaPost=postLambda_R,  factors=c(1,2))
biPlot(obj_TF, etaPost=postEta_TF, lambdaPost=postLambda_TF, factors=c(1,2))
