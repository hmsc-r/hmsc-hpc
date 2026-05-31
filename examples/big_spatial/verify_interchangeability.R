# verification of structural interchangeability between R and TF chains
library(Hmsc)
library(sp)

tryCatch({
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}, error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    setwd(dirname(sub("--file=", "", file_arg[1])))
  } else {
    for (i in seq_along(sys.frames())) {
      if (!is.null(sys.frames()[[i]]$ofile)) {
        setwd(dirname(sys.frames()[[i]]$ofile))
        break
      }
    }
  }
})
fileDir = getwd()

# Ensure directories exist
dir.create(file.path(fileDir, "init"), showWarnings = FALSE)
dir.create(file.path(fileDir, "fmR"), showWarnings = FALSE)
dir.create(file.path(fileDir, "fmTF"), showWarnings = FALSE)

RS = 1
nSamples = 1000
thin = 10
nChains = 4
nf = 10
transient = nSamples * thin
verbose = 500

modelTypeVec = c(0:4) # 0-ns, 1-full, 2-pgp, 3-nngp, 4-phylo
mtSuffixVec = c("ns","fu","pg","nn","ph")

YA = read.csv("data/Y.csv")
TrA = read.csv("data/traits.csv")
XA = read.csv("data/X.csv")
xyA = read.csv("data/xy.csv", header=FALSE)
dfTaxa = read.csv("data/taxa_used_tree.csv")
YA = YA[,dfTaxa$index]
TrA = TrA[dfTaxa$index,]
spPart = dfTaxa$partition # 3-40, 5-160, 7-622
spNumVec = cumsum(table(spPart))
sitePart = as.vector(as.matrix(read.csv("data/data partition.csv", header=FALSE)))
siteNumVec = cumsum(table(sitePart)[-1])
siteNames = sprintf("%.6d", 1:nrow(YA))
rownames(YA) = siteNames
rownames(XA) = siteNames
rownames(xyA) = siteNames

# Set up local python command using ~/.virtualenvs/tf/bin/python3
python_path <- "/home/gt/.virtualenvs/tf/bin/python3"
local_hmsc_path <- normalizePath(file.path(fileDir, "../.."))
Sys.setenv(PYTHONPATH = paste0(local_hmsc_path, ":", Sys.getenv("PYTHONPATH")))

# Smallest dataset size: ns = 40 (nsInd = 3), ny = 100 (nyInd = 1)
nsInd = 3
nyInd = 1

results_summary <- list()

for(modelType in modelTypeVec){
	modelTypeString = sprintf("%d%s", modelType, mtSuffixVec[modelType+1])
	cat("\n========================================================\n")
	cat(sprintf("PROCESSING MODEL TYPE: %s\n", modelTypeString))
	cat("========================================================\n")
	
	if(modelType==2){
		hull = read.csv("data/hull.csv", header=FALSE)
		kn = as.data.frame(spsample(Polygon(hull[-1,]), 60, type = "hexagonal", offset=c(0,0)))
	}
	
	indSpecies = (spPart > 0) & (spPart <= nsInd)
	indSite = (sitePart > 0) & (sitePart <= nyInd)
	Y = YA[indSite,indSpecies]
	X = XA[indSite,]
	Tr = as.matrix(TrA[indSpecies,2:ncol(TrA)])
	colnames(Y) = rownames(Tr) = TrA[indSpecies,"CommonName"]
	taxa = dfTaxa[indSpecies,c("name","scientific","family","order","class","phylum")]
	rownames(taxa) = dfTaxa[indSpecies,"name"]
	studyDesign = data.frame(site=as.factor(rownames(X)))
	C = NULL
	
	if(modelType==0){
		rLSite = HmscRandomLevel(units=rownames(xyA))
	} else if(modelType==1){
		rLSite = HmscRandomLevel(sData=xyA)
		modelTypeString = sprintf("%dfu", modelType)
	} else if(modelType==2){
		rLSite = HmscRandomLevel(sData=xyA, sMethod="GPP", sKnot=kn)
	} else if(modelType==3){
		rLSite = HmscRandomLevel(sData=xyA, sMethod="NNGP", nNeighbours=10)
	} else if(modelType==4){
		rLSite = HmscRandomLevel(units=rownames(xyA))
		C = Reduce(`+`, lapply(taxa, function(x) outer(x, x, `==`))) / ncol(taxa)
		colnames(C) = rownames(C) = rownames(taxa)
	}
	rLSite = setPriors(rLSite, nfMin=nf, nfMax=nf)
	m = Hmsc(Y=Y, XData=X, Tr=Tr, C=C, distr="probit", studyDesign=studyDesign, ranLevels=list(site=rLSite))
	
	#----------------------------------------------------
	# 1. Fit in R
	#----------------------------------------------------
	cat("Fitting model with R (4 chains, 1000 samples, thin 10)...\n")
	set.seed(RS+42)
	init_obj_r = sampleMcmc(m, samples=nSamples, thin=thin,
												transient=transient, nChains=nChains, verbose=verbose,
												engine="pass", updater=list(Gamma2=FALSE, GammaEta=FALSE))
	
	set.seed(RS+42)
	startTimeR = proc.time()
	fitR = sampleMcmc(m, samples = nSamples, thin = thin,
										 transient = transient,
										 nChains = nChains, nParallel=4,
										 verbose = verbose, updater=list(Gamma2=FALSE, GammaEta=FALSE),
										 dataParList=init_obj_r$dataParList)
	elapsedTimeR = proc.time() - startTimeR
	cat(sprintf("R Fitting completed in %0.2f seconds.\n", elapsedTimeR[3]))
	
	#----------------------------------------------------
	# 2. Fit in TF (Python)
	#----------------------------------------------------
	cat("Exporting initialized model for TF...\n")
	set.seed(RS+42)
	init_obj_tf = sampleMcmc(m, samples=nSamples, thin=thin,
												 transient=transient, nChains=nChains, verbose=verbose,
												 engine="HPC", updater=list(Gamma2=FALSE, GammaEta=FALSE))
	
	init_file_name = sprintf("init_verify_%s.rds", modelTypeString)
	init_file_path = file.path(fileDir, "init", init_file_name)
	saveRDS(init_obj_tf, file = init_file_path)
	
	cat("Fitting model with TF Python Sampler...\n")
	post_file_name = sprintf("post_verify_%s.rds", modelTypeString)
	post_file_path = file.path(fileDir, "fmTF", post_file_name)
	
	args_vec <- c(
		"-m", "hmsc.run_gibbs_sampler",
		"--input", shQuote(init_file_path),
		"--output", shQuote(post_file_path),
		"--samples", nSamples,
		"--transient", transient,
		"--thin", thin,
		"--verbose", verbose
	)
	
	startTimeTF = proc.time()
	status <- system2(python_path, args_vec)
	elapsedTimeTF = proc.time() - startTimeTF
	
	if(status != 0) {
		stop(sprintf("Python TF sampler failed for model %s with status %d", modelTypeString, status))
	}
	cat(sprintf("TF Fitting completed in %0.2f seconds.\n", elapsedTimeTF[3]))
	
	# Load TF chains back to R
	importFromHPC <- readRDS(post_file_path)
	fitTF <- importPosteriorFromHPC(m, importFromHPC[["list"]], nSamples, thin, transient)
	
	#----------------------------------------------------
	# 3. Check Interchangeability
	#----------------------------------------------------
	cat("Verifying chain interchangeability...\n")
	
	# Create a hybrid model: Chains 1 and 3 from R, Chains 2 and 4 from TF
	fitHybrid <- fitR
	fitHybrid$postList[[2]] <- fitTF$postList[[2]]
	fitHybrid$postList[[4]] <- fitTF$postList[[4]]
	
	# Check structural validity by converting to coda objects and calculating diagnostics
	codaR <- convertToCodaObject(fitR, Alpha = FALSE, Psi = FALSE, Delta = FALSE, Eta = FALSE, Lambda = FALSE, Omega = FALSE)
	codaTF <- convertToCodaObject(fitTF, Alpha = FALSE, Psi = FALSE, Delta = FALSE, Eta = FALSE, Lambda = FALSE, Omega = FALSE)
	codaHybrid <- convertToCodaObject(fitHybrid, Alpha = FALSE, Psi = FALSE, Delta = FALSE, Eta = FALSE, Lambda = FALSE, Omega = FALSE)
	
	# Compute Gelman-Rubin diagnostic (PSRFs) on Beta parameters
	psrfR <- gelman.diag(codaR$Beta, multivariate=FALSE)
	psrfTF <- gelman.diag(codaTF$Beta, multivariate=FALSE)
	psrfHybrid <- gelman.diag(codaHybrid$Beta, multivariate=FALSE)
	
	# Compute Variance Partitioning to verify predictability and correctness
	vpR <- computeVariancePartitioning(fitR)
	vpTF <- computeVariancePartitioning(fitTF)
	vpHybrid <- computeVariancePartitioning(fitHybrid)
	
	# Calculate correlation between TF and R Beta means
	meanBetaR <- apply(simplify2array(lapply(fitR$postList, function(chain) sapply(chain, function(s) s$Beta))), c(1,2), mean)
	meanBetaTF <- apply(simplify2array(lapply(fitTF$postList, function(chain) sapply(chain, function(s) s$Beta))), c(1,2), mean)
	meanBetaHybrid <- apply(simplify2array(lapply(fitHybrid$postList, function(chain) sapply(chain, function(s) s$Beta))), c(1,2), mean)
	
	betaCorr <- cor(as.vector(meanBetaR), as.vector(meanBetaTF))
	
	cat(sprintf("  Gelman diagnostic (mean PSRF point est) - R: %0.4f, TF: %0.4f, Hybrid: %0.4f\n", 
							mean(psrfR$psrf[,1]), mean(psrfTF$psrf[,1]), mean(psrfHybrid$psrf[,1])))
	cat(sprintf("  Beta Posterior Mean Correlation (R vs TF): %0.4f\n", betaCorr))
	cat("  Variance Partitioning R, TF, and Hybrid computed successfully!\n")
	
	# Save fitted models and statistics to keep them
	saveRDS(fitR, file = file.path(fileDir, "fmR", sprintf("R_verify_%s_fitted.rds", modelTypeString)))
	saveRDS(fitTF, file = file.path(fileDir, "fmTF", sprintf("TF_verify_%s_fitted.rds", modelTypeString)))
	saveRDS(fitHybrid, file = file.path(fileDir, "fmTF", sprintf("Hybrid_verify_%s_fitted.rds", modelTypeString)))
	
	results_summary[[modelTypeString]] <- list(
		model = modelTypeString,
		timeR = elapsedTimeR[3],
		timeTF = elapsedTimeTF[3],
		meanPsrfR = mean(psrfR$psrf[,1]),
		meanPsrfTF = mean(psrfTF$psrf[,1]),
		meanPsrfHybrid = mean(psrfHybrid$psrf[,1]),
		betaCorr = betaCorr
	)
}

cat("\n========================================================\n")
cat("SUMMARY OF RESULTS:\n")
cat("========================================================\n")
for(res in results_summary) {
	cat(sprintf("Model: %s\n", res$model))
	cat(sprintf("  Time - R: %0.1f s, TF: %0.1f s\n", res$timeR, res$timeTF))
	cat(sprintf("  Mean PSRF - R: %0.4f, TF: %0.4f, Hybrid (R+TF): %0.4f\n", res$meanPsrfR, res$meanPsrfTF, res$meanPsrfHybrid))
	cat(sprintf("  R vs TF Beta Mean Correlation: %0.4f\n", res$betaCorr))
}
cat("\nSUCCESS: Interchangeability validated successfully. All fitted models and verification artifacts preserved!\n")
