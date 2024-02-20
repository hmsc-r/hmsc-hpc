library(Hmsc)
library(sp)
library(jsonify)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
fileDir = getwd()
dir.create(file.path(fileDir, "init"))
dir.create(file.path(fileDir, "fmR"))
dir.create(file.path(fileDir, "fmTF"))

RS = 1
nSamples = 100
thin = 1
nChains = 1
nf = 10
transient = nSamples * thin
verbose = 1

modelTypeVec = c(0:4) # 0-ns, 1-full, 2-pgp, 3-nngp, 4-phylo
nParallel = 1
flagInit = 0
flagFitR = 1

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
mtSuffixVec = c("ns","fu","pg","nn","ph")
siteNames = sprintf("%.6d", 1:nrow(YA))
rownames(YA) = siteNames
rownames(XA) = siteNames
rownames(xyA) = siteNames

for(modelType in modelTypeVec){ 
	if(modelType==2){
		hull = read.csv("data/hull.csv", header=FALSE)
		kn = as.data.frame(spsample(Polygon(hull[-1,]), 60, type = "hexagonal", offset=c(0,0)))
		plot(hull, type="b")
		points(kn, col="red")
	}
	
	nsIndVec = c(3,5,7)
	if(modelType != 1){
		nyIndVec = c(1:9)
		if(modelType==2) nyIndVec = c(1:9)
	} else{
		nyIndVec = c(1:5)
	}
	for(nyInd in nyIndVec){
		for(nsInd in nsIndVec){
			print(sprintf("%d --- %d", nsInd, nyInd))
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
			modelTypeString = sprintf("%d%s", modelType, mtSuffixVec[modelType+1])
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
				C = matrix(NA,spNumVec[nsInd],spNumVec[nsInd])
				for(i in 1:spNumVec[nsInd]){
					for(j in 1:spNumVec[nsInd]){
						C[i,j] = mean((taxa[i,] == taxa[j,]))
					}
				}
				colnames(C) = rownames(C) = rownames(taxa)
			}
			rLSite = setPriors(rLSite, nfMin=nf, nfMax=nf)
			m = Hmsc(Y=Y, XData=X, Tr=Tr, C=C, distr="probit", studyDesign=studyDesign, ranLevels=list(site=rLSite))
			init_file_name = sprintf("init_%s_ns%.3d_ny%.5d_chain%.2d.rds", modelTypeString, m$ns, m$ny, nChains)
			fitR_file_name = sprintf("R_%s_ns%.3d_ny%.5d_chain%.2d_sam%.4d_thin%.4d.RData", modelTypeString, m$ns, m$ny, nChains, nSamples, thin)
			fitTF_file_name = sprintf("TF_%s_ns%.3d_ny%.5d_chain%.2d_sam%.4d_thin%.4d.rds", modelTypeString, m$ns, m$ny, nChains, nSamples, thin)
			
			if(flagInit){
				cat("Initializing for TF\n")
				path = dirname(rstudioapi::getSourceEditorContext()$path)
				set.seed(RS+42)
				init_obj = sampleMcmc(m, samples=nSamples, thin=thin,
															transient=transient, nChains=nChains, verbose=verbose,
															engine="HPC", updater=list(Gamma2=FALSE, GammaEta=FALSE))
				
				saveRDS(to_json(init_obj), file = file.path(fileDir, "init", init_file_name))
				cat("Export to TF saved\n")
			}
			
			if(flagFitR){
				set.seed(RS+42)
				cat("Initializing for R\n")
				init_obj = sampleMcmc(m, samples=nSamples, thin=thin,
															transient=transient, nChains=nChains, verbose=verbose,
															engine="pass", updater=list(Gamma2=FALSE, GammaEta=FALSE))
				cat("Initialization for R finished\n")
				set.seed(RS+42)
				cat("Fitting in R\n")
				startTime = proc.time()
				obj.R = sampleMcmc(m, samples = nSamples, thin = thin,
													 transient = transient,
													 nChains = nChains, nParallel=min(nChains,nParallel),
													 verbose = verbose, updater=list(Gamma2=FALSE, GammaEta=FALSE),
													 dataParList=init_obj$dataParList) #fitted by R
				elapsedTime = proc.time() - startTime
				print(elapsedTime)
				save(obj.R, elapsedTime, file=file.path(fileDir, "fmR", fitR_file_name))
			}
		}
	}
}
