library(Hmsc)
library(jsonify)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
fileDir = getwd()

dirRFit = "fmR"
prefixR = "R"
typeNameR = "RData"
dirTFFit = "fmTF"
prefixTF = "TF"
typeNameTF = "rds"

nChains = 1
msFigFlag = TRUE

dfTaxa = read.csv("data/taxa_used_tree.csv")
spPart = dfTaxa$partition # 3-40, 5-160, 7-623
spNumVec = cumsum(table(spPart))
sitePart = as.vector(as.matrix(read.csv("data/data partition.csv", header=FALSE)))
siteNumVec = cumsum(table(sitePart)[-1])
mtVec = c(0:4)
mtSuffixVec = c("ns","fu","pg","nn","ph")
modelNamesVec = c("non-spatial", "full GP", "predictive GP", "nearest neighbour GP", "phylogeny")
nsVec = spNumVec[c(3,5,7)]
nyVec = siteNumVec
mtN = length(mtVec)
nsN = length(nsVec)
nyN = length(nyVec)

samR = array(100, c(mtN,nsN,nyN))
thinR = array(1, c(mtN,nsN,nyN))

samTF = array(100, c(mtN,nsN,nyN))
thinTF = array(10, c(mtN,nsN,nyN))

execTimeR = array(NA, c(mtN,nsN,nyN))
execTimeTF = array(NA, c(mtN,nsN,nyN))
for(mtInd in 1:mtN){
	modelTypeString = paste0(mtVec[mtInd], mtSuffixVec[mtInd])
	for(nsInd in 1:nsN){
		ns = nsVec[nsInd]
		for(nyInd in 1:nyN){
			ny = nyVec[nyInd]
			print(paste(modelTypeString, ns, ny, sep="_"))
			fitR_file_name = sprintf("%s_%s_ns%.3d_ny%.5d_chain%.2d_sam%.4d_thin%.4d.%s", prefixR, modelTypeString,
															 ns, ny, nChains, samR[mtInd,nsInd,nyInd], thinR[mtInd,nsInd,nyInd], typeNameR)
			fitTF_file_name = sprintf("%s_%s_ns%.3d_ny%.5d_chain%.2d_sam%.4d_thin%.4d.%s", prefixTF, modelTypeString,
																ns, ny, nChains, samTF[mtInd,nsInd,nyInd], thinTF[mtInd,nsInd,nyInd], typeNameTF)
			tryCatch({
				load(file.path(fileDir, dirRFit, fitR_file_name))
				execTimeR[mtInd,nsInd,nyInd] = elapsedTime[3]
			}, error = function(e){
				print(paste(paste(modelTypeString, ns, ny, sep="_"), "R", "missing"))
			})
			tryCatch({
				fitTF <- from_json(readRDS(file = file.path(fileDir, dirTFFit, fitTF_file_name))[[1]])
				execTimeTF[mtInd,nsInd,nyInd] = fitTF$time
			}, error = function(e){
				print(paste(paste(modelTypeString, ns, ny, sep="_"), "TF", "missing"))
			})
		}
	}
}

mtPch = c(1,15,17,8,2)
colVec = c("red","blue","darkgreen")
if(msFigFlag){
	png("perfomance_combined.png", width=1200, height=1200)
	layout(matrix(c(1,2,3,4,6,6,5,6,6), 3, 3, byrow=TRUE),
				 widths=c(1,1,1), heights=c(1,1,1))
}

normTimeR = execTimeR * (1 / (samR*thinR))
normTimeTF = execTimeTF * (1 / (samTF*thinTF))
normTimeLogRatio = log10(normTimeTF / normTimeR)
ratioLim = c(-1,1)*max(abs(range(normTimeLogRatio,na.rm=TRUE)))
for(mtInd in 1:mtN){
	plot(NA, xlim=range(nyVec), ylim=range(c(normTimeR,normTimeTF),na.rm=TRUE), log="xy",
			 xlab=NA, ylab=NA, main=NA, xaxt="n", cex.axis=2)
	axis(1, at=nyVec, labels=c(100,200,400,800,"1.6K","3.2K","6.4K","12.8K","26K"), cex.axis=1.5)
	
	for(nsInd in 1:nsN){
		lines(nyVec, normTimeR[mtInd,nsInd,], lty=1, col=colVec[nsInd])
		points(nyVec, normTimeR[mtInd,nsInd,], pch=mtPch[mtInd], col=colVec[nsInd], cex=2)
		lines(nyVec, normTimeTF[mtInd,nsInd,], lty=5, col=colVec[nsInd])
		points(nyVec, normTimeTF[mtInd,nsInd,], pch=mtPch[mtInd], col=colVec[nsInd], cex=2)
	}
}

normTimeR = execTimeR * (100000 / (samR*thinR))
normTimeTF = execTimeTF * (100000 / (samTF*thinTF))
xRange = range(c(600,normTimeR,normTimeTF),na.rm=TRUE)
plot(NA, xlim=xRange, ylim=xRange, log="xy", xaxt="n", yaxt="n",
		 xlab=NA, ylab=NA, main=NA, cex=2)
xTick = c(60,600,60^2,6*50^2,24*60^2,7*24*60^2,30*24*60^2,6*30*24*60^2,60*30*24*60^2)
xLabel = c("min","10min","1h","6h","day","week","month","6 months", "5 years")
axis(1, at=xTick, labels=xLabel, cex.axis=2)
axis(2, at=xTick, labels=xLabel, cex.axis=2)
abline(0,1,lty=2)
lines(xTick,xTick/10, lty=3)
lines(xTick,xTick/100, lty=3)
lines(xTick,xTick/1000, lty=3)
lines(xTick,xTick*10, lty=3)
lines(xTick,xTick*100, lty=3)
lines(xTick,xTick*1000, lty=3)
points(normTimeR, normTimeTF, pch=rep(mtPch[1:mtN],nsN*nyN), col=rep(colVec,each=mtN), cex=rep(c(3,3,3,2,3)[1:mtN],nsN*nyN))

if(msFigFlag)	dev.off()
