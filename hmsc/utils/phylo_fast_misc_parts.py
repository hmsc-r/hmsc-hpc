import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfla, tfm, tfr, tfs = tf.linalg, tf.math, tf.random, tf.sparse
tfd, tfb = tfp.distributions, tfp.bijectors

def withTranspose_phyloFastBilinearDetBatched(treeList, X, Y, root, iV, rho, dtype=tf.float64):
  print("phyloFast phyloFastBilinearDetBatched")
  batchShape = list(rho.shape[:-1])
  if len(batchShape) > 1:
    raise ValueError("Batch shape in phyloFastBilinearDetBatched() can be no more than one")
  LiV = tfla.cholesky(iV)
  V = tfla.cholesky_solve(LiV, tf.eye(iV.shape[-1],dtype=dtype))
  logDetV = -2 * tf.reduce_sum(tfm.log(tfla.diag_part(LiV)))
  d105 = tfm.sqrt(rho)
  d205 = tfm.sqrt(1 - rho)
  V1 = d105[...,:,None] * V * d105[...,None,:]
  V2 = d205[...,:,None] * V * d205[...,None,:]
  depth = treeList[root]["depth"]
  indMatList, parentEdgeLenList = getIndMatListFromTree(treeList, depth=depth)
    
  XiSYList, OneiSXList, OneiSYList, OneiSOneList, logDetList = [[None]*(depth+1) for i in range(5)]
  d = 0
  parentEdgeLen = parentEdgeLenList[d]
  S = parentEdgeLen[:,None,None] * V1[...,None,:,:] + V2[...,None,:,:]
  transposeInd = np.arange(X.ndim-3).tolist() + (X.ndim-3+np.array([1,2,0])).tolist()
  X1 = tf.transpose(X, transposeInd)
  Y1 = tf.transpose(Y, transposeInd)
  iSX = tfla.solve(S, X1)
  iSY = tfla.solve(S, Y1)
  XiSYList[d] = tf.matmul(X1, iSY, transpose_a=True)
  OneiSXList[d], OneiSYList[d] = iSX, iSY
  LS = tfla.cholesky(S)
  OneiSOneList[d] = tfla.cholesky_solve(LS, tf.eye(S.shape[-1],batch_shape=S.shape[:-2],dtype=dtype))
  logDetList[d] = 2 * tf.reduce_sum(tfm.log(tfla.diag_part(LS)), -1)
  for d in range(depth):
    nextLevelNodeN = indMatList[d+1].shape[0] if d < depth-1 else 1
    scatterInd = indMatList[d][:,-1,None]
    #TODO consider rewriting so that the batch shape is not leading as transpositions are slow
    if len(batchShape) == 0:
      XiSYSum, OneiSXSum, OneiSYSum, OneiSOneSum = [tf.scatter_nd(scatterInd, vChild, [nextLevelNodeN]+vChild.shape[-2:]) \
                                                    for vChild in [XiSYList[d], OneiSXList[d], OneiSYList[d], OneiSOneList[d]]]
      logDetSum = tf.scatter_nd(scatterInd, logDetList[d], [nextLevelNodeN])
    elif len(batchShape) == 1:
      def scatter2(A):
        trInd = [1,0] + np.arange(2,len(XiSYList[d].shape)).tolist()
        res = tf.transpose(tf.scatter_nd(scatterInd, tf.transpose(A,trInd), [nextLevelNodeN]+batchShape+A.shape[-2:]), trInd)
        return(res)
      XiSYSum = scatter2(XiSYList[d])
      XiSYSum, OneiSXSum, OneiSYSum, OneiSOneSum = [scatter2(vChild) for vChild in [XiSYList[d], OneiSXList[d], OneiSYList[d], OneiSOneList[d]]]
      logDetSum = tf.transpose(tf.scatter_nd(indMatList[d][:,-1,None], tf.transpose(logDetList[d]), [nextLevelNodeN]+batchShape))
      
    parentEdgeLen = parentEdgeLenList[d+1]
    W = iV[...,None,:,:] + parentEdgeLen[:,None,None] * (d105[...,None,:,None] * OneiSOneSum * d105[...,None,None,:])
    LW = tfla.cholesky(W)
    logDetList[d+1] = logDetSum + logDetV + 2*tf.reduce_sum(tfm.log(tfla.diag_part(LW)), -1)
    iLW_Drho05_OneiSXSum = tfla.triangular_solve(LW, d105[...,None,:,None] * OneiSXSum)
    iLW_Drho05_OneiSYSum = tfla.triangular_solve(LW, d105[...,None,:,None] * OneiSYSum)
    iLW_Drho05_OneiSOneSum = tfla.triangular_solve(LW, d105[...,None,:,None] * OneiSOneSum)
    
    XiSYList[d+1] = XiSYSum - parentEdgeLen[:,None,None] * tf.matmul(iLW_Drho05_OneiSXSum, iLW_Drho05_OneiSYSum, transpose_a=True)
    OneiSXList[d+1] = OneiSXSum - parentEdgeLen[:,None,None] * tf.matmul(iLW_Drho05_OneiSOneSum, iLW_Drho05_OneiSXSum, transpose_a=True)
    OneiSYList[d+1] = OneiSYSum - parentEdgeLen[:,None,None] * tf.matmul(iLW_Drho05_OneiSOneSum, iLW_Drho05_OneiSYSum, transpose_a=True)
    OneiSOneList[d+1] = OneiSOneSum - parentEdgeLen[:,None,None] * tf.matmul(iLW_Drho05_OneiSOneSum, iLW_Drho05_OneiSOneSum, transpose_a=True)
    
  XiSY = tf.squeeze(XiSYList[depth], -3)
  logDet = tf.squeeze(logDetList[depth], -1)
  return XiSY, logDet


def recFunBilinearDet(node, treeList, X, Y, V, rho05, iV, V1, V2, logDetV, dtype=tf.float64):
  nChild = treeList[node]["n"]
  parentEdgeLen = treeList[node]["parentEdgeLen"]
  XiSYChild, OneiSXChild, OneiSYChild, OneiSOneChild, logDetChild = [[None]*nChild for i in range(5)]
  for i in range(nChild):
    childNode = treeList[node]["child"][i]
    if treeList[childNode]["n"] > 0:
      recRes = recFunBilinearDet(childNode, treeList, X, Y, V, rho05, iV, V1, V2, logDetV, dtype)
      XiSYChild[i], OneiSXChild[i], OneiSYChild[i], OneiSOneChild[i], logDetChild[i] = recRes
    else:
      S = treeList[childNode]["parentEdgeLen"] * V1 + V2
      X1 = tfla.matrix_transpose(X[...,:,childNode,:])
      Y1 = tfla.matrix_transpose(Y[...,:,childNode,:])
      iSX = tfla.solve(S, X1)
      iSY = tfla.solve(S, Y1)
      XiSYChild[i] = tf.matmul(X1, iSY, transpose_a=True)
      OneiSXChild[i], OneiSYChild[i] = iSX, iSY
      LS = tfla.cholesky(S)
      OneiSOneChild[i] = tfla.cholesky_solve(LS, tf.eye(S.shape[-1],batch_shape=S.shape[:-2],dtype=dtype))
      logDetChild[i] = 2 * tf.reduce_sum(tfm.log(tfla.diag_part(LS)), -1)
  
  XiSYSum, OneiSXSum, OneiSYSum, OneiSOneSum, logDetSum = [tf.add_n(vChild) for vChild in [XiSYChild, OneiSXChild, OneiSYChild, OneiSOneChild, logDetChild]]
  if parentEdgeLen == 0:
    return XiSYSum, OneiSXSum, OneiSYSum, OneiSOneSum, logDetSum
  
  W = iV + parentEdgeLen * (rho05[...,:,None] * OneiSOneSum * rho05[...,None,:])
  LW = tfla.cholesky(W)
  logDet = logDetSum + logDetV + 2*tf.reduce_sum(tfm.log(tfla.diag_part(LW)), -1)
  iLW_Drho05_OneiSXSum = tfla.triangular_solve(LW, rho05[...,:,None] * OneiSXSum)
  iLW_Drho05_OneiSYSum = tfla.triangular_solve(LW, rho05[...,:,None] * OneiSYSum)
  iLW_Drho05_OneiSOneSum = tfla.triangular_solve(LW, rho05[...,:,None] * OneiSOneSum)
  
  XiSY = XiSYSum - parentEdgeLen * tf.matmul(iLW_Drho05_OneiSXSum, iLW_Drho05_OneiSYSum, transpose_a=True)
  OneiSX = OneiSXSum - parentEdgeLen * tf.matmul(iLW_Drho05_OneiSOneSum, iLW_Drho05_OneiSXSum, transpose_a=True)
  OneiSY = OneiSYSum - parentEdgeLen * tf.matmul(iLW_Drho05_OneiSOneSum, iLW_Drho05_OneiSYSum, transpose_a=True)
  OneiSOne = OneiSOneSum - parentEdgeLen * tf.matmul(iLW_Drho05_OneiSOneSum, iLW_Drho05_OneiSOneSum, transpose_a=True)
  return XiSY, OneiSX, OneiSY, OneiSOne, logDet


#TODO need to check/fix the dimensions after modifications made to the batched version
def phyloFastBilinearDet(treeList, X, Y, root, iV, rho, dtype=tf.float64):
  print("phyloFast phyloFastBilinearDet")
  rho05 = tfm.sqrt(rho)
  LiV = tfla.cholesky(iV)
  V = tfla.cholesky_solve(LiV, tf.eye(iV.shape[-1],dtype=dtype))
  logDetV = -2 * tf.reduce_sum(tfm.log(tfla.diag_part(LiV)))
  V1 = rho05[...,:,None] * V * rho05[...,None,:]
  V2 = tf.sqrt(1-rho)[...,:,None] * V * tf.sqrt(1-rho)[...,None,:]
  XiSY, _, _, _, logDet = recFunBilinearDet(root, treeList, X, Y, V, rho05, iV, V1, V2, logDetV, dtype)
  return XiSY, logDet


def recFunSampleUp(node, treeList, treeListTemp, iV, rho05, rho205Mat, XTiDX, XTiDS, diV2x, dtype=tf.float64):
  nChild = treeList[node]["n"]
  edgeLenVec = treeList[node]["edgeLen"]
  iSigmaChild_m_list, iSigmaChild_list = [None]*nChild, [None]*nChild
  for i in range(nChild):
    childNode = treeList[node]["child"][i]
    if treeList[childNode]["n"] > 0:
      recRes = recFunSampleUp(childNode, treeList, treeListTemp, iV, rho05, rho205Mat, XTiDX, XTiDS, diV2x, dtype)
      treeListTemp[childNode]["iSm"] = iSigmaChild_m_list[i] = recRes[0]
      treeListTemp[childNode]["iS"] = iSigmaChild_list[i] = recRes[1]
    else: # can be redone in vectorized manner
      treeListTemp[childNode]["iS"] = iSigmaAdded = XTiDX[:,:,childNode]
      treeListTemp[childNode]["iSm"] = iSigmaAdded_beta = XTiDS[:,childNode,None]
      rho205 = rho205Mat[:,childNode]
      D2_iSigmaAdded = rho205[:,None] * iSigmaAdded
      W = iV + D2_iSigmaAdded * rho205
      LW = tfla.cholesky(W)
      iLW_D2_iSigmaAdded = tfla.triangular_solve(LW, D2_iSigmaAdded)
      iSigmaChild_list[i] = iSigmaAdded - tf.matmul(iLW_D2_iSigmaAdded, iLW_D2_iSigmaAdded, transpose_a=True)
      iSigmaChild_m_list[i] = iSigmaAdded_beta - tf.matmul(iLW_D2_iSigmaAdded, tfla.triangular_solve(LW, rho205[:,None]*iSigmaAdded_beta), transpose_a=True)
    
  iSigmaChild, iSigmaChild_m = [tf.stack(vChild_list, 0) for vChild_list in [iSigmaChild_list, iSigmaChild_m_list]]
  D1_iSigmaChild = rho05[:,None] * iSigmaChild
  W = iV + edgeLenVec[:,None,None] * (D1_iSigmaChild * rho05)
  LW = tfla.cholesky(W)
  iLW_D1_iSigmaChild = tfla.triangular_solve(LW, D1_iSigmaChild)
  iSigmaHat = iSigmaChild - edgeLenVec[:,None,None] * tf.matmul(iLW_D1_iSigmaChild, iLW_D1_iSigmaChild, transpose_a=True)
  iSigmaHat_m = iSigmaChild_m - edgeLenVec[:,None,None] * tf.matmul(iLW_D1_iSigmaChild, tfla.triangular_solve(LW, rho05[:,None]*iSigmaChild_m), transpose_a=True)
 
  treeList[node]["iSm"] = tf.reduce_sum(iSigmaHat_m, 0)
  treeList[node]["iS"] = tf.reduce_sum(iSigmaHat, 0)
  return treeList[node]["iSm"], treeList[node]["iS"]


def recFunSampleDown(node, treeList, treeListTemp, V, V1, rho05, rho205Mat, sdMult, dtype=tf.float64):
  nc = tf.shape(V)[0]
  nChild = treeList[node]["n"]
  edgeLenVec = treeList[node]["edgeLen"]
  beta = treeListTemp[node]["beta"]
  for i in range(nChild):
    childNode = treeList[node]["child"][i]
    iS = treeListTemp[childNode]["iS"]
    iSm = treeListTemp[childNode]["iSm"]
    if treeList[childNode]["n"] > 0:
      # W = edgeLenVec[i] * V %*% matmulDiagMat(sqrt(rho), matmulMatDiag(iS, sqrt(rho))) %*% V + V
      W = edgeLenVec[i] * tf.einsum("ik,kc,cj->ij", V, rho05[:,None] * iS * rho05, V) + V
      LW = tfla.cholesky(W)
      U = edgeLenVec[i] * tf.matmul(V1, iS) + tf.eye(nc, dtype=dtype)
      mu = tfla.solve(U, beta + edgeLenVec[i] * tf.matmul(V1,iSm))
      BetaRanPart = edgeLenVec[i]**0.5 * rho05[:,None] * tf.matmul(V, tfla.triangular_solve(LW, tfr.normal([nc,1],dtype=dtype), adjoint=True))
      treeListTemp[childNode]["beta"] = mu + sdMult*BetaRanPart
      recFunSampleDown(childNode, treeList, treeListTemp, V, V1, rho05, rho205Mat, sdMult, dtype)
    else:
      rho205 = rho205Mat[:,childNode]
      W = edgeLenVec[i] * V1 + rho205[:,None] * V * rho205
      LW = tfla.cholesky(W)
      iSigma = iS + tfla.cholesky_solve(LW, tf.eye(nc,dtype=dtype))
      iSigma_mu = iSm + tfla.cholesky_solve(LW, beta)
      LiSigma = tfla.cholesky(iSigma)
      BetaRanPart = tfla.triangular_solve(LiSigma, tfr.normal([nc,1],dtype=dtype), adjoint=True)
      treeListTemp[childNode]["beta"] = tfla.triangular_solve(LiSigma, tfla.triangular_solve(LiSigma, iSigma_mu), adjoint=True) + sdMult*BetaRanPart
  return(treeList)


def phyloFastSample(treeList, root, V, iV, rho, rho2Mat, XTiDX, XTiDS, sdMult=1, dtype=tf.float64):
  print("phyloFast phyloFastSample")
  ns = XTiDS.shape[1]
  nc = tf.shape(V)[0]
  treeListTemp = [None] * len(treeList)
  for i in range(len(treeListTemp)):
    treeListTemp[i] = {"iS":None, "iSm":None, "beta":None}
  
  recFunSampleUp(root, treeList, treeListTemp, iV, tfm.sqrt(rho), tfm.sqrt(rho2Mat), XTiDX, XTiDS, dtype)
  treeListTemp[root]["beta"] = tf.zeros([nc,1], dtype)
  V1 = tfm.sqrt(rho)[:,None] * V * tfm.sqrt(rho)
  recFunSampleDown(root, treeList, treeListTemp, V, V1, tfm.sqrt(rho), tfm.sqrt(rho2Mat), sdMult, dtype)
  Beta = tf.concat([node["beta"] for node in treeListTemp[:ns]], -1)
  return(Beta)