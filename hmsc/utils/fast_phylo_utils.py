import numpy as np
import tensorflow as tf
tfla, tfm, tfr, tfs = tf.linalg, tf.math, tf.random, tf.sparse

# class PhyloTreeNode():
#   def __init__(self):
#     self.n = None
#     self.child = None
#     self.edgeLen = None
#     self.parentEdgeLen = None


def recFunDepth(node, treeList):
  nChild = treeList[node]["n"]
  depthChild = [None]*nChild
  for i in range(nChild):
    childNode = treeList[node]["child"][i]
    if treeList[childNode]["n"] > 0:
      depthChild[i] = recFunDepth(childNode, treeList)
    else:
      treeList[childNode]["depth"] = depthChild[i] = 0
  treeList[node]["depth"] = np.max(np.array(depthChild)) + 1
  return treeList[node]["depth"]


def recFunBalanceDepth(node, treeList):
  nChild = treeList[node]["n"]
  for i in range(nChild):
    childNode = treeList[node]["child"][i]
    if treeList[childNode]["n"] > 0:
      recFunBalanceDepth(childNode, treeList)
    if treeList[childNode]["depth"] < treeList[node]["depth"] - 1:
      nN = treeList[node]["depth"] - treeList[childNode]["depth"] - 1
      
      segEdgeLen = treeList[node]["edgeLen"][i] / (nN+1)
      for k in range(nN):
        newNode = {
          "n": 1,
          "child": np.array([len(treeList) + 1]),
          "edgeLen": np.array([segEdgeLen]),
          "parent": len(treeList) - 1,
          "parentEdgeLen": segEdgeLen,
          "depth": treeList[node]["depth"] - (k+1)
        }
        if k == 0:
          newNode["parent"] = node
          treeList[node]["child"][i] = len(treeList)
          treeList[node]["edgeLen"][i] = segEdgeLen
        if k == nN - 1:
          newNode["child"] = np.array([childNode])
          treeList[childNode]["parent"] = len(treeList)
          treeList[childNode]["parentEdgeLen"] = segEdgeLen
      
        # for k in range(nN):
        #   newNode = {
        #     "n": 1,
        #     "child": np.array([len(treeList) + 1]),
        #     "edgeLen": np.array([0.0]),
        #     "parent": len(treeList) - 1,
        #     "parentEdgeLen": 0.0,
        #     "depth": treeList[node]["depth"] - (k+1)
        #   }
        #   if k == 0:
        #     newNode["parent"] = node
        #     newNode["parentEdgeLen"] = treeList[childNode]["parentEdgeLen"]
        #     treeList[node]["child"][i] = len(treeList)
        #   if k == nN - 1:
        #     newNode["child"] = np.array([childNode])
        #     treeList[childNode]["parent"] = len(treeList)
        #     treeList[childNode]["parentEdgeLen"] = 0.0
        treeList.append(newNode)
    

def getIndMatListFromTree(treeList, root=None, depth=None):
  if depth is None:
    depth = treeList[root]["depth"]
  indMatList, parentEdgeLenList = [None]*depth, [None]*(depth+1)
  for d in range(depth):
    ind = np.where(np.array([node["depth"] for node in treeList]) == d)[0]
    parent = np.array([treeList[i]["parent"] for i in ind])
    parentEdgeLenList[d] = np.array([treeList[i]["parentEdgeLen"] for i in ind])
    indMatList[d] = np.stack([np.arange(ind.size), ind, parent], axis=-1)
  parentEdgeLenList[depth] = np.array([0.0])
  
  for d in range(depth-1):
    asort = np.argsort(indMatList[d+1][:,1])
    indParent = asort[np.searchsorted(indMatList[d+1][:,1], indMatList[d][:,2], sorter=asort)]
    indMatList[d] = np.hstack([indMatList[d], indParent[:,None]])
  indMatList[depth-1] = np.hstack([indMatList[depth-1], np.zeros([indMatList[depth-1].shape[0],1],dtype=np.int32)])
  return indMatList, parentEdgeLenList
  

def phyloFastBilinearDetBatched(treeList, X, Y, root, iV, rho, dtype=tf.float64):
  print("phyloFast phyloFastBilinearDetBatched")
  batchShape = list(rho.shape[:-1])
  if len(batchShape) > 1:
    raise ValueError("Batch shape in phyloFastBilinearDetBatched() can be no more than one")
  rho05 = tfm.sqrt(rho)
  LiV = tfla.cholesky(iV)
  V = tfla.cholesky_solve(LiV, tf.eye(iV.shape[-1],dtype=dtype))
  logDetV = -2 * tf.reduce_sum(tfm.log(tfla.diag_part(LiV)))
  V1 = rho05[...,:,None] * V * rho05[...,None,:]
  V2 = tf.sqrt(1-rho)[...,:,None] * V * tf.sqrt(1-rho)[...,None,:]
  depth = treeList[root]["depth"]
  indMatList, parentEdgeLenList = getIndMatListFromTree(treeList, depth=depth)
    
  XiSYList, OneiSXList, OneiSYList, OneiSOneList, logDetList = [[None]*(depth+1) for i in range(5)]
  d = 0
  parentEdgeLen = tf.reshape(tf.cast(parentEdgeLenList[d],dtype), [-1] + [1]*len(batchShape) + [1, 1])
  S = parentEdgeLen * V1 + V2
  iSX = tfla.solve(S, X)
  iSY = tfla.solve(S, Y)
  XiSYList[d] = tf.matmul(X, iSY, transpose_a=True)
  OneiSXList[d], OneiSYList[d] = iSX, iSY
  LS = tfla.cholesky(S)
  OneiSOneList[d] = tfla.cholesky_solve(LS, tf.eye(S.shape[-1],batch_shape=S.shape[:-2],dtype=dtype))
  logDetList[d] = 2 * tf.reduce_sum(tfm.log(tfla.diag_part(LS)), -1)
  for d in range(depth):
    nextLevelNodeN = indMatList[d+1].shape[0] if d < depth-1 else 1
    scatterInd = indMatList[d][:,-1,None]
    XiSYSum, OneiSXSum, OneiSYSum, OneiSOneSum = [tf.scatter_nd(scatterInd, vChild, [nextLevelNodeN]+batchShape+vChild.shape[-2:]) \
                                                  for vChild in [XiSYList[d], OneiSXList[d], OneiSYList[d], OneiSOneList[d]]]
    logDetSum = tf.scatter_nd(scatterInd, logDetList[d], [nextLevelNodeN]+batchShape)
      
    parentEdgeLen = tf.reshape(tf.cast(parentEdgeLenList[d+1],dtype), [-1] + [1]*len(batchShape) + [1,1])
    W = iV + parentEdgeLen * (rho05[...,:,None] * OneiSOneSum * rho05[...,None,:])
    LW = tfla.cholesky(W)
    logDetList[d+1] = logDetSum + logDetV + 2*tf.reduce_sum(tfm.log(tfla.diag_part(LW)), -1)
    iLW_Drho05_OneiSXSum = tfla.triangular_solve(LW, rho05[...,:,None] * OneiSXSum)
    iLW_Drho05_OneiSYSum = tfla.triangular_solve(LW, rho05[...,:,None] * OneiSYSum)
    iLW_Drho05_OneiSOneSum = tfla.triangular_solve(LW, rho05[...,:,None] * OneiSOneSum)
    
    XiSYList[d+1] = XiSYSum - parentEdgeLen * tf.matmul(iLW_Drho05_OneiSXSum, iLW_Drho05_OneiSYSum, transpose_a=True)
    OneiSXList[d+1] = OneiSXSum - parentEdgeLen * tf.matmul(iLW_Drho05_OneiSOneSum, iLW_Drho05_OneiSXSum, transpose_a=True)
    OneiSYList[d+1] = OneiSYSum - parentEdgeLen * tf.matmul(iLW_Drho05_OneiSOneSum, iLW_Drho05_OneiSYSum, transpose_a=True)
    OneiSOneList[d+1] = OneiSOneSum - parentEdgeLen * tf.matmul(iLW_Drho05_OneiSOneSum, iLW_Drho05_OneiSOneSum, transpose_a=True)
    
  XiSY = tf.squeeze(XiSYList[depth], 0)
  logDet = tf.squeeze(logDetList[depth], 0)
  return XiSY, logDet


def withTranspose_phyloFastBilinearDetBatched(treeList, X, Y, root, iV, rho, dtype=tf.float64):
  print("phyloFast phyloFastBilinearDetBatched")
  batchShape = list(rho.shape[:-1])
  if len(batchShape) > 1:
    raise ValueError("Batch shape in phyloFastBilinearDetBatched() can be no more than one")
  rho05 = tfm.sqrt(rho)
  LiV = tfla.cholesky(iV)
  V = tfla.cholesky_solve(LiV, tf.eye(iV.shape[-1],dtype=dtype))
  logDetV = -2 * tf.reduce_sum(tfm.log(tfla.diag_part(LiV)))
  V1 = rho05[...,:,None] * V * rho05[...,None,:]
  V2 = tf.sqrt(1-rho)[...,:,None] * V * tf.sqrt(1-rho)[...,None,:]
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
    W = iV[...,None,:,:] + parentEdgeLen[:,None,None] * (rho05[...,None,:,None] * OneiSOneSum * rho05[...,None,None,:])
    LW = tfla.cholesky(W)
    logDetList[d+1] = logDetSum + logDetV + 2*tf.reduce_sum(tfm.log(tfla.diag_part(LW)), -1)
    iLW_Drho05_OneiSXSum = tfla.triangular_solve(LW, rho05[...,None,:,None] * OneiSXSum)
    iLW_Drho05_OneiSYSum = tfla.triangular_solve(LW, rho05[...,None,:,None] * OneiSYSum)
    iLW_Drho05_OneiSOneSum = tfla.triangular_solve(LW, rho05[...,None,:,None] * OneiSOneSum)
    
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


def phyloFastSampleBatched(treeList, root, V, iV, rho, rho2Mat, XTiDX, XTiDS, sdMult=1, dtype=tf.float64):
  EPS = tf.cast(0*1e-6, dtype)
  print("phyloFast phyloFastSampleBatched")
  ns = XTiDS.shape[1]
  nc = tf.shape(V)[0]
  depth = treeList[root]["depth"]
  indMatList, parentEdgeLenList = getIndMatListFromTree(treeList, depth=depth)
  rho05, rho205Mat = tfm.sqrt(rho), tfm.sqrt(rho2Mat)
  V1 = tfm.sqrt(rho)[:,None] * V * tfm.sqrt(rho)
  
  # recFunSampleUp(...)  
  iSigmaList, nuList = [[None]*(depth+1) for i in range(2)]
  iSigmaBase = iS = tf.transpose(XTiDX, [2,0,1])
  nuBase = nu = tf.transpose(XTiDS)[:,:,None]
  rho205 = tf.transpose(rho205Mat)
  D2_iS = rho205[:,:,None] * iS
  W = iV + D2_iS * rho205[:,None,:]
  LW = tfla.cholesky(W + EPS*tf.eye(nc, dtype=dtype))
  iLW_D2_iS = tfla.triangular_solve(LW, D2_iS)
  iSigmaList[0] = iS - tf.matmul(iLW_D2_iS, iLW_D2_iS, transpose_a=True)
  nuList[0] = nu - tf.matmul(iLW_D2_iS, tfla.triangular_solve(LW, rho205[:,:,None]*nu), transpose_a=True)
  for d in range(depth):
    nextLevelNodeN = indMatList[d+1].shape[0] if d < depth-1 else 1
    scatterInd = indMatList[d][:,-1,None]
    parentEdgeLen = parentEdgeLenList[d]
    nu, iS = nuList[d], iSigmaList[d]
    D1_iS = rho05[:,None] * iS
    W = iV + parentEdgeLen[:,None,None] * (D1_iS * rho05)
    LW = tfla.cholesky(W + EPS*tf.eye(nc, dtype=dtype))
    iLW_D1_iS = tfla.triangular_solve(LW, D1_iS)
    iSigmaHat = iS - parentEdgeLen[:,None,None] * tf.matmul(iLW_D1_iS, iLW_D1_iS, transpose_a=True)
    nuHat = nu - parentEdgeLen[:,None,None] * tf.matmul(iLW_D1_iS, tfla.triangular_solve(LW, rho05[:,None]*nu), transpose_a=True)
    nuList[d+1] = tf.scatter_nd(scatterInd, nuHat, [nextLevelNodeN,nc,1])
    iSigmaList[d+1] = tf.scatter_nd(scatterInd, iSigmaHat, [nextLevelNodeN,nc,nc])

  # recFunSampleDown(...)
  betaList = [None]*(depth+1)
  betaList[depth] = tf.zeros([1,nc,1], dtype)
  for d in range(depth,1,-1):
    levelNodeN = indMatList[d-1].shape[0]
    parentEdgeLen = parentEdgeLenList[d-1]
    betaParent = tf.gather(betaList[d], indMatList[d-1][:,-1], axis=0)
    nu, iS = nuList[d-1], iSigmaList[d-1]
    W = parentEdgeLen[:,None,None] * tf.einsum("ck,jkp,pq->jcq", V, rho05[:,None]*iS*rho05, V) + V
    LW = tfla.cholesky(W + EPS*tf.eye(nc, dtype=dtype))
    U = parentEdgeLen[:,None,None] * tf.matmul(V1, iS) + tf.eye(nc, dtype=dtype)
    mu = tfla.solve(U, betaParent + parentEdgeLen[:,None,None] * tf.matmul(V1,nu))
    betaRanPart = parentEdgeLen[:,None,None]**0.5 * rho05[:,None] * tf.matmul(V, tfla.triangular_solve(LW, tfr.normal([levelNodeN,nc,1],dtype=dtype), adjoint=True))
    betaList[d-1] = mu + sdMult*betaRanPart
    
  levelNodeN = indMatList[0].shape[0]
  parentEdgeLen = parentEdgeLenList[0]
  betaParent = tf.gather(betaList[1], indMatList[0][:,-1], axis=0)
  W = parentEdgeLen[:,None,None] * V1 + rho205[:,:,None] * V * rho205[:,None,:]
  LW = tfla.cholesky(W + EPS*tf.eye(nc, dtype=dtype))
  iSigma = iSigmaBase + tfla.cholesky_solve(LW, tf.eye(nc,batch_shape=[ns],dtype=dtype))
  nu = nuBase + tfla.cholesky_solve(LW, betaParent)
  LiSigma = tfla.cholesky(iSigma)
  BetaRanPart = tfla.triangular_solve(LiSigma, tfr.normal([ns,nc,1],dtype=dtype), adjoint=True)
  betaTemp = tfla.triangular_solve(LiSigma, tfla.triangular_solve(LiSigma, nu), adjoint=True) + sdMult*BetaRanPart
  Beta = tf.transpose(tf.squeeze(betaTemp, -1))
  return(Beta)