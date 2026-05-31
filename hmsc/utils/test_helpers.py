import numpy as np
import tensorflow as tf

def complete_model_data(modelData, params=None, modelDims=None):
    if params is None:
        params = {}
    if modelDims is None:
        modelDims = {}
    
    ns = modelDims.get("ns", 1001)
    ny = modelDims.get("ny", 701)
    nr = modelDims.get("nr", 7)
    
    if "Yo" not in modelData:
        if "Y" in modelData:
            Y = modelData["Y"]
            if isinstance(Y, (tf.Tensor, tf.Variable)):
                modelData["Yo"] = tf.math.logical_not(tf.math.is_nan(Y))
            else:
                modelData["Yo"] = np.logical_not(np.isnan(Y))
        else:
            ny_val = ny
            ns_val = ns
            if "Z" in params:
                ny_val, ns_val = params["Z"].shape
            elif "Beta" in params:
                ns_val = params["Beta"].shape[-1]
            modelData["Yo"] = np.ones([ny_val, ns_val], dtype=bool)
            
    defaults = {
        "Loff": None,
        "phyloFlag": False,
        "phyloFast": False,
        "phyloTreeList": None,
        "phyloTreeRoot": None,
        "phyloTreeDepth": None,
        "covRhoGroup": None,
        "C": None,
        "eC": None,
        "VC": None,
        "XSel": None,
        "XRRR": None,
        "rhoGroup": None,
        "T": None,
        "distr": None,
    }
    for k, v in defaults.items():
        if k not in modelData:
            modelData[k] = v
            
    if modelData.get("distr") is None:
        modelData["distr"] = np.ones([ns, 2])
        
    return modelData
