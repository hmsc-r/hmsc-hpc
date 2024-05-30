import rdata
import tensorflow as tf
import xarray as xr


def convert_to_numpy(obj):
    if isinstance(obj, xr.DataArray):
        return obj.to_numpy()

    if isinstance(obj, tf.Tensor):
        return obj.numpy()

    if isinstance(obj, dict):
        new = {}
        for key, value in obj.items():
            new[key] = convert_to_numpy(value)
        return new

    if isinstance(obj, list):
        new = []
        for value in obj:
            new.append(convert_to_numpy(value))
        return new

    return obj


def load_model_from_rds(rds_file_path):
    init_obj = rdata.read_rds(rds_file_path)
    init_obj = convert_to_numpy(init_obj)
    return init_obj, init_obj["hM"]


def save_chains_postList_to_rds(postList, postList_file_path, nChains, elapsedTime=-1, flag_save_eta=True):
    data = {}
    data["list"] = convert_to_numpy(postList)
    if not flag_save_eta:
        for i in range(len(data["list"])):
            for j in range(len(data["list"][i])):
                data["list"][i][j]["Eta"] = None
    data["time"] = elapsedTime
    rdata.write_rds(postList_file_path, data)
