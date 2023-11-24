import os
import sys
import h5py
import numpy as np

from collections import abc


def load_model_from_h5(rds_file_path: str) -> None:

    pass


def is_empty(a: list) -> bool:
    """Check whether a nested list is empty."""

    return all(map(is_empty, a)) if isinstance(a, list) else False


def get_empty_params(postList: list, params: abc.KeysView) -> set:
    """Get set of None-valued params."""

    empty_params = set()

    for param in params:

        a = postList[0][0][param]

        if is_empty(a) or a is None:
            empty_params.add(param)
    
    return empty_params


def get_listed_params(postList: list, params: abc.KeysView) -> set:
    """Get set of list-based params."""

    listed_params = set()

    for param in params:
        if isinstance(postList[0][0][param], list):
            listed_params.add(param)

    return listed_params


def get_param_arr(
    postList: list, 
    param: str, 
    nChains: int, 
    num_samples: int, 
    listed_params: abc.KeysView
) -> np.ndarray:
    """
    Get params values as numpy arrays.
    
    Say for (chain, sample) = (0,0), postList[0][0]['Beta'].shape is an (16,2) 2d-tensor.

    The function returns all 'Beta' tensors stacked into single 4d numpy array.
    e.g. for 8 chains and 25 samples, arr[8,25,16,2] is returned.
    """

    if param in listed_params:
        a = np.moveaxis(np.dstack(postList[0][0][param]), -1, 0)
    else:
        a = postList[0][0][param].numpy()

    if a.ndim == 1:
        x = a.shape
        arr = np.empty(shape=(nChains, num_samples, x[0]))

    elif a.ndim == 2:
        y, x = a.shape
        arr = np.empty(shape=(nChains, num_samples, y, x))

    elif a.ndim == 3:
        z, y, x = a.shape
        arr = np.empty(shape=(nChains, num_samples, z, y, x))

    else:
        raise NotImplemented

    for j in range(nChains):
        for k in range(num_samples):
            if param not in listed_params:
                arr[j,k,...] = postList[j][k][param].numpy()
            else:
                arr[j,k,...] = np.moveaxis(np.dstack(postList[j][k][param]), -1, 0)

    return arr


def save_chains_postList_to_h5(
    postList: list, 
    h5_filename: str, 
    nChains: int, 
    elapsedTime: float = -1, 
    flag_save_eta: bool = True, 
    dtype: str = 'float32'
) -> None:

    """
    Save posteriors from MCMC chains to a h5 dataset.

    Parameters
    ----------
    postList: list
        list of posteriors, per chain per sample.
    h5_filename: str
        file with posteriors in h5 dataset.
    nChains: int
        number of MCMC chains run.
    elapsedTime: float
        time elapsed by whole Gibbs sampler. By default not recorded, set to -1.
    flag_save_eta: bool
        whether to save Eta posterior. By default saved to h5 dataset.
    dtype: str
        data type for h5 dataset. By default set to 'float32'.

    Returns
    -------
    None

    See Also
    --------
    save_chains_postList_to_json
    save_chains_postList_to_rds

    """

    num_chains = len(postList[0])
    num_samples = len(postList[0][0])

    params = postList[0][0].keys()

    listed_params = get_listed_params(postList, params)
    empty_params = get_empty_params(postList, params)

    if os.path.exists(h5_filename):
        # prevent overwriting a file
        sys.exit('File already exists!')

    h5_fout = h5py.File(h5_filename, 'w')

    for param in params:
        if param not in empty_params:

            h5_fout.create_dataset(
                name=param,
                data=get_param_arr(postList, param, nChains, num_samples, listed_params),
                compression='gzip', compression_opts=4,
                dtype=dtype)

    h5_fout.close()
