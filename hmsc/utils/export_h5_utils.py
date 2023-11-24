import os
import sys
import h5py
import numpy as np


def load_model_from_h5(rds_file_path):
    pass


def is_empty(a):

    return all(map(is_empty, a)) if isinstance(a, list) else False


def get_empty_params(postList, params):

  empty_params = set()

  for param in params:

    a = postList[0][0][param]

    if is_empty(a) or a is None:
      empty_params.add(param)
  
  return empty_params


def get_listed_params(postList, params):

  listed_params = set()

  for param in params:
    if isinstance(postList[0][0][param], list):
      listed_params.add(param)

  return listed_params


def get_param_arr(postList, param, nChains, num_samples, listed_params):

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


def save_chains_postList_to_h5(postList, h5_filename, nChains, elapsedTime=-1, flag_save_eta=True, dtype='float32'):

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
