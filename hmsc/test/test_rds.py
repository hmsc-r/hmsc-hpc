import tempfile
import os
import numpy as np
import pytest
import rdata
import tensorflow as tf
import xarray as xr

from hmsc.utils.rds import convert_to_numpy, save_chains_postList_to_rds


def test_convert_to_numpy_tensors():
    t_f32 = tf.constant([1.0, 2.5, 3.75], dtype=tf.float32)
    t_f64 = tf.constant([1.0, 2.5, 3.75], dtype=tf.float64)
    t_i32 = tf.constant([1, 2, 3], dtype=tf.int32)

    res_f32 = convert_to_numpy(t_f32)
    assert isinstance(res_f32, np.ndarray)
    assert res_f32.dtype == np.float64
    np.testing.assert_allclose(res_f32, [1.0, 2.5, 3.75])

    res_f64 = convert_to_numpy(t_f64)
    assert isinstance(res_f64, np.ndarray)
    assert res_f64.dtype == np.float64

    res_i32 = convert_to_numpy(t_i32)
    assert isinstance(res_i32, np.ndarray)
    assert res_i32.dtype == np.int32


def test_convert_to_numpy_xarray_and_scalars():
    xr_arr = xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    res_xr = convert_to_numpy(xr_arr)
    assert isinstance(res_xr, np.ndarray)
    assert res_xr.dtype == np.float64

    scalar_f32 = np.float32(3.14)
    res_scalar = convert_to_numpy(scalar_f32)
    assert isinstance(res_scalar, (np.float64, float))
    assert res_scalar.dtype == np.float64


def test_convert_to_numpy_nested():
    data = {
        "tensor": tf.constant([1.0, 2.0], dtype=tf.float32),
        "nested_list": [
            np.array([3.0, 4.0], dtype=np.float32),
            {"inner_tensor": tf.constant(5.0, dtype=tf.float32)},
        ],
        "int_array": np.array([1, 2, 3], dtype=np.int32),
    }

    converted = convert_to_numpy(data)
    assert converted["tensor"].dtype == np.float64
    assert converted["nested_list"][0].dtype == np.float64
    assert converted["nested_list"][1]["inner_tensor"].dtype == np.float64
    assert converted["int_array"].dtype == np.int32


def test_save_chains_postlist_fp32():
    with tempfile.TemporaryDirectory() as tmpdir:
        rds_path = os.path.join(tmpdir, "test_fp32_postList.rds")

        # Simulate postList structure produced by Gibbs sampler with fp32
        postList = [
            [
                {
                    "Beta": tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32),
                    "rhoInd": np.array([0], dtype=np.int32),
                    "alphaInd": [np.array([0], dtype=np.int32)],
                    "sigma": tf.constant([0.5, 1.5], dtype=tf.float32),
                    "Eta": [tf.constant([[0.1, 0.2]], dtype=tf.float32)],
                }
            ]
        ]

        # Should not raise AssertionError when saving fp32 samples
        save_chains_postList_to_rds(postList, rds_path, nChains=1, elapsedTime=1.23, flag_save_eta=True)

        assert os.path.exists(rds_path)
        loaded = rdata.read_rds(rds_path)
        assert "list" in loaded
        assert "time" in loaded
