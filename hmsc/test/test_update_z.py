import pytest
import numpy as np
import tensorflow as tf

from numpy import nan
from pytest import approx

from hmsc.updaters.updateZ import updateZ
from hmsc.utils.test_helpers import complete_model_data

SEED = 42

def convert_to_tf(data):
    if isinstance(data, np.ndarray):
        new = tf.convert_to_tensor(data, dtype=data.dtype)
    elif isinstance(data, list):
        new = []
        for value in data:
            new.append(convert_to_tf(value))
    elif isinstance(data, dict):
        new = {}
        for key, value in data.items():
            new[key] = convert_to_tf(value)
    else:
        new = data

    return new


def run_test(input_values, ref_values, *,
             tnlib='tf',
             poisson_preupdate_z=False,
             poisson_marginalize_z=False,
             ):
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    # Convert inputs to tf
    params, data, rLHyperparams = input_values
    data['Yo'] = np.logical_not(np.isnan(data['Y']))
    params, data, rLHyperparams = map(convert_to_tf, (params, data, rLHyperparams))
    dtype = data['Y'].dtype
    assert dtype == tf.float64

    # Calculate
    Z, iD, omega = updateZ(
        params, data, rLHyperparams,
        poisson_preupdate_z=poisson_preupdate_z,
        poisson_marginalize_z=poisson_marginalize_z,
        truncated_normal_library=tnlib,
        dtype=dtype,
        )
    Z = Z.numpy()
    iD = iD.numpy()
    omega = omega.numpy()

    assert Z.shape == params['Z'].shape
    assert iD.shape == params['Z'].shape
    assert omega.shape == params['poisson_omega'].shape

    # Print values
    print()
    for name, array in (('Z', Z),
                        ('iD', iD),
                        ('omega', omega)):
        print(f'    {name} = \\')
        print(np.array2string(array, separator=', ', max_line_width=200))

    # Test against reference
    ref_Z, ref_iD, ref_omega = map(np.asarray, ref_values)
    assert Z == approx(ref_Z, nan_ok=True)
    assert iD == approx(ref_iD)
    if omega.size > 0 or ref_omega.size > 0:
        assert omega == approx(ref_omega, nan_ok=True)


def default_input_values(rng):
    ns = 7
    ny = 5
    nb = 11
    nr = 2
    ne = 3

    distr = np.array(
        [[1, 1],
         [2, 1],
         [3, 1],
         [1, 1],
         [3, 1],
         [2, 1],
         [1, 1]],
        dtype=np.int32,
    )

    Pi = np.array(
        [[0, 2],
         [1, 2],
         [2, 2],
         [0, 0],
         [1, 0]],
        dtype=np.int32,
    )

    ns3 = np.sum(distr[:, 0] == 3)

    params = dict(
        Z=rng.random(size=(ny, ns)),
        Beta=rng.random(size=(nb, ns)),
        sigma=rng.random(size=ns),
        Xeff=rng.random(size=(ny, nb)),
        poisson_omega=rng.random(size=(ny, ns3)),
    )
    data = dict(
        Y=rng.integers(2, size=(ny, ns)).astype(np.float64),
        Pi=Pi,
        distr=distr,
    )

    assert data['distr'].shape == (ns, 2)
    assert data['Pi'].shape == (ny, nr)

    EtaList = [
        rng.random(ne * 2).reshape(ne, 2),
        rng.random(ne * 2).reshape(ne, 2),
    ]
    LambdaList = [
        rng.random(2 * ns).reshape(2, ns),
        rng.random(2 * ns).reshape(2, ns),
    ]
    rLHyperparams = [
        dict(xDim=0),
        dict(xDim=0),
    ]
    assert len(EtaList) == len(LambdaList) == len(rLHyperparams) == nr

    params["Eta"] = EtaList
    params["Lambda"] = LambdaList

    data = complete_model_data(data, params)
    return params, data, rLHyperparams


def default_reference_values():
    Z = \
[[0, 2.5361722, 1.4075434, 1, 0.13129104, -0.027216023, 1],
 [0, 3.0099937, 1.3567631, 0, 0.89406079, -0.0065145159, 1],
 [1, -0.030795585, 1.7788299, 0, 0.6501287, 3.001938, 1],
 [0, 2.7197177, 1.6854148, 0, 0.11587453, 3.9297946, 1],
 [0, -0.051366734, 1.4497069, 0, 0.32676159, 4.3054389, 0]]
    iD = \
[[10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 5.3560603]]
    omega = \
[[82.695649, 73.524467],
 [75.731191, 80.315436],
 [77.777866, 79.681864],
 [81.555963, 73.141613],
 [80.272408, 75.35137]]
    return Z, iD, omega


def test_defaults():
    rng = np.random.default_rng(seed=SEED)
    run_test(
        default_input_values(rng),
        default_reference_values(),
    )


@pytest.mark.parametrize("tnlib", ['tf', 'tfd'])
def test_tnlib(tnlib):
    rng = np.random.default_rng(seed=SEED)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    if tnlib == 'tfd':
        Z = \
[[0, 2.1792131, 1.4075434, 1, 0.13129104, -0.056942036, 1],
 [0, 2.7920181, 1.3567631, 0, 0.89406079, -0.13361183, 1],
 [1, -0.016631796, 1.7788299, 0, 0.6501287, 4.7008139, 1],
 [0, 4.6868599, 1.6854148, 0, 0.11587453, 4.4379023, 1],
 [0, -0.1906678, 1.4497069, 0, 0.32676159, 2.6956344, 0]]
        omega = \
[[82.695649, 73.524467],
 [75.731191, 80.315436],
 [77.777866, 79.681864],
 [81.555963, 73.141613],
 [80.272408, 75.35137]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
        tnlib=tnlib,
    )


def test_y_nan():
    rng = np.random.default_rng(seed=SEED)
    params, data, rLHyperparams = default_input_values(rng)

    Y = rng.integers(3, size=data['Y'].shape).astype(np.float64)
    Y[Y == 2] = nan
    data['Y'] = Y

    Z = \
[[2.6390101, 2.5361722, 1.4075434, 1, 0.14450654, 2.7572329, 1],
 [4.2579463, 3.0099937, 0, 1, 0.89406079, 3.1342097, 0],
 [1, 3.4230493, 1.787066, 0, 0.66243985, 3.001938, 4.6532661],
 [0, 2.7197177, 1.6935875, 0, 0.1025117, -0.058522219, 3.1974797],
 [3.3675925, 3.5074093, 0, 3.1999928, 0.31382702, 4.3054389, 0]]
    iD = \
[[0, 2.9806676, 32.001416, 1.3627914, 1.7380668, 0, 5.3560603],
 [0, 0, 0, 1.3627914, 1.7380668, 1.9318932, 5.3560603],
 [10.682681, 0, 32.001416, 1.3627914, 1.7380668, 0, 0],
 [10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 0],
 [0, 0, 0, 0, 1.7380668, 0, 5.3560603]]
    omega = \
[[82.695649, 73.59784],
 [nan, 80.315436],
 [77.855969, 79.762635],
 [81.638104, 73.067929],
 [nan, 75.275854]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
    )


@pytest.mark.parametrize("distr_case", [1, 2, 3])
def test_distr_case(distr_case):
    rng = np.random.default_rng(seed=SEED)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    if distr_case in [1, 2, 3]:
        data['distr'][:, 0] = distr_case
        ns3 = np.sum(data['distr'][:, 0] == 3)
        ny = params['poisson_omega'].shape[0]
        params['poisson_omega'] = rng.random(size=(ny, ns3))

    if distr_case == 1:
        Z = \
[[0, 1, 0, 1, 0, 0, 1],
 [0, 1, 1, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 1, 1],
 [0, 1, 0, 0, 1, 1, 1],
 [0, 0, 0, 0, 1, 1, 0]]
        omega = []
    elif distr_case == 2:
        Z = \
[[-0.034903786, 1.8709593, -0.0022825973, 1.086493, -0.46063598, -0.029954962, 2.4355521],
 [-0.0047732086, 2.5993001, 3.7630349, -0.023796106, -0.16076815, -0.014827197, 2.97104],
 [3.6562421, -0.12845149, -0.029751199, -0.095328732, -0.078873834, 3.8221587, 4.2187888],
 [-0.019088003, 2.2230759, -0.000747233, -0.18328067, 2.6459235, 3.4708877, 2.5109508],
 [-0.045505706, -0.23786057, -0.0010687482, -0.065087231, 4.0724145, 3.3129991, -0.024179475]]
        omega = []
    elif distr_case == 3:
        Z = \
[[0.92624665, 0.45003419, 1.3198471, 0.80502735, 0.19884378, 0.9822903, 0.72215214],
 [0.92802478, 0.32279442, 1.465807, 0.19966078, 0.80088564, 0.41636724, 1.1255588],
 [0.85577721, 0.35917522, 1.7250669, 0.33495094, 1.042279, 0.55410351, 1.0912914],
 [0.57492908, 0.97391634, 1.7169863, 0.69326334, 0.22491319, 0.52106726, 0.12154811],
 [0.51790385, 0.92475001, 1.3405784, 0.97366586, 0.52101259, 0.67950832, 0.99466291]]
        omega = \
[[79.192298, 78.034138, 81.126084, 82.29634, 73.307385, 82.861867, 79.149279],
 [78.535481, 74.973533, 79.981103, 75.657009, 83.753253, 77.446055, 82.7547],
 [76.740366, 73.973894, 77.965487, 74.312075, 82.07981, 78.444247, 80.3828],
 [76.03236, 81.887229, 82.059597, 80.717699, 74.73542, 77.017019, 72.111641],
 [73.968662, 80.254828, 78.883159, 85.259888, 77.127616, 80.020205, 77.723596]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
    )


def test_x_ndim_3():
    rng = np.random.default_rng(seed=SEED)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    _, ns = params['Z'].shape
    ny, nb = params['Xeff'].shape
    params['Xeff'] = rng.random(size=(ns, ny, nb))

    Z = \
[[0, 3.5320043, 1.753429, 1, 0.13864651, -0.018645369, 1],
 [0, 2.6807404, 1.4385575, 0, 0.87347456, -0.0077176881, 1],
 [1, -0.029152124, 1.7213341, 0, 0.64247312, 2.9572088, 1],
 [0, 2.4097049, 1.5781643, 0, 0.14337646, 3.1950511, 1],
 [0, -0.052660417, 1.6337807, 0, 0.29615809, 4.0595912, 0]]
    omega = \
[[82.695649, 73.524467],
 [75.731191, 80.315436],
 [77.777866, 79.681864],
 [81.555963, 73.141613],
 [80.272408, 75.35137]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
    )


@pytest.mark.parametrize("preupdate", [True, False])
@pytest.mark.parametrize("marginalize", [True, False])
def test_poisson_flags(preupdate, marginalize):
    rng = np.random.default_rng(seed=SEED)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    if preupdate and marginalize:
        Z = \
[[0, 2.5361722, -12.635516, 1, -277.10556, -0.027216023, 1],
 [0, 3.0099937, -11.727865, 0, -179.72621, -0.0065145159, 1],
 [1, -0.030795585, -10.442803, 0, -197.24705, 3.001938, 1],
 [0, 2.7197177, -11.25108, 0, -185.9209, 3.9297946, 1],
 [0, -0.051366734, -11.915486, 0, -239.02875, 4.3054389, 0]]
        iD = \
[[10.682681, 2.9806676, 14.21764, 1.3627914, 0.87460102, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 14.586367, 1.3627914, 1.0541631, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 15.163064, 1.3627914, 1.0166099, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 14.800281, 1.3627914, 1.0401549, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 14.51482, 1.3627914, 0.93657754, 1.9318932, 5.3560603]]
    elif not preupdate and marginalize:
        Z = \
[[0, 2.5361722, 0.86148798, 1, 0.10729794, -0.027216023, 1],
 [0, 3.0099937, 0.31205812, 0, 0.68230191, -0.0065145159, 1],
 [1, -0.030795585, 0.47919116, 0, 0.63280165, 3.001938, 1],
 [0, 2.7197177, 0.77699572, 0, 0.0785376, 3.9297946, 1],
 [0, -0.051366734, 0.67896492, 0, 0.27881144, 4.3054389, 0]]
        iD = \
[[10.682681, 2.9806676, 23.07276, 1.3627914, 1.697929, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 22.49556, 1.3627914, 1.7012509, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 22.672783, 1.3627914, 1.7009644, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 22.98315, 1.3627914, 1.6977237, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 22.88005, 1.3627914, 1.6988802, 1.9318932, 5.3560603]]
    elif preupdate and not marginalize:
        Z = \
[[0, 2.5361722, -4.0607028, 1, -137.24598, -0.027216023, 1],
 [0, 3.0099937, -3.143395, 0, -106.93627, -0.0065145159, 1],
 [1, -0.030795585, -2.3560851, 0, -113.78047, 3.001938, 1],
 [0, 2.7197177, -2.949795, 0, -110.79138, 3.9297946, 1],
 [0, -0.051366734, -3.3001579, 0, -127.19099, 4.3054389, 0]]
        iD = \
[[10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 5.3560603],
 [10.682681, 2.9806676, 32.001416, 1.3627914, 1.7380668, 1.9318932, 5.3560603]]
    elif not preupdate and not marginalize:
        Z, iD, omega = default_reference_values()

    if preupdate:
        if marginalize:
            omega = \
[[25.584253, 1.7604809],
 [26.803507, 2.6790408],
 [28.817516, 2.4491219],
 [27.534806, 2.5903826],
 [26.562906, 2.031012]]
        else:
            omega = \
[[25.584253, 1.7604809],
 [26.803507, 2.6790408],
 [28.817516, 2.4491219],
 [27.534806, 2.5903826],
 [26.562906, 2.031012]]
    else:
        if marginalize:
            omega = \
[[82.695649, 73.524467],
 [75.731191, 80.315436],
 [77.777866, 79.681864],
 [81.555963, 73.141613],
 [80.272408, 75.35137]]
        else:
            omega = \
[[82.695649, 73.524467],
 [75.731191, 80.315436],
 [77.777866, 79.681864],
 [81.555963, 73.141613],
 [80.272408, 75.35137]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
        poisson_preupdate_z=preupdate,
        poisson_marginalize_z=marginalize,
    )
