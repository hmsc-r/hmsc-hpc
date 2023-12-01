import pytest
import numpy as np
import tensorflow as tf

from numpy import nan
from pytest import approx

from hmsc.updaters.updateZ import updateZ


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
             seed=42,
             ):

    # Convert inputs to tf
    params, data, rLHyperparams = map(convert_to_tf, input_values)
    dtype = data['Y'].dtype
    assert dtype == tf.float64

    # Calculate
    Z, iD, omega = updateZ(
        *input_values,
        poisson_preupdate_z=poisson_preupdate_z,
        poisson_marginalize_z=poisson_marginalize_z,
        truncated_normal_library=tnlib,
        seed=seed,
        dtype=dtype,
        )
    Z = Z.numpy()
    iD = iD.numpy()
    omega = omega.numpy()

    assert Z.shape == params['Z'].shape
    assert iD.shape == params['Z'].shape
    assert omega.shape == params['poisson_omega'].shape

    # Print values
    for name, array in (('Z', Z),
                        ('iD', iD),
                        ('omega', omega)):
        print(f'{name} = \\')
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

    return params, data, rLHyperparams


def default_reference_values():
    Z = \
[[ 0.        ,  2.53617218,  1.40754338,  1.        ,  0.13129104, -0.07285226,  1.        ],
 [ 0.        ,  2.97700928,  1.35676311,  0.        ,  0.89406079, -0.16933972,  1.        ],
 [ 1.        , -0.01509904,  1.77882987,  0.        ,  0.6501287 ,  3.88307341,  1.        ],
 [ 0.        ,  2.47942512,  1.68541477,  0.        ,  0.11587453,  3.06711182,  1.        ],
 [ 0.        , -0.06391573,  1.44970694,  0.        ,  0.32676159,  2.30020615,  0.        ]]
    iD = \
[[10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031]]
    omega = \
[[82.69564925, 73.52446685],
 [75.73119083, 80.31543575],
 [77.77786621, 79.68186375],
 [81.55596303, 73.14161346],
 [80.27240777, 75.35136998]]
    return Z, iD, omega


def test_defaults():
    seed = 42
    rng = np.random.default_rng(seed=seed)
    run_test(
        default_input_values(rng),
        default_reference_values(),
        seed=seed,
    )


@pytest.mark.parametrize("tnlib", ['tf', 'tfd'])
def test_tnlib(tnlib):
    seed = 42
    rng = np.random.default_rng(seed=seed)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    if tnlib == 'tfd':
        Z = \
[[ 0.        ,  2.17921306,  1.40754338,  1.        ,  0.13129104, -0.05694204,  1.        ],
 [ 0.        ,  2.7920181 ,  1.35676311,  0.        ,  0.89406079, -0.13361183,  1.        ],
 [ 1.        , -0.0166318 ,  1.77882987,  0.        ,  0.6501287 ,  4.70081388,  1.        ],
 [ 0.        ,  4.68685993,  1.68541477,  0.        ,  0.11587453,  4.4379023 ,  1.        ],
 [ 0.        , -0.1906678 ,  1.44970694,  0.        ,  0.32676159,  2.69563436,  0.        ]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
        tnlib=tnlib,
        seed=seed,
    )


def test_y_nan():
    seed = 42
    rng = np.random.default_rng(seed=seed)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    Y = rng.integers(3, size=data['Y'].shape).astype(np.float64)
    Y[Y == 2] = nan
    data['Y'] = Y

    Z = \
[[ 2.6390101 ,  2.53617218,  1.40754338,  1.        ,  0.14450654,  3.479958  ,  1.        ],
 [ 4.25794634,  2.97700928,         nan,  1.        ,  0.89406079,  3.95376739,  0.        ],
 [ 1.        ,  3.35260822,  1.78706598,  0.        ,  0.66243985,  3.88307341,  4.65326605],
 [ 0.        ,  2.47942512,  1.6935875 ,  0.        ,  0.1025117 , -0.02948891,  3.19747972],
 [ 3.36759254,  2.97494929,         nan,  3.19999282,  0.31382702,  2.30020615,  0.        ]]
    iD = \
[[ 0.        ,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  0.        ,  5.35606031],
 [ 0.        ,  0.        ,  0.        ,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  0.        , 32.00141647,  1.36279137,  1.73806681,  0.        ,  0.        ],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  1.73806681,  0.        ,  5.35606031]]
    omega = \
[[82.69564925, 73.59783994],
 [        nan, 80.31543575],
 [77.85596886, 79.76263482],
 [81.63810387, 73.06792919],
 [        nan, 75.27585429]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
        seed=seed,
    )


@pytest.mark.parametrize("distr_case", [1, 2, 3])
def test_distr_case(distr_case):
    seed = 42
    rng = np.random.default_rng(seed=seed)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    if distr_case in [1, 2, 3]:
        data['distr'][:, 0] = distr_case
        ns3 = np.sum(data['distr'][:, 0] == 3)
        ny = params['poisson_omega'].shape[0]
        params['poisson_omega'] = rng.random(size=(ny, ns3))

    if distr_case == 1:
        Z = \
[[0., 1., 0., 1., 0., 0., 1.],
 [0., 1., 1., 0., 0., 0., 1.],
 [1., 0., 0., 0., 0., 1., 1.],
 [0., 1., 0., 0., 1., 1., 1.],
 [0., 0., 0., 0., 1., 1., 0.]]
    elif distr_case == 2:
        Z = \
[[-2.08403799e-02,  2.75149757e+00, -1.14963779e-03,  2.95664772e+00, -2.90638976e-02, -1.94257235e-01,  2.48641501e+00],
 [-5.06593421e-03,  2.53575181e+00,  3.50824437e+00, -1.25499296e-01, -6.67515314e-02, -1.15103214e-01,  2.49429846e+00],
 [ 3.31221818e+00, -9.73045797e-02, -7.28061119e-03, -1.04481659e-01, -1.39073459e-01,  3.53843102e+00,  3.36584739e+00],
 [-1.31982845e-02,  3.35420016e+00, -5.91590284e-03, -3.77819950e-01,  2.86462315e+00,  3.15219413e+00,  2.39663354e+00],
 [-1.61291747e-02, -2.21578580e-01, -1.14318636e-02, -1.41671552e-01,  4.09264949e+00,  2.70324074e+00, -5.99368596e-02]]
    elif distr_case == 3:
        Z = \
[[1.08068846, 0.40776981, 1.27446972, 0.44300929, 0.13549434, 0.8884085 , 0.77713921],
 [1.0086427 , 0.16956534, 1.31792991, 0.2774678 , 0.82880334, 0.72597065, 1.0033521 ],
 [0.97619397, 0.47797544, 1.94174992, 0.27476818, 1.17503413, 0.64529485, 0.93802499],
 [0.45121264, 1.09131311, 1.56435178, 0.69240529, 0.1806462 , 0.62468013, 0.07027166],
 [0.5864055 , 0.81608219, 1.3552109 , 0.86939957, 0.42999498, 0.25728539, 0.83878168]]

    if distr_case in [1, 2]:
        omega = \
[]
    elif distr_case == 3:
        omega = \
[[81.70836675, 75.63906919, 81.88817839, 79.53973508, 73.68801896, 84.70176497, 80.65851229],
 [81.23612849, 73.43414908, 76.19355286, 74.56216626, 80.63686202, 78.19513098, 83.49581729],
 [77.96440037, 75.55425842, 79.08140616, 74.00632178, 84.05436711, 78.59819836, 83.42744085],
 [75.25213129, 84.97194624, 82.77510123, 80.53394158, 73.76514384, 77.7371134 , 72.00083597],
 [72.77825139, 82.02297995, 79.25287452, 82.3816733 , 76.11819717, 75.78188646, 81.89521818]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
        seed=seed,
    )


def test_x_ndim_3():
    seed = 42
    rng = np.random.default_rng(seed=seed)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    _, ns = params['Z'].shape
    ny, nb = params['Xeff'].shape
    params['Xeff'] = rng.random(size=(ns, ny, nb))

    Z = \
[[ 0.        ,  3.53200434,  1.75342901,  1.        ,  0.13864651, -0.0499102 ,  1.        ],
 [ 0.        ,  2.64775599,  1.43855754,  0.        ,  0.87347456, -0.20061524,  1.        ],
 [ 1.        , -0.01429325,  1.72133414,  0.        ,  0.64247312,  3.83834427,  1.        ],
 [ 0.        ,  2.16941234,  1.57816434,  0.        ,  0.14337646,  2.3323683 ,  1.        ],
 [ 0.        , -0.06552546,  1.63378067,  0.        ,  0.29615809,  2.05435849,  0.        ]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
        seed=seed,
    )


@pytest.mark.parametrize("preupdate", [True, False])
@pytest.mark.parametrize("marginalize", [True, False])
def test_poisson_flags(preupdate, marginalize):
    seed = 42
    rng = np.random.default_rng(seed=seed)
    params, data, rLHyperparams = default_input_values(rng)
    Z, iD, omega = default_reference_values()

    if preupdate and marginalize:
        Z = \
[[ 0.00000000e+00,  2.53617218e+00, -1.26355164e+01,  1.00000000e+00, -2.77105558e+02, -7.28522575e-02,  1.00000000e+00],
 [ 0.00000000e+00,  2.97700928e+00, -1.17278655e+01,  0.00000000e+00, -1.79726207e+02, -1.69339725e-01,  1.00000000e+00],
 [ 1.00000000e+00, -1.50990382e-02, -1.04428030e+01,  0.00000000e+00, -1.97247049e+02,  3.88307341e+00,  1.00000000e+00],
 [ 0.00000000e+00,  2.47942512e+00, -1.12510798e+01,  0.00000000e+00, -1.85920900e+02,  3.06711182e+00,  1.00000000e+00],
 [ 0.00000000e+00, -6.39157325e-02, -1.19154864e+01,  0.00000000e+00, -2.39028750e+02,  2.30020615e+00,  0.00000000e+00]]
    elif not preupdate and marginalize:
        Z = \
[[ 0.        ,  2.53617218,  0.86148798,  1.        ,  0.10729794, -0.07285226,  1.        ],
 [ 0.        ,  2.97700928,  0.31205812,  0.        ,  0.68230191, -0.16933972,  1.        ],
 [ 1.        , -0.01509904,  0.47919116,  0.        ,  0.63280165,  3.88307341,  1.        ],
 [ 0.        ,  2.47942512,  0.77699572,  0.        ,  0.0785376 ,  3.06711182,  1.        ],
 [ 0.        , -0.06391573,  0.67896492,  0.        ,  0.27881144,  2.30020615,  0.        ]]
    elif preupdate and not marginalize:
        Z = \
[[ 0.00000000e+00,  2.53617218e+00, -4.06070277e+00,  1.00000000e+00, -1.37245984e+02, -7.28522575e-02,  1.00000000e+00],
 [ 0.00000000e+00,  2.97700928e+00, -3.14339495e+00,  0.00000000e+00, -1.06936268e+02, -1.69339725e-01,  1.00000000e+00],
 [ 1.00000000e+00, -1.50990382e-02, -2.35608505e+00,  0.00000000e+00, -1.13780474e+02,  3.88307341e+00,  1.00000000e+00],
 [ 0.00000000e+00,  2.47942512e+00, -2.94979503e+00,  0.00000000e+00, -1.10791375e+02,  3.06711182e+00,  1.00000000e+00],
 [ 0.00000000e+00, -6.39157325e-02, -3.30015786e+00,  0.00000000e+00, -1.27190998e+02,  2.30020615e+00,  0.00000000e+00]]

    if preupdate and marginalize:
        iD = \
[[10.68268147,  2.98066761, 14.21764018,  1.36279137,  0.87460102,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 14.58636705,  1.36279137,  1.05416311,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 15.16306357,  1.36279137,  1.0166099 ,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 14.80028053,  1.36279137,  1.04015494,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 14.51482035,  1.36279137,  0.93657754,  1.93189325,  5.35606031]]
    elif not preupdate and marginalize:
        iD = \
[[10.68268147,  2.98066761, 23.07276037,  1.36279137,  1.69792896,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 22.49556043,  1.36279137,  1.70125087,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 22.67278332,  1.36279137,  1.70096439,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 22.98315046,  1.36279137,  1.69772374,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 22.88005036,  1.36279137,  1.69888016,  1.93189325,  5.35606031]]

    if preupdate:
        omega = \
[[25.58425259,  1.76048085],
 [26.80350743,  2.67904081],
 [28.81751649,  2.44912189],
 [27.53480601,  2.59038264],
 [26.5629061 ,  2.03101203]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
        poisson_preupdate_z=preupdate,
        poisson_marginalize_z=marginalize,
        seed=seed,
    )
