import pytest
import numpy as np
import tensorflow as tf

from pytest import approx

from hmsc.updaters.updateZ import updateZ


def input_values(*, distr_case, x_ndim, seed, dtype):
    assert distr_case in (0, 1, 2, 3)
    assert x_ndim in (2, 3)
    # Add test Y including nans

    rng = np.random.default_rng(seed=seed)
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
         [1, 1]]
    )

    if distr_case in (1, 2, 3):
        distr[:, 0] = distr_case

    Pi = np.array(
        [[0, 2],
         [1, 2],
         [2, 2],
         [0, 0],
         [1, 0]]
    )

    ns3 = np.sum(distr[:, 0] == 3)

    params = dict(
        Z=tf.convert_to_tensor(rng.random(size=(ny, ns)), dtype=dtype),
        Beta=tf.convert_to_tensor(rng.random(size=(nb, ns)), dtype=dtype),
        sigma=tf.convert_to_tensor(rng.random(size=ns), dtype=dtype),
        Xeff=tf.convert_to_tensor(rng.random(size=(ny, nb)), dtype=dtype),
        poisson_omega=tf.convert_to_tensor(rng.random(size=(ny, ns3)), dtype=dtype),
    )
    data = dict(
        Y=tf.convert_to_tensor(rng.integers(2, size=(ny, ns)), dtype=dtype),
        Pi=tf.convert_to_tensor(Pi, dtype=tf.int32),
        distr=tf.convert_to_tensor(distr, dtype=tf.int32),
    )

    assert data['distr'].shape == (ns, 2)
    assert data['Pi'].shape == (ny, nr)

    EtaList = [
        tf.convert_to_tensor(rng.random(ne * 2).reshape(ne, 2), dtype=dtype),
        tf.convert_to_tensor(rng.random(ne * 2).reshape(ne, 2), dtype=dtype),
    ]
    LambdaList = [
        tf.convert_to_tensor(rng.random(2 * ns).reshape(2, ns), dtype=dtype),
        tf.convert_to_tensor(rng.random(2 * ns).reshape(2, ns), dtype=dtype),
    ]
    rLHyperparams = [
        dict(xDim=0),
        dict(xDim=0),
    ]
    assert len(EtaList) == len(LambdaList) == len(rLHyperparams) == nr

    if x_ndim == 3:
        params['Xeff'] = tf.convert_to_tensor(rng.random(size=(ns, ny, nb)), dtype=dtype)

    params["Eta"] = EtaList
    params["Lambda"] = LambdaList

    return params, data, rLHyperparams


def run_test(distr_case=0, x_ndim=2, tnlib='tf',
             poisson_preupdate_z=False,
             poisson_marginalize_z=False,
             ref_values=None,
             seed=42, dtype=np.float64):
    input_kwargs = dict(
        distr_case=distr_case,
        x_ndim=x_ndim,
        seed=seed,
    )

    # Prepare input matrices
    params, data, rLHyperparams = input_values(**input_kwargs, dtype=dtype)

    # Calculate
    Z, iD, omega = updateZ(
        params, data, rLHyperparams,
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
    assert Z == approx(ref_Z)
    assert iD == approx(ref_iD)
    if omega.size > 0 or ref_omega.size > 0:
        assert omega == approx(ref_omega)


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
    run_test(ref_values=default_reference_values())


@pytest.mark.parametrize("tnlib", ['tf', 'tfd'])
def test_tnlib(tnlib):
    Z, iD, omega = default_reference_values()

    if tnlib == 'tfd':
        Z = \
[[ 0.        ,  2.17921306,  1.40754338,  1.        ,  0.13129104, -0.05694204,  1.        ],
 [ 0.        ,  2.7920181 ,  1.35676311,  0.        ,  0.89406079, -0.13361183,  1.        ],
 [ 1.        , -0.0166318 ,  1.77882987,  0.        ,  0.6501287 ,  4.70081388,  1.        ],
 [ 0.        ,  4.68685993,  1.68541477,  0.        ,  0.11587453,  4.4379023 ,  1.        ],
 [ 0.        , -0.1906678 ,  1.44970694,  0.        ,  0.32676159,  2.69563436,  0.        ]]

    run_test(
        tnlib=tnlib,
        ref_values=(Z, iD, omega),
    )


@pytest.mark.parametrize("distr_case", [0, 1, 2, 3])
def test_distr_case(distr_case):
    Z, iD, omega = default_reference_values()

    if distr_case == 1:
        Z = \
[[0., 0., 1., 0., 0., 0., 1.],
 [1., 0., 1., 0., 1., 1., 1.],
 [0., 1., 0., 1., 0., 0., 0.],
 [1., 0., 1., 0., 0., 1., 0.],
 [1., 1., 0., 0., 0., 1., 1.]]
    elif distr_case == 2:
        Z = \
[[-2.16731687e-02, -7.26993294e-02,  2.97247856e+00, -2.56566343e-01, -2.82493186e-02, -2.15090045e-01,  1.82051501e+00],
 [ 3.44111121e+00, -8.08565606e-02,  3.02281863e+00, -1.12269164e-01,  3.85723288e+00,  3.10194142e+00,  1.86781389e+00],
 [-1.97487684e-02,  3.19087864e+00, -7.81100111e-03,  3.84275918e+00, -1.37848714e-01, -1.85120295e-02, -9.35641902e-02],
 [ 2.98396344e+00, -2.44607084e-03,  4.53406253e+00, -2.99807014e-01, -1.35385546e-01,  2.85929925e+00, -1.04299383e-01],
 [ 3.86621611e+00,  4.19202896e+00, -1.22456847e-02, -1.25848590e-01, -1.53532906e-01,  2.53178412e+00,  2.94148879e+00]]
    elif distr_case == 3:
        Z = \
[[1.0157138 , 0.4100475 , 1.3975398 , 0.45718054, 0.14703785, 0.89889734, 0.77702612],
 [0.93864013, 0.16124331, 1.27282378, 0.28755889, 0.83547283, 0.71256083, 0.98335849],
 [0.93121183, 0.47809308, 1.93095805, 0.29002695, 1.18488449, 0.65230171, 0.92880115],
 [0.42809156, 1.1224092 , 1.78635417, 0.70666318, 0.18722986, 0.63402328, 0.11470677],
 [0.5574293 , 0.85028616, 1.43130618, 0.87880138, 0.44498424, 0.25420793, 0.84656094]]

    if distr_case in [1, 2]:
        omega = \
[]
    elif distr_case == 3:
        omega = \
[[81.70836675, 75.63906919, 81.970256  , 79.53973508, 73.76147381, 84.78603609, 80.57772435],
 [81.23612849, 73.43414908, 76.19355286, 74.56216626, 80.7187692 , 78.19513098, 83.49581729],
 [77.96440037, 75.55425842, 79.08140616, 74.00632178, 84.13732389, 78.59819836, 83.34528905],
 [75.25213129, 84.97194624, 82.77510123, 80.53394158, 73.69114804, 77.7371134 , 72.00083597],
 [72.77825139, 82.103995  , 79.33289536, 82.3816733 , 76.11819717, 75.78188646, 81.89521818]]

    run_test(
        distr_case=distr_case,
        ref_values=(Z, iD, omega),
    )


@pytest.mark.parametrize("x_ndim", [2, 3])
def test_x_ndim(x_ndim):
    Z, iD, omega = default_reference_values()

    if x_ndim == 3:
        Z = \
[[ 0.        ,  3.53200434,  1.75342901,  1.        ,  0.13864651, -0.0499102 ,  1.        ],
 [ 0.        ,  2.64775599,  1.43855754,  0.        ,  0.87347456, -0.20061524,  1.        ],
 [ 1.        , -0.01429325,  1.72133414,  0.        ,  0.64247312,  3.83834427,  1.        ],
 [ 0.        ,  2.16941234,  1.57816434,  0.        ,  0.14337646,  2.3323683 ,  1.        ],
 [ 0.        , -0.06552546,  1.63378067,  0.        ,  0.29615809,  2.05435849,  0.        ]]

    run_test(
        x_ndim=x_ndim,
        ref_values=(Z, iD, omega),
    )


@pytest.mark.parametrize("preupdate", [True, False])
@pytest.mark.parametrize("marginalize", [True, False])
def test_poisson_flags(preupdate, marginalize):
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
        poisson_preupdate_z=preupdate,
        poisson_marginalize_z=marginalize,
        ref_values=(Z, iD, omega),
    )
