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
[[ 0.        ,  2.75995407,  1.46132533,  1.        ,  0.02141102, -0.04065752,  1.        ],
 [ 0.        ,  2.63407915,  1.48657442,  0.        ,  0.9295774 , -0.02897899,  1.        ],
 [ 1.        , -0.05083014,  1.55560853,  0.        ,  0.56173484,  4.55706943,  1.        ],
 [ 0.        ,  2.23395789,  1.72425458,  0.        ,  0.24091763,  2.58591124,  1.        ],
 [ 0.        , -0.17355643,  1.65879016,  0.        ,  0.46961335,  4.22843305,  0.        ]]
    iD = \
[[10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  5.35606031]]
    omega = \
[[81.86756908, 72.81064334],
 [75.56812802, 82.39418253],
 [76.11790758, 80.1500393 ],
 [82.49025145, 74.07839893],
 [82.86859862, 77.14692351]]
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
[[ 0.        ,  2.20355172,  1.46132533,  1.        ,  0.02141102, -0.25247957,  1.        ],
 [ 0.        ,  4.03489762,  1.48657442,  0.        ,  0.9295774 , -0.04786586,  1.        ],
 [ 1.        , -0.01506489,  1.55560853,  0.        ,  0.56173484,  4.05305873,  1.        ],
 [ 0.        ,  2.80027739,  1.72425458,  0.        ,  0.24091763,  2.95680797,  1.        ],
 [ 0.        , -0.13374268,  1.65879016,  0.        ,  0.46961335,  4.17948469,  0.        ]]

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
[[ 3.58527222,  2.75995407,  1.46132533,  1.        ,  0.03480601,  3.18984633,  1.        ],
 [ 3.49418004,  2.63407915,         nan,  1.        ,  0.9295774 ,  3.40594864,  0.        ],
 [ 1.        ,  2.986176  ,  1.56403975,  0.        ,  0.5740196 ,  4.55706943,  4.10783732],
 [ 0.        ,  2.23395789,  1.73235256,  0.        ,  0.22778023, -0.11755489,  3.40126362],
 [ 3.80187667,  2.42038501,         nan,  1.31432367,  0.45704017,  4.22843305,  0.        ]]
    iD = \
[[ 0.        ,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  0.        ,  5.35606031],
 [ 0.        ,  0.        ,  0.        ,  1.36279137,  1.73806681,  1.93189325,  5.35606031],
 [10.68268147,  0.        , 32.00141647,  1.36279137,  1.73806681,  0.        ,  0.        ],
 [10.68268147,  2.98066761, 32.00141647,  1.36279137,  1.73806681,  1.93189325,  0.        ],
 [ 0.        ,  0.        ,  0.        ,  0.        ,  1.73806681,  0.        ,  5.35606031]]
    omega = \
[[81.86756908, 72.88365961],
 [        nan, 82.39418253],
 [76.19518046, 80.2310444 ],
 [82.57285932, 74.00424662],
 [        nan, 77.07051072]]

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
[[-0.00932817,  2.51793675, -0.00962837,  2.3043982 , -0.09784212, -0.07316939,  2.30329848],
 [-0.02019489,  1.98118754,  3.98201169, -0.30505784, -0.11742882, -0.1673493 ,  2.9924404 ],
 [ 3.55660506, -0.20311671, -0.00525189, -0.33395153, -0.1046308 ,  2.92314937,  4.51508742],
 [-0.02520902,  3.7431588 , -0.01465407, -0.01307165,  3.89120239,  3.56748974,  3.38585414],
 [-0.02597788, -0.02477555, -0.00625335, -0.09287741,  4.560472  ,  3.54324023, -0.06668396]]
    elif distr_case == 3:
        Z = \
[[ 1.08103897,  0.37993643,  1.47041143,  0.69601159, -0.08822384,  0.69146803,  0.86472671],
 [ 1.11050164,  0.41749746,  1.4509428 ,  0.42697021,  0.98806989,  0.86438855,  0.85050444],
 [ 0.55970649,  0.46331567,  1.55808719,  0.34136444,  0.68821819,  0.88021107,  1.12753029],
 [ 0.82030682,  1.03358407,  1.49131052,  0.92445352,  0.23852094,  0.41601653,  0.09681335],
 [ 0.76083148,  0.56711784,  1.56857781,  0.97783873,  0.14586467,  0.5390802 ,  0.52696148]]

    if distr_case in [1, 2]:
        omega = \
[]
    elif distr_case == 3:
        omega = \
[[80.77120024, 76.68751094, 80.40521585, 79.52522313, 71.13358233, 82.07061724, 80.85085594],
 [80.98122479, 75.47164617, 78.61788911, 77.67120823, 84.37740876, 79.86271159, 81.08145304],
 [75.41946522, 75.49403298, 79.46724498, 73.73826995, 78.96416624, 81.18048187, 82.22697569],
 [76.37707526, 84.43640824, 79.69663393, 83.50548746, 75.36752544, 77.29429568, 72.34069915],
 [73.80687095, 77.85947481, 80.41778216, 84.76475036, 75.66752127, 76.58853369, 76.5842063 ]]

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
[[ 0.        ,  3.75578623,  1.80972632,  1.        ,  0.02883692, -0.02785398,  1.        ],
 [ 0.        ,  2.30482585,  1.56849284,  0.        ,  0.90949982, -0.03433115,  1.        ],
 [ 1.        , -0.0481175 ,  1.49723007,  0.        ,  0.55412304,  4.51234028,  1.        ],
 [ 0.        ,  1.92394511,  1.61787935,  0.        ,  0.26807974,  1.85116773,  1.        ],
 [ 0.        , -0.17792748,  1.83870361,  0.        ,  0.43970644,  3.98258539,  0.        ]]

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
[[ 0.00000000e+00,  2.75995407e+00, -1.25377157e+01,  1.00000000e+00, -2.77534451e+02, -4.06575192e-02,  1.00000000e+00],
 [ 0.00000000e+00,  2.63407915e+00, -1.14777075e+01,  0.00000000e+00, -1.78888445e+02, -2.89789882e-02,  1.00000000e+00],
 [ 1.00000000e+00, -5.08301390e-02, -1.09308255e+01,  0.00000000e+00, -1.97044699e+02,  4.55706943e+00,  1.00000000e+00],
 [ 0.00000000e+00,  2.23395789e+00, -1.11493435e+01,  0.00000000e+00, -1.85468574e+02,  2.58591124e+00,  1.00000000e+00],
 [ 0.00000000e+00, -1.73556426e-01, -1.14199601e+01,  0.00000000e+00, -2.38074954e+02,  4.22843305e+00,  0.00000000e+00]]
    elif not preupdate and marginalize:
        Z = \
[[ 0.        ,  2.75995407,  0.80033074,  1.        ,  0.04062738, -0.04065752,  1.        ],
 [ 0.        ,  2.63407915,  0.29782576,  0.        ,  0.83936568, -0.02897899,  1.        ],
 [ 1.        , -0.05083014,  0.33899878,  0.        ,  0.66945516,  4.55706943,  1.        ],
 [ 0.        ,  2.23395789,  0.84643299,  0.        ,  0.16489896,  2.58591124,  1.        ],
 [ 0.        , -0.17355643,  0.87410673,  0.        ,  0.43309657,  4.22843305,  0.        ]]
    elif preupdate and not marginalize:
        Z = \
[[ 0.00000000e+00,  2.75995407e+00, -3.98170344e+00,  1.00000000e+00, -1.37797878e+02, -4.06575192e-02,  1.00000000e+00],
 [ 0.00000000e+00,  2.63407915e+00, -3.43097796e+00,  0.00000000e+00, -1.07840576e+02, -2.89789882e-02,  1.00000000e+00],
 [ 1.00000000e+00, -5.08301390e-02, -2.54554635e+00,  0.00000000e+00, -1.14274773e+02,  4.55706943e+00,  1.00000000e+00],
 [ 0.00000000e+00,  2.23395789e+00, -2.78898168e+00,  0.00000000e+00, -1.09959689e+02,  2.58591124e+00,  1.00000000e+00],
 [ 0.00000000e+00, -1.73556426e-01, -3.18235135e+00,  0.00000000e+00, -1.28265981e+02,  4.22843305e+00,  0.00000000e+00]]

    if preupdate and marginalize:
        iD = \
[[10.68268147,  2.98066761, 14.25728957,  1.36279137,  0.87394536,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 14.69370601,  1.36279137,  1.05602835,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 14.94192557,  1.36279137,  1.01702832,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 14.84498549,  1.36279137,  1.04113561,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 14.72666265,  1.36279137,  0.93825551,  1.93189325,  5.35606031]]
    elif not preupdate and marginalize:
        iD = \
[[10.68268147,  2.98066761, 23.00782923,  1.36279137,  1.69754463,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 22.48115067,  1.36279137,  1.70216053,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 22.52956058,  1.36279137,  1.70117651,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 23.05674238,  1.36279137,  1.69822221,  1.93189325,  5.35606031],
 [10.68268147,  2.98066761, 23.08620343,  1.36279137,  1.69977212,  1.93189325,  5.35606031]]

    if preupdate:
        omega = \
[[25.71292822,  1.75782633],
 [27.16820383,  2.69112069],
 [28.02913558,  2.45155177],
 [27.68994108,  2.59647329],
 [27.28108716,  2.03891941]]

    run_test(
        (params, data, rLHyperparams),
        (Z, iD, omega),
        poisson_preupdate_z=preupdate,
        poisson_marginalize_z=marginalize,
        seed=seed,
    )
