import numpy as np


def test_igm():
    """
    Test IGM implementation
    """

    try:
        from ..resample_numba import compute_igm
    except ImportError:
        return None

    wobs = np.linspace(0.2, 1.0, 32) * 1.0e4

    igmz = compute_igm(3.1, wobs, scale_tau=1.0)
    _tau = np.log(igmz)

    assert np.allclose(
        igmz[:6],
        np.array(
            [
                0.09095587,
                0.09023704,
                0.09558526,
                0.1098157,
                0.13983981,
                0.20491541,
            ]
        ),
        rtol=1.0e-3,
    )

    # Scaled tau
    igm2 = compute_igm(3.1, wobs, scale_tau=2.0)
    _tau2 = np.log(igm2)
    assert np.allclose(_tau * 2, _tau2)

    # Low-z, no IGM
    igmz = compute_igm(0.1, wobs, scale_tau=1.0)
    assert np.allclose(igmz, 1.0)

    # High-z, completely absorbed
    igmz = compute_igm(10, wobs, scale_tau=1.0)
    assert np.allclose(igmz, 0.0, rtol=1.0e-6)
