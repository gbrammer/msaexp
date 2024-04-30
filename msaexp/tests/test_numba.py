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


def test_prf():
    """
    Test pixel-integrated Gaussian
    """

    try:
        from ..resample_numba import pixel_integrated_gaussian_numba
    except ImportError:
        return None

    #########
    # 1D
    x = np.arange(-64, 64.1, 0.5, dtype=float)
    prf = pixel_integrated_gaussian_numba(x, 0.0, 1.0, dx=None)
    assert np.allclose(prf.sum() * 0.5, 1.0)

    x = np.arange(-64, 65, dtype=float)
    prf = pixel_integrated_gaussian_numba(x, 0.0, 1.0, dx=None)
    assert np.allclose(prf.sum(), 1.0)

    x = np.arange(-64, 65, dtype=float)
    prf = pixel_integrated_gaussian_numba(x, 0.0, 1.0, dx=1.0)
    assert np.allclose(prf.sum(), 1.0)

    for mu in np.linspace(-1.0, 1.0, 32):
        for sigma in [0.3, 1.0, 2.0, 5.0]:
            prf = pixel_integrated_gaussian_numba(x, mu, sigma, dx=None)
            assert np.allclose(prf.sum(), 1.0)

    ##########
    # 2D
    sh = (32, 32)
    yp, xp = np.indices(sh)
    ytrace = np.linspace(15, 16, sh[1])

    # Vary mu
    x = yp.flatten() - 15.0
    mu = (yp * 0.0 + ytrace).flatten() - 15
    prf1 = pixel_integrated_gaussian_numba(x, mu, 1.0, dx=1.0)
    assert np.allclose(prf1.reshape(sh).sum(axis=0), 1.0)

    # Vary sample
    x = (yp - ytrace).flatten()
    mu = 0.0
    prf2 = pixel_integrated_gaussian_numba(x, mu, 1.0, dx=1.0)
    assert np.allclose(prf2.reshape(sh).sum(axis=0), 1.0)

    # Should be the same
    assert np.allclose(prf1, prf2, rtol=1.0e-6)

    # Vary mu
    sx = np.linspace(0.5, 2, sh[1])
    sigma = (yp * 0.0 + sx).flatten()
    mu = 0.0
    prf_mu = pixel_integrated_gaussian_numba(x, mu, sigma, dx=1.0)

    assert np.allclose(prf_mu.reshape(sh).sum(axis=0), 1.0)


def test_prf_line():
    """
    Test pixel-integrated emission line model
    """
    try:
        from ..resample_numba import sample_gaussian_line_numba
    except ImportError:
        return None

    N = 80
    rtol = 1.0e-4

    dx = 0.2 / N

    spec_wobs = np.linspace(1.4, 1.6, N + 1)
    spec_R_fwhm = np.full_like(spec_wobs, 500)
    line_um = 1.5
    line_flux = 1.0
    velocity_sigma = 300.0

    line = sample_gaussian_line_numba(
        spec_wobs,
        spec_R_fwhm,
        line_um,
        line_flux=line_flux,
        velocity_sigma=velocity_sigma,
    )
    assert np.allclose(line.sum() * dx, 1.0, rtol=rtol)

    for x_shift in np.linspace(-0.5, 0.5, 32):
        line = sample_gaussian_line_numba(
            spec_wobs,
            spec_R_fwhm,
            line_um + dx * x_shift,
            line_flux=line_flux,
            velocity_sigma=velocity_sigma,
        )
        assert np.allclose(line.sum() * dx, 1.0, rtol=rtol)

    # Variable wavelength spacing
    spec_wobs = np.logspace(np.log10(1.4), np.log10(1.6), N + 1)
    line = sample_gaussian_line_numba(
        spec_wobs,
        spec_R_fwhm,
        line_um,
        line_flux=line_flux,
        velocity_sigma=velocity_sigma,
    )
    assert np.allclose(np.trapz(line, spec_wobs), 1.0, rtol=rtol)

    # Set R from the pixel spacing
    spec_R_fwhm = 2.35 * spec_wobs / np.gradient(spec_wobs)

    line = sample_gaussian_line_numba(
        spec_wobs,
        spec_R_fwhm,
        line_um,
        line_flux=line_flux,
        velocity_sigma=velocity_sigma,
    )
    assert np.allclose(np.trapz(line, spec_wobs), 1.0, rtol=rtol)

    for v in [0, 100, 500, 1000, 2000, 5000]:
        # c = None
        for x_shift in np.linspace(-0.5, 0.5, 8):
            line = sample_gaussian_line_numba(
                spec_wobs,
                spec_R_fwhm,
                line_um + dx * x_shift,
                line_flux=line_flux,
                velocity_sigma=v,
            )
            assert np.allclose(np.trapz(line, spec_wobs), 1.0, rtol=rtol)
            # if 1:
            #     pl = plt.plot(spec_wobs, line, color=c, alpha=0.5)
            #     c = pl[0].get_color()
