import numpy as np

__all__ = [
    "resample_template",
    "sample_gaussian_line",
    "pixel_integrated_gaussian",
]


def resample_template(
    spec_wobs,
    spec_R_fwhm,
    templ_wobs,
    templ_flux,
    velocity_sigma=100,
    nsig=5,
    fill_value=0.0,
    wave_min=0.0,
    wave_max=1.0e6,
    left=0.0,
    right=0.0,
):
    """
    Resample a high resolution template/model on the wavelength grid of a
    spectrum with (potentially) wavelength dependent dispersion

    Parameters
    ----------
    spec_wobs : array-like
        Spectrum wavelengths

    spec_R_fwhm : array-like
        Spectral resolution `wave/d(wave)`, FWHM

    templ_wobs : array-like
        Template wavelengths, observed frame.  Same units as `spec_wobs`.
        **NB:** both `spec_wobs` and `templ_wobs` assumed to be sorted!

    templ_flux : array-like
        Template flux densities sampled at `templ_wobs`

    velocity_sigma : float
        Kinematic velocity width, km/s

    nsig : float
        Number of sigmas of the Gaussian convolution kernel to sample

    wave_min : float
        Minimum wavelength to consider

    wave_max : float
        Maximum wavelength to consider

    left, right : float
        Fill values when wavelengths less (greater) than wave_min (wave_max)

    Returns
    -------
    resamp : array-like
        Template resampled at the `spec_wobs` wavelengths, convolved with a
        Gaussian kernel with sigma width

        >>> Rw = 1./np.sqrt((velocity_sigma/3.e5)**2 + 1./(spec_R_fwhm*2.35)**2)
        >>> dw = spec_wobs / Rw

    """
    dw = (
        np.sqrt(
            (velocity_sigma / 3.0e5) ** 2 + (1.0 / 2.35 / spec_R_fwhm) ** 2
        )
        * spec_wobs
    )

    ix = np.arange(templ_wobs.shape[0])
    ilo = np.interp(spec_wobs - nsig * dw, templ_wobs, ix).astype(int)
    ihi = np.interp(spec_wobs + nsig * dw, templ_wobs, ix).astype(int) + 1

    N = len(spec_wobs)
    fres = np.ones(N) * fill_value
    range_idx = np.where((spec_wobs >= wave_min) & (spec_wobs <= wave_max))[0]

    for i in range_idx:
        sl = slice(ilo[i], ihi[i])
        lsl = templ_wobs[sl]
        g = np.exp(-((lsl - spec_wobs[i]) ** 2) / 2 / dw[i] ** 2)
        g *= 1.0 / np.sqrt(2 * np.pi * dw[i] ** 2)
        fres[i] = np.trapz(templ_flux[sl] * g, lsl)

    fres[spec_wobs < wave_min] = left
    fres[spec_wobs > wave_max] = right

    return fres


def sample_gaussian_line(
    spec_wobs,
    spec_R_fwhm,
    line_um,
    dx=None,
    line_flux=1.0,
    velocity_sigma=100,
    lorentz=False,
):
    """
    Sample a Gaussian emission line on the spectrum wavelength grid accounting
    for pixel integration

    Parameters
    ----------
    spec_wobs : array-like
        Spectrum wavelengths

    spec_R_fwhm : array-like
        Spectral resolution `wave/d(wave)`, FWHM

    line_um : float
        Emission line central wavelength, in microns

    line_flux : float
        Normalization of the line

    velocity_sigma : float
        Kinematic velocity width, km/s

    Returns
    -------
    resamp : array-like
        Emission line "template" resampled at the `spec_wobs` wavelengths

    """
    Rw = np.interp(line_um, spec_wobs, spec_R_fwhm)
    dw = (
        np.sqrt((velocity_sigma / 3.0e5) ** 2 + (1.0 / 2.35 / Rw) ** 2)
        * line_um
    )

    if lorentz:
        resamp = pixel_integrated_lorentzian(
            spec_wobs, line_um, dw, dx=dx, normalization=line_flux
        )
    else:
        resamp = pixel_integrated_gaussian(
            spec_wobs, line_um, dw, normalization=line_flux
        )

    return resamp


def pixel_integrated_gaussian(x, mu, sigma, normalization=1.0):
    """
    Low level function for a pixel-integrated gaussian

    Parameters
    ----------
    x : array-like
        Sample centers

    mu : float
        Gaussian center

    sigma : float
        Gaussian width

    normalization : float
        Scaling

    Returns
    -------
    samp : array-like
        Pixel-integrated Gaussian

    """
    from math import erf

    N = len(x)
    samp = np.zeros_like(x)
    s2dw = np.sqrt(2) * sigma

    # i = 0
    i = 0
    x0 = x[i] - mu
    dx = x[i + 1] - x[i]

    left = erf((x0 - dx / 2) / s2dw)
    right = erf((x0 + dx / 2) / s2dw)
    samp[i] = (right - left) / 2 / dx * normalization

    for i in range(1, N):
        x0 = x[i] - mu
        dx = x[i] - x[i - 1]

        left = erf((x0 - dx / 2) / s2dw)
        right = erf((x0 + dx / 2) / s2dw)
        samp[i] = (right - left) / 2 / dx * normalization

    return samp


def pixel_integrated_lorentzian(
    x, mu, sigma, dx=None, normalization=1.0, clip_sigma=100
):
    """
    Low level function for a pixel-integrated Lorentzian / Cauchy function

    Parameters
    ----------
    x : array-like
        Sample centers

    mu : float, array-like
        Lorentzian center

    sigma : float, array-like
        Lorentzian half-width

    dx : float, array-like
        Difference to override ``x[i+1] - x[i]`` if not provided as zero

    normalization : float
        Scaling

    Returns
    -------
    samp : array-like
        Pixel-integrated Gaussian

    """
    from numpy import pi as math_pi
    from numpy import arctan as atan

    N = len(x)
    samp = np.zeros_like(x)

    s2dw = sigma * np.ones_like(x)
    clip_sigma2 = clip_sigma**2
    mux = mu * np.ones_like(x)

    if dx is None:
        # Like np.gradient
        # https://github.com/numba/numba/issues/6302
        xdx = np.ones_like(x)
        xdx[1:-1] = (x[2:] - x[:-2]) / 2.0
        xdx[0] = x[1] - x[0]
        xdx[-1] = x[-1] - x[-2]
    else:
        xdx = dx * np.ones_like(x)

    i = 0
    x0 = x[i] - mux[i]

    left = atan((x0 - xdx[i] / 2) / s2dw[i])
    right = atan((x0 + xdx[i] / 2) / s2dw[i])
    samp[i] = (right - left) / xdx[i] * normalization / math_pi

    for i in range(1, N):
        x0 = x[i] - mux[i]
        xleft = (x0 - xdx[i] / 2) / s2dw[i]
        if xleft**2 > clip_sigma2:
            continue

        left = atan(xleft)
        right = atan((x0 + xdx[i] / 2) / s2dw[i])
        samp[i] = (right - left) / xdx[i] * normalization / math_pi

    return samp
