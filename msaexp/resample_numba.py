import numpy as np
from numba import jit
from math import erf

__all__ = [
    "resample_template_numba",
    "sample_gaussian_line_numba",
    "pixel_integrated_gaussian_numba",
]


@jit(nopython=True, fastmath=True, error_model="numpy")
def resample_template_numba(
    spec_wobs,
    spec_R_fwhm,
    templ_wobs,
    templ_flux,
    velocity_sigma=100,
    nsig=5,
    fill_value=0.0,
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

    # Rw = 1./np.sqrt((velocity_sigma/3.e5)**2 + 1./(spec_R_fwhm*2.35)**2)
    # dw = spec_wobs / Rw

    ilo = 0
    ihi = 1

    N = len(spec_wobs)
    resamp = np.zeros_like(spec_wobs) * fill_value

    Nt = len(templ_wobs)

    for i in range(N):
        # sl = slice(ilo[i], ihi[i])
        while (templ_wobs[ilo] < spec_wobs[i] - nsig * dw[i]) & (ilo < Nt - 1):
            ilo += 1

        if ilo == 0:
            continue

        ilo -= 1

        while (templ_wobs[ihi] < spec_wobs[i] + nsig * dw[i]) & (ihi < Nt):
            ihi += 1

        if ilo >= ihi:
            resamp[i] = templ_flux[ihi]
            continue
        elif ilo == Nt - 1:
            break

        sl = slice(ilo, ihi)
        lsl = templ_wobs[sl]
        g = np.exp(-((lsl - spec_wobs[i]) ** 2) / 2 / dw[i] ** 2) / np.sqrt(
            2 * np.pi * dw[i] ** 2
        )
        # g *= 1./np.sqrt(2*np.pi*dw[i]**2)
        resamp[i] = np.trapz(templ_flux[sl] * g, lsl)

    return resamp


@jit(nopython=True, fastmath=True, error_model="numpy")
def sample_gaussian_line_numba(
    spec_wobs, spec_R_fwhm, line_um, line_flux=1.0, velocity_sigma=100
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

    resamp = pixel_integrated_gaussian_numba(
        spec_wobs, line_um, dw, normalization=line_flux
    )

    return resamp


@jit(nopython=True, fastmath=True, error_model="numpy")
def pixel_integrated_gaussian_numba(x, mu, sigma, dx=None, normalization=1.0):
    """
    Low level function for a pixel-integrated gaussian

    Parameters
    ----------
    x : array-like
        Sample centers

    mu : float, array-like
        Gaussian center

    sigma : float, array-like
        Gaussian width

    dx : float, array-like
        Difference to override ``x[i+1] - x[i]`` if not provided as zero

    normalization : float
        Scaling

    Returns
    -------
    samp : array-like
        Pixel-integrated Gaussian

    """
    N = len(x)
    samp = np.zeros_like(x)

    s2dw = np.sqrt(2) * sigma * np.ones_like(x)

    mux = mu * np.ones_like(x)

    if dx is None:
        xdx = x[1:] - x[:-1]
    else:
        xdx = dx * np.ones_like(x)

    i = 0
    x0 = x[i] - mux[i]

    left = erf((x0 - xdx[i] / 2) / s2dw[i])
    right = erf((x0 + xdx[i] / 2) / s2dw[i])
    samp[i] = (right - left) / 2 / xdx[i] * normalization

    for i in range(1, N):
        x0 = x[i] - mux[i]
        left = erf((x0 - xdx[i] / 2) / s2dw[i])
        right = erf((x0 + xdx[i] / 2) / s2dw[i])
        samp[i] = (right - left) / 2 / xdx[i] * normalization

    return samp
