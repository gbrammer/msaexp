import numpy as np

__all__ = [
    "resample_template",
    "sample_gaussian_line",
    "pixel_integrated_gaussian",
]


def resample_template(
    spec_wobs, spec_R_fwhm, templ_wobs, templ_flux, velocity_sigma=100, nsig=5
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

    ix = np.arange(templ_wobs.shape[0])
    ilo = np.cast[int](np.interp(spec_wobs - nsig * dw, templ_wobs, ix))
    ihi = np.cast[int](np.interp(spec_wobs + nsig * dw, templ_wobs, ix)) + 1

    N = len(spec_wobs)
    fres = np.zeros(N)
    for i in range(N):
        sl = slice(ilo[i], ihi[i])
        lsl = templ_wobs[sl]
        g = np.exp(-((lsl - spec_wobs[i]) ** 2) / 2 / dw[i] ** 2)
        g *= 1.0 / np.sqrt(2 * np.pi * dw[i] ** 2)
        fres[i] = np.trapz(templ_flux[sl] * g, lsl)

    return fres


def sample_gaussian_line(
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
