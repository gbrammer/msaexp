import numpy as np
from numba import jit
from math import erf, atan, pow as _pow, pi as math_pi

__all__ = [
    "simpson",
    "trapz",
    "trapz_dx",
    "resample_template_numba",
    "sample_gaussian_line_numba",
    "pixel_integrated_gaussian_numba",
    "compute_igm",
    "calzetti2000_alambda",
    "calzetti2000_attenuation",
    "drude_profile",
    "salim_alambda",
    "smc_alambda",
    "smc_attenuation",
]

CLIGHT = 299792458.0  # m/s


@jit(nopython=True, fastmath=True, error_model="numpy")
def simpson(y, x):
    """
    Simpson's rule integration by Mason Stoeker
    See: https://masonstoecker.com/2021/04/03/Simpson-and-Numba.html

    Parameters
    ----------
    y : array-like
        dependent variable

    x : array-like
        independent variable

    Returns
    -------
    result : float
        Numerical integral of y(x)

    """

    n = len(y) - 1
    h = np.zeros(n)
    for i in range(n):
        h[i] = x[i + 1] - x[i]

    n = len(h) - 1
    s = 0
    for i in range(1, n, 2):
        a = h[i] * h[i]
        b = h[i] * h[i - 1]
        c = h[i - 1] * h[i - 1]
        d = h[i] + h[i - 1]
        alpha = (2 * a + b - c) / h[i]
        beta = d * d * d / b
        gamma = (-a + b + 2 * c) / h[i - 1]
        s += alpha * y[i + 1] + beta * y[i] + gamma * y[i - 1]

    if (n + 1) % 2 == 0:
        alpha = h[n - 1] * (3 - h[n - 1] / (h[n - 1] + h[n - 2]))
        beta = h[n - 1] * (3 + h[n - 1] / h[n - 2])
        gamma = (
            -h[n - 1]
            * h[n - 1]
            * h[n - 1]
            / (h[n - 2] * (h[n - 1] + h[n - 2]))
        )
        return (s + alpha * y[n] + beta * y[n - 1] + gamma * y[n - 2]) / 6
    else:
        return s / 6


@jit(nopython=True, fastmath=True, error_model="numpy")
def trapz_dx(x):
    """
    Return trapezoid rule coefficients, useful for numerical integration
    using a dot product

    Parameters
    ----------
    x : array-like
        Independent variable

    Returns
    -------
    h : array_like
        Coefficients for trapezoidal rule integration.

    """
    N = len(x)

    h = np.zeros(N)
    h[0] = (x[1] - x[0]) / 2.0
    h0 = h[0]
    for i in range(1, N - 1):
        h1 = (x[i + 1] - x[i]) / 2.0
        h[i] = h0 + h1
        h0 = h1

    h[i - 1] = h[i - 2] / h1

    return h


@jit(nopython=True, fastmath=True, error_model="numpy")
def trapz(y, x):
    """
    Accelerated trapezoid rule integration

    Parameters
    ----------
    y : array-like
        dependent variable

    x : array-like
        independent variable

    Returns
    -------
    result : float
        Numerical integral of y(x)

    """
    N = len(x)

    h = (x[1] - x[0]) / 2.0
    result = y[0] * h
    h0 = h

    for i in range(1, N - 1):
        h1 = (x[i + 1] - x[i]) / 2.0
        result += y[i] * (h0 + h1)
        h0 = h1

    result += y[-1] * h1

    return result


@jit(nopython=True, fastmath=True, error_model="numpy")
def resample_template_numba(
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

    fill_value : float
        Value to fill in the resampled array where no template data is available

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

    # Rw = 1./np.sqrt((velocity_sigma/3.e5)**2 + 1./(spec_R_fwhm*2.35)**2)
    # dw = spec_wobs / Rw

    ilo = 0
    ihi = 1

    N = len(spec_wobs)
    resamp = np.ones_like(spec_wobs) * fill_value

    Nt = len(templ_wobs)

    for i in range(N):
        if spec_wobs[i] < wave_min:
            resamp[i] = left
            continue
        elif spec_wobs[i] > wave_max:
            resamp[i] = right
            continue

        ilo_i = ilo
        while (templ_wobs[ilo] < spec_wobs[i] - nsig * dw[i]) & (ilo < Nt - 1):
            ilo += 1

        if ilo == 0:
            continue

        # did we take a step?
        if ilo > ilo_i:
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
        resamp[i] = trapz(templ_flux[sl] * g, lsl)

    return resamp


@jit(nopython=True, fastmath=True, error_model="numpy")
def sample_gaussian_line_numba(
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
        resamp = pixel_integrated_lorentzian_numba(
            spec_wobs, line_um, dw, dx=dx, normalization=line_flux
        )
    else:
        resamp = pixel_integrated_gaussian_numba(
            spec_wobs, line_um, dw, dx=dx, normalization=line_flux
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

    left = erf((x0 - xdx[i] / 2) / s2dw[i])
    right = erf((x0 + xdx[i] / 2) / s2dw[i])
    samp[i] = (right - left) / 2 / xdx[i] * normalization

    left = right

    for i in range(1, N):
        x0 = x[i] - mux[i]
        xleft = (x0 - xdx[i] / 2) / s2dw[i]
        left = erf(xleft)
        right = erf((x0 + xdx[i] / 2) / s2dw[i])
        samp[i] = (right - left) / 2 / xdx[i] * normalization
        # left = right

    return samp


@jit(nopython=True, fastmath=True, error_model="numpy")
def pixel_integrated_lorentzian_numba(
    x, mu, sigma, dx=None, normalization=1.0
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
    N = len(x)
    samp = np.zeros_like(x)

    s2dw = sigma * np.ones_like(x)
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
    left = right

    for i in range(1, N):
        x0 = x[i] - mux[i]
        xleft = (x0 - xdx[i] / 2) / s2dw[i]
        left = atan(xleft)
        right = atan((x0 + xdx[i] / 2) / s2dw[i])
        samp[i] = (right - left) / xdx[i] * normalization / math_pi
        # left = right

    return samp


import grizli.utils_numba.interp

INTERP_CONSERVE = grizli.utils_numba.interp.interp_conserve_c


@jit(nopython=True, fastmath=True, error_model="numpy")
def integrate_filter(
    filter_wave,
    filter_throughput,
    filter_norm,
    templ_wobs,
    templ_fnu,
):
    """
    Integrate the template through a `FilterDefinition` filter object.

    .. note:: The `grizli` interpolation function
              `grizli.utils_c.interp.interp_conserve_c` will be used if
              available.

    Parameters
    ----------
    filt : `~eazy.filters.FilterDefinition` object or a list of them
        Filter(s) to interpolate

    flam : bool
        Return integrated fluxes in f-lambda, rather than f-nu

    scale : float, array
        Scale factor applied to template before integrating.  If an
        array is specified, it must have the same size as the template
        ``wave`` array.

    z : float
        Redshift the template before integrating through the filter

    include_igm : bool
        Include IGM absorption

    redshift_type : str
        See `~eazy.templates.Template.zindex`.

    iz : int
        Evaluate for a specific index of the ``flux`` array rather than
        calculating with ``zindex``

    Returns
    -------
    fnu : float or array
        Template integrated through one or more filters from ``filt``.  By
        defaults has units of fnu

        .. note:: The interpolated fluxes *do not* include factors of
                  (1+z) from the redshifted templates.

    """

    templ_filt = INTERP_CONSERVE(
        filter_wave, templ_wobs, templ_fnu, left=0, right=0
    )

    # f_nu/lam dlam == f_nu d (ln nu)
    filter_flux = trapz(
        filter_throughput * templ_filt / filter_wave, filter_wave
    )

    filter_flux /= filter_norm

    return filter_flux


@jit(nopython=True, fastmath=True, error_model="numpy")
def compute_igm(z, wobs, scale_tau=1.0):
    """
    Calculate
    `Inoue+ (2014) <https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.1805I>`_
    IGM transmission, reworked from `~eazy.igm.Inoue14`

    Parameters
    ----------
    z : float
        Redshift

    wobs : array-like
        Observed-frame wavelengths, Angstroms

    scale_tau : float
        Scalar multiplied to tau_igm

    Returns
    -------
    igmz : array-like
        IGM transmission factor
    """

    _LAF = np.array(
        [
            [2, 1215.670, 1.68976e-02, 2.35379e-03, 1.02611e-04],
            [3, 1025.720, 4.69229e-03, 6.53625e-04, 2.84940e-05],
            [4, 972.537, 2.23898e-03, 3.11884e-04, 1.35962e-05],
            [5, 949.743, 1.31901e-03, 1.83735e-04, 8.00974e-06],
            [6, 937.803, 8.70656e-04, 1.21280e-04, 5.28707e-06],
            [7, 930.748, 6.17843e-04, 8.60640e-05, 3.75186e-06],
            [8, 926.226, 4.60924e-04, 6.42055e-05, 2.79897e-06],
            [9, 923.150, 3.56887e-04, 4.97135e-05, 2.16720e-06],
            [10, 920.963, 2.84278e-04, 3.95992e-05, 1.72628e-06],
            [11, 919.352, 2.31771e-04, 3.22851e-05, 1.40743e-06],
            [12, 918.129, 1.92348e-04, 2.67936e-05, 1.16804e-06],
            [13, 917.181, 1.62155e-04, 2.25878e-05, 9.84689e-07],
            [14, 916.429, 1.38498e-04, 1.92925e-05, 8.41033e-07],
            [15, 915.824, 1.19611e-04, 1.66615e-05, 7.26340e-07],
            [16, 915.329, 1.04314e-04, 1.45306e-05, 6.33446e-07],
            [17, 914.919, 9.17397e-05, 1.27791e-05, 5.57091e-07],
            [18, 914.576, 8.12784e-05, 1.13219e-05, 4.93564e-07],
            [19, 914.286, 7.25069e-05, 1.01000e-05, 4.40299e-07],
            [20, 914.039, 6.50549e-05, 9.06198e-06, 3.95047e-07],
            [21, 913.826, 5.86816e-05, 8.17421e-06, 3.56345e-07],
            [22, 913.641, 5.31918e-05, 7.40949e-06, 3.23008e-07],
            [23, 913.480, 4.84261e-05, 6.74563e-06, 2.94068e-07],
            [24, 913.339, 4.42740e-05, 6.16726e-06, 2.68854e-07],
            [25, 913.215, 4.06311e-05, 5.65981e-06, 2.46733e-07],
            [26, 913.104, 3.73821e-05, 5.20723e-06, 2.27003e-07],
            [27, 913.006, 3.45377e-05, 4.81102e-06, 2.09731e-07],
            [28, 912.918, 3.19891e-05, 4.45601e-06, 1.94255e-07],
            [29, 912.839, 2.97110e-05, 4.13867e-06, 1.80421e-07],
            [30, 912.768, 2.76635e-05, 3.85346e-06, 1.67987e-07],
            [31, 912.703, 2.58178e-05, 3.59636e-06, 1.56779e-07],
            [32, 912.645, 2.41479e-05, 3.36374e-06, 1.46638e-07],
            [33, 912.592, 2.26347e-05, 3.15296e-06, 1.37450e-07],
            [34, 912.543, 2.12567e-05, 2.96100e-06, 1.29081e-07],
            [35, 912.499, 1.99967e-05, 2.78549e-06, 1.21430e-07],
            [36, 912.458, 1.88476e-05, 2.62543e-06, 1.14452e-07],
            [37, 912.420, 1.77928e-05, 2.47850e-06, 1.08047e-07],
            [38, 912.385, 1.68222e-05, 2.34330e-06, 1.02153e-07],
            [39, 912.353, 1.59286e-05, 2.21882e-06, 9.67268e-08],
            [40, 912.324, 1.50996e-05, 2.10334e-06, 9.16925e-08],
        ]
    ).T

    ALAM = _LAF[1]
    ALAF1 = _LAF[2]
    ALAF2 = _LAF[3]
    ALAF3 = _LAF[4]

    _DLA = np.array(
        [
            [2, 1215.670, 1.61698e-04, 5.38995e-05],
            [3, 1025.720, 1.54539e-04, 5.15129e-05],
            [4, 972.537, 1.49767e-04, 4.99222e-05],
            [5, 949.743, 1.46031e-04, 4.86769e-05],
            [6, 937.803, 1.42893e-04, 4.76312e-05],
            [7, 930.748, 1.40159e-04, 4.67196e-05],
            [8, 926.226, 1.37714e-04, 4.59048e-05],
            [9, 923.150, 1.35495e-04, 4.51650e-05],
            [10, 920.963, 1.33452e-04, 4.44841e-05],
            [11, 919.352, 1.31561e-04, 4.38536e-05],
            [12, 918.129, 1.29785e-04, 4.32617e-05],
            [13, 917.181, 1.28117e-04, 4.27056e-05],
            [14, 916.429, 1.26540e-04, 4.21799e-05],
            [15, 915.824, 1.25041e-04, 4.16804e-05],
            [16, 915.329, 1.23614e-04, 4.12046e-05],
            [17, 914.919, 1.22248e-04, 4.07494e-05],
            [18, 914.576, 1.20938e-04, 4.03127e-05],
            [19, 914.286, 1.19681e-04, 3.98938e-05],
            [20, 914.039, 1.18469e-04, 3.94896e-05],
            [21, 913.826, 1.17298e-04, 3.90995e-05],
            [22, 913.641, 1.16167e-04, 3.87225e-05],
            [23, 913.480, 1.15071e-04, 3.83572e-05],
            [24, 913.339, 1.14011e-04, 3.80037e-05],
            [25, 913.215, 1.12983e-04, 3.76609e-05],
            [26, 913.104, 1.11972e-04, 3.73241e-05],
            [27, 913.006, 1.11002e-04, 3.70005e-05],
            [28, 912.918, 1.10051e-04, 3.66836e-05],
            [29, 912.839, 1.09125e-04, 3.63749e-05],
            [30, 912.768, 1.08220e-04, 3.60734e-05],
            [31, 912.703, 1.07337e-04, 3.57789e-05],
            [32, 912.645, 1.06473e-04, 3.54909e-05],
            [33, 912.592, 1.05629e-04, 3.52096e-05],
            [34, 912.543, 1.04802e-04, 3.49340e-05],
            [35, 912.499, 1.03991e-04, 3.46636e-05],
            [36, 912.458, 1.03198e-04, 3.43994e-05],
            [37, 912.420, 1.02420e-04, 3.41402e-05],
            [38, 912.385, 1.01657e-04, 3.38856e-05],
            [39, 912.353, 1.00908e-04, 3.36359e-05],
            [40, 912.324, 1.00168e-04, 3.33895e-05],
        ]
    ).T

    ADLA1 = _DLA[2]
    ADLA2 = _DLA[3]

    # def _pow(a, b):
    #     return a**b

    ####
    # Lyman series, Lyman-alpha forest
    ####
    z1LAF = 1.2
    z2LAF = 4.7

    ###
    # Lyman Series, DLA
    ###
    z1DLA = 2.0

    ###
    # Lyman continuum, DLA
    ###
    lamL = 911.8

    tau = np.zeros_like(wobs)
    zS = z

    # Explicit iteration should be fast in JIT
    for i, wi in enumerate(wobs):
        if wi > 1300.0 * (1 + zS):
            continue

        # Iterate over Lyman series
        for j, lsj in enumerate(ALAM):
            # LS LAF
            if wi < lsj * (1 + zS):
                if wi < lsj * (1 + z1LAF):
                    # x1
                    tau[i] += ALAF1[j] * (wi / lsj) ** 1.2
                elif (wi >= lsj * (1 + z1LAF)) & (wi < lsj * (1 + z2LAF)):
                    tau[i] += ALAF2[j] * (wi / lsj) ** 3.7
                else:
                    tau[i] += ALAF3[j] * (wi / lsj) ** 5.5

            # LS DLA
            if wi < lsj * (1 + zS):
                if wi < lsj * (1.0 + z1DLA):
                    tau[i] += ADLA1[j] * (wi / lsj) ** 2
                else:
                    tau[i] += ADLA2[j] * (wi / lsj) ** 3

        # Lyman Continuum
        if wi < lamL * (1 + zS):
            # LC DLA
            if zS < z1DLA:
                tau[i] += (
                    0.2113 * _pow(1 + zS, 2)
                    - 0.07661 * _pow(1 + zS, 2.3) * _pow(wi / lamL, (-3e-1))
                    - 0.1347 * _pow(wi / lamL, 2)
                )
            else:
                x1 = wi >= lamL * (1 + z1DLA)
                if wi >= lamL * (1 + z1DLA):
                    tau[i] += (
                        0.04696 * _pow(1 + zS, 3)
                        - 0.01779
                        * _pow(1 + zS, 3.3)
                        * _pow(wi / lamL, (-3e-1))
                        - 0.02916 * _pow(wi / lamL, 3)
                    )
                else:
                    tau[i] += (
                        0.6340
                        + 0.04696 * _pow(1 + zS, 3)
                        - 0.01779
                        * _pow(1 + zS, 3.3)
                        * _pow(wi / lamL, (-3e-1))
                        - 0.1347 * _pow(wi / lamL, 2)
                        - 0.2905 * _pow(wi / lamL, (-3e-1))
                    )

            # LC LAF
            if zS < z1LAF:
                tau[i] += 0.3248 * (
                    _pow(wi / lamL, 1.2)
                    - _pow(1 + zS, -9e-1) * _pow(wi / lamL, 2.1)
                )
            elif zS < z2LAF:
                if wi >= lamL * (1 + z1LAF):
                    tau[i] += 2.545e-2 * (
                        _pow(1 + zS, 1.6) * _pow(wi / lamL, 2.1)
                        - _pow(wi / lamL, 3.7)
                    )
                else:
                    tau[i] += (
                        2.545e-2 * _pow(1 + zS, 1.6) * _pow(wi / lamL, 2.1)
                        + 0.3248 * _pow(wi / lamL, 1.2)
                        - 0.2496 * _pow(wi / lamL, 2.1)
                    )
            else:
                if wi > lamL * (1.0 + z2LAF):
                    tau[i] += 5.221e-4 * (
                        _pow(1 + zS, 3.4) * _pow(wi / lamL, 2.1)
                        - _pow(wi / lamL, 5.5)
                    )
                elif (wi >= lamL * (1 + z1LAF)) & (wi < lamL * (1 + z2LAF)):
                    tau[i] += (
                        5.221e-4 * _pow(1 + zS, 3.4) * _pow(wi / lamL, 2.1)
                        + 0.2182 * _pow(wi / lamL, 2.1)
                        - 2.545e-2 * _pow(wi / lamL, 3.7)
                    )
                elif wi < lamL * (1 + z1LAF):
                    tau[i] += (
                        5.221e-4 * _pow(1 + zS, 3.4) * _pow(wi / lamL, 2.1)
                        + 0.3248 * _pow(wi / lamL, 1.2)
                        - 3.140e-2 * _pow(wi / lamL, 2.1)
                    )

    igmz = np.exp(-scale_tau * tau)

    return igmz


@jit(nopython=True, fastmath=True, error_model="numpy")
def _calzetti2000_alambda(w0):
    """
    Calzetti (2000) attenuation law adapted from `bagpipes` (A. Carnall)

    Parameters
    ----------
    w0 : float
        Rest-frame wavelength in microns

    Returns
    -------
    Alam : float
        A(w0) / Av
    """
    if w0 < 0.1200:
        Alam = (w0 / 0.12) ** -0.77 * (
            4.05
            + 2.695
            * (-2.156 + 1.509 / 0.12 - 0.198 / 0.12**2 + 0.011 / 0.12**3)
        )
    elif w0 < 0.6300:
        Alam = 4.05 + 2.695 * (
            -2.156 + 1.509 / w0 - 0.198 / w0**2 + 0.011 / w0**3
        )
    elif w0 < 3.1:
        Alam = 2.659 * (-1.857 + 1.040 / w0) + 4.05

    else:
        Alam = 0.0

    Alam /= 4.05

    return Alam


@jit(nopython=True, fastmath=True, error_model="numpy")
def calzetti2000_alambda(wobs, z):
    """
    Calzetti (2000) attenuation law adapted from `bagpipes` (A. Carnall)

    Parameters
    ----------
    wobs : array-like
        Observed-frame wavelength, microns

    z : float
        Redshift

    Returns
    -------
    A_lambda : array-like
        Attenuation A(lam)/Av as a function of wavelength, magnitudes

    """

    A_lambda = np.zeros_like(wobs)

    for i, w in enumerate(wobs):
        A_lambda[i] = _calzetti2000_alambda(w / (1 + z))

    return A_lambda


@jit(nopython=True, fastmath=True, error_model="numpy")
def calzetti2000_attenuation(wobs, z, Av):
    """
    Calzetti (2000) attenuation law adapted from `bagpipes` (A. Carnall)

    Parameters
    ----------
    wobs : array-like
        Observed-frame wavelength, microns

    z : float
        Redshift

    Av : float
        Extinction in the V band, magnitudes.

    Returns
    -------
    A_linear : array-like
        Linear attenuation as a function of wavelength
    """
    A_lambda = calzetti2000_alambda(wobs, z)
    A_linear = 10 ** (-0.4 * A_lambda * Av)

    return A_linear


@jit(nopython=True, fastmath=True, error_model="numpy")
def drude_profile(w0, center, width):
    """
    Drude profile

    Parameters
    ----------
    w0 : float
        Rest-frame wavelength

    center : float
        Center of the profile, e.g., 0.2175 microns

    width : float
        Width of the profile, e.g., 0.035 microns

    Returns
    -------
    drude : float
        Drude profile
    """
    drude = w0**2 * width**2
    drude /= (w0**2 - center**2) ** 2 + w0**2 * width**2
    return drude


@jit(nopython=True, fastmath=True, error_model="numpy")
def salim_alambda(wobs, z, delta, B):
    """
    Salim (2018) attenuation law adapted from `bagpipes` (A. Carnall)

    Parameters
    ----------
    wobs : array-like
        Observed-frame wavelength, microns

    z : float
        Redshift

    delta : float
        Modification of slope w.r.t. Calzetti (2000)

    B : float
        Amplitude of the dust bump Drude profile

    Returns
    -------
    A_lambda : array-like
        Attenuation as a function of wavelength, magnitudes
    """
    Rv_m = 4.05 / ((4.05 + 1) * (4400.0 / 5500.0) ** delta - 4.05)

    drude_center = 0.2175  # microns
    drude_width = 0.035  # microns

    A_lambda = np.zeros_like(wobs)

    for i, w in enumerate(wobs):
        w0 = w / (1 + z)
        A_lambda[i] = (
            _calzetti2000_alambda(w0) * Rv_m * (w0 / 0.55) ** delta
            + B * drude_profile(w0, drude_center, drude_width)
        ) / Rv_m

    return A_lambda


@jit(nopython=True, fastmath=True, error_model="numpy")
def salim_polynomial_klambda(w, Rv, B, a0, a1, a2, a3):
    """
    Generalized Salim+2018 attenuation curve

    Parameters
    ----------
    w : float, array-like
        Rest-frame wavelength

    Returns
    -------
    k_lambda : float, array-like

        ``k_lambda = = a0 + a1 / w + a2 / w**2 + a3 / w**3 + B * Drude(w) + Rv``

    """
    drude_center = 0.2175  # microns
    drude_width = 0.035  # microns

    k_lambda = (
        a0
        + a1 / w
        + a2 / w**2
        + a3 / w**3
        + B * drude_profile(w, drude_center, drude_width)
        + Rv
    )

    return k_lambda


@jit(nopython=True, fastmath=True, error_model="numpy")
def salim2018_fit_alambda(wobs, z, index):
    """
    Attenuation curve fits with coefficients from Salim+2018, Table 1

    Parameters
    ----------
    wobs : array-like
        Observed-frame wavelength, microns

    z : float
        Redshift

    index : int, [0 - 7]
        Set of coefficients to use:
            0: Star-forming galaxies
            1: 8.5 < logM* < 9.5.
            2: 9.5 < logM* < 10.5
            3: 10.5 < logM* < 11.5
            4: High-z analogs
            5: logM* < 10
            6: logM* > 10
            7: Quiescent galaxies

    Returns
    -------
    A_lambda : array-like
        Attenuation A(lam)/Av as a function of wavelength, magnitudes

    """

    ############
    #       Rv     B     a0    a1      a2      a3   lmax    n
    table1 = [
        [3.15, 1.57, -4.30, 2.71, -0.191, 0.0121, 2.28, 1.15],
        [2.61, 2.62, -3.66, 2.13, -0.043, 0.0086, 2.01, 1.43],
        [2.99, 1.73, -4.13, 2.56, -0.153, 0.0105, 2.18, 1.20],
        [3.47, 1.09, -4.66, 3.03, -0.271, 0.0147, 2.45, 1.00],
        [2.88, 2.27, -4.01, 2.46, -0.128, 0.0098, 2.12, 1.24],
        [2.72, 2.74, -3.80, 2.25, -0.073, 0.0092, 2.05, 1.38],
        [2.93, 2.11, -4.12, 2.56, -0.152, 0.0104, 2.09, 1.19],
        [2.61, 2.21, -3.72, 2.20, -0.062, 0.0080, 1.95, 1.35],
    ]

    Rv, B, a0, a1, a2, a3, lmax, _n = table1[index]
    k_lambda = np.zeros_like(wobs)
    for i, w in enumerate(wobs):
        w0 = w / (1 + z)
        if w0 < lmax:
            k_lambda[i] = salim_polynomial_klambda(w0, Rv, B, a0, a1, a2, a3)

    return k_lambda / Rv


@jit(nopython=True, fastmath=True, error_model="numpy")
def smc_alambda(wobs, z):
    """
    SMC extinction law from Gordon (2003) attenuation law adapted from
    `bagpipes` (A. Carnall)

    Parameters
    ----------
    wobs : array-like
        Observed-frame wavelength, microns

    z : float
        Redshift

    Returns
    -------
    A_lambda : array-like
        Attenuation A(lam)/Av as a function of wavelength, magnitudes

    """

    A_lambda = np.zeros_like(wobs)

    c1 = -4.959
    c2 = 2.264
    c3 = 0.389
    c4 = 0.461
    x0 = 4.6

    gamma = 1.0
    Rv = 2.74

    #####
    # Interpolate beyond 2760 Angstroms
    Nint = 11

    int_w = [
        0.2760,
        0.296,
        0.37,
        0.44,
        0.55,
        0.65,
        0.81,
        1.25,
        1.65,
        2.198,
        3.1,
    ]

    int_y = [
        2.220,
        2.000,
        1.672,
        1.374,
        1.00,
        0.801,
        0.567,
        0.25,
        0.169,
        0.11,
        0.0,
    ]

    # Precompute slopes for interpolation
    interp_m = [
        (int_y[j] - int_y[j - 1]) / (int_w[j] - int_w[j - 1])
        for j in range(1, Nint)
    ]

    for i, w in enumerate(wobs):
        # k = 1 / wave_microns
        w0 = w / (1 + z)
        k = 1.0 / w0

        if w0 > 0.2760:

            alam = 0.0

            for j in range(1, Nint):
                if int_w[j] > w0:
                    # Linear interpolation
                    alam = (w0 - int_w[j - 1]) * interp_m[j - 1] + int_y[j - 1]
                    break

            A_lambda[i] = alam

        else:
            D = k**2 / ((k**2 - x0**2) ** 2 + k**2 * gamma**2)
            if k < 5.9:
                F = 0.0
            else:
                F = 0.5392 * (k - 5.9) ** 2 + 0.05644 * (k - 5.9) ** 3

            A_lambda[i] = (c1 + c2 * k + c3 * D + c4 * F) / Rv + 1.0

    return A_lambda


@jit(nopython=True, fastmath=True, error_model="numpy")
def smc_attenuation(wobs, z, Av):
    """
    SMC extinction law from Gordon (2003) attenuation law adapted from
    `bagpipes` (A. Carnall)

    Parameters
    ----------
    wobs : array-like
        Observed-frame wavelength, microns

    z : float
        Redshift

    Av : float
        Extinction in the V band, magnitudes.

    Returns
    -------
    A_linear : array-like
        Linear attenuation as a function of wavelength
    """
    A_lambda = smc_alambda(wobs, z)
    A_linear = 10 ** (-0.4 * A_lambda * Av)

    return A_linear
