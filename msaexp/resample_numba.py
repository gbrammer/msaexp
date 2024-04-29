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


@jit(nopython=True, fastmath=True, error_model="numpy")
def calculate_igm(z, wobs, scale_tau=1.):
    """
    Calculate Inoue+14 IGM transmission
    
    Parameters
    ----------
    z : float
        Redshift
    
    wobs : array-like
        Observed-frame wavelengths, Angstroms
    
    scale_tau : float
        Scale factor multiplied to tau_igm
    
    Returns
    -------
    igmz : array-like
        IGM transmission
    """
    
    _LAF = np.array([
       [  2 ,1215.670, 1.68976E-02, 2.35379E-03, 1.02611E-04 ],
       [  3 ,1025.720, 4.69229E-03, 6.53625E-04, 2.84940E-05 ],
       [  4 , 972.537, 2.23898E-03, 3.11884E-04, 1.35962E-05 ],
       [  5 , 949.743, 1.31901E-03, 1.83735E-04, 8.00974E-06 ],
       [  6 , 937.803, 8.70656E-04, 1.21280E-04, 5.28707E-06 ],
       [  7 , 930.748, 6.17843E-04, 8.60640E-05, 3.75186E-06 ],
       [  8 , 926.226, 4.60924E-04, 6.42055E-05, 2.79897E-06 ],
       [  9 , 923.150, 3.56887E-04, 4.97135E-05, 2.16720E-06 ],
       [ 10 , 920.963, 2.84278E-04, 3.95992E-05, 1.72628E-06 ],
       [ 11 , 919.352, 2.31771E-04, 3.22851E-05, 1.40743E-06 ],
       [ 12 , 918.129, 1.92348E-04, 2.67936E-05, 1.16804E-06 ],
       [ 13 , 917.181, 1.62155E-04, 2.25878E-05, 9.84689E-07 ],
       [ 14 , 916.429, 1.38498E-04, 1.92925E-05, 8.41033E-07 ],
       [ 15 , 915.824, 1.19611E-04, 1.66615E-05, 7.26340E-07 ],
       [ 16 , 915.329, 1.04314E-04, 1.45306E-05, 6.33446E-07 ],
       [ 17 , 914.919, 9.17397E-05, 1.27791E-05, 5.57091E-07 ],
       [ 18 , 914.576, 8.12784E-05, 1.13219E-05, 4.93564E-07 ],
       [ 19 , 914.286, 7.25069E-05, 1.01000E-05, 4.40299E-07 ],
       [ 20 , 914.039, 6.50549E-05, 9.06198E-06, 3.95047E-07 ],
       [ 21 , 913.826, 5.86816E-05, 8.17421E-06, 3.56345E-07 ],
       [ 22 , 913.641, 5.31918E-05, 7.40949E-06, 3.23008E-07 ],
       [ 23 , 913.480, 4.84261E-05, 6.74563E-06, 2.94068E-07 ],
       [ 24 , 913.339, 4.42740E-05, 6.16726E-06, 2.68854E-07 ],
       [ 25 , 913.215, 4.06311E-05, 5.65981E-06, 2.46733E-07 ],
       [ 26 , 913.104, 3.73821E-05, 5.20723E-06, 2.27003E-07 ],
       [ 27 , 913.006, 3.45377E-05, 4.81102E-06, 2.09731E-07 ],
       [ 28 , 912.918, 3.19891E-05, 4.45601E-06, 1.94255E-07 ],
       [ 29 , 912.839, 2.97110E-05, 4.13867E-06, 1.80421E-07 ],
       [ 30 , 912.768, 2.76635E-05, 3.85346E-06, 1.67987E-07 ],
       [ 31 , 912.703, 2.58178E-05, 3.59636E-06, 1.56779E-07 ],
       [ 32 , 912.645, 2.41479E-05, 3.36374E-06, 1.46638E-07 ],
       [ 33 , 912.592, 2.26347E-05, 3.15296E-06, 1.37450E-07 ],
       [ 34 , 912.543, 2.12567E-05, 2.96100E-06, 1.29081E-07 ],
       [ 35 , 912.499, 1.99967E-05, 2.78549E-06, 1.21430E-07 ],
       [ 36 , 912.458, 1.88476E-05, 2.62543E-06, 1.14452E-07 ],
       [ 37 , 912.420, 1.77928E-05, 2.47850E-06, 1.08047E-07 ],
       [ 38 , 912.385, 1.68222E-05, 2.34330E-06, 1.02153E-07 ],
       [ 39 , 912.353, 1.59286E-05, 2.21882E-06, 9.67268E-08 ],
       [ 40 , 912.324, 1.50996E-05, 2.10334E-06, 9.16925E-08 ],
    ]).T

    ALAM =  _LAF[1]
    ALAF1 = _LAF[2]
    ALAF2 = _LAF[3]
    ALAF3 = _LAF[4]
    
    _DLA = np.array([
       [  2 , 1215.670, 1.61698E-04, 5.38995E-05 ],
       [  3 , 1025.720, 1.54539E-04, 5.15129E-05 ],
       [  4 ,  972.537, 1.49767E-04, 4.99222E-05 ],
       [  5 ,  949.743, 1.46031E-04, 4.86769E-05 ],
       [  6 ,  937.803, 1.42893E-04, 4.76312E-05 ],
       [  7 ,  930.748, 1.40159E-04, 4.67196E-05 ],
       [  8 ,  926.226, 1.37714E-04, 4.59048E-05 ],
       [  9 ,  923.150, 1.35495E-04, 4.51650E-05 ],
       [ 10 ,  920.963, 1.33452E-04, 4.44841E-05 ],
       [ 11 ,  919.352, 1.31561E-04, 4.38536E-05 ],
       [ 12 ,  918.129, 1.29785E-04, 4.32617E-05 ],
       [ 13 ,  917.181, 1.28117E-04, 4.27056E-05 ],
       [ 14 ,  916.429, 1.26540E-04, 4.21799E-05 ],
       [ 15 ,  915.824, 1.25041E-04, 4.16804E-05 ],
       [ 16 ,  915.329, 1.23614E-04, 4.12046E-05 ],
       [ 17 ,  914.919, 1.22248E-04, 4.07494E-05 ],
       [ 18 ,  914.576, 1.20938E-04, 4.03127E-05 ],
       [ 19 ,  914.286, 1.19681E-04, 3.98938E-05 ],
       [ 20 ,  914.039, 1.18469E-04, 3.94896E-05 ],
       [ 21 ,  913.826, 1.17298E-04, 3.90995E-05 ],
       [ 22 ,  913.641, 1.16167E-04, 3.87225E-05 ],
       [ 23 ,  913.480, 1.15071E-04, 3.83572E-05 ],
       [ 24 ,  913.339, 1.14011E-04, 3.80037E-05 ],
       [ 25 ,  913.215, 1.12983E-04, 3.76609E-05 ],
       [ 26 ,  913.104, 1.11972E-04, 3.73241E-05 ],
       [ 27 ,  913.006, 1.11002E-04, 3.70005E-05 ],
       [ 28 ,  912.918, 1.10051E-04, 3.66836E-05 ],
       [ 29 ,  912.839, 1.09125E-04, 3.63749E-05 ],
       [ 30 ,  912.768, 1.08220E-04, 3.60734E-05 ],
       [ 31 ,  912.703, 1.07337E-04, 3.57789E-05 ],
       [ 32 ,  912.645, 1.06473E-04, 3.54909E-05 ],
       [ 33 ,  912.592, 1.05629E-04, 3.52096E-05 ],
       [ 34 ,  912.543, 1.04802E-04, 3.49340E-05 ],
       [ 35 ,  912.499, 1.03991E-04, 3.46636E-05 ],
       [ 36 ,  912.458, 1.03198E-04, 3.43994E-05 ],
       [ 37 ,  912.420, 1.02420E-04, 3.41402E-05 ],
       [ 38 ,  912.385, 1.01657E-04, 3.38856E-05 ],
       [ 39 ,  912.353, 1.00908E-04, 3.36359E-05 ],
       [ 40 ,  912.324, 1.00168E-04, 3.33895E-05 ],
    ]).T
    
    ADLA1 = _DLA[2]
    ADLA2 = _DLA[3]
    
    _pow = lambda a, b: a**b
    
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
    
    for i, wi in enumerate(wobs):
        if wi > 1300.*(1 + zS):
            continue
        
        for j, lsj in enumerate(ALAM):
            # LS LAF
            if wi < lsj*(1 + zS):
                if wi < lsj*(1 + z1LAF):
                    # x1
                    tau[i] += ALAF1[j] * (wi / lsj)**1.2
                elif (wi >= lsj*(1 + z1LAF)) & (wi < lsj*(1 + z2LAF)):
                    tau[i] += ALAF2[j] * (wi / lsj)**3.7
                else:
                    tau[i] += ALAF3[j] * (wi / lsj)**5.5
            
            # LS DLA
            if wi < lsj*(1 + zS):
                if wi < lsj*(1. + z1DLA):
                    tau[i] += ADLA1[j] * (wi / lsj)**2
                else:
                    tau[i] += ADLA2[j] * (wi / lsj)**3
        
        # LC DLA
        if wi < lamL*(1+zS):
            if zS < z1DLA:
                tau[i] += (0.2113 * _pow(1 + zS, 2)
                                    - 0.07661 * _pow(1 + zS, 2.3) 
                                    * _pow(wi/lamL, (-3e-1))
                                    - 0.1347 * _pow(wi/lamL, 2)
                                    )
            else:
                x1 = wi >= lamL*(1 + z1DLA)
                if wi >= lamL*(1 + z1DLA):
                    tau[i] += (0.04696 * _pow(1 + zS, 3)
                                - 0.01779 * _pow(1 + zS, 3.3) 
                                * _pow(wi/lamL, (-3e-1))
                                - 0.02916 * _pow(wi/lamL, 3)
                                )
                else:
                    tau[i] += (0.6340 + 0.04696 * _pow(1 + zS, 3)
                                - 0.01779 * _pow(1 + zS, 3.3)
                                * _pow(wi/lamL, (-3e-1))
                                - 0.1347 * _pow(wi/lamL, 2)
                                - 0.2905 * _pow(wi/lamL, (-3e-1))
                               )
        
            # LC LAF
            if zS < z1LAF:
                tau[i] += (0.3248 * (_pow(wi/lamL, 1.2)
                            - _pow(1 + zS, -9e-1) * _pow(wi/lamL, 2.1))
                            )
            elif zS < z2LAF:
                if wi >= lamL*(1 + z1LAF):
                    tau[i] += (2.545e-2 * (_pow(1 + zS, 1.6)
                               * _pow(wi/lamL, 2.1)
                               - _pow(wi/lamL, 3.7))
                               )
                else:
                    tau[i] += (2.545e-2 * _pow(1 + zS, 1.6)
                               * _pow(wi/lamL, 2.1)
                               + 0.3248 * _pow(wi/lamL, 1.2)
                               - 0.2496 * _pow(wi/lamL, 2.1)
                               )
            else:
                
                if wi > lamL*(1.+z2LAF):
                    tau[i] += (5.221e-4 * (_pow(1 + zS, 3.4)
                                             * _pow(wi/lamL, 2.1)
                                             - _pow(wi/lamL, 5.5))
                                             )
                elif (wi >= lamL*(1 + z1LAF)) & (wi < lamL*(1 + z2LAF)):
                    tau[i] += (5.221e-4 * _pow(1 + zS, 3.4)
                                             * _pow(wi/lamL, 2.1)
                                             + 0.2182 * _pow(wi/lamL, 2.1)
                                             - 2.545e-2 * _pow(wi/lamL, 3.7)
                                             )
                elif wi < lamL*(1 + z1LAF):
                    tau[i] += (5.221e-4 * _pow(1 + zS, 3.4)
                                             * _pow(wi/lamL, 2.1)
                                             + 0.3248 * _pow(wi/lamL, 1.2)
                                             - 3.140e-2 * _pow(wi/lamL, 2.1)
                                             )

    igmz = np.exp(-scale_tau*tau)
    return igmz
    