import numpy as np
import os
from grizli import utils

def resample_template(spec_wobs, spec_R_fwhm, templ_wobs, templ_flux, velocity_sigma=100, nsig=5):
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
    dw = np.sqrt((velocity_sigma/3.e5)**2 + (1./2.35/spec_R_fwhm)**2)*spec_wobs
    
    ix = np.arange(templ_wobs.shape[0])
    ilo = np.cast[int](np.interp(spec_wobs-nsig*dw, templ_wobs, ix))
    ihi = np.cast[int](np.interp(spec_wobs+nsig*dw, templ_wobs, ix))+1
    
    N = len(spec_wobs)
    fres = np.zeros(N)
    for i in range(N):
        sl = slice(ilo[i], ihi[i])
        lsl = templ_wobs[sl]
        g = np.exp(-(lsl-spec_wobs[i])**2/2/dw[i]**2)
        g *= 1./np.sqrt(2*np.pi*dw[i]**2) 
        fres[i] = np.trapz(templ_flux[sl]*g, lsl)
        
    return fres
