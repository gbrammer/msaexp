import numpy as np
import os
from grizli import utils
from numba import jit, njit

@jit(nopython=True, fastmath=True, error_model='numpy')
def resample_template_numba(spec_wobs, spec_R_fwhm, templ_wobs, templ_flux, velocity_sigma=100, nsig=5, fill_value=0.):
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
    
    #Rw = 1./np.sqrt((velocity_sigma/3.e5)**2 + 1./(spec_R_fwhm*2.35)**2)
    # dw = spec_wobs / Rw
    
    ilo = 0
    ihi = 1
    
    N = len(spec_wobs)
    resamp = np.zeros_like(spec_wobs)*fill_value
    
    Nt = len(templ_wobs)
    
    for i in range(N):
        #sl = slice(ilo[i], ihi[i])
        while (templ_wobs[ilo] < spec_wobs[i]-nsig*dw[i]) & (ilo < Nt-1):
            ilo += 1
        
        if ilo == 0:
            continue
            
        ilo -= 1
        
        while (templ_wobs[ihi] < spec_wobs[i]+nsig*dw[i]) & (ihi < Nt):
            ihi += 1
        
        if (ilo >= ihi):
            resamp[i] = templ_flux[ihi]
            continue
        elif (ilo == Nt-1):
            break
            
        sl = slice(ilo, ihi)
        lsl = templ_wobs[sl]
        g = np.exp(-(lsl-spec_wobs[i])**2/2/dw[i]**2)/np.sqrt(2*np.pi*dw[i]**2)
        # g *= 1./np.sqrt(2*np.pi*dw[i]**2) 
        resamp[i] = np.trapz(templ_flux[sl]*g, lsl)
        
    return resamp
