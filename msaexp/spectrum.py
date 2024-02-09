
"""
Fits, etc. to extracted spectra
"""
import os
import time
import warnings

import numpy as np

import scipy.ndimage as nd
from scipy.optimize import nnls

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec

import astropy.io.fits as pyfits

from grizli import utils
utils.set_warnings()

try:
    import eazy
    wave = np.exp(np.arange(np.log(2.4), np.log(4.5), 1./4000))*1.e4
    _temp = utils.pah33(wave)
    PAH_TEMPLATES = {}
    for t in _temp:
        if '3.47' in t:
            continue
            
        _tp = _temp[t]
        PAH_TEMPLATES[t] = eazy.templates.Template(name=t, arrays=(_tp.wave, _tp.flux))
except:
    print('Failed to initialize PAH_TEMPLATES')
    PAH_TEMPLATES = {}

import grizli.utils_c
import astropy.units as u

import eazy.igm
igm = eazy.igm.Inoue14()

from . import drizzle
from . import utils as msautils

SCALE_UNCERTAINTY = 1.0

# try:
#     from prospect.utils.smoothing import smoothspec
# except (FileNotFoundError, TypeError):
#     if 'SPS_HOME' not in os.environ:
#         sps_home = 'xxxxdummyxxxx' #os.path.dirname(__file__)
#         print(f'msaexp: setting environment variable SPS_HOME={sps_home} '
#               'to be able to import prospect.utils.smoothing')
#         os.environ['SPS_HOME'] = sps_home

FFTSMOOTH = False

__all__ = ["fit_redshift", "fit_redshift_grid", "plot_spectrum", 
           "read_spectrum", "calc_uncertainty_scale",
           "SpectrumSampler"]

def test():
    
    from importlib import reload
    import msaexp.spectrum

    from tqdm import tqdm

    import msaexp.resample_numba
    from grizli import utils
    
    reload(msaexp.resample_numba); reload(msaexp.spectrum)
    reload(msaexp.resample_numba); reload(msaexp.spectrum)
    
    from msaexp.spectrum import SpectrumSampler
    import eazy.templates
    
    self = SpectrumSampler('macsj0647_1169.v1.spec.fits')
    
    t = eazy.templates.Template('templates/sfhz/fsps_4590.fits')
    z = 4.2418
    
    res = self.resample_eazy_template(t, z=z)
    
    line = self.resample_eazy_template(t, z=z)
    
    lw, lr = utils.get_line_wavelengths()
    
    k = 'highO32'
    
    zg = np.linspace(z-0.1, z+0.1, 256)
    chi2 = zg*0.
    
    bspl = self.bspline_array(nspline=13, log=True)
    bspl2 = self.bspline_array(nspline=3, log=True)

    scale_disp = 1.2
    velocity_sigma = 100
    
    for i, zi in tqdm(enumerate(zg)):
        lines = [self.fast_emission_line(w*(1+zi)/1.e4,
                                       line_flux=r,
                                       scale_disp=scale_disp,
                                       velocity_sigma=velocity_sigma,
                                       nsig=4)
                 for w, r in zip(lw[k], lr[k])]
    
        A = np.vstack([np.array(lines).sum(axis=0)*bspl2] + [bspl])
    
        Ax = (A / self.spec['full_err'])
        yx = self.spec['flux'] / self.spec['full_err']
    
        x = np.linalg.lstsq(Ax[:,self.valid].T, yx[self.valid].data, rcond=None)
    
        model = A.T.dot(x[0])
        resid = (self.spec['flux'] - model)/self.spec['full_err']
        chi2[i] = (resid[self.valid]**2).sum()
    
    zi = zg[np.argmin(chi2)]
    
    lines = [self.fast_emission_line(w*(1+zi)/1.e4,
                                   line_flux=r,
                                   scale_disp=scale_disp,
                                   velocity_sigma=velocity_sigma,
                                   nsig=4)
             for w, r in zip(lw[k], lr[k])]

    A = np.vstack([np.array(lines).sum(axis=0)*bspl2] + [bspl])

    Ax = (A / self.spec['full_err'])
    yx = self.spec['flux'] / self.spec['full_err']

    x = np.linalg.lstsq(Ax[:,self.valid].T, yx[self.valid].data, rcond=None)

    model = A.T.dot(x[0])


class SpectrumSampler(object):
        
    spec = {}
    spec_wobs = None
    spec_R_fwhm = None
    valid = None
    
    def __init__(self, spec_input, **kwargs):
        """
        Helper functions for sampling templates onto the wavelength grid
        of an observed spectrum
        
        Parameters
        ----------
        spec_input : str, `~astropy.io.fits.HDUList`
            - `str` : spectrum filename, usually `[root].spec.fits`
            - `~astropy.io.fits.HDUList` : FITS data
        
        Attributes
        ----------
        resample_func : func
            Template resampling function, from 
            `msaexp.resample_template_numba.msaexp.resample_numba` if possible and 
            `msaexp.resample.resample_template` otherwise
        
        sample_line_func : func
            Emission line function, from 
            `msaexp.resample_template_numba.msaexp.sample_gaussian_line_numba` if 
             possible and `msaexp.resample.sample_line_func` otherwise
        
        spec : `~astropy.table.Table`
            1D spectrum table from the `SPEC1D HDU of ``file``
        
        spec_wobs : array-like
            Observed wavelengths, microns
        
        spec_R_fwhm : array-like
            Tabulated spectral resolution `R = lambda / dlambda`, assumed to be
            defined as FWHM
        
        valid : array-like
            Boolean array of valid 1D data
        
        """
        try:
            from .resample_numba import resample_template_numba as resample_func
            from .resample_numba import sample_gaussian_line_numba as sample_line_func
        except ImportError:
            from .resample import resample_template as resample_func
            from .resample import sample_gaussian_line as sample_line_func
        
        self.resample_func = resample_func
        self.sample_line_func = sample_line_func
                
        self.initialize_spec(spec_input, **kwargs)

        self.initialize_emission_line()


    def __getitem__(self, key):
        """
        Return column of the `spec` table
        """
        return self.spec[key]


    @property
    def meta(self):
        """
        Metadata of `spec` table
        """
        return self.spec.meta
    
    
    def initialize_emission_line(self, nsamp=64):
        """
        Initialize emission line
        """
        self.xline = np.linspace(-nsamp, nsamp, 2*nsamp+1)/nsamp*0.1+1
        self.yline = self.xline*0.
        self.yline[nsamp] = 1
        self.yline /= np.trapz(self.yline, self.xline)


    def initialize_spec(self, spec_input, **kwargs):
        """
        Read spectrum data from file and initialize attributes
        
        Parameters
        ----------
        spec_input : str
            Filename, usually `[root].spec.fits`
        
        kwargs : dict
            Keyword arguments passed to `msaexp.spectrum.read_spectrum`
        
        """
        self.spec_input = spec_input
        if isinstance(spec_input, str):
            self.file = spec_input
        else:
            self.file = None
            
        self.spec = read_spectrum(spec_input, **kwargs)
        self.spec_wobs = self.spec['wave'].astype(np.float32)
        self.spec_R_fwhm = self.spec['R'].astype(np.float32)
        
        self.valid = np.isfinite(self.spec['flux']/self.spec['full_err'])


    @property
    def meta(self):
        return self.spec.meta


    def resample_eazy_template(self, template, z=0, scale_disp=1.0, velocity_sigma=100., fnu=True, nsig=4):
        """
        Smooth and resample an `eazy.templates.Template` object onto the observed
        wavelength grid of a spectrum
        
        Parameters
        ----------
        template : `eazy.templates.Template`
            Template object
        
        z : float
            Redshift
        
        scale_disp : float
            Factor multiplied to the tabulated spectral resolution before sampling
        
        velocity_sigma : float
            Gaussian velocity broadening factor, km/s
        
        fnu : bool
            Return resampled template in f-nu flux densities
        
        nsig : int
            Number of standard deviations to sample for the convolution
        
        Returns
        -------
        res : array-like
            Template flux density smoothed and resampled at the spectrum wavelengths
        
        """
        templ_wobs = template.wave.astype(np.float32)*(1+z)/1.e4
        if fnu:
            templ_flux = template.flux_fnu(z=z).astype(np.float32)
        else:
            templ_flux = template.flux_flam(z=z).astype(np.float32)
            
        
        res = self.resample_func(self.spec_wobs,
                                 self.spec_R_fwhm*scale_disp,
                                 templ_wobs,
                                 templ_flux,
                                 velocity_sigma=velocity_sigma,
                                 nsig=nsig)
        return res
    
    
    def emission_line(self, line_um, line_flux=1, scale_disp=1.0, velocity_sigma=100., nsig=4):
        """
        Make an emission line template - *deprecated in favor of*
        `~msaexp.spectrum.SpectrumSampler.fast_emission_line`
        
        Parameters
        ----------
        line_um : float
            Line center, microns
        
        line_flux : float
            Line normalization
        
        scale_disp : float
            Factor by which to scale the tabulated resolution FWHM curve
        
        velocity_sigma : float
            Velocity sigma width in km/s
        
        nsig : int
            Number of sigmas of the convolution kernel to sample
        
        Returns
        -------
        res : array-like
            Gaussian emission line sampled at the spectrum wavelengths
        """
        res = self.resample_func(self.spec_wobs,
                                 self.spec_R_fwhm*scale_disp, 
                                 self.xline*line_um,
                                 self.yline,
                                 velocity_sigma=velocity_sigma,
                                 nsig=nsig)
        
        return res*line_flux/line_um


    def fast_emission_line(self, line_um, line_flux=1, scale_disp=1.0, velocity_sigma=100.):
        """
        Make an emission line template with numerically correct pixel integration
        function
        
        Parameters
        ----------
        line_um : float
            Line center, microns
        
        line_flux : float
            Line normalization
        
        scale_disp : float
            Factor by which to scale the tabulated resolution FWHM curve
        
        velocity_sigma : float
            Velocity sigma width in km/s
        
        Returns
        -------
        res : array-like
            Gaussian emission line sampled at the spectrum wavelengths
        """
        res = self.sample_line_func(self.spec_wobs,
                                    self.spec_R_fwhm*scale_disp, 
                                    line_um,
                                    line_flux=line_flux,
                                    velocity_sigma=velocity_sigma, 
                                    )
        return res


    def bspline_array(self, nspline=13, log=False, by_wavelength=False, get_matrix=True):
        """
        Initialize bspline templates for continuum fits
        
        Parameters
        ----------
        nspline : int
            Number of spline functions to sample across the wavelength range
        
        log : bool
            Sample in log(wavelength)
        
        get_matrix : bool
            If true, return array data.  Otherwise, return template objects
        
        Returns
        -------
        bspl : array-like
            bspline data, depending on ``get_matrix``
        
        """
        # if get_matrix:
        #     if by_wavelength:
        #         bspl = utils.bspline_templates(wave=self.spec_wobs*1.e4,
        #                                degree=3,
        #                                df=nspline,
        #                                log=log,
        #                                get_matrix=get_matrix
        #                                )
        #     else:
        #         bspl = utils.bspline_templates(wave=np.arange(len(self.spec_wobs)),
        #                                degree=3,
        #                                df=nspline,
        #                                log=log,
        #                                get_matrix=get_matrix
        #                                )
        #
        #     bspl = bspl.T
        # else:
        #     if by_wavelength:
        #         bspl = utils.bspline_templates(wave=self.spec_wobs*1.e4,
        #                                degree=3,
        #                                df=nspline,
        #                                log=log,
        #                                get_matrix=get_matrix
        #                                )
        #     else:
        #         bspl = utils.bspline_templates(wave=np.arange(len(self.spec_wobs)),
        #                                degree=3,
        #                                df=nspline,
        #                                log=log,
        #                                get_matrix=get_matrix
        #                                )
        #         for t in bspl:
        #             bspl[t].wave = self.spec_wobs
        if by_wavelength:
            bspl = utils.bspline_templates(wave=self.spec_wobs*1.e4,
                                   degree=3,
                                   df=nspline,
                                   log=log,
                                   get_matrix=get_matrix
                                   )
        else:
            bspl = utils.bspline_templates(wave=np.arange(len(self.spec_wobs)),
                                   degree=3,
                                   df=nspline,
                                   log=log,
                                   get_matrix=get_matrix
                                   )

        if get_matrix:
            bspl = bspl.T
        else:
            for t in bspl:
                bspl[t].wave = self.spec_wobs
            
        return bspl


    def redo_1d_extraction(self, **kwargs):
        """
        Redo 1D extraction from 2D arrays with `msaexp.drizzle.make_optimal_extraction`
        
        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to `msaexp.drizzle.make_optimal_extraction`
        
        Returns
        -------
        output : `~msaexp.spectrum.SpectrumSampler`
            A new `~msaexp.spectrum.SpectrumSampler` object
        
        Examples
        --------
        
        .. plot::
            :include-source:
        
            # Compare 1D extractions
            from msaexp import spectrum
            import matplotlib.pyplot as plt

            sp = spectrum.SpectrumSampler('https://s3.amazonaws.com/msaexp-nirspec/extractions/ceers-ddt-v1/ceers-ddt-v1_prism-clear_2750_1598.spec.fits')

            fig, axes = plt.subplots(2,1,figsize=(8,5), sharex=True, sharey=True)

            # Boxcar extraction, center pixel +/- 2 pix
            ax = axes[0]
            new = sp.redo_1d_extraction(ap_radius=2, bkg_offset=-6)

            ax.plot(sp['wave'], sp['flux'], alpha=0.5, label='Original optimal extraction')
            ax.plot(new['wave'], new['aper_flux'], alpha=0.5, label='Boxcar, y = 23 ± 2')

            ax.grid()
            ax.legend()

            # Extractions above and below the center
            ax = axes[1]
            low = sp.redo_1d_extraction(ap_center=21, ap_radius=1)
            hi = sp.redo_1d_extraction(ap_center=25, ap_radius=1)

            ax.plot(low['wave'], low['aper_flux']*1.5, alpha=0.5, label='Below, y = 21 ± 1', color='b')
            ax.plot(hi['wave'], hi['aper_flux']*3, alpha=0.5, label='Above, y = 25 ± 1', color='r')

            ax.set_xlim(0.9, 5.3)
            ax.grid()
            ax.legend()

            ax.set_xlabel(r'$\lambda$')
            for ax in axes:
                ax.set_ylabel(r'$\mu\mathrm{Jy}$')

            fig.tight_layout(pad=1)
        """
        
        if isinstance(self.spec_input, pyfits.HDUList):
            out_hdul = drizzle.extract_from_hdul(self.spec_input, **kwargs)
        else:
            with pyfits.open(self.file) as hdul:
                out_hdul = drizzle.extract_from_hdul(hdul, **kwargs)
        
        output = SpectrumSampler(out_hdul)
        
        return output


    def drizzled_hdu_figure(self, **kwargs):
        """
        Run `msaexp.utils.drizzled_hdu_figure` on array data
        
        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to `msaexp.utils.drizzled_hdu_figure`
        
        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Spectrum figure
        
        """
        if isinstance(self.spec_input, pyfits.HDUList):
            fig = msautils.drizzled_hdu_figure(self.spec_input, **kwargs)
        else:
            with pyfits.open(self.file) as hdul:
                fig = msautils.drizzled_hdu_figure(hdul, **kwargs)
        
        return fig


def smooth_template_disp_eazy(templ, wobs_um, disp, z, velocity_fwhm=80, scale_disp=1.3, flambda=True, with_igm=True):
    """
    Smooth a template with a wavelength-dependent dispersion function.
    
    *NB:* Not identical to the preferred
    `~msaexp.spectrum.SpectrumSampler.resample_eazy_template`
    
    Parameters
    ----------
    templ : `eazy.template.Template`
        Template object
    
    wobs_um : array-like
        Target observed-frame wavelengths, microns
    
    disp : table
        NIRSpec dispersion table with columns ``WAVELENGTH``, ``R``
    
    z : float
        Target redshift
    
    velocity_fwhm : float
        Velocity dispersion FWHM, km/s
    
    scale_disp : float
        Scale factor applied to ``disp['R']``
    
    flambda : bool
        Return smoothed template in units of f_lambda or f_nu.
    
    Returns
    -------
    tsmooth : array-like
        Template convolved with spectral resolution + velocity dispersion.  
        Same length as `wobs_um`
    
    """
    dv = np.sqrt(velocity_fwhm**2 + (3.e5/disp['R']/scale_disp)**2)
    disp_ang = disp['WAVELENGTH']*1.e4
    dlam_ang = disp_ang*dv/3.e5/2.35
    
    def _lsf(wave):
        return np.interp(wave,
                         disp_ang,
                         dlam_ang,
                         left=dlam_ang[0], right=dlam_ang[-1],
                         )
    
    if hasattr(wobs_um,'value'):
        wobs_ang = wobs_um.value*1.e4
    else:
        wobs_ang = wobs_um*1.e4
        
    flux_model = templ.to_observed_frame(z=z,
                                         lsf_func=_lsf,
                                         clip_wavelengths=None,
                                         wavelengths=wobs_ang,
                                         smoothspec_kwargs={'fftsmooth':FFTSMOOTH},
                                         )
                                         
    if flambda:
        flux_model = np.squeeze(flux_model.flux_flam())
    else:
        flux_model = np.squeeze(flux_model.flux_fnu())
        
    return flux_model


def smooth_template_disp_sedpy(templ, wobs_um, disp, z, velocity_fwhm=80, scale_disp=1.3, flambda=True, with_igm=True):
    """
    Smooth a template with a wavelength-dependent dispersion function using 
    the `sedpy`/`prospector` LSF smoothing function
    
    Parameters
    ----------
    templ : `eazy.template.Template`
        Template object
    
    wobs_um : array-like
        Target observed-frame wavelengths, microns
    
    disp : table
        NIRSpec dispersion table with columns ``WAVELENGTH``, ``R``
    
    z : float
        Target redshift
    
    velocity_fwhm : float
        Velocity dispersion FWHM, km/s
    
    scale_disp : float
        Scale factor applied to ``disp['R']``
    
    flambda : bool
        Return smoothed template in units of f_lambda or f_nu.
    
    Returns
    -------
    tsmooth : array-like
        Template convolved with spectral resolution + velocity dispersion.  
        Same length as `wobs_um`
    """
    from sedpy.smoothing import smoothspec
    
    wobs = templ.wave*(1+z)
    trim = (wobs > wobs_um[0]*1.e4*0.95)
    trim &= (wobs < wobs_um[-1]*1.e4*1.05)
    
    if flambda:
        fobs = templ.flux_flam(z=z)#[wclip]
    else:
        fobs = templ.flux_fnu(z=z)#[wclip]
    
    if with_igm:
        fobs *= templ.igm_absorption(z)
    
    wobs = wobs[trim]
    fobs = fobs[trim]
    
    R = np.interp(wobs, disp['WAVELENGTH']*1.e4, disp['R'],
                  left=disp['R'][0], right=disp['R'][-1])*scale_disp
                  
    dv = np.sqrt(velocity_fwhm**2 + (3.e5/R)**2)
    dlam_ang = wobs*dv/3.e5/2.35
    
    def _lsf(wave):
        return np.interp(wave, wobs, dlam_ang)
    
    tsmooth = smoothspec(wobs, fobs,
                         smoothtype='lsf', lsf=_lsf,
                         outwave=wobs_um*1.e4,
                         fftsmooth=FFTSMOOTH,
                        )
    
    return tsmooth


def smooth_template_disp(templ, wobs_um, disp, z, velocity_fwhm=80, scale_disp=1.3, flambda=True, with_igm=True):
    """
    Smooth a template with a wavelength-dependent dispersion function
    
    Parameters
    ----------
    templ : `eazy.template.Template`
        Template object
    
    wobs_um : array-like
        Target observed-frame wavelengths, microns
    
    disp : table
        NIRSpec dispersion table with columns ``WAVELENGTH``, ``R``
    
    z : float
        Target redshift
    
    velocity_fwhm : float
        Velocity dispersion FWHM, km/s
    
    scale_disp : float
        Scale factor applied to ``disp['R']``
    
    flambda : bool
        Return smoothed template in units of f_lambda or f_nu.
    Returns
    -------
    tsmooth : array-like
        Template convolved with spectral resolution + velocity dispersion.  
        Same length as `wobs_um`
    """
    
    wobs = templ.wave*(1+z)/1.e4
    if flambda:
        fobs = templ.flux_flam(z=z)#[wclip]
    else:
        fobs = templ.flux_fnu(z=z)#[wclip]
    
    if with_igm:
        fobs *= templ.igm_absorption(z)
        
    disp_r = np.interp(wobs, disp['WAVELENGTH'], disp['R'])*scale_disp
    fwhm_um = np.sqrt((wobs/disp_r)**2 + (velocity_fwhm/3.e5*wobs)**2)
    sig_um = np.maximum(fwhm_um/2.35, 0.5*np.gradient(wobs))
    
    x = wobs_um[:,np.newaxis] - wobs[np.newaxis,:]
    gaussian_kernel = 1./np.sqrt(2*np.pi*sig_um**2)*np.exp(-x**2/2/sig_um**2)
    tsmooth = np.trapz(gaussian_kernel*fobs, x=wobs, axis=1)
    
    return tsmooth


SMOOTH_TEMPLATE_DISP_FUNC = smooth_template_disp_eazy


def fit_redshift(file='jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits', z0=[0.2, 10], zstep=None, eazy_templates=None, nspline=None, scale_disp=1.3, vel_width=100, Rline=None, is_prism=False, use_full_dispersion=False, ranges=None, sys_err=0.02, **kwargs):
    """
    Fit spectrum for the redshift
    
    Parameters
    ----------
    file : str
        Spectrum filename
    
    z0 : (float, float)
        Redshift range
    
    zstep : (float, float)
        Step sizes in `dz/(1+z)`
    
    eazy_templates : list, None
        List of `eazy.templates.Template` objects.  If not provided, just use 
        dummy spline continuum and emission line templates
    
    nspline : int
        Number of splines to use for dummy continuum
    
    scale_disp : float
        Scale factor of nominal dispersion files, i.e., `scale_disp > 1` 
        *increases* the spectral resolution
    
    vel_width : float
        Velocity width the emission line templates
    
    Rline : float
        Original spectral resolution used to sample the line templates
    
    is_prism : bool
        Is the spectrum from the prism?
    
    use_full_dispersion : bool
        Convolve `eazy_templates` with the full wavelength-dependent
        dispersion function
    
    ranges : list of tuples
        Wavelength ranges for the subplots
    
    sys_err : float
        Systematic uncertainty added in quadrature with nominal uncertainties
    
    Returns
    -------
    fig : Figure
        Diagnostic figure
    
    sp : `~astropy.table.Table`
        A copy of the 1D spectrum as fit with additional columns describing the 
        best-fit templates
    
    data : dict
        Fit metadata
    
    """
    import yaml
    def float_representer(dumper, value):
        text = '{0:.6f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
    
    yaml.add_representer(float, float_representer)
    
    #is_prism |= ('clear' in file)
    spec = read_spectrum(file, sys_err=sys_err, **kwargs)
    is_prism |= spec.grating in ['prism']
    
    if 'spec.fits' in file:
        froot = file.split('.spec.fits')[0]
    else:
        froot = file.split('.fits')[0]
    
    if zstep is None:
        if (is_prism):
            step0 = 0.002
            step1 = 0.0001
        else:
            step0 = 0.001
            step1 = 0.00002
    else:
        step0, step1 = zstep
        
    if Rline is None:
        if is_prism:
            Rline = 1000
        else:
            Rline = 5000
    
    # First pass
    zgrid = utils.log_zgrid(z0, step0)
    zg0, chi0 = fit_redshift_grid(file, zgrid=zgrid,
                                  line_complexes=False, 
                                  vel_width=vel_width,
                                  scale_disp=scale_disp, 
                                  eazy_templates=eazy_templates,
                                  Rline=Rline,
                                  use_full_dispersion=use_full_dispersion,
                                  sys_err=sys_err,
                                  **kwargs)

    zbest0 = zg0[np.argmin(chi0)]
    
    # Second pass
    zgrid = utils.log_zgrid(zbest0 + np.array([-0.005, 0.005])*(1+zbest0), 
                            step1)
    
    zg1, chi1 = fit_redshift_grid(file, zgrid=zgrid,
                                  line_complexes=False, 
                                  vel_width=vel_width,
                                  scale_disp=scale_disp, 
                                  eazy_templates=eazy_templates,
                                  Rline=Rline,
                                  use_full_dispersion=use_full_dispersion,
                                  sys_err=sys_err,
                                  **kwargs)
                                  
    zbest = zg1[np.argmin(chi1)]
    
    fz, az = plt.subplots(1,1,figsize=(6,4))
    az.plot(zg0, chi0)
    az.plot(zg1, chi1)
    
    az.set_ylim(chi1.min()-50, chi1.min() + 10**2)
    az.grid()
    az.set_xlabel('redshift')
    az.set_ylabel(r'$\chi^2$')
    az.set_title(os.path.basename(file))
    
    fz.tight_layout(pad=1)
    
    fz.savefig(froot+'.chi2.png')
    
    if is_prism:
        if ranges is None:
            ranges = [(3427, 5308), (6250, 9700)]
        if nspline is None:
            nspline = 41
    else:
        if ranges is None:
            ranges = [(3680, 4400), (4861-50, 5008+50), (6490, 6760)]
        if nspline is None:
            nspline = 23
    
    fig, sp, data = plot_spectrum(file, z=zbest, show_cont=True,
                              draws=100, nspline=nspline,
                              figsize=(16, 8), vel_width=vel_width,
                              ranges=ranges, Rline=Rline,
                              scale_disp=scale_disp,
                              eazy_templates=eazy_templates,
                              use_full_dispersion=use_full_dispersion,
                              sys_err=sys_err,
                              **kwargs)
    
    if eazy_templates is not None:
        spl_fig, sp2, spl_data = plot_spectrum(file, z=zbest, show_cont=True,
                              draws=100, nspline=nspline,
                              figsize=(16, 8), vel_width=vel_width,
                              ranges=ranges, Rline=Rline,
                              scale_disp=scale_disp,
                              eazy_templates=None,
                              use_full_dispersion=use_full_dispersion,
                              sys_err=sys_err,
                              **kwargs)
        
        for k in ['coeffs', 'covar', 'model', 'mline', 'fullchi2', 'contchi2','eqwidth']:
            if k in spl_data:
                data[f'spl_{k}'] = spl_data[k]
        
        spl_fig.savefig(froot+'.spl.png')
    
        sp['spl_model'] = sp2['model']
    
    sp['wave'].unit = u.micron
    sp['flux'].unit = u.microJansky
    
    sp.write(froot+'.spec.zfit.fits', overwrite=True)
    
    zdata = {}
    zdata['zg0'] = zg0.tolist()
    zdata['chi0'] = chi0.tolist()
    zdata['zg1'] = zg1.tolist()
    zdata['chi1'] = chi1.tolist()
    
    data['dchi2'] = float(np.nanmedian(chi0) - np.nanmin(chi0))
    
    for k in ['templates','spl_covar','covar']:
        if k in data:
            _ = data.pop(k)
            
    with open(froot+'.zfit.yaml', 'w') as fp:
        yaml.dump(zdata, stream=fp)
    
    with open(froot+'.yaml', 'w') as fp:
        yaml.dump(data, stream=fp)
    
    fig.savefig(froot+'.zfit.png')
    
    return fig, sp, data

H_RECOMBINATION_LINES = ['Ha+NII', 'Ha','Hb','Hg','Hd',
                         'PaA','PaB','PaG','PaD','Pa8',
                         'BrA','BrB','BrG','BrD']

def make_templates(sampler, z, bspl={}, eazy_templates=None, vel_width=100, broad_width=4000, broad_lines=[], scale_disp=1.3, use_full_dispersion=False, disp=None, grating='prism', halpha_prism=['Ha+NII'], oiii=['OIII'], o4363=[], sii=['SII'], lorentz=False, with_pah=True, **kwargs):
    """
    Generate fitting templates
    
    wobs : array
        Observed-frame wavelengths of the spectrum to fit, microns
    
    z : float
        Redshift
    
    bspl : dict
        Spline templates for dummy continuum
    
    eazy_templates : list
        Optional list of `eazy.templates.Template` template objects to use in 
        place of the spline + line templates
    
    vel_width : float
        Velocity width of the individual emission line templates
    
    halpha_prism : ['Ha+NII'], ['Ha','NII']
        Line template names to use for Halpha and [NII], i.e., ``['Ha+NII']`` 
        fits with a fixed line ratio and `['Ha','NII']` fits them separately 
        but with a fixed line ratio 6548:6584 = 1:3
    
    oiii : ['OIII'], ['OIII-4959','OIII-5007']
        Similar for [OIII]4959+5007, ``['OIII']`` fits as a doublet with fixed
        ratio 4959:5007 = 1:2.98 and ``['OIII-4949', 'OIII-5007']`` fits them
        independently.
    
    o4363 : [] or ['OIII-4363']
        How to fit [OIII]4363.
    
    sii : ['SII'], ['SII-6717','SII-6731']
        [SII] doublet
    
    lorentz : bool
        Use Lorentzian profile for lines
    
    Returns
    -------
    templates : list
        List of the computed template objects
    
    tline : array
        Boolean list of which templates are line components
    
    _A : (NT, NWAVE) array
        Design matrix of templates interpolated at `wobs`
    
    """
    from grizli import utils
    
    wobs = sampler.spec_wobs
    wrest = wobs/(1+z)*1.e4
    
    wmask = sampler.valid
    
    wmin = wobs[wmask].min()
    wmax = wobs[wmask].max()
    
    templates = []
    tline = []
    
    if eazy_templates is None:
        lw, lr = utils.get_line_wavelengths()
        
        _A = [bspl*1]
        for i in range(bspl.shape[0]):
            templates.append(f'spl {i}')
            tline.append(False)
            
        #templates = {}
        #for k in bspl:
        #    templates[k] = bspl[k]

        # templates = {}
        if grating in ['prism']:
            hlines = ['Hb', 'Hg', 'Hd']
            
            if z > 4:
                oiii = ['OIII-4959','OIII-5007']
                hene = ['HeII-4687', 'NeIII-3867','HeI-3889']
                o4363 = ['OIII-4363']
                
            else:
                #oiii = ['OIII']
                hene = ['HeI-3889']
                #o4363 = []
                
            #sii = ['SII']
            #sii = ['SII-6717', 'SII-6731']
            
            hlines += halpha_prism + ['NeIII-3968']
            fuv = ['OIII-1663']
            oii_7320 = ['OII-7325']
            extra = []
            
        else:
            hlines = ['Hb', 'Hg', 'Hd','H8','H9', 'H10', 'H11', 'H12']
            
            hene = ['HeII-4687', 'NeIII-3867']
            o4363 = ['OIII-4363']
            oiii = ['OIII-4959','OIII-5007']
            sii = ['SII-6717', 'SII-6731']
            hlines += ['Ha', 'NII-6549', 'NII-6584']
            hlines += ['H7', 'NeIII-3968']
            fuv = ['OIII-1663', 'HeII-1640', 'CIV-1549']
            oii_7320 = ['OII-7323', 'OII-7332']
            
            extra = ['HeI-6680', 'SIII-6314']
            
        line_names = []
        line_waves = []
        
        for l in [*hlines, *oiii, *o4363, 'OII',
                  *hene, 
                  *sii,
                  *oii_7320,
                  'ArIII-7138', 'ArIII-7753', 'SIII-9068', 'SIII-9531',
                  'OI-6302', 'PaD', 'PaG', 'PaB', 'PaA', 'HeI-1083',
                  'BrA','BrB','BrG','BrD','PfB','PfG','PfD','PfE',
                  'Pa8','Pa9','Pa10',
                  'HeI-5877', 
                  *fuv,
                  'CIII-1906', 'NIII-1750', 'Lya',
                  'MgII', 'NeV-3346', 'NeVI-3426',
                  'HeI-7065', 'HeI-8446',
                  *extra
                   ]:

            if l not in lw:
                continue
            
            lwi = lw[l][0]*(1+z)

            if lwi < wmin*1.e4:
                continue

            if lwi > wmax*1.e4:
                continue
            
            line_names.append(l)
            line_waves.append(lwi)
        
        so = np.argsort(line_waves)
        line_waves = np.array(line_waves)[so]
        
        for iline in so:
            l = line_names[iline]
            lwi = lw[l][0]*(1+z)

            if lwi < wmin*1.e4:
                continue

            if lwi > wmax*1.e4:
                continue
            
            # print(l, lwi, disp_r)

            name = f'line {l}'

            for i, (lwi0, lri) in enumerate(zip(lw[l], lr[l])):
                lwi = lwi0*(1+z)/1.e4
                if l in broad_lines:
                    vel_i = broad_width
                else:
                    vel_i = vel_width
                    
                line_i = sampler.fast_emission_line(lwi,
                                    line_flux=lri/np.sum(lr[l]),
                                    scale_disp=scale_disp,
                                    velocity_sigma=vel_i,)
                if i == 0:
                    line_0 = line_i
                else:
                    line_0 += line_i
                
            _A.append(line_0/1.e4)
            templates.append(name)
            tline.append(True)
        
        if with_pah:
            xpah = 3.3*(1+z)
            if ((xpah > wmin) & (xpah < wmax)) | (0):
                for t in PAH_TEMPLATES:
                    tp = PAH_TEMPLATES[t]
                    tflam = sampler.resample_eazy_template(tp,
                                            z=z,
                                            velocity_sigma=vel_width,
                                            scale_disp=scale_disp,
                                            fnu=False)
            
                    _A.append(tflam)
            
                    templates.append(t)
                    tline.append(True)
                    
                
        _A = np.vstack(_A)
        
        ll = wobs.value*1.e4/(1+z) < 1215.6

        igmz = igm.full_IGM(z, wobs.value*1.e4)
        _A *= np.maximum(igmz, 0.01)
        
    else:
        if isinstance(eazy_templates[0], dict) & (len(eazy_templates) == 2):
            # lw, lr dicts
            lw, lr = eazy_templates
            
            _A = [bspl*1]
            for i in range(bspl.shape[0]):
                templates.append(f'spl {i}')
                tline.append(False)
            
            for l in lw:
                name = f'line {l}'
                
                line_0 = None
                
                for i, (lwi0, lri) in enumerate(zip(lw[l], lr[l])):
                    lwi = lwi0*(1+z)/1.e4
                    
                    if lwi < wmin:
                        continue

                    elif lwi > wmax:
                        continue
                    
                    if l in broad_lines:
                        vel_i = broad_width
                    else:
                        vel_i = vel_width
                    
                    line_i = sampler.fast_emission_line(lwi,
                                        line_flux=lri/np.sum(lr[l]),
                                        scale_disp=scale_disp,
                                        velocity_sigma=vel_i,)
                    if line_0 is None:
                        line_0 = line_i
                    else:
                        line_0 += line_i
                
                if line_0 is not None:
                    _A.append(line_0/1.e4)
                    templates.append(name)
                    tline.append(True)
            
            _A = np.vstack(_A)
        
            ll = wobs.value*1.e4/(1+z) < 1215.6

            igmz = igm.full_IGM(z, wobs.value*1.e4)
            _A *= np.maximum(igmz, 0.01)
        
        elif len(eazy_templates) == 1:
            # Scale single template by spline
            t = eazy_templates[0]
            
            for i in range(bspl.shape[0]):
                templates.append(f'{t.name} spl {i}')
                tline.append(False)
            
            tflam = sampler.resample_eazy_template(t,
                                    z=z,
                                    velocity_sigma=vel_width,
                                    scale_disp=scale_disp,
                                    fnu=False)
            
            _A = np.vstack([bspl*tflam])
        
            ll = wobs.value*1.e4/(1+z) < 1215.6

            igmz = igm.full_IGM(z, wobs.value*1.e4)
            _A *= np.maximum(igmz, 0.01)
            
        else:
            templates = []
            tline = []
        
            _A = []
            for i, t in enumerate(eazy_templates):
                tflam = sampler.resample_eazy_template(t,
                                        z=z,
                                        velocity_sigma=vel_width,
                                        scale_disp=scale_disp,
                                        fnu=False)
            
                _A.append(tflam)
            
                templates.append(t.name)
                tline.append(False)
            
            _A = np.vstack(_A)
            
    return templates, np.array(tline), _A


def old_make_templates(wobs, z, wfull, wmask=None, bspl={}, eazy_templates=None, vel_width=100, broad_width=4000, broad_lines=[], scale_disp=1.3, use_full_dispersion=False, disp=None, grating='prism', halpha_prism=['Ha+NII'], oiii=['OIII'], o4363=[], sii=['SII'], lorentz=False, **kwargs):
    """
    Generate fitting templates
    
    wobs : array
        Observed-frame wavelengths of the spectrum to fit, microns
    
    z : float
        Redshift
    
    wfull : array
        Full wavelength array of the templates
    
    wmask : array-like
        Boolean mask on `wobs` for valid data
    
    bspl : dict
        Spline templates for dummy continuum
    
    eazy_templates : list
        Optional list of `eazy.templates.Template` template objects to use in 
        place of the spline + line templates
    
    vel_width : float
        Velocity width of the individual emission line templates
    
    halpha_prism : ['Ha+NII'], ['Ha','NII']
        Line template names to use for Halpha and [NII], i.e., ``['Ha+NII']`` 
        fits with a fixed line ratio and `['Ha','NII']` fits them separately 
        but with a fixed line ratio 6548:6584 = 1:3
    
    oiii : ['OIII'], ['OIII-4959','OIII-5007']
        Similar for [OIII]4959+5007, ``['OIII']`` fits as a doublet with fixed
        ratio 4959:5007 = 1:2.98 and ``['OIII-4949', 'OIII-5007']`` fits them
        independently.
    
    o4363 : [] or ['OIII-4363']
        How to fit [OIII]4363.
    
    sii : ['SII'], ['SII-6717','SII-6731']
        [SII] doublet
    
    lorentz : bool
        Use Lorentzian profile for lines
    
    Returns
    -------
    templates : list
        List of the computed template objects
    
    tline : array
        Boolean list of which templates are line components
    
    _A : (NT, NWAVE) array
        Design matrix of templates interpolated at `wobs`
    
    """
    from grizli import utils
    
    lw, lr = utils.get_line_wavelengths()
    
    wrest = wobs/(1+z)*1.e4
    
    if wmask is None:
        wmask = np.isfinite(wobs)
        
    wmin = wobs[wmask].min()
    wmax = wobs[wmask].max()
    
    if eazy_templates is None:
        templates = {}
        for k in bspl:
            templates[k] = bspl[k]

        # templates = {}
        if grating in ['prism']:
            hlines = ['Hb', 'Hg', 'Hd']
            
            if z > 4:
                oiii = ['OIII-4959','OIII-5007']
                hene = ['HeII-4687', 'NeIII-3867','HeI-3889']
                o4363 = ['OIII-4363']
                
            else:
                #oiii = ['OIII']
                hene = ['HeI-3889']
                #o4363 = []
                
            #sii = ['SII']
            #sii = ['SII-6717', 'SII-6731']
            
            hlines += halpha_prism + ['NeIII-3968']
            fuv = ['OIII-1663']
            oii_7320 = ['OII-7325']
            extra = []
            
        else:
            hlines = ['Hb', 'Hg', 'Hd','H8','H9', 'H10', 'H11', 'H12']
            
            hene = ['HeII-4687', 'NeIII-3867']
            o4363 = ['OIII-4363']
            oiii = ['OIII-4959','OIII-5007']
            sii = ['SII-6717', 'SII-6731']
            hlines += ['Ha', 'NII-6549', 'NII-6584']
            hlines += ['H7', 'NeIII-3968']
            fuv = ['OIII-1663', 'HeII-1640', 'CIV-1549']
            oii_7320 = ['OII-7323', 'OII-7332']
            
            extra = ['HeI-6680', 'SIII-6314']
            
        for l in [*hlines, *oiii, *o4363, 'OII',
                  *hene, 
                  *sii,
                  *oii_7320,
                  'ArIII-7138', 'ArIII-7753', 'SIII-9068', 'SIII-9531',
                  'OI-6302', 'PaD', 'PaG', 'PaB', 'PaA', 'HeI-1083',
                  'BrA','BrB','BrG','BrD','PfB','PfG','PfD','PfE',
                  'Pa8','Pa9','Pa10',
                  'HeI-5877', 
                  *fuv,
                  'CIII-1906', 'NIII-1750', 'Lya',
                  'MgII', 'NeV-3346', 'NeVI-3426',
                  'HeI-7065', 'HeI-8446',
                  *extra
                   ]:

            if l not in lw:
                continue
            
            lwi = lw[l][0]*(1+z)

            if lwi < wmin*1.e4:
                continue

            if lwi > wmax*1.e4:
                continue

            # print(l, lwi, disp_r)

            name = f'line {l}'

            for i, (lwi0, lri) in enumerate(zip(lw[l], lr[l])):
                lwi = lwi0*(1+z)
                disp_r = np.interp(lwi/1.e4, disp['WAVELENGTH'], 
                                   disp['R'])*scale_disp
                
                if l in broad_lines:
                    vel_i = broad_width
                else:
                    vel_i = vel_width
                    
                fwhm_ang = np.sqrt((lwi/disp_r)**2 + (vel_i/3.e5*lwi)**2)
                
                # print(f'Add component: {l} {lwi0} {lri}')

                if i == 0:
                    templates[name] = utils.SpectrumTemplate(wave=wfull,
                                                             flux=None,
                                                             central_wave=lwi,
                                                             fwhm=fwhm_ang,
                                                             name=name,
                                                             lorentz=lorentz)
                    templates[name].flux *= lri/np.sum(lr[l])
                else:
                    templates[name].flux += utils.SpectrumTemplate(wave=wfull,
                                                                   flux=None,
                                                            central_wave=lwi,
                                                            fwhm=fwhm_ang,
                                                            lorentz=lorentz,
                                            name=name).flux*lri/np.sum(lr[l])

        _, _A, tline = utils.array_templates(templates,
                                             max_R=10000,
                                             wave=wobs.astype(float)*1.e4,
                                             apply_igm=False)

        ll = wobs.value*1.e4/(1+z) < 1215.6

        igmz = igm.full_IGM(z, wobs.value*1.e4)
        _A *= np.maximum(igmz, 0.01)
    else:
        templates = {}
        if use_full_dispersion:
            _A = []
            tline = np.zeros(len(eazy_templates), dtype=bool)
            for i, t in enumerate(eazy_templates):
                templates[t.name] = 0.
                tflam = SMOOTH_TEMPLATE_DISP_FUNC(t,
                                             wobs,
                                             disp,
                                             z,
                                             velocity_fwhm=vel_width,
                                             scale_disp=scale_disp,
                                             flambda=True)
                _A.append(tflam)
                tline[i] = t.name.startswith('line ')
            
            _A = np.array(_A)
        else:
            for t in eazy_templates:
                tflam = t.flux_flam(z=z)
                templates[t.name] = utils.SpectrumTemplate(wave=t.wave,
                                                    flux=tflam, name=t.name)
    
            # ToDo: smooth with dispersion
            _, _A, tline = utils.array_templates(templates,
                                                 max_R=10000,
                                                 wave=wrest,
                                                 z=z, apply_igm=True)
    
            for i in range(len(templates)):
                _A[i,:] = nd.gaussian_filter(_A[i,:], 0.5)
    
    return templates, tline, _A
    
    
def fit_redshift_grid(file='jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits', zgrid=None, vel_width=100, bkg=None, scale_disp=1.3, nspline=27, line_complexes=True, Rline=1000, eazy_templates=None, use_full_dispersion=True, sys_err=0.02, use_aper_columns=False, **kwargs):
    """
    Fit redshifts on a grid
    
    Parameters
    ----------
    zgrid : array
        Redshifts to fit
    
    others : see `msaexp.spectrum.fit_redshift`
    
    Returns
    -------
    zgrid : array
        Copy of `zgrid`
    
    chi2 : array
        Chi-squared of the template fits at redshifts from `zgrid`
    
    """
    import time
    import os
    from tqdm import tqdm
    
    import astropy.io.fits as pyfits
    
    import numpy as np
    from grizli import utils
    import grizli.utils_c
    import astropy.units as u
    
    import eazy.igm
    
    import matplotlib.pyplot as plt
    
    #spec = read_spectrum(file, sys_err=sys_err)
    sampler = SpectrumSampler(file, **kwargs)
    spec = sampler.spec
        
    if (use_aper_columns > 0) & ('aper_flux' in spec.colnames):
        if ('aper_corr' in spec.colnames) & (use_aper_columns > 1):
            ap_corr = spec['aper_corr']*1
        else:
            ap_corr = 1
        
        flam = spec['aper_flux']*spec['to_flam']*ap_corr
        eflam = spec['aper_full_err']*spec['to_flam']*ap_corr
    else:
        flam = spec['flux']*spec['to_flam']
        eflam = spec['full_err']*spec['to_flam']
        
    wobs = spec['wave']
    mask = spec['valid']
    
    flam[~mask] = np.nan
    eflam[~mask] = np.nan
    
    #spline = utils.bspline_templates(wave=spec['wave']*1.e4, degree=3, df=nspline)
    bspl = sampler.bspline_array(nspline=nspline, get_matrix=True)
    
    chi2 = zgrid*0.
    
    #bspl = utils.bspline_templates(wave=spec['wave']*1.e4, degree=3, df=nspline) #, log=True)
    # w0 = utils.log_zgrid([spec['wave'].min()*1.e4,
    #                       spec['wave'].max()*1.e4], 1./Rline)
                                                  
    for iz, z in tqdm(enumerate(zgrid)):
        
        templates, tline, _A = make_templates(sampler, z,
                            bspl=bspl,
                            eazy_templates=eazy_templates,
                            vel_width=vel_width,
                            scale_disp=scale_disp,
                            use_full_dispersion=use_full_dispersion,
                            disp=spec.disp,
                            grating=spec.grating,
                            **kwargs,
                            )
        
        okt = _A[:,mask].sum(axis=1) > 0
        
        _Ax = _A[okt,:]/eflam
        _yx = flam/eflam
        
        if eazy_templates is None:
            _x = np.linalg.lstsq(_Ax[:,mask].T, 
                                 _yx[mask], rcond=None)
        else:
            _x = nnls(_Ax[:,mask].T, _yx[mask])
        
        coeffs = np.zeros(_A.shape[0])
        coeffs[okt] = _x[0]
        _model = _A.T.dot(coeffs)
        
        chi = (flam - _model) / eflam
        
        chi2_i = (chi[mask]**2).sum()
        # print(z, chi2_i)
        chi2[iz] = chi2_i
        
    return zgrid, chi2


def calc_uncertainty_scale(file=None, data=None, order=0, initial_mask=(0.2, 5), sys_err=0.02, fit_sys_err=False, method='bfgs', tol=None, student_df=10, update=True, verbose=True, **kwargs):
    """
    Compute a polynomial scaling of the spectrum uncertainties.  The procedure is to fit for
    coefficients of a polynomial multiplied to the `err` array of the spectrum such that 
    `(flux - model)/(err*scl)` residuals are `N(0,1)`
    
    Parameters
    ----------
    file : str
        Spectrum filename
    
    data : tuple
        Precomputed outputs from `msaexp.spectrum.plot_spectrum`
    
    order : int
        Degree of the correction polynomial
    
    initial_mask : (float, float)
        Masking for the fit initialization.  First parameter is zeroth-order 
        uncertainty scaling and the second parameter is the mask threshold of the 
        residuals
    
    sys_err : float
        Systematic component of the uncertainties
    
    fit_sys_err : bool
        Fit for adjusted ``sys_err`` parameter

    method : str
        Optimization method for `scipy.optimize.minimize`
    
    tol : None, float
        Tolerance parameter passed to `scipy.optimize.minimize`
    
    student_df : int, None
        If specified, calculate log likelihood of a `scipy.stats.t Student-t 
        distribution with ``df=student_df``.  Otherwise, calculate log-likelihood 
        of the `scipy.stats.normal` distribution.
    
    update : bool
        Update the global `msaexp.spectrum.SCALE_UNCERTAINTY` array with the fit result
    
    verbose : bool
        Print status messages.  If ``verbose > 1`` will also print status at each
        step of the optimization.
    
    kwargs : dict
        Keyword arguments for `msaexp.spectrum.plot_spectrum` if `data` not specified
    
    Returns
    -------
    spec : `~astropy.table.Table`
        The spectrum as fit
    
    escale : array
        The wavelength-dependent scaling of the uncertainties
    
    sys_err : float
        The systematic uncertainty used, fixed or adjusted depending on ``fit_sys_err``
    
    res : object
        Output from `scipy.optimize.minimize`
    
    """
    from scipy.stats import norm
    from scipy.stats import t as student
    
    from scipy.optimize import minimize
    
    global SCALE_UNCERTAINTY
    SCALE_UNCERTAINTY = 1.0
    
    if data is None:
        spec, spl = plot_spectrum(inp=file,
                                     eazy_templates=None,
                                     get_spl_templates=True,
                                     sys_err=sys_err,
                                     get_init_data=True,
                                     **kwargs
                                  )
    else:
        spec, spl = data
        
    # if 'full_err' in spec.colnames:
    #     _err = spec['full_err']
    # else:
    #     _err = spec['err']
    _err = spec['err']
    
    ok = (_err > 0) & (spec['flux'] != 0)
    ok &= np.isfinite(_err+spec['flux'])
    
    # print('escale: ', np.nanpercentile(spec['escale'], [10,50,90]))
    
    def objfun_scale_uncertainties(c, ok, ret):
        
        if fit_sys_err:
            _sys_err = c[0] / 100
            _coeffs = c[1:]
        else:
            _coeffs = c[:]
            _sys_err = sys_err
        
        err = 10**np.polyval(_coeffs, spec['wave'] - 3)*_err
        if 'escale' in spec.colnames:
            err *= spec['escale']

        _full_err = np.sqrt(err**2 + np.maximum(_sys_err*spec['flux'], 0)**2)
        
        _Ax = spl / _full_err
        _yx = spec['flux'] / _full_err
        _x = np.linalg.lstsq(_Ax[:,ok].T, _yx[ok], rcond=None)
        _model = spl.T.dot(_x[0])
        
        if ret == 1:
            return _sys_err, _full_err, _model
            
        if student_df is None:
            _lnp = norm.logpdf((spec['flux'] - _model)[ok],
                          loc=_model[ok]*0.,
                          scale=_full_err[ok]).sum()
            chi2 = -1*_lnp/2
        else:
            _lnp = student.logpdf((spec['flux'] - _model)[ok],
                          student_df,
                          loc=_model[ok]*0.,
                          scale=_full_err[ok]).sum()
            chi2 = -1*_lnp/2
        
        if 0:
            _resid = (spec['flux'] - _model)[ok] / _full_err[ok]
            chi2 = np.log(utils.nmad(_resid))**2
        
        if verbose > 1:
            cstr = ' '.join([f'{ci:6.2f}' for ci in c])
            print(f"{cstr}: {chi2:.6e}")
    
        return chi2

    if fit_sys_err:
        c0 = np.zeros(order+2)
        c0[0] = sys_err*100
    else:
        c0 = np.zeros(order+1)

    if initial_mask is not None:
        c0[-1] = initial_mask[0]
        _sys, _efit, _model = objfun_scale_uncertainties(c0, ok, 1)
        bad = np.abs((spec['flux'] - _model)/_efit) > initial_mask[1]
        
        msg = f'calc_uncertainty_scale: Mask additional {(bad & ok).sum()} pixels'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        
        ok &= ~bad

    # objfun_scale_uncertainties([0.0])
    # print('xxx: fit_sys_err', fit_sys_err, sys_err)
    
    #c0[-1] = np.log10(3)
    
    res = minimize(objfun_scale_uncertainties, c0, method=method, args=(ok, 0))
    _sys, _efit, _model = objfun_scale_uncertainties(res.x, ok, 1)
    spec.meta['calc_syse'] = _sys
    spec['calc_err'] = _efit
    spec['calc_model'] = _model
    spec['calc_valid'] = ok

    if fit_sys_err:
        sys_err, _coeffs = res.x[0]/100., res.x[1:]
    else:
        _coeffs = res.x[:]

    _resid = (spec['flux'] - _model)/_efit
    msg = f'calc_uncertainty_scale: sys_err = {sys_err:.4f}'
    msg += f'\ncalc_uncertainty_scale: coeffs = {_coeffs}'
    msg += f'\ncalc_uncertainty_scale: NMAD = {utils.nmad(_resid[ok]):.3f}'
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
    
    if update:

        msg = f'calc_uncertainty_scale: Set SCALE_UNCERTAINTY: {_coeffs}'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            
        SCALE_UNCERTAINTY = _coeffs
    
    return spec, 10**np.polyval(_coeffs, spec['wave'] - 3), sys_err, res


def setup_spectrum(file, **kwargs):
    """
    Deprecated, use `msaexp.spectrum.read_spectrum`
    """
    return read_spectrum(file, **kwargs)


def set_spec_sys_err(spec, sys_err=0.02):
    """
    Set `full_err` columns including a systematic component
    
        >>> full_err**2 = (err * escale)**2 + (sys_err * maximum(flux,0))**2
    
    """
    spec['full_err'] = np.sqrt((spec['err']*spec['escale'])**2 +
                                np.maximum(sys_err*spec['flux'], 0)**2)
    
    if 'aper_err' in spec.colnames:
        spec['aper_full_err'] = np.sqrt((spec['aper_err']*spec['escale'])**2 +
                                   np.maximum(sys_err*spec['aper_flux'],0)**2)
        
    spec.meta['sys_err'] = sys_err


def read_spectrum(inp, spectrum_extension='SPEC1D', sys_err=0.02, err_mask=(10,0.5), err_median_filter=[11, 0.8], **kwargs):
    """
    Read a spectrum and apply flux and/or uncertainty scaling
    
    Flux scaling `corr` is applied if there are `POLY[i]` keywords in the spectrum 
    metadata, with
    
    .. code-block:: python
        :dedent:
        
        >>> coeffs = [header[f'POLY{i}'] for i in range(order+1)]
        >>> corr = np.polyval(coeffs, np.log(spec['wave']*1.e4))
    
    Parameters
    ----------
    inp : str or `~astropy.io.fits.HDUList`
        Fits filename of a file that includes a `~astropy.io.fits.BinTableHDU` table of 
        an extracted spectrum.  Alternatively, can be an `~astropy.io.fits.HDUList`
        itself
    
    spectrum_extension : str
        Extension name of 1D spectrum in file or HDUList input
    
    sys_err : float
        Systematic uncertainty added in quadrature with `err` array
    
    err_mask : float, float or None
        Mask pixels where ``err < np.percentile(err[err > 0], err_mask[0])*err_mask[1]``
    
    err_median_filter : int, float or None
        Mask pixels where
        ``err < nd.median_filter(err, err_median_filter[0])*err_median_filter[1]``
    
    Returns
    -------
    spec : `~astropy.table.Table`
        Spectrum table.  Existing columns in `file` should be
            
            - ``wave`` : observed-frame wavelength, microns
            - ``flux`` : flux density, `~astropy.units.microJansky`
            - ``err`` : Uncertainty on ```flux```

        Columns calculated here are
        
            - ``corr`` : flux scaling
            - ``escale`` : extra scaling of uncertainties
            - ``full_err`` : Full uncertainty including `sys_err`
            - ``R`` : spectral resolution
            - ``valid`` : Data are valid
    
    """
    global SCALE_UNCERTAINTY
    
    import scipy.ndimage as nd
    
    if isinstance(inp, str):
        if 'fits' in inp:
            with pyfits.open(inp) as hdul:
                if spectrum_extension in hdul:
                    spec = utils.read_catalog(hdul[spectrum_extension])
                else:
                    spec = utils.read_catalog(inp)
        else:
            spec = utils.read_catalog(inp)
            
    elif isinstance(inp, pyfits.HDUList):
        if spectrum_extension in inp:
            spec = utils.read_catalog(inp[spectrum_extension])
        else:
            msg = f'{spectrum_extension} extension not found in HDUList input'
            raise ValueError(msg)
    else:
        spec = utils.read_catalog(inp)

    if 'POLY0' in spec.meta:
        pc = []
        for pi in range(10):
            if f'POLY{pi}' in spec.meta:
                pc.append(spec.meta[f'POLY{pi}'])
        
        corr = np.polyval(pc, np.log(spec['wave']*1.e4))
        spec['flux'] *= corr
        spec['err'] *= corr
        spec['corr'] = corr
    else:
        spec['corr'] = 1.

    if 'escale' not in spec.colnames:  
        if hasattr(SCALE_UNCERTAINTY,'__len__'):
            if len(SCALE_UNCERTAINTY) < 6:
                spec['escale'] = 10**np.polyval(SCALE_UNCERTAINTY, spec['wave'])
            elif len(SCALE_UNCERTAINTY) == len(spec):
                spec['escale'] = SCALE_UNCERTAINTY
        else:
            spec['escale'] = SCALE_UNCERTAINTY
            # print('xx scale scalar', SCALE_UNCERTAINTY)
    
    for c in ['flux','err']:
        if hasattr(spec[c], 'filled'):
            spec[c] = spec[c].filled(0)
        
    valid = np.isfinite(spec['flux']+spec['err'])
    valid &= spec['err'] > 0
    valid &= spec['flux'] != 0 
    
    if (valid.sum() > 0) & (err_mask is not None):
        _min_err = np.nanpercentile(spec['err'][valid], err_mask[0])*err_mask[1]
        valid &= spec['err'] > _min_err
        
    if err_median_filter is not None:
        med = nd.median_filter(spec['err'][valid], err_median_filter[0])
        medi = np.interp(spec['wave'], spec['wave'][valid], med, left=0, right=0)
        valid &= spec['err'] > err_median_filter[1]*medi
        
    set_spec_sys_err(spec, sys_err=sys_err)
    
    # spec['full_err'] = np.sqrt((spec['err']*spec['escale'])**2 +
    #                             np.maximum(sys_err*spec['flux'], 0)**2)
    #
    # if 'aper_err' in spec.colnames:
    #     spec['aper_full_err'] = np.sqrt((spec['aper_err']*spec['escale'])**2 +
    #                                np.maximum(sys_err*spec['aper_flux'],0)**2)
    #
    # spec.meta['sys_err'] = sys_err
    
    spec['full_err'][~valid] = 0
    spec['flux'][~valid] = 0.
    spec['err'][~valid] = 0.
    
    spec['valid'] = valid
    
    grating = spec.meta['GRATING'].lower()
    _filter = spec.meta['FILTER'].lower()
    
    _data_path = os.path.dirname(__file__)
    disp = utils.read_catalog(f'{_data_path}/data/jwst_nirspec_{grating}_disp.fits')
    
    spec.disp = disp
    
    spec['R'] = np.interp(spec['wave'], disp['WAVELENGTH'], disp['R'],
                          left=disp['R'][0], right=disp['R'][-1])
    
    spec.grating = grating
    spec.filter = _filter
    
    flam_unit = 1.e-20*u.erg/u.second/u.cm**2/u.Angstrom
        
    um = spec['wave'].unit
    if um is None:
        um = u.micron
    
    spec.equiv = u.spectral_density(spec['wave'].data*um)
    
    spec['to_flam'] = (1*spec['flux'].unit).to(flam_unit, equivalencies=spec.equiv).value
    spec.meta['flamunit'] = flam_unit.unit
    
    spec.meta['fluxunit'] = spec['flux'].unit
    spec.meta['waveunit'] = spec['wave'].unit
    
    spec['wave'] = spec['wave'].value
    spec['flux'] = spec['flux'].value
    spec['err'] = spec['err'].value
    
    return spec


def plot_spectrum(inp='jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits', z=9.505, vel_width=100, bkg=None, scale_disp=1.3, nspline=27, show_cont=True, draws=100, figsize=(16, 8), ranges=[(3650, 4980)], Rline=1000, full_log=False, write=False, eazy_templates=None, use_full_dispersion=True, get_init_data=False, get_spl_templates=False, scale_uncertainty_kwargs=None, plot_unit=None, spline_single=True, sys_err=0.02, return_fit_results=False, use_aper_columns=False, label=None, **kwargs):
    """
    Make a diagnostic figure
    
    Parameters
    ----------
    ...
    return_fit_results : bool
        Just return the fit results - 
        ``templates, coeffs, flam, eflam, _model, mask, full_chi2``
    
    """
    global SCALE_UNCERTAINTY
    
    lw, lr = utils.get_line_wavelengths()
        
    if isinstance(inp, str):
        sampler = SpectrumSampler(inp, sys_err=sys_err, **kwargs)
        file = inp
    elif isinstance(inp, pyfits.HDUList):
        sampler = SpectrumSampler(inp, sys_err=sys_err, **kwargs)
        file = None
    else:
        file = None
        sampler = inp
    
    if (label is None) & (file is not None):
        label = os.path.basename(file)
    
    spec = sampler.spec
        
    if (use_aper_columns > 0) & ('aper_flux' in spec.colnames):
        if ('aper_corr' in spec.colnames) & (use_aper_columns > 1):
            ap_corr = spec['aper_corr']*1
        else:
            ap_corr = 1
        
        flux_column = 'aper_flux'
        err_column = 'aper_full_err'
        
        #flam = spec['aper_flux']*spec['to_flam']*ap_corr
        #eflam = spec['aper_full_err']*spec['to_flam']*ap_corr
    else:
        
        flux_column = 'flux'
        err_column = 'full_err'
        
        #flam = spec['flux']*spec['to_flam']
        #eflam = spec['full_err']*spec['to_flam']
        ap_corr = 1.
    
    flam = spec[flux_column] * spec['to_flam']*ap_corr
    eflam = spec[err_column] * spec['to_flam']*ap_corr
    
    wrest = spec['wave']/(1+z)*1.e4
    wobs = spec['wave']
    mask = spec['valid']
    
    flam[~mask] = np.nan
    eflam[~mask] = np.nan
    
    bspl = sampler.bspline_array(nspline=nspline, get_matrix=True)
                                   
    w0 = utils.log_zgrid([spec['wave'].min()*1.e4,
                          spec['wave'].max()*1.e4], 1./Rline)
    
    templates, tline, _A = make_templates(sampler, z,
                            bspl=bspl,
                        eazy_templates=eazy_templates,
                        vel_width=vel_width,
                        scale_disp=scale_disp,
                        use_full_dispersion=use_full_dispersion,
                        disp=spec.disp,
                        grating=spec.grating,
                        **kwargs,
                        )
    
    if get_init_data:
        return spec, _A
    
    if scale_uncertainty_kwargs is not None:
        _, escl, _sys_err, _ = calc_uncertainty_scale(file=None,
                                            data=(spec, _A),
                                            sys_err=sys_err,
                                            **scale_uncertainty_kwargs)
        # eflam *= escl
        spec['escale'] *= escl
        set_spec_sys_err(spec, sys_err=_sys_err)
        eflam = spec[err_column] * spec['to_flam'] * ap_corr
        
    okt = _A[:,mask].sum(axis=1) > 0
    
    _Ax = _A[okt,:]/eflam
    _yx = flam/eflam
    
    if eazy_templates is None:
        _x = np.linalg.lstsq(_Ax[:,mask].T, 
                             _yx[mask], rcond=None)
    else:
        _x = nnls(_Ax[:,mask].T, _yx[mask])
    
    coeffs = np.zeros(_A.shape[0])
    coeffs[okt] = _x[0]
    
    _model = _A.T.dot(coeffs)
    _mline = _A.T.dot(coeffs*tline)
    _mcont = _model - _mline
    
    full_chi2 = ((flam - _model)**2/eflam**2)[mask].sum()
    cont_chi2 = ((flam - _mcont)**2/eflam**2)[mask].sum()
    
    if return_fit_results:
        return templates, coeffs, flam, eflam, _model, mask, full_chi2
    
    try:
        oktemp = okt & (coeffs != 0)
            
        AxT = (_A[oktemp,:]/eflam)[:,mask].T
    
        covar_i = utils.safe_invert(np.dot(AxT.T, AxT))
        covar = utils.fill_masked_covar(covar_i, oktemp)
        covard = np.sqrt(covar.diagonal())
            
        has_covar = True
    except:
        has_covar = False
        covard = coeffs*0.
        N = len(templates)
        covar = np.eye(N, N)
    
    print(f'\n# line flux err\n# flux x 10^-20 erg/s/cm2')
    if label is not None:
        print(f'# {label}')
    
    print(f'# z = {z:.5f}\n# {time.ctime()}')
    
    cdict = {}
    eqwidth = {}
    
    for i, t in enumerate(templates):
        cdict[t] = [float(coeffs[i]), float(covard[i])]
        if t.startswith('line '):
            lk = t.split()[-1]
            
            # Equivalent width:
            # coeffs, line fluxes are in units of 1e-20 erg/s/cm2
            # _mcont, continuum model is in units of 1-e20 erg/s/cm2/A
            # so observed-frame equivalent width is roughly
            # eqwi = coeffs[i] / _mcont[ wave_obs[i] ]
            
            if lk in lw:
                lwi = lw[lk][0]*(1+z)/1.e4
                continuum_i = np.interp(lwi, spec['wave'], _mcont)
                eqwi = coeffs[i]/continuum_i
            else:
                eqwi = np.nan
            
            eqwidth[t] = [float(eqwi)]
            
            print(f'{t:>20}   {coeffs[i]:8.1f} ± {covard[i]:8.1f} (EW={eqwi:9.1f})')
            
            
    if 'srcra' not in spec.meta:
        spec.meta['srcra'] = 0.0
        spec.meta['srcdec'] = 0.0
        spec.meta['srcname'] = 'unknown'
    
    spec['model'] = _model/spec['to_flam']
    spec['mline'] = _mline/spec['to_flam']
    
    data = {'z': float(z),
            'file':file,
            'label':label,
            'ra': float(spec.meta['srcra']),
            'dec': float(spec.meta['srcdec']),
            'name': str(spec.meta['srcname']),
            'wmin':float(spec['wave'][mask].min()),
            'wmax':float(spec['wave'][mask].max()),
            'coeffs':cdict,
            'covar':covar.tolist(),
            'wave': [float(m) for m in spec['wave']],
            'flux': [float(m) for m in spec['flux']],
            'err': [float(m) for m in spec['err']],
            'escale': [float(m) for m in spec['escale']],
            'model': [float(m) for m in _model/spec['to_flam']],
            'mline':[float(m) for m in _mline/spec['to_flam']],
            'templates':templates, 
            'dof': int(mask.sum()), 
            'fullchi2': float(full_chi2), 
            'contchi2': float(cont_chi2),
            'eqwidth': eqwidth,
           }
            
    for k in ['z','wmin','wmax','dof','fullchi2','contchi2']:
        spec.meta[k] = data[k]
        
    #fig, axes = plt.subplots(len(ranges)+1,1,figsize=figsize)
    if len(ranges) > 0:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(2, len(ranges), figure=fig)
        axes = []
        for i, _ra in enumerate(ranges):
            axes.append(fig.add_subplot(gs[0,i]))
    
        axes.append(fig.add_subplot(gs[1,:]))
        
    else:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        axes = [ax]
        
    _Acont = (_A.T*coeffs)[mask,:][:,:nspline]
    _Acont[_Acont < 0.001*_Acont.max()] = np.nan
    
    if (draws is not None) & has_covar:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu = np.random.multivariate_normal(coeffs[oktemp], covar_i, size=draws)
            
        #print('draws', draws, mu.shape, _A.shape)
        mdraws = _A[oktemp,:].T.dot(mu.T)
    else:
        mdraws = None
    
    if plot_unit is not None:
        unit_conv = (1*spec.meta['flamunit']).to(plot_unit,
                                   equivalencies=spec.equiv).value
    else:
        unit_conv = np.ones(len(wobs))
        
    for ax in axes:
        if 1:
            ax.errorbar(wobs, flam*unit_conv, eflam*unit_conv,
                        marker='None', linestyle='None',
                        alpha=0.5, color='k', ecolor='k', zorder=100)

        ax.step(wobs, flam*unit_conv, color='k', where='mid', lw=1, alpha=0.8)
        # ax.set_xlim(3500, 5100)

        #ax.plot(_[1]['templz']/(1+z), _[1]['templf'])
        
        ax.step(wobs[mask], (_mcont*unit_conv)[mask],
                color='pink', alpha=0.8, where='mid')
        ax.step(wobs[mask], (_model*unit_conv)[mask],
                color='r', alpha=0.8, where='mid')
        
        cc = utils.MPL_COLORS
        for w, c in zip([3727, 4980, 6565, 9070, 9530, 1.094e4, 1.282e4, 
                         1.875e4], 
                    [cc['purple'], cc['b'], cc['g'], 'darkred', 'darkred', 
                     cc['pink'], cc['pink'], cc['pink']]):
            wz = w*(1+z)/1.e4
            dw = 70*(1+z)/1.e4
            ax.fill_between([wz-dw, wz+dw], [0,0], [100,100], 
                            color=c, alpha=0.07, zorder=-100)
                        
            
        if mdraws is not None:
            ax.step(wobs[mask], (mdraws.T*unit_conv).T[mask,:],
                    color='r', alpha=np.maximum(1./draws, 0.02), zorder=-100, where='mid')

        if show_cont:
            ax.plot(wobs[mask], (_Acont.T*unit_conv[mask]).T,
                    color='olive', alpha=0.3)
            
        ax.fill_between(ax.get_xlim(), [-100, -100], [0, 0], color='0.8', 
                        alpha=0.5, zorder=-1)

        ax.fill_betweenx([0, 100], [0,0], [1215.67*(1+z)/1.e4]*2, 
                         color=utils.MPL_COLORS['orange'], alpha=0.2,
                         zorder=-1)
    
        ax.grid()

    # axes[0].set_xlim(1000, 2500)
    # ym = 0.15; axes[0].set_ylim(-0.1*ym, ym)
    
    for i, r in enumerate(ranges):
        axes[i].set_xlim(*[ri*(1+z)/1.e4 for ri in r])
        # print('xxx', r)
        
    if spec.filter == 'clear':
        axes[-1].set_xlim(0.6, 5.29)
        axes[-1].xaxis.set_minor_locator(MultipleLocator(0.1))
        axes[-1].xaxis.set_major_locator(MultipleLocator(0.5))
    elif spec.filter == 'f070lp':
        axes[-1].set_xlim(0.69, 1.31)
        axes[-1].xaxis.set_minor_locator(MultipleLocator(0.02))
    elif spec.filter == 'f100lp':
        axes[-1].set_xlim(0.99, 1.91)
        axes[-1].xaxis.set_minor_locator(MultipleLocator(0.02))
        axes[-1].xaxis.set_major_locator(MultipleLocator(0.1))
    elif spec.filter == 'f170lp':
        axes[-1].set_xlim(1.69, 3.21)
    elif spec.filter == 'f290lp':
        axes[-1].set_xlim(2.89, 5.31)
    else:
        axes[-1].set_xlim(wrest[mask].min(), wrest[mask].max())
    
    axes[-1].set_xlabel(f'obs wavelenth, z = {z:.5f}')
    
    #axes[0].set_title(os.path.basename(file))
    
    for ax in axes:
        xl = ax.get_xlim()
        ok = wobs > xl[0]
        ok &= wobs < xl[1]
        ok &= np.abs(wrest-5008) > 100
        ok &= np.abs(wrest-6564) > 100
        ok &= mask
        if ok.sum() == 0:
            ax.set_visible(False)
            continue
        
        ymax = np.maximum((_model*unit_conv)[ok].max(), 10*np.median((eflam*unit_conv)[ok]))
        
        ymin = np.minimum(-0.1*ymax, -3*np.median((eflam*unit_conv)[ok]))
        ax.set_ylim(ymin, ymax*1.3)
        # print(xl, ymax)
    
    if ok.sum() > 0:
        if (np.nanmax((flam/eflam)[ok]) > 20) & (full_log):
            ax.set_ylim(0.005*ymax, ymax*5)
            ax.semilogy()
        
    if len(axes) > 0:
        gs.tight_layout(fig, pad=0.8)
    else:
        fig.tight_layout(pad=0.8)
    
    if label is not None:
        fig.text(0.015*12./12, 0.005, f'{label}',
             ha='left', va='bottom',
             transform=fig.transFigure, fontsize=8)
    
    fig.text(1-0.015*12./12, 0.005, time.ctime(),
             ha='right', va='bottom',
             transform=fig.transFigure, fontsize=6)
    
    
    return fig, spec, data


DEFAULT_FNUMBERS = [239, 205, # F814W, F160W
                    362, 363, 364, 365, 366, 370, 371, 375, 376, 377, # NRC BB
                    379, 380, 381, 382, 383, 384, 385, 386, # NRC MB
                    ]
                    
DEFAULT_REST_FNUMBERS = [120, 121, # GALEX
                         218, 219, 270, 271, 272, 274, # FUV
                         153, 154, 155, # UBV
                         156, 157, 158, 159, 160, # SDSS ugriz
                         161, 162, 163, # 2MASS JHK
                         414, 415, 416, # ugi Antwi-Danso 2022
                         ]

DEFAULT_SCALE_KWARGS = dict(order=0, sys_err=0.02,
                            nspline=31, scale_disp=1.3, vel_width=100
                            )

DLA_KWS = dict(wrange=[1180., 1350], slope_filters=[270, 271],
               filter_fraction_threshold=0.1, RES=None, make_plot=False)
               
def do_integrate_filters(file, z=0, RES=None, fnumbers=DEFAULT_FNUMBERS, rest_fnumbers=DEFAULT_REST_FNUMBERS, scale_kwargs=DEFAULT_SCALE_KWARGS, dla_kwargs=DLA_KWS):
    """
    Integrate a spectrum through a list of filter bandpasses
    
    Parameters
    ----------
    file : str
        Spectrum filename
    
    z : float
        Redshift
    
    RES : `eazy.filters.FilterFile`
        Container of filter bandpasses
    
    fnumbers : list
        List of observed-frame ``f_numbers``
    
    rest_fnumbers : list
        List of rest-frame ``f_numbers`` to evaluate at ``z``
    
    scale_kwargs : dict, None
        If provided, initialize the spectrum by first passing through
        `msaexp.spectrum.calc_uncertainty_scale`
    
    Returns
    -------
    fdict : dict
        "horizontal" dictionary with keys for each separate filter
    
    sed : `~astropy.table.Table`
        "vertical" table of the integrated flux densities
    
    """
    import eazy.filters
    
    global SCALE_UNCERTAINTY
    SCALE_UNCERTAINTY = 1.
    
    if RES is None:
        RES = eazy.filters.FilterFile(path=None)
    
    if scale_kwargs is not None:
        # Initialize spectrum rescaling uncertainties
        spec, escl, _sys_err, _ = calc_uncertainty_scale(file, z=z, **scale_kwargs)
        spec['escale'] *= escl
        set_spec_sys_err(spec, sys_err=_sys_err)
        
    else:
        # Just read the catalog
        spec = utils.read_catalog(file)
    
    rows = []
    fdict = {'file':file, 'z':z}
    
    # Observed-frame filters
    for fn in fnumbers:
        obs_flux = integrate_spectrum_filter(spec, RES[fn], z=0)
        rows.append([RES[fn].name, RES[fn].pivot/1.e4, fn, 0] + list(obs_flux))

        for c, v in zip(['valid', 'frac', 'flux', 'err', 'full_err'], obs_flux):
            fdict[f'obs_{fn}_{c}'] = v
    
    for fn in rest_fnumbers:
        rest_flux = integrate_spectrum_filter(spec, RES[fn], z=z)
        rows.append([RES[fn].name, RES[fn].pivot*(1+z)/1.e4, fn, z] + list(rest_flux))

        for c, v in zip(['valid', 'frac', 'flux', 'err', 'full_err'], rest_flux):
            fdict[f'rest_{fn}_{c}'] = v

    sed = utils.GTable(names=['name', 'pivot','f_number', 'z',
                              'valid', 'frac', 'flux', 'err', 'full_err'],
                       rows=rows)
    
    if dla_kwargs is not None:
        dla_kwargs['RES'] = RES
        dla_kwargs['z'] = z
        
        _ = measure_dla_eqw(spec, **dla_kwargs)
        beta, beta_unc, ndla, dla_value, dla_unc, _fig = _
        
        fdict['beta'] = beta
        fdict['beta_unc'] = beta_unc
        fdict['ndla'] = ndla
        fdict['dla_value'] = dla_value
        fdict['dla_unc'] = dla_unc
    
    return fdict, sed


def integrate_spectrum_filter(spec, filt, z=0, filter_fraction_threshold=0.1):
    """Integrate spectrum data through a filter bandpass

    Parameters
    ----------
    spec : `~astropy.table.Table`
        Spectrum data with columns ``wave`` (microns), ``flux`` and ``err``
        [``full_err``] (fnu) and ``valid``

    filt : `~eazy.filters.FilterDefinition`
        Filter bandpass object

    z : float
        Redshift

    filter_fraction_threshold : float
        Minimum allowed ``filter_fraction``, i.e. for filters that overlap with the 
        observed spectrum
        
    Returns
    -------
    npix : int
        Number of "valid" pixels
        
    filter_fraction : float
        Fraction of the integrated bandpass that falls on valid spectral wavelength bins

    filter_flux : float
        Integrated flux density, in units of ``spec['flux']``

    filter_err : float
        Propagated uncertainty from ``spec['err']``

    filter_full_err : float
        Propagated uncertianty from ``spec['full_err']``, if available.  Returns -1 if 
        not available.
    
    """
    
    # Interpolate bandpass to wavelength grid
    filter_int = np.interp(spec['wave'],
                           filt.wave/1.e4*(1+z),
                           filt.throughput,
                           left=0.,
                           right=0.)

    if 'valid' in spec.colnames:
        valid = spec['valid']
    else:
        valid = (spec['err'] > 0) & np.isfinite(spec['err'] + spec['flux'])

    if valid.sum() < 2:
        return (valid.sum(), 0., 0., -1., -1.)

    # Full filter normalization
    filt_norm_full = np.trapz(filt.throughput/(filt.wave/1.e4), filt.wave/1.e4)

    # Normalization of filter sampled by the spectrum
    filt_norm = np.trapz( (filter_int / spec['wave'])[valid], spec['wave'][valid])

    filter_fraction = filt_norm / filt_norm_full

    if filter_fraction < filter_fraction_threshold:
        return (valid.sum(), filter_fraction, 0., -1., -1.)
        
    ## Integrals

    # Trapezoid rule steps
    trapz_dx = utils.trapz_dx(spec['wave'])

    # Integrated flux
    fnu_flux = ( (filter_int / filt_norm * trapz_dx *
                  spec['flux'] / spec['wave'])[valid] ).sum()

    # Propagated uncertainty
    fnu_err = np.sqrt(( (filter_int / filt_norm * trapz_dx *
                         spec['err'] / spec['wave'])[valid]**2 ).sum())

    if 'full_err' in spec.colnames:
        fnu_full_err = np.sqrt(( (filter_int / filt_norm * trapz_dx *
                                  spec['full_err'] / spec['wave'])[valid]**2 ).sum())
    else:
        fnu_full_err = -1.
        
    return valid.sum(), filter_fraction, fnu_flux, fnu_err, fnu_full_err


def measure_dla_eqw(spec, z=0, wrange=[1180., 1350], slope_filters=[270,271], filter_fraction_threshold=0.1, RES=None, make_plot=False):
    """
    Measure DLA parameter (Heintz+24)
    
    >>> DLA = Integrate(1 - Fobs/Fcont, dlam)
    
    over a limited range near Ly-alpha
    
    Parameters
    ----------
    spec : `~astropy.table.Table`
        Spectrum data with columns ``wave`` (microns), ``flux`` and ``err``
        [``full_err``] (fnu) and ``valid``

    z : float
        Redshift
    
    wrange : (float, float)
        Wavelength range for the integral
    
    slope_filters : (int, int)
        Filter indices of two filters to define the UV slope.  The default values of
        ``slope_filters = (270, 271)`` fits the UV slope beta between rest-frame 
        1400--1700 Angstroms.
    
    RES : `eazy.filters.FilterFile`
        Container of filter bandpasses

    make_plot : bool
        Make a diagnostic plot
    
    Returns
    -------
    beta : float
        Derived UV slope ``flam = lam**beta``
    
    beta_unc : float
        Propagated uncertainty on ``beta``
    
    ndla : int
        Number of wavelength pixels satisfying ``wrange``
    
    dla_value : float
        Integrated DLA parameter
    
    dla_unc : float
        Propagated uncertainty on ``dla_value``

    fig : `~matplotlib.Figure`, None
        Figure if ``make_plot=True``.
    
    """
    if RES is None:
        RES = eazy.filters.FilterFile(path=None)
    
    f0 = integrate_spectrum_filter(spec,
                                   RES[slope_filters[0]],
                                   z=z,
                                   filter_fraction_threshold=filter_fraction_threshold)
    
    w0 = RES[slope_filters[0]].pivot / 1.e4
    
    f1 = integrate_spectrum_filter(spec,
                                   RES[slope_filters[1]],
                                   z=z,
                                   filter_fraction_threshold=filter_fraction_threshold)
    
    w1 = RES[slope_filters[1]].pivot / 1.e4
    
    if (f0[1] < filter_fraction_threshold) | (f1[1] < filter_fraction_threshold):
        return (-1, -1, 0, -1, -1, None)
    
    wrest = spec['wave'] / (1+z)
    xdla = (wrest >= wrange[0]/1.e4) & (wrest <= wrange[1]/1.e4) & (spec['valid'])
    if xdla.sum() <= 2:
        return (-1, -1, xdla.sum(), -1, -1, None)
        
    x = wrest[xdla]
    
    # UV slope and uncertainty
    beta = (np.log(f0[2]) - 2*np.log(w0) - np.log(f1[2]) + 2*np.log(w1)) / np.log(w0/w1)
    vbeta = ((f0[4]/f0[2])**2 + (f1[4]/f1[2])**2) / np.abs(np.log(w0/w1))
    beta_unc = np.sqrt(vbeta)
    
    # Continuum fit and uncertainty
    log_fuv = (beta+2) * np.log(wrest/w0) + np.log(f0[2])
    vlog_fuv = (vbeta) * np.abs(np.log(wrest/w0)) + (f0[4]/f0[2])**2
    
    fcont = np.exp(log_fuv)[xdla]
    econt = (np.sqrt(vlog_fuv)*np.exp(log_fuv))[xdla]
    
    dx = utils.trapz_dx(x)*1.e4
    sx = spec[xdla]
    ydata = 1 - sx['flux'] / fcont
    vdata = (sx['flux'] / fcont)**2 * ( (sx['full_err'] / sx['flux'])**2 +
                                         (econt/fcont)**2 )
    
    # Do trapezoid rule integration and propagation of uncertainty
    dla_value = (ydata*dx).sum()
    dla_unc = np.sqrt((vdata*dx).sum())
    
    if make_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
        ax.errorbar(wrest, spec['flux'], spec['full_err'], color='k',
                    marker='.',
                    linestyle='None',
                    alpha=0.5)
        ax.step(wrest, spec['flux'], where='mid', color='k', alpha=0.5)
        
        ax.errorbar(w0, f0[2], f0[4], color='r', zorder=100, marker='o', alpha=0.4)
        ax.errorbar(w1, f1[2], f1[4], color='r', zorder=100, marker='o', alpha=0.4)
        
        if dla_value < 0:
            y0 = 2*np.interp(0.1216, x, fcont)
        else:
            y0 = 0.
            
        ax.vlines(0.1216 + np.array([-1,1])*dla_value/2.e4, y0,
                  np.interp(0.1216, x, fcont), color='magenta',
                  label=f'EW_DLA = {dla_value:.1f} ({dla_unc:.1f})')

        ax.fill_between(x, fcont-econt, fcont+econt, color='red', alpha=0.1)
        ax.plot(x, fcont, color='r', alpha=0.2,
                label=f'Beta = {beta:5.2f} ({beta_unc:.2f})')

        ymax = np.maximum(f0[2]+f0[4], f1[2]+f1[4])*2
        ax.set_ylim(-0.1*ymax, ymax)
        ax.set_xlim(0.11, w1 + 0.2*(w1-w0))
        ax.grid()
        ax.set_xlabel(r'$\lambda_\mathrm{rest}$')
        ax.set_ylabel(r'$f_\nu$')
        leg = ax.legend(loc='upper right')
        leg.set_title(f"{spec.meta['SRCNAME']}\nz = {z:.4f}")
        # ax.set_title(spec.meta['FILENAME'])
        
        fig.tight_layout(pad=1)
    else:
        fig = None
        
    return beta, beta_unc, xdla.sum(), dla_value, dla_unc, fig

