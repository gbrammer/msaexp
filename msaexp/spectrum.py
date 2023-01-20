
"""
Fits, etc. to extracted spectra
"""
import os
import time
import numpy as np

import scipy.ndimage as nd
from scipy.optimize import nnls

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec

import astropy.io.fits as pyfits

from grizli import utils
utils.set_warnings()

import grizli.utils_c
import astropy.units as u

import eazy.igm
igm = eazy.igm.Inoue14()

SCALE_UNCERTAINTY = 2.0

try:
    from prospect.utils.smoothing import smoothspec
except (FileNotFoundError, TypeError):
    if 'SPS_HOME' not in os.environ:
        sps_home = 'xxxxdummyxxxx' #os.path.dirname(__file__)
        print(f'msaexp: setting environment variable SPS_HOME={sps_home} '
              'to be able to import prospect.utils.smoothing')
        os.environ['SPS_HOME'] = sps_home


def smooth_template_disp_prospector(templ, wobs_um, disp, z, velocity_fwhm=80, scale_disp=1.0, flambda=True, with_igm=True, fftsmooth=True):
    """
    Smooth a template with a wavelength-dependent dispersion function using 
    the `prospector` LSF smoothing function
    
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
    from prospect.utils.smoothing import smoothspec
    
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
                         fftsmooth=fftsmooth,
                        )
    
    return tsmooth


def smooth_template_disp(templ, wobs_um, disp, z, velocity_fwhm=80, scale_disp=1.5, flambda=True, with_igm=True):
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


SMOOTH_TEMPLATE_DISP_FUNC = smooth_template_disp_prospector


def fit_redshift(file='jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits', z0=[0.2, 10], step0=None, eazy_templates=None, nspline=None, scale_disp=1.5, vel_width=100, Rline=None, is_prism=False, use_full_dispersion=False, ranges=None):
    """
    """
    import yaml
    def float_representer(dumper, value):
        text = '{0:.6f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
    
    yaml.add_representer(float, float_representer)
    
    is_prism |= ('clear' in file)
    
    if step0 is None:
        if (is_prism):
            step0 = 0.002
        else:
            step0 = 0.001
    
    if is_prism:
        step1 = 0.0001
    else:
        step1 = 0.00002
    
    if Rline is None:
        if is_prism:
            Rline = 1000
        else:
            Rline = 5000
              
    zgrid = utils.log_zgrid(z0, step0)
    zg0, chi0 = fit_redshift_grid(file, zgrid=zgrid, line_complexes=False, 
                                  vel_width=vel_width, scale_disp=scale_disp, 
                                  eazy_templates=eazy_templates, Rline=Rline,
                                  use_full_dispersion=use_full_dispersion)

    zbest0 = zg0[np.argmin(chi0)]
            
    zgrid = utils.log_zgrid(zbest0 + np.array([-0.005, 0.005])*(1+zbest0), 
                            step1)
    zg1, chi1 = fit_redshift_grid(file, zgrid=zgrid, line_complexes=False, 
                                  vel_width=vel_width, scale_disp=scale_disp, 
                                  eazy_templates=eazy_templates, Rline=Rline,
                                  use_full_dispersion=use_full_dispersion)
                                  
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
    
    fz.savefig(file.replace('spec.fits', 'spec.chi2.png'))
    
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
    
    fig, data = plot_spectrum(file, z=zbest, show_cont=True,
                              draws=100, nspline=nspline,
                              figsize=(16, 8), vel_width=vel_width,
                              ranges=ranges, Rline=Rline,
                              scale_disp=scale_disp,
                              eazy_templates=eazy_templates,
                              use_full_dispersion=use_full_dispersion)
    
    if eazy_templates is not None:
        spl_fig, spl_data = plot_spectrum(file, z=zbest, show_cont=True,
                              draws=100, nspline=nspline,
                              figsize=(16, 8), vel_width=vel_width,
                              ranges=ranges, Rline=Rline,
                              scale_disp=scale_disp,
                              eazy_templates=None,
                              use_full_dispersion=use_full_dispersion)
        
        for k in ['coeffs', 'covar', 'model', 'mline', 'full_chi2', 'cont_chi2']:
            if k in spl_data:
                data[f'spl_{k}'] = spl_data[k]
        
        spl_fig.savefig(file.replace('spec.fits', 'spec.spl.png'))
    
    zdata = {}
    zdata['zg0'] = zg0.tolist()
    zdata['chi0'] = chi0.tolist()
    zdata['zg1'] = zg1.tolist()
    zdata['chi1'] = chi1.tolist()
    
    data['dchi2'] = float(np.nanmedian(chi0) - np.nanmin(chi0))
    
    for k in ['templates','spl_covar','covar']:
        if k in data:
            _ = data.pop(k)
    
    with open(file.replace('spec.fits', 'spec.zfit.yaml'), 'w') as fp:
        yaml.dump(zdata, stream=fp)
    
    with open(file.replace('spec.fits', 'spec.yaml'), 'w') as fp:
        yaml.dump(data, stream=fp)
    
    fig.savefig(file.replace('spec.fits', 'spec.zfit.png'))
    
    return fig, data


def fit_redshift_grid(file='jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits', zgrid=None, vel_width=100, bkg=None, scale_disp=1.5, nspline=27, line_complexes=True, Rline=1000, eazy_templates=None, use_full_dispersion=True):
    """
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

    lw, lr = utils.get_line_wavelengths()
    
    with pyfits.open(file) as im:
        if 'SPEC1D' in im:
            spec = utils.read_catalog(im['SPEC1D'])
            if 'POLY0' in spec.meta:
                pc = []
                for pi in range(3):
                    if f'POLY{pi}' in spec.meta:
                        pc.append(spec.meta[f'POLY{pi}'])
                
                corr = np.polyval(pc, np.log(spec['wave']*1.e4))
                spec['flux'] *= corr
                spec['err'] *= corr
        else:
            spec = utils.read_catalog(file)    

    spec['err'] *= SCALE_UNCERTAINTY

    if bkg is None:
        if '2767_11027' in file:
            bkg = 0.015
        else:
            bkg = 0.
    
    eqw = u.spectral_density(spec['wave'].data*spec['wave'].unit)
    
    flam_unit = 1.e-20*u.erg/u.second/u.cm**2/u.Angstrom
    
    # flam_unit = u.microJansky
    
    if hasattr(spec['flux'], 'filled'):
        flam = ((spec['flux'].filled(0).data + bkg) * spec['flux'].unit).to(flam_unit, equivalencies=eqw).value
    else:
        flam = ((spec['flux'].data + bkg) * spec['flux'].unit).to(flam_unit, equivalencies=eqw).value
        
    eflam = spec['err'].to(flam_unit, equivalencies=eqw).value    
    eflam = np.sqrt(eflam**2 + (0.02*flam)**2)
    
    bad = (flam == 0) | (eflam == 0) | (eflam < 0)
    # bad |= wrest < 1000
    
    flam[bad] = np.nan
    eflam[bad] = np.nan
    
    grating = spec.meta['GRATING'].lower()
    _filter = spec.meta['FILTER'].lower()
    
    _data_path = os.path.dirname(__file__)
    disp = utils.read_catalog(f'{_data_path}/data/jwst_nirspec_{grating}_disp.fits')
    
    spline = utils.bspline_templates(wave=spec['wave']*1.e4, degree=3, df=nspline) #, log=True)

    w0 = utils.log_zgrid([spec['wave'].min()*1.e4, spec['wave'].max()*1.e4], 1./Rline)
    
    chi2 = zgrid*0.
    for iz, z in tqdm(enumerate(zgrid)):
        wrest = spec['wave']/(1+z)*1.e4
        
        if eazy_templates is None:
            templates = utils.bspline_templates(wave=spec['wave']*1.e4, degree=3, df=nspline) #, log=True)

            w0 = utils.log_zgrid([spec['wave'].min()*1.e4, spec['wave'].max()*1.e4], 1./Rline)

            # templates = {}
            hlines = ['Ha','Hb', 'Hg', 'Hd'] 
            if z > 4:
                hlines += ['H7','H8','H9', 'H10', 'H11', 'H12']

            if z > 5:
                oiii = ['OIII-4959','OIII-5007']
            else:
                oiii = ['OIII']

            if 'g140m' in file:
                oiii = ['OIII-4959','OIII-5007']
                sii = ['SII-6717', 'SII-6731']
            else:
                sii = ['SII']

            for l in [*hlines, *oiii, 'OIII-4363', 'OII',
                      'HeII-4687', 
                      *sii,
                      'OII-7325', 'ArIII-7138', 'NII', 'SIII-9068', 'SIII-9531',
                      'OI-6302', 'PaD', 'PaG', 'PaB', 'PaA', 'HeI-1083', 
                      'NeIII-3867', 'HeI-3889', 'NeIII-3968', 'HeI-5877', 
                      'HeII-1640', 'CIV-1549',
                      'CIII-1908', 'OIII-1663', 'NIII-1750', 'Lya',
                      'MgII', 'NeV-3346', 'NeVI-3426']:

                lwi = lw[l][0]*(1+z)

                if lwi < spec['wave'][~bad].min()*1.e4:
                    continue

                if lwi > spec['wave'][~bad].max()*1.e4:
                    continue

                # print(l, lwi, disp_r)

                name = f'line {l}'

                for i, (lwi0, lri) in enumerate(zip(lw[l], lr[l])):
                    lwi = lwi0*(1+z)
                    disp_r = np.interp(lwi/1.e4, disp['WAVELENGTH'], disp['R'])*scale_disp

                    vel_fwhm = np.sqrt((lwi/disp_r)**2 + (vel_width/3.e5*lwi)**2)

                    # print(f'Add component: {l} {lwi0} {lri}')

                    if i == 0:
                        templates[name] = utils.SpectrumTemplate(wave=w0, flux=None, central_wave=lwi, fwhm=vel_width, name=name)
                        templates[name].flux *= lri/np.sum(lr[l])
                    else:
                        templates[name].flux += utils.SpectrumTemplate(wave=w0, flux=None, central_wave=lwi, fwhm=vel_width, name=name).flux*lri/np.sum(lr[l])

            _, _A, tline = utils.array_templates(templates, wave=spec['wave'].astype(float)*1.e4, apply_igm=False)
            # print(tline)

            ll = spec['wave'].value*1.e4/(1+z) < 1215.6

            igmz = igm.full_IGM(z, spec['wave'].value*1.e4)
            _A *= np.maximum(igmz, 0.01)
        else:
            templates = {}
            if use_full_dispersion:
                _A = []
                tline = np.zeros(len(eazy_templates), dtype=bool)
                for i, t in enumerate(eazy_templates):
                    templates[t.name] = 0.
                    tflam = SMOOTH_TEMPLATE_DISP_FUNC(t,
                                                 spec['wave'],
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
                                                     wave=wrest,
                                                     z=z, apply_igm=True)
        
                for i in range(len(templates)):
                    _A[i,:] = nd.gaussian_filter(_A[i,:], 0.5)

        _Ax = _A/eflam
        _yx = flam/eflam
        #_x = np.linalg.lstsq(_Ax[:,~spec['flux'].mask].T, _yx[~spec['flux'].mask], rcond=None)
        if hasattr(spec['flux'], 'mask'):
            mask = ~spec['flux'].mask
        else:
            mask = np.isfinite(spec['flux']+spec['err'])
            mask &= spec['err'] > 0
            mask &= spec['flux'] != 0
        
        if eazy_templates is None:
            _x = np.linalg.lstsq(_Ax[:,mask].T, 
                                 _yx[mask], rcond=None)
        else:
            _x = nnls(_Ax[:,mask].T, _yx[mask])

        _model = _A.T.dot(_x[0])
        
        chi = (flam - _model) / eflam
        
        chi2_i = (chi[~bad]**2).sum()
        # print(z, chi2_i)
        chi2[iz] = chi2_i
        
    return zgrid, chi2


def plot_spectrum(file='jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits', z=9.505, vel_width=100, bkg=None, scale_disp=1.5, nspline=27, show_cont=True, draws=100, figsize=(16, 8), ranges=[(3650, 4980)], Rline=1000, full_log=False, write=False, eazy_templates=None, use_full_dispersion=True):
    """
    """
    lw, lr = utils.get_line_wavelengths()

    with pyfits.open(file) as im:
        if 'SPEC1D' in im:
            spec = utils.read_catalog(im['SPEC1D'])
            if 'POLY0' in spec.meta:
                pc = []
                for pi in range(3):
                    if f'POLY{pi}' in spec.meta:
                        pc.append(spec.meta[f'POLY{pi}'])
                
                corr = np.polyval(pc, np.log(spec['wave']*1.e4))
                spec['flux'] *= corr
                spec['err'] *= corr
            
        else:
            spec = utils.read_catalog(file)    
    
    spec['err'] *= SCALE_UNCERTAINTY
    
    if bkg is None:
        if '2767_11027' in file:
            bkg = 0.015
        else:
            bkg = 0.
    
    eqw = u.spectral_density(spec['wave'].data*spec['wave'].unit)
    
    flam_unit = 1.e-20*u.erg/u.second/u.cm**2/u.Angstrom
    
    # flam_unit = u.microJansky
    
    if hasattr(spec['flux'], 'filled'):
        flam = ((spec['flux'].filled(0).data + bkg) * spec['flux'].unit).to(flam_unit, equivalencies=eqw).value
    else:
        flam = ((spec['flux'].data + bkg) * spec['flux'].unit).to(flam_unit, equivalencies=eqw).value
        
    eflam = spec['err'].to(flam_unit, equivalencies=eqw).value
    eflam = np.sqrt(eflam**2 + (0.02*flam)**2)

    wrest = spec['wave']/(1+z)*1.e4
    wobs = spec['wave']

    bad = (flam == 0) | (eflam == 0) | (eflam < 0)
    # bad |= wrest < 1000
    
    flam[bad] = np.nan
    eflam[bad] = np.nan
    
    grating = spec.meta['GRATING'].lower()
    _filter = spec.meta['FILTER'].lower()
    
    _data_path = os.path.dirname(__file__)
    disp = utils.read_catalog(f'{_data_path}/data/jwst_nirspec_{grating}_disp.fits')
    
    #templates = utils.cheb_templates(wave=spec['wave']*1.e4, order=33, log=True)
    
    if eazy_templates is None:
        templates = utils.bspline_templates(wave=spec['wave']*1.e4, degree=3, df=nspline) #, log=True)

        w0 = utils.log_zgrid([spec['wave'].min()*1.e4, spec['wave'].max()*1.e4], 1./Rline)

        # templates = {}
        hlines = ['Ha','Hb', 'Hg', 'Hd'] 
        if z > 4:
            hlines += ['H7','H8','H9', 'H10', 'H11', 'H12']
        
        if z > 5:
            oiii = ['OIII-4959','OIII-5007']
        else:
            oiii = ['OIII']

        if 'g140m' in file:
            oiii = ['OIII-4959','OIII-5007']
            sii = ['SII-6717', 'SII-6731']
        else:
            sii = ['SII']
        
        for l in [*hlines, *oiii, 'OIII-4363', 'OII',
                  'HeII-4687', 
                  *sii,
                  'OII-7325', 'ArIII-7138', 'NII', 'SIII-9068', 'SIII-9531',
                  'OI-6302', 'PaD', 'PaG', 'PaB', 'PaA', 'HeI-1083', 
                  'NeIII-3867', 'HeI-3889', 'NeIII-3968', 'HeI-5877', 
                  'HeII-1640', 'CIV-1549',
                  'CIII-1908', 'OIII-1663', 'NIII-1750', 'Lya',
                  'MgII', 'NeV-3346', 'NeVI-3426']:
        
            lwi = lw[l][0]*(1+z)

            if lwi < spec['wave'][~bad].min()*1.e4:
                continue

            if lwi > spec['wave'][~bad].max()*1.e4:
                continue

            # print(l, lwi, disp_r)
        
            name = f'line {l}'
        
            for i, (lwi0, lri) in enumerate(zip(lw[l], lr[l])):
                lwi = lwi0*(1+z)
                disp_r = np.interp(lwi/1.e4, disp['WAVELENGTH'], disp['R'])*scale_disp
        
                vel_width = np.sqrt((lwi/disp_r)**2 + (vel_width/3.e5*lwi)**2)

                # print(f'Add component: {l} {lwi0} {lri}')

                if i == 0:
                    templates[name] = utils.SpectrumTemplate(wave=w0, flux=None, central_wave=lwi, fwhm=vel_width, name=name)
                    templates[name].flux *= lri/np.sum(lr[l])
                else:
                    templates[name].flux += utils.SpectrumTemplate(wave=w0, flux=None, central_wave=lwi, fwhm=vel_width, name=name).flux*lri/np.sum(lr[l])

        _, _A, tline = utils.array_templates(templates, wave=spec['wave'].astype(float)*1.e4, apply_igm=False)
        # print(tline)
    
        ll = spec['wave'].value*1.e4/(1+z) < 1215.6
    
        igmz = igm.full_IGM(z, spec['wave'].value*1.e4)
        _A *= np.maximum(igmz, 0.01)
    else:
        templates = {}
        if use_full_dispersion:
            _A = []
            tline = np.zeros(len(eazy_templates), dtype=bool)
            for i, t in enumerate(eazy_templates):
                templates[t.name] = 0.
                tflam = SMOOTH_TEMPLATE_DISP_FUNC(t,
                                             spec['wave'],
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
                                                 wave=wrest,
                                                 z=z, apply_igm=True)
        
            for i in range(len(templates)):
                _A[i,:] = nd.gaussian_filter(_A[i,:], 0.5)
                
    _Ax = _A/eflam
    _yx = flam/eflam
    
    if hasattr(spec['flux'], 'mask'):
        mask = ~spec['flux'].mask
    else:
        mask = np.isfinite(spec['flux']+spec['err'])
        mask &= spec['err'] > 0
        mask &= spec['flux'] != 0 
        
    if eazy_templates is None:
        _x = np.linalg.lstsq(_Ax[:,mask].T, _yx[mask], rcond=None)
    else:
        _x = nnls(_Ax[:,mask].T, _yx[mask])
    
    _model = _A.T.dot(_x[0])
    _mline = _A.T.dot(_x[0]*tline)
    _mcont = _model - _mline
    
    full_chi2 = ((flam - _model)**2/eflam**2)[~bad].sum()
    cont_chi2 = ((flam - _mcont)**2/eflam**2)[~bad].sum()
    
    try:
        #covar = utils.safe_invert(np.dot(_Ax[:,~spec['flux'].mask].T.T, _Ax[:,~spec['flux'].mask].T))
        #covard = np.sqrt(covar.diagonal())
        
        oktemp = (_x[0] != 0)
        # oktemp[:3] = False
        
        AxT = _Ax[:,mask][oktemp,:].T
        
        covar_i = utils.safe_invert(np.dot(AxT.T, AxT))
        covar = utils.fill_masked_covar(covar_i, oktemp)
        covard = np.sqrt(covar.diagonal())

        has_covar = True
    except:
        has_covar = False
        covard = _x[0]*0.
        N = len(templates)
        covar = np.eye(N, N)
    
    print(f'\n# line flux err\n# flux x 10^-20 erg/s/cm2\n# {file}\n# z = {z:.5f}\n# {time.ctime()}')
    coeffs = {}
    for i, t in enumerate(templates):
        coeffs[t] = [float(_x[0][i]), float(covard[i])]
        if t.startswith('line '):
            print(f'{t:>20}   {_x[0][i]:8.1f} ± {covard[i]:8.1f}')
    
    if 'source_ra' not in spec.meta:
        spec.meta['source_ra'] = 0.0
        spec.meta['source_dec'] = 0.0
        spec.meta['source_name'] = 'unknown'
            
    data = {'z': float(z),
            'file':file,
            'ra': float(spec.meta['source_ra']),
            'dec': float(spec.meta['source_dec']),
            'name': str(spec.meta['source_name']),
            'wmin':float(spec['wave'][~bad].min()),
            'wmax':float(spec['wave'][~bad].max()),
            'coeffs':coeffs,
            'covar':covar.tolist(),
            'model': [float(m) for m in _model],
            'mline':[float(m) for m in _mline],
            'templates':templates, 
            'dof': int((~bad).sum()), 
            'full_chi2': float(full_chi2), 
            'cont_chi2': float(cont_chi2),
           }
            
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
        
    ok = np.isfinite(spec['flux']+spec['err']) & (spec['err'] > 0)

    _Acont = (_A.T*_x[0])[ok,:][:,:nspline]
    _Acont[_Acont < 0.001*_Acont.max()] = np.nan
    
    if (draws is not None) & has_covar:
        mu = np.random.multivariate_normal(_x[0][oktemp], covar_i, size=draws)
        #print('draws', draws, mu.shape, _A.shape)
        mdraws = _A[oktemp,:].T.dot(mu.T)
    else:
        mdraws = None
        
    for ax in axes:
        if 1:
            ax.errorbar(wobs, flam, eflam, marker='None', linestyle='None',
                    alpha=0.5, color='k', ecolor='k', zorder=100) #log_flux[0], log_flux[1])

        ax.step(wobs, flam, color='k', where='mid', lw=1, alpha=0.8)
        # ax.set_xlim(3500, 5100)

        #ax.plot(_[1]['templz']/(1+z), _[1]['templf'])
        
        ax.step(wobs[ok], _mcont[ok], color='pink', alpha=0.8, where='mid')
        ax.step(wobs[ok], _model[ok], color='r', alpha=0.8, where='mid')
        
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
            ax.step(wobs[ok], mdraws[ok,:], color='r', alpha=np.maximum(1./draws, 0.02), zorder=-100, where='mid')

        if show_cont:
            ax.plot(wobs[ok], _Acont, color='olive', alpha=0.3)
            
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
        
    if _filter == 'clear':
        axes[-1].set_xlim(0.6, 5.29)
        axes[-1].xaxis.set_minor_locator(MultipleLocator(0.1))
        axes[-1].xaxis.set_major_locator(MultipleLocator(0.5))
    elif _filter == 'f070lp':
        axes[-1].set_xlim(0.69, 1.31)
        axes[-1].xaxis.set_minor_locator(MultipleLocator(0.02))
    elif _filter == 'f100lp':
        axes[-1].set_xlim(0.99, 1.91)
        axes[-1].xaxis.set_minor_locator(MultipleLocator(0.02))
        axes[-1].xaxis.set_major_locator(MultipleLocator(0.1))
    elif _filter == 'f170lp':
        axes[-1].set_xlim(1.69, 3.21)
    elif _filter == 'f290lp':
        axes[-1].set_xlim(2.89, 5.31)
    else:
        axes[-1].set_xlim(wrest[~bad].min(), wrest[~bad].max())
    
    axes[-1].set_xlabel(f'obs wavelenth, z = {z:.5f}')
    
    #axes[0].set_title(os.path.basename(file))
    
    for ax in axes:
        xl = ax.get_xlim()
        ok = wobs > xl[0]
        ok &= wobs < xl[1]
        ok &= np.abs(wrest-5008) > 100
        ok &= np.abs(wrest-6564) > 100
        ok &= np.isfinite(spec['flux']+spec['err']) & (spec['err'] > 0)
        if ok.sum() == 0:
            ax.set_visible(False)
            continue
        
        ymax = np.maximum(_model[ok].max(), 10*np.median(eflam[ok]))
        
        ymin = np.minimum(-0.1*ymax, -3*np.median(eflam[ok]))
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
    
    fig.text(0.015*12./12, 0.005, f'{os.path.basename(file)}',
             ha='left', va='bottom',
             transform=fig.transFigure, fontsize=8)
    
    fig.text(1-0.015*12./12, 0.005, time.ctime(),
             ha='right', va='bottom',
             transform=fig.transFigure, fontsize=6)
    
    
    return fig, data

#PATH = '/Users/gbrammer/Research/JWST/Projects/RXJ2129/Nirspec/'
# if 0:
#     fig, data = plot_spectrum(PATH + 'jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits', z=9.503, show_cont=True, draws=100, nspline=41,
#                     figsize=(16, 8), ranges=[(3650, 5100)], Rline=2000)

    