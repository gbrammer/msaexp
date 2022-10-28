
"""
Fits, etc. to extracted spectra
"""

import eazy.igm
igm = eazy.igm.Inoue14()

def plot_spectrum(file='jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits', z=9.505, vel_width=100, bkg=None, scale_disp=1.5, nspline=27, show_cont=True, draws=100, figsize=(16, 8), ranges=[(3650, 4980)], Rline=1000, full_log=False):
    """
    """
    import time
    import os
    
    import numpy as np
    from grizli import utils
    import grizli.utils_c
    import astropy.units as u
    
    import eazy.igm
    
    import matplotlib.pyplot as plt

    lw, lr = utils.get_line_wavelengths()

    spec = utils.read_catalog(file)
    
    if bkg is None:
        if '2767_11027' in file:
            bkg = 0.015
        else:
            bkg = 0.
    
    eqw = u.spectral_density(spec['wave'].data*spec['wave'].unit)
    
    flam_unit = 1.e-20*u.erg/u.second/u.cm**2/u.Angstrom
    
    # flam_unit = u.microJansky
    
    flam = ((spec['flux'].filled(0).data + bkg) * spec['flux'].unit).to(flam_unit, equivalencies=eqw).value
    eflam = spec['err'].to(flam_unit, equivalencies=eqw).value
    wrest = spec['wave']/(1+z)*1.e4

    bad = (flam == 0) | (eflam == 0) | (eflam < 0)
    # bad |= wrest < 1000
    
    flam[bad] = np.nan
    eflam[bad] = np.nan
    
    grating = spec.meta['GRATING'].lower()
    disp = utils.read_catalog(f'/Users/gbrammer/Research/JWST/Projects/NIRSpec/jwst_nirspec_{grating}_disp.fits')
    
    #templates = utils.cheb_templates(wave=spec['wave']*1.e4, order=33, log=True)
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
        
    for l in [*hlines, *oiii, 'OIII-4363', 'OII',
              'HeII-4687', 
              'SII', 'OII-7325', 'ArIII-7138', 'NII', 'SIII-9068', 'SIII-9531', 'OI-6302', 'PaD', 'PaG', 'PaB', 'PaA', 'HeI-1083', 
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

    _Ax = _A/eflam
    _yx = flam/eflam
    _x = np.linalg.lstsq(_Ax[:,~spec['flux'].mask].T, _yx[~spec['flux'].mask], rcond=None)
    
    _model = _A.T.dot(_x[0])
    _mline = _A.T.dot(_x[0]*tline)
    _mcont = _model - _mline
    
    try:
        #covar = utils.safe_invert(np.dot(_Ax[:,~spec['flux'].mask].T.T, _Ax[:,~spec['flux'].mask].T))
        #covard = np.sqrt(covar.diagonal())
        
        oktemp = (_x[0] != 0)
        # oktemp[:3] = False
        
        AxT = _Ax[:,~spec['flux'].mask][oktemp,:].T
        
        covar_i = utils.safe_invert(np.dot(AxT.T, AxT))
        covar = utils.fill_masked_covar(covar_i, oktemp)
        covard = np.sqrt(covar.diagonal())

        has_covar = True
    except:
        has_covar = False
        covard = _x[0]*0.
        N = len(templates)
        covar = np.eye(N, N)
    
    print(f'\n# line flux err\n# flux x 10^-20 erg/s/cm2\n# {file}\n# z = {z:.4f}\n# {time.ctime()}')
    coeffs = {}
    for i, t in enumerate(templates):
        coeffs[t] = (_x[0][i], covard[i])
        if t.startswith('line '):
            print(f'{t:>20}   {_x[0][i]:8.1f} ± {covard[i]:8.1f}')
            
    data = {'z': z, 'file':file, 'coeffs':coeffs, 'covar':covar, 'model': _model, 'mline':_mline, 'templates':templates}
        
    fig, axes = plt.subplots(len(ranges)+1,1,figsize=figsize)
    
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
            ax.errorbar(spec['wave']/(1+z)*1.e4, flam, eflam, marker='None', linestyle='None',
                    alpha=0.5, color='k', ecolor='k', zorder=100) #log_flux[0], log_flux[1])

        ax.step(spec['wave']/(1+z)*1.e4, flam, color='k', where='mid', lw=1, alpha=0.8)
        # ax.set_xlim(3500, 5100)

        #ax.plot(_[1]['templz']/(1+z), _[1]['templf'])
        
        ax.step(spec['wave'][ok]/(1+z)*1.e4, _mcont[ok], color='pink', alpha=0.8, where='mid')
        ax.step(spec['wave'][ok]/(1+z)*1.e4, _model[ok], color='r', alpha=0.8, where='mid')
        
        if mdraws is not None:
            ax.step(spec['wave'][ok]/(1+z)*1.e4, mdraws[ok,:], color='r', alpha=np.maximum(1./draws, 0.02), zorder=-100, where='mid')

        if show_cont:
            ax.plot(spec['wave'][ok]/(1+z)*1.e4, _Acont, color='olive', alpha=0.3)
            
        ax.fill_between(ax.get_xlim(), [-10,-10], [0, 0], color='0.8', alpha=0.5, zorder=-1)

        ax.fill_betweenx([0, 1000], [0,0], [1215.67, 1215.67], color=utils.MPL_COLORS['orange'], alpha=0.2, zorder=-1)
    
        ax.grid()

    # axes[0].set_xlim(1000, 2500)
    # ym = 0.15; axes[0].set_ylim(-0.1*ym, ym)
    
    for i, r in enumerate(ranges):
        axes[i].set_xlim(*r)

    axes[-1].set_xlim(wrest[~bad].min(), wrest[~bad].max())
    
    axes[-1].set_xlabel(f'rest wavelenth, z = {z:.3f}')
    axes[0].set_title(os.path.basename(file))
    
    for ax in axes:
        xl = ax.get_xlim()
        ok = wrest > xl[0]
        ok &= wrest < xl[1]
        ok &= np.abs(wrest-5008) > 100
        ok &= np.abs(wrest-6564) > 100
        ok &= np.isfinite(spec['flux']+spec['err']) & (spec['err'] > 0)
        
        
        ymax = np.maximum(_model[ok].max(), 10*np.median(eflam[ok]))
        
        ymin = np.minimum(-0.1*ymax, -3*np.median(eflam[ok]))
        ax.set_ylim(ymin, ymax*1.3)
        # print(xl, ymax)
    
    if (np.nanmax((flam/eflam)[ok]) > 20) & (full_log):
        ax.set_ylim(0.005*ymax, ymax*5)
        ax.semilogy()
        
    fig.tight_layout(pad=1)
    
    return fig, data

#PATH = '/Users/gbrammer/Research/JWST/Projects/RXJ2129/Nirspec/'
if 0:
    fig, data = plot_spectrum(PATH + 'jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits', z=9.503, show_cont=True, draws=100, nspline=41,
                    figsize=(16, 8), ranges=[(3650, 5100)], Rline=2000)
