import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import astropy.io.fits as pyfits

from tqdm import tqdm

import jwst.datamodels

from grizli import utils
import msaexp.utils as msautils

from scipy.special import huber

EVAL_COUNT = 0
CHI2_MASK = None
SKIP_COUNT = 100

def split_visit_groups(files, join=[0, 3], gratings=['PRISM']):
    """
    """
    # keys = []
    # for file in files:
    #     key = ['_'.join([file.split('_')[i] for i in join])]
    #     keys.append(key)
    keys = []
    all_files = []
    for file in files:
        with pyfits.open(file) as im:
            sh = im['SCI'].data.shape
            if im[0].header['GRATING'] not in gratings:
                continue
            
            fi = os.path.basename(file)
            row = [file, sh[0], sh[1], im[0].header['GRATING'], im[0].header['FILTER']]
            fk = '_'.join([file.replace('.','_').split('_')[i] for i in join])
            key = f"{fk}-{im[0].header['GRATING']}"
            #key = f"{fi.split('_')[0]}-{im[0].header['GRATING']}-{fi.split('_')[3]}"
            # key += f"-{sh[0]}x{sh[1]}"
            keys.append(key.lower())
            all_files.append(file)
            
    keys = np.array(keys)
    un = utils.Unique(keys, verbose=False)
    groups = {}
    for k in np.unique(keys):
        groups[k] = np.array(all_files)[un[k]].tolist()
    
    return groups


def slit_prf_fraction(wave, sigma=0., x_pos=0., slit_width=0.2, pixel_scale=0.1, verbose=True):
    """
    Rough slit-loss correction given derived source width and x_offset shutter centering
    
    Parameters
    ----------
    sigma : float
        Derived source width (pixels) in quadtrature with the tabulated intrinsic PSF 
        width from ``msaexp.utils.get_nirspec_psf_fwhm``
    
    wave : array-like, float
        Spectrum wavelengths, microns
    
    x_pos : float
        Shutter-normalized source center in range (-0.5, 0.5) (``source_xpos`` in slit
        metadata)
    
    slit_width : float
        Slit/shutter width, arcsec
    
    pixel_scale : float
        NIRSpec pixel scale, arcsec/pix
    
    Returns
    -------
    prf_frac : array-like
        Wavelength-dependent flux fraction within the shutter
    
    """
    from msaexp.resample_numba import pixel_integrated_gaussian_numba as PRF
    
    # Tabulated PSF FWHM, pix
    psf_fw = msautils.get_nirspec_psf_fwhm(wave)
    
    pix_center = np.zeros_like(wave)
    pix_mu = x_pos * slit_width / pixel_scale
    pix_sigma = np.sqrt((psf_fw/2.35)**2 + sigma**2)
    
    msg = f'slit_prf_fraction: mu = {pix_mu:.2f}, sigma = {sigma:.1f} pix'
    if verbose:
        print(msg)
    
    prf_frac = PRF(pix_center, pix_mu, pix_sigma,
                   dx=slit_width / pixel_scale, normalization=1)
    
    return prf_frac


def objfun_prof_trace(theta, base_coeffs, wave, xpix, ypix, yslit0, diff, vdiff, mask, ipos, ineg, sh, fix_sigma, force_positive, ret):
    """
    """
    from msaexp.resample_numba import pixel_integrated_gaussian_numba as PRF
    
    global EVAL_COUNT
    global CHI2_MASK
    global SKIP_COUNT
    
    EVAL_COUNT += 1

    if fix_sigma > 0:
        sigma = fix_sigma/10.
        i0 = 0
    else:
        sigma = theta[0]/10.
        i0 = 1
        
    yslit = yslit0*1.
    for j in np.where(ipos)[0]:
        xj = (xpix[j,:] - sh[1]/2) / sh[1]
        _ytr = np.polyval(theta[i0:], xj)
        _ytr += np.polyval(base_coeffs, xj)
        yslit[j,:] = ypix[j,:] - _ytr
    
    psf_fw = msautils.get_nirspec_psf_fwhm(wave)
    
    sig2 = np.sqrt((psf_fw/2.35)**2 + sigma**2)
    
    ppos = PRF(yslit[ipos,:].flatten(), 0., sig2[ipos,:].flatten(), dx=1)
    ppos = ppos.reshape(yslit[ipos,:].shape)
        
    if ineg.sum() > 0:
        pneg = PRF(yslit[ineg,:].flatten(), 0., sig2[ineg,:].flatten(), dx=1)
        pneg = pneg.reshape(yslit[ineg,:].shape)
    else:
        pneg = np.zeros_like(ppos)

    if 0:
        ppos  = np.nansum(ppos, axis=0) / ipos.sum() #np.nansum(mask[ipos,:], axis=0)
        if ineg.sum() > 0:
            pneg  = np.nansum(pneg, axis=0) / ineg.sum() #np.nansum(mask[ineg,:], axis=0)
        else:
            pneg = np.zeros_like(ppos)            
    else:
        ppos  = np.nansum(ppos, axis=0) / np.nansum(mask[ipos,:], axis=0)
        if ineg.sum() > 0:
            pneg  = np.nansum(pneg, axis=0) / np.nansum(mask[ineg,:], axis=0)
        else:
            pneg = np.zeros_like(ppos)

    pdiff = ppos - pneg
    
    if (pneg.sum() == 0) & (len(theta) == 1000):
        # print('bkg: ', EVAL_COUNT, theta)
        bkg = theta[2]/10.
    else:
        bkg = 0.
    
    if force_positive:
        pdiff *= pdiff > 0
    
    # Remove any masked pixels
    pmask = mask.sum(axis=0) == mask.shape[0]
    
    snum = np.nansum(((diff-bkg)*pdiff/vdiff*pmask).reshape(sh), axis=0)
    sden = np.nansum((pdiff**2/vdiff*pmask).reshape(sh), axis=0)
    smod = snum/sden*pdiff.reshape(sh)
    
    chi = (diff - (smod + bkg).flatten())/np.sqrt(vdiff)
    
    # if (CHI2_MASK is None) or (CHI2_MASK.size != chi2.size):
    #     CHI2_MASK = np.isfinite(diff)
    # elif (EVAL_COUNT == 1): #| (CHI2_MASK.size != chi2.size):
    #     CHI2_MASK = chi2 < 40**2
    # else:
    #     CHI2_MASK = np.isfinite(diff)

    if 0:
        # two-sided
        # CHI2_MASK = (chi < 40) & (chi > -10)
        # CHI2_MASK &= ((smod+bkg).flatten()/np.sqrt(vdiff) > -10)
        CHI2_MASK = diff/np.sqrt(vdiff) > -10
        
    elif 0:
        # absolute value
        CHI2_MASK = (chi**2 < 40**2) #& ((smod+bkg).flatten()/np.sqrt(vdiff) > -10)
    else:
        # no mask
        CHI2_MASK = np.isfinite(diff)
    
    ok = np.isfinite(chi)
    #chi2 = np.nansum(chi[CHI2_MASK]**2)
    chi2 = np.nansum(huber(7, chi[CHI2_MASK & ok]))
    
    # "prior" on sigma with logistic bounds
    peak = 10000
    chi2 += peak / (1 + np.exp(-1*(sigma - 2.8))) # right
    chi2 += peak - peak / (1 + np.exp(-3*(sigma - 0))) # left
    chi2 += (sigma - 0.6)**2/2/0.1**2
    
    if (EVAL_COUNT % SKIP_COUNT == 0) | (ret == 1):
        tval = ' '.join([f'{t:6.3f}' for t in theta[i0:]])
        tfix = '*' if i0 == 0 else ' '
        print(f"{EVAL_COUNT:>8} {tfix}sigma={sigma*10:.2f}{tfix} [{tval}]  {chi2:.1f}")

    if ret == 1:
        trace_coeffs = theta[i0:]
        return snum, sden, smod, sigma, trace_coeffs, chi2
    else:
        return chi2


def objfun_prof(theta, wave, yslit, diff, vdiff, mask, ipos, ineg, sh, force_positive, ret):
    """
    """
    from msaexp.resample_numba import pixel_integrated_gaussian_numba as PRF
    
    global EVAL_COUNT
    global CHI2_MASK
    global SKIP_COUNT
    
    EVAL_COUNT += 1

    center = theta[0]
    if len(theta) > 1:
        sigma = theta[1]/10.
    else:
        sigma = 0.4
    
    psf_fw = msautils.get_nirspec_psf_fwhm(wave)
    
    sig2 = np.sqrt((psf_fw/2.35)**2 + sigma**2)
    ppos = PRF(yslit[ipos,:].flatten(), center, sig2[ipos,:].flatten(), dx=1)
    ppos = ppos.reshape(yslit[ipos,:].shape)
    
    pneg = PRF(yslit[ineg,:].flatten(), center, sig2[ineg,:].flatten(), dx=1)
    if ineg.sum() > 0:
        pneg = pneg.reshape(yslit[ineg,:].shape)
    else:
        pneg = np.zeros_like(ppos)

    if 0:
        ppos  = np.nansum(ppos, axis=0) / ipos.sum() #np.nansum(mask[ipos,:], axis=0)
        if ineg.sum() > 0:
            pneg  = np.nansum(pneg, axis=0) / ineg.sum() #np.nansum(mask[ineg,:], axis=0)
        else:
            pneg = np.zeros_like(ppos)            
    else:
        ppos  = np.nansum(ppos, axis=0) / np.nansum(mask[ipos,:], axis=0)
        if pneg.sum() > 0:
            pneg  = np.nansum(pneg, axis=0) / np.nansum(mask[ineg,:], axis=0)
        else:
            pneg = np.zeros_like(ppos)

    pdiff = ppos - pneg
    
    if (pneg.sum() == 0) & (len(theta) == 3):
        # print('bkg: ', EVAL_COUNT, theta)
        bkg = theta[2]/10.
    else:
        bkg = 0.
    
    if (len(theta) > 2) & (pneg.sum() > 0):
        sig2 = 3
        if len(theta) == 3:
            scales = [theta[2], theta[2]]
        else:
            scales = theta[2:]
        
        for k, dy in enumerate([-1.,1.]):
            ppos = PRF(yslit[ipos,:].flatten(), center+dy, sig2, dx=1)
            ppos = ppos.reshape(yslit[ipos,:].shape)
            if ineg.sum() > 0:
                pneg = PRF(yslit[ineg,:].flatten(), center+dy, sig2, dx=1)
                pneg = pneg.reshape(yslit[ineg,:].shape)
            else:
                pneg = np.zeros_like(ppos)

            if 0:
                ppos  = np.nansum(ppos, axis=0) / ipos.sum() #np.nansum(mask[ipos,:], axis=0)
                if ineg.sum() > 0:
                    pneg  = np.nansum(pneg, axis=0) / ineg.sum() #np.nansum(mask[ineg,:], axis=0)
                else:
                    pneg = np.zeros_like(ppos)
                    
            else:
                ppos  = np.nansum(ppos, axis=0) / np.nansum(mask[ipos,:], axis=0)
                if ineg.sum() > 0:
                    pneg  = np.nansum(pneg, axis=0) / np.nansum(mask[ineg,:], axis=0)
                else:
                    pneg = np.zeros_like(ppos)

            pdiff += (ppos - pneg)*scales[k]
        
        pdiff /= 1 + np.sum(scales)

    if force_positive:
        pdiff *= pdiff > 0
    
    # Remove any masked pixels
    pmask = mask.sum(axis=0) == mask.shape[0]
    
    snum = np.nansum(((diff-bkg)*pdiff/vdiff*pmask).reshape(sh), axis=0)
    sden = np.nansum((pdiff**2/vdiff*pmask).reshape(sh), axis=0)
    smod = snum/sden*pdiff.reshape(sh)
    
    chi = (diff - (smod + bkg).flatten())/np.sqrt(vdiff)
    
    # if (CHI2_MASK is None) or (CHI2_MASK.size != chi2.size):
    #     CHI2_MASK = np.isfinite(diff)
    # elif (EVAL_COUNT == 1): #| (CHI2_MASK.size != chi2.size):
    #     CHI2_MASK = chi2 < 40**2
    # else:
    #     CHI2_MASK = np.isfinite(diff)

    if 1:
        # two-sided
        CHI2_MASK = (chi < 40) & (chi > -10)
        CHI2_MASK &= ((smod+bkg).flatten()/np.sqrt(vdiff) > -10)
    elif 0:
        # absolute value
        CHI2_MASK = (chi**2 < 40**2) #& ((smod+bkg).flatten()/np.sqrt(vdiff) > -10)
    else:
        # no mask
        CHI2_MASK = np.isfinite(diff)
    
    chi2 = np.nansum(chi[CHI2_MASK]**2)
    
    if (EVAL_COUNT % SKIP_COUNT == 0) | (ret == 1):
        print(f'{EVAL_COUNT:>8} {theta}  {chi2:.1f}')

    if ret == 0:
        return chi2
    else:
        return snum, sden, smod, 0, chi2


class SlitGroup():
    def __init__(self, files, name, position_key='position_number', diffs=True, stuck_min_sn=0.9, undo_barshadow=False, sky_arrays=None, undo_pathloss=True):
        """
        """
        self.name = name
        
        self.diffs = diffs
        
        self.slits = []
        keep_files = []
        
        self.stuck_min_sn = stuck_min_sn
        self.undo_barshadow = undo_barshadow
        
        self.sky_arrays = sky_arrays
        self.undo_pathloss = undo_pathloss
        
        self.shapes = []
        
        for i, file in enumerate(tqdm(files)):
            slit = jwst.datamodels.open(file)
            visit = f"{file.split('_')[0]}_{file.split('_')[3]}"
    
            # if i > 0:
            #     if slit.data.size != self.slits[0].data.size:
            #         # print(i,slit.data.shape, slits[0].data.shape)
            #         continue
            
            self.slits.append(slit)
            keep_files.append(file)
            self.shapes.append(slit.data.shape)
            
        self.files = keep_files
        self.info = self.parse_metadata()
        self.sh = np.min(np.array(self.shapes), axis=0)
        # print('shape: ', self.sh)
        #self.slits[0].data.shape
        
        self.position_key = position_key
        
        # self.unp = utils.Unique(self.info[position_key], verbose=False)
        
        self.parse_data()
    
    @property
    def N(self):
        return len(self.slits)
    
    @property
    def grating(self):
        return self.info['grating'][0]
    
    @property
    def unp(self):
        return utils.Unique(self.info[self.position_key], verbose=False)
    
    def parse_metadata(self, verbose=True):
        rows = []
        for i, slit in enumerate(self.slits):
            print(i, slit.meta.filename, slit.data.shape)
            md = slit.meta.dither.instance
            mi = slit.meta.instrument.instance
            rows.append([slit.meta.filename,
                         slit.meta.filename.split('_')[0],
                         slit.data.shape] + [md[k] for k in md] + [mi[k] for k in mi])

        names = ['filename','visit', 'shape'] + [k for k in md] + [k for k in mi]
    
        info = utils.GTable(names=names, rows=rows)
        info['x_position'] = np.round(info['x_offset']*10)/10.
        info['y_position'] = np.round(info['y_offset']*10)/10.
        info['y_index'] = utils.Unique(info['y_position'], verbose=False).indices + 1
        
        return info
    
    
    def parse_data(self, verbose=True):
        """
        """
        slits = self.slits
        
        sl = (slice(0,self.sh[0]), slice(0,self.sh[1]))
        
        sci = np.array([slit.data[sl].flatten()*1 for slit in slits])
        try:
            bar = np.array([slit.barshadow[sl].flatten()*1 for slit in slits])
        except:
            bar = None
            
        dq = np.array([slit.dq[sl].flatten()*1 for slit in slits])
        var = np.array([(slit.var_poisson + slit.var_rnoise)[sl].flatten()
                        for slit in slits])
        var_flat = np.array([slit.var_flat[sl].flatten()*1 for slit in slits])
        
        bad = sci == 0
        sci[bad] = np.nan
        var[bad] = np.nan
        var_flat[bad] = np.nan
        dq[bad] = 1
        if bar is not None:
            bar[bad] = np.nan
        
        sh = slits[0].data.shape
        yp, xp = np.indices(sh)
        
        # 2D
        xslit = []
        ypix = []
        yslit = []
        wave = []
        
        # 1D
        xtr = []
        ytr = []
        wtr = []
                
        for slit in slits:
            # try:
            #     wcs = slit.meta.wcs
            #     d2w = wcs.get_transform('detector', 'world')
            # except AttributeError:
            #     wcs = gwcs.WCS(slit.meta.wcs.instance['steps'])
            #     d2w = wcs.get_transform('detector', 'world')
            #     slit.meta.wcs = wcs

            _res = msautils.slit_trace_center(slit, 
                                  with_source_ypos=True,
                                  index_offset=0.0)
            
            _xtr, _ytr, _wtr, slit_ra, slit_dec = _res
            
            xslit.append(xp[sl].flatten())
            yslit.append((yp[sl] - (_ytr[sl[1]])).flatten())
            ypix.append(yp[sl].flatten())
            
            xtr.append(_xtr[sl[1]])
            ytr.append(_ytr[sl[1]])
            wtr.append(_wtr[sl[1]])
            
            wcs = slit.meta.wcs
            d2w = wcs.get_transform('detector', 'world')
            _ypi, _xpi = np.indices(slit.data.shape)
            _ras, _des, _wave = d2w(_xpi, _ypi)
            wave.append(_wave[sl].flatten())
            
            #_ras, _des, ws = d2w(_xtr, _ytr)
            #
            #wave.append((xp[sl]*0. + ws[sl[1]]).flatten())
        
        xslit = np.array(xslit)
        yslit = np.array(yslit)
        ypix = np.array(ypix)
        wave = np.array(wave)
        
        xtr = np.array(xtr)
        ytr = np.array(ytr)
        wtr = np.array(wtr)
        
        self.bad_exposures = np.zeros(self.N, dtype=bool)
        
        bad = (dq & 1025) > 0
        if (self.grating in ['PRISM']):
            low = sci/np.sqrt(var) < self.stuck_min_sn
            nlow = low.sum(axis=1)
            bad_exposures = nlow > 2*self.sh[1]
            print('  Prism exposures with stuck shutters: ', bad_exposures.sum())
            for j in np.where(bad_exposures)[0]:
                bad_j = nd.binary_dilation(low[j,:].reshape(self.sh), iterations=2)
                bad[j,:] |= bad_j.flatten()
            
            self.bad_exposures = bad_exposures
            
            bad |= sci > 100
            bad |= sci == 0
            
        # bad |= np.abs(yslit > 8)
        isfin = np.isfinite(sci)
        
        sci[bad] = np.nan
        mask = np.isfinite(sci)
        var[~mask] = np.nan
        
        self.sci = sci
        self.dq = dq
        self.mask = mask
        self.var = var
        self.var_flat = var_flat
        
        for j, slit in enumerate(slits):
            phot_scl = slit.meta.photometry.pixelarea_steradians*1.e12
            #phot_scl *= (slit.pathloss_uniform / slit.pathloss_point)[sl].flatten()
            # Remove pathloss correction
            if self.undo_pathloss:
                if slit.source_type is None:
                    pl_ext = 'PATHLOSS_UN'
                else:
                    pl_ext = 'PATHLOSS_PS'
                
                with pyfits.open(self.files[j]) as sim:
                    if pl_ext in sim:
                        if verbose:
                            msg = f'   {self.files[j]} source_type={slit.source_type} '
                            msg += pl_ext
                            print(msg)
                        
                        phot_scl *= sim[pl_ext].data.astype(sci.dtype)[sl].flatten()
            
            self.sci[j,:] *= phot_scl
            self.var[j,:] *= phot_scl**2
            self.var_flat[j,:] *= phot_scl**2
            
        self.xslit = xslit
        self.yslit = yslit
        self.ypix = ypix
        self.wave = wave
        self.bar = bar
        
        self.xtr = xtr
        self.ytr = ytr
        self.wtr = wtr
        
        self.set_trace_coeffs(degree=2)
        

    def set_trace_coeffs(self, degree=2):
        """
        Fit a polynomial to the trace
        """
        coeffs = []
        for i in range(self.N):
            xi =  (self.xtr[i,:] - self.sh[1]/2) / self.sh[1]
            yi = self.ytr[i,:]
            oki = np.isfinite(xi + yi)
            coeffs.append(np.polyfit(xi[oki], yi[oki], degree))
        
        self.base_coeffs = coeffs
        self.trace_coeffs = [c*0. for c in coeffs]


    def update_trace_from_coeffs(self):
        """
        """
        yslit = []
        for i in range(self.N):
            xi =  (self.xtr[i,:] - self.sh[1]/2) / self.sh[1]
            _ytr = np.polyval(self.base_coeffs[i], xi)
            _ytr += np.polyval(self.trace_coeffs[i], xi)
            yslit.append((self.ypix[i,:].reshape(self.sh) - _ytr).flatten())
        
        self.yslit = np.array(yslit)


    @property
    def sky_background(self):
        if self.sky_arrays is not None:
            sky = np.interp(self.wave, *self.sky_arrays, left=-1, right=-1)
            sky[sky < 0] = np.nan
        else:
            sky = 0.
        
        return sky
    
    
    @property
    def data(self):
        
        sky = self.sky_background
        
        if self.undo_barshadow:
            return (self.sci - sky) / self.bar
        else:
            return self.sci - sky
    
    def make_diff_image(self, exp=1):
        """
        """
        ipos = self.unp[exp]

        pos  = np.nansum(self.data[ipos,:], axis=0) / np.nansum(self.mask[ipos,:], axis=0)
        vpos = np.nansum(self.var[ipos,:], axis=0) / np.nansum(self.mask[ipos,:], axis=0)

        if self.diffs:
            ineg = ~self.unp[exp]
            neg  = np.nansum(self.data[ineg,:], axis=0) / np.nansum(self.mask[ineg,:], axis=0)
            vneg = np.nansum(self.var[ineg,:], axis=0) / np.nansum(self.mask[ineg,:], axis=0)
        else:
            ineg = np.zeros(self.N, dtype=bool)
            neg = np.zeros_like(pos)
            vneg = np.zeros_like(vpos)

        diff = pos - neg
        vdiff = vpos + vneg
        
        return ipos, ineg, diff, vdiff
    
    
    def plot_2d(self, exp=1, figsize=(10,3), yoffset=0, model=None, kws=dict(vmin=-0.02, vmax=0.1, cmap='plasma')):
        """
        """
        ipos, ineg, diff, vdiff = self.make_diff_image(exp=exp)
        
        if model is None:
            fig, ax = plt.subplots(1,1, figsize=figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(3,1, figsize=(figsize[0], figsize[1]*2), sharex=True, sharey=True)
        
        if model is not None:
            vmax = np.nanpercentile(model[np.isfinite(model)], 95)*2
            kws['vmin'] = -1*vmax
            kws['vmax'] = 1.5*vmax
            
        ax = axes[0]    
        ax.imshow(diff.reshape(self.sh), aspect='auto', **kws)
        
        if model is not None:
            axes[1].imshow(model, aspect='auto', **kws)
            axes[2].imshow(diff.reshape(self.sh) - model, aspect='auto', **kws)
            
        for ax in axes:
            for j in np.where(ipos)[0]:
                xj =  (self.xtr[j,:] - self.sh[1]/2) / self.sh[1]
                _ytr = np.polyval(self.trace_coeffs[j], xj)
                _ytr += np.polyval(self.base_coeffs[j], xj)
                _ = ax.plot(_ytr + yoffset, color='k', alpha=0.3, lw=1)

            for j in np.where(ineg)[0]:
                xj =  (self.xtr[j,:] - self.sh[1]/2) / self.sh[1]
                _ytr = np.polyval(self.trace_coeffs[j], xj)
                _ytr += np.polyval(self.base_coeffs[j], xj)
                _ = ax.plot(_ytr + yoffset, color='0.8', alpha=0.3, lw=1)
        
        fig.tight_layout(pad=1)
        
        return fig
    
    
    def fit_profile(self, x0=[0., 10], exp=1, force_positive=False, bounds=((-2,2), (3, 20)), method='powell', tol=1.e-6, evaluate=False):
        """
        """
        from scipy.optimize import minimize
        global EVAL_COUNT
        ipos, ineg, diff, vdiff = self.make_diff_image(exp=exp)
        
        args = (self.wave, self.yslit, diff, vdiff, self.mask,
                ipos, ineg, self.sh, force_positive, 0)
        xargs = (self.wave, self.yslit, diff, vdiff, self.mask,
                ipos, ineg, self.sh, force_positive, 1)

        if evaluate:
            theta = x0
        else:
            EVAL_COUNT = 0
            
            _res = minimize(objfun_prof, x0, args=args, method='powell',
                        bounds=bounds, tol=tol)
            
            theta = _res.x
        
        snum, sden, smod, chi2_init = objfun_prof(x0, *xargs)
        
        snum, sden, smod, chi2_fit = objfun_prof(theta, *xargs)
        
        out = {'theta':theta,
               'chi2_init': chi2_init, 'chi2_fit': chi2_fit,
               'ipos':ipos, 'ineg':ineg,
               'diff':diff, 'vdiff':vdiff,
                'snum':snum, 'sden':sden, 'smod':smod
               }
        
        return out


    def fit_all_traces(self, niter=3, dchi_threshold=-25, ref_exp=2, verbose=True, **kwargs):
        """
        """
        tfits = {}
        
        if ref_exp is None:
            exp_groups = self.unp.values
        else:
            exp_groups = [ref_exp]
            for p in self.unp.values:
                if p not in exp_groups:
                    exp_groups.append(p)
        
        if 'evaluate' in kwargs:
            force_evaluate = kwargs['evaluate']
        else:
            force_evaluate = None
            
        for k in range(niter):
            if verbose:
                print(f'\nfit_all_traces, iter {k}\n')
                
            for i, exp in enumerate(exp_groups):
                
                if k > 0:
                    kwargs['x0'] = tfits[exp]['theta']
                
                if ref_exp is not None:
                    if exp != ref_exp:
                        kwargs['evaluate'] = True
                        kwargs['x0'] = tfits[ref_exp]['theta']
                    else:
                        kwargs['evaluate'] = False
                else:
                    kwargs['evaluate'] = False
                    
                if force_evaluate is not None:
                    kwargs['evaluate'] = force_evaluate
                    
                tfits[exp] = self.fit_single_trace(exp=exp, **kwargs)
                dchi = tfits[exp]['chi2_fit'] - tfits[exp]['chi2_init']
                
                msg = f'\n Exposure group {exp}   dchi2 = {dchi:9.1f}'
                
                if (dchi < dchi_threshold) | (kwargs['evaluate']):
                    msg += '\n'
                    for j in np.where(tfits[exp]['ipos'])[0]:
                        self.trace_coeffs[j] = tfits[exp]['trace_coeffs']
                else:
                    msg += '*\n'
                
                if verbose:
                    print(msg)
                
            self.update_trace_from_coeffs()
        
        return tfits


    def fit_single_trace(self, x0=None, sigma0=3., exp=1, force_positive=True, method='powell', tol=1.e-6, evaluate=False, degree=2, sigma_bounds=(1,20), trace_bounds=(-1,1), fix_sigma=-1, with_bounds=True, **kwargs):
        """
        """
        from scipy.optimize import minimize
        global EVAL_COUNT
        ipos, ineg, diff, vdiff = self.make_diff_image(exp=exp)
        
        base_coeffs = self.base_coeffs[np.where(ipos)[0][0]]
        
        args = (base_coeffs, self.wave, self.xslit, self.ypix, self.yslit, diff,
                vdiff, self.mask,
                ipos, ineg, self.sh, fix_sigma, force_positive, 0)
        xargs = (base_coeffs, self.wave, self.xslit, self.ypix, self.yslit, diff,
                 vdiff, self.mask,
                ipos, ineg, self.sh, fix_sigma, force_positive, 1)
        
        if x0 is None:
            if fix_sigma > 0:
                x0 = np.zeros(degree+1)
            else:
                x0 = np.append([sigma0], np.zeros(degree+1))
        
        if with_bounds:
            if fix_sigma > 0:
                bounds = [trace_bounds]*(len(x0))
            else:
                bounds = [sigma_bounds] + [trace_bounds]*(len(x0)-1)
        else:
            bounds = None
        
        if evaluate:
            theta = x0
        else:
            EVAL_COUNT = 0
            
            _res = minimize(objfun_prof_trace, x0,
                            args=args,
                            method=method,
                            tol=tol,
                            bounds=bounds,
                            )
            
            theta = _res.x
        
        snum, sden, smod, sigma, trace_coeffs, chi2_init = objfun_prof_trace(x0, *xargs)
        
        snum, sden, smod, sigma, trace_coeffs, chi2_fit = objfun_prof_trace(theta, *xargs)
        
        out = {'theta':theta,
               'sigma':sigma,
               'trace_coeffs': trace_coeffs,
               'chi2_init': chi2_init, 'chi2_fit': chi2_fit,
               'ipos':ipos, 'ineg':ineg,
               'diff':diff, 'vdiff':vdiff,
                'snum':snum, 'sden':sden, 'smod':smod,
                'force_positive': force_positive,
                'bounds':bounds,
                'method':method,
                'tol':tol,
               }
        
        return out


    def extract_spectrum(self, x0=[0, 4], bounds=None, evaluate=False):
        """
        """
        ydata = []
        wdata = []
        fdata = []
        vdata = []
        pdata = []
        sodata = []
        
        theta = np.array(x0)
        
        for ex in self.unp.values:
    
            res = self.fit_profile(exp=ex, x0=theta, bounds=bounds,
                                   evaluate=((ex > 1) | evaluate),
                                   )

            # plt.imshow(diff.reshape(self.sh) ,aspect='auto', vmin=-0.1, vmax=0.1)
    
            ysl = self.yslit[res['ipos'],:][0,:].reshape(self.sh)
            wsl = self.wave[res['ipos'],:][0,:].reshape(self.sh)
            ydata.append(ysl)
            wdata.append(wsl)
            sodata.append(np.argsort(wsl[0,:]))
    
            fdata.append(res['diff'].reshape(self.sh))
            vdata.append(res['vdiff'].reshape(self.sh))
            pdata.append(res['smod']*res['sden']/res['snum'])

            theta = res['theta']
        
        # Sorting
        sodata = np.argsort( np.hstack(wdata )[0,:])
        ydata  = np.hstack(ydata )[:,sodata]
        wdata  = np.hstack(wdata )[:,sodata]
        fdata  = np.hstack(fdata )[:,sodata]
        vdata  = np.hstack(vdata )[:,sodata]
        pdata  = np.hstack(pdata )[:,sodata]
        
        # Optimal extraction
        msk = pdata > 0
        fnum = np.nansum(fdata/vdata*pdata*msk, axis=0)
        fden = np.nansum(pdata**2/vdata*msk, axis=0)
        w0 = wdata[0,:]

        mdata = pdata*fnum/fden
        out = {'wave2d': wdata,
               'yslit2d': ydata,
               'flux2d': fdata,
               'var2d':  vdata,
               'prof2d': pdata,
               'wave2d': wdata,
               'fnum': fnum,
               'fden': fden,
               'theta': theta,
               }
        
        return out
    
    
    def plot_profile(self, exp=1, ax=None, fit_result=None, ymax=0.2):
        """
        """
        ipos, ineg, diff, vdiff = self.make_diff_image(exp=exp)
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(5,5))
        
        ax.scatter(self.yslit[0,:], diff, alpha=0.1, color='0.5')
        if fit_result is not None:
            ymax = np.nanpercentile(fit_result['smod'][np.isfinite(fit_result['smod'])], 98)*2.
            
            ax.scatter(self.yslit[0,:], fit_result['smod'], alpha=0.1, color='r')
            ax.vlines(fit_result['sigma'], -ymax, ymax, linestyle=':', color='r')
            
        ax.set_ylim(-0.5*ymax, ymax)
        ax.grid()
        fig.tight_layout(pad=1)
        
        return fig
    
    
    def fit_all_positions(self, x0=[0., 10], evaluate=False, force_positive=False, fit_all=True, fit_kwargs={}, verbose=True):
        """
        """
        results = {}
        
        theta = None
        for i, exp in enumerate(self.unp.values):
            if fit_all | (theta is None):
                results[exp] = self.fit_profile(x0=x0, exp=exp,
                                                force_positive=force_positive, 
                                                evaluate=evaluate,
                                                **fit_kwargs)
                if verbose:
                    theta = results[exp]['theta']
                    # dchi2 = results[exp]['chi2_fit'] - results[exp]['chi2_init']
                    msg = f"Fit at position {exp}:  shift = {theta[0]:.2f} "
                    msg += f"sigma = {theta[1]:.2f}" #"   dchi2: {dchi2:9.1f}"
                    print(msg)
                    
            else:
                results[exp] = self.fit_profile(x0=theta, exp=exp,
                                                evaluate=True,
                                                force_positive=force_positive,
                                                **fit_kwargs)
        
        return results