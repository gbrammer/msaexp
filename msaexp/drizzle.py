"""
Tools for drizzle-combining MSA spectra
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.ndimage as nd

import jwst.datamodels
import astropy.io.fits as pyfits

import grizli.utils

from . import utils

class DummyBackground(object):
    def __init__(self, level=0.0):
        self.level = level
        self.subtracted = False


DRIZZLE_PARAMS = dict(output=None,
                      single=True,
                      blendheaders=True,
                      pixfrac=1.0,
                      kernel='square',
                      fillval=0,
                      wht_type='ivm',
                      good_bits=0,
                      pscale_ratio=1.0,
                      pscale=None)


def metadata_tuple(slit):
    """
    Tuple of (msa_metadata_file, msa_metadata_id)
    """
    return slit.meta.instrument.msa_metadata_file, slit.meta.instrument.msa_metadata_id


def center_wcs(slit, waves, center_on_source=False, force_nypix=31, fix_slope=None, slit_center=0.):
    """
    """
    # Centered on source
    wcs_data = utils.build_slit_centered_wcs(slit, waves,
                                             force_nypix=force_nypix,
                                             center_on_source=True,
                                             get_from_ypos=False,
                                             phase=0.,
                                             fix_slope=fix_slope,
                                             slit_center=slit_center)

    slit.drizzle_slit_offset_source = slit.drizzle_slit_offset
    
    # Centered on slitlet
    if center_on_source:
        offset_to_source = 0.0
    else:
        wcs_data = utils.build_slit_centered_wcs(slit, waves,
                                                    force_nypix=force_nypix,
                                                    center_on_source=False,
                                                    get_from_ypos=False,
                                                    phase=0.,
                                                    fix_slope=fix_slope,
                                                    slit_center=slit_center)
    
        # Derived offset between source and center of slitlet
        if slit_center != 0:
            offset_to_source = 0.
        else:
            offset_to_source = slit.drizzle_slit_offset_source - slit.drizzle_slit_offset
        
        
    #meta = slit.meta.instrument.msa_metadata_file

    return wcs_data, offset_to_source, metadata_tuple(slit)


def drizzle_slitlets(id, pipeline_extension='phot', root='msadriz', files=None, verbose=True, master_bkg=None, log_step=True, force_nypix=31, outlier_threshold=5, bkg_offset=5, bkg_parity=[1,-1], imshow_kws=dict(vmin=-0.1, vmax=0.5, aspect='auto', cmap='cubehelix_r'), mask_padded=False, center_on_source=False, fix_slope=None, make_figure=True, stuck_slit_test={'prism':(97.5, 7)}, **kwargs):
    """
    Implementing more direct drizzling of multiple 2D cutouts
    
    stuck_slit_test : dict
        Test for stuck slits.  The keys are the grating names and the values are tuples of 
         (``percentile``, ``threshold``).  If the S/N ``percentile`` is less than ``threshold``
        then the slit is interpreted to be stuck closed.  The background in the prism spectra
        should (almost?) always produce S/N > 10 pixels.
    
    """
    from gwcs import wcstools
    
    if files is None:
        files = glob.glob(f'*{pipeline_extension}*_{id}.fits')
        #files = glob.glob(f'*{pipeline_extension}*_{id}.fits')
        files.sort()
        
    ### Read the SlitModels
    gratings = {}
    for file in tqdm(files):
        slit = jwst.datamodels.SlitModel(file)
        grating = slit.meta.instance['instrument']['grating'].lower()
        if grating not in gratings:
            gratings[grating] = []
    
        gratings[grating].append(slit)
    
    if verbose:
        for g in gratings:
            print(g, len(gratings[g]))
    
    ### DQ mask
    for gr in gratings:
        slits = gratings[gr]
        for i in range(len(slits)): #[18:40]:

            slit = slits[i]

            # msk = ~np.isfinite(slit.data)
            # msk = nd.binary_dilation(msk, iterations=3)
            # if 0:
            #     #slit.data[msk] = np.nan
            #     slit.dq[msk] |= 4
            
            if mask_padded:
                print(f'Mask padded area of slitlets: {mask_padded*1:.1f}')
                _wcs = slit.meta.wcs
                d2s = _wcs.get_transform('detector', 'slit_frame')

                bbox = _wcs.bounding_box
                grid = wcstools.grid_from_bounding_box(bbox)
                _, sy, slam = np.array(d2s(*grid))
                msk = sy < np.nanmin(sy+mask_padded)
                msk |= sy > np.nanmax(sy-mask_padded)
                slit.data[msk] = np.nan
                
            slit.dq &= 1025
            
    #### Loop through gratings
    figs = {}
    data = {}
    wavedata = {}
    
    for gr in gratings:
        slits = gratings[gr]
        
        # ### Dummy background object
        # for slit in slits:
        #     slit.meta.background = DummyBackground()
        #     slit.dq &= 1025
        
        ## Default wavelengths
        waves = utils.get_standard_wavelength_grid(grating, sample=1, log_step=log_step)
        
        ## Drizzle 2D spectra
        drz = None
        drz_ids = []

        force_yoffset = None
        wcs_data = None
        
        # Approximate for default
        to_ujy = 1.e12*5.e-13
        
        wcs_meta = None
        
        # Get offset from one science slit
        for i in range(len(slits)): #[18:40]:
            slit = slits[i]
            if 'background' in slit.source_name:
                continue
            
            wcs_data, offset_to_source, wcs_meta = center_wcs(slit,
                                                          waves,
                                                          force_nypix=force_nypix,
                                                          center_on_source=center_on_source,
                                                          fix_slope=fix_slope)
            
            to_ujy = 1.e12*slit.meta.photometry.pixelarea_steradians                                              
            break
        
        if wcs_meta is None:
            # Run for background slit
            wcs_data, offset_to_source, wcs_meta = center_wcs(slits[0],
                                                          waves,
                                                          force_nypix=force_nypix,
                                                          center_on_source=False,
                                                          fix_slope=fix_slope,
                                                          slit_center=1.0)
            
        # Now do the drizzling
        exptime = 0.
        
        # FITS header metadata
        h = pyfits.Header()
        h['BKGOFF'] = bkg_offset, 'Background offset'
        h['OTHRESH'] = outlier_threshold, 'Outlier mask threshold, sigma'
        h['LOGWAVE'] = log_step, 'Target wavelengths are log spaced'
        
        h['BUNIT'] = 'mJy'
        
        inst = slit.meta.instance['instrument']
        for k in ['grating','filter']:
            h[k] = inst[k]
        
        h['NFILES'] = len(slits), 'Number of extracted slitlets'
        h['EFFEXPTM'] = 0., 'Total effective exposure time'
        
        for slit in slits:
            h[f'SRCNAME'] = slit.source_name, 'source_name from MSA file'
            h[f'SRCID'] = slit.source_id, 'source_id from MSA file'
            h[f'SRCRA'] = slit.source_ra, 'source_ra from MSA file'
            h[f'SRCDEC'] = slit.source_dec, 'source_dec from MSA file'
            
            if slit.source_ra > 0:
                break
        
        to_ujy_list = []
        
        for i in range(len(slits)): #[18:40]:
            if verbose:
                print(f'{gr} {i:2} {slit.source_name:18} {slit.source_id:9} '
                      f'{slit.source_ypos:.3f} {files[i]} {slit.data.shape}')
    
            slit = slits[i]
            drz_ids.append(slit.source_name)
            
            h['EFFEXPTM'] += slit.meta.exposure.effective_exposure_time
            
            if (metadata_tuple(slit) != wcs_meta) & center_on_source:
                
                wcs_data, offset_to_source, wcs_meta = center_wcs(slit,
                                                              waves,
                                                              force_nypix=force_nypix,
                                                              center_on_source=center_on_source,
                                                              fix_slope=fix_slope)
                
                print(f'Recenter on source ({metadata_tuple(slit)}) y={offset_to_source:.2f}')
                
                # Recalculate photometry scaling
                to_ujy = 1.e12*slit.meta.photometry.pixelarea_steradians
                            
            h[f'SRCNAM{i}'] = slit.source_name, 'source_name from MSA file'
            h[f'SRCID{i}'] = slit.source_id, 'source_id from MSA file'
            h[f'SRCRA{i}'] = slit.source_ra, 'source_ra from MSA file'
            h[f'SRCDEC{i}'] = slit.source_dec, 'source_dec from MSA file'
            h[f'SLITID{i}'] = slit.slitlet_id, 'slitlet_id from MSA file'
            h[f'FILE{i}'] = slit.meta.instance['filename']                
            h[f'TOUJY{i}'] = to_ujy, 'Conversion to uJy/pix'
            h[f'PIXAR{i}'] = (slit.meta.photometry.pixelarea_steradians,
                             'pixelarea_steradians')
                        
            _waves, _header, _drz = utils.drizzle_slits_2d([slit], build_data=wcs_data,
                                                              drizzle_params=DRIZZLE_PARAMS)
            
            to_ujy_list.append(to_ujy)
            
            if drz is None:
                drz = _drz
            else:
                drz.extend(_drz)
            
            # if verbose:
            #     fig, ax = plt.subplots(1,1,figsize=(8,3))
            #     ax.imshow(_drz[0].data, vmin=-0.2, vmax=2, aspect='auto')
            #     ax.set_ylabel(f'{i} {slit.source_id}')
    
        drz_ids = np.array(drz_ids)
        
        ## Are slitlets tagged as background?
        is_bkg = np.array([d.startswith('background') for d in drz_ids])
        is_bkg &= False # ignore, combine them all
        
        ############ Combined drizzled spectra
        
        ## First pass
        sci = np.array([d.data*to_ujy_list[i]
                        for i, d in enumerate(drz)])
        err = np.array([d.err*to_ujy_list[i]
                        for i, d in enumerate(drz)])
        scim = np.array([d.data*to_ujy_list[i]
                         for i, d in enumerate(drz)])
        
        stuck_slits = []
        if gr in stuck_slit_test:
            _test = stuck_slit_test[gr]
            for i, d in enumerate(drz):
                ok = np.isfinite(d.err + d.data) & (d.dq & 1025 == 0) & (d.err > 0)
                sn = (d.data/d.err)[ok]
                slit_sn = np.percentile(sn, _test[0])
                if slit_sn < _test[1]:
                    h[f'STUCK{i}'] = True, f'Slit is stuck closed ({slit_sn:.2f})'
                    if verbose:
                        slit_file = slits[i].meta.instance['filename'] 
                        print(f'Slit {slit_file} appears to be stuck closed:'
                              f' {_test[0]:.1f} S/N = {slit_sn:.2f} < {_test[1]:.1f}')
                               
                    stuck_slits.append(i)
        
        for i in stuck_slits:
            sci[i] *= np.nan
            err[i] *= np.nan
            scim[i] *= np.nan
        
        scima = np.nanmax(sci, axis=0)
        scim[(scim >= scima)] = np.nan
        
        ivar = 1./err**2
        ivar[err <= 0] = 0
        ivarm = ivar*1.
        ivarm[~np.isfinite(scim)] = 0
        ivar[~np.isfinite(sci)] = 0
        ivarm[scim == 0] = 0
        ivar[sci == 0] = 0

        scim[ivarm == 0] = 0
        sci[ivar == 0] = 0

        sci[ivar == 0] = np.nan
        avg = np.nanmedian(sci, axis=0)
        avg_w = ivarm.sum(axis=0)

        ## Second pass
        sci = np.array([d.data*to_ujy_list[i]
                        for i, d in enumerate(drz)])
        err = np.array([d.err*to_ujy_list[i]
                        for i, d in enumerate(drz)])
        scim = np.array([d.data*to_ujy_list[i]
                         for i, d in enumerate(drz)])
        
        for i in stuck_slits:
            sci[i] *= np.nan
            err[i] *= np.nan
            scim[i] *= np.nan
        
        scima = np.nanmax(sci, axis=0)
        scim[((sci - avg)/err > outlier_threshold)] = np.nan
        
        ivar = 1./err**2
        ivar[err <= 0] = 0
        ivarm = ivar*1.
        ivarm[~np.isfinite(scim)] = 0
        ivar[~np.isfinite(sci)] = 0
        ivarm[scim == 0] = 0
        ivar[sci == 0] = 0

        scim[ivarm == 0] = 0
        sci[ivar == 0] = 0
        
        # Weighted combination of unmasked pixels
        avg = (scim*ivarm)[~is_bkg,:].sum(axis=0)/ivarm[~is_bkg,:].sum(axis=0)
        avg_w = ivarm[~is_bkg,:].sum(axis=0)
        
        # Set masked pixels to zero
        msk = ~np.isfinite(avg + avg_w)
        avg[msk] = 0
        avg_w[msk] = 0
        
        # Background by rolling average array
        bkg_num = avg*0.
        bkg_w = avg*0
        for s in bkg_parity:
            bkg_num += np.roll(avg*avg_w, s*bkg_offset, axis=0)
            bkg_w += np.roll(avg_w, s*bkg_offset, axis=0)

        bkg_w[:bkg_offset] = 0
        bkg_w[-bkg_offset:] = 0
        
        # Set masked back to nan
        avg[msk] = np.nan
        avg_w[msk] = np.nan

        bkg = bkg_num/bkg_w
        
        # Use master background if supplied
        if master_bkg is not None:
            bkg = master_bkg[0]
            bkg_w = master_bkg[1]
        
        # Trace center
        to_source = (avg.shape[0]-1)//2 + offset_to_source
        x0 = np.cast[int](np.round(to_source))
        
        h['SRCYPIX'] = to_source, 'Expected row of source centering'
        
        # Valid data along wavelength axis
        xvalid = np.isfinite(avg).sum(axis=0) > 0
        xvalid &= nd.binary_erosion(nd.binary_dilation(xvalid, iterations=2), iterations=4)
        
        # Make a figure
        if make_figure:
            
            fig, axes = plt.subplots(3,1, figsize=(14,8), sharex=True, sharey=True)
            axes[0].imshow(avg, **imshow_kws)
            axes[1].imshow(avg-bkg, **imshow_kws)
            axes[2].imshow(bkg, **imshow_kws)
            xr = np.arange(avg.shape[1])[xvalid]
            axes[0].set_ylabel('Data')
            axes[1].set_ylabel('Cleaned')
            axes[2].set_ylabel('Background')
        
            for ax in axes:
                ax.set_yticks([0,x0-bkg_offset, x0,x0+bkg_offset, avg.shape[0]])
                ax.set_yticklabels([0, -bkg_offset, x0, bkg_offset, avg.shape[0]])
    
                ax.grid()
                ax.set_xlim(xr[0]-5, xr[-1]+5)
                ax.set_ylim(x0-2*bkg_offset, x0+2*bkg_offset)
        
            ax.set_xlabel('pixel')
            axes[0].set_title(f"{h['SRCNAME']} {gr}")
            fig.tight_layout(pad=1)
            
        else:
            fig = None
        
        
        # Build HDUList
        h['EXTNAME'] = 'SCI'
        hdul = pyfits.HDUList([pyfits.ImageHDU(data=avg, header=h)])
        h['EXTNAME'] = 'WHT'
        hdul.append(pyfits.ImageHDU(data=avg_w, header=h))
        h['EXTNAME'] = 'BKG'
        hdul.append(pyfits.ImageHDU(data=bkg, header=h))
        
        # Add to data dicts
        figs[gr] = fig
        data[gr] = hdul
        wavedata[gr] = waves
        
    return figs, data, wavedata


def fit_profile(waves, sci2d, wht2d, profile_slice=None, prf_center=None, prf_sigma=1.0, sigma_bounds=(0.5, 2.0), fit_prf=True, fix_center=False, center_limit=4, trim=0, fix_sigma=False, verbose=True):
    """
    """
    from photutils.psf import IntegratedGaussianPRF
    from astropy.modeling.models import Polynomial2D, Gaussian1D
    from astropy.modeling.fitting import LevMarLSQFitter
    import scipy.ndimage as nd
    import astropy.units as u
    
    sh = wht2d.shape
    yp, xp = np.indices(sh)
    
    if profile_slice is not None:
        prof1d = np.nansum((sci2d * wht2d)[:,profile_slice], axis=1) 
        prof1d /= np.nansum(wht2d[:,profile_slice], axis=1)
        slice_limits = profile_slice.start, profile_slice.stop
    else:
        prof1d = np.nansum(sci2d * wht2d, axis=1) / np.nansum(wht2d, axis=1)
        slice_limits = 0, sh[1]
    
    ok = np.isfinite(prof1d)
    # Trim edges
    ok[np.where(ok)[0][:1]] = False
    ok[np.where(ok)[0][-1:]] = False
    
    ok &= (prof1d > 0)
    
    xpix = np.arange(sh[0])
    # ytrace = np.nanmedian(ytr)
    # print('xxx', ytrace, sh[0]/2)
    ytrace = sh[0]/2.
    x0 = np.arange(sh[0]) - ytrace
    y0 = yp - ytrace
    
    if prf_center is None:
        msk = ok & (np.abs(x0) < center_limit)
        prf_center = np.nanargmax(prof1d*msk) - sh[0]/2.
        if verbose:
            print(f'Set prf_center: {prf_center} {sh} {ok.sum()}')
        
    ok &= np.abs(x0 - prf_center) < center_limit
    
    prf = IntegratedGaussianPRF(x_0=0, y_0=prf_center, sigma=prf_sigma)
    gau = Gaussian1D(mean=prf_center, stddev=prf_sigma)
    
    if (fit_prf > 0) & (ok.sum() > (2 - fix_center - fix_sigma)):
        fitter = LevMarLSQFitter()
        
        if fit_prf == 2:
            gau.bounds['stddev'] = sigma_bounds
            gfit = fitter(gau, x0[ok], prof1d[ok])
            prf.y_0 = gfit.mean
            prf.sigma = gfit.stddev
            sigma_bounds = [0.8, 1.5*gfit.stddev]
            prf_center = gfit.mean
            # print(gfit)
            
        prf.fixed['x_0'] = True
        prf.fixed['y_0'] = fix_center
        prf.fixed['sigma'] = fix_sigma    
        prf.bounds['sigma'] = sigma_bounds
        prf.bounds['y_0'] = (prf_center-center_limit, prf_center+center_limit)
        
        
        pfit = fitter(prf, x0[ok]*0., x0[ok], prof1d[ok])
        
        if verbose:
            msg = f'fit_prf: center = {pfit.y_0.value:.2f}'
            msg += f'. sigma = {pfit.sigma.value:.2f}'
            print(msg)
        
    else:
        pfit = prf
    
    #plt.plot(x0, prof1d)
    #plt.plot(x0, pfit(x0*0, x0))

    # Renormalize for 1D
    pfit.flux = prf.sigma.value*np.sqrt(2*np.pi)
    

    profile2d = pfit(y0*0., y0)
    wht1d = np.nansum(wht2d*profile2d**2, axis=0)
    sci1d = np.nansum(sci2d*wht2d*profile2d, axis=0) / wht1d
    
    if profile_slice is not None:
        pfit1d = np.nansum((wht2d*profile2d*sci1d)[:,profile_slice], axis=1) 
        pfit1d /= np.nansum(wht2d[:,profile_slice], axis=1)
    else:
        pfit1d = np.nansum(profile2d*sci1d * wht2d, axis=1) / np.nansum(wht2d, axis=1)
    
    if trim > 0:
        bad = nd.binary_dilation(wht1d <= 0, iterations=trim)
        wht1d[bad] = 0
        
    sci1d[wht1d <= 0] = 0
    err1d = np.sqrt(1/wht1d)
    err1d[wht1d <= 0] = 0
    
    #plt.imshow(profile2d)
    #plt.plot(sci1d)
    
    ####### Make tables
    # Flux conversion
    to_ujy = 1. # Already applied in drizzle_slitlets
    # to_ujy = 1.e12*5.e-13 #drizzled_slits[0].meta.photometry.pixelarea_steradians
    
    spec = grizli.utils.GTable()
    spec.meta['VERSION'] = msaexp.__version__, 'msaexp software version'
        
    # spec.meta['TOMUJY'] = to_ujy, 'Conversion from pixel values to microJansky'
    spec.meta['PROFCEN'] = pfit.y_0.value, 'PRF profile center'
    spec.meta['PROFSIG'] = pfit.sigma.value, 'PRF profile sigma'
    spec.meta['PROFSTRT'] = slice_limits[0], 'Start of profile slice'
    spec.meta['PROFSTOP'] = slice_limits[1], 'End of profile slice'
    spec.meta['YTRACE'] = ytrace, 'Expected center of trace'
    
    prof_tab = grizli.utils.GTable()
    prof_tab.meta['VERSION'] = msaexp.__version__, 'msaexp software version'
    prof_tab['pix'] = x0
    prof_tab['profile'] = prof1d
    prof_tab['pfit'] = pfit1d
    prof_tab.meta['PROFCEN'] = pfit.y_0.value, 'PRF profile center'
    prof_tab.meta['PROFSIG'] = pfit.sigma.value, 'PRF profile sigma'
    prof_tab.meta['PROFSTRT'] = slice_limits[0], 'Start of profile slice'
    prof_tab.meta['PROFSTOP'] = slice_limits[1], 'End of profile slice'
    prof_tab.meta['YTRACE'] = ytrace, 'Expected center of trace'
        
    spec['wave'] = waves
    spec['wave'].unit = u.micron
    spec['flux'] = sci1d*to_ujy
    spec['err'] = err1d*to_ujy
    spec['flux'].unit = u.microJansky
    spec['err'].unit = u.microJansky
    
    msk = np.isfinite(sci2d + wht2d)
    sci2d[~msk] = 0
    wht2d[~msk] = 0
    
    return sci2d, wht2d, profile2d, spec, prof_tab    