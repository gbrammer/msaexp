"""
Tools for drizzle-combining MSA spectra
"""
import glob
import warnings
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

import jwst.datamodels
import astropy.io.fits as pyfits

import grizli.utils

from . import utils

# Parameter defaults
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

FIGSIZE = (10,4)
IMSHOW_KWS = dict(vmin=-0.1,
                  vmax=None,
                  aspect='auto',
                  interpolation='nearest',
                  origin='lower',
                  cmap='cubehelix_r')


def metadata_tuple(slit):
    """
    Tuple of (msa_metadata_file, msa_metadata_id)
    
    Parameters
    ----------
    slit : `jwst.datamodels.SlitModel`
        Slitlet data object
    
    Returns
    -------
    meta : (str, str)
        Tuple of `(msa_metadata_file, msa_metadata_id)`
    
    """
    return slit.meta.instrument.msa_metadata_file, slit.meta.instrument.msa_metadata_id


def center_wcs(slit, waves, center_on_source=False, force_nypix=31, fix_slope=None, slit_center=0., center_phase=-0.5):
    """
    Derive a 2D spectral WCS centered on the expected source position along the slit
    
    Parameters
    ----------
    slit : `jwst.datamodels.SlitModel`
        Slitlet data object
    
    waves : array-like
        Target wavelength array
    
    center_on_source : bool
        Center on the source position along the slit.  If not, center on `slit_center`
        slit coordinate
    
    force_nypix : int
        Cross-dispersion size of the output 2D WCS
    
    fix_slope : float
        Fixed cross-dispersion pixel size, in units of the slit coordinate frame
    
    slit_center : float
        Define the center of the slit in the slit coordinate frame
    
    center_phase : float
        Pixel phase defining the center of the slit alignment
    
    Returns
    -------
    wcs_data : object
        Output from `msaexp.utils.build_slit_centered_wcs`
    
    offset_to_source : float
        Offset between center of the WCS and the expected source position
    
    meta : tuple
        MSA key from `msaexp.drizzle.metadata_tuple`
    
    """
    # Centered on source
    wcs_data = utils.build_slit_centered_wcs(slit, waves,
                                             force_nypix=force_nypix,
                                             center_on_source=True,
                                             get_from_ypos=False,
                                             phase=center_phase,
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
                                                    phase=center_phase,
                                                    fix_slope=fix_slope,
                                                    slit_center=slit_center)
    
        # Derived offset between source and center of slitlet
        offset_to_source = slit.drizzle_slit_offset_source - slit.drizzle_slit_offset
    
    return wcs_data, offset_to_source, metadata_tuple(slit)


def drizzle_slitlets(id, wildcard='*phot', files=None, output=None, verbose=True, drizzle_params=DRIZZLE_PARAMS, master_bkg=None, waves=None, wave_sample=1, log_step=True, force_nypix=31, center_on_source=False, center_phase=-0.5, fix_slope=None, outlier_threshold=5, sn_threshold=3, bar_threshold=0.7, err_threshold=1000, bkg_offset=5, bkg_parity=[1,-1], mask_padded=False, show_drizzled=True, show_slits=True, imshow_kws=IMSHOW_KWS, **kwargs):
    """
    Implementing more direct drizzling of multiple 2D slitlets
    
    Parameters
    ----------
    id, wildcard : object, str
        Values to search for extracted slitlet files:
    
        .. code-block:: python
            :dedent:
            
            files = glob.glob(f'{wildcard}*_{id}.fits')
    
    files : list
        Explicit list of slitlet files
    
    output : str
        Optional rootname of output figures and FITS data
    
    verbose : bool
        Verbose messaging
    
    drizzle_params : dict
        Drizzle parameters passed to `msaexp.utils.drizzle_slits_2d`
    
    master_bkg : array-like
        Master background to replace local background derived from the drizzled product
    
    waves : array-like
        Explicit arget wavelength array
    
    wave_sample, log_step : float, bool
        If `waves` not specified, generate with `msaexp.utils.get_standard_wavelength_grid`
    
    force_nypix, center_on_source, center_phase, fix_slope : int, bool, float, float
        Parameters of `msaexp.drizzle.center_wcs`
    
    outlier_threshold : int
        Outlier threshold in drizzle combination
    
    sn_threshold : float
        Mask pixels in slitlets where `data/err < sn_threshold`.  For the prism, essentially
        all pixels should have S/N > 5 from the background, so this mask can help identify
        and mask stuck-closed slitlets
    
    bar_threshold : float
        Mask pixels in slitlets where `barshadow < bar_threshold`
    
    err_threshold : float
        Mask pixels in slitlets where `err > err_threshold*median(err)`.  There are some
        strange pixels with very large uncertainties in the pipeline products.
    
    bkg_offset, bkg_parity : int, list
        Offset in pixels for defining the local background of the drizzled product, which is
        derived by rolling the data array by `bkg_offset*bkg_parity` pixels.  The standard
        three-shutter nod pattern corresponds to about 5 pixels.  An optimal combination seems
        to be ``fix_slope=0.2``, ``bkg_offset=6``.
    
    mask_padded : bool
        Mask pixels of slitlets that had been padded around the nominal MSA slitlets
    
    show_drizzled : bool
        Make a figure with `msaexp.drizzle.show_drizzled_product` showing the drizzled combined
        arrays.  If `output` specified, save to `{output}-{id}-[grating].d2d.png`.
    
    show_slits : bool
        Make a figure with `msaexp.drizzle.show_drizzled_slits` showing the individual drizzled
        slitlets.  If `output` specified, save to `{output}-{id}-[grating].slit2d.png`.
    
    imshow_kws : dict
        Keyword arguments for ``matplotlib.pyplot.imshow`` in `show_drizzled` and `show_slits`
        figures.
    
    Returns
    -------
    figs : dict
        Any figures that were created, keys are separated by grating+filter here and below
    
    data : dict
        `~astropy.io.fits.HDUList` FITS data for the drizzled output
    
    wavedata : dict
        Wavelength arrays
    
    all_slits : dict
        `SlitModel` objects for the input slitlets
    
    drz_data : dict
        3D `sci` and `wht` arrays of the drizzled slitlets that were combined into the drizzled
        stack
    
    """
    from gwcs import wcstools
    
    if files is None:
        files = glob.glob(f'{wildcard}*_{id}.fits')
        #files = glob.glob(f'*{pipeline_extension}*_{id}.fits')
        files.sort()
        
    ### Read the SlitModels
    gratings = {}
    for file in files:
        slit = jwst.datamodels.SlitModel(file)
        grating = slit.meta.instrument.grating.lower()
        filt = slit.meta.instrument.filter.lower()
        key = f'{grating}-{filt}'
        if key not in gratings:
            gratings[key] = []
    
        gratings[key].append(slit)
    
    if verbose:
        for g in gratings:
            msg = f'msaexp.drizzle.drizzle_slitlets: id={id}  {g} N={len(gratings[g])}'
            grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=verbose, 
                                     show_date=False)
            
    
    ### DQ mask
    for gr in gratings:
        slits = gratings[gr]
        for i in range(len(slits)): #[18:40]:

            slit = slits[i]

            if mask_padded:
                msg = f'Mask padded area of slitlets: {mask_padded*1:.1f}'
                grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=verbose, 
                                         show_date=False)
                
                _wcs = slit.meta.wcs
                d2s = _wcs.get_transform('detector', 'slit_frame')

                bbox = _wcs.bounding_box
                grid = wcstools.grid_from_bounding_box(bbox)
                _, sy, slam = np.array(d2s(*grid))
                msk = sy < np.nanmin(sy+mask_padded)
                msk |= sy > np.nanmax(sy-mask_padded)
                slit.data[msk] = np.nan
            
            if hasattr(slit, 'barshadow'):
                msk  = slit.barshadow < bar_threshold
                slit.data[msk] = np.nan
                
            slit.dq = (slit.dq & 1025 > 0)*1
            slit.data[slit.dq > 0] = np.nan
            
    #### Loop through gratings
    figs = {}
    data = {}
    wavedata = {}
    all_slits = {}
    drz_data = {}
    
    for gr in gratings:
        slits = gratings[gr]
        
        ## Default wavelengths
        if waves is None:
            waves = utils.get_standard_wavelength_grid(grating,
                                                       sample=wave_sample,
                                                       log_step=log_step)
        
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
            
            _center = center_wcs(slit,
                                 waves,
                                 force_nypix=force_nypix,
                                 center_on_source=center_on_source,
                                 fix_slope=fix_slope,
                                 center_phase=center_phase)
            
            wcs_data, offset_to_source, wcs_meta = _center
            to_ujy = 1.e12*slit.meta.photometry.pixelarea_steradians
            break
        
        if wcs_meta is None:
            # Run for background slits centered on slitlet if all skipped above
            _center = center_wcs(slits[0],
                                 waves,
                                 force_nypix=force_nypix,
                                 center_on_source=False,
                                 fix_slope=fix_slope,
                                 slit_center=1.0,
                                 center_phase=center_phase)
                                 
            wcs_data, offset_to_source, wcs_meta = _center
            to_ujy = 1.e12*slits[0].meta.photometry.pixelarea_steradians

        ##################
        # Now do the drizzling
        
        # FITS header metadata
        h = pyfits.Header()
        h['BKGOFF'] = bkg_offset, 'Background offset'
        h['OTHRESH'] = outlier_threshold, 'Outlier mask threshold, sigma'
        h['WSAMPLE'] = wave_sample, 'Wavelength sampling factor'
        h['LOGWAVE'] = log_step, 'Target wavelengths are log spaced'
        
        h['BUNIT'] = 'mJy'
        
        inst = slits[0].meta.instance['instrument']
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
            slit = slits[i]
            
            msg = f'msaexp.drizzle.drizzle_slitlets: '
            msg += f'{gr} {i:2} {slit.source_name:18} {slit.source_id:9}'
            msg += f' {slit.source_ypos:.3f} {files[i]} {slit.data.shape}'
            grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=verbose, 
                                     show_date=False)
    
            drz_ids.append(slit.source_name)
            
            h['EFFEXPTM'] += slit.meta.exposure.effective_exposure_time
            
            if (metadata_tuple(slit) != wcs_meta) & center_on_source:
                
                _center = center_wcs(slit,
                                     waves,
                                     force_nypix=force_nypix,
                                     center_on_source=center_on_source,
                                     fix_slope=fix_slope,
                                     center_phase=center_phase,
                                     )
                
                wcs_data, offset_to_source, wcs_meta = _center
                
                msg = f'msaexp.drizzle.drizzle_slitlets: '
                msg += f'Recenter on source ({metadata_tuple(slit)}) y={offset_to_source:.2f}'
                grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=verbose, 
                                         show_date=False)
                
                # Recalculate photometry scaling
                to_ujy = 1.e12*slit.meta.photometry.pixelarea_steradians
            
            # Slit metadata
            keys = utils.slit_metadata_to_header(slit, key=i+1, header=h)
            
            # Do the drizzling
            _waves, _header, _drz = utils.drizzle_slits_2d([slit],
                                                           build_data=wcs_data,
                                                           drizzle_params=drizzle_params)
            
            to_ujy_list.append(to_ujy)
            
            if drz is None:
                drz = _drz
            else:
                drz.extend(_drz)
        
        drz_ids = np.array(drz_ids)
        
        ## Are slitlets tagged as background?
        is_bkg = np.array([d.startswith('background') for d in drz_ids])
        is_bkg &= False # ignore, combine them all
        
        ############ Combined drizzled spectra
        
        ## First pass - max-clipped median
        sci = np.array([d.data*to_ujy_list[i]
                        for i, d in enumerate(drz)])
        err = np.array([d.err*to_ujy_list[i]
                        for i, d in enumerate(drz)])
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            scima = np.nanmax(sci, axis=0)
        
        flagged = (sci >= scima)
        flagged |= (err <= 0) | (~np.isfinite(sci)) | (~np.isfinite(err))
        flagged |= sci == 0
        flagged |= ~np.isfinite(sci/err)
        flagged |= sci/err < sn_threshold
        
        for i in range(len(drz_ids)):
            ei = err[i,:,:]
            emask =  ei > err_threshold*np.median(ei[np.isfinite(ei) & (ei > 0)])
            flagged[i,:,:] |= emask
        
        ivar = 1./err**2
        sci[flagged] = np.nan
        ivar[flagged] = 0

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            avg = np.nanmedian(sci, axis=0)
            
        avg_w = ivar.sum(axis=0)
        print('xxx init ', flagged.sum(), sci.size, avg.size)
        
        ## Subsequent passes - weighted average with outlier rejection
        for _iter in range(3):
            sci = np.array([d.data*to_ujy_list[i]
                            for i, d in enumerate(drz)])
            err = np.array([d.err*to_ujy_list[i]
                            for i, d in enumerate(drz)])
            
            #scim[(np.abs(sci - avg)/cerr > outlier_threshold)] = np.nan
            flagged = (np.abs(sci - avg)*np.sqrt(avg_w) > outlier_threshold)
            
            flagged |= (err <= 0) | (~np.isfinite(sci)) | (~np.isfinite(err))
            flagged |= sci == 0
            flagged |= ~np.isfinite(sci/err)
            flagged |= sci/err < sn_threshold
            
            for i in range(len(drz_ids)):
                ei = err[i,:,:]
                emask =  ei > err_threshold*np.median(ei[np.isfinite(ei) & (ei > 0)])
                flagged[i,:,:] |= emask
            
            ivar = 1./err**2
            sci[flagged] = 0
            ivar[flagged] = 0
                        
            # Weighted combination of unmasked pixels
            avg = (sci*ivar)[~is_bkg,:].sum(axis=0)/ivar[~is_bkg,:].sum(axis=0)
            avg_w = ivar[~is_bkg,:].sum(axis=0)
        
            # Set masked pixels to zero
            msk = ~np.isfinite(avg + avg_w)
            avg[msk] = 0
            avg_w[msk] = 0
            
        # Background by rolling full drizzled array
        bkg_num = avg*0.
        bkg_w = avg*0
        for s in bkg_parity:
            bkg_num += np.roll(avg*avg_w, s*bkg_offset, axis=0)
            bkg_w += np.roll(avg_w, s*bkg_offset, axis=0)

        bkg_w[:bkg_offset] = 0
        bkg_w[-bkg_offset:] = 0
        bkg = bkg_num/bkg_w
        
        # Use master background if supplied
        if master_bkg is not None:
            bkg = master_bkg[0]
            bkg_w = master_bkg[1]
        
        # Set masked back to nan
        avg[msk] = np.nan
        avg_w[msk] = np.nan
                
        # Trace center
        y0 = (avg.shape[0]-1)//2
        to_source = y0 + offset_to_source

        h['SRCYPIX'] = to_source, 'Expected row of source centering'
        
        # Valid data along wavelength axis
        xvalid = np.isfinite(avg).sum(axis=0) > 0
        xvalid &= nd.binary_erosion(nd.binary_dilation(xvalid, iterations=2), iterations=4)
                
        # Build HDUList
        h['EXTNAME'] = 'SCI'
        hdul = pyfits.HDUList([pyfits.PrimaryHDU()])
        hdul.append(pyfits.ImageHDU(data=avg, header=h))
        
        h['EXTNAME'] = 'WHT'
        hdul.append(pyfits.ImageHDU(data=avg_w, header=h))
        
        h['EXTNAME'] = 'BKG'
        hdul.append(pyfits.ImageHDU(data=bkg, header=h))
        
        if output is not None:
            hdul.writeto(f'{output}-{id}-{gr}.d2d.fits', overwrite=True,
                         output_verify='fix')
        
        if imshow_kws['vmax'] is None:
            vmax = np.nanpercentile(avg, 95)*2
            print('xxx', vmax)
            imshow_kws['vmax'] = vmax
            imshow_kws['vmin'] = -0.1*vmax
            reset_vmax = True
        else:
            reset_vmax = False
            
        # Make a figure
        if (show_drizzled):
            dfig = show_drizzled_product(hdul, imshow_kws=imshow_kws)
            if output is not None:
                dfig.savefig(f'{output}-{id}-{gr}.d2d.png')
        else:
            dfig = None

        if (show_slits):
            sfig = show_drizzled_slits(slits, sci, ivar, hdul, imshow_kws=imshow_kws,
                                       with_background=(show_slits > 1))
            if output is not None:
                sfig.savefig(f'{output}-{id}-{gr}.slit2d.png')
        else:
            sfig = None
        
        if reset_vmax:
            imshow_kws['vmax'] = None
            
        # Add to data dicts
        figs[gr] = dfig, sfig
        data[gr] = hdul
        wavedata[gr] = waves
        all_slits[gr] = slits
        drz_data[gr] = sci, ivar
        
    return figs, data, wavedata, all_slits, drz_data


def show_drizzled_slits(slits, sci, ivar, hdul, figsize=FIGSIZE, imshow_kws=IMSHOW_KWS, with_background=False):
    """
    Make a figure showing drizzled slitlets
    """
    avg = hdul['SCI'].data
    xvalid = np.isfinite(avg).sum(axis=0) > 0
    xr = np.arange(avg.shape[1])[xvalid]
    bkg = hdul['BKG'].data
    
    h = hdul['SCI'].header
    bkg_offset = h['BKGOFF']
    x0 = h['SRCYPIX']    
    y0 = (avg.shape[0]-1)//2
    
    msk = (ivar > 0)*1.
    msk[ivar <= 0] = np.nan
    
    fig, axes = plt.subplots(len(slits),1, figsize=figsize, sharex=True, sharey=True)
    for i, slit in enumerate(slits):
        axes[i].imshow((sci[i,:,:] - bkg*with_background)*msk[i,:,:], **imshow_kws)
        axes[i].text(0.02, 0.02*figsize[1]/figsize[0]*len(slits)*2, slit.meta.filename,
                     ha='left', va='bottom', transform=axes[i].transAxes,
                     bbox={'fc':'w', 'alpha':0.5,'ec':'None'},
                     fontsize=6)
    
    for ax in axes:
        ax.set_yticks([0,x0-bkg_offset, x0,x0+bkg_offset, avg.shape[0]])
        #ax.set_yticklabels([0, -bkg_offset, int(x0), bkg_offset, avg.shape[0]])
        ax.set_yticklabels([])
                
        ax.grid()
        ax.set_xlim(xr[0]-5, xr[-1]+5)
        ax.set_ylim(y0-2*bkg_offset, y0+2*bkg_offset)
        ax.hlines(x0, *ax.get_xlim(), color='k', linestyle='-', alpha=0.1)

    ax.set_xlabel('pixel')
    axes[0].set_title(f"{h['SRCNAME']} {h['GRATING']}-{h['FILTER']}")
    fig.tight_layout(pad=0.5)
    
    return fig


def show_drizzled_product(hdul, figsize=FIGSIZE, imshow_kws=IMSHOW_KWS):
    """
    Make a figure showing drizzled product
    """
    
    avg = hdul['SCI'].data
    xvalid = np.isfinite(avg).sum(axis=0) > 0
    xr = np.arange(avg.shape[1])[xvalid]
    bkg = hdul['BKG'].data
    
    h = hdul['SCI'].header
    bkg_offset = h['BKGOFF']
    
    x0 = h['SRCYPIX']    
    y0 = (avg.shape[0]-1)//2
    
    fig, axes = plt.subplots(3,1, figsize=figsize, sharex=True, sharey=True)
    
    axes[0].imshow(avg, **imshow_kws)
    axes[1].imshow(avg-bkg, **imshow_kws)
    axes[2].imshow(bkg, **imshow_kws)
    
    # Labels
    for i, label in enumerate(['Data','Cleaned','Background']):
        axes[i].text(0.02, 0.02*figsize[1]/figsize[0]*3*2, label,
                     ha='left', va='bottom', transform=axes[i].transAxes,
                     bbox={'fc':'w', 'alpha':0.5,'ec':'None'},
                     fontsize=8)
        
    for ax in axes:
        ax.set_yticks([0,x0-bkg_offset, x0,x0+bkg_offset, avg.shape[0]])
        ax.set_yticklabels([])
        ax.hlines(x0, *ax.get_xlim(), color='k', linestyle='-', alpha=0.1)
        
        ax.grid()
        ax.set_xlim(xr[0]-5, xr[-1]+5)
        ax.set_ylim(y0-2*bkg_offset, y0+2*bkg_offset)

    ax.set_xlabel('pixel')
    axes[0].set_title(f"{h['SRCNAME']} {h['GRATING']}-{h['FILTER']}")
    fig.tight_layout(pad=0.5)
    
    return fig


def make_optimal_extraction(waves, sci2d, wht2d, profile_slice=None,
                            prf_center=None, prf_sigma=1.0, sigma_bounds=(0.5, 2.5), 
                            center_limit=4,
                            fit_prf=True, fix_center=False, fix_sigma=False, trim=0,
                            bkg_offset=6, bkg_parity=[-1,1], verbose=True, **kwargs):
    """
    Optimal extraction from 2D arrays
    """
    import scipy.ndimage as nd
    import astropy.units as u
    from scipy.optimize import least_squares
    
    from .version import __version__ as msaexp_version
    
    sh = wht2d.shape
    yp, xp = np.indices(sh)
    
    ok = np.isfinite(sci2d*wht2d) & (wht2d > 0)
    
    if profile_slice is not None:
        if not isinstance(profile_slice, slice):
            if isinstance(profile_slice[0], int):
                # pixels
                profile_slice = slice(*profile_slice)
            else:
                # Wavelengths interpolated on pixel grid
                xpix = np.arange(sh[1])
                xsl = np.cast[int](np.round(np.interp(profile_slice, waves, xpix)))
                xsl = np.clip(xsl, 0, sh[1])
                print(f'Wavelength slice: {profile_slice} > {xsl} pix')
                profile_slice = slice(*xsl)
            
        prof1d = np.nansum((sci2d * wht2d)[:,profile_slice], axis=1) 
        prof1d /= np.nansum(wht2d[:,profile_slice], axis=1)
            
        slice_limits = profile_slice.start, profile_slice.stop
        
        pmask = ok & False
        pmask[:,profile_slice] = True
        ok &= ~pmask
        
    else:
        prof1d = np.nansum(sci2d * wht2d, axis=1) / np.nansum(wht2d, axis=1)
        slice_limits = 0, sh[1]
    
    xpix = np.arange(sh[0])
    ytrace = (sh[0]-1)/2.
    x0 = np.arange(sh[0]) - ytrace
    y0 = yp - ytrace
        
    if prf_center is None:
        prf_center = np.nanargmax(prof1d) - (sh[0]-1)/2.
        if verbose:
            print(f'Set prf_center: {prf_center} {sh} {ok.sum()}')
    
    msg = f"msaexp.drizzle.extract_from_hdul: Initial center = {prf_center:6.2f},"
    msg += f" sigma = {prf_sigma:6.2f}"
    grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False)
    
    ############# 
    #### Integrated gaussian profile
    fit_type = 3 - 2*fix_center - 1*fix_sigma
    
    wht_mask = wht2d*1
    wht_mask[~ok] = 0.
    
    if fit_type == 0:
        args = (waves, sci2d, wht_mask, prf_center, prf_sigma,
                bkg_offset, bkg_parity, 3, 1, (verbose > 1))
        pnorm, pmodel = utils.objfun_prf([prf_center, prf_sigma], *args)
        profile2d = pmodel/pnorm
    else:
        # Fit it
        if fix_sigma:
            p0 = [prf_center]
            bounds = (-center_limit, center_limit)
        elif fix_center:
            p0 = [prf_sigma]
            bounds = sigma_bounds
        else:
            p0 = [prf_center, prf_sigma]
            bounds = ((-center_limit+prf_center, sigma_bounds[0]),
                      (center_limit+prf_center, sigma_bounds[1]))
            
        args = (waves, sci2d, wht_mask, prf_center, prf_sigma,
                bkg_offset, bkg_parity, fit_type, 1, (verbose > 1))
        lmargs = (waves, sci2d, wht_mask, prf_center, prf_sigma,
                  bkg_offset, bkg_parity, fit_type, 2, (verbose > 1))

        _res = least_squares(utils.objfun_prf, p0, args=lmargs, method='trf',
                             bounds=bounds, loss='huber')
        
        pnorm, pmodel = utils.objfun_prf(_res.x, *args)
        profile2d = pmodel/pnorm
        pmask = (profile2d > 0) & np.isfinite(profile2d)
        profile2d[~pmask] = 0
        
        if fix_sigma:
            fit_center = _res.x[0]
            fit_sigma = prf_sigma
        elif fix_center:
            fit_sigma = _res.x[0]
            fit_center = prf_center
        else:
            fit_center, fit_sigma = _res.x
    
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
    
    ####### Make tables
    # Flux conversion
    # to_ujy = 1.e12*5.e-13 #drizzled_slits[0].meta.photometry.pixelarea_steradians
    to_ujy = 1.
    
    spec = grizli.utils.GTable()
    spec.meta['VERSION'] = msaexp_version, 'msaexp software version'
        
    spec.meta['TOMUJY'] = to_ujy, 'Conversion from pixel values to microJansky'
    spec.meta['PROFCEN'] = fit_center, 'PRF profile center'
    spec.meta['PROFSIG'] = fit_sigma, 'PRF profile sigma'
    spec.meta['PROFSTRT'] = slice_limits[0], 'Start of profile slice'
    spec.meta['PROFSTOP'] = slice_limits[1], 'End of profile slice'
    spec.meta['YTRACE'] = ytrace, 'Expected center of trace'
    
    prof_tab = grizli.utils.GTable()
    prof_tab.meta['VERSION'] = msaexp_version, 'msaexp software version'
    
    prof_tab['pix'] = x0
    prof_tab['profile'] = prof1d
    prof_tab['pfit'] = pfit1d
    prof_tab.meta['PROFCEN'] = fit_center, 'PRF profile center'
    prof_tab.meta['PROFSIG'] = fit_sigma, 'PRF profile sigma'
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
    
    return sci2d*to_ujy, wht2d/to_ujy**2, profile2d, spec, prof_tab


def extract_from_hdul(hdul, prf_center=None, master_bkg=None, verbose=True, **kwargs):
    """
    """
        
    if master_bkg is None:
        bkg_i = hdul['BKG'].data
    else:
        bkg_i = master_bkg
    
    sci = hdul['SCI']
    sci2d = sci.data - bkg_i
    
    wht2d = hdul['WHT'].data*1

    waves = utils.get_standard_wavelength_grid(sci.header['GRATING'].lower(),
                                               sample=sci.header['WSAMPLE'],
                                               log_step=sci.header['LOGWAVE'])
    
    if prf_center is None:
        prf_center = sci.header['SRCYPIX'] - (sci.data.shape[0]-1)/2. - 1
    
    _data = make_optimal_extraction(waves, sci2d, wht2d,
                                    prf_center=prf_center,
                                    verbose=verbose,
                                    **kwargs,
                                   )

    _sci2d, _wht2d, profile2d, spec, prof = _data

    hdul = pyfits.HDUList()
    hdul.append(pyfits.BinTableHDU(data=spec, name='SPEC1D'))

    header = sci.header

    for k in spec.meta:
        header[k] = spec.meta[k]
    
    msg = f"msaexp.drizzle.extract_from_hdul:  Output center = {header['PROFCEN']:6.2f}"
    msg += f", sigma = {header['PROFSIG']:6.2f}"
    grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False)

    hdul.append(pyfits.ImageHDU(data=sci2d, header=header, name='SCI'))
    hdul.append(pyfits.ImageHDU(data=wht2d, header=header, name='WHT'))
    hdul.append(pyfits.ImageHDU(data=profile2d, header=header, name='PROFILE'))

    hdul.append(pyfits.BinTableHDU(data=prof, name='PROF1D'))

    for k in hdul['SCI'].header:
        if k not in hdul['SPEC1D'].header:
            hdul['SPEC1D'].header[k] = hdul['SCI'].header[k]
    
    return hdul
