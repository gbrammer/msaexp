"""
"""
import os
import numpy as np
import astropy.io.fits as pyfits

def summary_from_metafiles():
    """
    """
    import glob
    from grizli import utils, prep
    import astropy.io.fits as pyfits
    
    files = glob.glob('*msa.fits')
    files.sort()
    
    for file in files:
        im = pyfits.open(file)
        tab = utils.GTable(im['SOURCE_INFO'].data)
        tab.write(file.replace('.fits', '.csv'), overwrite=True)
        prep.table_to_regions(tab, file.replace('.fits','.reg'), 
                              comment=tab['source_name'], size=0.1)


GRATING_LIMITS = {'prism': [0.58, 5.3, 0.01], 
                  'g140m': [0.7, 1.9, 0.00063], 
                  'g235m': [1.6, 3.1, 0.00106], 
                  'g395m': [2.8, 5.2, 0.00179],
                  'g140h': [0.7, 1.9, 0.000238], 
                  'g235h': [1.6, 3.1, 0.000396], 
                  'g395h': [2.8, 5.2, 0.000666]}

def get_standard_wavelength_grid(grating, sample=1, free_prism=True, log_step=False, grating_limits=GRATING_LIMITS):
    """
    Get a fixed wavelength grid for a given grating
    
    Parameters
    ----------
    grating : str
        NIRSpec grating name
    
    sample : float
        Oversample factor relative to the grating default step
    
    free_prism : bool
        Use irregular prism steps
    
    log_step : bool
        Use logarithmic steps
    
    grating_limits : dict
        Default grating limits and steps
    
    Returns
    -------
    target_waves : array
        Target wavelength grid
    """
    import grizli.utils
    
    if grating.lower() not in grating_limits:
        return None
        
    gr = grating_limits[grating.lower()]

    if (grating.lower() == 'prism') & free_prism:
        _path = os.path.join(os.path.dirname(__file__), 'data')
        _disp_file = f'{_path}/jwst_nirspec_{grating.lower()}_disp.fits'
        disp = grizli.utils.read_catalog(_disp_file)
        
        target_waves = [gr[0]]
        while target_waves[-1] < gr[1]:
            dw = np.interp(target_waves[-1], disp['WAVELENGTH'], disp['DLDS'])
            target_waves.append(target_waves[-1]+dw/sample)
        
        target_waves = np.array(target_waves)
    else:
        if log_step:
            # Step is dlam/l0/sample where l0 is the center of the grid
            dlog = gr[2] / (gr[0]+gr[1]) * 2. / sample
            target_waves = np.exp(np.arange(*np.log(gr[:2]), dlog))
        else:
            target_waves = np.arange(*gr[:2], gr[2]/sample)
    
    return target_waves


def build_regular_wavelength_wcs(slits, pscale_ratio=1, keep_wave=False, wave_scale=1, log_wave=False, refmodel=None, verbose=True, wave_range=None, wave_step=None, wave_array=None, **kwargs):
    """
    Create a spatial/spectral WCS covering footprint of the input
    
    Refactored from 
    `jwst.resample.resample_spec.ResampleSpecData.build_nirspec_output_wcs`      
    for regularly-spaced grids in linear or log wavelength.
    
    Parameters
    ----------
    wave_scale : float
        Factor by which to scale the wavelength grid relative to the median
        dispersion of a single slit
    
    log_wave : bool
        Wavelength grid is evenly spaced in log(wave)
    
    Returns
    -------
    target_waves : array-like
        Wavelength grid
    
    header : `~astropy.io.fits.Header`
        WCS header
        
    data_size : tuple
        Image dimensions of output array
    
    output_wcs : `gwcs.wcs.WCS`
        Spectroscopic WCS
    
    """
    import astropy.units as u
    from astropy import coordinates as coord
    from astropy.modeling.models import (
        Mapping, Tabular1D, Linear1D, Pix2Sky_TAN, RotateNative2Celestial, Identity
    )
    
    from gwcs import wcstools, WCS
    from gwcs import coordinate_frames as cf
    
    from jwst.resample.resample_spec import (
            ResampleSpecData, _find_nirspec_output_sampling_wavelengths,
            resample_utils
    )
        
    _max_virtual_slit_extent = ResampleSpecData._max_virtual_slit_extent
    
    all_wcs = [m.meta.wcs for m in slits if m is not refmodel]
    if refmodel:
        all_wcs.insert(0, refmodel.meta.wcs)
    else:
        refmodel = slits[0]

    # make a copy of the data array for internal manipulation
    refmodel_data = refmodel.data.copy()
    # renormalize to the minimum value, for best results when
    # computing the weighted mean below
    refmodel_data -= np.nanmin(refmodel_data)

    # save the wcs of the reference model
    refwcs = refmodel.meta.wcs

    # setup the transforms that are needed
    s2d = refwcs.get_transform('slit_frame', 'detector')
    d2s = refwcs.get_transform('detector', 'slit_frame')
    s2w = refwcs.get_transform('slit_frame', 'world')

    # estimate position of the target without relying on the meta.target:
    # compute the mean spatial and wavelength coords weighted
    # by the spectral intensity
    bbox = refwcs.bounding_box
    grid = wcstools.grid_from_bounding_box(bbox)
    _, s, lam = np.array(d2s(*grid))
    sd = s * refmodel_data
    ld = lam * refmodel_data
    good_s = np.isfinite(sd)
    if np.any(good_s):
        total = np.sum(refmodel_data[good_s])
        wmean_s = np.sum(sd[good_s]) / total
        wmean_l = np.sum(ld[good_s]) / total
    else:
        wmean_s = 0.5 * (refmodel.slit_ymax - refmodel.slit_ymin)
        wmean_l = d2s(*np.mean(bbox, axis=1))[2]

    # transform the weighted means into target RA/Dec
    targ_ra, targ_dec, _ = s2w(0, wmean_s, wmean_l)

    target_waves = _find_nirspec_output_sampling_wavelengths(
                       all_wcs,
                       targ_ra, targ_dec
                   )
    target_waves = np.array(target_waves)
    orig_lam = target_waves*1
    
    # Set linear dispersion
    if wave_range is None:
        lmin = np.nanmin(target_waves)
        lmax = np.nanmax(target_waves)
    else:
        lmin, lmax = wave_range
        
    if wave_step is None:
        dlam = np.nanmedian(np.diff(target_waves))*wave_scale
    else:
        dlam = wave_step*1.
    
    if log_wave:
        msg = f'Set log(lam) grid (dlam/lam={dlam/lmin*3.e5:.0f} km/s)'
        
        lam_step = dlam/lmin
        target_waves = np.exp(np.arange(np.log(lmin), np.log(lmax), lam_step))
    else:
        msg = f'Set linear wave grid (dlam={dlam*1.e4:.1f} Ang)'
        
        lam_step = dlam    
        target_waves = np.arange(lmin, lmax, lam_step)
    
    if keep_wave:
        target_waves = orig_lam
        
        if keep_wave == 2:
            msg = 'Oversample original wavelength grid x 2'
                
            # Oversample by x2
            dl = np.diff(target_waves)
            target_waves = np.append(target_waves, target_waves[:-1]+dl/2.)
            target_waves.sort()
            
        lmin = np.nanmin(target_waves)
        lmax = np.nanmax(target_waves)
        dlam = np.nanmedian(np.diff(target_waves))
        
    if wave_array is not None:
        msg = f'Set user-defined wavelength grid (size={wave_array.size})'
        
        target_waves = wave_array*1.
        lmin = np.nanmin(target_waves)
        lmax = np.nanmax(target_waves)
        dlam = np.nanmedian(np.diff(target_waves))
        
    if verbose:
        print('build_regular_wavelength_wcs: ' + msg)
        
    n_lam = target_waves.size
    if not n_lam:
        raise ValueError("Not enough data to construct output WCS.")

    x_slit = np.zeros(n_lam)
    lam = 1e-6 * target_waves

    # Find the spatial pixel scale:
    y_slit_min, y_slit_max = _max_virtual_slit_extent(None, all_wcs,
                                                      targ_ra, targ_dec)

    nsampl = 50
    xy_min = s2d(
        nsampl * [0],
        nsampl * [y_slit_min],
        lam[(tuple((i * n_lam) // nsampl for i in range(nsampl)), )]
    )
    xy_max = s2d(
        nsampl * [0],
        nsampl * [y_slit_max],
        lam[(tuple((i * n_lam) // nsampl for i in range(nsampl)), )]
    )

    good = np.logical_and(np.isfinite(xy_min), np.isfinite(xy_max))
    if not np.any(good):
        raise ValueError("Error estimating output WCS pixel scale.")

    xy1 = s2d(x_slit, np.full(n_lam, refmodel.slit_ymin), lam)
    xy2 = s2d(x_slit, np.full(n_lam, refmodel.slit_ymax), lam)
    xylen = np.nanmax(np.linalg.norm(np.array(xy1) - np.array(xy2), axis=0)) + 1
    pscale = (refmodel.slit_ymax - refmodel.slit_ymin) / xylen

    # compute image span along Y-axis (length of the slit in the detector plane)
    # det_slit_span = np.linalg.norm(np.subtract(xy_max, xy_min))
    det_slit_span = np.nanmax(np.linalg.norm(np.subtract(xy_max, xy_min), axis=0))
    ny = int(np.ceil(det_slit_span * pscale_ratio + 0.5)) + 1

    border = 0.5 * (ny - det_slit_span * pscale_ratio) - 0.5

    if xy_min[1][1] < xy_max[1][1]:
        y_slit_model = Linear1D(
            slope=pscale / pscale_ratio,
            intercept=y_slit_min - border * pscale * pscale_ratio
        )
    else:
        y_slit_model = Linear1D(
            slope=-pscale / pscale_ratio,
            intercept=y_slit_max + border * pscale * pscale_ratio
        )

    # extrapolate 1/2 pixel at the edges and make tabular model w/inverse:
    lam = lam.tolist()
    pixel_coord = list(range(n_lam))

    if len(pixel_coord) > 1:
        # left:
        slope = (lam[1] - lam[0]) / pixel_coord[1]
        lam.insert(0, -0.5 * slope + lam[0])
        pixel_coord.insert(0, -0.5)
        # right:
        slope = (lam[-1] - lam[-2]) / (pixel_coord[-1] - pixel_coord[-2])
        lam.append(slope * (pixel_coord[-1] + 0.5) + lam[-2])
        pixel_coord.append(pixel_coord[-1] + 0.5)

    else:
        lam = 3 * lam
        pixel_coord = [-0.5, 0, 0.5]

    wavelength_transform = Tabular1D(points=pixel_coord,
                                     lookup_table=lam,
                                     bounds_error=False, fill_value=np.nan)
    wavelength_transform.inverse = Tabular1D(points=lam,
                                             lookup_table=pixel_coord,
                                             bounds_error=False,
                                             fill_value=np.nan)
    
    data_size = (ny, len(target_waves))

    # Construct the final transform
    mapping = Mapping((0, 1, 0))
    mapping.inverse = Mapping((2, 1))
    out_det2slit = mapping | Identity(1) & y_slit_model & wavelength_transform

    # Create coordinate frames
    det = cf.Frame2D(name='detector', axes_order=(0, 1))
    slit_spatial = cf.Frame2D(name='slit_spatial', axes_order=(0, 1),
                              unit=("", ""), axes_names=('x_slit', 'y_slit'))
    spec = cf.SpectralFrame(name='spectral', axes_order=(2,),
                            unit=(u.micron,), axes_names=('wavelength',))
    slit_frame = cf.CompositeFrame([slit_spatial, spec], name='slit_frame')
    sky = cf.CelestialFrame(name='sky', axes_order=(0, 1),
                            reference_frame=coord.ICRS())
    world = cf.CompositeFrame([sky, spec], name='world')

    pipeline = [(det, out_det2slit), (slit_frame, s2w), (world, None)]
    output_wcs = WCS(pipeline)

    # Compute bounding box and output array shape.  Add one to the y (slit)
    # height to account for the half pixel at top and bottom due to pixel
    # coordinates being centers of pixels
    bounding_box = resample_utils.wcs_bbox_from_shape(data_size)
    output_wcs.bounding_box = bounding_box
    output_wcs.array_shape = data_size
    
    header = pyfits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = data_size[1]
    header['NAXIS2'] = data_size[0]
    
    if log_wave:
        header['CTYPE1'] = 'WAVE-LOG'
    else:
        header['CTYPE1'] = 'WAVE'
    
    header['CRPIX1'] = 1
    header['CRVAL1'] = lmin
    header['CDELT1'] = lam_step
    header['CUNIT1'] = 'um'

    header['CRPIX2'] = data_size[0]//2+1
    header['CRVAL2'] = 0.
    header['CDELT2'] = pscale / pscale_ratio
    header['CUNIT2'] = 'arcsec'
    header['DLAM'] = dlam, 'Wavelength step'
    header['LMIN'] = lmin, 'Min wavelength'
    header['LMAX'] = lmax, 'Max wavelength'
    
    return target_waves, header, data_size, output_wcs


DRIZZLE_PARAMS = dict(output=None,
                      single=False,
                      blendheaders=True,
                      pixfrac=1.0,
                      kernel='square',
                      fillval=0,
                      wht_type='ivm',
                      good_bits=0,
                      pscale_ratio=1.0,
                      pscale=None)

def drizzle_slits_2d(slits, drizzle_params=DRIZZLE_PARAMS, **kwargs):
    """
    Run `jwst.resample.resample_spec import ResampleSpecData` on a list of 
    List of `jwst.datamodels.slit.SlitModel` objects.
        
    Parameters
    ----------
    slits : list
        List of `jwst.datamodels.SlitModel` objects
    
    drizzle_params : dict
        Drizzle parameters passed on initialization of the 
        `jwst.resample.resample_spec.ResampleSpecData` step
    
    kwargs : dict
        Passed to `build_regular_wavelength_wcs` for building the wavelength
        grid.
        
    Returns
    -------
    target_waves : array
        The target wavelength array
    
    header : `~astropy.io.fits.Header`
        WCS header
    
    drizzled_slits : `jwst.datamodels.ModelContainer`
        The products of `ResampleSpecData.do_drizzle()` on the individual 
        slitlets
    
    """
    from jwst.datamodels import SlitModel, ModelContainer
    from jwst.resample.resample_spec import ResampleSpecData
    
    if slits in [None, []]:
        return None
    
    # Build WCS
    if 'pscale_ratio' in drizzle_params:
        kwargs['pscale_ratio'] = drizzle_params['pscale_ratio']
    
    _data = build_regular_wavelength_wcs(slits, **kwargs)
    target_waves, header, data_size, output_wcs = _data
    
    # Own drizzle single to get variances
    if 'single' in drizzle_params:
        run_single = drizzle_params['single']
    else:
        run_single = False
    
    if run_single:
        for i, s in enumerate(slits):
            container = ModelContainer()
            container.append(s)

            if i == 0:
                step = ResampleSpecData(container, **drizzle_params)
                step.single = False
            
                step.data_size = data_size
                step.output_wcs = output_wcs

                step.blank_output = SlitModel(tuple(step.output_wcs.array_shape))
                step.blank_output.update(step.input_models[0])
                step.blank_output.meta.wcs = step.output_wcs
            else:
                step.input_models = container
                
            drizzled_slits = step.do_drizzle()

    else:
        
        container = ModelContainer()
        for s in slits:
            container.append(s)
        
        step = ResampleSpecData(container, **drizzle_params)
    
        step.data_size = data_size
        step.output_wcs = output_wcs
    
        step.blank_output = SlitModel(tuple(step.output_wcs.array_shape))
        step.blank_output.update(step.input_models[0])
        step.blank_output.meta.wcs = step.output_wcs
    
        drizzled_slits = step.do_drizzle()
    
    return target_waves, header, drizzled_slits


def combine_2d_with_rejection(drizzled_slits, sigma=5, grow=0, trim=2, prf_center=None, prf_sigma=1.0, center_limit=4, fit_prf=True, fix_center=False, fix_sigma=False, verbose=True, profile_slice=None, **kwargs):
    """
    Combine single drizzled arrays with outlier detection
    
    Parameters
    ----------
    drizzled_slits : list
        List of drizzled 2D slitlets from `msapipe.utils.drizzle_slits_2d`, 
        i.e., from `msapipe.pipe.NirspecPipeline.get_background_slits`.  
        
        **N.B.** This can be a concatenation from mulitple `NirspecPipeline`
        objects, e.g., multiple visits or combining across the two detectors
        to a single output product.
    
    sigma : float
        Outlier threshold (absolute value) for identifying outliers between the
        different slitlets, e.g., bad pixels and cosmic rays
    
    grow : int
        not used
    
    trim : int
        Number of pixels to trim from the edges of the output spectra
    
    prf_center : float
        Center of the extraction profile relative to the center of the 2D 
        array
    
    prf_sigma : float
        Sigma width of the GaussianPRF profile
    
    center_limit : float
        Tolerance to search/fit for the profile relative to ``prf_center``

    fit_prf : bool
        Fit updates to the prf parameters
    
    fix_center : bool
        If `fit_prf`, this sets fixing the center of the fitted profile
    
    fix_sigma : bool
        If `fit_prf`, this sets fixing the width of the fitted profile
    
    Returns
    -------
    sci2d : array
        Combined 2D spectrum
    
    wht2d : array
        Inverse variance weights of the combination
    
    profile2d : array
        Profile used for the optimal 1D extraction
    
    spec : `astropy.table.Table`
        1D extraction
    
    """
    from photutils.psf import IntegratedGaussianPRF
    from astropy.modeling.models import Polynomial2D
    from astropy.modeling.fitting import LevMarLSQFitter
    import scipy.ndimage as nd
    
    import astropy.units as u
    
    import grizli.utils
    
    sci = np.array([s.data for s in drizzled_slits])
    dq = np.array([s.dq for s in drizzled_slits])
    err = np.array([s.err*(1+np.sqrt(2))/np.sqrt(2) for s in drizzled_slits])
    
    ivar = 1/err**2
    dq[(sci == 0) | (~np.isfinite(sci))] |= 1
    sci[dq > 0] = np.nan
    med = np.nanmedian(sci, axis=0)
    
    ivar[(dq > 0) | (err <= 0)] = 0
    
    if np.nanmax(ivar) == 0:
        mad = 1.48*np.nanmedian(np.abs(sci-med), axis=0)
        for i in range(ivar.shape[0]):
            ivar[i,:,:] = 1/mad**2.
        
        ivar[(dq > 0) | (~np.isfinite(mad))] = 0.
        
    bad = np.abs(sci-med)*np.sqrt(ivar) > sigma
    bad |= dq > 0

    sci[bad] = 0
    ivar[bad] = 0.
    
    wht2d = (ivar*(~bad)).sum(axis=0)
    sci2d = (sci*ivar*(~bad)).sum(axis=0) / wht2d
    sci2d[wht2d == 0] = 0
    wht2d[wht2d <= 0] = 0.
    
    sh = wht2d.shape
    yp, xp = np.indices(sh)
    
    if profile_slice is not None:
        prof1d = np.nansum((sci2d * wht2d)[:,profile_slice], axis=1) 
        prof1d /= np.nansum(wht2d[:,profile_slice], axis=1)
    else:
        prof1d = np.nansum(sci2d * wht2d, axis=1) / np.nansum(wht2d, axis=1)
    
    ok = np.isfinite(prof1d) & (prof1d > 0)
    
    x0 = np.arange(sh[0]) - sh[0]/2.
    y0 = yp - sh[0]/2.
    
    if prf_center is None:
        msk = ok & (np.abs(x0) < center_limit)
        prf_center = np.nanargmax(prof1d*msk) - sh[0]/2.
        if verbose:
            print(f'Set prf_center: {prf_center} {sh} {ok.sum()}')
        
    ok &= np.abs(x0 - prf_center) < center_limit
    
    prf = IntegratedGaussianPRF(x_0=0, y_0=prf_center, sigma=prf_sigma)
    
    if fit_prf & (ok.sum() > (2 - fix_center - fix_sigma)):
        fitter = LevMarLSQFitter()
        prf.fixed['x_0'] = True
        prf.fixed['y_0'] = fix_center
        prf.fixed['sigma'] = fix_sigma    
        pfit = fitter(prf, x0[ok]*0., x0[ok], prof1d[ok])
        
        if verbose:
            msg = f'fit_prf: center = {pfit.y_0.value:.2f}'
            msg += f'. sigma = {pfit.sigma.value:.2f}'
            print(msg)
        
    else:
        pfit = prf
    
    # Renormalize
    pfit.flux = 1.0
    
    profile2d = pfit(y0*0., y0)
    wht1d = (wht2d*profile2d**2).sum(axis=0)
    sci1d = (sci2d*wht2d*profile2d).sum(axis=0) / wht1d
    
    if trim > 0:
        bad = nd.binary_dilation(wht1d <= 0, iterations=trim)
        wht1d[bad] = 0
        
    sci1d[wht1d <= 0] = 0
    err1d = np.sqrt(1/wht1d)
    err1d[wht1d <= 0] = 0
    
    # Flux conversion
    to_ujy = 1.e12*drizzled_slits[0].meta.photometry.pixelarea_steradians
    
    spec = grizli.utils.GTable()
    spec.meta['NCOMBINE'] = len(drizzled_slits)
    spec.meta['MASKSIG'] = sigma, 'Mask sigma'
    
    spec.meta['TOMUJY'] = to_ujy, 'Conversion from pixel values to microJansky'
    spec.meta['PROFCEN'] = pfit.y_0.value, 'PRF profile center'
    spec.meta['PROFSIG'] = pfit.sigma.value, 'PRF profile sigma'
    
    met = drizzled_slits[0].meta.instrument.instance
    for k in ['detector','filter','grating']:
        spec.meta[k] = met[k]
        
    sh = drizzled_slits[0].data.shape
    xpix = np.arange(sh[1])
    ypix = np.zeros(sh[1]) + sh[0]/2 + pfit.y_0.value
    
    _wcs = drizzled_slits[0].meta.wcs
    ri, di, spec['wave'] = _wcs.forward_transform(xpix, ypix)
    
    spec['wave'].unit = u.micron
    spec['flux'] = sci1d*to_ujy
    spec['err'] = err1d*to_ujy
    spec['flux'].unit = u.microJansky
    spec['err'].unit = u.microJansky
    
    return sci2d*to_ujy, wht2d/to_ujy**2, profile2d, spec


def drizzle_2d_pipeline(slits, output_root=None, standard_waves=True, drizzle_params=DRIZZLE_PARAMS, **kwargs):
    """
    Drizzle list of background-subtracted slitlets
    
    Parameters
    ----------
    slits : list
        List of `jwst.datamodels.SlitModel` objects
    
    drizzle_params : dict
        Drizzle parameters passed on initialization of the 
        `jwst.resample.resample_spec.ResampleSpecData` step
    
    kwargs : dict
        Passed to `build_regular_wavelength_wcs` for building the wavelength
        grid.
    
    Returns
    -------
    hdul : `astropy.io.fits.HDUList`
        FITS HDU list with extensions ``SPEC1D``, ``SCI``, ``WHT``, ``PROFILE``
    
    """
    
    if (standard_waves > 0) & ('wave_array' not in kwargs):
        # Get fixed wavelength grid
        waves = get_standard_wavelength_grid(slits[0].meta.instrument.grating, 
                                    sample=standard_waves)
        _data0 = drizzle_slits_2d(slits,
                                  drizzle_params=drizzle_params,
                                  wave_array=waves, **kwargs)
    else:                                
        _data0 = drizzle_slits_2d(slits,
                                  drizzle_params=drizzle_params,
                                  **kwargs)
                                  
    target_wave, header, drizzled_slits = _data0
    
    _data1 = combine_2d_with_rejection(drizzled_slits, **kwargs)
    sci2d, wht2d, profile2d, spec = _data1
    
    hdul = pyfits.HDUList()
    hdul.append(pyfits.BinTableHDU(data=spec, name='SPEC1D'))
    
    # Add 2D arrays
    header['BUNIT'] = 'ujy'
    
    for k in spec.meta:
        header[k] = spec.meta[k]

    hdul.append(pyfits.ImageHDU(data=sci2d, header=header, name='SCI'))
    hdul.append(pyfits.ImageHDU(data=wht2d, header=header, name='WHT'))
    hdul.append(pyfits.ImageHDU(data=profile2d, header=header, name='PROFILE'))

    if output_root is not None:
        hdul.write(f'{output_root}.driz.fits', overwrite=True)
        
    return hdul


def drizzled_hdu_figure(hdul, tick_steps=None, xlim=None, subplot_args=dict(figsize=(10, 4), height_ratios=[1,3], width_ratios=[10,1]), cmap='plasma_r', ymax=None, z=None):
    """
    Figure showing drizzled hdu
    """
    import matplotlib.pyplot as plt
    import grizli.utils
    
    sp = grizli.utils.GTable(hdul['SPEC1D'].data)
    nx = len(sp)

    # Extraction profile
    pden = np.nansum(hdul['WHT'].data, axis=1)
    snum = np.nansum(hdul['SCI'].data*hdul['WHT'].data, axis=1)
    pnum = np.nansum(hdul['PROFILE'].data*hdul['WHT'].data*sp['flux'], axis=1)
    
    fig, a2d = plt.subplots(2,2, **subplot_args)
    axes = [a2d[0][0], a2d[1][0]]
    
    if ymax is None:
        ymax = np.nanpercentile(sp['flux'][sp['err'] > 0], 90)*2
        ymax = np.maximum(ymax, 7*np.median(sp['err'][sp['err'] > 0]))
    
    yscl = hdul['PROFILE'].data.max()
    
    axes[0].imshow(hdul['SCI'].data/yscl, vmin=-0.1*ymax, vmax=ymax, 
                   aspect='auto', cmap=cmap, 
                   interpolation='nearest')
                   
    axes[0].set_yticklabels([])
    y0 = hdul['PROFILE'].header['NAXIS2']/2. + hdul['SPEC1D'].header['PROFCEN']
    
    prof = pnum/pden
    sprof = snum/pden
    pmax = np.nanmax(prof)
        
    ap = a2d[0][1]
    ap.step(sprof/pmax, np.arange(len(prof)), color='k', where='mid', alpha=0.8, lw=2)
    ap.step(prof/pmax, np.arange(len(prof)), color='r', where='mid', alpha=0.5, lw=1)
    ap.fill_betweenx(np.arange(len(prof)), prof*0., prof/pmax, color='r', alpha=0.2, lw=1)
    ap.set_ylim(axes[0].get_ylim())
    ap.set_yticks([y0-4, y0+4])
    ap.grid()
    ap.set_xlim(-0.5, 1)
    ap.set_xticklabels([])
    ap.set_yticklabels([])
    a2d[1][1].set_visible(False)
    
    axes[0].set_yticks([y0-4, y0+4])
    
    axes[1].step(np.arange(len(sp)), sp['flux'], where='mid', color='0.5', alpha=0.9)
    axes[1].step(np.arange(len(sp)), sp['err'], where='mid', color='r', alpha=0.2)
    xl = axes[1].get_xlim()

    axes[1].fill_between(xl, [-ymax, -ymax], [0, 0], color='0.8', alpha=0.1)
    axes[1].set_xlim(*xl)
    
    if z is not None:
        cc = grizli.utils.MPL_COLORS
        for w, c in zip([3727, 4860, 4980, 6565, 9070, 9530, 1.094e4, 1.282e4, 
                         1.875e4, 1.083e4], 
                    [cc['purple'], cc['g'], cc['b'], cc['g'], 'darkred', 'darkred', 
                     cc['pink'], cc['pink'], cc['pink'], cc['orange']]):
            wz = w*(1+z)/1.e4
            dw = 70*(1+z)/1.e4
            wx = np.interp([wz-dw, wz+dw], sp['wave'], np.arange(len(sp)))

            axes[1].fill_between(wx, [0,0], [100,100], 
                            color=c, alpha=0.07, zorder=-100)
        
    axes[1].set_ylim(-0.1*ymax, ymax)
    axes[1].set_xlabel(r'$\lambda_\mathrm{obs}$ [$\mu$m]')
    axes[1].set_ylabel(r'$f_\nu$ [$\mu$Jy]')
                       
    if tick_steps is None:
        if hdul[1].header['GRATING'] == 'PRISM':
            minor = 0.1
            major = 0.5
        else:
            minor = 0.05
            major = 0.25
    else:
        major, minor = tick_steps
        
    xt = np.arange(0.5, 5.5, major)
    xtm = np.arange(0.5, 5.5, minor)
    
    if hdul[1].header['GRATING'] == 'PRISM':
        xt = np.append([0.7], xt)
        
    xt = xt[(xt > sp['wave'].min()) & (xt < sp['wave'].max())]
    xv = np.interp(xt, sp['wave'], np.arange(len(sp)))

    xtm = xtm[(xtm > sp['wave'].min()) & (xtm < sp['wave'].max())]
    xvm = np.interp(xtm, sp['wave'], np.arange(len(sp)))
    
    for ax in axes:
        ax.set_xticks(xvm, minor=True)
        ax.set_xticks(xv, minor=False)
        ax.set_xticklabels(xt)
        ax.grid()
        
    if xlim is not None:
        xvi = np.interp(xlim, sp['wave'], np.arange(len(sp)))
        for ax in axes:
            ax.set_xlim(*xvi)
    else:
        for ax in axes:
            ax.set_xlim(0, nx)
            
    fig.tight_layout(pad=0.5)

    return fig


def extract_all():
    """
    demo 
    """
    from importlib import reload
    
    from msaexp import pipeline
    groups = pipeline.exposure_groups()
    obj = {}
    
    for g in groups:
        if '395m' not in g:
            continue
            
        print(f'\n\nInitialize {g}\n\n')
        obj[g] = pipeline.NirspecPipeline(g)
        obj[g].full_pipeline(run_extractions=False)
        obj[g].set_background_slits()
    
    import msaexp.utils
    reload(msaexp.utils)
    
    groups = ['jw02756001001-01-clear-prism-nrs1', 'jw02756001001-02-clear-prism-nrs1']
    gg = groups
    
    key = '2756_80075'
    bad = []
    
    slits=[]
    step='bkg'
    
    for g in gg:
        if g not in obj:
            continue
            
        self = obj[g]
        
        if key not in self.slitlets:
            continue
        
        sl = self.slitlets[key]
        mode = self.mode
        
        _data = self.extract_all_slits(keys=[key], yoffset=None, prof_sigma=None, skip=bad,
                                       fit_profile_params=None, close=False)
        
        si = self.get_background_slits(key, step='bkg', check_background=True)
        
        if si is not None:
            for s in si:
                slits.append(s)
    
    kwargs = {'keep_wave':True}
    kwargs = {'keep_wave':False, 'wave_range':[0.6, 5.3], 'wave_step':0.0005, 
              'log_wave':True}
              
    # f290lp g395m
    kwargs = {'keep_wave':False, 'wave_range':[2.87, 5.2], 'wave_step':0.001, 
              'log_wave':False}
    
    #kwargs = {'keep_wave':1} #'wave_array':wx}
    
    prf_kwargs = {'prf_center':None, 'prf_sigma':1.0, 'prf_fit':True}
    
    drizzle_params = dict(output=None,
                          single=True,
                          blendheaders=True,
                          pixfrac=0.5,
                          kernel='square',
                          fillval=0,
                          wht_type='ivm',
                          good_bits=0,
                          pscale_ratio=1.0,
                          pscale=None)
                          
    wave, header, drizzled_slits = msaexp.utils.drizzle_slits_2d(slits, drizzle_params=drizzle_params, **kwargs)
    sci2d, wht2d, profile2d, spec = msaexp.utils.combine_2d_with_rejection(drizzled_slits, sigma=10, **prf_kwargs)
    
    for k in ['name','ra','dec']:
        spec.meta[k] = sl[f'source_{k}']
    
    _fitsfile = f'{mode}-{key}.driz.fits'
    
    spec.write(_fitsfile, overwrite=True)
    
    with pyfits.open(_fitsfile, mode='update') as hdul:
        
        hdul[1].header['EXTNAME'] = 'SPEC1D'
        
        for k in spec.meta:
            header[k] = spec.meta[k]
            
        hdul.append(pyfits.ImageHDU(data=sci2d, header=header, name='SCI'))
        hdul.append(pyfits.ImageHDU(data=wht2d, header=header, name='WHT'))
        hdul.append(pyfits.ImageHDU(data=profile2d, header=header, name='PROFILE'))
        
        hdul.flush()
        
    
        