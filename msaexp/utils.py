"""
"""
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
    ref_lam : array-like
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

    ref_lam = _find_nirspec_output_sampling_wavelengths(
        all_wcs,
        targ_ra, targ_dec
    )
    ref_lam = np.array(ref_lam)
    orig_lam = ref_lam*1
    
    # Set linear dispersion
    if wave_range is None:
        lmin = np.nanmin(ref_lam)
        lmax = np.nanmax(ref_lam)
    else:
        lmin, lmax = wave_range
        
    if wave_step is None:
        dlam = np.nanmedian(np.diff(ref_lam))*wave_scale
    else:
        dlam = wave_step*1.
    
    if log_wave:
        if verbose:
            msg = f'Set log(lam) grid (dlam/lam={dlam/lmin*3.e5:.0f} km/s)'
            print(msg)
        
        lam_step = dlam/lmin
        ref_lam = np.exp(np.arange(np.log(lmin), np.log(lmax), lam_step))
    else:
        if verbose:
            msg = f'Set linear wave grid (dlam={dlam*1.e4:.1f} Ang)'
            print(msg)
        
        lam_step = dlam    
        ref_lam = np.arange(lmin, lmax, lam_step)
    
    if keep_wave:
        ref_lam = orig_lam
        
        if keep_wave == 2:
            if verbose:
                msg = 'Oversample original wavelength grid x 2'
                print(msg)
                
            # Oversample by x2
            dl = np.diff(ref_lam)
            ref_lam = np.append(ref_lam, ref_lam[:-1]+dl/2.)
            ref_lam.sort()
            
        lmin = np.nanmin(ref_lam)
        lmax = np.nanmax(ref_lam)
        dlam = np.nanmedian(np.diff(ref_lam))
        
    if wave_array is not None:
        ref_lam = wave_array*1.
        lmin = np.nanmin(ref_lam)
        lmax = np.nanmax(ref_lam)
        dlam = np.nanmedian(np.diff(ref_lam))
        
    n_lam = ref_lam.size
    if not n_lam:
        raise ValueError("Not enough data to construct output WCS.")

    x_slit = np.zeros(n_lam)
    lam = 1e-6 * ref_lam

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
    
    data_size = (ny, len(ref_lam))

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
    
    return ref_lam, header, data_size, output_wcs


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


def combine_2d_with_rejection(result, sigma=5, grow=0, trim=2, prf_center=None, prf_sigma=1.0, fit_prf=True):
    """
    Combine single drizzled arrays with outlier detection
    """
    from photutils.psf import IntegratedGaussianPRF
    from astropy.modeling.models import Polynomial2D
    from astropy.modeling.fitting import LevMarLSQFitter
    import scipy.ndimage as nd
    
    import astropy.units as u
    
    import grizli.utils
    
    sci = np.array([s.data for s in result])
    dq = np.array([s.dq for s in result])
    err = np.array([s.err*(1+np.sqrt(2))/np.sqrt(2) for s in result])
    
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

    sh = wht2d.shape
    yp, xp = np.indices(sh)
    x = np.arange(sh[0])
    
    prof = np.nansum(sci2d * wht2d, axis=1) / np.nansum(wht2d, axis=1)
    ok = np.isfinite(prof) & (prof > 0)
    if prf_center is None:
        x0 = sh[0]//2
    else:
        x0 = prf_center
        
    ok &= np.abs(x - x0) < 8
    
    if prf_center is None:
        prf_center = np.nanargmax(prof*ok)
        
    prf = IntegratedGaussianPRF(x_0=0, y_0=prf_center, sigma=prf_sigma)
    
    if fit_prf:
        fitter = LevMarLSQFitter()
        prf.fixed['x_0'] = True
        prf.fixed['sigma'] = False    
        pfit = fitter(prf, x[ok]*0., x[ok], prof[ok])
    else:
        pfit = prf
    
    pfit.flux = 1.0
    
    p2d = pfit(yp*0., yp)
    
    # pclip = prof > 0
    # prof /= prof[pclip].sum()
    # p2d = (wht2d.T*0. + prof).T
    # p2d[~pclip,:] = 0
    
    wht1d = (wht2d*p2d**2).sum(axis=0)
    sci1d = (sci2d*wht2d*p2d).sum(axis=0) / wht1d
    
    if trim > 0:
        bad = nd.binary_dilation(wht1d <= 0, iterations=trim)
        wht1d[bad] = 0
        
    sci1d[wht1d <= 0] = 0
    err1d = np.sqrt(1/wht1d)
    err1d[wht1d <= 0] = 0
    
    tomjy = 1.e12*result[0].meta.photometry.pixelarea_steradians #*pfit.flux.value
    
    spec = grizli.utils.GTable()
    spec.meta['NCOMBINE'] = len(result)
    spec.meta['MASKSIG'] = sigma, 'Mask sigma'
    
    spec.meta['TOMUJY'] = tomjy, 'Conversion from pixel values to microJansky'
    spec.meta['PROFCEN'] = pfit.y_0.value, 'PRF profile center'
    spec.meta['PROFSIG'] = pfit.sigma.value, 'PRF profile sigma'
    
    met = result[0].meta.instrument.instance
    for k in ['detector','filter','grating']:
        spec.meta[k] = met[k]
        
    sh = result[0].data.shape
    x = np.arange(sh[1])
    
    ri, di, spec['wave'] = result[0].meta.wcs.forward_transform(x, x*0+sh[0]/2)
    
    spec['wave'].unit = u.micron
    spec['flux'] = sci1d*tomjy
    spec['err'] = err1d*tomjy
    spec['flux'].unit = u.microJansky
    spec['err'].unit = u.microJansky
    
    return sci2d*tomjy, wht2d/tomjy**2, p2d, spec


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
    result : `jwst.datamodels.ModelContainer`
        The product of `ResampleSpecData.do_drizzle()`
    
    header : `~astropy.io.fits.Header`
        WCS header
        
    """
    from jwst.datamodels import SlitModel, ModelContainer
    from jwst.resample.resample_spec import ResampleSpecData
    
    if slits in [None, []]:
        return None
    
    # Build WCS
    if 'pscale_ratio' in drizzle_params:
        kwargs['pscale_ratio'] = drizzle_params['pscale_ratio']
        
    _ = build_regular_wavelength_wcs(slits, **kwargs)
    ref_lam, header, data_size, output_wcs = _
    
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
                
            result = step.do_drizzle()

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
    
        result = step.do_drizzle()
    
    # if 'single' in drizzle_params:
    #     if drizzle_params['single']:
    #         comb = combine_2d_with_rejection(result)
            
    return ref_lam, header, result

def extract_all():
    
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
                          
    wave, header, result = msaexp.utils.drizzle_slits_2d(slits, drizzle_params=drizzle_params, **kwargs)
    sci2d, wht2d, p2d, spec = msaexp.utils.combine_2d_with_rejection(result, sigma=10, **prf_kwargs)
    
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
        hdul.append(pyfits.ImageHDU(data=p2d, header=header, name='PROFILE'))
        
        hdul.flush()
        
    
        