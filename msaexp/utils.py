import os
import yaml

import warnings
import numpy as np
import astropy.io.fits as pyfits

import grizli.utils


MSAEXP_BADPIX = None


def rename_source(source_name):
    """
    This function takes a source name as input and returns the adjusted
    name according to the following rules:

        - If the source name starts with "background", it is replaced with "b".
        - If the source name contains "_-", it is replaced with "_m".

    Parameters:
    -----------
    source_name : str
        The original source name.

    Returns:
    --------
    name : str
        The adjusted source name.

    """
    name = source_name.replace("background_", "b")
    name = name.replace("_-", "_m")
    return name


def update_slit_metadata(slit):
    """
    Try to update missing slit metadata.

    This function tries to update missing metadata for a given slit object.
        - If the slit's lamp mode is 'FIXEDSLIT' and the source name is missing,
          it sets the source name based on the target's proposer name and the slit's
          name.
        - If the slit's source type is missing, it sets it to 'EXTENDED'.
        - If the slit's source type is None, it sets it to 'EXTENDED'.
        - If the slit's slitlet ID is missing, it sets it to 9999.

    Parameters:
    -----------
    slit : `jwst.datamodels.SlitModel`
        The slit object to update.

    Returns:
    --------
    None
    """
    meta = slit.meta.instance
    is_fixed = meta["instrument"]["lamp_mode"] == "FIXEDSLIT"

    if is_fixed & (slit.source_name is None):
        # Set source info for fixed slit targets
        targ = meta["target"]
        _name = f"{targ['proposer_name'].lower()}_{slit.name}".lower()
        slit.source_name = _name
        slit.source_ra = targ["ra"]
        slit.source_dec = targ["dec"]

    if not hasattr(slit, "source_type"):
        slit.source_type = "EXTENDED"

    if slit.source_type is None:
        slit.source_type = "EXTENDED"

    if not hasattr(slit, "slitlet_id"):
        slit.slitlet_id = 9999


def update_slit_dq_mask(
    slit, mask_padded=False, bar_threshold=-1, verbose=True
):
    """
    Update slit dq array and masking for padded slits and barshadow

    Parameters
    ----------
    slit : `jwst.datamodels.SlitModel`, str
        Filename or data object of a 2D slitlet

    mask_padded : bool
        Mask pixels of slitlets that had been padded around the nominal MSA
        slitlets

    bar_threshold : float
        Mask pixels in slitlets where `barshadow < bar_threshold`

    Returns
    -------
    _slit : `jwst.datamodels.SlitModel`
        Input data model with modified DQ array, or the object loaded from a
        file if the filename was provided

    """
    from gwcs import wcstools
    from jwst.datamodels import SlitModel

    if isinstance(slit, str):
        _slit = SlitModel(slit)
        update_slit_metadata(_slit)
    else:
        _slit = slit

    if mask_padded:
        msg = "msaexp.utils.update_slit_dq_mask: Mask padded area of"
        msg += " slitlet: {mask_padded*1:.1f}"
        grizli.utils.log_comment(
            grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
        )

        _wcs = _slit.meta.wcs
        d2s = _wcs.get_transform("detector", "slit_frame")

        bbox = _wcs.bounding_box
        grid = wcstools.grid_from_bounding_box(bbox)
        _, sy, slam = np.array(d2s(*grid))
        msk = sy < np.nanmin(sy + mask_padded)
        msk |= sy > np.nanmax(sy - mask_padded)
        _slit.data[msk] = np.nan

    if hasattr(_slit, "barshadow") & (bar_threshold > 0):
        grizli.utils.log_comment(
            grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
        )
        msk = _slit.barshadow < bar_threshold

        msg = "msaexp.utils.update_slit_dq_mask: mask barshadow < "
        msg += "{bar_threshold:.2f}, N={msk.sum()} pix"
        grizli.utils.log_comment(
            grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
        )

        _slit.data[msk] = np.nan

    _slit.dq = (_slit.dq & 1025 > 0) * 1
    _slit.data[_slit.dq > 0] = np.nan

    # Does the scale look like MJr/sr?
    mederr = np.nanmedian(_slit.err)
    if mederr < 1.0e-11:
        msg = f"msaexp.utils.update_slit_dq_mask: med(err) = {mederr:.2e} ; "
        msg += f" scale by {1./_slit.meta.photometry.pixelarea_steradians:.2e}"

        grizli.utils.log_comment(
            grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
        )

        _slit.data /= _slit.meta.photometry.pixelarea_steradians
        _slit.err /= _slit.meta.photometry.pixelarea_steradians
        _slit.var_flat /= _slit.meta.photometry.pixelarea_steradians**2
        _slit.var_rnoise /= _slit.meta.photometry.pixelarea_steradians**2
        _slit.var_poisson /= _slit.meta.photometry.pixelarea_steradians**2

    return _slit


def slit_trace_center(
    slit, with_source_xpos=False, with_source_ypos=True, index_offset=0.0
):
    """
    Get detector coordinates along the center of a slit

    Parameters
    slit : `~stdatamodels.jwst.datamodels.slit.SlitModel`
        Slit object

    with_source_xpos : bool
        Apply wavelength correction based on source centering within the
        shutter assuming ``source_type = POINT``

    with_source_ypos : bool
        Get center of trace accounting for source y offset in the shutter

    index_offset : float
        Index offset applied to the trace center

    Returns
    -------
    x : array-like
        x pixel along trace

    ytr : array-like
        y pixel along trace

    wtr : array-like
        wavelength along trace, microns

    rs, ds : float
        Expected sky coordinates at center of the trace
    """
    from jwst.wavecorr import WavecorrStep
    from gwcs import wcstools

    sh = slit.data.shape

    if with_source_xpos:
        # Get wavelength offset for off-center source,
        # assuming source_type = POINT
        step = WavecorrStep()

        orig_srctype = slit.source_type
        slit.source_type = "POINT"

        wavecorr = step.process(slit)

        slit.source_type = orig_srctype

        _wcs = wavecorr.meta.wcs
        swave = wavecorr.wavelength * 1
    else:
        _wcs = slit.meta.wcs
        swave = slit.wavelength * 1

    d2s = _wcs.get_transform("detector", "slit_frame")
    d2w = _wcs.get_transform("detector", "world")

    bbox = _wcs.bounding_box
    grid = wcstools.grid_from_bounding_box(bbox)

    # Slit coordinates
    sx, sy, slam = np.array(d2s(*grid))

    # Direction of increasing sy shutter coordinate needed for interpolation
    dy = sy - np.roll(sy, 1, axis=0)
    ydir = 1 if np.nanmedian(dy) > 0 else -1

    x = np.arange(sh[1], dtype=float)
    yarr = np.arange(sy.shape[0])

    ok = np.isfinite(sy)

    # Interpolate 2D shutter and wavelength arrays at the desired trace center
    trace_center = 0.0
    if with_source_ypos & (slit.source_ypos is not None):
        trace_center += slit.source_ypos

    ytr = np.array(
        [
            np.interp(
                trace_center, sy[ok[:, i], i][::ydir], yarr[ok[:, i]][::ydir]
            )
            for i in range(sy.shape[1])
        ]
    )

    wtr = np.array(
        [
            np.interp(
                trace_center,
                sy[ok[:, i], i][::ydir],
                swave[ok[:, i], i][::ydir],
            )
            for i in range(sy.shape[1])
        ]
    )

    ytr += index_offset

    rs, ds, _ = d2w(x[sh[1] // 2], ytr[sh[1] // 2])

    return x, ytr, wtr, rs, ds


GRATING_LIMITS = {
    "prism": [0.58, 5.33, 0.01],
    "g140m": [0.68, 1.9, 0.00063],
    "g235m": [1.66, 3.17, 0.00106],
    "g395m": [2.83, 5.24, 0.00179],
    "g140h": [0.68, 1.9, 0.000238],
    "g235h": [1.66, 3.17, 0.000396],
    "g395h": [2.83, 5.24, 0.000666],
}


def get_standard_wavelength_grid(
    grating,
    sample=1,
    free_prism=True,
    log_step=False,
    grating_limits=GRATING_LIMITS,
):
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

    if (grating.lower() == "prism") & free_prism:
        _path = os.path.join(os.path.dirname(__file__), "data")
        _disp_file = f"{_path}/jwst_nirspec_{grating.lower()}_disp.fits"
        disp = grizli.utils.read_catalog(_disp_file)

        target_waves = [gr[0]]
        while target_waves[-1] < gr[1]:
            dw = np.interp(target_waves[-1], disp["WAVELENGTH"], disp["DLDS"])
            target_waves.append(target_waves[-1] + dw / sample)

        target_waves = np.array(target_waves)
    else:
        if log_step:
            # Step is dlam/l0/sample where l0 is the center of the grid
            dlog = gr[2] / (gr[0] + gr[1]) * 2.0 / sample
            target_waves = np.exp(np.arange(*np.log(gr[:2]), dlog))
        else:
            target_waves = np.arange(*gr[:2], gr[2] / sample)

    return target_waves


def get_slit_sign(slit):
    """
    sign convention for slit pixels

    Parameters
    ----------
    slit : `jwst.datamodels.SlitModel`
        Slit object

    Returns
    -------
    sign : int
        sign convention
    """

    from jwst.resample.resample_spec import (
        ResampleSpecData,
        _find_nirspec_output_sampling_wavelengths,
    )

    _max_virtual_slit_extent = ResampleSpecData._max_virtual_slit_extent

    refmodel = slit

    all_wcs = [refmodel.meta.wcs]

    refwcs = refmodel.meta.wcs

    # setup the transforms that are needed
    s2d = refwcs.get_transform("slit_frame", "detector")
    d2s = refwcs.get_transform("detector", "slit_frame")
    s2w = refwcs.get_transform("slit_frame", "world")

    # estimate position of the target without relying on the meta.target:
    # compute the mean spatial and wavelength coords weighted
    # by the spectral intensity
    bbox = refwcs.bounding_box
    wmean_s = 0.5 * (refmodel.slit_ymax - refmodel.slit_ymin)
    wmean_l = d2s(*np.mean(bbox, axis=1))[2]

    # transform the weighted means into target RA/Dec
    targ_ra, targ_dec, _ = s2w(0, wmean_s, wmean_l)

    target_waves = _find_nirspec_output_sampling_wavelengths(
        all_wcs, targ_ra, targ_dec
    )
    target_waves = np.array(target_waves)

    n_lam = target_waves.size
    if not n_lam:
        raise ValueError("Not enough data to construct output WCS.")

    lam = 1e-6 * target_waves

    # Find the spatial pixel scale:
    y_slit_min, y_slit_max = _max_virtual_slit_extent(
        None, all_wcs, targ_ra, targ_dec
    )

    nsampl = 50
    xy_min = s2d(
        nsampl * [0],
        nsampl * [y_slit_min],
        lam[(tuple((i * n_lam) // nsampl for i in range(nsampl)),)],
    )
    xy_max = s2d(
        nsampl * [0],
        nsampl * [y_slit_max],
        lam[(tuple((i * n_lam) // nsampl for i in range(nsampl)),)],
    )

    if xy_min[1][1] < xy_max[1][1]:
        sign = 1
    else:
        sign = -1

    return sign


def build_regular_wavelength_wcs(
    slits,
    pscale_ratio=1,
    keep_wave=False,
    wave_scale=1,
    log_wave=False,
    refmodel=None,
    verbose=True,
    wave_range=None,
    wave_step=None,
    wave_array=None,
    ypad=2,
    force_nypix=None,
    force_yoffset=None,
    get_weighted_center=False,
    center_on_source=True,
    fix_slope=None,
    **kwargs,
):
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

    get_weighted_center : bool
        Try to find the source centroid in the slit

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
        Mapping,
        Tabular1D,
        Linear1D,
        Identity,
    )

    from gwcs import wcstools, WCS
    from gwcs import coordinate_frames as cf

    from jwst.resample.resample_spec import (
        ResampleSpecData,
        _find_nirspec_output_sampling_wavelengths,
        resample_utils,
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
    s2d = refwcs.get_transform("slit_frame", "detector")
    d2s = refwcs.get_transform("detector", "slit_frame")
    s2w = refwcs.get_transform("slit_frame", "world")

    # estimate position of the target without relying on the meta.target:
    # compute the mean spatial and wavelength coords weighted
    # by the spectral intensity
    bbox = refwcs.bounding_box
    grid = wcstools.grid_from_bounding_box(bbox)
    _, s, lam = np.array(d2s(*grid))

    if get_weighted_center:
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
    else:
        wmean_s = 0.5 * (refmodel.slit_ymax - refmodel.slit_ymin)
        wmean_l = d2s(*np.mean(bbox, axis=1))[2]

    # transform the weighted means into target RA/Dec
    targ_ra, targ_dec, _ = s2w(0, wmean_s, wmean_l)

    target_waves = _find_nirspec_output_sampling_wavelengths(
        all_wcs, targ_ra, targ_dec
    )
    target_waves = np.array(target_waves)
    orig_lam = target_waves * 1

    # Set linear dispersion
    if wave_range is None:
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            lmin = np.nanmin(target_waves)
            lmax = np.nanmax(target_waves)
    else:
        lmin, lmax = wave_range

    if wave_step is None:
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            dlam = np.nanmedian(np.diff(target_waves)) * wave_scale
    else:
        dlam = wave_step * 1.0

    if log_wave:
        msg = f"Set log(lam) grid (dlam/lam={dlam/lmin*3.e5:.0f} km/s)"

        lam_step = dlam / lmin
        target_waves = np.exp(np.arange(np.log(lmin), np.log(lmax), lam_step))
    else:
        msg = f"Set linear wave grid (dlam={dlam*1.e4:.1f} Ang)"

        lam_step = dlam
        target_waves = np.arange(lmin, lmax, lam_step)

    if keep_wave:
        target_waves = orig_lam

        if keep_wave == 2:
            msg = "Oversample original wavelength grid x 2"

            # Oversample by x2
            dl = np.diff(target_waves)
            target_waves = np.append(
                target_waves, target_waves[:-1] + dl / 2.0
            )
            target_waves.sort()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)

            lmin = np.nanmin(target_waves)
            lmax = np.nanmax(target_waves)
            dlam = np.nanmedian(np.diff(target_waves))

    if wave_array is not None:
        msg = f"Set user-defined wavelength grid (size={wave_array.size})"

        target_waves = wave_array * 1.0
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)

            lmin = np.nanmin(target_waves)
            lmax = np.nanmax(target_waves)
            dlam = np.nanmedian(np.diff(target_waves))

    if verbose:
        print("build_regular_wavelength_wcs: " + msg)

    n_lam = target_waves.size
    if not n_lam:
        raise ValueError("Not enough data to construct output WCS.")

    x_slit = np.zeros(n_lam)
    lam = 1e-6 * target_waves

    # Find the spatial pixel scale:
    y_slit_min, y_slit_max = _max_virtual_slit_extent(
        None, all_wcs, targ_ra, targ_dec
    )

    nsampl = 50
    xy_min = s2d(
        nsampl * [0],
        nsampl * [y_slit_min],
        lam[(tuple((i * n_lam) // nsampl for i in range(nsampl)),)],
    )
    xy_max = s2d(
        nsampl * [0],
        nsampl * [y_slit_max],
        lam[(tuple((i * n_lam) // nsampl for i in range(nsampl)),)],
    )

    good = np.logical_and(np.isfinite(xy_min), np.isfinite(xy_max))
    if not np.any(good):
        raise ValueError("Error estimating output WCS pixel scale.")

    xy1 = s2d(x_slit, np.full(n_lam, refmodel.slit_ymin), lam)
    xy2 = s2d(x_slit, np.full(n_lam, refmodel.slit_ymax), lam)
    xylen = (
        np.nanmax(np.linalg.norm(np.array(xy1) - np.array(xy2), axis=0)) + 1
    )
    pscale = (refmodel.slit_ymax - refmodel.slit_ymin) / xylen

    # compute image span along Y-axis
    # (length of the slit in the detector plane)
    det_slit_span = np.nanmax(
        np.linalg.norm(np.subtract(xy_max, xy_min), axis=0)
    )
    ny = int(np.ceil(det_slit_span * pscale_ratio + 0.5)) + 1

    if ypad > 0:
        if verbose:
            print(f"Pad {ypad} pixels on 2D cutout")

        ny += 2 * ypad

    if center_on_source:
        slit_center = 0.5 * (y_slit_max - y_slit_min) + y_slit_min
        slit_center_offset = -(refmodel.source_ypos - slit_center)
        slit_pad = np.abs(slit_center_offset / (pscale / pscale_ratio))
        slit_pad = int(np.round(slit_pad))
        ny += 2 * slit_pad

    border = 0.5 * (ny - det_slit_span * pscale_ratio) - 0.5

    if xy_min[1][1] < xy_max[1][1]:
        intercept = y_slit_min - border * pscale * pscale_ratio
        sign = 1
        # slope = pscale / pscale_ratio
    else:
        intercept = y_slit_max + border * pscale * pscale_ratio
        sign = -1
        # slope = -pscale / pscale_ratio

    slope = sign * pscale / pscale_ratio
    if fix_slope is not None:
        slope = sign * fix_slope

    if center_on_source:
        intercept += sign * slit_center_offset

    if force_nypix is not None:
        ny = force_nypix

    if force_yoffset is not None:
        intercept += slope * force_yoffset

    y_slit_model = Linear1D(slope=slope, intercept=intercept)

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

    wavelength_transform = Tabular1D(
        points=pixel_coord,
        lookup_table=lam,
        bounds_error=False,
        fill_value=np.nan,
    )
    wavelength_transform.inverse = Tabular1D(
        points=lam,
        lookup_table=pixel_coord,
        bounds_error=False,
        fill_value=np.nan,
    )

    data_size = (ny, len(target_waves))

    # Construct the final transform
    mapping = Mapping((0, 1, 0))
    mapping.inverse = Mapping((2, 1))
    out_det2slit = mapping | Identity(1) & y_slit_model & wavelength_transform

    # Create coordinate frames
    det = cf.Frame2D(name="detector", axes_order=(0, 1))
    slit_spatial = cf.Frame2D(
        name="slit_spatial",
        axes_order=(0, 1),
        unit=("", ""),
        axes_names=("x_slit", "y_slit"),
    )
    spec = cf.SpectralFrame(
        name="spectral",
        axes_order=(2,),
        unit=(u.micron,),
        axes_names=("wavelength",),
    )
    slit_frame = cf.CompositeFrame([slit_spatial, spec], name="slit_frame")
    sky = cf.CelestialFrame(
        name="sky", axes_order=(0, 1), reference_frame=coord.ICRS()
    )
    world = cf.CompositeFrame([sky, spec], name="world")

    pipeline = [(det, out_det2slit), (slit_frame, s2w), (world, None)]
    output_wcs = WCS(pipeline)

    # Compute bounding box and output array shape.  Add one to the y (slit)
    # height to account for the half pixel at top and bottom due to pixel
    # coordinates being centers of pixels
    bounding_box = resample_utils.wcs_bbox_from_shape(data_size)
    output_wcs.bounding_box = bounding_box
    output_wcs.array_shape = data_size

    if 1:
        header = fixed_rectified_slit_header(slits[0].meta.wcs, output_wcs)

    return target_waves, header, data_size, output_wcs


def build_slit_centered_wcs(
    slit,
    waves,
    pscale_ratio=1,
    ypad=0,
    force_nypix=21,
    slit_center=0.0,
    center_on_source=False,
    get_from_ypos=True,
    phase=-0.5,
    fix_slope=None,
    **kwargs,
):
    """
    Build a 2D WCS centered on the target in a slit

    Parameters
    ----------
    slit : `jwst.datamodels.SlitModel`

    Returns
    -------
    _data : tuple
        See `~msaexp.utils.build_regular_wavelength_wcs`

    """
    from gwcs import wcstools

    if get_from_ypos:
        yoff = slit.source_ypos / 0.1 - 0.5
    else:
        sc = slit.copy()

        refwcs = slit.meta.wcs
        d2s = refwcs.get_transform("detector", "slit_frame")

        bbox = refwcs.bounding_box
        grid = wcstools.grid_from_bounding_box(bbox)
        _, s, lam = np.array(d2s(*grid))

        if center_on_source:
            s0 = slit.source_ypos
        else:
            s0 = slit_center

        sc.data = sc.data * 0.0 + np.exp(-((s - s0) ** 2) / 2 / 0.3**2)

        DRIZZLE_PARAMS = dict(
            output=None,
            single=True,
            blendheaders=True,
            pixfrac=1.0,
            kernel="square",
            fillval=0,
            wht_type="ivm",
            good_bits=0,
            pscale_ratio=1.0,
            pscale=None,
        )

        _data = build_regular_wavelength_wcs(
            [sc],
            center_on_source=False,
            wave_array=waves,
            force_nypix=force_nypix,
            ypad=ypad,
            pscale_ratio=pscale_ratio,
            verbose=False,
            fix_slope=fix_slope,
        )

        _waves, _header, _drz = drizzle_slits_2d(
            [sc],
            build_data=_data,
            drizzle_params=DRIZZLE_PARAMS,
        )

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            prof = np.nansum(_drz[0].data, axis=1)

        if prof.max() == 0:
            return _data

        xc = (np.arange(len(prof)) * prof).sum() / prof.sum()
        yoff = xc - (force_nypix - 1) / 2 + phase

    slit.drizzle_slit_offset = yoff

    _xdata = build_regular_wavelength_wcs(
        [slit],
        center_on_source=False,
        wave_array=waves,
        force_nypix=force_nypix,
        force_yoffset=yoff,
        ypad=0,
        verbose=False,
        fix_slope=fix_slope,
    )

    return _xdata


def longslit_header_from_wcs(wcs):
    """
    Generate a FITS-compliant long-slit header from a NIRSpec spectral WCS

    Parameters
    ----------
    wcs : `gwcs.wcs.WCS`
        Spectral WCS, with minimal transforms ``detector`` and ``world``

    Returns
    -------
    header : `~astropy.io.fits.Header`
        WCS header following the FITS WCS
        `Paper II <https://fits.gsfc.nasa.gov/fits_wcs.html>`_
        standard.

    """
    import gwcs

    if not isinstance(wcs, gwcs.wcs.WCS):
        raise ValueError("`wcs` is not a gwcs.wcs.WCS object")

    for frame in ["detector", "world", "slit_frame"]:
        if frame not in wcs.available_frames:
            raise ValueError(f"Frame '{frame}' not in wcs.available_frames")

    bounds = wcs.pixel_bounds
    shape = (int(bounds[1][1] + 0.5), int(bounds[0][1] + 0.5))

    # Transformations
    d2w = wcs.get_transform("detector", "world")
    s2d = wcs.get_transform("slit_frame", "detector")
    d2s = wcs.get_transform("detector", "slit_frame")
    s2w = wcs.get_transform("slit_frame", "world")

    # Wavelength bounds at x edges
    # w0 = d2w(0, shape[0]//2)
    # w1 = d2w(shape[1]-1, shape[0]//2)
    ypi, xpi = np.indices(shape)
    w0 = d2w(xpi, ypi)

    # Pixel offsets
    crpix00 = (shape[1] // 2, shape[0] // 2)
    crpix01 = (shape[1] // 2, shape[0] // 2 + 1)
    crpix10 = (shape[1] // 2 + 1, shape[0] // 2)

    # Sky from detector
    rd00 = d2w(*crpix00)
    rd10 = d2w(*crpix10)

    cosd = np.cos(rd00[1] / 180 * np.pi)

    # Slit from detector
    s00 = d2s(*crpix00)

    # dy_slit_frame / dy_detector
    dy0 = 0.1
    d00 = s2d(s00[0], s00[1] + dy0, s00[2])
    dslit_dy = dy0 / np.sqrt(np.sum((np.array(d00) - np.array(crpix00)) ** 2))

    if d00[1] < crpix00[1]:
        dslit_dy *= -1

    # sky pixel scale and slit position angle towards +slit_frame_y
    sd00 = s2w(s00[0], s00[1], s00[2])
    sd01 = s2w(s00[0], s00[1] + dslit_dy, s00[2])
    cd1 = (sd01[0] - sd00[0]) * cosd
    cd2 = sd01[1] - sd00[1]

    pscale = np.sqrt(cd1**2 + cd2**2)
    posang = np.arctan2(cd1, cd2) / np.pi * 180

    # Build the header
    h = pyfits.Header()
    h["NAXIS"] = 3

    h["NAXIS1"] = shape[1]
    h["NAXIS2"] = shape[0]
    h["NAXIS3"] = 1

    h["CRPIX1"] = shape[1] // 2 + 1
    h["CRPIX2"] = shape[0] // 2 + 1
    h["CRPIX3"] = 1

    h["CRVAL1"] = rd00[2] * 1.0e4
    h["CRVAL2"] = rd00[0]
    h["CRVAL3"] = rd00[1]

    h["CD1_1"] = (
        rd10[2] - rd00[2]
    ) * 1.0e4, "dlam/dpix at the reference pixel"
    h["CD2_2"] = np.cos((90 + posang) / 180 * np.pi) * pscale
    h["CD3_2"] = -np.sin((90 + posang) / 180 * np.pi) * pscale
    h["CD2_3"] = 1.0
    h["CD3_3"] = 1.0

    h["CTYPE1"] = "WAVELEN"
    h["CTYPE2"] = "RA---TAN"
    h["CTYPE3"] = "DEC--TAN"

    h["CUNIT1"] = "Angstrom"
    h["CUNIT2"] = "deg"
    h["CUNIT3"] = "deg"

    h["RADESYS"] = "ICRS"
    # h['EPOCH'] = 2000.
    h["WCSNAME"] = "SLITWCS"

    h["SLIT_PA"] = posang, "Position angle of the slit, degrees"
    h["PSCALE"] = pscale * 3600, "Pixel scale, arcsec/pix"

    h["SLIT_Y0"] = s00[1], "y_slit_frame at reference pixel"
    h["SLIT_DY"] = dslit_dy, "d(slit_frame)/dy  at reference pixel"

    wmin = np.nanmedian(w0[2][:, 0])
    wmax = np.nanmedian(w0[2][:, -1])
    if ~np.isfinite(wmin):
        wmin = -1
    if ~np.isfinite(wmax):
        wmax = -1

    h["LMIN"] = wmin, "Minimum wavelength, micron"
    h["LMIN"] = wmax, "Minimum wavelength, micron"

    # h['LMIN'] = w0[2], 'Minimum wavelength, micron'
    # h['LMAX'] = w1[2], 'Maximum wavelength, micron'
    h["DLAM"] = h["CD1_1"] / 1.0e4, "Wavelength step at reference pixel"

    # h['LONPOLE'] = 90 + posang

    return h


def fixed_rectified_slit_header(slit_wcs, rectified_wcs):
    """
    Merge 2D WCS header from a slit WCS to a rectified WCS, e.g., from
    `build_slit_centered_wcs`.

    Something gets lost in the transformation of the latter, so assume the
    ``slit_wcs`` is correct, derive the cross-dispersion WCS based on the
    `slit_frame` coordinate frame there and then propagate to the
    ``rectified_wcs``.

    Parameters
    ----------
    slit_wcs : `gwcs.wcs.WCS`
        WCS object of a slitlet

    rectified_wcs : `gwcs.wcs.WCS`
        Rectified WCS

    Returns
    -------
    header : `~astropy.io.fits.Header`
        Fits header with spectral WCS.

    """
    header = longslit_header_from_wcs(rectified_wcs)

    hslit = longslit_header_from_wcs(slit_wcs)

    yscl = header["SLIT_DY"] / hslit["SLIT_DY"]
    dy = header["SLIT_Y0"] - hslit["SLIT_Y0"]

    cosd = np.cos(hslit["CRVAL3"] / 180 * np.pi)

    header["CRVAL2"] = (
        hslit["CRVAL2"] + hslit["CD2_2"] * dy / hslit["SLIT_DY"] / cosd
    )
    header["CRVAL3"] = hslit["CRVAL3"] + hslit["CD3_2"] * dy / hslit["SLIT_DY"]
    header["CD2_2"] = hslit["CD2_2"] * yscl
    header["CD3_2"] = hslit["CD3_2"] * yscl
    header["SLIT_PA"] = hslit["SLIT_PA"]

    return header


DRIZZLE_PARAMS = dict(
    output=None,
    single=False,
    blendheaders=True,
    pixfrac=1.0,
    kernel="square",
    fillval=0,
    wht_type="ivm",
    good_bits=0,
    pscale_ratio=1.0,
    pscale=None,
)


def drizzle_slits_2d(
    slits,
    build_data=None,
    drizzle_params=DRIZZLE_PARAMS,
    centered_wcs=False,
    **kwargs,
):
    """
    Run `jwst.resample.resample_spec.ResampleSpecData` on a list of
    List of `jwst.datamodels.slit.SlitModel` objects.

    Parameters
    ----------
    slits : list
        List of `jwst.datamodels.SlitModel` objects

    build_data : tuple
        Data like output from `msaexp.utils.build_regular_wavelength_wcs`,
        i.e., ``target_waves, header, data_size, output_wcs``

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
    if "pscale_ratio" in drizzle_params:
        kwargs["pscale_ratio"] = drizzle_params["pscale_ratio"]

    if build_data is None:
        _data = build_regular_wavelength_wcs(slits, **kwargs)
        target_waves, header, data_size, output_wcs = _data
    else:
        target_waves, header, data_size, output_wcs = build_data

    if centered_wcs:
        _ = build_slit_centered_wcs(slits[0], target_waves, **kwargs)
        target_waves, header, data_size, output_wcs = _
        print(f"Center on target: output shape = {data_size}")

    # Own drizzle single to get variances
    if "single" in drizzle_params:
        run_single = drizzle_params["single"]
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

                step.blank_output = SlitModel(
                    tuple(step.output_wcs.array_shape)
                )
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


def combine_2d_with_rejection(
    drizzled_slits,
    outlier_threshold=5,
    grow=0,
    trim=2,
    prf_center=None,
    prf_sigma=1.0,
    center_limit=8,
    fit_prf=True,
    fix_center=False,
    fix_sigma=False,
    verbose=True,
    profile_slice=None,
    sigma_bounds=(0.5, 2.0),
    background=None,
    **kwargs,
):
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

    outlier_threshold : float
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

    prof_tab : `astropy.table.Table`
        Cross-dispersion profile

    """
    from photutils.psf import IntegratedGaussianPRF
    from astropy.modeling.fitting import LevMarLSQFitter
    import scipy.ndimage as nd

    import astropy.units as u

    import grizli.utils
    from .version import __version__

    sci = np.array([s.data for s in drizzled_slits])
    dq = np.array([s.dq for s in drizzled_slits])

    if 0:
        err = np.array([s.err * 2.0 for s in drizzled_slits])
        ivar = 1 / err**2

    # jwst resample uses 1/var_rnoise as the weight
    ivar = np.array([1 / s.var_rnoise for s in drizzled_slits])
    ivar[~np.isfinite(ivar)] = 0
    ivar *= 0.25

    err = 1 / np.sqrt(ivar)
    err[ivar == 0] = 0

    dq[(sci == 0) | (~np.isfinite(sci))] |= 1
    sci[dq > 0] = np.nan
    med = np.nanmedian(sci, axis=0)

    ivar[(dq > 0) | (err <= 0)] = 0

    if np.nanmax(ivar) == 0:
        mad = 1.48 * np.nanmedian(np.abs(sci - med), axis=0)
        for i in range(ivar.shape[0]):
            ivar[i, :, :] = 1 / mad**2.0

        ivar[(dq > 0) | (~np.isfinite(mad))] = 0.0

    bad = np.abs(sci - med) * np.sqrt(ivar) > outlier_threshold
    bad |= dq > 0

    sci[bad] = 0
    ivar[bad] = 0.0

    wht2d = (ivar * (~bad)).sum(axis=0)
    sci2d = (sci * ivar * (~bad)).sum(axis=0) / wht2d
    sci2d[wht2d == 0] = 0
    wht2d[wht2d <= 0] = 0.0

    if background is not None:
        print("use background")
        sci2d -= background
        bmask = (background == 0) | (~np.isfinite(background))
        sci2d[bmask] = 0
        wht2d[bmask] = 0

    sh = wht2d.shape
    yp, xp = np.indices(sh)

    if profile_slice is not None:
        if not isinstance(profile_slice, slice):
            if isinstance(profile_slice[0], int):
                # pixels
                profile_slice = slice(*profile_slice)
            else:
                # Wavelengths interpolated on pixel grid
                sh = drizzled_slits[0].data.shape
                xpix = np.arange(sh[1])
                ypix = np.zeros(sh[1]) + sh[0] / 2

                _wcs = drizzled_slits[0].meta.wcs
                _, _, wave0 = _wcs.forward_transform(xpix, ypix)
                xsl = np.cast[int](
                    np.round(np.interp(profile_slice, wave0, xpix))
                )
                xsl = np.clip(xsl, 0, sh[1])
                print(f"Wavelength slice: {profile_slice} > {xsl} pix")
                profile_slice = slice(*xsl)

        prof1d = np.nansum((sci2d * wht2d)[:, profile_slice], axis=1)
        prof1d /= np.nansum(wht2d[:, profile_slice], axis=1)

        slice_limits = profile_slice.start, profile_slice.stop
    else:
        prof1d = np.nansum(sci2d * wht2d, axis=1) / np.nansum(wht2d, axis=1)
        slice_limits = 0, sh[1]

    ok = np.isfinite(prof1d)
    # Trim edges
    ok[np.where(ok)[0][:1]] = False
    ok[np.where(ok)[0][-1:]] = False

    ok &= prof1d > 0

    xpix = np.arange(sh[0])
    ytrace = sh[0] / 2.0
    x0 = np.arange(sh[0]) - ytrace
    y0 = yp - ytrace

    if prf_center is None:
        msk = ok & (np.abs(x0) < center_limit)
        prf_center = np.nanargmax(prof1d * msk) - sh[0] / 2.0
        if verbose:
            print(f"Set prf_center: {prf_center} {sh} {ok.sum()}")

    ok &= np.abs(x0 - prf_center) < center_limit

    prf = IntegratedGaussianPRF(x_0=0, y_0=prf_center, sigma=prf_sigma)

    if fit_prf & (ok.sum() > (2 - fix_center - fix_sigma)):
        fitter = LevMarLSQFitter()
        prf.fixed["x_0"] = True
        prf.fixed["y_0"] = fix_center
        prf.fixed["sigma"] = fix_sigma
        prf.bounds["sigma"] = sigma_bounds
        prf.bounds["y_0"] = (
            prf_center - center_limit,
            prf_center + center_limit,
        )

        pfit = fitter(prf, x0[ok] * 0.0, x0[ok], prof1d[ok])

        if verbose:
            msg = f"fit_prf: center = {pfit.y_0.value:.2f}"
            msg += f". sigma = {pfit.sigma.value:.2f}"
            print(msg)

    else:
        pfit = prf

    # Renormalize for 1D
    pfit.flux = prf.sigma.value * np.sqrt(2 * np.pi)

    profile2d = pfit(y0 * 0.0, y0)
    wht1d = (wht2d * profile2d**2).sum(axis=0)
    sci1d = (sci2d * wht2d * profile2d).sum(axis=0) / wht1d

    if profile_slice is not None:
        pfit1d = np.nansum(
            (wht2d * profile2d * sci1d)[:, profile_slice], axis=1
        )
        pfit1d /= np.nansum(wht2d[:, profile_slice], axis=1)
    else:
        pfit1d = np.nansum(profile2d * sci1d * wht2d, axis=1) / np.nansum(
            wht2d, axis=1
        )

    if trim > 0:
        bad = nd.binary_dilation(wht1d <= 0, iterations=trim)
        wht1d[bad] = 0

    sci1d[wht1d <= 0] = 0
    err1d = np.sqrt(1 / wht1d)
    err1d[wht1d <= 0] = 0

    # Flux conversion
    to_ujy = 1.0e12 * drizzled_slits[0].meta.photometry.pixelarea_steradians

    spec = grizli.utils.GTable()
    spec.meta["VERSION"] = __version__, "msaexp software version"

    spec.meta["NCOMBINE"] = len(drizzled_slits), "Number of combined exposures"

    exptime = 0
    for s in drizzled_slits:
        exptime += s.meta.exposure.effective_exposure_time

    spec.meta["EXPTIME"] = exptime, "Total effective exposure time"

    spec.meta["OTHRESH"] = outlier_threshold, "Outlier mask threshold, sigma"

    spec.meta["TOMUJY"] = to_ujy, "Conversion from pixel values to microJansky"
    spec.meta["PROFCEN"] = pfit.y_0.value, "PRF profile center"
    spec.meta["PROFSIG"] = pfit.sigma.value, "PRF profile sigma"
    spec.meta["PROFSTRT"] = slice_limits[0], "Start of profile slice"
    spec.meta["PROFSTOP"] = slice_limits[1], "End of profile slice"
    spec.meta["YTRACE"] = ytrace, "Expected center of trace"

    prof_tab = grizli.utils.GTable()
    prof_tab.meta["VERSION"] = __version__, "msaexp software version"
    prof_tab["pix"] = x0
    prof_tab["profile"] = prof1d
    prof_tab["pfit"] = pfit1d
    prof_tab.meta["PROFCEN"] = pfit.y_0.value, "PRF profile center"
    prof_tab.meta["PROFSIG"] = pfit.sigma.value, "PRF profile sigma"
    prof_tab.meta["PROFSTRT"] = slice_limits[0], "Start of profile slice"
    prof_tab.meta["PROFSTOP"] = slice_limits[1], "End of profile slice"
    prof_tab.meta["YTRACE"] = ytrace, "Expected center of trace"

    met = drizzled_slits[0].meta.instrument.instance
    for k in ["detector", "filter", "grating"]:
        spec.meta[k] = met[k]

    sh = drizzled_slits[0].data.shape
    xpix = np.arange(sh[1])
    ypix = np.zeros(sh[1]) + sh[0] / 2 + pfit.y_0.value

    _wcs = drizzled_slits[0].meta.wcs
    ri, di, spec["wave"] = _wcs.forward_transform(xpix, ypix)

    spec["wave"].unit = u.micron
    spec["flux"] = sci1d * to_ujy
    spec["err"] = err1d * to_ujy
    spec["flux"].unit = u.microJansky
    spec["err"].unit = u.microJansky

    return sci2d * to_ujy, wht2d / to_ujy**2, profile2d, spec, prof_tab


def drizzle_2d_pipeline(
    slits,
    output_root=None,
    standard_waves=True,
    drizzle_params=DRIZZLE_PARAMS,
    include_separate=True,
    **kwargs,
):
    """
    Drizzle list of background-subtracted slitlets

    Parameters
    ----------
    slits : list
        List of `jwst.datamodels.SlitModel` objects

    output_root : str
        Optional rootname of output files

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

    if (standard_waves > 0) & ("wave_array" not in kwargs):
        # Get fixed wavelength grid
        waves = get_standard_wavelength_grid(
            slits[0].meta.instrument.grating, sample=standard_waves
        )
        _data0 = drizzle_slits_2d(
            slits, drizzle_params=drizzle_params, wave_array=waves, **kwargs
        )
    else:
        _data0 = drizzle_slits_2d(
            slits, drizzle_params=drizzle_params, **kwargs
        )

    target_wave, header, drizzled_slits = _data0

    _data1 = combine_2d_with_rejection(drizzled_slits, **kwargs)
    sci2d, wht2d, profile2d, spec, prof = _data1

    hdul = pyfits.HDUList()
    hdul.append(pyfits.BinTableHDU(data=spec, name="SPEC1D"))

    # Add 2D arrays
    header["BUNIT"] = "ujy"

    for k in spec.meta:
        header[k] = spec.meta[k]

    hdul.append(pyfits.ImageHDU(data=sci2d, header=header, name="SCI"))
    hdul.append(pyfits.ImageHDU(data=wht2d, header=header, name="WHT"))
    hdul.append(pyfits.ImageHDU(data=profile2d, header=header, name="PROFILE"))

    # if include_separate:
    #     for s in drizzled_slits.
    hdul.append(pyfits.BinTableHDU(data=prof, name="PROF1D"))

    if output_root is not None:
        hdul.writeto(f"{output_root}.spec.fits", overwrite=True)

    return hdul


def drizzled_hdu_figure(
    hdul,
    tick_steps=None,
    xlim=None,
    subplot_args=dict(
        figsize=(10, 4), height_ratios=[1, 3], width_ratios=[10, 1]
    ),
    cmap="bone_r",
    interpolation="nearest",
    ymax=None,
    ymax_percentile=90,
    ymax_scale=1.5,
    ymax_sigma_scale=7,
    vmin=-0.2,
    z=None,
    ny=None,
    output_root=None,
    unit="fnu",
    flam_scale=-20,
    recenter=True,
    use_aper_columns=False,
    smooth_sigma=None,
):
    """
    Figure showing drizzled hdu

    Parameters:
    ----------
    hdul : `~astropy.io.fits.HDUList`
        The HDUList object containing the data.
    tick_steps : tuple, optional
        The major and minor tick steps for the x-axis.
    xlim : tuple, optional
        The x-axis limits.
    subplot_args : dict, optional
        Additional arguments for creating the subplots. Default is
        ``dict(figsize=(10, 4), height_ratios=[1,3], width_ratios=[10,1])``.
    cmap : str, optional
        The colormap for the image. Default is 'plasma_r'.
    ymax : float, optional
        The maximum y-axis value. Default is None.
    ymax_sigma_scale : float, optional
        The scale factor for setting the maximum y-axis value based on the
        median error. Default is 7.
    vmin : float, optional
        The minimum value for the 2D cutout. Default is -0.2.
    z : float, optional
        If the redshift is indicated, draw axes with rest-frame wavelengths
        and indicate some common emission lines
    ny : float, optional
        Number of pixels to show on y-axis
    output_root : str, optional
        Rootname of the output file. Default is None.
    unit : str, optional
        Controls the output flux units: 'fnu' (default) for `microJansky` or
        `flam` for f-lambda cgs.
    flam_scale : float, optional
        The scale factor for the flux unit in the y-axis label. Default is -20.
    recenter : bool, optional
        Whether to recenter the y-axis on the expected source location.
        Default is True.
    use_aper_columns : bool, optional
        Whether to use boxcar aperture extraction columns for the 1D spectra.
        Default is False.
    smooth_sigma : float, optional
        The sigma value for smoothing the 2D spectra. Default is None.

    Returns:
    -------
    fig : `~matplotlib.pyplot.Figure`
        The Figure object containing the plot.

    """

    import matplotlib.pyplot as plt
    import astropy.units as u
    import scipy.ndimage as nd
    import grizli.utils

    sp = grizli.utils.read_catalog(hdul["SPEC1D"])
    nx = len(sp)

    fig, a2d = plt.subplots(2, 2, **subplot_args)
    axes = [a2d[0][0], a2d[1][0]]

    if (use_aper_columns > 0) & ("aper_flux" in sp.colnames):
        if ("aper_corr" in sp.colnames) & (use_aper_columns > 1):
            ap_corr = sp["aper_corr"] * 1
        else:
            ap_corr = 1

        if "aper_full_err" in sp.colnames:
            err = sp["aper_full_err"] * ap_corr
        else:
            err = sp["aper_err"] * ap_corr

        flux = sp["aper_flux"] * ap_corr
    else:
        if "full_err" in sp.colnames:
            err = sp["full_err"] * 1
        else:
            err = sp["err"] * 1

        flux = sp["flux"] * 1

        ap_corr = 1

    equiv = u.spectral_density(sp["wave"].data * u.micron)
    flam_unit = 10**flam_scale * u.erg / u.second / u.cm**2 / u.Angstrom
    to_flam = (1 * u.microJansky).to(flam_unit, equivalencies=equiv).value

    if unit != "fnu":
        flux *= to_flam  # (sp['wave']/2.)**-2
        err *= to_flam  # (sp['wave']/2.)**-2

    if ymax is None:
        _msk = (err > 0) & np.isfinite(err) & np.isfinite(flux)
        if _msk.sum() == 0:
            ymax = 1.0
        else:
            ymax = np.nanpercentile(flux[_msk], ymax_percentile) * ymax_scale
            ymax = np.maximum(ymax, ymax_sigma_scale * np.nanmedian(err[_msk]))

    yscl = hdul["PROFILE"].data.max() * ap_corr
    if unit == "flam":
        yscl = yscl / to_flam  # (sp['wave']/2.)**2

    if smooth_sigma is not None:
        xp = np.arange(-4 * int(smooth_sigma), 5 * int(smooth_sigma))
        xg = np.exp(-(xp**2) / 2 / smooth_sigma**2)[None, :]
        # xg /= xg.sum()
        num = hdul["SCI"].data * hdul["WHT"].data
        den = hdul["WHT"].data * 1
        ok = np.isfinite(num + den)
        num[~ok] = den[~ok] = 0.0

        smdata = nd.convolve(num, xg) / nd.convolve(den, xg**2)
        # smdata = nd.gaussian_filter(hdul['SCI'].data, smooth_sigma)
    else:
        smdata = hdul["SCI"].data

    axes[0].imshow(
        smdata / yscl,
        vmin=vmin * ymax,
        vmax=ymax,
        aspect="auto",
        cmap=cmap,
        interpolation=interpolation,
    )

    axes[0].set_yticklabels([])
    y0 = hdul["SPEC1D"].header["YTRACE"] + hdul["SPEC1D"].header["PROFCEN"]

    if "BKGOFF" in hdul["SCI"].header:
        yt = hdul["SCI"].header["BKGOFF"]
    else:
        yt = 5

    if ny is not None:
        axes[0].set_ylim(y0 - ny, y0 + ny)
        axes[0].set_yticks([y0])
        if recenter:
            axes[0].set_ylim(y0 - 2 * ny, y0 + 2 * ny)
    else:
        axes[0].set_yticks([y0 - yt, y0, y0 + yt])
        if recenter:
            axes[0].set_ylim(y0 - 2 * yt, y0 + 2 * yt)

    if (use_aper_columns > 0) & ("APER_Y0" in sp.meta):
        y0 = sp.meta["APER_Y0"]
        yt = sp.meta["APER_DY"]
        axes[0].set_yticks([y0 - yt, y0 + yt + 1])

    # Extraction profile
    ap = a2d[0][1]
    ptab = grizli.utils.GTable(hdul["PROF1D"].data)
    pmax = np.nanmax(ptab["pfit"])

    sh = hdul["SCI"].data.shape
    sli = (
        hdul["PROF1D"].header["PROFSTRT"],
        hdul["PROF1D"].header["PROFSTOP"],
    )

    if sli != (0, sh[1]):

        dy = np.diff(axes[0].get_ylim())[0]

        axes[0].scatter(sli[0], y0 + 0.25 * dy, marker=">", fc="w", ec="k")
        axes[0].scatter(sli[1], y0 + 0.25 * dy, marker="<", fc="w", ec="k")

    if pmax > 0:
        xpr = np.arange(len(ptab))
        ap.step(
            ptab["profile"] / pmax,
            xpr,
            color="k",
            where="pre",
            alpha=0.8,
            lw=2,
        )
        ap.step(
            ptab["pfit"] / pmax, xpr, color="r", where="pre", alpha=0.5, lw=1
        )
        ap.fill_betweenx(
            xpr + 0.5, xpr * 0.0, ptab["pfit"] / pmax, color="r", alpha=0.2
        )

        ap.text(
            0.99,
            0.0,
            f"{hdul['SPEC1D'].header['PROFCEN']:5.2f} ±"
            + f" {hdul['SPEC1D'].header['PROFSIG']:4.2f}",
            ha="right",
            va="center",
            bbox={"fc": "w", "ec": "None", "alpha": 1.0},
            transform=ap.transAxes,
            fontsize=7,
        )

    ap.spines.right.set_visible(False)
    ap.spines.top.set_visible(False)

    ap.set_ylim(axes[0].get_ylim())
    ap.set_yticks(axes[0].get_yticks())

    ap.grid()
    ap.set_xlim(-0.5, 1.3)
    ap.set_xticklabels([])
    ap.set_yticklabels([])

    a2d[1][1].set_visible(False)

    axes[1].step(np.arange(len(sp)), flux, where="mid", color="0.5", alpha=0.9)
    axes[1].step(np.arange(len(sp)), err, where="mid", color="r", alpha=0.2)
    xl = axes[1].get_xlim()

    axes[1].fill_between(xl, [-ymax, -ymax], [0, 0], color="0.8", alpha=0.1)
    axes[1].set_xlim(*xl)

    axes[1].set_ylim(-0.1 * ymax, ymax)
    axes[1].set_xlabel(r"$\lambda_\mathrm{obs}$ [$\mu$m]")

    if unit == "fnu":
        axes[1].set_ylabel(r"$f_\nu\ [\mu\mathrm{Jy}]$")
    else:
        _ylabel = r"$f_\lambda\ [10^{xxx}\ \mathrm{erg} \mathrm{s}^{-1} \mathrm{cm}^{-2} \mathrm{\AA}^{-1}]$"
        axes[1].set_ylabel(_ylabel.replace("xxx", f"{flam_scale:.0f}"))

    if tick_steps is None:
        if hdul[1].header["GRATING"] == "PRISM":
            minor = 0.1
            major = 0.5
        else:
            minor = 0.05
            major = 0.25
    else:
        major, minor = tick_steps

    xt = np.arange(0.5, 5.5, major)
    xtm = np.arange(0.5, 5.5, minor)

    if hdul[1].header["GRATING"] == "PRISM":
        xt = np.append([0.7], xt)

    xt = xt[(xt > sp["wave"].min()) & (xt < sp["wave"].max())]
    xv = np.interp(xt, sp["wave"], np.arange(len(sp)))
    if major < 0.1:
        xtl = [f"{v:.2f}" for v in xt]
    elif major >= 1:
        xtl = [f"{v:.0f}" for v in xt]
    else:
        xtl = [
            f"{v:.1f}" if (np.round(v * 10) == v * 10) else f"{v:.2f}"
            for v in xt
        ]

    xtm = xtm[(xtm > sp["wave"].min()) & (xtm < sp["wave"].max())]
    xvm = np.interp(xtm, sp["wave"], np.arange(len(sp)))

    for ax in axes:
        ax.set_xticks(xvm, minor=True)
        ax.set_xticks(xv, minor=False)
        ax.xaxis.set_ticks_position("both")

    axes[1].set_xticklabels(xtl)
    axes[1].grid()

    if z is not None:
        cc = grizli.utils.MPL_COLORS
        for w, c in zip(
            [
                1216.0,
                1909.0,
                2799.0,
                3727,
                4860,
                5007,
                6565,
                9070,
                9530,
                1.094e4,
                1.282e4,
                1.875e4,
                1.083e4,
            ],
            [
                "purple",
                "olive",
                "skyblue",
                cc["purple"],
                cc["g"],
                cc["b"],
                cc["g"],
                "darkred",
                "darkred",
                cc["pink"],
                cc["pink"],
                cc["pink"],
                cc["orange"],
            ],
        ):
            wz = w * (1 + z) / 1.0e4
            # dw = 70*(1+z)/1.e4
            # dw = 0.01*(sp['wave'].max()-sp['wave'].min())

            dw = 0.005 * len(sp["wave"])
            wx = np.interp(wz, sp["wave"], np.arange(len(sp)))

            axes[1].fill_between(
                [wx - dw, wx + dw],
                [0, 0],
                [100, 100],
                color=c,
                alpha=0.07,
                zorder=-100,
            )

        # Rest ticks on top
        wrest = np.arange(0.1, 1.91, 0.05)
        wrest = np.append(wrest, np.arange(2.0, 5.33, 0.1))
        mrest = np.arange(0.1, 1.91, 0.01)
        mrest = np.append(mrest, np.arange(2.0, 5.33, 0.05))

        xtr = wrest * (1 + z)
        in_range = (xtr > sp["wave"].min()) & (xtr < sp["wave"].max())
        if in_range.sum() > 9:
            wrest = np.arange(0.1, 1.91, 0.1)
            wrest = np.append(wrest, np.arange(2.0, 5.33, 0.2))
            xtr = wrest * (1 + z)
            in_range = (xtr > sp["wave"].min()) & (xtr < sp["wave"].max())

            mrest = np.arange(0.1, 1.91, 0.05)
            mrest = np.append(mrest, np.arange(2.0, 5.33, 0.1))

        if in_range.sum() > 12:
            wrest = np.arange(0.2, 1.81, 0.2)
            wrest = np.append(wrest, np.arange(2.0, 5.33, 0.2))
            xtr = wrest * (1 + z)
            in_range = (xtr > sp["wave"].min()) & (xtr < sp["wave"].max())

            mrest = np.arange(0.1, 1.91, 0.05)
            mrest = np.append(mrest, np.arange(2.0, 5.33, 0.1))

        xtr = xtr[in_range]
        xvr = np.interp(xtr, sp["wave"], np.arange(len(sp)))
        axes[0].set_xticks(xvr, minor=False)
        xticks = [f"{w:0.2}" for w in wrest[in_range]]
        # xticks[0] = f'z={z:.3f}' #' {xticks[0]}'
        axes[0].set_xticklabels(xticks)
        axes[1].text(
            0.03,
            0.90,
            f"z={z:.4f}",
            ha="left",
            va="top",
            transform=axes[1].transAxes,
            fontsize=8,
            bbox={"fc": "w", "alpha": 0.5, "ec": "None"},
        )

        xtr = mrest * (1 + z)
        xtr = xtr[(xtr > sp["wave"].min()) & (xtr < sp["wave"].max())]
        xvr = np.interp(xtr, sp["wave"], np.arange(len(sp)))
        axes[0].set_xticks(xvr, minor=True)

    else:
        axes[0].set_xticklabels([])
        axes[0].grid()

    if "SRCNAME" in hdul["SCI"].header:
        if output_root is not None:
            label = f"{output_root} {hdul['SCI'].header['SRCNAME']}"
        else:
            label = f"{hdul['SCI'].header['SRCNAME']}"

        axes[1].text(
            0.03,
            0.97,
            label,
            ha="left",
            va="top",
            transform=axes[1].transAxes,
            fontsize=8,
            bbox={"fc": "w", "alpha": 0.5, "ec": "None"},
        )

    if xlim is not None:
        xvi = np.interp(xlim, sp["wave"], np.arange(len(sp)))
        for ax in axes:
            ax.set_xlim(*xvi)

    else:
        for ax in axes:
            ax.set_xlim(0, nx)

    fig.tight_layout(pad=0.5)

    return fig


def extract_all():
    """
    demo [not used]
    """
    from importlib import reload

    from msaexp import pipeline

    groups = pipeline.exposure_groups()
    obj = {}

    for g in groups:
        if "395m" not in g:
            continue

        print(f"\n\nInitialize {g}\n\n")
        obj[g] = pipeline.NirspecPipeline(g)
        obj[g].full_pipeline(run_extractions=False)
        obj[g].set_background_slits()

    import msaexp.utils

    reload(msaexp.utils)

    groups = [
        "jw02756001001-01-clear-prism-nrs1",
        "jw02756001001-02-clear-prism-nrs1",
    ]
    gg = groups

    key = "2756_80075"

    slits = []

    for g in gg:
        if g not in obj:
            continue

        self = obj[g]

        if key not in self.slitlets:
            continue

        sl = self.slitlets[key]
        mode = self.mode

        si = self.get_background_slits(key, step="bkg", check_background=True)

        if si is not None:
            for s in si:
                slits.append(s)

    kwargs = {"keep_wave": True}
    kwargs = {
        "keep_wave": False,
        "wave_range": [0.6, 5.3],
        "wave_step": 0.0005,
        "log_wave": True,
    }

    # f290lp g395m
    kwargs = {
        "keep_wave": False,
        "wave_range": [2.87, 5.2],
        "wave_step": 0.001,
        "log_wave": False,
    }

    # kwargs = {'keep_wave':1} #'wave_array':wx}

    prf_kwargs = {"prf_center": None, "prf_sigma": 1.0, "prf_fit": True}

    drizzle_params = dict(
        output=None,
        single=True,
        blendheaders=True,
        pixfrac=0.5,
        kernel="square",
        fillval=0,
        wht_type="ivm",
        good_bits=0,
        pscale_ratio=1.0,
        pscale=None,
    )

    wave, header, drizzled_slits = msaexp.utils.drizzle_slits_2d(
        slits, drizzle_params=drizzle_params, **kwargs
    )
    sci2d, wht2d, profile2d, spec, prof = (
        msaexp.utils.combine_2d_with_rejection(
            drizzled_slits, outlier_threshold=10, **prf_kwargs
        )
    )

    for k in ["name", "ra", "dec"]:
        spec.meta[k] = sl[f"source_{k}"]

    _fitsfile = f"{mode}-{key}.driz.fits"

    spec.write(_fitsfile, overwrite=True)

    with pyfits.open(_fitsfile, mode="update") as hdul:

        hdul[1].header["EXTNAME"] = "SPEC1D"

        for k in spec.meta:
            header[k] = spec.meta[k]

        hdul.append(pyfits.ImageHDU(data=sci2d, header=header, name="SCI"))
        hdul.append(pyfits.ImageHDU(data=wht2d, header=header, name="WHT"))
        hdul.append(
            pyfits.ImageHDU(data=profile2d, header=header, name="PROFILE")
        )

        hdul.flush()


def array_to_bin_edges(array):
    """
    Compute bin edges of an input array where the bin widths are the array steps

    Parameters
    ----------
    array : array-like
        Input array with length ``N``

    Returns
    -------
    bins : array-like
        Bin edges with length ``N+1``

    """
    dw = np.diff(array)
    bins = np.hstack(
        [
            array[0] - dw[0] / 2,
            array[:-1] + dw / 2.0,
            array[-1] + dw[-1] / 2.0,
        ]
    )
    return bins


def pixfrac_steps(oversample, pixfrac):
    """
    Grid for oversampling cross dispersion axis
    """
    steps = (
        np.linspace(
            1.0 / oversample,
            2 + 1.0 / oversample,
            oversample + 1
        )[:-1] - 1
    ) * pixfrac
    return steps


def calculate_psf_fwhm():
    """
    Use WebbPSF to calculate the FWHM as a function of wavelength assuming
    0.2" fixed slit

    Parameters:
    -----------
    None

    Returns:
    --------
    None

    """
    import scipy.stats
    import webbpsf
    import matplotlib.pyplot as plt

    nspec = webbpsf.NIRSpec()
    nspec.image_mask = "S200A1"  # 0.2 arcsec slit

    psfs = {}
    for wave in [0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 5.5]:
        print(wave)
        psfs[wave] = nspec.calc_psf(
            monochromatic=wave * 1.0e-6, oversample=4, detector_oversample=4
        )

    norm = scipy.stats.norm()

    N = len(psfs)
    fig, axes = plt.subplots(
        2, N, figsize=(2 * len(psfs), 4), sharex=False, sharey=False
    )

    fwhm_wave = []

    for i, w in enumerate(psfs):
        psf = psfs[w]
        ax = axes[0][i]
        ext = "OVERSAMP"
        ax.imshow(np.log10(psf[ext].data))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        prof = psf[ext].data.sum(axis=1)
        ax = axes[1][i]
        Nx = len(prof)
        ax.plot(prof / prof.sum(), marker=".")
        cdf = np.cumsum(prof / prof.sum())
        ax.plot(cdf / 2)

        x = np.arange(Nx)
        fwhm = np.diff(
            np.interp(norm.cdf(np.array([-1, 1]) * 2.35 / 2), cdf, x)
        )
        fwhm_wave.append(fwhm[0])

        print(w, fwhm / 4)
        ax.set_ylim(-0.1, 0.5)
        ax.set_xlim(Nx / 2 - 20, Nx / 2 + 20)
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def get_nirspec_psf_fwhm(wave):
    """
    Get PSF FWHM in pix from tabulated array derived from `webbpsf`

    Parameters
    ----------
    wave : float, array-like
        Wavelength in microns

    Returns
    -------
    fwhm : float
        Full Width at Half Maximum (FWHM) of the Point Spread Function (PSF)
        in pixels.

    """
    from .data.fwhm_data import fwhm_data

    fwhm_table = grizli.utils.read_catalog(fwhm_data)

    fwhm = np.interp(
        wave,
        fwhm_table["wave"],
        fwhm_table["fwhm_pix"],
        left=fwhm_table["fwhm_pix"][0],
        right=fwhm_table["fwhm_pix"][-1],
    )

    return fwhm


def get_prism_bar_correction(
    scaled_yshutter,
    num_shutters=3,
    wrap="auto",
    wrap_pad=0.2,
    mask=True,
    bar_data=None,
):
    """
    Generate the `msaexp` prism bar-shadow correction

    Parameters
    ----------
    scaled_yshutter : array-like
        Cross-dispersion pixel coordinates, scaled by a factor of 1/5 to roughly have
        units of the MSA shutters

    num_shutters : [1,2,3]
        Number of shutters in the slitlet

    wrap : bool, str
        If ``auto``, determine if the bounds of ``scaled_yshutter`` are more than 0.5
        shutters outside of the (-1.5, 1.5) range used to determine the correction.
        If so, or if ``wrap=True``, replicate the center shutter to all specified
        shutters.

    wrap_pad : float
        If ``wrap``, pad the outer edges that are unilluminated and won't be
        properly calibrated

    mask : bool
        Apply mask where the bar correction is within the shutter pixels of the
        calibration

    bar_data : None, dict
        Correction data.  If not specified, read from
        ``prism_{num_shutters}_bar_coeffs_wave.yaml``

    Returns
    -------
    bar : array-like
        The estimated bar throughput.  To correct, divide by ``bar``

    is_wrapped : bool
        Was the bar profile wrapped using the single central shutter, e.g., from
        ``wrap='auto'``?

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from msaexp.utils import get_prism_bar_correction

        scaled_yshutter = np.linspace(-1.6, 1.6, 512)

        fig, ax = plt.subplots(1,1,figsize=(6,4))

        for n in [1,2,3]:
            bar, _wrapped = get_prism_bar_correction(scaled_yshutter,
                                                     num_shutters=n,
                                                     wrap=False)
            ax.plot(scaled_yshutter, bar, label=f'{n}-shutter', alpha=0.5)

        ax.legend(loc='lower right', fontsize=6)
        ax.grid()

        ax.set_xlabel('scaled_yshutter = cross-dispersion pixel / 5')
        ax.set_ylabel('bar shadow factor')

        fig.tight_layout(pad=1)
        fig.show()

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import msaexp.utils

        # 5-shutter slitlet
        scaled_yshutter = np.linspace(-2.51, 2.51, 1024)

        fig, ax = plt.subplots(1,1,figsize=(6,4))
        bar, _wrapped = msaexp.utils.get_prism_bar_correction(scaled_yshutter)
        ax.plot(scaled_yshutter, bar, label=f'wrapped: {_wrapped}')
        ax.grid()
        ax.legend()

        ax.set_xlabel('scaled_yshutter = cross-dispersion pixel / 5')
        ax.set_ylabel('bar shadow factor')

        fig.tight_layout(pad=1)
        fig.show()

    .. code-block:: python
        :dedent:

        slit_file = 'jw04233006001_03101_00002_nrs2_phot.076.4233_75646.fits'
        slit = jwst.datamodels.open(slit_file)
        # get the trace center
        _trace = msaexp.utils.slit_trace_center(
            slit,
            with_source_xpos=False,
            with_source_ypos=False
        )
        # scaled coordinates
        yp, xp = np.indices(slit.data.shape)
        scaled_yshutter = (yp - _trace[1]) / 5
        bar, _wrapped = msaexp.utils.get_prism_bar_correction(scaled_yshutter)

        plt.imshow(bar, aspect='auto')

    """
    import yaml

    if bar_data is None:
        path_to_ref = os.path.join(
            os.path.dirname(__file__),
            "data",
            f"prism_{num_shutters}_bar_coeffs.yaml",
        )

        with open(path_to_ref) as fp:
            bar_data = yaml.load(fp, Loader=yaml.Loader)

    df = len(bar_data["coeffs"])

    if wrap in ["auto"]:
        is_wrapped = np.nanmin(scaled_yshutter) < bar_data["minmax"][0] - 0.5
        is_wrapped |= np.nanmin(scaled_yshutter) > bar_data["minmax"][1] + 0.5
    elif wrap in [True]:
        is_wrapped = True
    else:
        is_wrapped = False

    if is_wrapped:
        pass_yshutter = ((scaled_yshutter.flatten() + 0.5) % 1) - 0.5
        mask = False
    else:
        pass_yshutter = scaled_yshutter.flatten()

    _spl = grizli.utils.bspline_templates(
        pass_yshutter,
        df=df,
        get_matrix=True,
        minmax=bar_data["minmax"],
    )

    sh = scaled_yshutter.shape
    bar = _spl.dot(bar_data["coeffs"]).reshape(sh)

    if mask:
        bar[pass_yshutter.reshape(sh) < (bar_data["minmax"][0])] = np.nan
        bar[pass_yshutter.reshape(sh) > (bar_data["minmax"][1])] = np.nan

        if is_wrapped:
            extra_mask = scaled_yshutter < (
                np.nanmin(scaled_yshutter) + wrap_pad
            )
            extra_mask |= scaled_yshutter > (
                np.nanmin(scaled_yshutter) - wrap_pad
            )
            bar[extra_mask] = np.nan

    return bar, is_wrapped


def get_prism_wave_bar_correction(
    scaled_yshutter,
    wavelengths,
    num_shutters=3,
    wrap="auto",
    wrap_pad=0.2,
    mask=True,
    bar_data=None,
):
    """
    Generate the `msaexp` prism bar-shadow correction including wavelength dependence

    Parameters
    ----------
    scaled_yshutter : array-like
        Cross-dispersion pixel coordinates, scaled by a factor of 1/5 to roughly have
        units of the MSA shutters

    wavelengths : array-like
        Wavelength of the pixel samples, microns

    num_shutters : [1,2,3]
        Number of shutters in the slitlet

    wrap : bool, str
        If ``auto``, determine if the bounds of ``scaled_yshutter`` are more than 0.5
        shutters outside of the (-1.5, 1.5) range used to determine the correction.
        If so, or if ``wrap=True``, replicate the center shutter to all specified
        shutters.

    wrap_pad : float
        If ``wrap``, pad the outer edges that are unilluminated and won't be
        properly calibrated

    mask : bool
        Apply mask where the bar correction is within the shutter pixels of the
        calibration

    bar_data : None, dict
        Correction data.  If not specified, read from
        ``prism_{num_shutters}_bar_coeffs_wave.yaml``

    Returns
    -------
    bar : array-like
        The estimated bar throughput.  To correct, divide by ``bar``

    is_wrapped : bool
        Was the bar profile wrapped using the single central shutter, e.g., from
        ``wrap='auto'``?

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from msaexp.utils import get_prism_wave_bar_correction

        scaled_yshutter = np.linspace(-1.6, 1.6, 512)

        fig, ax = plt.subplots(1,1,figsize=(6,4))

        for w in [1.0, 2.0, 3.0, 4.0, 5.0]:
            bar, _wrapped = get_prism_wave_bar_correction(
                                    scaled_yshutter,
                                    np.full_like(scaled_yshutter, w),
                                    num_shutters=3,
                                    wrap=False)

            ax.plot(scaled_yshutter, bar,
                    label=f'{w:.0f} um',
                    alpha=0.5,
                    color=plt.cm.RdYlBu_r(np.interp(w, [0.8, 5.3], [0, 1]))
                    )

        ax.legend(loc='lower right', fontsize=6)
        ax.grid()

        ax.set_xlabel('scaled_yshutter = cross-dispersion pixel / 5')
        ax.set_ylabel('bar shadow factor')

        fig.tight_layout(pad=1)
        fig.show()
    """
    import yaml

    if bar_data is None:
        path_to_ref = os.path.join(
            os.path.dirname(__file__),
            "data",
            f"prism_{num_shutters}_bar_coeffs_wave.yaml",
        )

        with open(path_to_ref) as fp:
            bar_data = yaml.load(fp, Loader=yaml.Loader)

    df = len(bar_data["coeffs"])

    if wrap in ["auto"]:
        is_wrapped = np.nanmin(scaled_yshutter) < bar_data["minmax"][0] - 0.5
        is_wrapped |= np.nanmin(scaled_yshutter) > bar_data["minmax"][1] + 0.5
    elif wrap in [True]:
        is_wrapped = True
    else:
        is_wrapped = False

    if is_wrapped:
        pass_yshutter = ((scaled_yshutter.flatten() + 0.5) % 1) - 0.5
        mask = False
    else:
        pass_yshutter = scaled_yshutter.flatten()

    _spl = grizli.utils.bspline_templates(
        pass_yshutter,
        df=df,
        get_matrix=True,
        minmax=bar_data["minmax"],
    ).T

    wave_coeffs = np.array(
        [
            np.interp(
                wavelengths.flatten(),
                bar_data["wavelengths"],
                c_i,
                left=c_i[0],
                right=c_i[-1],
            )
            for c_i in bar_data["coeffs"]
        ]
    )

    sh = scaled_yshutter.shape
    bar = (wave_coeffs * _spl).sum(axis=0).reshape(sh)

    if mask:
        bar[pass_yshutter.reshape(sh) < (bar_data["minmax"][0])] = np.nan
        bar[pass_yshutter.reshape(sh) > (bar_data["minmax"][1])] = np.nan

        if is_wrapped:
            extra_mask = scaled_yshutter < (
                np.nanmin(scaled_yshutter) + wrap_pad
            )
            extra_mask |= scaled_yshutter > (
                np.nanmin(scaled_yshutter) - wrap_pad
            )
            bar[extra_mask] = np.nan

    return bar, is_wrapped


def cache_badpix_arrays():
    """
    Load badpixel arrays into global data
    """
    global MSAEXP_BADPIX

    MSAEXP_BADPIX = {}

    for detector in ["NRS1", "NRS2"]:
        path_to_ref = os.path.join(
            os.path.dirname(__file__),
            "data",
            f"msaexp_badpix_{detector}.yaml".lower(),
        )

        with open(path_to_ref) as fp:
            dq_data = yaml.load(fp, Loader=yaml.Loader)

        dq_arr = np.zeros(dq_data["shape"], dtype=int).flatten()
        dq_arr[dq_data["pixel_index"]] = dq_data["pixel_value"]

        MSAEXP_BADPIX[detector] = (
            dq_arr.reshape(dq_data["shape"]),
            dq_data["flags"],
            path_to_ref,
        )


def extra_slit_dq_flags(slit, dq_arr=None, verbose=True):
    """
    Set extra DQ flags, including a big stuck open shutter on NRS1

    Parameters
    ----------
    slit : `~jwst.datamodels.SlitModel`
        Slitlet data

    dq_arr : array-like
        Full-frame dq array for the appropriate detector.  If not specified,
        read from the reference files in ``msaexp/data``.

    Returns
    -------
    dq : array-like
        Extra DQ flags

    flags : dict
        Description of DQ flags
    """
    import yaml

    global MSAEXP_BADPIX

    if MSAEXP_BADPIX is None:
        cache_badpix_arrays()

    if dq_arr is None:
        dq_arr, flags, path_to_ref = MSAEXP_BADPIX[
            slit.meta.instrument.detector
        ]
        path_to_ref = os.path.basename(path_to_ref)
    else:
        path_to_ref = "(user-supplied)"
        flags = {}

    slx = slice(slit.xstart - 1, slit.xstart - 1 + slit.xsize)
    sly = slice(slit.ystart - 1, slit.ystart - 1 + slit.ysize)

    dq = dq_arr[sly, slx]  # .reshape(dq_data["shape"])[sly, slx]

    msg = f" extra_slit_dq_flags: ({os.path.basename(path_to_ref)})  N={(dq > 0).sum()}"
    grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=verbose)

    return dq, flags


def slit_normalization_correction(slit, verbose=True):
    """
    Run `~msaexp.utils.get_normalization_correction` for a ``SlitModel`` object

    Parameters
    ----------
    slit : `~jwst.datamodels.SlitModel`
        Slitlet data

    Returns
    -------
    corr : array-like
        Correction to apply to slit data, i.e., ``corrected = slit.data * corr``

    """
    corr = get_normalization_correction(
        slit.wavelength,
        slit.quadrant,
        slit.xcen,
        slit.ycen,
        grating=slit.meta.instrument.grating,
        verbose=verbose,
    )
    return corr


def get_normalization_correction(
    wavelengths, quadrant, xcen, ycen, grating="PRISM", verbose=True
):
    """
    Normalization correction derived from empty sky slits, analagous to a correction
    to the SFLAT calibration

    Parameters
    ----------
    wavelengths : array-like
        Sample wavelengths

    quadrant : [1,2,3,4]
        MSA quadrant

    xcen : int
        MSA column, 1-365

    ycen : int
        MSA row, 1-171

    grating : str
        Grating name (just ``'PRISM'`` implemented so far)

    Returns
    -------
    corr : array-like
        Correction to apply to slit data, i.e., ``corrected = slit.data * corr``

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from msaexp.utils import get_normalization_correction

        waves = np.linspace(0.8, 5.2, 256)

        fig, ax = plt.subplots(1, 1, figsize=(6,4))

        corr = get_normalization_correction(waves, 1, 180, 85, grating="PRISM")
        ax.plot(waves, corr)

        ax.legend(loc='lower right', fontsize=6)
        ax.grid()
        ax.set_ylim(0.8, 1.2)
        ax.hlines([1.], 0.7, 5.3, color='r', linestyle=':')

        ax.set_xlabel('wavelength, um')
        ax.set_ylabel('normalization')

        fig.tight_layout(pad=1)
        fig.show()
    """

    import yaml

    if grating not in ["PRISM"]:
        return np.ones_like(wavelengths)

    if quadrant not in [1, 2, 3, 4]:
        msg = "msaexp.utils.get_normalization_correction: "
        msg += f"quadrant={quadrant} must be one of [1,2,3,4]"
        grizli.utils.log_comment(
            grizli.utils.LOGFILE,
            msg,
            verbose=verbose,
        )
        return np.ones_like(wavelengths)

    path_to_ref = os.path.join(
        os.path.dirname(__file__),
        "data",
        f"{grating.lower()}_slit_renormalize.yaml",
    )

    if not os.path.exists(path_to_ref):
        msg = f"msaexp.utils.get_normalization_correction: {path_to_ref} not found"
        grizli.utils.log_comment(
            grizli.utils.LOGFILE,
            msg,
            verbose=verbose,
        )
        return np.ones_like(wavelengths)

    msg = f" get_normalization_correction: {os.path.basename(path_to_ref)}"
    msg += f" quadrant={quadrant} xcen={xcen} ycen={ycen}"
    grizli.utils.log_comment(
        grizli.utils.LOGFILE,
        msg,
        verbose=verbose,
    )

    with open(path_to_ref) as fp:
        norm_data = yaml.load(fp, Loader=yaml.Loader)

    cdata = np.array(norm_data["coeffs"])

    iq = quadrant - 1
    coeffs = cdata[:, iq, 0] + cdata[:, iq, 1] * xcen + cdata[:, iq, 2] * ycen

    ref_wave = np.array(norm_data["reference_wavelengths"])

    nbin = len(ref_wave)
    xbin = np.interp(
        wavelengths,
        ref_wave,
        np.arange(nbin) / nbin,
        left=np.nan,
        right=np.nan,
    )

    spl_full = grizli.utils.bspline_templates(
        xbin.flatten(),
        df=norm_data["df"],
        minmax=(0, 1),
        get_matrix=True,
    )

    corr = 1.0 / spl_full.dot(coeffs)

    return corr.reshape(wavelengths.shape)


def make_nirspec_gaussian_profile(
    waves,
    sigma=0.5,
    ycenter=0.0,
    ny=31,
    weight=1,
    bkg_offset=6,
    bkg_parity=[-1, 1],
):
    """
    Make a pixel-integrated Gaussian profile

    Parameters:
    ----------
    waves : array-like
        Array of wavelengths in microns
    sigma : float, optional
        Standard deviation of the Gaussian profile. Default is 0.5.
    ycenter : float, optional
        Y-coordinate of the center of the profile. Default is 0.
    ny : int, optional
        Number of pixels in the y-direction. Default is 31.
    weight : int, optional
        Weight of the profile. Default is 1.
    bkg_offset : int, optional
        Offset for nodded background subtraction. Default is 6.
    bkg_parity : list, optional
        List of integers specifying the parity of the background nod offsets.
        Default is [-1, 1].

    Returns:
    -------
    prf : array-like
        2D pixel-integrated Gaussian profile.

    """
    import scipy.special

    yp, xp = np.indices((ny + 1, len(waves)))
    y0 = (ny - 1) / 2

    psf_fwhm = get_nirspec_psf_fwhm(waves)
    sig = np.sqrt((psf_fwhm / 2.35) ** 2 + sigma**2)

    ysig = (yp - y0 - 0.5 - ycenter) / sig

    cdf = 1 / 2 * (1 + scipy.special.erf(ysig / np.sqrt(2)))

    prf = np.diff(cdf, axis=0)

    ###
    # Don't use weights
    # weight = np.ones_like(prf)

    weight = weight > 0

    if bkg_offset is not None:
        if bkg_offset > 0:
            bkgn = prf * 0.0
            bkgd = bkgn * 0

            for p in bkg_parity:
                bkgn += np.roll(prf * weight, p * bkg_offset, axis=0)
                bkgd += np.roll(weight, p * bkg_offset, axis=0)

            prf -= bkgn / bkgd

    return prf


def objfun_prf(
    params,
    waves,
    sci2d,
    wht2d,
    ycenter,
    sigma,
    bkg_offset,
    bkg_parity,
    fit_type,
    ret,
    verbose,
):
    """
    Objective function for fitting the 2D profile

    Parameters:
    ----------
    params : array_like
        The parameters for the fit
    waves : array_like
        The wavelength values.
    sci2d : array_like
        The 2D science data.
    wht2d : array_like
        The 2D weight data.
    ycenter : float
        The y-coordinate of the center of the profile.
    sigma : float
        The standard deviation of the profile.
    bkg_offset : float
        The background offset, pixels.
    bkg_parity : int
        Parity of the nod offsets, multiplied to ``bkg_offset``
    fit_type : int
        Fit behavior
    ret : int
        Control fit outputs for either fitting or returning the
        model given ``params``
    verbose : bool
        Print verbose output.

    Returns:
    -------
    if ret == 1:
    norm : float
        The normalization factor.
    model : array_like
        The model profile.

    if ret == 2:
    chi : array_like
        The chi values.

    if ret == 3:
    chi2 : float
        The chi-squared value.

    """
    if fit_type == 1:
        sigma = params[0]
    elif fit_type == 2:
        ycenter = params[0]
    else:
        ycenter, sigma = params

    ny = sci2d.shape[0]
    prf = make_nirspec_gaussian_profile(
        waves,
        sigma=sigma,
        ycenter=ycenter,
        ny=ny,
        weight=wht2d,
        bkg_offset=bkg_offset,
        bkg_parity=bkg_parity,
    )

    prf *= prf > 0

    ok = (
        np.isfinite(sci2d * prf * wht2d) & (wht2d > 0) & (sci2d != 0)
    )  # & (prf != 0)
    # ok &= sci2d*np.sqrt(wht2d) > -5

    okp = np.isfinite(prf)

    norm = (sci2d * prf * wht2d)[ok & okp].sum() / (prf**2 * wht2d)[
        ok & okp
    ].sum()
    model = norm * prf

    if 0:
        p2 = prf * 1
        p2 *= p2 > 0

        num = np.nansum((sci2d * p2 * wht2d), axis=0)
        den = np.nansum((p2**2 * wht2d), axis=0)
        norm = num / den

        model = norm * prf

    chi = (sci2d - model) * np.sqrt(wht2d)

    chi2 = (chi[ok] ** 2).sum()
    if verbose:
        print(params, chi2)

    if ret == 1:
        return norm, model
    elif ret == 2:
        return chi[ok].flatten()
    elif ret == 3:
        return chi2

