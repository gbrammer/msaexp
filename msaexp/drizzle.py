"""
Tools for drizzle-combining MSA spectra
"""

import glob
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

import jwst.datamodels
import astropy.io.fits as pyfits

import grizli.utils

from . import utils

# Parameter defaults
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

FIGSIZE = (10, 4)

IMSHOW_KWS = dict(
    vmin=-0.1,
    vmax=None,
    aspect="auto",
    interpolation="nearest",
    origin="lower",
    cmap="cubehelix_r",
)


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
    return (
        slit.meta.instrument.msa_metadata_file,
        slit.meta.instrument.msa_metadata_id,
    )


def center_wcs(
    slit,
    waves,
    center_on_source=False,
    force_nypix=31,
    fix_slope=None,
    slit_center=0.0,
    center_phase=-0.5,
):
    """
    Derive a 2D spectral WCS centered on the expected source position along
    the slit

    Parameters
    ----------
    slit : `jwst.datamodels.SlitModel`
        Slitlet data object

    waves : array-like
        Target wavelength array

    center_on_source : bool
        Center on the source position along the slit.  If not, center on
        `slit_center` slit coordinate

    force_nypix : int
        Cross-dispersion size of the output 2D WCS

    fix_slope : float
        Fixed cross-dispersion pixel size, in units of the slit coordinate
        frame

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
    wcs_data = utils.build_slit_centered_wcs(
        slit,
        waves,
        force_nypix=force_nypix,
        center_on_source=True,
        get_from_ypos=False,
        phase=center_phase,
        fix_slope=fix_slope,
        slit_center=slit_center,
    )

    slit.drizzle_slit_offset_source = slit.drizzle_slit_offset

    # Centered on slitlet
    if center_on_source:
        offset_to_source = 0.0
    else:
        wcs_data = utils.build_slit_centered_wcs(
            slit,
            waves,
            force_nypix=force_nypix,
            center_on_source=False,
            get_from_ypos=False,
            phase=center_phase,
            fix_slope=fix_slope,
            slit_center=slit_center,
        )

        # Derived offset between source and center of slitlet
        offset_to_source = (
            slit.drizzle_slit_offset_source - slit.drizzle_slit_offset
        )

    return wcs_data, offset_to_source, metadata_tuple(slit)


def drizzle_slitlets(
    id,
    wildcard="*phot",
    files=None,
    output=None,
    verbose=True,
    drizzle_params=DRIZZLE_PARAMS,
    master_bkg=None,
    wave_arrays={},
    wave_sample=1,
    log_step=True,
    force_nypix=31,
    center_on_source=False,
    center_phase=-0.5,
    fix_slope=None,
    outlier_threshold=5,
    sn_threshold=3,
    bar_threshold=-0.7,
    err_threshold=1000,
    bkg_offset=5,
    bkg_parity=[1, -1],
    mask_padded=False,
    show_drizzled=True,
    show_slits=True,
    imshow_kws=IMSHOW_KWS,
    get_quick_data=False,
    max_sn_threshold=20,
    reopen=True,
    **kwargs,
):
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
        Explicit list of either slitlet filenames or
        `jwst.datamodels.SlitModel` objects.

    output : str
        Optional rootname of output figures and FITS data

    verbose : bool
        Verbose messaging

    drizzle_params : dict
        Drizzle parameters passed to `msaexp.utils.drizzle_slits_2d`

    master_bkg : array-like, int
        Master background to replace local background derived from the
        drizzled product

    wave_arrays : dict
        Explicit target wavelength arrays with keys for `{grating}-{filter}`
        combinations

    wave_sample, log_step : float, bool
        If `waves` not specified, generate with
        `msaexp.utils.get_standard_wavelength_grid`

    force_nypix, center_on_source, center_phase, fix_slope : int, bool, float
        Parameters of `msaexp.drizzle.center_wcs`

    outlier_threshold : int
        Outlier threshold in drizzle combination

    sn_threshold : float
        Mask pixels in slitlets where `data/err < sn_threshold`.  For the
        prism, essentially all pixels should have S/N > 5 from the background,
        so this mask can help identify and mask stuck-closed slitlets

    bar_threshold : float
        Mask pixels in slitlets where `barshadow < bar_threshold`

    err_threshold : float
        Mask pixels in slitlets where `err > err_threshold*median(err)`.  There
        are some strange pixels with very large uncertainties in the pipeline
        products.

    bkg_offset, bkg_parity : int, list
        Offset in pixels for defining the local background of the drizzled
        product, which is derived by rolling the data array by
        `bkg_offset*bkg_parity` pixels.  The standard three-shutter nod pattern
        corresponds to about 5 pixels.  An optimal combination seems
        to be ``fix_slope=0.2``, ``bkg_offset=6``.

        If ``bkg_offset < 0``, then don't do shifted offset.

    mask_padded : bool
        Mask pixels of slitlets that had been padded around the nominal MSA
        slitlets

    show_drizzled : bool
        Make a figure with `msaexp.drizzle.show_drizzled_product` showing the
        drizzled combined arrays.  If `output` specified, save to
        `{output}-{id}-[grating].d2d.png`.

    show_slits : bool
        Make a figure with `msaexp.drizzle.show_drizzled_slits` showing the
        individual drizzled slitlets.  If `output` specified, save to
        `{output}-{id}-[grating].slit2d.png`.

    imshow_kws : dict
        Keyword arguments for ``matplotlib.pyplot.imshow`` in `show_drizzled`
        and `show_slits` figures.

    get_quick_data : bool
        Just return `waves`, `slits`, and the drizzled `sci` and `err` data
        arrays before doing any outlier rejection, etc.

    max_sn_threshold : float
        S/N threshold for initial rejection of the maximum pixel in the set

    reopen : bool
        Re-initialize `jwst.datamodels.SlitModel` before drizzling to fix
        apparent memory leak issue

    Returns
    -------
    figs : dict
        Any figures that were created, keys are separated by grating+filter
        here and below

    data : dict
        `~astropy.io.fits.HDUList` FITS data for the drizzled output

    wavedata : dict
        Wavelength arrays

    all_slits : dict
        `SlitModel` objects for the input slitlets

    drz_data : dict
        3D `sci` and `wht` arrays of the drizzled slitlets that were combined
        into the drizzled stack

    """

    if files is None:
        files = glob.glob(f"{wildcard}*_{id}.fits")
        # files = glob.glob(f'*{pipeline_extension}*_{id}.fits')
        files.sort()

    # Read the SlitModels
    gratings = {}
    grating_files = {}

    msg = f"msaexp.drizzle.drizzle_slitlets: {id} read {len(files)} files"
    grizli.utils.log_comment(
        grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
    )

    for file in files:
        if isinstance(file, str):
            slit = jwst.datamodels.SlitModel(file)
            utils.update_slit_metadata(slit)
        else:
            slit = file
            reopen = False

        grating = slit.meta.instrument.grating.lower()
        filt = slit.meta.instrument.filter.lower()
        key = f"{grating}-{filt}"
        if key not in gratings:
            gratings[key] = []
            grating_files[key] = []

        gratings[key].append(slit)
        grating_files[key].append(file)

    if verbose:
        for g in gratings:
            msg = f"msaexp.drizzle.drizzle_slitlets: id={id}  {g}"
            msg += f" N={len(gratings[g])}"
            grizli.utils.log_comment(
                grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
            )

    # DQ mask
    for gr in gratings:
        slits = gratings[gr]
        for slit in slits:
            utils.update_slit_dq_mask(
                slit,
                mask_padded=mask_padded,
                bar_threshold=bar_threshold,
                verbose=False,
            )

    # Loop through gratings
    figs = {}
    data = {}
    wavedata = {}
    all_slits = {}
    drz_data = {}

    for gr in gratings:
        slits = gratings[gr]

        # Default wavelengths
        if gr in wave_arrays:
            waves = wave_arrays[gr]
        else:
            waves = utils.get_standard_wavelength_grid(
                gr.split("-")[0], sample=wave_sample, log_step=log_step
            )

        # Drizzle 2D spectra
        drz = None
        drz_ids = []

        wcs_data = None

        # Approximate for default
        to_ujy = 1.0e12 * 5.0e-13

        wcs_meta = None

        # Get offset from one science slit
        for i in range(len(slits)):  # [18:40]:
            slit = slits[i]
            if "background" in slit.source_name:
                continue
            elif "-" in slit.source_name:
                continue
            elif slit.data.shape[1] < 50:
                continue

            msg = f"msaexp.drizzle.drizzle_slitlets: get wcs from slit {i} = "
            msg += f" {slit.source_name}"
            grizli.utils.log_comment(
                grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
            )

            _center = center_wcs(
                slit,
                waves,
                force_nypix=force_nypix,
                center_on_source=center_on_source,
                fix_slope=fix_slope,
                center_phase=center_phase,
            )

            wcs_data, offset_to_source, wcs_meta = _center
            try:
                to_ujy = 1.0e12 * slit.meta.photometry.pixelarea_steradians
            except TypeError:
                to_ujy = 1.0

            break

        if wcs_meta is None:
            # Run for background slits centered on slitlet if all skipped above
            _center = center_wcs(
                slits[0],
                waves,
                force_nypix=force_nypix,
                center_on_source=False,
                fix_slope=fix_slope,
                slit_center=1.0,
                center_phase=center_phase,
            )

            wcs_data, offset_to_source, wcs_meta = _center
            try:
                to_ujy = 1.0e12 * slits[0].meta.photometry.pixelarea_steradians
            except TypeError:
                to_ujy = 1.0

        ##################
        # Now do the drizzling

        msg = f"msaexp.drizzle.drizzle_slitlets: output size = {wcs_data[2]}"
        grizli.utils.log_comment(
            grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
        )

        # FITS header metadata
        h = pyfits.Header()

        wcs_header = wcs_data[1]
        for k in wcs_header:
            h[k] = wcs_header[k], wcs_header.comments[k]

        h["BKGOFF"] = bkg_offset, "Background offset"
        h["OTHRESH"] = outlier_threshold, "Outlier mask threshold, sigma"
        h["WSAMPLE"] = wave_sample, "Wavelength sampling factor"
        h["LOGWAVE"] = log_step, "Target wavelengths are log spaced"

        h["BUNIT"] = "mJy"

        inst = slits[0].meta.instance["instrument"]
        for k in ["grating", "filter"]:
            h[k] = inst[k]

        h["NFILES"] = len(slits), "Number of extracted slitlets"
        h["EFFEXPTM"] = 0.0, "Total effective exposure time"

        for slit in slits:
            h["SRCNAME"] = slit.source_name, "source_name from MSA file"
            h["SRCID"] = slit.source_id, "source_id from MSA file"
            h["SRCRA"] = slit.source_ra, "source_ra from MSA file"
            h["SRCDEC"] = slit.source_dec, "source_dec from MSA file"

            if slit.source_ra > 0:
                break

        to_ujy_list = []

        for i in range(len(slits)):  # [18:40]:
            slit = slits[i]

            _file = grating_files[gr][i]

            msg = "msaexp.drizzle.drizzle_slitlets: "
            msg += f"{gr} {i:2} {slit.source_name:18} {slit.source_id:9}"
            msg += f" {slit.source_ypos:6.3f} {_file} {slit.data.shape}"
            grizli.utils.log_comment(
                grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
            )

            drz_ids.append(slit.source_name)

            h["EFFEXPTM"] += slit.meta.exposure.effective_exposure_time

            if (metadata_tuple(slit) != wcs_meta) & center_on_source:

                _center = center_wcs(
                    slit,
                    waves,
                    force_nypix=force_nypix,
                    center_on_source=center_on_source,
                    fix_slope=fix_slope,
                    center_phase=center_phase,
                )

                wcs_data, offset_to_source, wcs_meta = _center

                msg = "msaexp.drizzle.drizzle_slitlets: "
                msg += f"Recenter on source ({metadata_tuple(slit)})"
                msg += f" y={offset_to_source:.2f}"
                grizli.utils.log_comment(
                    grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
                )

                # Recalculate photometry scaling
                try:
                    to_ujy = 1.0e12 * slit.meta.photometry.pixelarea_steradians
                except TypeError:
                    to_ujy = 1.0

            # Slit metadata

            # Do the drizzling
            if reopen:
                # _slit = jwst.datamodels.SlitModel(grating_files[gr][i])
                # utils.update_slit_metadata(_slit)
                _slit = utils.update_slit_dq_mask(
                    grating_files[gr][i],
                    mask_padded=mask_padded,
                    bar_threshold=bar_threshold,
                    verbose=False,
                )

            else:
                _slit = slit

            _waves, _header, _drz = utils.drizzle_slits_2d(
                [_slit], build_data=wcs_data, drizzle_params=drizzle_params
            )

            to_ujy_list.append(to_ujy)

            if drz is None:
                drz = _drz
            else:
                drz.extend(_drz)

            slit.close()
            if reopen:
                _slit.close()

        drz_ids = np.array(drz_ids)

        # Are slitlets tagged as background?
        is_bkg = np.zeros(len(drz_ids), dtype=bool)

        ############
        # Combined drizzled spectra

        # First pass - max-clipped median
        sci = np.array([d.data * to_ujy_list[i] for i, d in enumerate(drz)])
        err = np.array([d.err * to_ujy_list[i] for i, d in enumerate(drz)])

        if get_quick_data:
            return waves, slits, sci, err

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            scimax = np.nanmax(sci, axis=0)

        flagged = (sci >= scimax) & (sci / err > max_sn_threshold)
        flagged |= (err <= 0) | (~np.isfinite(sci)) | (~np.isfinite(err))
        flagged |= sci == 0
        flagged |= ~np.isfinite(sci / err)
        flagged |= sci / err < sn_threshold

        for i in range(len(drz_ids)):
            ei = err[i, :, :]
            emi = np.isfinite(ei) & (ei > 0)
            emask = ei > err_threshold * np.median(ei[emi])
            flagged[i, :, :] |= emask

        ivar = 1.0 / err**2
        sci[flagged] = np.nan
        ivar[flagged] = 0

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            avg = np.nanmedian(sci, axis=0)

        avg_w = ivar.sum(axis=0)

        # Subsequent passes - weighted average with outlier rejection
        for _iter in range(3):
            sci = np.array(
                [d.data * to_ujy_list[i] for i, d in enumerate(drz)]
            )
            err = np.array([d.err * to_ujy_list[i] for i, d in enumerate(drz)])

            flagged = np.abs(sci - avg) * np.sqrt(avg_w) > outlier_threshold

            flagged |= (err <= 0) | (~np.isfinite(sci)) | (~np.isfinite(err))
            flagged |= sci == 0
            flagged |= ~np.isfinite(sci / err)
            flagged |= sci / err < sn_threshold

            for i in range(len(drz_ids)):
                ei = err[i, :, :]
                emi = np.isfinite(ei) & (ei > 0)
                emask = ei > err_threshold * np.median(ei[emi])
                flagged[i, :, :] |= emask

            ivar = 1.0 / err**2
            sci[flagged] = 0
            ivar[flagged] = 0

            # Weighted combination of unmasked pixels
            avg = (sci * ivar)[~is_bkg, :].sum(axis=0) / ivar[~is_bkg, :].sum(
                axis=0
            )
            avg_w = ivar[~is_bkg, :].sum(axis=0)

            # Set masked pixels to zero
            msk = ~np.isfinite(avg + avg_w)
            avg[msk] = 0
            avg_w[msk] = 0

        # Use master background if supplied
        if master_bkg is not None:
            if hasattr(master_bkg, "__len__"):
                bkg = master_bkg[0]
                bkg_w = master_bkg[1]
            elif master_bkg in [0, 0.0]:
                bkg = np.zeros_like(avg)
                bkg_w = np.zeros_like(avg)
        elif bkg_offset < 0:
            bkg = np.zeros_like(avg)
            bkg_w = np.zeros_like(avg)
        else:
            # Background by rolling full drizzled array
            bkg_num = avg * 0.0
            bkg_w = avg * 0
            for s in bkg_parity:
                bkg_num += np.roll(avg * avg_w, s * bkg_offset, axis=0)
                bkg_w += np.roll(avg_w, s * bkg_offset, axis=0)

            bkg_w[:bkg_offset] = 0
            bkg_w[-bkg_offset:] = 0
            bkg = bkg_num / bkg_w

        # Set masked back to nan
        avg[msk] = np.nan
        avg_w[msk] = np.nan

        # Trace center
        y0 = (avg.shape[0] - 1) // 2
        to_source = y0 + offset_to_source

        h["SRCYPIX"] = to_source, "Expected row of source centering"

        # Valid data along wavelength axis
        xvalid = np.isfinite(avg).sum(axis=0) > 0
        xvalid &= nd.binary_erosion(
            nd.binary_dilation(xvalid, iterations=2), iterations=4
        )

        # Build HDUList
        h["EXTNAME"] = "SCI"
        hdul = pyfits.HDUList([pyfits.PrimaryHDU()])
        hdul.append(pyfits.ImageHDU(data=avg, header=h))

        h["EXTNAME"] = "WHT"
        hdul.append(pyfits.ImageHDU(data=avg_w, header=h))

        h["EXTNAME"] = "BKG"
        hdul.append(pyfits.ImageHDU(data=bkg, header=h))

        h["EXTNAME"] = "WAVE"
        h["BUNIT"] = "micron"
        hdul.append(pyfits.ImageHDU(data=waves, header=h))

        # output_key = f'{output}-{id}-{gr}'
        output_key = f"{output}_{gr}_{id}"  # match spec.fits

        if output is not None:
            hdul.writeto(
                f"{output_key}.d2d.fits", overwrite=True, output_verify="fix"
            )

        if imshow_kws["vmax"] is None:
            vmax = np.nanpercentile(avg, 95) * 2
            print("xxx", vmax)
            imshow_kws["vmax"] = vmax
            imshow_kws["vmin"] = -0.1 * vmax
            reset_vmax = True
        else:
            reset_vmax = False

        # Make a figure
        if show_drizzled:
            dfig = show_drizzled_product(hdul, imshow_kws=imshow_kws)
            if output is not None:
                dfig.savefig(f"{output_key}.d2d.png")
        else:
            dfig = None

        if show_slits:
            sfig = show_drizzled_slits(
                slits,
                sci,
                ivar,
                hdul,
                imshow_kws=imshow_kws,
                with_background=(show_slits > 1),
            )
            if output is not None:
                sfig.savefig(f"{output_key}.slit2d.png")
        else:
            sfig = None

        if reset_vmax:
            imshow_kws["vmax"] = None

        # Add to data dicts
        figs[gr] = dfig, sfig
        data[gr] = hdul
        wavedata[gr] = waves
        all_slits[gr] = slits
        drz_data[gr] = sci, ivar

    return figs, data, wavedata, all_slits, drz_data


def show_drizzled_slits(
    slits,
    sci,
    ivar,
    hdul,
    figsize=FIGSIZE,
    variable_size=True,
    imshow_kws=IMSHOW_KWS,
    with_background=False,
):
    """
    Make a figure showing drizzled slitlets

    Parameters
    ----------
    slits : list
        List of slitlet objects

    sci : (N,NY,NX) array
        Science array of drizzled slitlets

    ivar : (N,NY,NX) array
        Science array if inverse variance weights

    hdul : `~astropy.io.fits.HDUList`
        Drizzle-combined HDU

    figsize : tuple
        Figure size

    variable_size : bool, optional
        Whether to use variable size for the figure.

    imshow_kws : dict
        Keywords passed to `~matplotlib.pyplot.imshow`

    with_background : bool, optional
        Whether to include the background in the displayed slitlets.

    Returns
    -------
    fig : Figure
    """
    avg = hdul["SCI"].data
    xvalid = np.isfinite(avg).sum(axis=0) > 0
    if xvalid.sum() > 1:
        xr = np.arange(avg.shape[1])[xvalid]
    else:
        xr = np.arange(avg.shape[1])

    bkg = hdul["BKG"].data

    h = hdul["SCI"].header
    bkg_offset = np.abs(h["BKGOFF"])
    x0 = h["SRCYPIX"]
    y0 = (avg.shape[0] - 1) // 2

    msk = (ivar > 0) * 1.0
    msk[ivar <= 0] = np.nan

    if variable_size:
        fs = (figsize[0], figsize[1] / 3 * len(slits))
    else:
        fs = figsize

    fig, axes = plt.subplots(
        len(slits), 1, figsize=fs, sharex=True, sharey=True
    )

    for i, slit in enumerate(slits):
        axes[i].imshow(
            (sci[i, :, :] - bkg * with_background) * msk[i, :, :], **imshow_kws
        )

        axes[i].text(
            0.02,
            0.02 * figsize[1] / figsize[0] * len(slits) * 2,
            slit.meta.filename,
            ha="left",
            va="bottom",
            transform=axes[i].transAxes,
            bbox={"fc": "w", "alpha": 0.5, "ec": "None"},
            fontsize=6,
        )

    for ax in axes:
        ax.set_yticks([0, x0 - bkg_offset, x0, x0 + bkg_offset, avg.shape[0]])
        ax.set_yticklabels([])

        ax.grid()
        ax.set_xlim(xr[0] - 5, xr[-1] + 5)
        ax.set_ylim(y0 - 2 * bkg_offset, y0 + 2 * bkg_offset)
        ax.hlines(x0, *ax.get_xlim(), color="k", linestyle="-", alpha=0.1)

    ax.set_xlabel("pixel")
    axes[0].set_title(f"{h['SRCNAME']} {h['GRATING']}-{h['FILTER']}")
    fig.tight_layout(pad=0.5)

    return fig


def show_drizzled_product(hdul, figsize=FIGSIZE, imshow_kws=IMSHOW_KWS):
    """
    Make a figure showing drizzled product

    Parameters
    ----------
    hdul : `~astropy.io.fits.HDUList`
        Drizzle combined HDU

    figsize : tuple
        Figure size

    imshow_kws : dict
        kwargs for `~matplotlib.pyplot.imshow`

    Returns
    -------
    fig : Figure
        Figure object

    """

    avg = hdul["SCI"].data
    xvalid = np.isfinite(avg).sum(axis=0) > 0
    # xr = np.arange(avg.shape[1])[xvalid]
    if xvalid.sum() > 1:
        xr = np.arange(avg.shape[1])[xvalid]
    else:
        xr = np.arange(avg.shape[1])

    bkg = hdul["BKG"].data

    h = hdul["SCI"].header
    bkg_offset = np.abs(h["BKGOFF"])

    x0 = h["SRCYPIX"]
    y0 = (avg.shape[0] - 1) // 2

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)

    axes[0].imshow(avg, **imshow_kws)
    axes[1].imshow(avg - bkg, **imshow_kws)
    axes[2].imshow(bkg, **imshow_kws)

    # Labels
    for i, label in enumerate(["Data", "Cleaned", "Background"]):
        axes[i].text(
            0.02,
            0.02 * figsize[1] / figsize[0] * 3 * 2,
            label,
            ha="left",
            va="bottom",
            transform=axes[i].transAxes,
            bbox={"fc": "w", "alpha": 0.5, "ec": "None"},
            fontsize=8,
        )

    for ax in axes:
        ax.set_yticks([0, x0 - bkg_offset, x0, x0 + bkg_offset, avg.shape[0]])
        ax.set_yticklabels([])
        ax.hlines(x0, *ax.get_xlim(), color="k", linestyle="-", alpha=0.1)

        ax.grid()
        ax.set_xlim(xr[0] - 5, xr[-1] + 5)
        ax.set_ylim(y0 - 2 * bkg_offset, y0 + 2 * bkg_offset)

    ax.set_xlabel("pixel")
    axes[0].set_title(f"{h['SRCNAME']} {h['GRATING']}-{h['FILTER']}")
    fig.tight_layout(pad=0.5)

    return fig


def get_xlimits_from_lines(
    hdul,
    sn_thresh=2,
    max_dy=4,
    n_erode=2,
    n_dilate=4,
    pad=10,
    verbose=True,
):
    """
    Find emission lines in 2D spectrum

    Parameters
    ----------
    hdul : HDUList
        The HDUList object containing the 2D spectrum.

    sn_thresh : float, optional
        The signal-to-noise threshold for identifying emission lines.
        Default is 2.

    max_dy : float, optional
        The maximum deviation in the y-direction from the central pixel.
        Default is 4.

    n_erode : int, optional
        The number of iterations for binary erosion. Default is 2.

    n_dilate : int, optional
        The number of iterations for binary dilation. Default is 4.

    pad : int, optional
        The padding value for the x-limits. Default is 10.

    verbose : bool, optional
        Whether to print verbose output. Default is True.

    Returns
    -------
    xlim : tuple
        A tuple containing the x-limits of the emission lines.

    """
    import scipy.ndimage as nd

    sh = hdul["SCI"].data.shape
    yp, xp = np.indices(sh)

    if "SRCYPIX" in hdul["SCI"].header:
        y0 = hdul["SCI"].header["SRCYPIX"]
    else:
        y0 = sh[0] / 2

    msk = hdul["SCI"].data * np.sqrt(hdul["WHT"].data) > sn_thresh
    msk &= np.abs(yp - y0) < max_dy

    msk_erode = nd.binary_erosion(msk, iterations=n_erode)
    msk_dilate = nd.binary_dilation(msk_erode, iterations=n_dilate)

    if msk_dilate.sum() == 0:
        xlim = (0, sh[1])
    else:
        xpx = xp[msk_dilate]
        xlim = np.clip([xpx.min() - pad, xpx.max() + pad], 0, sh[1]).tolist()

    msg = f"msaexp.drizzle.get_xlimits_from_lines: {msk_dilate.sum()} pixels, "
    msg += f"slice: {xlim}"
    grizli.utils.log_comment(
        grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
    )

    return xlim


def make_optimal_extraction(
    waves,
    sci2d,
    wht2d,
    var2d=None,
    profile_slice=None,
    prf_center=None,
    prf_sigma=1.0,
    sigma_bounds=(0.5, 2.5),
    center_limit=4,
    fix_center=False,
    fix_sigma=False,
    trim=0,
    bkg_offset=6,
    bkg_parity=[-1, 1],
    offset_for_chi2=1.0,
    max_wht_percentile=None,
    max_med_wht_factor=10,
    verbose=True,
    ap_radius=None,
    ap_center=None,
    **kwargs,
):
    """
    Optimal extraction from 2D arrays

    Parameters
    ----------
    waves : 1D array
        Wavelengths, microns

    sci2d : 2D array
        Data array

    wht2d : 2D array
        Inverse variance weight array

    profile_slice : tuple, slice
        Slice along wavelength axis where to determine the cross-dispersion
        profile.  If a tuple of floats, interpret as wavelength limits in
        microns

    prf_center : float
        Profile center, relative to the cross-dispersion center of the array.
        If `None`, then try to estimate it from the data

    prf_sigma : float
        Width of the extraction profile in pixels

    sigma_bounds : (float, float)
        Parameter bounds for `prf_sigma`

    center_limit : float
        Maximum offset from `prf_center` allowed

    fix_center : bool
        Fix the centering in the fit

    fix_sigma : bool
        Fix the width in the fit

    trim : int
        Number of pixels to trim from the edges of the extracted spectrum

    bkg_offset, bkg_parity : int, list
        Parameters for the local background determination (see
        `~msaexp.drizzle.drizzle_slitlets`).  The profile is "subtracted" in
        the same way as the data.

    offset_for_chi2 : float
        If specified, compute chi2 of the profile fit offseting the first
        parameter by +/- this value

    max_wht_percentile : float
        Maximum percentile of WHT to consider valid

    max_med_wht_factor : float
        Maximum weight value relative to the median nonzero weight to consider
        valid

    verbose : bool
        Status messages

    ap_center, ap_radius : int, int
        Center and radius of fixed-width aperture extraction, in pixels.  If
        not specified, then

        .. code-block:: python
            :dedent:

            >>> ap_center = int(np.round(ytrace + fit_center))
            >>> ap_radius = np.clip(int(np.round(prof_sigma*2.35/2)), 1, 3)

    kwargs : dict
        Ignored keyword args

    Returns
    -------
    sci2d_out : array
        Output 2D sci array

    wht2d_out : array
        Output 2D wht array

    profile2d : array
        2D optimal extraction profile

    spec : `~astropy.table.Table`
        Optimally-extracted 1D spectrum

    prof_tab : `~astropy.table.Table`
        Table of the collapsed 1D profile
    """
    import scipy.ndimage as nd
    import astropy.units as u
    from scipy.optimize import least_squares

    from .version import __version__ as msaexp_version

    sh = wht2d.shape
    yp, xp = np.indices(sh)

    ok = np.isfinite(sci2d * wht2d) & (wht2d > 0)
    if max_wht_percentile is not None:
        wperc = np.percentile(wht2d[ok], max_wht_percentile)
        ok &= wht2d < wperc

    if max_med_wht_factor is not None:
        med_wht = np.nanmedian(wht2d[ok])
        ok &= wht2d < max_med_wht_factor * med_wht

    if var2d is None:
        wht_mask = wht2d * 1
    else:
        wht_mask = 1.0 / var2d

    wht_mask[~ok] = 0.0

    if profile_slice is not None:
        if not isinstance(profile_slice, slice):
            if isinstance(profile_slice[0], int):
                # pixels
                profile_slice = slice(*profile_slice)
            else:
                # Wavelengths interpolated on pixel grid
                xpix = np.arange(sh[1])
                xsl = np.round(np.interp(profile_slice, waves, xpix)).astype(int)
                xsl = np.clip(xsl, 0, sh[1])

                print(f"Wavelength slice: {profile_slice} > {xsl} pix")
                profile_slice = slice(*xsl)

        prof1d = np.nansum((sci2d * wht_mask)[:, profile_slice], axis=1)
        prof1d /= np.nansum(wht_mask[:, profile_slice], axis=1)

        slice_limits = profile_slice.start, profile_slice.stop

        pmask = ok & True
        pmask[:, profile_slice] &= True
        ok &= pmask

    else:
        prof1d = np.nansum(sci2d * wht_mask, axis=1) / np.nansum(
            wht_mask, axis=1
        )
        slice_limits = 0, sh[1]

    xpix = np.arange(sh[0])
    ytrace = (sh[0] - 1) / 2.0
    x0 = np.arange(sh[0]) - ytrace

    if prf_center is None:
        prf_center = np.nanargmax(prof1d) - (sh[0] - 1) / 2.0

        if verbose:
            print(f"Set prf_center: {prf_center} {sh} {ok.sum()}")

    msg = "msaexp.drizzle.extract_from_hdul: Initial center = "
    msg += f" {prf_center:6.2f}, sigma = {prf_sigma:6.2f}"
    grizli.utils.log_comment(
        grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
    )

    #############
    # Integrated gaussian profile
    fit_type = 3 - 2 * fix_center - 1 * fix_sigma

    wht_mask[~ok] = 0.0

    p00_name = None

    if fit_type == 0:
        args = (
            waves,
            sci2d,
            wht_mask,
            prf_center,
            prf_sigma,
            bkg_offset,
            bkg_parity,
            3,
            1,
            (verbose > 1),
        )

        pnorm, pmodel = utils.objfun_prf([prf_center, prf_sigma], *args)
        profile2d = pmodel / pnorm
        pmask = (profile2d > 0) & np.isfinite(profile2d)
        profile2d[~pmask] = 0

        fit_center = prf_center
        fit_sigma = prf_sigma

    else:
        # Fit it
        if fix_sigma:
            p00_name = "center"
            p0 = [prf_center]
            bounds = (-center_limit, center_limit)
        elif fix_center:
            p00_name = "sigma"
            p0 = [prf_sigma]
            bounds = sigma_bounds
        else:
            p00_name = "center"
            p0 = [prf_center, prf_sigma]
            bounds = (
                (-center_limit + prf_center, sigma_bounds[0]),
                (center_limit + prf_center, sigma_bounds[1]),
            )

        args = (
            waves,
            sci2d,
            wht_mask,
            prf_center,
            prf_sigma,
            bkg_offset,
            bkg_parity,
            fit_type,
            1,
            (verbose > 1),
        )
        lmargs = (
            waves,
            sci2d,
            wht_mask,
            prf_center,
            prf_sigma,
            bkg_offset,
            bkg_parity,
            fit_type,
            2,
            (verbose > 1),
        )

        _res = least_squares(
            utils.objfun_prf,
            p0,
            args=lmargs,
            method="trf",
            bounds=bounds,
            loss="huber",
        )

        # dchi2 / dp0
        if offset_for_chi2 is not None:
            chiargs = (
                waves,
                sci2d,
                wht_mask,
                prf_center,
                prf_sigma,
                bkg_offset,
                bkg_parity,
                fit_type,
                3,
                (verbose > 1),
            )
            delta = _res.x * 0.0
            dchi2dp = []
            for d in [-offset_for_chi2, 0.0, offset_for_chi2]:
                delta[0] = d
                dchi2dp.append(utils.objfun_prf(_res.x + delta, *chiargs))

            msg = f"msaexp.drizzle.extract_from_hdul: dchi2/d{p00_name} = "
            dchi = dchi2dp[0] - dchi2dp[1]
            dchi += dchi2dp[2] - dchi2dp[1]
            msg += f"{dchi/2.:.1f}"
            grizli.utils.log_comment(
                grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
            )

        else:
            dchi2dp = None

        pnorm, pmodel = utils.objfun_prf(_res.x, *args)
        profile2d = pmodel / pnorm
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

    wht1d = np.nansum(wht_mask * profile2d**2, axis=0)
    sci1d = np.nansum(sci2d * wht_mask * profile2d, axis=0) / wht1d

    if profile_slice is not None:
        pfit1d = np.nansum(
            (wht_mask * profile2d * sci1d)[:, profile_slice], axis=1
        )
        pfit1d /= np.nansum((wht_mask)[:, profile_slice], axis=1)
    else:
        pfit1d = np.nansum(profile2d * sci1d * wht_mask, axis=1)
        pfit1d /= np.nansum(wht_mask, axis=1)

    if trim > 0:
        bad = nd.binary_dilation(wht1d <= 0, iterations=trim)
        wht1d[bad] = 0

    sci1d[wht1d <= 0] = 0
    err1d = np.sqrt(1 / wht1d)
    err1d[wht1d <= 0] = 0

    #######
    # Make tables

    # Flux conversion
    to_ujy = 1.0

    spec = grizli.utils.GTable()
    spec.meta["VERSION"] = msaexp_version, "msaexp software version"

    spec.meta["TOMUJY"] = to_ujy, "Conversion from pixel values to microJansky"
    spec.meta["PROFCEN"] = fit_center, "PRF profile center"
    spec.meta["PROFSIG"] = fit_sigma, "PRF profile sigma"
    spec.meta["PROFSTRT"] = slice_limits[0], "Start of profile slice"
    spec.meta["PROFSTOP"] = slice_limits[1], "End of profile slice"
    spec.meta["YTRACE"] = ytrace, "Expected center of trace"

    spec.meta["MAXWPERC"] = max_wht_percentile, "Maximum weight percentile"
    spec.meta["MAXWFACT"] = max_med_wht_factor, "Maximum weight factor"

    prof_tab = grizli.utils.GTable()
    prof_tab.meta["VERSION"] = msaexp_version, "msaexp software version"

    prof_tab["pix"] = x0
    prof_tab["profile"] = prof1d
    prof_tab["pfit"] = pfit1d
    prof_tab.meta["PROFCEN"] = fit_center, "PRF profile center"
    prof_tab.meta["PROFSIG"] = fit_sigma, "PRF profile sigma"
    prof_tab.meta["PROFSTRT"] = slice_limits[0], "Start of profile slice"
    prof_tab.meta["PROFSTOP"] = slice_limits[1], "End of profile slice"
    prof_tab.meta["YTRACE"] = ytrace, "Expected center of trace"

    if (dchi2dp is not None) & (p00_name is not None):
        prof_tab.meta["DCHI2PAR"] = (p00_name, "Parameter for dchi2/dparam")
        prof_tab.meta["CHI2A"] = (
            dchi2dp[0],
            "Chi2 with d{p00_name} = -{offset_for_chi2}",
        )
        prof_tab.meta["CHI2B"] = dchi2dp[1], "Chi2 with dparam = 0. (best fit)"
        prof_tab.meta["CHI2C"] = (
            dchi2dp[2],
            "Chi2 with d{p00_name} = +{offset_for_chi2}",
        )

    spec["wave"] = waves
    spec["wave"].unit = u.micron
    spec["flux"] = sci1d * to_ujy
    spec["err"] = err1d * to_ujy
    spec["flux"].unit = u.microJansky
    spec["err"].unit = u.microJansky

    # Aperture extraction
    if ap_center is None:
        ap_center = int(np.round(ytrace + fit_center))
    elif ap_center < 0:
        ap_center = np.nanargmax(prof1d)

    if ap_radius is None:
        ap_radius = np.clip(int(np.round(fit_sigma * 2.35 / 2)), 1, 3)

    sly = slice(ap_center - ap_radius, ap_center + ap_radius + 1)

    aper_sci = np.nansum(sci2d[sly, :], axis=0)
    aper_var = np.nansum(1.0 / wht_mask[sly, :], axis=0)
    aper_corr = np.nansum(profile2d, axis=0) / np.nansum(
        profile2d[sly, :], axis=0
    )
    spec["aper_flux"] = aper_sci * to_ujy
    spec["aper_err"] = np.sqrt(aper_var) * to_ujy
    spec["aper_corr"] = aper_corr
    spec["aper_flux"].unit = u.microJansky
    spec["aper_err"].unit = u.microJansky

    spec["aper_flux"].description = (
        f"Flux in trace aperture ({ap_center}, {ap_radius})"
    )
    spec["aper_err"].description = (
        "Flux uncertainty in trace aperture " f"({ap_center}, {ap_radius})"
    )

    spec.meta["APER_Y0"] = (ap_center, "Fixed aperture center")
    spec.meta["APER_DY"] = (ap_radius, "Fixed aperture radius, pix")

    msg = "msaexp.drizzle.extract_from_hdul: aperture extraction = "
    msg += f"({ap_center}, {ap_radius})"
    grizli.utils.log_comment(
        grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
    )

    msk = np.isfinite(sci2d + wht_mask)
    sci2d[~msk] = 0
    wht_mask[~msk] = 0

    return sci2d * to_ujy, wht2d / to_ujy**2, profile2d, spec, prof_tab


def extract_from_hdul(
    hdul,
    prf_center=None,
    master_bkg=None,
    verbose=True,
    line_limit_kwargs={},
    **kwargs,
):
    """
    Run 1D extraction on arrays from a combined dataset

    Parameters
    ----------
    hdul : `~astropy.io.fits.HDUList`
        Output data from from `~msaexp.drizzle.drizzle_slitlets`

    prf_center : float, None
        Initial profile center.  If not specified, get from the `'SRCYPIX'`
        keyword in `hdul['SCI'].header`

    master_bkg : array
        Optional master background to use instead of `hdul['BKG'].data`

    verbose : bool
        Printing status messages

    line_limit_kwargs : dict
        Keyword arguments passed to `msaexp.drizzle.get_xlimits_from_lines`

    kwargs : dict
        Keyword arguments passed to `msaexp.drizzle.make_optimal_extraction`

    Returns
    -------
    outhdu : `~astropy.io.fits.HDUList`
        Modified HDU including 1D extraction

    """

    if master_bkg is None:
        if "BKG" in hdul:
            bkg_i = hdul["BKG"].data
        else:
            bkg_i = None
    else:
        bkg_i = master_bkg

    sci = hdul["SCI"]
    sci2d = sci.data * 1
    if bkg_i is not None:
        sci2d -= bkg_i

    wht2d = hdul["WHT"].data * 1

    if "WAVE" in hdul:
        waves = hdul["WAVE"].data
    elif "SPEC1D" in hdul:
        tab = grizli.utils.read_catalog(hdul["SPEC1D"])
        waves = tab["wave"].data
    else:
        _gr = sci.header["GRATING"].lower()
        waves = utils.get_standard_wavelength_grid(
            _gr, sample=sci.header["WSAMPLE"], log_step=sci.header["LOGWAVE"]
        )

    if line_limit_kwargs:
        kwargs["profile_slice"] = get_xlimits_from_lines(
            hdul, **line_limit_kwargs
        )

    if prf_center is None:
        if "SRCYPIX" in sci.header:
            y0 = sci.header["SRCYPIX"]
        else:
            y0 = (sci.data.shape[0] - 1) / 2

        prf_center = y0 - (sci.data.shape[0] - 1) / 2.0

    _data = make_optimal_extraction(
        waves,
        sci2d,
        wht2d,
        prf_center=prf_center,
        verbose=verbose,
        **kwargs,
    )

    _sci2d, _wht2d, profile2d, spec, prof = _data

    hdul = pyfits.HDUList()
    hdul.append(pyfits.BinTableHDU(data=spec, name="SPEC1D"))

    header = sci.header

    for k in spec.meta:
        header[k] = spec.meta[k]

    msg = "msaexp.drizzle.extract_from_hdul:  Output center = "
    msg += f" {header['PROFCEN']:6.2f}, sigma = {header['PROFSIG']:6.2f}"
    grizli.utils.log_comment(
        grizli.utils.LOGFILE, msg, verbose=verbose, show_date=False
    )

    hdul.append(pyfits.ImageHDU(data=sci2d, header=header, name="SCI"))
    hdul.append(pyfits.ImageHDU(data=wht2d, header=header, name="WHT"))
    hdul.append(pyfits.ImageHDU(data=profile2d, header=header, name="PROFILE"))

    hdul.append(pyfits.BinTableHDU(data=prof, name="PROF1D"))

    for k in hdul["SCI"].header:
        if k not in hdul["SPEC1D"].header:
            hdul["SPEC1D"].header[k] = hdul["SCI"].header[k]

    return hdul
