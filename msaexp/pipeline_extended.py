"""
Extend CRDS reference files for fixed-slit observations
"""

import os
import time

import numpy as np
import astropy.io.fits as pyfits
import astropy.table
import matplotlib.pyplot as plt

import jwst.datamodels
from jwst.assign_wcs import AssignWcsStep
from jwst.msaflagopen import MSAFlagOpenStep
from jwst.extract_2d import Extract2dStep
from jwst.flatfield import FlatFieldStep
from jwst.pathloss import PathLossStep
from jwst.barshadow import BarShadowStep
from jwst.photom import PhotomStep
from jwst.assign_wcs.util import NoDataOnDetectorError

from jwst.assign_wcs.assign_wcs import load_wcs
from jwst.flatfield import flat_field
import jwst.photom.photom

from grizli import utils
import msaexp.utils as msautils
from .msa import slit_best_source_alias
from . import pipeline
from .fork.assign_wcs.nirspec import zeroth_order_mask

# NEW_WAVERANGE_REF = "jwst_nirspec_wavelengthrange_0008_ext.asdf"

EXTENDED_RANGES = {
    "F070LP_G140M": [0.6, 3.3],
    "F100LP_G140M": [0.9, 3.3],
    "F170LP_G235M": [1.5, 5.3],
    "F290LP_G395M": [2.6, 5.6],
    "F070LP_G140H": [0.6, 3.3],
    "F100LP_G140H": [0.9, 3.3],
    "F170LP_G235H": [1.5, 5.3],
    "F290LP_G395H": [2.6, 5.6],
    "CLEAR_PRISM": [0.5, 5.6],
}

__all__ = [
    "step_reference_files",
    "extend_wavelengthrange",
    "extend_fs_fflat",
    "extend_quad_fflat",
    "extend_sflat",
    "extend_dflat",
    "extend_photom",
    "run_pipeline",
]


def step_reference_files(step, input_model):
    """
    Get reference filenames for a `jwst` pipeline step

    Parameters
    ----------
    step : `jwst` pipeline step instance
        Step object, e.g., `jwst.assign_wcs.AssignWcsStep()`

    input_model : `jwsts.datamodels.Model`
        Data model instance

    Returns
    -------
    reference_file_names : dict
        List of reference filenames by type

    """
    reference_file_names = {}
    for reftype in step.reference_file_types:
        reffile = step.get_reference_file(input_model, reftype)
        reference_file_names[reftype] = reffile if reffile else ""

    return reference_file_names


VERBOSITY = True


def extend_wavelengthrange(
    ref_file="jwst_nirspec_wavelengthrange_0008.asdf", ranges=EXTENDED_RANGES
):
    """
    Extend limits of ``wavelengthrange`` reference file

    Parameters
    ----------
    ref_file : str
        Filename of ``wavelengthrange`` asdf reference file.  If not provided as an
        absolute path starting with ``/``, will look for the file in ``./`` and
        ``$CRDS_PATH/references/jwst/nirspec``.

    ranges : dict
        Extended ranges by grating / filter

    Returns
    -------
    status : bool
        True if completed without exception

    Writes a file ``ref_file.replace(".asdf", "_ext.asdf")``.

    """

    if os.path.exists(ref_file):
        NIRSPEC_CRDS = "./"
    elif ref_file.startswith("/"):
        NIRSPEC_CRDS = os.path.dirname(ref_file)
    else:
        NIRSPEC_CRDS = os.path.join(
            os.getenv("CRDS_PATH"), "references/jwst/nirspec"
        )

    waverange = jwst.datamodels.open(os.path.join(NIRSPEC_CRDS, ref_file))

    for k in ranges:
        i = waverange.waverange_selector.index(k)
        waverange.wavelengthrange[i] = [v * 1e-6 for v in ranges[k]]

    new_waverange = ref_file.replace(".asdf", "_ext.asdf")

    waverange.meta.author = "G. Brammer"
    waverange.meta.date = utils.nowtime().iso.replace(" ", "T")
    waverange.meta.description = "Extended M grating wavelength ranges"

    so = np.argsort(waverange.waverange_selector)

    msg = f"New wavelength range file {new_waverange}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    for i in so:
        key = waverange.waverange_selector[i]
        spl = key.split("_")
        if (
            spl[0]
            in [
                "CLEAR",
                "FLAT1",
                "FLAT2",
                "FLAT3",
                "LINE1",
                "LINE2",
                "LINE3",
                "REF",
                "TEST",
            ]
        ) & (key != "CLEAR_PRISM"):
            continue
        if (spl[1] == "MIRROR") | (key == "F100LP_PRISM"):
            continue
        if spl[0] in ["OPAQUE", "F110W", "F140X"]:
            continue
        if (spl[1] == "PRISM") & (spl[0] != "CLEAR"):
            continue

        msg = f"{i:>2} {waverange.waverange_selector[i]:>16} {waverange.order[i]:>3}   {waverange.wavelengthrange[i][0]*1.e6:.2f}  -  {waverange.wavelengthrange[i][1]*1.e6:.2f}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    waverange.write(new_waverange)
    waverange.close()

    return True


def extend_fs_fflat(
    fflat_filename,
    full_range=[0.55, 5.6],
    FILL_VALUE=2.0e20,
    log_prefix="extend_fs_fflat",
):
    """
    Extend fflat reference for ``EXP_TYPE = NRS_FIXEDSLIT``

    Parameters
    ----------
    fflat_filename : str
        Filename

    full_range : list
        Wavelength limits

    FILL_VALUE : float
        Value to use for the monochromatic flat table.  The default is set such that
        the products processed with the updated references files have roughly the same
        pixel values as with the default calibrations.

    log_prefix : str
        String for log messages

    Returns
    -------
    ff : ``jwst.datamodels.NirspecFlat``
        F-Flat object
    """
    fbase = os.path.basename(fflat_filename)

    ff = jwst.datamodels.open(fflat_filename)

    fftab = astropy.table.Table(ff.flat_table)
    nelem = len(fftab[0]["wavelength"])

    non_zero = fftab["wavelength"][0] > 0

    newtab = astropy.table.Table()
    newtab["slit_name"] = fftab["slit_name"]

    med_FILL_VALUE = np.nanmedian(fftab["data"][0][non_zero])

    msg = f"{log_prefix} {fbase} fill fflat with {FILL_VALUE} (med = {med_FILL_VALUE})"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    wave = np.logspace(*(np.log10(full_range)), nelem)
    newtab["nelem"] = nelem
    newtab["wavelength"] = [wave.astype(np.float32).tolist()] * len(fftab)
    newtab["data"] = [
        np.full(nelem, FILL_VALUE, dtype=np.float32).tolist()
    ] * len(fftab)
    newtab["error"] = [
        np.full(nelem, FILL_VALUE * 1.0e-5, dtype=np.float32).tolist()
    ] * len(fftab)
    for c in ["wavelength", "data", "error"]:
        newtab[c] = newtab[c].astype(np.float32)

    ff.flat_table = pyfits.BinTableHDU(newtab).data
    for c in ["wavelength", "data", "error"]:
        newtab[c] = newtab[c].astype(np.float32)

    return ff


def extend_quad_fflat(
    fflat_filename,
    full_range=[0.55, 5.6],
    FILL_VALUE=2.0e20,
    log_prefix="extend_quad_fflat",
):
    """
    Extend fflat reference for ``EXP_TYPE = NRS_MSASPEC``

    Parameters
    ----------
    fflat_filename : str
        Filename

    full_range : list
        Wavelength limits

    FILL_VALUE : float
        Value to use for the monochromatic flat table.  The default is set such that
        the products processed with the updated references files have roughly the same
        pixel values as with the default calibrations.

    log_prefix : str
        String for log messages

    Returns
    -------
    ff : ``jwst.datamodels.NirspecFlat``
        F-Flat object

    """
    fbase = os.path.basename(fflat_filename)

    ff = jwst.datamodels.open(fflat_filename)

    for i in range(4):
        fftab = astropy.table.Table(ff.quadrants[i].flat_table)
        non_zero = fftab["wavelength"][0] > 0

        newtab = astropy.table.Table()
        newtab["slit_name"] = fftab["slit_name"]

        med_FILL_VALUE = np.nanmedian(fftab["data"][0][non_zero])

        msg = f"{log_prefix} {fbase} Q={i} fill fflat with {FILL_VALUE} (med = {med_FILL_VALUE})"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        nelem = len(fftab[0]["wavelength"])

        wave = np.logspace(*(np.log10(full_range)), nelem)
        newtab["nelem"] = nelem
        newtab["wavelength"] = [wave.astype(np.float32).tolist()] * len(fftab)
        newtab["data"] = [
            np.full(nelem, FILL_VALUE, dtype=np.float32).tolist()
        ] * len(fftab)
        newtab["error"] = [
            np.full(nelem, FILL_VALUE * 1.0e-5, dtype=np.float32).tolist()
        ] * len(fftab)

        ff.quadrants[i].flat_table = pyfits.BinTableHDU(newtab).data
        for c in ["wavelength", "data", "error"]:
            newtab[c] = newtab[c].astype(np.float32)

    return ff


def extend_sflat(
    sflat_filename,
    full_range=[0.55, 5.6],
    FILL_VALUE=5.0e-10,
    log_prefix="extend_sflat",
):
    """
    Extend NIRSpec sflat reference file

    Parameters
    ----------
    fflat_filename : str
        Filename

    full_range : list
        Wavelength limits

    FILL_VALUE : float
        Value to use for the monochromatic flat table.  The default is set such that
        the products processed with the updated references files have roughly the same
        pixel values as with the default calibrations.

    log_prefix : str
        String for log messages

    Returns
    -------
    sf : ``jwst.datamodels.NirspecFlat``
        S-Flat object

    """
    fbase = os.path.basename(sflat_filename)

    sf = jwst.datamodels.open(sflat_filename)

    msg = f"{log_prefix} {fbase} Unset NO_FLAT_FIELD from SFlat DQ"
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    sf.dq -= sf.dq & 2**18
    if sf.meta.exposure.type == "NRS_IFU":
        msg = f"{log_prefix} {fbase} Unset 1 from IFU SFlat DQ"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        sf.dq -= sf.dq & 1

    # Reset all
    msg = f"{log_prefix} {fbase} Set unity SFlat SCI"
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    sf.data = np.ones_like(sf.data)
    sf.err = np.ones_like(sf.data) * 1.0e-3

    # Just reset wavelength metadata, don't add extensions
    head_ = sf.extra_fits.SCI.header
    nflat = sf.data.shape[0]

    inds = []

    for i, row in enumerate(head_):
        if row[0] == "FLAT_01":
            head_[i][1] = full_range[0]
            inds.append(i)

        elif row[0] == f"FLAT_{nflat:02d}":
            head_[i][1] = full_range[-1]
            inds.append(i)

    for i in inds:
        utils.log_comment(
            utils.LOGFILE,
            f"{log_prefix} {fbase} header {sf.extra_fits.SCI.header[i]}",
            verbose=True,
        )

    if len(sf.wavelength) > 0:
        sf.wavelength[0] = (full_range[0],)
        sf.wavelength[-1] = (full_range[-1],)

    # Make flat table
    sft = astropy.table.Table(sf.flat_table)
    non_zero = sft["wavelength"][0] > 0

    newtab = astropy.table.Table()
    newtab["slit_name"] = sft["slit_name"]

    nelem = len(sft[0]["wavelength"])
    if nelem > 50000:
        nelem = 1024
        head_ = sf.extra_fits.FAST_VARIATION.header
        for i in range(len(head_)):
            head_[i][1] = f"({nelem})"  # wierd format
            # print(head_[i])

    wave = np.logspace(*(np.log10(full_range)), nelem)

    med_FILL_VALUE = np.nanmax(sft["data"][0][non_zero])

    msg = f"{log_prefix} {fbase} fill sflat with {FILL_VALUE} (med = {med_FILL_VALUE})"
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    newtab["nelem"] = nelem
    newtab["wavelength"] = [wave.astype(np.float32).tolist()] * len(sft)
    newtab["data"] = [
        np.full(nelem, FILL_VALUE, dtype=np.float32).tolist()
    ] * len(sft)
    newtab["error"] = [
        np.full(nelem, FILL_VALUE * 1.0e-5, dtype=np.float32).tolist()
    ] * len(sft)

    for c in ["wavelength", "data", "error"]:
        newtab[c] = newtab[c].astype(np.float32)

    sf.flat_table = pyfits.BinTableHDU(newtab).data

    return sf


def extend_dflat(
    dflat_filename,
    full_range=[0.55, 5.6],
    nelem=1024,
    log_prefix="extend_dflat",
):
    """
    Extend NIRSpec sflat reference file

    Parameters
    ----------
    dflat_filename : str
        Filename

    full_range : list
        Wavelength limits

    nelem : int
        Number of elements in the flattened "fast_variation" table

    log_prefix : str
        String for log messages

    Returns
    -------
    df : ``jwst.datamodels.NirspecFlat``
        D-Flat object

    """
    fbase = os.path.basename(dflat_filename)

    df = jwst.datamodels.open(dflat_filename)

    dflat_waves = df.extra_fits.WAVELENGTH.data.wavelength.tolist()

    prism_min = full_range[0]

    prism_max = full_range[1]

    ####################
    # Maximum wavelength
    if dflat_waves[-1] < prism_max:

        msg = f"{log_prefix} {fbase} max wave = {dflat_waves[-1]:.2f} ->  append {prism_max}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        dflat_waves.append(prism_max)

        astropy.table.Table(df.extra_fits.WAVELENGTH.data)

        newtab = astropy.table.Table()
        newtab["wavelength"] = dflat_waves

        df.extra_fits.WAVELENGTH.data = pyfits.BinTableHDU(newtab).data

        df.extra_fits.SCI.header[0][1] += 1
        df.extra_fits.SCI.header.append(
            [
                f"PFLAT_{df.extra_fits.SCI.header[0][1]}",
                prism_max,
                "central wavelength monochromatic pflat (micron)",
            ]
        )
        df.data = np.vstack([df.data, df.data[-1:, :, :] * 1])

    ####################
    # Minimum wavelength
    dflat_waves = df.extra_fits.WAVELENGTH.data.wavelength.tolist()

    if dflat_waves[0] > prism_min:

        msg = f"{log_prefix} {fbase} min wave = {dflat_waves[0]:.2f} -> prepend {prism_min}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        dflat_waves.insert(0, prism_min)

        astropy.table.Table(df.extra_fits.WAVELENGTH.data)

        newtab = astropy.table.Table()
        newtab["wavelength"] = dflat_waves

        df.extra_fits.WAVELENGTH.data = pyfits.BinTableHDU(newtab).data

        df.extra_fits.SCI.header[0][1] += 1
        df.extra_fits.SCI.header.insert(
            1,
            [
                f"PFLAT_1",
                prism_min,
                "central wavelength monochromatic pflat (micron)",
            ],
        )
        N = df.extra_fits.SCI.header[0][1]
        for i in range(N):
            df.extra_fits.SCI.header[i + 1][0] = f"PFLAT_{i+1}"

        df.data = np.vstack([df.data[:1, :, :] * 1, df.data])

    ##############################
    # Flatten FAST_VARIATION table
    in_fast = astropy.table.Table(df.extra_fits.FAST_VARIATION.data)
    wgrid = np.logspace(*np.log10(full_range), nelem)
    resp = (wgrid > 0) * 1

    fast_1d = astropy.table.Table()
    fast_1d["slit_name"] = in_fast["slit_name"]
    fast_1d["nelem"] = nelem
    fast_1d["wavelength"] = [wgrid.astype(in_fast["wavelength"].dtype)] * len(
        in_fast
    )
    fast_1d["data"] = [resp.astype(in_fast["data"].dtype)] * len(in_fast)

    df.extra_fits.FAST_VARIATION.data = pyfits.BinTableHDU(fast_1d).data

    return df


def extend_photom(phot_filename, ranges=EXTENDED_RANGES):
    """
    Extend wavelength ranges in photom reference file

    Parameters
    ----------
    phot_filename : str
        Filename of a ``photom`` reference file

    ranges : dict:
        Wavelength ranges by grating, filter

    Returns
    -------
    pref : `stdatamodels.jwst.datamodels.photom.NrsMosPhotomModel`
        Updated reference file object

    """
    pref = jwst.datamodels.open(phot_filename)
    ptab = astropy.table.Table(pref.phot_table)

    for k in ranges:
        filt, grat = k.split("_")
        nfull = len(ptab[0]["wavelength"])
        nelem = nfull - 4
        wgrid = np.zeros(nfull, dtype=np.float32)
        wgrid[:nelem] = np.logspace(*np.log10(ranges[k]), nelem)
        resp = (wgrid > 0) * 1

        rows = (ptab["grating"] == grat) & (ptab["filter"] == filt)

        ptab["nelem"][rows] = nelem
        ptab["wavelength"][rows] = wgrid
        ptab["relresponse"][rows] = resp

    pref.phot_table = pyfits.BinTableHDU(ptab).data

    return pref


def extend_pathloss(
    path_filename,
    full_range=[0.55, 5.6],
):
    """
    Extend wavelength ranges in pathloss reference file.  This function just resets the
    first pixel of the reference file to the minimum of the desired wavelength range.

    Parameters
    ----------
    path_filename : str
        Filename of a ``pathloss`` reference file

    full_range : list
        Minimum and maximum wavelengths, microns.  Assumes long wavelength side doesn't
        need to be extended.

    Returns
    -------
    hdu : `astropy.io.fits.HDUList`
        Updated HDUlist of

    """
    hdu = pyfits.open(path_filename)
    for ext in hdu:
        h = hdu[ext].header
        if "CRVAL1" not in h:
            continue

        if h["NAXIS"] == 3:
            crval = "CRVAL3"
            cdelt = "CDELT3"
            naxis = "NAXIS3"
        else:
            crval = "CRVAL1"
            cdelt = "CDELT1"
            naxis = "NAXIS1"

        to_meters = 1.0e-6 if h[crval] < 1.0e-5 else 1.0
        h[crval] = full_range[0] * to_meters

        max_wavelength = h[crval] + h[naxis] * h[cdelt]
        if max_wavelength / to_meters < full_range[1]:
            h[cdelt] = (full_range[1] * to_meters - h[crval]) / h[naxis]

    return hdu


def extend_barshadow(
    bar_filename,
    full_range=[0.55, 5.6],
):
    """
    Extend wavelength ranges in barshadow reference file.  This function just resets the
    first pixel of the reference file to the minimum of the desired wavelength range.
    It also updates the wavelength step if necessary to cover the full desired range.

    Parameters
    ----------
    path_filename : str
        Filename of a ``barshadow`` reference file

    full_range : list
        Minimum and maximum wavelengths, microns.  Assumes long wavelength side doesn't
        need to be extended.

    Returns
    -------
    hdu : `astropy.io.fits.HDUList`
        Updated HDUlist of

    """
    hdu = pyfits.open(bar_filename)

    msg = f"extend_barshadow: full_range = {full_range[0]:.2f} {full_range[1]:.2f}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    for ext in hdu:
        h = hdu[ext].header
        if "CRVAL1" not in h:
            continue

        key = "CRVAL1"

        to_meters = 1.0e-6 if h[key] < 1.0e-5 else 1.0
        msg = f"extend_barshadow: {os.path.basename(bar_filename)}[{h['EXTNAME']}] "
        msg += (
            f"set minimum wave  {h[key]/to_meters:.2f} -> {full_range[0]:.2f}"
        )
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        h[key] = full_range[0] * to_meters

        wmax = h[key] + h["CDELT1"] * h["NAXIS1"]

        if wmax / to_meters < full_range[1]:
            old_cd1 = h["CDELT1"]
            new_cd1 = (full_range[1] * to_meters - h[key]) / h["NAXIS1"]

            msg = f"extend_barshadow: {os.path.basename(bar_filename)}[{h['EXTNAME']}] "
            msg += f"set cdelt1  {old_cd1/to_meters:.4f} -> {new_cd1/to_meters:.4f}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

            h["CDELT1"] = new_cd1

    return hdu


def assign_wcs_with_extended(dm, file=None, ranges=EXTENDED_RANGES, **kwargs):
    """ """
    import jwst.assign_wcs.nirspec
    import jwst.assign_wcs.assign_wcs_step

    # Fork of load_wcs to ignore limits on IFU detectors
    from .fork.assign_wcs.assign_wcs import load_wcs as load_wcs_fork

    jwst.assign_wcs.load_wcs = load_wcs_fork
    jwst.assign_wcs.assign_wcs_step.load_wcs = load_wcs_fork

    from jwst.assign_wcs import AssignWcsStep

    ORIG_LOGFILE = utils.LOGFILE

    wstep = AssignWcsStep()

    ##############
    # AssignWCS with extended wavelength range
    wcs_reference_files = step_reference_files(wstep, dm)
    new_waverange = os.path.basename(wcs_reference_files["wavelengthrange"])
    new_waverange = new_waverange.replace(".asdf", "_ext.asdf")

    msg = f"msaexp.pipeline_extended.assign_wcs_with_extended {new_waverange}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    if not os.path.exists(new_waverange):
        extend_wavelengthrange(
            ref_file=os.path.basename(wcs_reference_files["wavelengthrange"]),
            ranges=ranges,
        )

    # Use new reference
    wcs_reference_files["wavelengthrange"] = new_waverange
    # slit_y_range = [wstep.slit_y_low, wstep.slit_y_high]
    try:
        # with_wcs = load_wcs(dm, wcs_reference_files, slit_y_range)
        with_wcs = wstep.call(dm, override_wavelengthrange=new_waverange)
    except NoDataOnDetectorError:
        msg = f"{file} No open slits found to work on"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
        utils.LOGFILE = ORIG_LOGFILE
        return None

    return with_wcs


def run_pipeline(
    file,
    slit_index=0,
    all_slits=True,
    write_output=True,
    set_log=True,
    skip_existing_log=False,
    undo_flat=True,
    preprocess_kwargs={},
    ranges=EXTENDED_RANGES,
    make_trace_figures=False,
    run_pathloss=True,
    run_barshadow=True,
    mask_zeroth_kwargs={},
    **kwargs,
):
    """
    Pipeline for extending reference files

    Parameters
    ----------
    file : str
        Exposure (rate.fits) filename

    slit_index : int
        Index of a single slit to extract

    all_slits : bool
        Extract all slits of the final `jwst.datamodles.MultiSlit` model

    write_output : bool
        Write output extracted `jwst.datamodels.SlitModel` objects to separate files

    set_log : bool
        Set log file based on the input ``file``

    skip_existing_log : bool
        Skip subsequent processing if the log file from ``set_log`` is found in the
        working directory.

    undo_flat : bool
        Main switch for using dummy monochromatic flat reference files.  This needs
        to be True for the extracted spectra to not be masked with NaN beyond where the
        current reference files are defined

    preprocess_kwargs : dict
        Keyword arguments for `msaexp.pipeline.exposure_detector_effects`
        preprocessing.  Skip if ``None``

    ranges : dict
        Full extended wavelength ranges by FILTER / GRATING

    make_trace_figures : bool
        Make some diagnostic figures

    run_pathloss : bool
        Run PathLoss step for MSA exposures

    run_barshadow : bool
        Run BarShadow step for MSA exposures

    Returns
    -------
    result : None, `jwst.datamodels.MultiSlitModel`
        None if an existing log was found, otherwise the final calibrated product.
        Also returns ``None`` if `jwst.assign_wcs.AssignWcsStep` raises a
        ``NoDataOnDetectorError`` exception.

    """

    ORIG_LOGFILE = utils.LOGFILE
    if set_log:
        utils.LOGFILE = file.replace("_rate.fits", "_rate.wave_log.txt")

        if os.path.exists(utils.LOGFILE) & skip_existing_log:
            utils.LOGFILE = ORIG_LOGFILE
            print(f"log file {utils.LOGFILE} found, skip")
            return None

    msg = f"""
########
# extend_reference_files {time.ctime()}
# {file}
# slit_index={slit_index} all_slits={all_slits} write_output={write_output}
# log to {utils.LOGFILE}
"""
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    # Preprocessing
    if preprocess_kwargs is not None:
        # 1/f, bias & rnoise
        status = pipeline.exposure_detector_effects(file, **preprocess_kwargs)

    with jwst.datamodels.open(file) as dm:
        EXPOSURE_TYPE = dm.meta.exposure.type

    if EXPOSURE_TYPE == "NRS_IFU":
        make_trace_figures = False
        run_pathloss = False
        run_barshadow = False
        mask_zeroth_kwargs = None
        # write_output = False

    # Mask zeroth orders
    if mask_zeroth_kwargs is not None:
        with pyfits.open(file, mode="update") as hdul:
            with jwst.datamodels.open(hdul) as input_model:
                dq, slits_, bounding_boxes = zeroth_order_mask(
                    input_model, **mask_zeroth_kwargs
                )
                if dq.max() > 0:
                    if dq.size != hdul["DQ"].data.size:
                        hdul["DQ"].data |= dq
                    else:
                        subarray_ = input_model.meta.subarray
                        slx_ = slice(
                            subarray_.xstart - 1,
                            subarray_.xstart - 1 + subarray_.xsize,
                        )
                        sly_ = slice(
                            subarray_.ystart - 1,
                            subarray_.ystart - 1 + subarray_.ysize,
                        )
                        hdul["DQ"].data |= dq[sly_, slx_]

            hdul.flush()

    wstep = AssignWcsStep()

    msg = f"{file} AssignWCS"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    ##############
    # AssignWCS with extended wavelength range
    with jwst.datamodels.open(file) as dm:

        wcs_reference_files = step_reference_files(wstep, dm)
        new_waverange = os.path.basename(
            wcs_reference_files["wavelengthrange"]
        )
        new_waverange = new_waverange.replace(".asdf", "_ext.asdf")

        print(f"extend_wavelengthrange: {new_waverange}")
        if not os.path.exists(new_waverange):
            extend_wavelengthrange(
                ref_file=os.path.basename(
                    wcs_reference_files["wavelengthrange"]
                ),
                ranges=ranges,
            )

        # Use new reference
        wcs_reference_files["wavelengthrange"] = new_waverange
        # slit_y_range = [wstep.slit_y_low, wstep.slit_y_high]
        try:
            # with_wcs = load_wcs(dm, wcs_reference_files, slit_y_range)
            with_wcs = wstep.call(dm, override_wavelengthrange=new_waverange)
        except NoDataOnDetectorError:
            msg = f"{file} No open slits found to work on"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
            utils.LOGFILE = ORIG_LOGFILE
            return None

    if EXPOSURE_TYPE in ["NRS_MSASPEC", "NRS_IFU"]:
        ############
        # MSAFlagOpen
        msg = f"{file} MSAFlagOpen"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
        flag_open = MSAFlagOpenStep().call(with_wcs)
    else:
        flag_open = with_wcs

    ############
    # Extract2D
    if EXPOSURE_TYPE == "NRS_IFU":
        _slit = ext2d = flag_open
        _slit.name = "ifu"
        all_slits = False
    else:
        msg = f"{file} Extract2D"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        step2d = Extract2dStep()
        ext2d = step2d.call(flag_open)

        _slit_index = None
        if EXPOSURE_TYPE == "NRS_FIXEDSLIT":
            for ind, slit in enumerate(ext2d.slits):
                if f"NRS_{slit.name}_SLIT".upper() == ext2d.meta.aperture.name:
                    _slit_index = ind
                    break

        if _slit_index is not None:
            msg = f"{file} use slit_index = {_slit_index} for {slit.name}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
            slit_index = _slit_index
            all_slits = False

        _slit = ext2d[slit_index]

    if _slit.meta.target.catalog_name is not None:
        targ_ = _slit.meta.target.catalog_name.replace(" ", "-").replace(
            "_", "-"
        )
        targ_ = targ_.replace("(", "").replace(")", "").replace("/", "")
    else:
        targ_ = "cat"

    inst_key = (
        f"{_slit.meta.instrument.filter}_{_slit.meta.instrument.grating}"
    )

    if EXPOSURE_TYPE in ["NRS_FIXEDSLIT", "NRS_IFU"]:
        slit_prefix_ = (
            f"{file.split('_rate')[0]}_{targ_}_{inst_key}_{_slit.name}".lower()
        )
    else:
        if undo_flat:
            plabel = "raw"
        else:
            plabel = "phot"

        slit_prefix_ = f"{file.split('_rate')[0]}_{inst_key}_{plabel}.{_slit.name}.{_slit.source_name}".lower()

    det = _slit.meta.instrument.detector

    msg = f"{file} {inst_key} {det}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    ###############
    # Trace figures
    if make_trace_figures:
        xtr, ytr, wtr, rs, ds = msautils.slit_trace_center(
            _slit, with_source_xpos=False, with_source_ypos=False
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        ax.imshow(_slit.data, aspect="auto", vmin=-0.1, vmax=2, cmap="magma_r")
        ax.plot(xtr, ytr, color="magenta")
        ax.set_title(f"{inst_key} {det}")

        fig.tight_layout(pad=1)
        fig.savefig(f"{slit_prefix_}_trace.png".lower())

    ############
    # Flat-field
    msg = f"{file} FlatFieldStep"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    flat_step = FlatFieldStep()
    flat_reference_files = step_reference_files(flat_step, ext2d)

    if inst_key in ranges:
        full_range = ranges[inst_key]
    else:
        full_range = None
        undo_flat = False

    if undo_flat:
        ########
        # F-Flat
        new_fflat_filename = os.path.basename(
            flat_reference_files["fflat"]
        ).replace(".fits", "_ext.fits")

        msg = f"{file}  fflat = '{new_fflat_filename}'"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        if not os.path.exists(new_fflat_filename):
            if EXPOSURE_TYPE == "NRS_MSASPEC":
                ff = extend_quad_fflat(
                    flat_reference_files["fflat"],
                    full_range=full_range,
                    log_prefix=file,
                )
            else:
                ff = extend_fs_fflat(
                    flat_reference_files["fflat"],
                    full_range=full_range,
                    log_prefix=file,
                )

            ff.write(new_fflat_filename, overwrite=True)

        flat_reference_files["fflat"] = new_fflat_filename

        ########
        # S-Flat
        new_sflat_filename = os.path.basename(
            flat_reference_files["sflat"]
        ).replace(".fits", "_ext.fits")

        msg = f"{file}  sflat = '{new_sflat_filename}'"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        if not os.path.exists(new_sflat_filename):
            sf = extend_sflat(
                flat_reference_files["sflat"],
                full_range=full_range,
                log_prefix=file,
            )
            sf.write(new_sflat_filename, overwrite=True)

        flat_reference_files["sflat"] = new_sflat_filename

        ########
        # D-FLAT
        new_dflat_filename = os.path.basename(
            flat_reference_files["dflat"]
        ).replace(".fits", "_ext.fits")

        msg = f"{file}  dflat = '{new_dflat_filename}'"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        if not os.path.exists(new_dflat_filename):
            wmin, wmax = 10, 0
            for k in ranges:
                wmin = np.minimum(ranges[k][0], wmin)
                wmax = np.maximum(ranges[k][1], wmax)

            df = extend_dflat(
                flat_reference_files["dflat"],
                full_range=[wmin, wmax],
                log_prefix=file,
            )

            df.write(new_dflat_filename, overwrite=True)

        flat_reference_files["dflat"] = new_dflat_filename

    #########################
    # Metadata for fixed slit
    if EXPOSURE_TYPE == "NRS_FIXEDSLIT":
        for _slit in ext2d.slits:
            msautils.update_slit_metadata(_slit)

    # Run the pipeline step with the updated references (or the originals)
    flat_corr = flat_step.call(
        ext2d,
        override_fflat=flat_reference_files["fflat"],
        override_sflat=flat_reference_files["sflat"],
        override_dflat=flat_reference_files["dflat"],
    )

    if run_pathloss:
        ##########
        # PathLoss
        msg = f"{file} NRS_MSASPEC run PathLossStep"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        path_step = PathLossStep()
        path_filename = path_step.get_reference_file(flat_corr, "pathloss")

        if full_range is not None:
            new_path_filename = os.path.basename(path_filename).replace(
                ".fits", "_ext.fits"
            )

            if not os.path.exists(new_path_filename):
                path_hdul = extend_pathloss(
                    path_filename, full_range=full_range
                )
                path_hdul.writeto(new_path_filename, overwrite=True)
                path_hdul.close()

            path_result = path_step.call(
                flat_corr, override_pathloss=new_path_filename
            )
        else:
            path_result = path_step.call(flat_corr)
    else:
        path_result = flat_corr

    last_result = path_result

    if EXPOSURE_TYPE == "NRS_MSASPEC":

        if run_barshadow:
            ###########
            # BarShadow
            msg = f"{file} NRS_MSASPEC run BarShadowStep"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

            bar_step = BarShadowStep()

            bar_filename = bar_step.get_reference_file(
                path_result, "barshadow"
            )

            if full_range is not None:
                new_bar_filename = os.path.basename(bar_filename).replace(
                    ".fits", "_ext.fits"
                )

                if not os.path.exists(new_bar_filename):
                    bar_hdul = extend_barshadow(
                        bar_filename, full_range=full_range
                    )
                    bar_hdul.writeto(new_bar_filename, overwrite=True)
                    bar_hdul.close()

                bar_result = bar_step.call(
                    path_result, override_barshadow=new_bar_filename
                )
            else:
                bar_result = bar_step.call(path_result)

            last_result = bar_result

    ########
    # Photom
    phot_step = PhotomStep()
    phot_filename = phot_step.get_reference_file(last_result, "photom")
    area_filename = phot_step.get_reference_file(last_result, "area")

    phot_step = PhotomStep()

    if full_range is not None:

        new_phot_filename = os.path.basename(phot_filename).replace(
            ".fits", "_ext.fits"
        )

        msg = f"{file}  photom = '{new_phot_filename}'"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        if not os.path.exists(new_phot_filename):
            pref = extend_photom(phot_filename, ranges=ranges)
            pref.write(new_phot_filename, overwrite=True)

        # Apply photom
        result = phot_step.call(last_result, override_photom=new_phot_filename)

    else:
        result = phot_step.call(last_result)

    ########
    # Figure
    if make_trace_figures:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        ax.imshow(
            result.slits[slit_index].data,
            aspect="auto",
            vmin=-0.1 * 20,
            vmax=2 * 20,
            cmap="magma_r",
        )

        ax.plot(xtr, ytr, color="magenta")
        ax.set_title(f"{inst_key} {det}")
        fig.tight_layout(pad=1)

        fig.savefig(f"{slit_prefix_}_final.png".lower())

    # result.write(os.path.basename(file).replace('_rate.fits', '_photom.fits'))

    ########
    # Write calibrated slitlet files
    if write_output & (EXPOSURE_TYPE == "NRS_IFU"):
        cal_file = file.replace("_rate.fits", "_cal.fits")
        msg = f"{file} write calibrated IFU exposure cal_file"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
        result.write(cal_file, overwrite=True)

    elif write_output:
        if all_slits:
            slit_list = result.slits
        else:
            slit_list = result.slits[slit_index : slit_index + 1]

        for _slit in slit_list:
            if _slit.meta.exposure.type == "NRS_FIXEDSLIT":
                slit_prefix_ = f"{file.split('_rate')[0]}_{targ_}_{inst_key}_{_slit.name}".lower()
            else:
                if undo_flat:
                    plabel = "raw"
                else:
                    plabel = "phot"

                try:
                    source_alias = slit_best_source_alias(
                        _slit,
                        require_primary=False,
                        which="min",
                        verbose=False,
                    )
                except:
                    source_alias = _slit.source_name

                slit_prefix_ = f"{file.split('_rate')[0]}_{inst_key}_{plabel}.{_slit.name}.{source_alias}".lower()

            slit_file = slit_prefix_ + ".fits"

            msg = f"{file} write slitlet {slit_file}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

            slit_model = jwst.datamodels.SlitModel(_slit.instance)
            slit_model.write(slit_file, overwrite=True)

            with pyfits.open(slit_file, mode="update") as im:
                im[0].header["NOFLAT"] = (
                    True,
                    "Dummy flat field with extended wavelength references",
                )
                im[0].header["R_FFLAT"] = flat_reference_files["fflat"]
                im[0].header["R_SFLAT"] = flat_reference_files["sflat"]
                im[0].header["R_DFLAT"] = flat_reference_files["dflat"]
                im[0].header["R_PHOTOM"] = new_phot_filename

                im.flush()

    utils.LOGFILE = ORIG_LOGFILE

    return result
