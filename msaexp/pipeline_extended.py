"""
Extend CRDS reference files for fixed-slit observations
"""

import os
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

from jwst.assign_wcs.assign_wcs import load_wcs
from jwst.flatfield import flat_field
import jwst.photom.photom

from grizli import utils
import msaexp.utils

# NEW_WAVERANGE_REF = "jwst_nirspec_wavelengthrange_0008_ext.asdf"

NEW_RANGE = {
    "F070LP_G140M": [0.6, 3.3],
    "F100LP_G140M": [0.9, 3.3],
    "F170LP_G235M": [1.5, 5.3],
    "F290LP_G395M": [2.6, 6.0],
    "CLEAR_PRISM": [0.5, 5.6],
}


def step_reference_files(step, input_model):
    """
    Get reference filenames for a `jwst` pipeline step
    """
    reference_file_names = {}
    for reftype in step.reference_file_types:
        reffile = step.get_reference_file(input_model, reftype)
        reference_file_names[reftype] = reffile if reffile else ""

    return reference_file_names


VERBOSITY = True


def extend_wavelengthrange(ref_file="jwst_nirspec_wavelengthrange_0008.asdf"):
    """
    """

    if ref_file.startswith('/'):
        NIRSPEC_CRDS = ref_file
    else:
        NIRSPEC_CRDS = os.path.join(
        os.getenv("CRDS_PATH"), "references/jwst/nirspec"
    )

    waverange = jwst.datamodels.open(os.path.join(NIRSPEC_CRDS, ref_file))

    for k in NEW_RANGE:
        i = waverange.waverange_selector.index(k)
        waverange.wavelengthrange[i] = [v * 1e-6 for v in NEW_RANGE[k]]

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
        if ("H" in spl[1]) | (spl[1] == "MIRROR") | (key == "F100LP_PRISM"):
            continue
        if spl[0] in ["OPAQUE", "F110W", "F140X"]:
            continue
        if (spl[1] == "PRISM") & (spl[0] != "CLEAR"):
            continue

        msg = f"{i:>2} {waverange.waverange_selector[i]:>16} {waverange.order[i]:>3}   {waverange.wavelengthrange[i][0]*1.e6:.2f}  -  {waverange.wavelengthrange[i][1]*1.e6:.2f}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    waverange.write(new_waverange)


def extend_fs_fflat(
    fflat_filename, full_range=[0.55, 5.6], log_prefix="extend_fs_fflat"
):
    """ """
    fbase = os.path.basename(fflat_filename)

    ff = jwst.datamodels.open(fflat_filename)

    fftab = astropy.table.Table(ff.flat_table)
    nelem = len(fftab[0]["wavelength"])

    non_zero = fftab["wavelength"][0] > 0

    newtab = astropy.table.Table()
    newtab["slit_name"] = fftab["slit_name"]

    med_FILL_VALUE = np.nanmedian(fftab["data"][0][non_zero])
    FILL_VALUE = 2.0e20

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
    fflat_filename, full_range=[0.55, 5.6], log_prefix="extend_quad_fflat"
):
    """ """
    fbase = os.path.basename(fflat_filename)

    ff = jwst.datamodels.open(fflat_filename)

    for i in range(4):
        fftab = astropy.table.Table(ff.quadrants[i].flat_table)
        non_zero = fftab["wavelength"][0] > 0

        newtab = astropy.table.Table()
        newtab["slit_name"] = fftab["slit_name"]

        med_FILL_VALUE = np.nanmedian(fftab["data"][0][non_zero])
        FILL_VALUE = 2.0e20

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
    """ """
    fbase = os.path.basename(sflat_filename)

    sf = jwst.datamodels.open(sflat_filename)

    msg = f"{log_prefix} {fbase} Unset NO_FLAT_FIELD from SFlat DQ"
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    sf.dq -= sf.dq & 2**18

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
    dflat_filename, full_range=[0.55, 5.6], log_prefix="extend_dflat"
):

    fbase = os.path.basename(dflat_filename)

    df = jwst.datamodels.open(dflat_filename)

    dflat_waves = df.extra_fits.WAVELENGTH.data.wavelength.tolist()

    prism_min = full_range[0]

    prism_max = full_range[1]

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

    # Min
    dflat_waves = df.extra_fits.WAVELENGTH.data.wavelength.tolist()

    if dflat_waves[0] > prism_min:

        msg = f"{log_prefix} {fbase} max wave = {dflat_waves[0]:.2f} -> prepend {prism_min}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        # dflat_waves.append(prism_max)
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

    return df


def extend_reference_files(
    file,
    slit_index=0,
    all_slits=True,
    write_output=True,
    set_log=True,
    skip_existing_log=False,
    undo_flat=True,
    **kwargs,
):
    """ """
    import time
    from jwst.assign_wcs.util import NoDataOnDetectorError

    ORIG_LOGFILE = utils.LOGFILE
    if set_log:
        utils.LOGFILE = file.replace("_rate.fits", "_rate.wave_log.txt")

        if os.path.exists(utils.LOGFILE) & skip_existing_log:
            utils.LOGFILE = ORIG_LOGFILE
            print(f"log file {utils.LOGFILE} found, skip")
            return True

    msg = f"""
########
# extend_reference_files {time.ctime()}
# {file}
# slit_index={slit_index} all_slits={all_slits} write_output={write_output}
# log to {utils.LOGFILE}
"""
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

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

        if not os.path.exists(new_waverange):
            extend_wavelengthrange(
                ref_file=os.path.basename(
                    wcs_reference_files["wavelengthrange"]
                )
            )

        # Use new reference
        wcs_reference_files["wavelengthrange"] = new_waverange
        slit_y_range = [wstep.slit_y_low, wstep.slit_y_high]
        try:
            with_wcs = load_wcs(dm, wcs_reference_files, slit_y_range)
        except NoDataOnDetectorError:
            msg = f"{file} No open slits found to work on"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
            utils.LOGFILE = ORIG_LOGFILE
            return False

    ############
    # Extract2D
    msg = f"{file} Extract2D"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    step2d = Extract2dStep()
    ext2d = step2d.call(with_wcs)

    _slit_index = None
    if ext2d.meta.exposure.type == "NRS_FIXEDSLIT":
        for ind, slit in enumerate(ext2d.slits):
            if f"NRS_{slit.name}_SLIT".upper() == ext2d.meta.aperture.name:
                _slit_index = ind
                break

    if _slit_index is not None:
        msg = f"{file} use slit_index = {_slit_index} for {slit.name}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
        slit_index = _slit_index
        all_slits = False

    slit = ext2d[slit_index]

    targ_ = slit.meta.target.catalog_name.replace(" ", "-").replace("_", "-")

    inst_key = f"{slit.meta.instrument.filter}_{slit.meta.instrument.grating}"

    slit_prefix = (
        f"{file.split('_rate')[0]}_{targ_}_{inst_key}_{slit.name}".lower()
    )

    det = slit.meta.instrument.detector

    msg = f"{file} {inst_key} {det}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    xtr, ytr, wtr, rs, ds = msaexp.utils.slit_trace_center(
        slit, with_source_xpos=False, with_source_ypos=False
    )

    # Trace
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.imshow(slit.data, aspect="auto", vmin=-0.1, vmax=2, cmap="magma_r")
    ax.plot(xtr, ytr, color="magenta")
    ax.set_title(f"{inst_key} {det}")
    fig.tight_layout(pad=1)

    fig.savefig(f"{slit_prefix}_trace.png".lower())

    ############
    # Flat-field
    msg = f"{file} FlatFieldStep"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    flat_step = FlatFieldStep()
    flat_reference_files = step_reference_files(flat_step, ext2d)

    ########
    # FFlat
    new_fflat_filename = os.path.basename(
        flat_reference_files["fflat"]
    ).replace(".fits", "_ext.fits")
    msg = f"{file}  fflat = '{new_fflat_filename}'"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    full_range = NEW_RANGE[inst_key]

    # if inst_key == 'F100LP_G140M':
    #     full_range[0] = 0.952
    #     msg = f"{file}  set minimum wavelength {full_range[0]}"
    #     utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    if (not os.path.exists(new_fflat_filename)) & undo_flat:
        if ext2d.meta.exposure.type == "NRS_MSASPEC":
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

    if undo_flat:
        flat_reference_files["fflat"] = new_fflat_filename

    ## SFlat
    new_sflat_filename = os.path.basename(
        flat_reference_files["sflat"]
    ).replace(".fits", "_ext.fits")

    msg = f"{file}  sflat = '{new_sflat_filename}'"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    if (not os.path.exists(new_sflat_filename)) & undo_flat:
        sf = extend_sflat(
            flat_reference_files["sflat"],
            full_range=full_range,
            log_prefix=file,
        )
        sf.write(new_sflat_filename, overwrite=True)

    if undo_flat:
        flat_reference_files["sflat"] = new_sflat_filename

    ###############
    # DFLAT
    new_dflat_filename = os.path.basename(
        flat_reference_files["dflat"]
    ).replace(".fits", "_ext.fits")

    msg = f"{file}  dflat = '{new_dflat_filename}'"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    wmin, wmax = 10, 0
    for k in NEW_RANGE:
        wmin = np.minimum(NEW_RANGE[k][0], wmin)
        wmax = np.maximum(NEW_RANGE[k][1], wmax)

    if not os.path.exists(new_dflat_filename):
        df = extend_dflat(
            flat_reference_files["dflat"],
            full_range=[wmin, wmax],
            log_prefix=file,
        )

        df.write(new_dflat_filename, overwrite=True)

    flat_reference_files["dflat"] = new_dflat_filename

    ### Apply flake flats
    flat_models = flat_step._get_references(ext2d, ext2d.meta.exposure.type)
    for k in flat_models:
        if flat_models[k] is not None:
            flat_models[k] = flat_models[k].__class__(flat_reference_files[k])

    # Metadata for fixed slit
    if ext2d.meta.exposure.type == "NRS_FIXEDSLIT":
        for _slit in ext2d.slits:
            msaexp.utils.update_slit_metadata(_slit)

    flat_corr, flat_applied = flat_field.do_correction(
        ext2d, **flat_models, inverse=flat_step.inverse
    )

    # Close reference models
    for k in flat_models:
        if flat_models[k] is not None:
            flat_models[k].close()

    # Path and bar
    if ext2d.meta.exposure.type == "NRS_MSASPEC":

        msg = f"{file} NRS_MSASPEC run PathLossStep"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        path_step = PathLossStep()
        path_result = path_step.call(flat_corr)

        msg = f"{file} NRS_MSASPEC run BarShadowStep"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        bar_step = BarShadowStep()
        bar_result = bar_step.call(path_result)

        last_result = bar_result

    else:
        last_result = flat_corr

    ###########
    # Photom

    phot_step = PhotomStep()
    phot_filename = phot_step.get_reference_file(last_result, "photom")
    area_filename = phot_step.get_reference_file(last_result, "area")

    new_phot_filename = os.path.basename(phot_filename).replace(
        ".fits", "_ext.fits"
    )

    msg = f"{file}  photom = '{new_phot_filename}'"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    if not os.path.exists(new_phot_filename):
        pref = jwst.datamodels.open(phot_filename)
        ptab = astropy.table.Table(pref.phot_table)

        for k in NEW_RANGE:
            filt, grat = k.split("_")
            nfull = len(ptab[0]["wavelength"])
            nelem = nfull - 4
            wgrid = np.zeros(nfull, dtype=np.float32)
            wgrid[:nelem] = np.logspace(*np.log10(NEW_RANGE[k]), nelem)
            resp = (wgrid > 0) * 1

            rows = (ptab["grating"] == grat) & (ptab["filter"] == filt)
            # print(k, rows.sum(), NEW_RANGE[k])

            ptab["nelem"][rows] = nelem
            ptab["wavelength"][rows] = wgrid
            ptab["relresponse"][rows] = resp

        pref.phot_table = pyfits.BinTableHDU(ptab).data

        pref.write(new_phot_filename, overwrite=True)

    # Apply photom

    correction_pars = None

    phot = jwst.photom.photom.DataSet(
        last_result,
        phot_step.inverse,
        phot_step.source_type,
        phot_step.mrs_time_correction,
        correction_pars,
    )

    result = phot.apply_photom(new_phot_filename, area_filename)

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

    fig.savefig(f"{slit_prefix}_final.png".lower())

    if write_output:
        if all_slits:
            slit_list = result.slits
        else:
            slit_list = result.slits[slit_index : slit_index + 1]

        for _slit in slit_list:
            if _slit.meta.exposure.type == "NRS_FIXEDSLIT":
                slit_prefix_ = f"{file.split('_rate')[0]}_{targ_}_{inst_key}_{_slit.name}".lower()
            else:
                slit_prefix_ = f"{file.split('_rate')[0]}_{inst_key}_phot.{_slit.name}.{_slit.source_name}".lower()

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


def run_pipeline_with_extended_wavelengths(file, **kwargs):
    result = extend_reference_files(file, **kwargs)
    return result
