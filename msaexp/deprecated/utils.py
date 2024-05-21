import os
import yaml

import warnings
import numpy as np
import astropy.io.fits as pyfits

import grizli.utils

from .. import utils as msautils

def summary_from_metafiles():
    """
    Generate summary files from metafiles.

    This function reads in metafiles with the extension '.msa.fits'
    and generates summary files in CSV and DS9 region files

    Parameters:
    -----------
    None

    Returns:
    --------
    None

    """
    import glob
    from grizli import utils, prep
    import astropy.io.fits as pyfits

    files = glob.glob("*msa.fits")
    files.sort()

    for file in files:
        im = pyfits.open(file)
        tab = utils.GTable(im["SOURCE_INFO"].data)
        tab.write(file.replace(".fits", ".csv"), overwrite=True)
        prep.table_to_regions(
            tab,
            file.replace(".fits", ".reg"),
            comment=tab["source_name"],
            size=0.1,
        )


def update_output_files(mode):
    """
    Rename output slitlet files and metadata with updated target names

    This function renames the output slitlet files and updates the metadata
    with the updated target names. It reads a YAML file containing the target
    names and their corresponding slit indices, and renames the slitlet files
    accordingly. The function also updates the YAML file with the new target
    names and writes it back to disk.

    Parameters:
    -----------
    mode : str
        The mode for which to update the output files.

    Returns:
    --------
        False if the YAML file does not exist, else None.

    """
    import glob
    import yaml

    import astropy.io.fits as pyfits

    from . import pipeline

    groups = pipeline.exposure_groups()

    yaml_file = f"{mode}.slits.yaml"

    if not os.path.exists(yaml_file):
        print(f"Skip {yaml_file}")
        return False

    with open(yaml_file) as fp:
        yaml_data = yaml.load(fp, Loader=yaml.Loader)

    files = groups[mode]

    orig_sources = []
    for i in range(len(yaml_data)):
        this_src = None
        row = []

        for file in files:
            base = file.split("_rate.fits")[0]
            with pyfits.open(f"{base}_phot.{i:03d}.fits") as im:
                src = im[1].header["srcname"]
                row.append(src)
                if not src.lower().startswith("background"):
                    if src in yaml_data:
                        this_src = src

        if this_src is None:
            this_src = src

        if this_src not in yaml_data:
            print(i, " ".join(row))

        orig_sources.append(this_src)

    for i, src in enumerate(orig_sources):
        if src not in yaml_data:
            print(f"! {src} not in {mode}")
            continue

        yaml_data[src]["slit_index"] = i
        new = msautils.rename_source(src)
        print(f"{i:>2d}   {src}  >>>  {new}")

        yaml_data[new] = yaml_data.pop(src)

    # Move phot files
    for i, src in enumerate(orig_sources):
        new = msautils.rename_source(src)
        print(f"{i:>2} {src} >> {new}")

        for file in files:
            base = file.split("_rate.fits")[0]
            old_file = f"{base}_phot.{i:03d}.fits"
            new_file = f"{base}_phot.{i:03d}.{new}.fits"
            if os.path.exists(old_file):
                cmd = f"mv {old_file} {new_file}"
                os.system(cmd)
                print(f"  {cmd}")
            else:
                print(f"  {old_file}  {new_file}")

    # Write updated yaml file
    with open(yaml_file, "w") as fp:
        yaml.dump(yaml_data, stream=fp)

    print(f"Fix {yaml_file}")


def update_slitlet_filenames(files, script_only=True, verbose=True):
    """
    Update slitlet filenames to reflect new convention with the
    correct `slitlet_id`:

    `{ROOT}_phot.{SLITLET_ID:03d}.{SOURCE_ID}.fits`

    Note: the function just prints messages that can be pasted into the shell
    for renaming the files

    Parameters
    ----------
    files : str
        List of slitlet files, `{ROOT}_phot.{SLITLET_ID:03d}.{SOURCE_ID}.fits`

    script_only : bool
        If False, rename the files.  Otherwise, just print messages.

    Returns
    -------
    commands : list
        List of shell commands for renaming files

    """

    import shutil
    import astropy.io.fits as pyfits

    commands = []

    for file in files:
        with pyfits.open(file) as im:
            slitlet_id = im[1].header["SLITID"]

        old_key = file.split("phot.")[1].split(".")[0]
        new_file = file.replace(f".{old_key}.", f".{slitlet_id:03d}.")
        if new_file != file:
            cmd = f"mv {file:<58} {new_file}"
            if not script_only:
                shutil.move(file, new_file)
        else:
            cmd = f"# {file:<58} - filename OK"

        if verbose:
            print(cmd)

        commands.append(cmd)

    return commands


def slit_metadata_to_header(slit, key="", header=None):
    """
    Get selected metadata.

    This would be similar to

    .. code-block:: python
        :dedent:

        from stdatamodels import fits_support
        hdul, _asdf = fits_support.to_fits(slit._instance, schema=slit._schema)

    and getting the keywords from `hdul`, but here the keywords are renamed
    slightly to allow adding the `key` suffix.

    Parameters
    ----------
    slit : `jwst.datamodels.SlitModel`
        Slit data model

    key : str, int
        Key that will be appended to header keywords, e.g.,
        ``header[f'FILE{key}']``. Should have just one or two characters to
        avoid making HIERARCH keywords longer than the FITS standard eight
        characters.

    header : `~astropy.io.fits.Header`
        Add keys to existing header

    Returns
    -------
    header : `~astropy.io.fits.Header`

    """
    if header is None:
        h = pyfits.Header()
    else:
        h = header

    meta = slit.meta.instance

    h[f"FILE{key}"] = meta["filename"], "Data filename"
    h[f"CALVER{key}"] = (
        meta["calibration_software_version"],
        "Calibration software version",
    )
    h[f"CRDS{key}"] = (
        meta["ref_file"]["crds"]["context_used"],
        "CRDS context",
    )

    h[f"GRAT{key}"] = meta["instrument"]["grating"], "Instrument grating"
    h[f"FILTER{key}"] = meta["instrument"]["filter"], "Instrument filter"

    if "msa_metadata_file" in meta["instrument"]:
        h[f"MSAMET{key}"] = (
            meta["instrument"]["msa_metadata_file"],
            "MSAMETF metadata file",
        )

        h[f"MSAID{key}"] = (
            meta["instrument"]["msa_metadata_id"],
            "MSAMETF metadata id",
        )
        h[f"MSACNF{key}"] = (
            meta["instrument"]["msa_configuration_id"],
            "MSAMETF metadata configuration id",
        )

    h[f"SLITID{key}"] = slit.slitlet_id, "slitlet_id from MSA file"
    h[f"SRCNAM{key}"] = slit.source_name, "source_name from MSA file"
    h[f"SRCID{key}"] = slit.source_id, "source_id from MSA file"
    h[f"SRCRA{key}"] = slit.source_ra, "source_ra from MSA file"
    h[f"SRCDEC{key}"] = slit.source_dec, "source_dec from MSA file"

    if slit.meta.photometry.pixelarea_steradians is None:
        h[f"PIXSR{key}"] = (1.0e-12, "Pixel area, sr")
    else:
        h[f"PIXSR{key}"] = (
            slit.meta.photometry.pixelarea_steradians,
            "Pixel area, sr",
        )

    h[f"TOUJY{key}"] = 1.0e12 * h[f"PIXSR{key}"], "Conversion to uJy/pix"

    h[f"DETECT{key}"] = (
        meta["instrument"]["detector"],
        "Instrument detector name",
    )

    h[f"XSTART{key}"] = slit.xstart, "Left detector pixel of 2D cutout"
    h[f"XSIZE{key}"] = slit.xsize, "X size of 2D cutout"
    h[f"YSTART{key}"] = slit.ystart, "Lower detector pixel of 2D cutout"
    h[f"YSIZE{key}"] = slit.ysize, "Y size of 2D cutout"

    h[f"RA_REF{key}"] = (
        meta["wcsinfo"]["ra_ref"],
        "Exposure RA reference position",
    )
    h[f"DE_REF{key}"] = (
        meta["wcsinfo"]["dec_ref"],
        "Exposure Dec reference position",
    )
    h[f"RL_REF{key}"] = meta["wcsinfo"]["roll_ref"], "Exposure roll referencev"
    h[f"V2_REF{key}"] = (
        meta["wcsinfo"]["v2_ref"],
        "Exposure V2 reference position",
    )
    h[f"V3_REF{key}"] = (
        meta["wcsinfo"]["v3_ref"],
        "Exposure V3 reference position",
    )
    h[f"V3YANG{key}"] = meta["wcsinfo"]["v3yangle"], "Exposure V3 y angle"

    h[f"RA_V1{key}"] = (
        meta["pointing"]["ra_v1"],
        "[deg] RA of telescope V1 axis",
    )
    h[f"DEC_V1{key}"] = (
        meta["pointing"]["dec_v1"],
        "[deg] Dec of telescope V1 axis",
    )
    h[f"PA_V3{key}"] = (
        meta["pointing"]["pa_v3"],
        "[deg] Position angle of V3 axis",
    )

    h[f"EXPSTR{key}"] = (
        meta["exposure"]["start_time_mjd"],
        "Exposure start MJD",
    )
    h[f"EXPEND{key}"] = meta["exposure"]["start_time_mjd"], "Exposure end MJD"
    h[f"EFFEXP{key}"] = (
        meta["exposure"]["effective_exposure_time"],
        "Effective exposure time",
    )

    h[f"DITHN{key}"] = (
        meta["dither"]["position_number"],
        "Dither position number",
    )
    h[f"DITHX{key}"] = meta["dither"]["x_offset"], "Dither x offset"
    h[f"DITHY{key}"] = meta["dither"]["y_offset"], "Dither y offset"
    if "nod_type" in meta["dither"]:
        h[f"DITHT{key}"] = meta["dither"]["nod_type"], "Dither nod type"
    else:
        h[f"DITHT{key}"] = "INDEF", "Dither nod type"

    h[f"NFRAM{key}"] = meta["exposure"]["nframes"], "Number of frames"
    h[f"NGRP{key}"] = meta["exposure"]["ngroups"], "Number of groups"
    h[f"NINTS{key}"] = meta["exposure"]["nints"], "Number of integrations"
    h[f"RDPAT{key}"] = meta["exposure"]["readpatt"], "Readout pattern"

    return h


def detector_bounding_box(file):
    """
    This function reads slit files and metadata for slits and calculates the
    bounding box coordinates for each slit.

    Parameters:
    -----------
    file : str
        The path to the file containing slits.

    Returns:
    --------
    slit_borders : dict
        A dictionary containing the bounding box coordinates for each slit.

    """

    import glob
    from tqdm import tqdm
    import yaml

    from jwst.datamodels import SlitModel
    from grizli import utils

    froot = file.split("_rate.fits")[0]

    slit_files = glob.glob(f"{froot}*phot.[01]*fits")

    slit_borders = {}

    fp = open(f"{froot}.detector_trace.reg", "w")
    fp.write("image\n")

    fpr = open(f"{froot}.detector_bbox.reg", "w")
    fpr.write("image\n")

    for _file in tqdm(slit_files[:]):
        fslit = SlitModel(_file)

        tr = fslit.meta.wcs.get_transform("slit_frame", "detector")
        waves = np.linspace(
            np.nanmin(fslit.wavelength), np.nanmax(fslit.wavelength), 32
        )

        xmin, ymin = tr(
            waves * 0.0, waves * 0.0 + fslit.slit_ymin, waves * 1.0e-6
        )
        xmin += fslit.xstart
        ymin += fslit.ystart

        # pl = plt.plot(xmin, ymin)
        xmax, ymax = tr(
            waves * 0.0, waves * 0.0 + fslit.slit_ymax, waves * 1.0e-6
        )
        xmax += fslit.xstart
        ymax += fslit.ystart

        xcen, ycen = tr(
            waves * 0.0, waves * 0.0 + fslit.source_ypos, waves * 1.0e-6
        )
        xcen += fslit.xstart
        ycen += fslit.ystart

        # plt.plot(xmax, ymax, color=pl[0].get_color())
        _x = [
            fslit.xstart,
            fslit.xstart + fslit.xsize,
            fslit.xstart + fslit.xsize,
            fslit.xstart,
        ]
        _y = [
            fslit.ystart,
            fslit.ystart,
            fslit.ystart + fslit.ysize,
            fslit.ystart + fslit.ysize,
        ]

        sr = utils.SRegion(np.array([_x, _y]), wrap=False)

        if fslit.source_name.startswith("background"):
            props = "color=white"
        elif "_-" in fslit.source_name:
            props = "color=magenta"
        else:
            props = "color=cyan"

        sr.label = fslit.source_name
        sr.ds9_properties = props
        fpr.write(sr.region[0] + "\n")

        _key = _file.split(".")[-2]

        slit_borders[_key] = {
            "min": [xmin.tolist(), ymin.tolist()],
            "max": [xmax.tolist(), ymax.tolist()],
            "cen": [xcen.tolist(), ycen.tolist()],
            "wave": waves,
            "xstart": fslit.xstart,
            "xsize": fslit.xsize,
            "ystart": fslit.ystart,
            "ysize": fslit.ysize,
            "src_name": fslit.source_name,
            "slit_ymin": fslit.slit_ymin,
            "slit_ymax": fslit.slit_ymax,
            "source_ypos": fslit.source_ypos,
        }

        # sr = utils.SRegion(np.array([np.append(xmin, xmax[::-1]),
        #                              np.append(ymin, ymax[::-1])]),
        #                    wrap=False)
        sr = utils.SRegion(
            np.array(
                [
                    np.append(xmin, xmax[::-1]),
                    np.append(ycen - 1, (ycen + 1)[::-1]),
                ]
            ),
            wrap=False,
        )

        sr.label = fslit.source_name
        sr.ds9_properties = props
        fp.write(sr.region[0] + "\n")

    fp.close()
    fpr.close()

    with open(f"{froot}.detector_trace.yaml", "w") as fp:
        yaml.dump(slit_borders, stream=fp)

    return slit_borders


def slit_cutout_region(slitfile, as_text=True, skip=8, verbose=False):
    """
    Generate the region in the original exposure corresponding to a slit cutout

    Parameters:
    -----------
    slitfile : str
        The path to the slit file.
    as_text : bool, optional
        If True, the region is returned as a text string.
        If False, the region is returned as a `grizli.utils.SRegion` object.
        Default is True.
    skip : int, optional
        The number of pixels to skip between each point in the region. Default
        is 8.
    verbose : bool, optional
        Verbose output.

    Returns:
    --------
    str or `grizli.utils.SRegion`
        If `as_text` is True, the region is returned as a text string.
        If `as_text` is False, the region is returned as a
        `grizli.utils.SRegion` object.

    """
    import jwst.datamodels

    obj = jwst.datamodels.open(slitfile)
    wcs = obj.meta.wcs

    if verbose:
        print(f"Get slit region: {obj.source_name}")

    sh = obj.data.shape
    yp, xp = np.indices(sh)

    d2s = wcs.get_transform("detector", "slit_frame")
    s2d = wcs.get_transform("slit_frame", "detector")

    ss = d2s(xp, yp)
    sx, sy = s2d(
        ss[0][sh[0] // 2, :],
        ss[0][sh[0] // 2, :] * 0.0 + obj.source_ypos,
        np.nanmedian(ss[2], axis=0),
    )

    ypi = yp * 1.0
    ypi[~np.isfinite(ss[1])] = np.nan
    ymi = np.nanmin(ypi, axis=0)
    yma = np.nanmax(ypi, axis=0)

    xy = [
        np.array(
            [
                np.hstack([sx, sx[::-1]]) + obj.xstart + 1,
                np.hstack([sy + 2.5, sy[::-1] - 2.5]) + obj.ystart + 1,
            ]
        ).T[::skip, :],
        np.array(
            [
                np.hstack([sx, sx[::-1]]) + obj.xstart + 1,
                np.hstack([ymi, yma[::-1]]) + obj.ystart + 1,
            ]
        ).T[::skip, :],
    ]

    for i in range(2):
        ok = np.isfinite(xy[i]).sum(axis=1) == 2
        xy[i] = xy[i][ok, :]

    obj.close()

    sr = grizli.utils.SRegion(xy, wrap=False)
    x0 = sr.centroid[0]
    if as_text:
        colors = ["white", "cyan"]
        txt = "\n".join(
            [
                "polygon(" + r[2:-2] + f" # color={colors[i]}"
                for i, r in enumerate(sr.polystr(precision=2))
            ]
        )
        txt += "\n# text({0:.2f},{1:.2f}) text={{{2}}} color={3}\n".format(
            x0[0], x0[1] + 2, obj.source_name, colors[-1]
        )
        txt = txt.replace("),(", ",")
        return txt
    else:
        sr.label = obj.source_name
        return sr


def all_slit_cutout_regions(files, output="slits.reg", **kwargs):
    """
    Generate slit cutout regions for multiple files and save them to a file.

    Parameters:
    ----------
    files : list
        A list of file paths that will be passed to
       `~msaexp.utils.slit_cutout_region`
    output : str, optional
        The output file path where the slit cutout regions will be saved.
        Default is 'slits.reg'.
    kwargs : dict, optional
        Additional keyword arguments to be passed to the slit_cutout_region
        function.

    Returns:
    -------
    None

    """

    with open(output, "w") as fp:
        for file in files:
            txt = slit_cutout_region(file, as_text=True, skip=1, verbose=True)
            fp.write(txt)
