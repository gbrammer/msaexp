import os
import numpy as np
import subprocess

import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

from grizli import jwst_utils

jwst_utils.set_quiet_logging(50)

from grizli import utils

from .. import ifu as msaifu
from .. import utils as msautils

__all__ = [
    "preprocess_ifu_file",
    "fs_to_ptab",
    "run_one_preprocess_ifu",
    "combine_ifu_pipeline",
]


def run_one_preprocess_ifu(clean=False, sync=False, rowid=None, **kwargs):
    """
    Process with interaction with DJA db
    """
    import glob
    from grizli.aws import db
    import time

    if rowid is None:
        row = db.SQL(
            f"""
            SELECT rowid, \"fileSetName\", LOWER(detector) as detector
            FROM nirspec_ifu_exposures
            WHERE status = 0
            ORDER BY RANDOM()
            LIMIT 1
        """
        )
    else:
        row = db.SQL(
            f"""
            SELECT rowid, \"fileSetName\", LOWER(detector) as detector
            FROM nirspec_ifu_exposures
            WHERE rowid = {rowid}
            ORDER BY RANDOM()
            LIMIT 1
        """
        )

    if len(row) == 0:
        return None

    rate_file = "{fileSetName}_{detector}_rate.fits".format(**row[0])

    rowid = row["rowid"][0]
    if sync:
        now = time.time()
        db.execute(
            f"UPDATE nirspec_ifu_exposures SET status = 1, ctime = {now} WHERE rowid = {rowid}"
        )

    cube = preprocess_ifu_file(rate_file, sync=sync, **kwargs)

    if sync:
        if isinstance(cube, str):
            # rate_file not found / downloaded
            status = 3
        elif not os.path.exists(rate_file.replace("_rate.fits", "_ptab.fits")):
            # Script seemed to finish but ptab product not found
            status = 9
        else:
            status = 2

        now = time.time()
        db.execute(
            f"UPDATE nirspec_ifu_exposures SET status = {status}, ctime = {now} WHERE rowid = {rowid}"
        )

    if clean:
        files = glob.glob(rate_file.replace("_rate.fits", "*"))
        files.sort()
        for file in files:
            print(f"rm {file}")
            os.remove(file)

    return cube


def preprocess_ifu_file(
    file,
    perform_saturation=True,
    do_flatfield=False,
    do_photom=False,
    extend_wavelengths=True,
    skip_pixel_table_background=True,
    sync=False,
    s3_prefix="s3://msaexp-nirspec/ifu_exposures",
    **kwargs,
):
    """
    Parameters
    ----------
    file : str
        ``rate.fits`` filename

    perform_saturation : bool
        ``perform`` argument to `msaexp.utils.exposure_ramp_saturation`

    do_flatfield, do_photom, extend_wavelengths : bool
        Parameters to `msaexp.ifu.ExposureCube`

    kwargs : dict
        Additional keyword args passed to `msaexp.ifu.ExposureCube`
    Returns
    -------
    cube : `msaexp.ifu.ExposureCube`
        Cube object
    """
    import mastquery.utils

    if (not os.path.exists(file)) & (
        not os.path.exists(file.replace("_rate.fits", "_ptab.fits"))
    ):
        mastquery.utils.download_from_mast([file])

        if not os.path.exists(file):
            return f"{file} not found"

    cube = msaifu.ExposureCube(
        file,
        do_flatfield=do_flatfield,
        do_photom=do_photom,
        extend_wavelengths=extend_wavelengths,
        **kwargs,
    )

    if cube.ptab is None:
        # Shorten for FITS header
        try:
            desc = cube.input.meta.target.description.split("; ")
            cube.input.meta.target.description = "; ".join(desc[:2])
        except AttributeError:
            pass

        if "ref_coord" not in kwargs:
            kwargs["ref_coord"] = (cube.target_ra, cube.target_dec)

        cube.process_pixel_table(
            skip_pixel_table_background=skip_pixel_table_background, **kwargs
        )
        plt.close("all")

    # Saturated mask
    sat_file, sat_i = msautils.exposure_ramp_saturation(
        file, perform=perform_saturation, **kwargs
    )
    cube.saturated_mask = sat_i
    cube.saturated_file = sat_file

    # Sky from fixed slit
    fs_file = file.replace("_rate.fits", "_fs.fits")

    if os.path.exists(fs_file):
        print(f"Load fixed_slit file {fs_file}")
        fs_data = {}
        with pyfits.open(fs_file) as fs_im:
            for ext in fs_im[1:]:
                k = ext.header["FSNAME"]
                fs_data[k] = utils.read_catalog(ext)
    else:
        fs_data = msaifu.ifu_sky_from_fixed_slit(
            rate_file=file,
            slit_names=["S200A1", "S200A2", "S400A1", "S200B1", "S1600A1"][
                :-1
            ],
        )

        hdul = pyfits.HDUList(
            [pyfits.PrimaryHDU()]
            + [pyfits.BinTableHDU(fs_data[k]) for k in fs_data]
        )
        hdul.writeto(fs_file, overwrite=True)

    cube.fixed_slit = fs_data

    if sync:
        visit = os.path.basename(file.split("_")[0])
        s3_path = (
            os.path.join(s3_prefix, visit)
            .replace("//", "/")
            .replace("s3:/", "s3://")
        )
        file_prefix = os.path.basename(file.split("_rate")[0])
        for ext in [
            "cal.yaml",
            "ptab.fits",
            "satflag.fits.gz",
            "fs.fits",
            "rate_onef_axis0.png",
            "rate_onef_axis1.png",
        ]:
            file_ext = file_prefix + "_" + ext
            if os.path.exists(file_ext):
                send_command = (
                    f"aws s3 cp {file_ext} {s3_path}/ --acl public-read"
                )
                print(send_command)
                sync_result = subprocess.run(
                    send_command, shell=True, capture_output=True
                )

    return cube


def fs_to_ptab(fs_file, slice_offset=50):
    """
    Convert a ``fs.fits`` table of the fixed-slit spectra to a pixel table

    Parameters
    ----------
    fs_file : str
        Filename of a fixed slit summary fits file generated from the
        `msaexp.ifu.ifu_sky_from_fixed_slit` table

    slice_offset : int
        Integer of first fixed slit "slice".  The IFU slices are numbered 0-29.

    Returns
    -------
    fs_ptab : Table
        Pixel table

    """
    import astropy.table

    FIXED_SLIT_NAMES = ["S200A1", "S200A2", "S400A1", "S1600A1", "S200B1"]

    fs_data = {}
    with pyfits.open(fs_file) as fs_im:
        for ext in fs_im[1:]:
            k = ext.header["FSNAME"]
            fs_data[k] = utils.read_catalog(ext)

    tabs = []
    for i, k in enumerate(fs_data):
        if k not in FIXED_SLIT_NAMES:
            continue

        fsk = fs_data[k]
        fsk.meta = msaifu.meta_lowercase(fsk.meta)

        yp, xp = np.indices((fsk.meta["ysize"], fsk.meta["xsize"]))
        xp += fsk.meta["xstart"] - 1
        yp += fsk.meta["ystart"] - 1

        # Like pixtab
        fsk["xpix"] = xp.flatten()
        fsk["ypix"] = yp.flatten()
        fsk["slice"] = slice_offset + FIXED_SLIT_NAMES.index(k)
        # fsk['xsky'] = 0.
        tabs.append(fsk[np.isfinite(fsk["wave"] + fsk["data"])])

    fs_ptab = astropy.table.vstack(tabs)
    fs_ptab.meta["nslits"] = len(tabs)

    for i, k in enumerate(fs_data):
        if k not in FIXED_SLIT_NAMES:
            continue

        slice_id = 50 + FIXED_SLIT_NAMES.index(k)
        fs_ptab.meta[f"slice{i}"] = slice_id
        fs_ptab.meta[f"slit{i}"] = k

    return fs_ptab


def run_one_products_ifu(
    rowid=None,
    sync=True,
    clean=True,
    s3_prefix="s3://msaexp-nirspec/ifu_exposures/",
    **kwargs,
):
    """ """
    import glob
    import yaml

    from grizli.aws import db
    import time

    if rowid is None:
        row = db.SQL(
            f"""
            SELECT *
            FROM nirspec_ifu_products
            WHERE status = 0
            ORDER BY RANDOM()
            LIMIT 1
        """
        )
    else:
        row = db.SQL(
            f"""
            SELECT *
            FROM nirspec_ifu_products
            WHERE rowid = {rowid}
            ORDER BY RANDOM()
            LIMIT 1
        """
        )

    if len(row) == 0:
        return None

    if row["yaml_kwargs"][0]:
        yaml_kwargs = yaml.load(row["yaml_kwargs"][0], Loader=yaml.Loader)

        for k in yaml_kwargs:
            if k not in kwargs:
                kwargs[k] = yaml_kwargs[k]

    rowid = row["rowid"][0]
    obsid = row["obsid"][0]
    gfilt = row["gfilt"][0]

    if sync:
        now = time.time()
        db.execute(
            f"UPDATE nirspec_ifu_products SET status = 1, ctime = {now} WHERE rowid = {rowid}"
        )

    # Do it
    result = combine_ifu_pipeline(obsid=obsid, gfilt=gfilt, **kwargs)

    outroot = result["outroot"]
    result_files = glob.glob(f"{outroot}*")

    resp = result["resp"]
    for row in resp:
        result_files += glob.glob(
            "{fileSetName}_{detector}*".format(**row).lower()
        )

    result_files.sort()

    if sync:

        now = time.time()
        db.execute(
            f"UPDATE nirspec_ifu_products SET status = 2, ctime = {now}, outroot = '{outroot}' WHERE rowid = {rowid}"
        )

        s3_path = os.path.join(s3_prefix, "jw" + obsid) + "/"
        s3_path = s3_path.replace("//", "/").replace("s3:/", "s3://")

        send_command = f'aws s3 sync ./ {s3_path} --exclude "*" --include "{outroot}*" --acl public-read'

        print(send_command)
        sync_result = subprocess.run(
            send_command, shell=True, capture_output=True
        )

    if clean:
        for file in result_files:
            print(f"rm {file}")
            os.remove(file)

    return result


def combine_ifu_pipeline(
    obsid="02913001001",
    gfilt="F170LP_G235M",
    scale=0.05,
    pixfrac=0.75,
    recenter_cube=True,
    perform_saturation=True,
    run_cube_diagnostics=True,
    write_cube_diagnostics=True,
    **kwargs,
):
    """
    Parameters
    ---------
    obsid : str
        Observation ID (same as ``visit_id`` in the MAST query)

    gfilt : str
        Filter and grating combination

    scale : float
        Drizzle pixel scale

    pixfrac : float
        Pseudo-drizzle pixfrac

    """
    from grizli.aws import db

    filter, grating = gfilt.split("_")

    resp = db.SQL(
        f"""
    select * from nirspec_ifu_exposures
    where visit_id = '{obsid}' AND grating = '{grating}' AND filter = '{filter}' and status = 2
    order by \"fileSetName\", detector
    """
    )

    print(f"DB: {len(resp)} exposures")

    s3_prefix = "s3://msaexp-nirspec/ifu_exposures"

    for row in resp:
        for ext in ["ptab.fits", "fs.fits", "satflag.fits.gz"]:
            s3_file = "{s3_prefix}/jw{visit_id}/{fileSetName}_{detector}_{ext}".format(
                s3_prefix=s3_prefix, ext=ext, **row
            ).lower()
            # print(s3_file)
            if not os.path.exists(os.path.basename(s3_file)):
                os.system(f"aws s3 cp {s3_file} . ")
            else:
                print(os.path.basename(s3_file))

    files = [
        "{fileSetName}_{detector}_rate.fits".format(**row).lower()
        for row in resp
    ]

    kwargs["files"] = files
    kwargs["pixel_size"] = scale
    kwargs["pixfrac"] = pixfrac
    kwargs["filter"] = filter
    kwargs["grating"] = grating
    kwargs["obsid"] = obsid
    kwargs["recenter_cube"] = recenter_cube
    kwargs["perform_saturation"] = perform_saturation

    outroot, cubes, ptab, hdul = msaifu.ifu_pipeline(
        # # outroot=None,
        # pixel_size=scale,
        # pixfrac=pixfrac,
        # side="auto",
        # # wave_sample=1.05,
        # files=files,
        # obsid=obsid,
        # filter=filter,
        # grating=grating,
        # # download=False,
        # use_token=False,
        # sky_annulus=None,
        # exposure_type="rate",
        # do_flatfield=False,
        # do_photom=False,
        # extend_wavelengths=True,
        # use_first_center=True,
        # slice_wavelength_range=[0.5e-6, 5.6e-6],
        # # make_drizzled=1,
        # bad_pixel_flag=(
        #     msautils.BAD_PIXEL_FLAG & ~1024 | 4096 | 1073741824 | 16777216
        # ),
        # # detectors=["nrs1", "nrs2"],
        # # BAD_PIXEL_FLAG=1, dilate_failed_open=False,
        # perform_saturation=True,
        # recenter_cube=True,
        # drizzle_wave_limits=(4.5, 4.9),
        **kwargs,
    )

    cube_file = outroot + ".fits"

    # cube products
    if os.path.exists(cube_file) & run_cube_diagnostics:
        cube_hdu = pyfits.open(cube_file)

        result = msaifu.cube_make_diagnostics(
            cube_hdu,
            **kwargs,
            # scale_func=np.arcsinh, figsize=(12, 4), wave_power=-1, erode_background=None,
            # thresh_percentile=80, min_thresh=3, cmap=plt.cm.rainbow,
        )

        utils.figure_timestamp(result["fig"])

        result["fig"].text(
            0.005,
            0.005,
            outroot,
            ha="left",
            va="bottom",
            transform=result["fig"].transFigure,
            fontsize=8,
        )

        utils.figure_timestamp(result["img_fig"])

        result["img_fig"].text(
            0.5,
            0.005,
            outroot,
            ha="center",
            va="bottom",
            transform=result["img_fig"].transFigure,
            fontsize=8,
        )

        if write_cube_diagnostics:
            result["img_fig"].savefig(cube_file.replace(".fits", ".thumb.png"))

            result["fig"].savefig(cube_file.replace(".fits", ".1d.png"))

            result["stab"].write(
                cube_file.replace(".fits", ".1d.fits"), overwrite=True
            )
    else:
        result = {}

    result["outroot"] = outroot
    result["ptab"] = ptab
    result["cubes"] = cubes
    result["resp"] = resp

    return result
