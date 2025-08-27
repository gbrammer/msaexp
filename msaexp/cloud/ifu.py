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
]

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

    if not os.path.exists(file):
        mastquery.utils.download_from_mast([file])

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
            slit_names=["S200A1", "S200A2", "S400A1", "S200B1", "S1600A1"][:-1],
        )

        hdul = pyfits.HDUList(
            [pyfits.PrimaryHDU()]
            + [pyfits.BinTableHDU(fs_data[k]) for k in fs_data]
        )
        hdul.writeto(fs_file, overwrite=True)

    cube.fixed_slit = fs_data

    if sync:
        visit = os.path.basename(file.split("_")[0])
        s3_path = os.path.join(s3_prefix, visit).replace("//","/").replace("s3:/", "s3://")
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
                send_command = f"aws s3 cp {file_ext} {s3_path}/ --acl public-read"
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

    FIXED_SLIT_NAMES = ['S200A1', 'S200A2', 'S400A1', 'S1600A1', 'S200B1']

    fs_data = {}
    with pyfits.open(fs_file) as fs_im:
        for ext in fs_im[1:]:
            k = ext.header['FSNAME']
            fs_data[k] = utils.read_catalog(ext)

    tabs = []
    for i, k in enumerate(fs_data):
        if k not in FIXED_SLIT_NAMES:
            continue

        fsk = fs_data[k]
        fsk.meta = msaifu.meta_lowercase(fsk.meta)

        yp, xp = np.indices((fsk.meta['ysize'], fsk.meta['xsize']))
        xp += fsk.meta['xstart'] - 1
        yp += fsk.meta['ystart'] - 1

        # Like pixtab
        fsk['xpix'] = xp.flatten()
        fsk['ypix'] = yp.flatten()
        fsk['slice'] = slice_offset + FIXED_SLIT_NAMES.index(k)
        # fsk['xsky'] = 0.
        tabs.append(fsk[np.isfinite(fsk['wave'] + fsk['data'])])

    fs_ptab = astropy.table.vstack(tabs)
    fs_ptab.meta['nslits'] = len(tabs)

    for i, k in enumerate(fs_data):
        if k not in FIXED_SLIT_NAMES:
            continue

        slice_id = 50 + FIXED_SLIT_NAMES.index(k)
        fs_ptab.meta[f'slice{i}'] = slice_id
        fs_ptab.meta[f'slit{i}'] = k

    return fs_ptab

