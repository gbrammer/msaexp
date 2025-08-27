import os
import numpy as np
import subprocess

import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

from grizli import jwst_utils

jwst_utils.set_quiet_logging(50)

from grizli import utils

from .. import ifu
from .. import utils as msautils


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

    cube = msaexp.ifu.ExposureCube(
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
        fs_data = msaexp.ifu.ifu_sky_from_fixed_slit(
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
        s3_path = os.path.join(s3_prefix, visit).replace("//","/")
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
