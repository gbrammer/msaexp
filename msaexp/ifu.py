"""
Preliminary handling for NIRSpec IFU cubes
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage as nd
import jwst.datamodels

import astropy.table
import astropy.io.fits as pyfits

import scipy.stats
from scipy.optimize import minimize

from grizli import utils

from . import utils as msautils
from .utils import BAD_PIXEL_FLAG
from .slit_combine import pseudo_drizzle

IFU_BAD_PIXEL_FLAG = BAD_PIXEL_FLAG & ~1024 | 4096 | 1073741824 | 16777216

VERBOSITY = True

def rotation_matrix(angle):
    """
    Compute a 2x2 rotation matrix for an input ``angle`` in degrees
    """
    theta = angle / 180 * np.pi
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rot


class ExposureCube:
    input = None
    ptab = None

    def __init__(
        self, file="jw06595001001_02101_00001_nrs1_rate.fits", **kwargs
    ):
        """
        Handler for extracting cube data from a full-frame image
        """
        self.file = file

        self.process_pixel_table(**kwargs)

        if self.ptab is None:
            self.load_data(**kwargs)

            self.slice_data = slice_corners(self.input, **kwargs)

    def load_data(
        self, dilate_failed_open=True, force_reprocess=False, **kwargs
    ):
        """
        Load pixel data
        """
        if not os.path.exists(self.file.replace("_rate", "_cal")):
            self.preprocess(**kwargs)

        msg = f'cal file: {self.file.replace("_rate", "_cal")}'
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        self.input = jwst.datamodels.open(self.file.replace("_rate", "_cal"))

        if dilate_failed_open:
            self._dilate_failed_open_mask()

        self.position_angle = self.input.meta.aperture.position_angle
        self.ra_ref = self.input.meta.wcsinfo.ra_ref
        self.dec_ref = self.input.meta.wcsinfo.dec_ref
        self.target_ra = self.input.meta.target.proposer_ra
        self.target_dec = self.input.meta.target.proposer_dec
        if self.target_ra is None:
            self.target_ra = self.input.meta.target.ra
            self.target_dec = self.input.meta.target.dec

    def preprocess(
        self,
        do_flatfield=True,
        do_photom=True,
        extend_wavelengths=True,
        local_sflat=True,
        # run_oneoverf=True,
        # prism_oneoverf_rows=True,
        **kwargs,
    ):
        """
        Generate cal files with Spec1 steps
        """
        import astropy.io.fits as pyfits
        from astropy.utils.data import download_file

        msg = (
            f"Preprocess do_flatfield={do_flatfield} do_photom={do_photom} "
            + f" local_sflat={local_sflat} extend_wavelengths={extend_wavelengths}"
        )
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        from jwst.assign_wcs import AssignWcsStep
        from jwst.msaflagopen import MSAFlagOpenStep
        from jwst.flatfield import FlatFieldStep
        from jwst.photom import PhotomStep

        from grizli import jwst_utils
        from .pipeline_extended import assign_wcs_with_extended
        from . import utils as msautils

        det = detector_corrections(self.file, **kwargs)

        input = jwst.datamodels.open(self.file)
        _ = msautils.slit_hot_pixels(
            input, verbose=VERBOSITY, max_allowed_flagged=4096 * 8
        )

        if local_sflat & extend_wavelengths:
            inst_ = input.meta.instrument.instance
            sflat_file = "sflat_{grating}-{filter}_{detector}.fits".format(
                **inst_
            ).lower()

            if not os.path.exists(sflat_file):
                URL_ = (
                    "https://s3.amazonaws.com/msaexp-nirspec/ifu-sflat/"
                    + sflat_file
                )
                msg = f"Download SFLAT file from {URL_}"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

                sflat_file = download_file(URL_, cache=True)

            if os.path.exists(sflat_file):

                with pyfits.open(sflat_file) as sflat_:

                    outside_sflat = (~np.isfinite(sflat_[0].data)) & (
                        input.dq & 1 == 0
                    )
                    med_outside = np.nanmedian(input.data[outside_sflat])
                    msg = f"{self.file} SFLAT: {sflat_file} med={med_outside:.3f}"
                    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

                    input.data -= med_outside

                    input.data /= sflat_[0].data
                    input.err /= sflat_[0].data
                    input.var_rnoise /= sflat_[0].data ** 2
                    input.var_poisson /= sflat_[0].data ** 2
                    input.dq |= ((~np.isfinite(input.data)) * 1).astype(
                        input.dq.dtype
                    )

        if extend_wavelengths:
            input_wcs = assign_wcs_with_extended(input, **kwargs)
        else:
            input_wcs = AssignWcsStep().call(input)

        input_open = MSAFlagOpenStep().call(input_wcs)
        if do_flatfield:
            input_flat = FlatFieldStep().call(input_open)
        else:
            input_flat = input_open

        if do_photom:
            input_photom = PhotomStep().call(input_flat)
        else:
            input_photom = input_flat

        input_photom.to_fits(
            self.file.replace("_rate", "_cal"), overwrite=True
        )

    def _dilate_failed_open_mask(
        self,
    ):
        import scipy.ndimage as nd

        msg = "dilate failed open mask"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        _flag = jwst.datamodels.dqflags.pixel["MSA_FAILED_OPEN"]
        open_mask = self.input.dq & _flag > 0

        sh = (5, 48)
        footprint = np.ones(sh, dtype=int)
        footprint[:, : sh[1] // 4] = 0
        open_mask = nd.binary_dilation(open_mask, structure=footprint)
        self.input.dq |= (open_mask * _flag).astype(self.input.dq.dtype)

    @property
    def ref_coord(self):
        """
        Reference coordinate
        """
        if self.target_ra is None:
            return (self.ra_ref, self.dec_ref)
        else:
            return (self.target_ra, self.target_dec)

    def pixel_table(
        self,
        ref_coord=None,
        mask_params=(2, 6),
        image_attrs=["data", "dq", "var_poisson", "var_rnoise"],
        **kwargs,
    ):
        """
        Create a single pixel table from the cutouts
        """

        if ref_coord is None:
            ra_ref, dec_ref = self.ref_coord
        else:
            ra_ref, dec_ref = ref_coord

        _ = self.slice_data
        x_slice, y_slice, yslit_data, coord1_data, coord2_data, lam_data = _

        ptab = {}

        for c in image_attrs:
            ptab[c] = []

        coord_cols = ["slice", "wave", "ra", "dec", "xpix", "ypix"]
        for c in coord_cols:
            ptab[c] = []

        shapes = []
        ypix, xpix = np.indices(self.input.data.shape, dtype=np.int16)

        for i, l in enumerate(lam_data):
            slx = slice(*x_slice[i])
            sly = slice(*y_slice[i])

            # cut = input_photom.data[sly, slx]
            for c in image_attrs:
                ptab[c].append(getattr(self.input, c)[sly, slx].flatten())

            if mask_params is not None:
                msk = np.isfinite(self.input.data[sly, slx])
                msk = nd.binary_closing(msk, iterations=mask_params[0])
                shrink = nd.binary_erosion(
                    msk, structure=np.ones((mask_params[1], 1), dtype=bool)
                )
                ptab["dq"][i][~shrink.flatten()] |= 1024

            sh = coord1_data[i].shape
            shapes.append(sh)
            ptab["slice"].append(np.ones(sh, dtype=int).flatten() * i)
            ptab["wave"].append(lam_data[i].flatten())
            ptab["ra"].append(coord1_data[i].flatten())
            ptab["dec"].append(coord2_data[i].flatten())
            ptab["xpix"].append(xpix[sly, slx].flatten())
            ptab["ypix"].append(ypix[sly, slx].flatten())

        for c in ptab:
            ptab[c] = np.hstack(ptab[c])

        ptab = utils.GTable(ptab)
        ptab.meta["RA_REF"] = ra_ref
        ptab.meta["DEC_REF"] = dec_ref
        ptab.meta["NSLICE"] = len(shapes)
        ptab.meta["FILE"] = self.file

        # Save to YAML
        meta_to_yaml(self.input.meta)

        for meta in [
            self.input.meta.observation.instance,
            self.input.meta.target.instance,
            self.input.meta.instrument.instance,
            self.input.meta.exposure.instance,
            self.input.meta.aperture.instance,
            self.input.meta.pointing.instance,
            self.input.meta.wcsinfo.instance,
            self.input.meta.dither.instance,
            self.input.meta.cal_step.instance,
        ]:
            for k in meta:
                if k not in ptab.meta:
                    if k in ["observation_folder"]:
                        continue
                    ptab.meta[k] = meta[k]

        ptab.meta["bunit_data"] = self.input.meta.bunit_data
        ptab.meta["calver"] = self.input.meta.calibration_software_version

        # Trim NaN, e.g., from S-Flats
        valid_data = np.isfinite(ptab["data"] + ptab["var_rnoise"] + ptab["wave"])
        ptab = ptab[valid_data]

        # Slice info
        for i in range(len(shapes)):
            ptab.meta[f"XSTRT{i}"] = x_slice[i][0]
            ptab.meta[f"XSIZE{i}"] = shapes[i][1]
            ptab.meta[f"YSTRT{i}"] = y_slice[i][0]
            ptab.meta[f"YSIZE{i}"] = shapes[i][0]

        return ptab

    def process_pixel_table(self, load_existing=True, **kwargs):
        """ """
        ptab_file = self.file.replace("_rate", "_cal").replace(
            "_cal.fits", "_ptab.fits"
        )
        if os.path.exists(ptab_file) & load_existing:
            msg = f"process_pixel_table: load {ptab_file}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

            self.ptab = utils.read_catalog(ptab_file)
            self.ptab.meta = meta_lowercase(self.ptab.meta)
            self.target_ra = self.ra_ref = self.ptab.meta["ra_ref"]
            self.target_dec = self.dec_ref = self.ptab.meta["dec_ref"]
            self.position_angle = self.ptab.meta["position_angle"]

        elif self.input is not None:
            self.ptab = self.pixel_table(**kwargs)
            self.ptab.meta = meta_lowercase(self.ptab.meta)
            try:
                pixel_table_background(self.ptab, **kwargs)
            except:
                msg = f"pixel_table_background failed!"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
                # self.ptab["sky"] = 0

            self.ptab.write(ptab_file, overwrite=True)


def saturated_mask(files, ptab):
    """
    TBD: rework into a method if the mask doesn't use all of the exposures
    """
    import glob

    # Apply saturated mask to DQ
    sat_files = []
    for file in files:
        sat_files += glob.glob(file.replace("_cal", "_saturated"))

    sat_mask = np.array([np.zeros((2048, 2048), dtype=bool) for file in files])

    from grizli import jwst_utils

    bpix = msautils.cache_badpix_arrays()

    for j, file in enumerate(files):
        flag_sn, flag_dq, count = jwst_utils.flag_nirspec_hot_pixels(file)
        sat_mask[j, :, :] |= (
            flag_dq | bpix[file.split("_")[3].upper()][0]
        ) > 0
        msg = f"{file}  flagged {sat_mask[j,:,:].sum()}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    import skimage.morphology

    N = len(files) // 2  # For both detectors
    if len(sat_files) == len(files):
        for d in [0, 1]:
            for j in range(N):
                _file = sat_files[j * 2 + d]
                print(_file)
                with pyfits.open(_file) as sat:
                    sat_dil = skimage.morphology.isotropic_dilation(
                        sat[0].data > 0, radius=2
                    )
                    for k in range(j + 1, j + 3):
                        if k < N:
                            print("  >> ", sat_files[k * 2 + d])
                            sat_mask[k * 2 + d, :, :] |= sat_dil

    # apply mask
    for j in range(len(files)):
        ptab[j]["dq"] |= (
            sat_mask[j, :, :][ptab[j]["ypix"], ptab[j]["xpix"]] * 1024
        ).astype(ptab[j]["dq"].dtype)
        ovalid = ptab[j]["valid"].sum()
        ptab[j]["valid"] &= (ptab[j]["dq"] & 1024) == 0
        nvalid = ptab[j]["valid"].sum()
        print(f"{file}  {ovalid} > {nvalid}  ({ovalid - nvalid})")


def plot_cube_strips(ptabs, figsize=(10, 5), cmap="bone_r", **kwargs):
    """
    Plot spatial slices of a cube pixel table
    """
    from scipy.spatial import ConvexHull

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    image_attrs = ["data", "dq", "var_poisson", "var_rnoise"]

    for ptab in ptabs:
        dx, dy = pixel_table_dx_dy(ptab, **kwargs)

        un = utils.Unique(ptab["slice"], verbose=False)
        ok = np.isfinite(dx + dy)

        hull = ConvexHull(np.array([dx[ok], dy[ok]]).T)
        vert = hull.vertices

        # Slice coordinates
        ax = axes[0]
        pl = ax.plot(dx[ok][vert], dy[ok][vert], alpha=0.5, zorder=100)

        color = pl[0].get_color()

        for i, v in enumerate(un.values):
            ixi = un[v] & (
                np.abs(ptab["wave"] - np.nanmedian(ptab["wave"])) < 0.01
            )
            ax.scatter(dx[ixi], dy[ixi], alpha=0.1, color=color)

        # Cube data
        ax = axes[1]

        ax.plot(dx[ok][vert], dy[ok][vert], alpha=0.5, zorder=100, color=color)

        valid_dq = (
            ptab["dq"] & (jwst.datamodels.dqflags.pixel["MSA_FAILED_OPEN"] | 1)
        ) == 0

        for c in image_attrs:
            valid_dq &= np.isfinite(ptab[c])

        valid_dq &= ptab["var_poisson"] > 0
        valid_dq &= ptab["var_rnoise"] > 0

        test = np.isfinite(ptab["data"])
        test &= np.abs(ptab["wave"] - np.nanmedian(ptab["wave"])) < 0.02
        test &= valid_dq

        rsize = 0.0
        rx = np.random.rand(test.sum()) * rsize - rsize / 2
        ry = np.random.rand(test.sum()) * rsize - rsize / 2

        ax.scatter(
            dx[test] + rx,
            dy[test] + ry,
            c=ptab["data"][test],
            cmap=cmap,
            alpha=0.3,
            marker="s",
            vmin=-0.1,
            vmax=5.6,
            s=30,
            ec="None",
        )

    for ax in axes:
        ax.set_xlabel("dx (arcsec)")
        ax.grid()
        ax.set_aspect(1)

    axes[0].set_ylabel("dy (arcsec)")

    fig.tight_layout(pad=1)

    return fig


def detector_corrections(file, run_oneoverf=True, prism_oneoverf_rows=True, update_stats=True, skip_subarray=True, **kwargs):
    """
    Detector-level corrections on a rate file
    """
    from grizli import jwst_utils

    is_resized = msautils.resize_subarray_to_full(file, **kwargs)
    if is_resized & skip_subarray:
        return None

    do_corr = run_oneoverf

    with pyfits.open(file) as im:
        if "ONEFEXP" in im[0].header:
            if (im[0].header["ONEFEXP"]) & (run_oneoverf < 2):
                do_corr = False
        grating = im[0].header["GRATING"]
        filter = im[0].header["FILTER"]
        detector = im[0].header["DETECTOR"]
        rate_sci = im["SCI"].data * 1
        rate_err = im["ERR"].data * 1
        rate_dq = im["DQ"].data * 1

    sflat, sflat_mask = load_ifu_sflat(
        grating=grating, #"prism",
        filter=filter, #"clear",
        # detector="nrs1",
        detector=detector.lower(),
    )

    empty_mask = (
        (~sflat_mask) & (np.isfinite(rate_sci)) & (rate_dq & 1 == 0)
    )
    empty_mask &= rate_sci > -10 * rate_err

    if do_corr:

        jwst_utils.exposure_oneoverf_correction(
            file,
            erode_mask=False,
            in_place=True,
            axis=0,
            force_oneoverf=True,
            manual_mask=empty_mask,
            deg_pix=2048,
        )

        if (grating == "PRISM") & (prism_oneoverf_rows):
            jwst_utils.exposure_oneoverf_correction(
                file,
                erode_mask=False,
                in_place=True,
                axis=1,
                nirspec_prism_mask=False,
                manual_mask=empty_mask,
                force_oneoverf=True,
                deg_pix=2048,
            )

    # Reopen
    with pyfits.open(file) as im:
        if "ONEFEXP" in im[0].header:
            if (im[0].header["ONEFEXP"]) & (run_oneoverf < 2):
                do_corr = False
        grating = im[0].header["GRATING"]
        filter = im[0].header["FILTER"]
        detector = im[0].header["DETECTOR"]
        rate_sci = im["SCI"].data * 1
        rate_err = im["ERR"].data * 1
        rate_dq = im["DQ"].data * 1
        rate_var_rnoise = im["VAR_RNOISE"].data * 1
        rate_var_poisson = im["VAR_POISSON"].data * 1

    det_median = np.nanmedian(rate_sci[empty_mask])
    det_nmad = utils.nmad((rate_sci / np.sqrt(rate_var_rnoise))[empty_mask])

    msg = f"msaexp.ifu.detector_corrections: median {det_median:7.4f}  nmad {det_nmad:.3f}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    if update_stats:
        rate_sci -= det_median
        rate_var_rnoise *= det_nmad**2
        rate_err = np.sqrt(rate_var_rnoise + rate_var_poisson)

        with pyfits.open(file, mode="update") as im:
            if 'DETMED' in im[0].header:
                im[0].header['DETMED'] += det_median
            else:
                im[0].header['DETMED'] = (det_median, "Median from empty pixels")

            if 'DETNMAD' in im[0].header:
                im[0].header['DETNMAD'] *= det_nmad
            else:
                im[0].header['DETNMAD'] = (det_nmad, "RNOISE NMAD from empty pixels")

            im['SCI'].data -= det_median
            im['VAR_RNOISE'].data *= det_nmad**2
            im['ERR'].data = np.sqrt(im['VAR_RNOISE'].data + im['VAR_POISSON'].data)
            im.flush()

    res = {
        "file": file,
        "sci": rate_sci,
        "err": rate_err,
        "var_rnoise": rate_var_rnoise,
        "var_poisson": rate_var_poisson,
        "dq": rate_dq,
        "sflat": sflat,
        "sflat_mask": sflat_mask,
        "empty_mask": empty_mask,
    }

    return res


def objfun_scale_rnoise(theta, resid_num, resid_rnoise, resid_poisson):
    """
    objective function for scaling readnoise extension
    """
    try:
        theta_rn, theta_p = theta
    except:
        theta_rn, theta_p = theta, 1.0

    norm = scipy.stats.norm(
        scale=np.sqrt(theta_rn * resid_rnoise + theta_p * resid_poisson)
    )
    lnp = -2 * norm.logpdf(resid_num).sum()
    # print(theta_rn, theta_p, lnp)
    # print(theta, lnp)
    return lnp


def meta_lowercase(meta):
    """
    lowercase columns on metadata dictionary
    """
    out = {}
    for k in meta:
        out[k.lower()] = meta[k]

    return out


def meta_to_yaml(meta, yaml_file=None, **kwargs):
    """
    Save simple elements of a metadata object to a yaml file
    """
    import yaml

    mdict = meta_simpledict(meta, **kwargs)

    if yaml_file is None:
        yaml_file = mdict["filename"].replace(".fits", ".yaml")
        yaml_file = yaml_file.replace(".gz", "")

    msg = f"meta_to_yaml: {yaml_file}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    with open(yaml_file, "w") as fp:
        yaml.dump(mdict, fp)

    return mdict


def meta_simpledict(meta, exclude=["wcs"], **kwargs):
    """
    Get a simplified dictionary version of a metadata object
    """
    mdict = {}

    keys = meta.instance.keys()

    for k in keys:
        if k not in exclude:
            attr = getattr(meta, k)
            if hasattr(attr, "instance"):
                mdict[k] = attr.instance
            else:
                mdict[k] = attr

    return mdict


def load_ifu_sflat(
    grating="prism",
    filter="clear",
    detector="nrs1",
    dilate_iterations=4,
    **kwargs,
):
    """
    Load SFLAT for IFU data
    """
    import scipy.ndimage as nd
    from astropy.utils.data import download_file

    if (grating.lower() == "prism") & (detector.lower() == "nrs2"):
        # Doesn't exist
        sflat = np.zeros((2048, 2048))
        sflat[900:990, 400:950] = 1.
        sflat_mask = sflat > 0

        msg = f"msaexp.ifu.load_ifu_sflat: mask S200B1 for {grating} {filter} {detector}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        return sflat, sflat_mask

    sflat_file = f"sflat_{grating}-{filter}_{detector}.fits".lower()

    if sflat_file == "sflat_g395m-f290lp_nrs2.fits":
        return np.ones((2048, 2048)), np.zeros((2048, 2048), dtype=bool)

    msg = f"msaexp.ifu.load_ifu_sflat: {sflat_file}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    if not os.path.exists(sflat_file):
        URL_ = (
            "https://s3.amazonaws.com/msaexp-nirspec/ifu-sflat/" + sflat_file
        )
        msg = f"Download SFLAT file from {URL_}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        sflat_file = download_file(URL_, cache=True)

    with pyfits.open(sflat_file) as sflat_im:
        sflat = sflat_im[0].data

    sflat_mask = np.isfinite(sflat)
    if dilate_iterations > 0:
        sflat_mask = nd.binary_dilation(
            sflat_mask, iterations=dilate_iterations
        )

    return sflat, sflat_mask


def ifu_sky_from_fixed_slit(
    rate_file="jw02659003001_02101_00001_nrs1_rate.fits",
    slit_names=["S200A1", "S200A2", "S400A1"],
    **kwargs,
):
    """
    Estimate sky of IFU exposures using the Fixed Slits
    """
    from scipy.stats import binned_statistic
    from astropy.utils.data import download_file

    from jwst.extract_2d import Extract2dStep

    from .pipeline_extended import assign_wcs_with_extended

    _im = pyfits.open(rate_file)
    rate_sci = _im["SCI"].data * 1
    rate_err = _im["ERR"].data * 1
    rate_dq = _im["DQ"].data * 1
    rate_var_rnoise = _im["VAR_RNOISE"].data * 1
    rate_var_poisson = _im["VAR_POISSON"].data * 1

    ORIG_EXPTYPE = _im[0].header["EXP_TYPE"]
    fixed_slit = "S200A1"
    if ORIG_EXPTYPE != "NRS_FIXEDSLIT":
        _im[0].header["EXP_TYPE"] = "NRS_FIXEDSLIT"
        _im[0].header["APERNAME"] = f"NRS_{fixed_slit}_SLIT"
        _im[0].header["OPMODE"] = "FIXEDSLIT"
        _im[0].header["FXD_SLIT"] = fixed_slit

    gr = _im[0].header["GRATING"]
    if gr == "PRISM":
        _im[0].header["FILTER"] = "CLEAR"
    elif gr.startswith("G140"):
        _im[0].header["FILTER"] = "F100LP"
    elif gr.startswith("G235"):
        _im[0].header["FILTER"] = "F170LP"
    elif gr.startswith("G395"):
        _im[0].header["FILTER"] = "F290LP"

    with_wcs = assign_wcs_with_extended(_im, **kwargs)
    _im.close()
    ext2d = Extract2dStep().process(with_wcs)

    meta = meta_simpledict(ext2d.meta)["instrument"]
    grating = meta["grating"].upper()

    wave_grid = msautils.get_standard_wavelength_grid(grating, sample=1.05)
    wave_bin = msautils.array_to_bin_edges(wave_grid)

    sflat_file = "sflat_{grating}-{filter}_{detector}.fits".format(
        **meta
    ).lower()

    if not os.path.exists(sflat_file):
        URL_ = (
            "https://s3.amazonaws.com/msaexp-nirspec/ifu-sflat/" + sflat_file
        )
        msg = f"Download SFLAT file from {URL_}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        sflat_file = download_file(URL_, cache=True)

    with pyfits.open(sflat_file) as sflat_im:
        sflat = sflat_im[0].data

    imask = ~np.isfinite(sflat) & ((rate_dq & 1) == 0)
    rate_median = np.median(rate_sci[imask])
    rate_sci -= rate_median

    fs = {}

    for slit in ext2d.slits:
        if slit.name not in slit_names:
            continue

        msg = f"ifu_sky_from_fixed_slit: {rate_file} {slit.name}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        slx = slice(slit.xstart - 1, slit.xstart - 1 + slit.xsize)
        sly = slice(slit.ystart - 1, slit.ystart - 1 + slit.ysize)

        slit_data = (rate_sci / sflat)[sly, slx]
        # slit_err = (rate_err / sflat)[sly, slx]
        slit_var_rnoise = (rate_var_rnoise / sflat**2)[sly, slx]
        slit_var_poisson = (rate_var_poisson / sflat**2)[sly, slx]
        slit_dq = rate_dq[sly, slx]

        # xspl_int = np.interp(slit.wavelength.flatten(), wave_grid, xspl)
        #
        # xbspl = utils.bspline_templates(
        #     xspl_int, df=nspl, minmax=(0, 1), get_matrix=True
        # )
        #
        # Ax = xbspl.T
        # yx = slit_data.flatten()
        ok = np.isfinite((slit_data + slit.wavelength).flatten())  # & (yx > 0)
        ok &= (slit_dq.flatten() & (1 + 1024 + 4096)) == 0

        wavef = slit.wavelength.flatten()

        med = binned_statistic(
            wavef[ok],
            slit_data.flatten()[ok],
            statistic="median",
            bins=wave_bin,
        )

        dev = slit_data.flatten() - np.interp(
            wavef, wave_grid, med.statistic, left=np.nan, right=np.nan
        )

        mad = binned_statistic(
            wavef[ok], np.abs(dev)[ok], statistic=np.nanmedian, bins=wave_bin
        )

        # ount = binned_statistic(fs['wave'][fs['ok']], np.abs(dev)[fs['ok']]**0, statistic=np.nansum, bins=wave_bin)

        nmad = 1.48 * mad.statistic  # / np.sqrt(count.statistic)
        nmad_i = np.interp(wavef, wave_grid, nmad, left=np.nan, right=np.nan)

        valid = ok & np.isfinite(nmad_i) & (np.abs(dev) < nmad_i * 3)

        slit_ptab = utils.GTable()
        slit_ptab.meta["rate_file"] = rate_file
        slit_ptab.meta["ny"], slit_ptab.meta["nx"] = slit_data.shape
        slit_ptab.meta["fsname"] = slit.name
        slit_ptab.meta["datamed"] = rate_median
        for k in ["xstart", "xsize", "ystart", "ysize"]:
            slit_ptab.meta[k] = getattr(slit, k)

        for k in meta:
            slit_ptab.meta[k] = meta[k]

        slit_ptab["wave"] = slit.wavelength.flatten()
        slit_ptab["data"] = slit_data.flatten()
        slit_ptab["var_rnoise"] = slit_var_rnoise.flatten()
        slit_ptab["var_poisson"] = slit_var_poisson.flatten()
        # slit_ptab["err"] = slit_err.flatten()
        slit_ptab["dq"] = slit_dq.flatten()
        slit_ptab["ok"] = ok
        slit_ptab["valid"] = valid

        fs[slit.name] = slit_ptab

    return fs


def pixel_table_background(
    ptab,
    sky_center=(0, 0),
    sky_annulus=(0.5, 1.0),
    make_plot=True,
    skip_pixel_table_background=False,
    **kwargs,
):
    """
    Extract background spectrum from pixel table cube data
    """
    if skip_pixel_table_background:
        return False

    meta = meta_lowercase(ptab.meta)

    grating = meta["grating"].upper()

    wave_grid = msautils.get_standard_wavelength_grid(
        grating,
        sample=1.05,
    )

    if grating == "PRISM":
        nspl = 71
    else:
        nspl = 21

    nw = len(wave_grid)
    xspl = np.arange(0, nw) / nw

    bspl = utils.bspline_templates(
        xspl, df=nspl, minmax=(0, 1), get_matrix=True
    )

    if "valid" in ptab.colnames:
        valid_dq = ptab["valid"] & True
    else:
        valid_dq = pixel_table_valid_data(ptab, **kwargs)

    for c in ptab.colnames:
        valid_dq &= np.isfinite(ptab[c])

    # valid_dq &= ptab['var_poisson'] > 0
    valid_dq &= ptab["var_rnoise"] > 0

    poisson_threshold = np.nanpercentile(ptab["var_poisson"][valid_dq], 97)
    rnoise_threshold = np.nanpercentile(ptab["var_rnoise"][valid_dq], 97)

    valid_dq &= ptab["var_poisson"] < 100
    valid_dq &= ptab["var_rnoise"] < 100

    valid_dq &= np.isfinite(ptab["wave"])
    # valid_dq &= ptab["wave"] > 0.65

    # ptab["valid"] = valid_dq & True

    dx, dy = pixel_table_dx_dy(ptab)

    ###
    if sky_annulus is None:
        coord_test = valid_dq & True
    else:
        dr = np.sqrt(
            (dx - sky_center[0]) ** 2
            + (dy - sky_center[1]) ** 2
        )
        coord_test = valid_dq & (dr > sky_annulus[0])
        coord_test &= valid_dq & (dr < sky_annulus[1])

    if coord_test.sum() > 0:
        test = coord_test
    else:
        test = valid_dq & True

    test &= ptab["data"] > -6 * np.sqrt(ptab["var_rnoise"])
    test &= ptab["data"] < 10

    if grating.endswith("H"):
        test &= ptab["data"] < 20 * np.sqrt(ptab["var_rnoise"])

    if make_plot:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax = axes[0]

    ok = np.isfinite(ptab["wave"][test])

    var_total = ptab["var_rnoise"] + ptab["var_poisson"]

    for _iter in range(3):

        so = np.argsort(ptab["wave"][test][ok])
        xsky = ptab["wave"][test][ok][so]
        sky_med = nd.median_filter(ptab["data"][test][ok][so].astype(float), 64)
        if make_plot:
            ax.plot(ptab["wave"][test][ok][so], sky_med, alpha=0.5)

        sky_interp = np.interp(ptab["wave"][test], xsky, sky_med)
        ok = np.abs(ptab["data"][test] - sky_interp) < 5 * np.sqrt(var_total[test])
        ok &= var_total[test] < 8 * np.median(var_total[test][ok])
        msg = f"iter {_iter} {ok.sum()}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    ix_bad = np.where(test)[0][~ok]
    # ptab['valid'][ix_bad] = False

    skip = ok.sum() // 10000 + 1

    # Fit splines to the median filter
    xspl_int = np.interp(xsky, wave_grid, xspl)
    xbspl = utils.bspline_templates(
        xspl_int, df=nspl, minmax=(0, 1), get_matrix=True
    )
    Ax = xbspl.T
    yx = sky_med
    coeffs = np.linalg.lstsq(Ax.T, yx, rcond=None)
    splm = sky_med = xbspl.dot(coeffs[0])

    if make_plot:
        axes[0].plot(xsky, splm, color="yellow", alpha=0.5)

        ax.scatter(
            ptab["wave"][test][ok][::skip],
            ptab["data"][test][ok][::skip],
            alpha=0.1,
            zorder=-10,
        )
        ax.set_ylim(-1, 5)

    resid_num = ptab["data"][test][ok] - np.interp(
        ptab["wave"][test][ok], xsky, sky_med
    )
    resid_rnoise = ptab["var_rnoise"][test][ok]
    resid_poisson = ptab["var_poisson"][test][ok]
    resid = resid_num / np.sqrt(resid_rnoise + resid_poisson)

    res = minimize(
        objfun_scale_rnoise,
        x0=[1.0, 1.0],
        args=(resid_num, resid_rnoise, resid_poisson),
        method="bfgs",
        options={'eps': 1.e-2},
        tol=1.0e-6,
    )
    scale_rnoise = res.x[0]
    if len(res.x) > 1:
        scale_poisson = res.x[1]
    else:
        scale_poisson = 1.0

    msg = f"  uncertainty nmad={utils.nmad(resid):.3f} std={np.std(resid):.3f}  scale_rnoise={np.sqrt(scale_rnoise):.3f}  scale_poisson={np.sqrt(scale_poisson):.3f}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    ptab.meta["rnoise_nmad"] = utils.nmad(resid)
    ptab.meta["rnoise_std"] = np.std(resid)
    if "scale_rnoise" in ptab.meta:
        ptab.meta["scale_rnoise"] *= scale_rnoise
        ptab.meta["scale_poisson"] *= scale_poisson
    else:
        ptab.meta["scale_rnoise"] = scale_rnoise
        ptab.meta["scale_poisson"] = scale_poisson

    ptab["var_rnoise"] *= scale_rnoise
    ptab["var_poisson"] *= scale_poisson

    resid = resid_num / np.sqrt(resid_rnoise * scale_rnoise + resid_poisson * scale_poisson)

    if make_plot:
        ax = axes[1]
        ax.scatter(
            ptab["wave"][test][ok][::skip],
            resid[::skip],
            c=dy[test][ok][::skip],
            alpha=0.1,
            marker=".",
        )

    xspl_int = np.interp(ptab["wave"], wave_grid, xspl)

    xbspl = utils.bspline_templates(
        xspl_int, df=nspl, minmax=(0, 1), get_matrix=True
    )

    Ax = xbspl[test, :][ok, :].T / np.sqrt(ptab["var_rnoise"][test][ok])
    yx = ptab["data"][test][ok] / np.sqrt(ptab["var_rnoise"][test][ok])
    coeffs = np.linalg.lstsq(Ax.T, yx, rcond=None)
    splm = bspl.dot(coeffs[0])

    if make_plot:
        ax.set_ylim(-5, 5)
        axes[0].plot(wave_grid, splm, color="magenta", alpha=0.5, zorder=1000)
        fig.tight_layout(pad=1)

    sky = np.interp(ptab["wave"], wave_grid, splm)  #

    ptab["sky"] = np.interp(ptab["wave"], xsky, sky_med)

    return True


def slice_corners(
    input,
    coord_system="skyalign",
    slice_indices=range(30),
    slice_wavelength_range="auto",
    **kwargs,
):
    """Find the sky footprint of a slice of a NIRSpec exposure

    For each slice find:
    a. the min and max spatial coordinates (along slice, across slice) or
       (ra,dec) depending on coordinate system of the output cube.
    b. min and max wavelength

    Parameters
    ----------
    input: data model
       input model (or file)

    coord_system : str
       coordinate system of output cube: skyalign, ifualign, internal_cal

    slice_indices : list
        indices of the IFU slices to extract (0-30)

    slice_wavelength_range : [float, float], "auto"
        Wavelength range to extract for a particular slice.  If ``auto``, then determine from
        `msaexp.pipeline.extended.EXTENDED_RANGES` for the grating and filter of ``input``.

    Notes
    -----
    Returns
    -------
    min and max spatial coordinates and wavelength for slice.

    """
    from tqdm import tqdm
    from jwst.assign_wcs import nirspec
    from jwst.assign_wcs.util import wrap_ra
    from gwcs import wcstools
    from .pipeline_extended import EXTENDED_RANGES

    if slice_wavelength_range == "auto":
        inst_ = input.meta.instrument.instance
        gfilt = '{filter}_{grating}'.format(**inst_).upper()

        slice_wavelength_range = [w*1.e-6 for w in EXTENDED_RANGES[gfilt]]

        msg = f"slice_wavelength_range: auto {gfilt} {EXTENDED_RANGES[gfilt]}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    # nslices = 30
    nslices = len(slice_indices)

    # for NIRSPEC there are 30 regions
    # log.info('Looping over slices to determine cube size')

    x_slice = np.zeros((nslices, 2), dtype=int)
    y_slice = np.zeros((nslices, 2), dtype=int)

    yslit_data = []
    lam_data = []
    coord1_data = []
    coord2_data = []

    for ind, i in tqdm(enumerate(slice_indices)):
        slice_wcs = nirspec.nrs_wcs_set_input(
            input, i, wavelength_range=slice_wavelength_range
        )
        x, y = wcstools.grid_from_bounding_box(
            slice_wcs.bounding_box, step=(1, 1), center=True
        )

        x_slice[ind, :] = x.min(), x.max() + 1
        y_slice[ind, :] = y.min(), y.max() + 1

        #############
        # "slit" coordinate
        # coord1 = along slice
        # coord2 = across slice
        detector2slicer = slice_wcs.get_transform("detector", "slicer")
        coord2, coord1, lam = detector2slicer(
            x, y
        )  # lam ~0 for this transform
        gr = np.gradient(coord1, axis=0)
        sh = x.shape

        y0 = coord1[sh[0] // 2, sh[1] // 2]
        yslit = (coord1 - y0) / gr
        yslit_data.append(yslit)

        ###############
        # "sky" coordinate
        # coord1 = ra
        # coord2 = dec
        coord1, coord2, lam = slice_wcs(x, y)

        coord1_data.append(coord1 * 1)
        coord2_data.append(coord2 * 1)
        lam_data.append(lam * 1)

    return x_slice, y_slice, yslit_data, coord1_data, coord2_data, lam_data


def query_obsid_exposures(
    obsid="04056002001",
    grating="G395H",
    filter=None,
    download=True,
    exposure_type="cal",
    extend_wavelengths=True,
    detectors=None,
    fixed_slit=False,
    trim_prism_nrs2=True,
    extra_query=[],
    param_ranges=None,
    **kwargs,
):
    """
    Query mast for IFU exposures
    """
    import mastquery.jwst
    import mastquery.utils

    extensions = ["rate", "s2d", "s3d"]

    query = []
    query += mastquery.jwst.make_query_filter(
        "productLevel",
        values=["1", "1a", "1b", "2", "2a", "2b"],
    )

    if grating is not None:
        query += mastquery.jwst.make_query_filter("grating", values=[grating])
    if filter is not None:
        query += mastquery.jwst.make_query_filter("filter", values=[filter])

    query += mastquery.jwst.make_query_filter("obs_id", text=f"V{obsid}%")

    if not fixed_slit:
        query += mastquery.jwst.make_query_filter("is_imprt", values=["f"])

    if (detectors is None) & (grating is not None):
        if grating.upper() in ["G395M", "PRISM"]:
            detectors = ["nrs1"]
        elif (
            (grating.endswith("H") & (1))
            | (grating in ["G235M"])
            | extend_wavelengths
        ):
            detectors = ["nrs1", "nrs2"]
        else:
            detectors = ["nrs1"]

    if detectors is not None:
        query += mastquery.jwst.make_query_filter("detector", values=detectors)

    query += extra_query

    msg = f"QUERY = {query}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    res = mastquery.jwst.query_jwst(
        instrument="NRS",
        filters=query,
        extensions=extensions,
        rates_and_cals=False,
    )

    if len(res) == 0:
        return None, None

    # Unique rows
    rates = []
    unique_indices = []

    for i, u in enumerate(res["dataURI"]):
        ui = u.replace("s2d", exposure_type)
        for e in extensions:
            ui = ui.replace(e, exposure_type)

        if ui not in rates:
            unique_indices.append(i)

        rates.append(ui)

    res.remove_column("dataURI")
    res["dataURI"] = rates

    res = res[unique_indices]

    if param_ranges is not None:
        in_range = np.ones(len(res), dtype=bool)
        for p in param_ranges:
            if p in res.colnames:
                p_test = (res[p] >= param_ranges[p][0]) & (res[p] <= param_ranges[p][1])
                print(f'param_range: {p} = {param_ranges[p]}  {p_test.sum()} / {len(res)}')
                in_range &= p_test

        res = res[in_range]

    if (grating is None) & trim_prism_nrs2:
        prism_nrs2 = (res['grating'] == 'PRISM') & (res['detector'] == 'NRS2')
        prism_nrs2 &= (res['apername'] != 'NRS_S200B2_SLIT')

        if prism_nrs2.sum() > 0:
            msg = f'Remove {prism_nrs2.sum()} PRISM NRS2 exposures that will be empty'
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
            res = res[~prism_nrs2]

    msg = f"Found {len(res)} exposures"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    if download:
        mast = mastquery.utils.download_from_mast(res, **kwargs)
    else:
        mast = None

    return res, mast


def drizzle_cube_data(
    ptab,
    wave_sample=1.05,
    drizzle_wave_limits=None,
    pixel_size=0.1,
    pixfrac=0.75,
    side="auto",
    center=(0.0, 0.0),
    column=None,
    **kwargs,
):
    """
    Drizzle resample cube
    """
    from tqdm import tqdm

    dx, dy = pixel_table_dx_dy(ptab, **kwargs)

    if side in ["auto"]:
        sx = (int(np.nanmax(np.abs(dx)) / pixel_size) + 4) * pixel_size
        sy = (int(np.nanmax(np.abs(dy)) / pixel_size) + 4) * pixel_size
        side = [sx, sy]

    if not hasattr(side, "__len__"):
        sides = [side, side]
    else:
        sides = side

    Nside = [int(np.ceil(s / pixel_size)) for s in sides]
    xbin = np.arange(-Nside[0], Nside[0] + 1) * pixel_size  # - pixel_size / 2
    ybin = np.arange(-Nside[1], Nside[1] + 1) * pixel_size  # - pixel_size / 2

    wave_grid = msautils.get_standard_wavelength_grid(
        ptab.meta["grating"], sample=wave_sample
    )
    if drizzle_wave_limits is not None:
        wsub = (wave_grid >= drizzle_wave_limits[0]) & (wave_grid <= drizzle_wave_limits[1])
        wave_grid = wave_grid[wsub]

    wbin = msautils.array_to_bin_edges(wave_grid)

    nx = len(xbin) - 1
    ny = len(ybin) - 1

    msg = f"drizzle_cube_data: (NW, NY, NX) = ({len(wbin)-1}, {ny}, {nx})"
    msg += f"\ndrizzle_cube_data: {wave_grid[0]:.2f} - {wave_grid[-1]:.2f} um, wave_sample: {wave_sample:.2f}"
    msg += f"\ndrizzle_cube_data: {pixel_size:.2f} arcsec/pix ({sides[1]:.2f}, {sides[0]:.2f}) arcsec"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    num = np.zeros((len(wave_grid), ny, nx))
    den = np.zeros((len(wave_grid), ny, nx))
    vnum = np.zeros((len(wave_grid), ny, nx))

    var_total = ptab["var_rnoise"] + ptab["var_poisson"]

    for k in tqdm(range(len(wave_grid))):

        sub = (ptab["wave"] > wbin[k]) & (ptab["wave"] < wbin[k + 1])
        sub &= ptab["valid"]
        sub &= ptab["slice"] < 30

        if sub.sum() == 0:
            continue

        ysl = dy[sub] - center[1]
        xsl = dx[sub] - center[0]

        if column is None:
            if "sky" in ptab.colnames:
                data = (ptab["data"] - ptab["sky"])[sub] * 1
            else:
                data = ptab["data"][sub] * 1
        else:
            data = ptab[column][sub] * 1

        var = var_total[sub] * 1
        wht = 1.0 / ptab["var_rnoise"][sub] * 1

        dkws = dict(oversample=5, pixfrac=pixfrac * 1, sample_axes="xy")

        arrays = pseudo_drizzle(
            xsl,
            ysl,
            data,
            var,
            wht,
            xbin,
            ybin,
            **dkws,
        )

        num[k, :, :], vnum[k, :, :], den[k, :, :], ntot = arrays

    return num, den, vnum


def make_drizzle_hdul(
    ptab,
    num,
    den,
    vnum,
    wave_sample=1.05,
    drizzle_wave_limits=None,
    pixel_size=0.1,
    pixfrac=0.75,
    center=(0.0, 0.0),
    side="auto",
    **kwargs,
):
    """
    Drizzle pixel table into a rectified cube

    Parameters
    ----------
    ptab : `~astropy.table.Table`

    ... TBD

    """
    import astropy.wcs as pywcs
    
    dx, dy = pixel_table_dx_dy(ptab, **kwargs)

    if side in ["auto"]:
        sx = (int(np.nanmax(np.abs(dx)) / pixel_size) + 4) * pixel_size
        sy = (int(np.nanmax(np.abs(dy)) / pixel_size) + 4) * pixel_size
        side = [sx, sy]

    if not hasattr(side, "__len__"):
        sides = [side, side]
    else:
        sides = side

    Nside = [int(np.ceil(s / pixel_size)) for s in sides]
    xbin = np.arange(-Nside[0], Nside[0] + 1) * pixel_size  # - pixel_size / 2
    ybin = np.arange(-Nside[1], Nside[1] + 1) * pixel_size  # - pixel_size / 2

    wave_grid = msautils.get_standard_wavelength_grid(
        ptab.meta["grating"], sample=wave_sample
    )
    if drizzle_wave_limits is not None:
        wsub = (wave_grid >= drizzle_wave_limits[0]) & (wave_grid <= drizzle_wave_limits[1])
        wave_grid = wave_grid[wsub]

    wbin = msautils.array_to_bin_edges(wave_grid)

    meta = ptab.meta

    pa_aper = meta["position_angle"]

    hdul = utils.make_wcsheader(
        meta["ra_ref"],
        meta["dec_ref"],
        size=[2 * s * pixel_size for s in Nside],
        pixscale=pixel_size,
        get_hdu=True,
        theta=-pa_aper,
    )
    sky_wcs = pywcs.WCS(hdul.header, relax=True)

    hdul.header["NAXIS"] = 3

    hdul.header["CUNIT3"] = "Angstrom"

    # hdul.header["CD1_1"] *= -1
    # hdul.header["CD1_2"] *= -1

    if np.std(np.diff(wave_grid)) == 0:
        hdul.header["CTYPE3"] = "WAVE"
        hdul.header["CRVAL3"] = wave_grid[0] * 1.0e4
        hdul.header["CRPIX3"] = 1
        hdul.header["CD3_3"] = np.diff(wave_grid)[0] * 1.0e4
        hdul.header["CD1_3"] = 0.0
        hdul.header["CD2_3"] = 0.0
        hdul.header["CD3_1"] = 0.0
        hdul.header["CD3_2"] = 0.0

    else:
        hdul.header["CTYPE3"] = "WAVE-TAB"
        hdul.header["PS3_0"] = "WCS-TAB"
        hdul.header["PS3_1"] = "WAVELENGTH"

    wtab = utils.GTable()
    wtab["ROW"] = np.arange(len(wave_grid))
    wtab["WAVELENGTH"] = wave_grid * 1.0e4
    wave_hdu = pyfits.BinTableHDU(wtab, name="WCS-TAB")

    dcube = num / den
    # dcube[~np.isfinite(dcube)] = 0.0
    dvar = vnum / den**2

    hdul = pyfits.HDUList(
        [
            pyfits.PrimaryHDU(),
            pyfits.ImageHDU(
                data=dcube[:, :, :], name="SCI", header=hdul.header
            ),
            pyfits.ImageHDU(data=dvar, name="VAR", header=hdul.header),
            pyfits.ImageHDU(data=den, name="WHT", header=hdul.header),
            wave_hdu,
        ]
    )

    return hdul


def ifu_pipeline(
    obsid="04056002001",
    grating="G395H",
    output_suffix="",
    outroot=None,
    use_first_center=True,
    files=None,
    make_drizzled=True,
    perform_saturation=False,
    cumulative_saturation=True,
    **kwargs,
):
    """
    Full for pipeline for NIRSpec IFU processing

    Parameters
    ----------
    obsid : str
        Dataset observation id

    grating : string
        Grating name

    outroot : str, None
        Rootname of output file

    use_first_center : bool
        Flag to use the reference coordinate as the first exposure for all exposures

    make_drizzled : bool
        Make rectified resampled cube

    kwargs : dict
        Keyword arguments passed to all sub functions

    Returns
    -------
    outroot : string
        Output file rootname, calculated internally if input ``outroot = None``

    cubes : list
        List of `msaexp.ifu.ExposureCube` objects

    ptab : `~astropy.table.Table`
        Merged pixel table across all exposures

    hdul : `~astropy.fits.io.HDUList`
        Drizzled image HDU, if requested

    """
    if files is None:
        res, mast = query_obsid_exposures(
            obsid=obsid, grating=grating, **kwargs
        )
        if res is None:
            return None

        files = [os.path.basename(file) for file in res["dataURI"]]
        files.sort()
    else:
        obsid = files[0][2:13]
        with pyfits.open(files[0]) as im:
            grating = im[0].header["GRATING"]

    msg = f"ifu_pipeline: file={files[0]}  {obsid} {grating}"
    utils.LOGFILE = f"cube-{obsid}-{grating}{output_suffix}.log.txt".lower()
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    # Initialize
    cubes = []
    for file in files:
        cube = ExposureCube(file, **kwargs)
        cubes.append(cube)

        if cube.ptab is None:
            # Shorten for FITS header
            try:
                desc = cube.input.meta.target.description.split("; ")
                cube.input.meta.target.description = "; ".join(desc[:2])
            except AttributeError:
                pass

            if use_first_center & ("ref_coord" not in kwargs):
                kwargs["ref_coord"] = (cubes[0].target_ra, cubes[0].target_dec)

            cube.process_pixel_table(**kwargs)
            plt.close("all")
        else:
            if use_first_center & ("ref_coord" not in kwargs):
                kwargs["ref_coord"] = (cubes[0].target_ra, cubes[0].target_dec)

    if "ref_coord" in kwargs:
        for cube in cubes:
            cube.ptab.meta["ra_ref"], cube.ptab.meta["dec_ref"] = kwargs["ref_coord"]

    #### TBD: saturated mask when running on cal files

    # Set exposure index for full table
    for i, cube in enumerate(cubes):
        cube.ptab["exposure"] = i

    # Saturated mask
    sat_files, sat_data = [], []
    for file in files:
        sat_file, sat_i = msautils.exposure_ramp_saturation(
            file, perform=perform_saturation, **kwargs
        )
        if sat_i is None:
            sat_i = np.zeros((2048, 2048), dtype=bool)

        sat_files.append(sat_file)
        sat_data.append(sat_i)

    sat_det = {}
    for i, cube in enumerate(cubes):
        this_sat = sat_data[i]
        det = cube.ptab.meta["detector"]
        if det in sat_det:
            sat_mask = this_sat | sat_det[det]
        else:
            sat_mask = this_sat

        ptab_sat = sat_mask[cube.ptab["ypix"], cube.ptab["xpix"]]

        # print("xxx", files[i], ptab_sat.sum())

        cube.ptab["dq"] |= (ptab_sat * 4096).astype(cube.ptab["dq"].dtype)
        if cumulative_saturation:
            # Full cumulative mask
            sat_det[det] = sat_mask
        else:
            # Previous exposure
            sat_det[det] = this_sat

    ptabs = [cube.ptab for cube in cubes]

    try:
        SOURCE = ptabs[0].meta["proposer_name"].lower().replace(" ", "_")
    except:
        SOURCE = "indef"

    if outroot is None:
        outroot = f"cube-{obsid}_{grating}-{ptabs[0].meta['filter']}_{SOURCE}{output_suffix}".lower()

    strip_fig = plot_cube_strips(ptabs)
    strip_fig.text(
        0.5,
        0.005,
        outroot,
        ha="center",
        va="bottom",
        transform=strip_fig.transFigure,
        fontsize=8,
    )
    utils.figure_timestamp(strip_fig)

    strip_fig.savefig(f"{outroot}.strips.png")

    # Full ptab cube
    ptab = astropy.table.vstack(ptabs)
    ptab.meta["srcname"] = SOURCE
    ptab.meta["obsid"] = obsid
    if "proposer_ra" in ptabs[0].meta:
        ptab.meta["ra_ref"] = ptabs[0].meta["proposer_ra"]
        ptab.meta["dec_ref"] = ptabs[0].meta["proposer_dec"]

    ptab.meta["nfiles"] = len(files)
    for i, file_ in enumerate(files):
        ptab.meta[f"file{i:04d}"] = file_

    ptab["valid"] = pixel_table_valid_data(ptab, **kwargs)

    if make_drizzled:
        num, den, vnum = drizzle_cube_data(ptab, **kwargs)
        hdul = make_drizzle_hdul(ptab, num, den, vnum, **kwargs)

        for ext in range(3):
            hdul[ext].header["SRCNAME"] = SOURCE
            hdul[ext].header["NFILES"] = len(files)
            for i, file_ in enumerate(files):
                hdul[ext].header[f"FILE{i:04d}"] = file_

        cube_file = f"{outroot}.fits"
        msg = f"cube_file: {cube_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        hdul.writeto(cube_file, overwrite=True)
    else:
        hdul = None

    return outroot, cubes, ptab, hdul


def rerun_drizzle(ptab, output_suffix="", **kwargs):
    """
    Rerun drizzle cube steps
    """
    num, den, vnum = drizzle_cube_data(ptab, **kwargs)
    hdul = make_drizzle_hdul(ptab, num, den, vnum, **kwargs)

    outroot = "cube-{obsid}_{grating}-{filter}_{srcname}{output_suffix}".format(
        output_suffix=output_suffix, **ptab.meta
    ).lower()

    for ext in range(3):
        hdul[ext].header["SRCNAME"] = ptab.meta["srcname"]
        for i in range(100):
            k = f"FILE{i:04d}"
            if k.lower() in ptab.meta:
                hdul[ext].header[k] = ptab.meta[k.lower()]
            else:
                break

    cube_file = f"{outroot}.fits"
    msg = f"cube_file: {cube_file}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    hdul.writeto(cube_file, overwrite=True)
    return cube_file, hdul


def pixel_table_to_fits(
    ptab,
    columns=[
        "data",
        "dq",
        "var_poisson",
        "var_rnoise",
        "sky",
        "slice",
        "dx",
        "dy",
        "ra",
        "dec",
        "wave",
    ],
    **kwargs,
):
    """
    Make full HDUList
    """
    hdul = [pyfits.PrimaryHDU()]
    for c in columns:
        if c in ptab.colnames:
            hdul.append(pixel_table_to_detector(ptab, column=c, as_hdu=True, **kwargs))

    hdul = pyfits.HDUList(hdul)

    return hdul


def pixel_table_valid_data(ptab, low_threshold=-4, bad_pixel_flag=IFU_BAD_PIXEL_FLAG, **kwargs):
    """
    Determine "valid" pixels in a pixel table with optional bad pixel flagging

    Parameters
    ----------
    ptab : Table
        Pixel table

    Returns
    -------
    valid : array-like
        Boolean array True where data are determined to be valid

    """
    valid = np.isfinite(ptab["wave"] + ptab["data"])

    if bad_pixel_flag is not None:
        msg = f'Update "valid" with bad_pixel_flag = {bad_pixel_flag}'
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        ptab.meta["bad_pixel_flag"] = bad_pixel_flag
        valid &= (ptab["dq"] & bad_pixel_flag) == 0

    if low_threshold is not None:
        var_total = ptab["var_poisson"] + ptab["var_rnoise"]
        bad_low = ptab["data"] < low_threshold * np.sqrt(var_total)
        msg = (
            f"Pixels below {low_threshold} x sqrt(var_total): {bad_low.sum()}"
        )
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
        valid &= ~bad_low

    return valid


def pixel_table_to_detector(
    ptab, column="data", split_exposures=True, as_hdu=False, **kwargs
):
    """
    Map pixel table back to detector frame

    Parameters
    ----------
    ptab : `~astropy.table.Table`
        Pixel table

    column : str
        Data column to use

    as_hdu : bool
        Output as fits HDU or simple array

    Returns
    -------
    hdu, detector_array : `~astropy.io.fits.ImageHDU`, `np.array`
        2048 x 2048 array data

    """

    if split_exposures & ("exposure" in ptab.colnames):
        uexp = utils.Unique(ptab["exposure"], verbose=False)
        nexp = np.max(uexp.values) + 1
    else:
        nexp = 1

    detector_array = np.zeros((nexp, 2048, 2048), dtype=ptab[column].dtype)
    if nexp > 1:
        for exp_index in uexp.values:
            mask = uexp[exp_index]
            detector_array[
                exp_index, ptab["ypix"][mask], ptab["xpix"][mask]
            ] = ptab[column][mask]
    else:
        detector_array[0, ptab["ypix"], ptab["xpix"]] = ptab[column]
        detector_array = detector_array[0, :, :]

    if as_hdu:
        hdu = pyfits.ImageHDU(
            data=detector_array, header=pyfits.Header(ptab.meta)
        )
        hdu.header["EXTNAME"] = column.upper()
        return hdu
    else:
        return detector_array


def pixel_table_dx_dy(ptab, ref_coords=None, wcs=None, **kwargs):
    """
    Parameters
    ----------
    ptab : Table
        IFU cube pixel table with (at least) columns ``ra``, ``dec`` and attribute
        ``meta["position_angle"]``

    ref_coords : (float, float), None
        Reference decimal ra, dec.  If not specified, then get from ``ra_ref``, ``dec_ref``
        entries in the metadata

    Returns
    -------
    dx, dy : array-like
        Offset coordinates in arcsec relative to the reference center

    """
    if wcs is not None:
        dx, dy = wcs.all_world2pix(ptab["ra"], ptab["dec"], 0)
        return dx, dy

    rot = rotation_matrix(ptab.meta["position_angle"])

    if ref_coords is None:
        ra_ref = ptab.meta["ra_ref"]
        dec_ref = ptab.meta["dec_ref"]
    else:
        ra_ref, dec_ref = ref_coords

    dra = (
        -(ptab["ra"] - ra_ref)
        * np.cos(dec_ref / 180 * np.pi)
        * 3600
    )
    dde = (ptab["dec"] - dec_ref) * 3600

    dx, dy = np.array([dra, dde]).T.dot(rot).T

    return dx, dy

def run_dja_pipeline(obsid="02659003001", gfilt="CLEAR_PRISM", **kwargs):
    """
    Run full pipeline for DJA
    """
    from .pipeline_extended import EXTENDED_RANGES
    import yaml

    slice_wrange = [w * 1.0e-6 for w in EXTENDED_RANGES[gfilt]]

    filter, grating = gfilt.split("_")

    params = dict(
        outroot=None,
        pixel_size=0.08,
        pixfrac=0.75,
        side="auto",
        wave_sample=1.05,
        drizzle_wave_limits=None,
        files=None,
        obsid=obsid,
        filter=filter,
        grating=grating,
        download=True,
        use_token=False,
        sky_annulus=None,
        exposure_type="rate",
        do_flatfield=False,
        do_photom=False,
        extend_wavelengths=True,
        use_first_center=True,
        slice_wavelength_range=slice_wrange,  # [0.5e-6, 5.6e-6],
        make_drizzled=True,
        bad_pixel_flag=(
            msautils.BAD_PIXEL_FLAG & ~1024 | 4096 | 1073741824 | 16777216
        ),
        perform_saturation=True,
        cumulative_saturation=True,
        output_suffix="",
    )

    for k in kwargs:
        params[k] = kwargs[k]

    yaml_file = f"cube-{obsid}-{grating}{params['output_suffix']}.params.yaml".lower()
    with open(yaml_file, "w") as fp:
        yaml.dump(params, fp)

    outroot, cubes, ptab, hdul = ifu_pipeline(**params)

    return outroot, cubes, ptab, hdul
