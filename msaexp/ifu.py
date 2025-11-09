"""
Preliminary handling for NIRSpec IFU cubes
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import scipy.ndimage as nd
import jwst.datamodels

import astropy.table
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u
from astropy.visualization import simple_norm

import scipy.stats
from scipy.optimize import minimize

from grizli import utils

from . import utils as msautils
from .utils import BAD_PIXEL_FLAG
from .slit_combine import pseudo_drizzle
from .version import __version__ as msaexp_version

IFU_BAD_PIXEL_FLAG = BAD_PIXEL_FLAG & ~1024 | 4096 | 1073741824 | 16777216

VERBOSITY = True

LINE_LABELS_LATEX = {
    "Ha": r"H$\alpha$",
    "Hb": r"H$\beta$",
    "Hg": r"H$\gamma$",
    "Hd": r"H$\delta$",
    "SII": r"[SII]$\lambda\lambda$6717,6731",
    "SIII": r"[SIII]$\lambda\lambda$9068,9531",
    "SIII-9531": r"[SIII]$\lambda$9531",
    "NII": r"[NII]$\lambda\lambda$6548,6584",
    "OII": r"[OII]$\lambda$3727",
    "OIII": r"[OIII]$\lambda$5007",
    "OIII-5007": r"[OIII]$\lambda$5007",
    "OIII-4363": r"[OIII]$\lambda$4363",
    "PaA": r"Pa$\alpha$",
    "PaB": r"Pa$\beta$",
    "PaG": r"Pa$\gamma$",
    "PaD": r"Pa$\delta$",
}

LINE_WAVELENGTHS, LINE_RATIOS = utils.get_line_wavelengths()


def linelist_velocity_offset(lines):
    """
    Get a list of velocity offsets for a list of emission line names
    """

    ref_wave = LINE_WAVELENGTHS[lines[0]][0]
    rows = []
    for li in lines:
        for j, wj in enumerate(LINE_WAVELENGTHS[li]):
            row = {
                "name": li,
                "index": j,
                "wave": wj,
                "dv": int(np.round((wj - ref_wave) / ref_wave * 3.0e5)),
            }
            rows.append(row)

    tab = utils.GTable(rows=rows)
    tab.meta["ref_wave"] = ref_wave
    tab.meta["min_dv"] = tab["dv"].min()
    tab.meta["max_dv"] = tab["dv"].max()

    msg = "line_list_velocity_differences: {lines}  {min_dv}  {max_dv}".format(
        lines=lines, **tab.meta
    )
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    return tab


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
        ptab.meta["MSAEXPV"] = msaexp_version

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
        valid_data = np.isfinite(
            ptab["data"] + ptab["var_rnoise"] + ptab["wave"]
        )
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


def detector_corrections(
    file,
    run_oneoverf=True,
    prism_oneoverf_rows=True,
    update_stats=True,
    skip_subarray=True,
    **kwargs,
):
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
        grating=grating,  # "prism",
        filter=filter,  # "clear",
        # detector="nrs1",
        detector=detector.lower(),
    )

    empty_mask = (~sflat_mask) & (np.isfinite(rate_sci)) & (rate_dq & 1 == 0)
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
            if "DETMED" in im[0].header:
                im[0].header["DETMED"] += det_median
            else:
                im[0].header["DETMED"] = (
                    det_median,
                    "Median from empty pixels",
                )

            if "DETNMAD" in im[0].header:
                im[0].header["DETNMAD"] *= det_nmad
            else:
                im[0].header["DETNMAD"] = (
                    det_nmad,
                    "RNOISE NMAD from empty pixels",
                )

            im["SCI"].data -= det_median
            im["VAR_RNOISE"].data *= det_nmad**2
            im["ERR"].data = np.sqrt(
                im["VAR_RNOISE"].data + im["VAR_POISSON"].data
            )
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
        sflat[900:990, 400:950] = 1.0
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
    rescale_noise=False,
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
        dr = np.sqrt((dx - sky_center[0]) ** 2 + (dy - sky_center[1]) ** 2)
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

    if "scale_rnoise" not in ptab.meta:
        ptab.meta["scale_rnoise"] = 1.0
        ptab.meta["scale_poisson"] = 1.0

    var_total = (
        ptab["var_rnoise"] * ptab.meta["scale_rnoise"]
        + ptab["var_poisson"] * ptab.meta["scale_poisson"]
    )

    for _iter in range(3):

        so = np.argsort(ptab["wave"][test][ok])
        xsky = ptab["wave"][test][ok][so]
        sky_med = nd.median_filter(
            ptab["data"][test][ok][so].astype(float), 64
        )
        # if make_plot:
        #     ax.plot(ptab["wave"][test][ok][so], sky_med, alpha=0.5)

        sky_interp = np.interp(ptab["wave"][test], xsky, sky_med)
        ok = np.abs(ptab["data"][test] - sky_interp) < 5 * np.sqrt(
            var_total[test]
        )
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

        axes[0].scatter(
            ptab["wave"][test][ok][::skip],
            ptab["data"][test][ok][::skip],
            alpha=0.1,
            zorder=-10,
        )
        ax.set_ylim(-1, 5)

    resid_num = ptab["data"][test][ok] - np.interp(
        ptab["wave"][test][ok], xsky, sky_med
    )
    resid_rnoise = ptab["var_rnoise"][test][ok] * ptab.meta["scale_rnoise"]
    resid_poisson = ptab["var_poisson"][test][ok] * ptab.meta["scale_poisson"]
    resid = resid_num / np.sqrt(resid_rnoise + resid_poisson)

    if rescale_noise:
        res = minimize(
            objfun_scale_rnoise,
            x0=[1.0],
            args=(resid_num, resid_rnoise, resid_poisson),
            method="bfgs",
            options={"eps": 1.0e-2},
            tol=1.0e-6,
        )
        scale_rnoise = res.x[0]
        if len(res.x) > 1:
            scale_poisson = res.x[1]
        else:
            scale_poisson = 1.0

        ptab.meta["scale_rnoise"] *= scale_rnoise
        ptab.meta["scale_poisson"] *= scale_poisson
    else:
        scale_rnoise = 1.0
        scale_poisson = 1.0

    msg = f"  uncertainty nmad={utils.nmad(resid):.3f} std={np.std(resid):.3f}  scale_rnoise={np.sqrt(scale_rnoise):.3f}  scale_poisson={np.sqrt(scale_poisson):.3f}"
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    ptab.meta["rnoise_nmad"] = utils.nmad(resid)
    ptab.meta["rnoise_std"] = np.std(resid)

    # ptab["var_rnoise"] *= scale_rnoise
    # ptab["var_poisson"] *= scale_poisson

    resid = resid_num / np.sqrt(
        resid_rnoise * scale_rnoise + resid_poisson * scale_poisson
    )

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
        ymax = np.nanpercentile(splm, 90)
        axes[0].set_ylim(-0.2 * ymax, 1.5 * ymax)

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
        gfilt = "{filter}_{grating}".format(**inst_).upper()

        slice_wavelength_range = [w * 1.0e-6 for w in EXTENDED_RANGES[gfilt]]

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
                p_test = (res[p] >= param_ranges[p][0]) & (
                    res[p] <= param_ranges[p][1]
                )
                print(
                    f"param_range: {p} = {param_ranges[p]}  {p_test.sum()} / {len(res)}"
                )
                in_range &= p_test

        res = res[in_range]

    if (grating is None) & trim_prism_nrs2:
        prism_nrs2 = (res["grating"] == "PRISM") & (res["detector"] == "NRS2")
        prism_nrs2 &= res["apername"] != "NRS_S200B2_SLIT"

        if prism_nrs2.sum() > 0:
            msg = f"Remove {prism_nrs2.sum()} PRISM NRS2 exposures that will be empty"
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
        wsub = (wave_grid >= drizzle_wave_limits[0]) & (
            wave_grid <= drizzle_wave_limits[1]
        )
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

    if "scale_rnoise" in ptab.meta:
        scale_rnoise = ptab.meta["scale_rnoise"]
        scale_poisson = ptab.meta["scale_poisson"]
    else:
        scale_rnoise = 1.0
        scale_poisson = 1.0

    var_total = (
        ptab["var_rnoise"] * scale_rnoise + ptab["var_poisson"] * scale_poisson
    )

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
        wht = 1.0 / (ptab["var_rnoise"][sub] * scale_rnoise)

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
        wsub = (wave_grid >= drizzle_wave_limits[0]) & (
            wave_grid <= drizzle_wave_limits[1]
        )
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

    if np.std(np.diff(wave_grid)) < 1.0e-9:
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

    hdul.header["NSIDE0"] = Nside[0]
    hdul.header["NSIDE1"] = Nside[1]
    hdul.header["PIXSIZE"] = pixel_size
    hdul.header["PIXFRAC"] = pixfrac
    hdul.header["WAVESAMP"] = wave_sample

    if drizzle_wave_limits is not None:
        hdul.header["WAVEMIN"] = drizzle_wave_limits[0]
        hdul.header["WAVEMAX"] = drizzle_wave_limits[1]

    wtab = utils.GTable()
    wtab["ROW"] = np.arange(len(wave_grid))
    wtab["WAVELENGTH"] = wave_grid * 1.0e4
    wave_hdu = pyfits.BinTableHDU(wtab, name="WCS-TAB")

    dcube = num / den
    # dcube[~np.isfinite(dcube)] = 0.0
    dvar = vnum / den**2

    for k in meta:
        if k.lower() in ["spectral_region"]:
            continue
        elif k.upper() in hdul.header:
            continue

        hdul.header[k] = meta[k]

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
    exclude_self_saturation=True,
    recenter_cube=False,
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
    from scipy.spatial import ConvexHull

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
        if grating is None:
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
            cube.ptab.meta["ra_ref"], cube.ptab.meta["dec_ref"] = kwargs[
                "ref_coord"
            ]

    #### TBD: saturated mask when running on cal files

    # Set exposure index for full table
    for i, cube in enumerate(cubes):
        cube.ptab["exposure"] = i

    msg = (
        f"exposure_ramp_saturation: perform_saturation={perform_saturation} "
        + f"cumulative_saturation={cumulative_saturation} "
        + f"exclude_self_saturation={exclude_self_saturation}"
    )
    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

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
        if exclude_self_saturation:
            ptab_sat ^= this_sat[cube.ptab["ypix"], cube.ptab["xpix"]]

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

    # Full ptab cube
    ptab = astropy.table.vstack(ptabs)
    ptab.meta["srcname"] = SOURCE
    ptab.meta["obsid"] = obsid
    if "proposer_ra" in ptabs[0].meta:
        ptab.meta["ra_ref"] = ptabs[0].meta["proposer_ra"]
        ptab.meta["dec_ref"] = ptabs[0].meta["proposer_dec"]

    if recenter_cube:
        oref = ptab.meta["ra_ref"] * 1, ptab.meta["dec_ref"] * 1

        coo_ = np.array([ptab["ra"], ptab["dec"]])
        hull_ = ConvexHull(coo_.T)
        vertices_ = coo_[:, hull_.vertices]
        sr_ = utils.SRegion(vertices_)
        center_ = np.squeeze(sr_.shapely[0].centroid.xy)

        ptab.meta["ra_ref"] = center_[0]
        ptab.meta["dec_ref"] = center_[1]

        msg = f"recenter_cube: ({oref[0]:>.6f}, {oref[1]:.6f}) -> "
        msg += f'({ptab.meta["ra_ref"]:>.6f}, {ptab.meta["dec_ref"] :.6f})'
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

    strip_fig = plot_cube_strips(
        ptabs, ref_coords=(ptab.meta["ra_ref"], ptab.meta["dec_ref"])
    )

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

    outroot = (
        "cube-{obsid}_{grating}-{filter}_{srcname}{output_suffix}".format(
            output_suffix=output_suffix, **ptab.meta
        ).lower()
    )

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
            hdul.append(
                pixel_table_to_detector(ptab, column=c, as_hdu=True, **kwargs)
            )

    hdul = pyfits.HDUList(hdul)

    return hdul


def pixel_table_valid_data(
    ptab, low_threshold=-4, bad_pixel_flag=IFU_BAD_PIXEL_FLAG, **kwargs
):
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
        if "scale_rnoise" in ptab.meta:
            scale_rnoise = ptab.meta["scale_rnoise"]
            scale_poisson = ptab.meta["scale_poisson"]
        else:
            scale_rnoise = 1.0
            scale_poisson = 1.0

        var_total = (
            ptab["var_poisson"] * scale_poisson
            + ptab["var_rnoise"] * scale_rnoise
        )
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
        if "spectral_region" in ptab.meta:
            _ = ptab.meta.pop("spectral_region")

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

    dra = -(ptab["ra"] - ra_ref) * np.cos(dec_ref / 180 * np.pi) * 3600
    dde = (ptab["dec"] - dec_ref) * 3600

    dx, dy = np.array([dra, dde]).T.dot(rot).T

    return dx, dy


def pixel_table_wave_step(ptab):
    """
    Get the gradient dlam / dx of the wavelength array in a pixel table needed for resampling models

    Parameters
    ----------
    ptab : Table
        Pixel table with at least ``wave``, ``xpix``, ``ypix`` columns.  If a "slice" column is found,
        the gradient will be calculated by slice, e.g., for the IFU slices.

    Returns
    -------
    wave_step : array-like
        Wavelength gradient
    """
    # Put wavelength and slice arrays back into 2D "detector" arrays
    wave = pixel_table_to_detector(
        ptab, column="wave", split_exposures=True, as_hdu=False
    )

    wave[wave == 0] = np.nan
    wave = np.nanmean(wave, axis=0)

    dwave = np.gradient(wave, axis=1)

    valid = np.isfinite(wave)
    valid = nd.binary_erosion(valid, iterations=2)

    if "slice" in ptab.colnames:
        sli = pixel_table_to_detector(
            ptab, column="slice", split_exposures=True, as_hdu=False
        )
        sli = np.nanmax(sli, axis=0)
    else:
        sli = np.zeros(valid.shape, dtype=int)

    sli = sli[valid]
    wave = wave[valid]
    dwave = dwave[valid]

    slices = utils.Unique(sli, verbose=False)
    if "slice" in ptab.colnames:
        ptab_slices = utils.Unique(ptab["slice"], verbose=False)
    else:
        ptab_slices = [np.isfinite(ptab["wave"])]

    # Polynomial fit:
    # dwave**dlp = P(wave**wp) works well for the prism
    wp = -1
    dlp = -1
    degree = 11

    wave_step = np.zeros_like(ptab["wave"])

    for sli in slices.values:
        c = np.polyfit(
            wave[slices[sli]] ** wp, dwave[slices[sli]] ** dlp, degree
        )

        wave_step[ptab_slices[sli]] = (
            np.polyval(c, ptab["wave"][ptab_slices[sli]] ** wp) ** dlp
        )

    return wave_step


def pixel_table_xwave(ptab, wave_sample=1.05):
    """
    Get linear coordinate along "wave" axis

    Parameters
    ----------
    ptab : Table
        Pixel table with at least ``wave`` column and ``grating`` metadata entry.

    Returns
    -------
    wave_grid : array-like
        nominal grating wavelength grid from `~msaexp.utils.get_standard_wavelength_grid``

    wave_xgrid : array-like
        Normalized grid from (0, 1)

    wx : array-like
        Wavelength coordinate of ``ptab["wave"]`` relative to the nominal grating wavelength grid

    """
    wave_grid = msautils.get_standard_wavelength_grid(
        ptab.meta["grating"], sample=wave_sample
    )

    NW = len(wave_grid)
    wave_xgrid = np.linspace(0, 1, NW)

    wx = np.interp(
        ptab["wave"], wave_grid, wave_xgrid, left=np.nan, right=np.nan
    )

    return wave_grid, wave_xgrid, wx


def pixel_table_sensitivity(
    ptab, prefix="msaexp_sensitivity", version="001", **kwargs
):
    """ """
    path_to_data = os.path.join(
        os.path.dirname(__file__), "data/extended_sensitivity"
    )

    grating = ptab.meta["grating"]
    filter = ptab.meta["filter"]

    file_template = "{prefix}_{grating}_{filter}_{version}.fits".lower()

    sens_file = os.path.join(
        path_to_data,
        file_template.format(
            prefix=prefix,
            filter=filter,
            grating=grating,
            version=version,
        ).lower(),
    )

    sens_data = utils.read_catalog(sens_file)

    sens = np.interp(
        ptab["wave"], sens_data["wavelength"], sens_data["sensitivity"]
    )

    return sens_data, sens


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

    yaml_file = (
        f"cube-{obsid}-{grating}{params['output_suffix']}.params.yaml".lower()
    )
    with open(yaml_file, "w") as fp:
        yaml.dump(params, fp)

    outroot, cubes, ptab, hdul = ifu_pipeline(**params)

    return outroot, cubes, ptab, hdul


CUBE_WHITE_DEFAULT = {
    "PRISM": 376,  # F356W
    "G140H": 364,  # F115W
    "G140M": 365,  # F150W
    "G235M": 375,  # F277W
    "G235H": 375,  # F277W
    "G395M": 375,  # F277W
    "G395H": 376,  # F356W
}


def cube_integrate_filter(cube_hdu, f_number=377, background=0, **kwargs):
    """
    Integrate a cube across a filter bandpass
    """
    import eazy.filters

    RES = eazy.filters.FilterFile()
    if f_number is None:
        f_number = CUBE_WHITE_DEFAULT[
            cube_hdu["SCI"].header["GRATING"].upper()
        ]

    fi = RES[f_number]

    sci = cube_hdu["SCI"].data
    wht = 1.0 / cube_hdu["VAR"].data

    # wtab = utils.GTable(cube_hdu['WCS-TAB'].data)
    stab = cube_load_sensitivity(cube_hdu, **kwargs)

    xfi = np.interp(stab["wave"] * 1.0e4, fi.wave, fi.throughput)

    num = (((sci.T - background).T * wht).T * xfi * stab["sensitivity"]).T
    den = (wht.T * xfi * stab["sensitivity"] ** 2).T

    img = np.nansum(num, axis=0) / np.nansum(den, axis=0)
    ivar = np.nansum(den, axis=0)
    ivar[ivar == 0] = np.nan

    return img, ivar, fi


def cube_load_sensitivity(
    cube_hdu, prefix="msaexp_sensitivity", version="001", **kwargs
):
    """ """
    path_to_data = os.path.join(
        os.path.dirname(__file__), "data/extended_sensitivity"
    )
    stab = utils.GTable(cube_hdu["WCS-TAB"].data)
    stab["wave"] = stab["WAVELENGTH"] / 1.0e4

    grating = cube_hdu["SCI"].header["GRATING"]
    filter = cube_hdu["SCI"].header["FILTER"]

    file_template = "{prefix}_{grating}_{filter}_{version}.fits".lower()

    sens_file = os.path.join(
        path_to_data,
        file_template.format(
            prefix=prefix,
            filter=filter,
            grating=grating,
            version=version,
        ).lower(),
    )

    sens_data = utils.read_catalog(sens_file)

    sens = np.interp(
        stab["wave"], sens_data["wavelength"], sens_data["sensitivity"]
    )

    stab["sensitivity"] = sens
    stab.meta["sensfile"] = sens_file

    return stab


CUBE_RGB_DEFAULT = {
    "PRISM": [364, 375, 377],
    "G140H": [364, 368, 369],  # F115W, 140m, 162m
    "G140M": [364, 365, 366],  # 115w, 150w, 200w
    "G235M": [366, 375, 377],  # F277W
    "G235H": [370, 371, 379],  # F277W
    "G395M": [380, 382, 384],  # F277W
    "G395H": [380, 382, 384],  # F356W
}


def cube_filter_rgb(
    cube_hdu,
    f_numbers=None,
    rgb_scale=None,
    wave_power=-1,
    total_scale=1.0,
    **kwargs,
):
    """
    Make an RGB image of three filters integrated through the cube
    """
    from astropy.visualization import make_lupton_rgb
    import eazy.filters

    RES = eazy.filters.FilterFile()

    if f_numbers is None:
        f_numbers = CUBE_RGB_DEFAULT[cube_hdu["SCI"].header["GRATING"].upper()]

    rgb_filters = [RES[fi] for fi in f_numbers]
    if rgb_scale is None:
        rgb_scale = np.array(
            [(f.pivot / 1.0e4) ** wave_power for f in rgb_filters]
        )

    rgb_img, rgb_ivar = [], []

    for fi, scale in zip(f_numbers, rgb_scale):
        img_i, ivar_i, filter_i = cube_integrate_filter(
            cube_hdu, f_number=fi, **kwargs
        )

        rgb_img.append(img_i * scale / rgb_scale[-1])
        rgb_ivar.append(ivar_i / (scale / rgb_scale[-1]) ** 2)

        print(f"{RES[fi].name}  scale = {scale / rgb_scale[-1]:.2f}")

    imax = np.nanpercentile(rgb_img[1], 84) * 50.2 * total_scale
    rgb = make_lupton_rgb(
        *rgb_img[::-1], stretch=0.1 * imax / 1.1, minimum=-0.01 * imax / 1.1
    )

    nvalid = np.isfinite(np.array(rgb_img)).sum(axis=0) == 3
    for i in range(3):
        rgb[:, :, i][~nvalid] = 255

    return rgb, rgb_img, rgb_ivar, rgb_filters


def cube_extract_segments(
    cube_hdu,
    min_thresh=3,
    thresh_percentile=86,
    iterate=True,
    f_number=None,
    deblend_nthresh=32,
    deblend_cont=1.0e-3,
    erode_background=4,
    filter_kernel=None,
    **kwargs,
):
    """ """
    import sep

    grating = cube_hdu["SCI"].header["GRATING"]

    img, ivar, filt = cube_integrate_filter(
        cube_hdu, f_number=f_number, **kwargs
    )
    img_clean = img - np.nanmedian(img[np.isfinite(ivar)])

    err = np.sqrt(1.0 / ivar)
    err[~np.isfinite(ivar)] = 0

    thresh = np.nanpercentile(img_clean / err, thresh_percentile)
    thresh = np.maximum(thresh, min_thresh)

    _ = sep.extract(
        img_clean,
        thresh,
        err=err,
        mask=(err == 0),
        segmentation_map=True,
        filter_kernel=filter_kernel,
        clean=True,
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
    )

    seg1 = _[-1]
    seg = seg1

    if iterate:
        thresh = np.nanpercentile(img_clean / err, thresh_percentile)
        thresh = np.maximum(thresh, min_thresh)
        print(f"Threshold: {thresh:.2f}")

        img_clean = img - np.nanmedian(img[(err > 0) & (seg1 == 0)])

        _ = sep.extract(
            img_clean,
            thresh,
            err=err,
            mask=(err == 0),
            segmentation_map=True,
            filter_kernel=filter_kernel,
            clean=True,
            deblend_nthresh=deblend_nthresh,
            deblend_cont=deblend_cont,
        )
        seg = _[-1]

    ### Extract segment spectra
    sci = cube_hdu["SCI"].data
    wht = 1.0 / cube_hdu["VAR"].data

    seg_ids = np.unique(seg)
    auto_src = False

    sh = img_clean.shape
    yp, xp = np.indices(sh)

    if len(seg_ids) == 1:
        # Force a source at the center
        auto_src = True

        R = np.sqrt((xp - sh[1] / 2) ** 2 + (yp - sh[0] / 2) ** 2)

        print(f"Force a source at {sh[0]/2}, {sh[1]/2}")

        seg = (R < 4) * 1
        seg_ids = np.unique(seg)

    # Reorder seg by flux
    seg_flux = []
    for seg_id in seg_ids[1:]:
        seg_mask = seg == seg_id
        seg_flux.append(np.nansum(img_clean[seg_mask]))

    so = np.argsort(seg_flux)[::-1]
    seg_sorted = seg * 1
    for i, j in enumerate(so):
        seg_mask = seg == i + 1
        seg_sorted[seg_mask] = j + 1

    seg = seg_sorted

    seg_bkg = 0

    stab = cube_load_sensitivity(cube_hdu, **kwargs)
    stab.meta["detfilt"] = filt.name.split()[0]
    stab.meta["detthrsh"] = thresh
    stab.meta["detdebln"] = deblend_nthresh
    stab.meta["detdeblc"] = deblend_cont
    stab.meta["nsrc"] = len(seg_ids) - 1
    stab.meta["autosrc"] = auto_src

    for i, seg_id in enumerate(seg_ids):
        seg_mask = seg == seg_id

        if (seg_id == 0) & (erode_background is not None):
            seg_mask = nd.binary_erosion(seg_mask, iterations=erode_background)

        stab.meta[f"nmask{seg_id}"] = seg_mask.sum()

        if seg_id == 0:
            prof = 1.0
        else:

            stab.meta[f"pflux{seg_id}"] = np.nansum(img_clean[seg_mask])

            prof = img_clean / stab.meta[f"pflux{seg_id}"]

            stab.meta[f"pmax{seg_id}"] = prof[seg_mask].max()

            stab.meta[f"xsrc{seg_id}"] = np.nansum((xp * prof)[seg_mask])
            stab.meta[f"ysrc{seg_id}"] = np.nansum((yp * prof)[seg_mask])

            prof *= (seg_mask).sum()

        total_num = np.nansum(
            prof * (sci.T - seg_bkg).T * (seg_mask) * wht, axis=(1, 2)
        )

        # No background
        total_num0 = np.nansum(prof * sci * (seg_mask) * wht, axis=(1, 2))

        # Denominator
        total_den = np.nansum(prof**2 * (seg_mask) * wht, axis=(1, 2))

        if seg_id == 0:
            seg_bkg = total_num / total_den
            stab["background"] = seg_bkg

        elif 1:
            err = 1 / np.sqrt(total_den)
            if grating not in "PRISM":
                err_mask = err > 5 * np.nanmedian(err)
                err_mask = nd.binary_dilation(err_mask, iterations=4)
            else:
                err_mask = ~np.isfinite(total_num)

            total_num[err_mask] = np.nan
            total_num0[err_mask] = np.nan

            stab[f"flux{seg_id}"] = total_num / total_den
            stab[f"oflux{seg_id}"] = total_num0 / total_den
            stab[f"ivar{seg_id}"] = total_den

    return seg, stab


def cube_make_diagnostics(
    cube_hdu,
    scale_func=np.log10,
    wave_power=-1,
    figsize=(9, 3),
    cmap=plt.cm.rainbow,
    **kwargs,
):
    """
    Make diagnostics from 3D cube data
    """

    grating = cube_hdu["SCI"].header["GRATING"]

    seg, stab = cube_extract_segments(cube_hdu, **kwargs)

    img, ivar, img_filter = cube_integrate_filter(
        cube_hdu, background=stab["background"], **kwargs
    )

    rgb, rgb_img, rgb_ivar, rgb_filters = cube_filter_rgb(
        cube_hdu,
        background=stab["background"],
        wave_power=wave_power,
        **kwargs,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

    img_scale = scale_func(img)
    perc = np.nanpercentile(img_scale, [2, 50, 98])

    seg_mask = seg * 1.0
    seg_mask[seg == 0] = np.nan

    nmax = np.maximum(len(np.unique(seg)) - 1, 1)

    axes[0].imshow(
        scale_func(img), vmin=perc[0], vmax=perc[2], cmap=plt.cm.gray
    )
    axes[1].imshow(rgb)
    axes[2].imshow(seg_mask, vmin=0, vmax=nmax, cmap=cmap, alpha=0.9)

    for i in range(2):
        for seg_id in np.unique(seg):
            if seg_id == 0:
                continue

            c = cmap(seg_id / nmax)

            axes[i].contour(
                (seg == seg_id) * 1,
                vmin=0,
                vmax=1,
                levels=0,
                origin=None,
                colors=["w"],
                alpha=0.3,
                zorder=10,
            )

            axes[i].contour(
                (seg == seg_id) * 1,
                vmin=0,
                vmax=1,
                levels=0,
                origin=None,
                colors=[c],
                alpha=0.8,
                zorder=11,
            )

    axes[0].text(
        0.015,
        0.01,
        img_filter.name.split()[0].split("_")[-1].upper(),
        ha="left",
        va="bottom",
        transform=axes[0].transAxes,
        fontsize=7,
        bbox={"fc": "w", "ec": "None", "alpha": 1.0},
    )

    for j, c in enumerate(["steelblue", "olive", "tomato"]):
        axes[1].text(
            0.015 + 0.18 * j,
            0.01,
            rgb_filters[j].name.split()[0].split("_")[-1].upper(),
            color=c,
            ha="left",
            va="bottom",
            transform=axes[1].transAxes,
            fontsize=7,
            bbox={"fc": "w", "ec": "None", "alpha": 1.0},
        )

    axes[2].text(
        0.015,
        0.01,
        f'threshold = {stab.meta["detthrsh"]:.1f}' + r"$\sigma$",
        ha="left",
        va="bottom",
        transform=axes[2].transAxes,
        fontsize=7,
        bbox={"fc": "w", "ec": "None", "alpha": 1.0},
    )

    for ax in axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.tight_layout(pad=1)
    img_fig = fig

    # Spectra
    # seg_ids = np.unique(seg)

    ny = np.minimum(stab.meta["nsrc"], 7)

    fig, axes = plt.subplots(ny, 1, sharex=True, figsize=(8, 2.0 * ny))
    if ny == 1:
        axes = [axes]

    # Sort order
    src_fluxes = np.array(
        [stab.meta[f"pflux{seg_id+1}"] for seg_id in range(stab.meta["nsrc"])]
    )
    src_sorted = np.argsort(src_fluxes)
    if len(src_sorted) > ny:
        src_sorted = src_sorted[-ny:]

    sens_flam = stab["sensitivity"] / stab["wave"] ** wave_power
    sens_flam[sens_flam == 0] = np.nan

    # for ax in axes:
    #     ax.plot(
    #         stab["wave"],
    #         stab["background"] / sens_flam,
    #         color="k",
    #         alpha=0.2,
    #     )

    for i, j in enumerate(src_sorted):
        if i >= len(axes):
            break

        ax = axes[i]

        seg_id = j + 1

        c = cmap(seg_id / nmax)

        flux = stab[f"flux{seg_id}"]
        oflux = stab[f"oflux{seg_id}"]

        bkg_diff = oflux - flux
        bkg_scl = np.nanmedian(bkg_diff / stab["background"])
        ax.plot(
            stab["wave"],
            stab["background"] / sens_flam * bkg_scl,
            color="k",
            alpha=0.1,
            zorder=-1,
        )

        # print(seg_id, stab.meta[f"nmask{seg_id}"], stab.meta[f"pmax{seg_id}"], stab.meta[f"pmax{seg_id}"] / stab.meta[f"nmask{seg_id}"])

        err = 1 / np.sqrt(stab[f"ivar{seg_id}"])

        if grating.upper() not in "PRISM":
            err_mask = err > 5 * np.nanmedian(err)
            err_mask = nd.binary_dilation(err_mask, iterations=4)
        else:
            err_mask = ~np.isfinite(err)

        if grating.upper().startswith("G140"):
            err_mask |= stab["wave"] < 0.97

        flux[err_mask] = np.nan
        oflux[err_mask] = np.nan

        # Gray background
        ax.plot(stab["wave"], flux / sens_flam, alpha=0.5, color="0.8")
        ax.plot(stab["wave"], oflux / sens_flam, alpha=0.5, color="0.8")

        ax.fill_between(
            stab["wave"], flux * 0.0, flux / sens_flam, alpha=0.05, color=c
        )

        ax.plot(stab["wave"], flux / sens_flam, alpha=0.5, color=c)
        ax.plot(stab["wave"], oflux / sens_flam, alpha=0.2, color=c)

        ymax = np.nanpercentile(flux / sens_flam, 99)
        ax.set_ylim(-0.2 * ymax, 1.2 * ymax)
        ax.set_yticklabels([])

    for ax in axes:
        ax.grid()

    ax.set_xlabel(r"wavelength")
    if wave_power == 0:
        ax.set_ylabel(r"$f_\nu$")
    elif wave_power == -1:
        ax.set_ylabel(r"$f_\nu~/~\lambda$")
    elif wave_power == -2:
        ax.set_ylabel(r"$f_\lambda$")
    else:
        ax.set_ylabel(
            r"$f_\nu~\cdot~\lambda^{x}$".replace("x", f"{wave_power}")
        )

    fig.tight_layout(pad=0.5)

    result = {
        "img": img,
        "ivar": ivar,
        "rgb": rgb,
        "rgb_img": rgb_img,
        "rgb_ivar": rgb_ivar,
        "rgb_filters": rgb_filters,
        "seg": seg,
        "stab": stab,
        "img_fig": img_fig,
        "fig": fig,
    }

    return result


class ReducedCube:

    sensitivity_file = None
    sensitivity = None
    sens = 1.0
    bwht = 0.0
    spec = None

    labels = None
    label_image = None
    label_counts = None
    label_uni = None

    def __init__(
        self,
        file="",
        data=None,
        redshift=0.0,
        fits_scale_err=0.37,
        slw=None,
        slx=None,
        sly=None,
        negative_threshold=-5,
    ):
        """
        Handler for reduced cubes
        """

        self.file = file
        self.redshift = redshift

        self.slw = slw
        self.slx = slx
        self.sly = sly

        self.negative_threshold = negative_threshold

        if data is not None:
            self.wave, self.header, self.sci, self.wht = data
        else:
            with pyfits.open(file) as cube_hdu:
                if "WCS-TAB" in cube_hdu:
                    self.wave = (
                        utils.GTable(cube_hdu["WCS-TAB"].data)[
                            "WAVELENGTH"
                        ].data
                        / 1.0e4
                    )
                    self.header = cube_hdu[1].header.copy()
                    self.sci = cube_hdu["SCI"].data * 1
                    self.wht = cube_hdu["WHT"].data / fits_scale_err**2
                else:
                    # Load MUSE cube
                    self.header = cube_hdu[1].header.copy()
                    self.sci = cube_hdu["DATA"].data * 1
                    self.wht = 1.0 / cube_hdu["STAT"].data

                    self.header["GRATING"] = "MUSE"
                    self.header["FILTER"] = "MUSE"

                    wcs_ = pywcs.WCS(self.header)
                    xpix = np.arange(self.sci.shape[0])
                    _, _, wave_meters = wcs_.all_pix2world([0], [0], xpix, 0)
                    self.wave = wave_meters * 1.0e6

                    to_fnu = (
                        1.0e-20 * u.erg / u.second / u.cm**2 / u.Angstrom
                    ).to(
                        u.microJansky,
                        equivalencies=u.spectral_density(self.wave * u.micron),
                    )
                    self.sci = (self.sci.T * to_fnu.value).T
                    self.wht = (self.wht.T / to_fnu.value**2).T

            if slw is not None:
                self.sci = self.sci[slw, :, :]
                self.wht = self.wht[slw, :, :]
                self.wave = self.wave[slw]

            if sly is not None:
                self.sci = self.sci[:, sly, :]
                self.wht = self.wht[:, sly, :]

            if slx is not None:
                self.sci = self.sci[:, :, slx]
                self.wht = self.wht[:, :, slx]

        self.yp, self.xp = np.indices(self.sci.shape[1:])

        try:
            self.load_sensitivity()
        except ValueError:
            pass

        self.pixbin = self.valid * 1
        self.mask = self.pixbin.sum(axis=0) > 0

    @property
    def valid(self):
        valid = np.isfinite(self.sci + self.wht) & (self.wht > 0)
        valid &= self.sci - self.bkg > self.negative_threshold * np.sqrt(
            self.wht
        )
        return valid

    @property
    def outroot(self):
        return os.path.basename(self.file).split(".fits")[0]

    @property
    def path(self):
        return os.path.dirname(self.file)

    @property
    def grating(self):
        return self.header["GRATING"]

    @property
    def filter(self):
        return self.header["FILTER"]

    @property
    def shape(self):
        return self.sci.shape

    @property
    def aspect(self):
        sh = self.shape
        return sh[1] / sh[2]

    @property
    def gradient_dv(self):
        """
        Gradient of the wavelength grid ``dwave / wave * c``
        """
        return np.gradient(self.wave) / self.wave * 3.0e5

    @property
    def bkg(self):
        """
        Background in units of ``sci`` (without sensitivity)
        """
        if hasattr(self.bwht, "shape"):
            return self.bwht / (self.wht.T * self.sens).T
        else:
            return 0

    @property
    def wcs_header(self):

        wcsh, _ = utils.make_wcsheader(get_hdu=False)
        for k in wcsh:
            if k not in self.header:
                print(k)
            else:
                wcsh[k] = self.header[k]

        if (self.slx is not None) | (self.sly is not None):
            if self.slx is None:
                slx = slice(0, self.shape[2])
            else:
                slx = self.slx

            if self.sly is None:
                sly = slice(0, self.shape[1])
            else:
                sly = self.sly

            wcs_ = pywcs.WCS(wcsh)
            wcsh = utils.get_wcs_slice_header(wcs_, slx, sly)

        return wcsh

    def wcs2d(self):

        wcs = pywcs.WCS(self.wcs_header)
        wcs.pscale = utils.get_wcs_pscale(wcs)
        return wcs

    def info(self):
        """ """
        info = f"{self.file} {self.shape}  {self.grating}_{self.filter}"
        info += f'  ({self.header["CRVAL1"]:.6f}, {self.header["CRVAL2"]:.6f})  z={self.redshift:.4f}'
        return info

    def load_sensitivity(self, **kwargs):
        """
        Load sensitivity curve
        """
        path_to_files = os.path.join(
            os.path.dirname(__file__),
            "data",
            "extended_sensitivity",
            "ifu",
            "msaexp_ifu_sens*{GRATING}_{FILTER}*fits".format(
                **self.header
            ).lower(),
        )

        sens_files = glob.glob(path_to_files)

        if len(sens_files) == 0:
            self.num = ((self.sci * self.wht).T).T
            self.den = (self.wht.T).T

            msg = f"load_sensitivity: no files found at {path_to_files}"
            raise ValueError(msg)

        sens_files.sort()

        utils.log_comment(
            utils.LOGFILE,
            f"load_sensitivity {self.file}: {sens_files[-1]}",
            verbose=(VERBOSITY & 2),
        )

        self.sensitivity_file = sens_files[-1]
        self.sensitivity = utils.read_catalog(self.sensitivity_file)

        self.sens = np.interp(
            self.wave,
            self.sensitivity["wavelength"],
            self.sensitivity["sensitivity"],
            left=0.0,
            right=0.0,
        )

        # Precompute sensititivity-weighted arrays
        self.num = ((self.sci * self.wht).T * self.sens).T
        self.den = (self.wht.T * self.sens**2).T

    @staticmethod
    def make_rebin_labels(shape=None, factor=4, **kwargs):
        """ """
        bfactor = int(np.round(factor))
        sh2 = (
            shape[0],
            shape[1] // bfactor + (shape[1] % bfactor > 0),
            shape[2] // bfactor + (shape[1] % bfactor > 0),
        )

        fbin_image = np.zeros(shape[1:], dtype=int)

        fbin_id = 0

        for i in range(sh2[1]):
            sli = slice(i * bfactor, (i + 1) * bfactor)
            for j in range(sh2[2]):
                slj = slice(j * bfactor, (j + 1) * bfactor)
                fbin_id += 1
                fbin_image[sli, slj] = fbin_id

        return fbin_image

    def rebin(self, factor=4, **kwargs):
        """
        Rebin cube data
        """

        label_image = self.make_rebin_labels(
            shape=self.shape, factor=factor, **kwargs
        )

        cube = self.rebin_labels(label_image=label_image, **kwargs)

        cube.file = self.file.replace(".fits", f".b{factor}.fits")

        return cube

    def rebin_labels(
        self,
        label_image=None,
        nmad_npix=10000,
        nmad_threshold=10,
        weight_type="wht",
        **kwargs,
    ):
        """ """
        from tqdm import tqdm

        uni = utils.Unique(label_image.flatten(), verbose=False)

        sh = self.shape
        sh2 = (sh[0], uni.N, 1)

        sci = np.zeros(sh2, dtype=self.sci.dtype)
        wht = np.zeros(sh2, dtype=self.sci.dtype)
        npix = np.zeros(sh2, dtype=int)

        xpf, ypf = self.xp.flatten(), self.yp.flatten()

        if weight_type == "wht":
            bnum_ = self.num - self.bwht

        sbkg_ = self.sci - self.bkg

        valid = self.valid & True
        has_data = (valid.sum(axis=0) > 0).flatten()

        for i, bin_i in tqdm(enumerate(uni.values)):
            msk = uni[bin_i] & has_data
            if msk.sum() == 0:
                continue

            scim = sbkg_[:, ypf[msk], xpf[msk]]
            whtm = self.wht[:, ypf[msk], xpf[msk]]
            validm = valid[:, ypf[msk], xpf[msk]]

            if weight_type == "wht":
                numm = bnum_[:, ypf[msk], xpf[msk]]
                denm = self.den[:, ypf[msk], xpf[msk]]

            if uni.counts[i] > nmad_npix:
                med = np.nanmedian(scim, axis=1)
                dmed = (scim.T - med).T
                nmad = 1.48 * np.nanmedian(np.abs(dmed), axis=1)

                ok = (np.abs(dmed).T < nmad_threshold * nmad).T & validm
            else:
                ok = validm

            if weight_type == "wht":
                # weighted average
                wht[:, i, 0] = np.nansum(denm * np.nan ** (1 - ok), axis=1)
                sci[:, i, 0] = (
                    np.nansum(numm * np.nan ** (1 - ok), axis=1) / wht[:, i, 0]
                )
            else:
                # simple average
                npix = np.sum(ok, axis=1)
                wht[:, i, 0] = (
                    1.0
                    / np.nansum(1.0 / whtm * np.nan ** (1 - ok), axis=1)
                    * npix**2
                )
                sci[:, i, 0] = (
                    np.nansum(scim * np.nan ** (1 - ok), axis=1) / npix
                )

        if weight_type == "wht":
            # Put sensitivity back in
            sci = (sci.T * self.sens).T
            wht = (wht.T / self.sens**2).T

        cube = ReducedCube(
            file=self.file.replace(".fits", f".rebin.fits"),
            redshift=self.redshift,
            data=(self.wave, self.header, sci, wht),
        )

        cube.label_image = label_image
        cube.labels = uni.values
        cube.label_counts = uni.counts
        cube.label_uni = uni

        return cube

    def mean_image(self, wave_range=None, **kwargs):
        """
        Weighted mean image with optional wavelength sub range
        """
        if wave_range is None:
            wsub = np.isfinite(self.wave)
        else:
            wsub = (self.wave > wave_range[0]) & (self.wave < wave_range[1])

        mean_weight = np.nansum(self.den[wsub, :, :], axis=0)

        mean_image = (
            np.nansum((self.num - self.bwht)[wsub, :, :], axis=0) / mean_weight
        )

        return mean_image, mean_weight

    def make_2d_profile(
        self,
        x0="max",
        niter=7,
        min_weight=0.1,
        center_radius=5,
        norm_radius=12,
        make_figure=True,
        **kwargs,
    ):
        """ """

        mean_image, mean_weight = self.mean_image(**kwargs)
        weight_mask = mean_weight > min_weight * np.nanmedian(mean_weight)
        weight_mask &= np.isfinite(mean_image)

        norm_mask = weight_mask & True
        center_mask = weight_mask & True

        if x0 in ["max"]:
            sn = mean_image * weight_mask * np.sqrt(mean_weight)
            yc, xc = np.unravel_index(np.nanargmax(sn), mean_image.shape)
            msg = f"make_2d_profile:  init max (xc, yc) = ({xc:.1f}, {yc:.1f})"
            utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY & 2))

        elif x0 is not None:
            xc, yc = x0
            msg = f"make_2d_profile:  init (xc, yc) = ({xc:.1f}, {yc:.1f})"
            utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY & 2))
        else:
            xc = None
            yc = None

        if xc is not None:

            Rp = np.sqrt((self.xp - xc) ** 2 + (self.yp - yc) ** 2)
            norm_mask = (Rp < norm_radius) & weight_mask
            center_mask = (Rp < center_radius) & weight_mask

        xc0 = 0
        yc0 = 0

        for iter_ in range(niter):

            yp, xp = np.indices(mean_image.shape)
            xc = np.nansum((self.xp * mean_image)[center_mask]) / np.nansum(
                mean_image[center_mask]
            )
            yc = np.nansum((self.yp * mean_image)[center_mask]) / np.nansum(
                mean_image[center_mask]
            )

            Rp = np.sqrt((self.xp - xc) ** 2 + (self.yp - yc) ** 2)
            norm_mask = (Rp < norm_radius) & weight_mask
            center_mask = (Rp < center_radius) & weight_mask

            msg = f"make_2d_profile:  iter {iter_:^3}  (xc, yc) = ({xc:.1f}, {yc:.1f})"
            utils.log_comment(utils.LOGFILE, msg, verbose=(VERBOSITY & 2))

            dx = (xc - xc0) ** 2 + (yc - yc0) ** 2
            if np.sqrt(dx) < 0.1:
                break

            xc0, yc0 = xc, yc

        if make_figure:
            fig, axes = plt.subplots(
                1, 3, figsize=(9, 3 * self.aspect), sharex=True, sharey=True
            )
            norm = np.nansum(mean_image[norm_mask])
            axes[0].imshow(
                np.log(mean_image * weight_mask / norm), vmin=-8, vmax=-5
            )
            axes[1].imshow(
                np.log(mean_image * norm_mask / norm), vmin=-10, vmax=-3
            )
            axes[2].imshow(
                np.log(mean_image * center_mask / norm), vmin=-10, vmax=-3
            )
            for ax in axes:
                ax.grid()
            fig.tight_layout(pad=1)
        else:
            fig = None

        result = {
            "mean_image": mean_image,
            "mean_weight": mean_weight,
            "norm_mask": norm_mask,
            "center_mask": center_mask,
            "profile2d": mean_image / norm,
            "x0": (xc, yc),
            "fig": fig,
        }
        return result

    def get_background(
        self, mean_image=None, norm_mask=None, bkg_mask=None, dilate_radius=16
    ):
        """ """
        from skimage.morphology import isotropic_dilation

        if bkg_mask is None:
            bkg_mask = np.isfinite(mean_image) & (
                ~isotropic_dilation(norm_mask, radius=8)
            )

        bnum1d = np.nansum(self.num * (bkg_mask), axis=(1, 2))
        bden1d = np.nansum(self.den * (bkg_mask), axis=(1, 2))
        bwht = ((bnum1d / bden1d * self.sens**2) * self.wht.T).T

        return bnum1d, bden1d, bwht

    def optimal_extraction(
        self,
        profile2d=1.0,
        mask_percentile=50,
        mask_threshold=0.1,
        erode="auto",
        **kwargs,
    ):
        """ """
        num1d = np.nansum((self.num - self.bwht) * profile2d, axis=(1, 2))
        den1d = np.nansum(self.den * profile2d**2, axis=(1, 2))

        mask_level = np.nanpercentile(den1d, mask_percentile) * mask_threshold
        mask1d = den1d > mask_level

        if erode in ["auto"]:
            if self.grating.endswith("H"):
                erode_iters = 16
            elif self.grating == "PRISM":
                erode_iters = 4
            else:
                erode_iters = 8

            mask1d = nd.binary_erosion(mask1d, iterations=erode_iters)

        elif erode > 0:
            mask1d = nd.binary_erosion(mask1d, iterations=erode)

        return num1d, den1d, mask1d

    def make_spec_hdu(
        self, profile2d=1.0, x0=None, norm_mask=1.0, slit_width=2, **kwargs
    ):
        """ """
        if x0 is None:
            sh = self.shape
            x0 = (sh[2] / 2.0, sh[1] / 2.0)

        p2d = profile2d * norm_mask

        num1d, den1d, mask1d = self.optimal_extraction(profile2d=p2d, **kwargs)
        den1d[~mask1d] = np.nan
        flux1d = num1d / den1d
        err1d = 1.0 / np.sqrt(den1d)

        tab = utils.GTable()

        tab["wave"] = self.wave * u.micron
        tab["flux"] = flux1d * u.microJansky
        tab["err"] = err1d * u.microJansky
        tab["escale"] = 1.0
        tab["valid"] = np.isfinite(err1d) & (err1d > 0)

        for k in self.header:
            if k.startswith("NAX"):
                continue
            elif k in ["COMMENT", "BITPIX", "XTENSION"]:
                continue

            # print(k)
            tab.meta[k.upper()] = self.header[k]

        # Pseudo-slit 2D extraction
        mask_slit = np.abs(self.xp - x0[0]) < slit_width
        pnorm2d = np.nansum(p2d * mask_slit)

        prof2 = np.nansum(p2d, axis=0)
        prof2 *= 1.0 / prof2.sum()

        num2d = np.nansum(
            (self.num - self.bwht) * mask_slit * prof2, axis=(2)
        ).T
        den2d = np.nansum(self.den * mask_slit * (prof2) ** 2, axis=(2)).T

        data2d = num2d / den2d
        wht2d = 1 / (1.0 / den2d + (0.05 * data2d) ** 2)

        msk = (wht2d > 0) & np.isfinite(wht2d + data2d)
        data2d[~msk] = 0
        wht2d[~msk] = 0

        prof2d = data2d * 1.0  # / np.nansum(data2d, axis=0)
        prof2d[~np.isfinite(prof2d) | (data2d < 0)] = 0
        prof2d /= np.nansum(prof2d, axis=0)
        prof2d[(prof2d**0 * tab["valid"][None, :]) == 0] = 0.0
        prof2d[~np.isfinite(prof2d)] = 0

        hdul = pyfits.HDUList(
            [
                pyfits.PrimaryHDU(),
                pyfits.BinTableHDU(tab, name="SPEC1D"),
                pyfits.ImageHDU(data=data2d, name="SCI"),
                pyfits.ImageHDU(data=wht2d, name="WHT"),
                pyfits.ImageHDU(data=prof2d, name="PROFILE"),
            ]
        )

        hdul["SCI"].header["SRCNAME"] = self.outroot
        hdul[1].header["YTRACE"] = x0[1] + 0.5
        hdul[1].header["PROFCEN"] = 0.0
        hdul[1].header["PROFSIG"] = -1.0

        ptab = utils.GTable()
        ptab["pfit"] = np.nansum(hdul["PROFILE"].data, axis=1)
        ptab["profile"] = ptab["pfit"]
        ptab.meta["PROFSTRT"] = 10
        ptab.meta["PROFSTOP"] = len(tab) - 10
        ptab.meta["PROFSIG"] = np.nan

        hdul.append(pyfits.BinTableHDU(ptab, name="PROF1D"))

        return hdul

    def set_spec(self, **kwargs):
        """ """
        from .spectrum import SpectrumSampler

        hdu = self.make_spec_hdu(**kwargs)
        self.spec = SpectrumSampler(hdu, sens_file=self.sensitivity_file)

    def fit_emission_lines(
        self,
        lines=["Ha"],
        dv_slice=[-800, 800],
        wave_slice=None,
        dv_range=800,
        velocity_sigma=150,
        lorentz=False,
        oversample=2.5,
        scale_disp=1.8,
        continuum_order=1,
        get_covariance=True,
        templates=None,
        optimize="loss",
        min_line_threshold=-4,
        broad_component_sigma=None,
        **kwargs,
    ):
        """ """
        from tqdm import tqdm

        if self.spec is None:
            self.set_spec(**kwargs)

        lwaves = [[w / 1.0e4 for w in LINE_WAVELENGTHS[li]] for li in lines]
        lratios = [LINE_RATIOS[li] for li in lines]

        line_wavelength = lwaves[0][0] * (1 + self.redshift)
        ii = np.where(self.wave > line_wavelength)[0][0]

        li = self.spec.fast_emission_line(
            line_wavelength,
            line_flux=1.0,
            velocity_sigma=velocity_sigma,
            scale_disp=scale_disp,
            lorentz=lorentz,
        )

        dv_ii = np.gradient(self.wave)[ii] / self.wave[ii] * 3.0e5
        nstep = np.maximum(int(np.ceil(dv_range / dv_ii)), 8)

        if wave_slice is not None:
            dv_slice = [
                (wave_slice[0] - line_wavelength) / line_wavelength * 3.0e5,
                (wave_slice[1] - line_wavelength) / line_wavelength * 3.0e5,
            ]

        dv_step = [
            ii - int(np.ceil(np.maximum(-dv_slice[0], dv_range) / dv_ii)),
            ii + int(np.ceil(np.maximum(dv_slice[1], dv_range) / dv_ii)),
        ]

        dv_step[0] = np.maximum(dv_step[0], 0)
        dv_step[1] = np.minimum(dv_step[1], self.shape[0])

        sl = slice(*dv_step)

        # Precompute slices
        sci_sl = (
            ((self.num - self.bwht) / self.den)[sl, :, :].T
            * self.spec["to_flam"][sl]
        ).T
        wht_sl = (self.den[sl, :, :].T / self.spec["to_flam"][sl] ** 2).T

        nsl = len(self.wave[sl])

        model = np.zeros_like(sci_sl)

        yx = sci_sl * np.sqrt(wht_sl)
        swht_sl = np.sqrt(wht_sl)

        ok = np.isfinite(yx)

        msk = self.mask & (np.nansum(sci_sl, axis=0) > 0)
        xpm = self.xp[msk]
        ypm = self.yp[msk]

        dv_grid = np.arange(-nstep, nstep + 1, 1.0 / oversample) * dv_ii
        # wave_grid = line_wavelength * (1 + dv_grid / 3.e5)
        ngrid = len(dv_grid)

        nlines = len(lines)

        if templates is None:
            ncoeffs = continuum_order + 1 + nlines
            xx = np.linspace(-1, 1.0, nsl)
            # c = np.polynomial.polynomial.polyvander(xx, continuum_order).T
            c = np.polynomial.chebyshev.chebvander(xx, continuum_order).T
        else:
            ncoeffs = len(templates) + nlines

            c = np.array(
                [
                    self.spec.resample_eazy_template(
                        t,
                        z=self.redshift,
                        scale_disp=scale_disp,
                        velocity_sigma=velocity_sigma,
                        fnu=False,
                    )[sl]
                    for t in templates
                ]
            )

            template_norm = np.median(c, axis=1)
            # print("xxx templates", template_norm)

        if broad_component_sigma is not None:
            broad_lines = []
            for i in range(len(lines)):
                for wi in lwaves[i]:
                    w_obs = wi * (1 + self.redshift)

                    li = self.spec.fast_emission_line(
                        w_obs,
                        line_flux=1.0,
                        velocity_sigma=broad_component_sigma,
                        scale_disp=scale_disp,
                        lorentz=True,
                    )
                    broad_lines.append(li[sl])

            c = np.vstack([c, np.array(broad_lines)])

            ncoeffs += len(broad_lines)

        coeffs = np.zeros((ngrid, ncoeffs, *msk.shape))
        vcoeffs = np.zeros((ngrid, ncoeffs, *msk.shape))
        model = np.zeros((ngrid, nsl, *msk.shape))

        nbins = msk.sum()

        msg = (
            f"fit_emission_line: {lines} {scale_disp:.1f} σ={velocity_sigma:.0f} "
            f"z={self.redshift:.4f} nslice={nsl} dv={dv_ii:.1f} km/s  {nbins} bins  {ngrid} steps"
        )
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        A = np.vstack([c, np.zeros((nlines, nsl))])
        is_line = np.arange(len(coeffs)) > (nlines - 1)

        for k, dv in tqdm(enumerate(dv_grid)):
            if templates is not None:
                zk = (1 + self.redshift) * (1 + dv / 3.0e5) - 1
                for j, t in enumerate(templates):
                    A[j, :] = (
                        self.spec.resample_eazy_template(
                            t,
                            z=zk,
                            scale_disp=scale_disp,
                            velocity_sigma=velocity_sigma,
                            fnu=False,
                        )[sl]
                        / template_norm[j]
                    )

            for j in range(nlines):
                A[-nlines + j, :] = 0.0
                for w0, r0 in zip(lwaves[j], lratios[j]):
                    wi = w0 * (1 + self.redshift) * (1 + dv / 3.0e5)
                    A[-nlines + j, :] += self.spec.fast_emission_line(
                        wi,
                        line_flux=r0,
                        velocity_sigma=velocity_sigma,
                        scale_disp=scale_disp,
                        lorentz=lorentz,
                    )[sl]

            for j, i in zip(ypm, xpm):
                oki = ok[:, j, i]

                Ax = (A * swht_sl[:, j, i])[:, oki]
                yxi = yx[:, j, i][oki]

                lsq = np.linalg.lstsq(Ax.T, yxi, rcond=None)

                c_ji = lsq[0] * 1.0

                if get_covariance:
                    try:
                        v_ji = utils.safe_invert(Ax.dot(Ax.T)).diagonal()
                        vcoeffs[k, :, j, i] = v_ji
                        clip_line = is_line & (
                            c_ji < min_line_threshold * np.sqrt(v_ji)
                        )
                        c_ji[clip_line] = 0.0
                    except:
                        pass

                coeffs[k, :, j, i] = c_ji
                m_ji = A.T.dot(c_ji)
                model[k, :, j, i] = m_ji

        sn_coeffs = coeffs / np.sqrt(vcoeffs)
        sn_coeffs[~np.isfinite(sn_coeffs)] = 0

        loss = np.nansum((sci_sl - model) ** 2 * wht_sl, axis=1)

        if optimize == "loss":
            imax = np.argmin(loss, axis=0)
        else:
            imax = np.nanargmax(sn_coeffs[:, -nlines, :, :], axis=0)

        max_line_dv = dv_grid[imax]
        max_line_dv[~msk] = np.nan

        max_line_sn = np.zeros((nlines, *msk.shape))
        max_line_flux = np.zeros((nlines, *msk.shape))
        max_model = np.zeros((nsl, *msk.shape))

        for j, i in zip(ypm, xpm):
            k = imax[j, i]
            max_line_sn[:, j, i] = sn_coeffs[k, -nlines:, j, i]
            max_line_flux[:, j, i] = coeffs[k, -nlines:, j, i]
            max_model[:, j, i] = model[k, :, j, i]

        ldata = {
            "lines": lines,
            "redshift": self.redshift,
            "dv_grid": dv_grid,
            "sci": sci_sl,
            "wht": wht_sl,
            "sl": sl,
            "coeffs": coeffs,
            "vcoeffs": vcoeffs,
            "model": model,
            "loss": loss,
            "imax": imax,
            "max_line_sn": max_line_sn,
            "max_line_flux": max_line_flux,
            "max_line_dv": max_line_dv,
            "max_model": max_model,
            "get_covariance": get_covariance,
        }

        return ldata

    def log_sigma_grid(
        self, min_dv_factor=0.5, max_sigma=500, step_factor=2, **kwargs
    ):
        """ """
        min_wave_dv = np.ceil(self.gradient_dv.min() * min_dv_factor / 25) * 25

        sigmas = np.exp(
            np.arange(
                np.log(min_wave_dv), np.log(max_sigma), np.log(step_factor)
            )
        )

        return sigmas

    def line_fit_pipeline(
        self,
        max_bin_factor=2,
        bin_sn_threshold=2.5,
        sigma_kwargs={},
        make_outputs=True,
        label_kwargs=None,
        **kwargs,
    ):
        """
        Do line fit over a grid of velocity bins
        """

        full_line_dv = np.zeros(self.mask.shape, dtype=float)
        full_line_dv_var = np.zeros(self.mask.shape, dtype=float)
        full_line_sig = np.zeros(self.mask.shape, dtype=float) - 1.0
        full_line_sig_var = np.zeros(self.mask.shape, dtype=float) - 1.0
        full_line_bin = np.zeros(self.mask.shape, dtype=float)
        full_line_bid = np.zeros(self.mask.shape, dtype=int)

        full_line_model = None

        full_sci = self.sci * 1
        full_wht = self.wht * 1

        if label_kwargs is not None:
            max_bin_factor = 2

        bfactor = max_bin_factor * 2
        sh = self.shape

        sigmas = self.log_sigma_grid(**sigma_kwargs)
        msg = (
            f"line_fit_pipeline: sigmas = {[int(np.round(s)) for s in sigmas]}"
        )
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

        grids = []
        bin_factors = []

        self.initial_mask = self.mask & True

        while bfactor > 1:
            bfactor = int(bfactor / 2)

            if bfactor > 1:
                if label_kwargs is not None:
                    msg = f"line_fit_pipeline: bin with label image"
                    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

                    WITH_LABELS = 2
                    bcube = self.rebin_labels(**label_kwargs)
                    sh2 = bcube.shape

                    msg = f"line_fit_pipeline: N={bcube.label_uni.N} labels"
                    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

                else:

                    WITH_LABELS = 1
                    msg = f"line_fit_pipeline: bin factor {bfactor}"
                    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

                    bcube = self.rebin(factor=bfactor)
                    sh2 = bcube.shape

                    # sh2 = (sh[0], sh[1]//bfactor, sh[2]//bfactor)
                    #
                    # sci = np.zeros(sh2, dtype=bcube.sci.dtype)
                    # var = np.zeros(sh2, dtype=bcube.sci.dtype)
                    # for i in range(sh2[1]):
                    #     for j in range(sh2[2]):
                    #         sli = slice(i*bfactor, (i+1)*bfactor)
                    #         slj = slice(j*bfactor, (j+1)*bfactor)
                    #         bcube.mask[i,j] = np.nanmax(full_line_sig[sli, slj]) <= 0

            else:
                WITH_LABELS = False
                bcube = self
                self.mask &= full_line_sig <= 0

            if self.mask.sum() == 0:
                break

            ldata_sig = {}

            for sig in sigmas:
                kwargs["velocity_sigma"] = sig
                ldata_sig[sig] = bcube.fit_emission_lines(**kwargs)

            grids.append(ldata_sig)
            bin_factors.append(bfactor)

            ######
            # Merge over sigmas
            # loss = np.array([ldata_sig[sig]['loss'] for sig in ldata_sig])
            #
            # lsh = loss.shape
            # ks = np.zeros(lsh[2:], dtype=int)
            # kv = np.zeros(lsh[2:], dtype=int)
            # for i in range(lsh[2]):
            #     for j in range(lsh[3]):
            #         ks[i,j], kv[i,j] = np.unravel_index(np.nanargmin(loss[:,:,i,j]), lsh[:2])
            #
            # coeffs = np.array([ldata_sig[sig]['coeffs'] for sig in ldata_sig])
            # vcoeffs = np.array([ldata_sig[sig]['vcoeffs'] for sig in ldata_sig])
            # model = np.array([ldata_sig[sig]['model'] for sig in ldata_sig])
            #
            # shc = ldata_sig[sig]['coeffs'].shape
            # ncoeffs = shc[1]
            #
            # shm = ldata_sig[sig]['model'].shape
            #
            # best_coeffs = coeffs[0,0,:,:,:]
            # best_vcoeffs = vcoeffs[0,0,:,:,:]
            # best_model = model[0,0,:,:,:]
            # best_sig = best_coeffs[0,:,:] * 0.
            # best_dv = best_coeffs[0,:,:] * 0.
            # nsl = best_model.shape[0]
            #
            # ldata = ldata_sig[sigmas[0]]
            # nlines = len(ldata["lines"])
            #
            # for ii in range(shc[2]):
            #     for jj in range(shc[3]):
            #         ksi = ks[ii,jj]
            #         kvi = kv[ii,jj]
            #         best_coeffs[:,ii,jj] = coeffs[ksi, kvi, :, ii, jj]
            #         best_vcoeffs[:,ii,jj] = vcoeffs[ksi, kvi, :, ii, jj]
            #         best_model[:,ii,jj] = model[ksi, kvi, :, ii, jj]
            #         best_dv[ii,jj] = ldata['dv_grid'][kvi]
            #         best_sig[ii,jj] = sigmas[ksi]
            #

            ldata = ldata_sig[sigmas[0]]
            shc = ldata_sig[sig]["coeffs"].shape
            ncoeffs = shc[1]
            nlines = len(ldata["lines"])

            # Marginalize fit parameters by lnp weight
            mres = marginalize_line_fit_grid(grids[-1])

            best_coeffs = mres["coeffs"]
            if ldata["get_covariance"]:
                best_vcoeffs = mres["vcoeffs"]
            else:
                best_vcoeffs = mres["coeffs_var"]

            best_model = mres["model"]
            best_dv = mres["dv"]
            best_dv_var = mres["dv_var"]
            best_sig = np.exp(mres["ln_sig"])
            best_sig_var = mres["ln_sig_var"]
            nsl = best_model.shape[0]

            best_sn = best_coeffs / np.sqrt(best_vcoeffs)

            pixbin = (
                np.sum(bcube.pixbin[ldata["sl"], :, :], axis=0)
                / ldata["model"].shape[1]
            )

            # Initialize full arrays
            if full_line_model is None:
                full_line_model = np.zeros((nsl, *self.shape[-2:]))
                full_line_coeffs = np.zeros((ncoeffs, *self.shape[-2:]))
                full_line_vcoeffs = np.zeros((ncoeffs, *self.shape[-2:]))

            if bfactor > 1:
                fill = pixbin > 0.3 * np.nanmax(pixbin)
                for k in range(nlines):
                    fill &= np.abs(
                        best_sn[-nlines + k, :, :]
                    ) < bin_sn_threshold * 2 / np.sqrt(pixbin / bfactor**2)

                if WITH_LABELS:
                    if WITH_LABELS > 1:
                        fill = np.ones(bcube.shape[1:], dtype=bool)

                    xpf = self.xp.flatten()
                    ypf = self.yp.flatten()

                label_i = np.min(full_line_bid) + 1

                for i in range(sh2[1]):
                    for j in range(sh2[2]):

                        if not fill[i, j]:
                            continue

                        if WITH_LABELS:
                            label_i = bcube.labels[i]
                            if label_i == 0:
                                continue

                            is_label_i = bcube.label_uni[label_i]
                            sli = ypf[is_label_i]
                            slj = xpf[is_label_i]
                        else:
                            ii = slice(i * bfactor, (i + 1) * bfactor)
                            jj = slice(j * bfactor, (j + 1) * bfactor)
                            sli = self.yp[ii, jj].flatten()
                            slj = self.xp[ii, jj].flatten()
                            label_i += 1

                        for i_, j_ in zip(sli, slj):
                            if np.nanmax(full_line_sig[i_, j_]) <= 0:
                                for k in range(ncoeffs):
                                    full_line_coeffs[k, i_, j_] = best_coeffs[
                                        k, i, j
                                    ]
                                    full_line_vcoeffs[k, i_, j_] = (
                                        best_vcoeffs[k, i, j]
                                    )

                                full_sci[:, i_, j_] = bcube.sci[:, i, j]
                                full_wht[:, i_, j_] = bcube.wht[:, i, j]

                                full_line_model[:, i_, j_] = best_model[
                                    :, i, j
                                ]

                                full_line_bin[i_, j_] = (
                                    bfactor / 2**WITH_LABELS
                                )

                                full_line_dv[i_, j_] = best_dv[i, j]
                                full_line_dv_var[i_, j_] = best_dv_var[i, j]

                                full_line_sig[i_, j_] = best_sig[i, j]
                                full_line_sig_var[i_, j_] = best_sig_var[i, j]

                                full_line_bid[sli, slj] = label_i

                if WITH_LABELS > 1:
                    self.mask &= False

            else:
                sh2 = self.shape
                fill = np.isfinite(best_sn[-nlines, :, :])
                for k in range(nlines):
                    fill &= np.isfinite(best_sn[-nlines + k, :, :])

                for k in range(ncoeffs):
                    for i in range(sh2[1]):
                        for j in range(sh2[2]):
                            if fill[i, j]:
                                full_line_coeffs[k, i, j] = best_coeffs[
                                    k, i, j
                                ]
                                full_line_vcoeffs[k, i, j] = best_vcoeffs[
                                    k, i, j
                                ]

                for i in range(sh2[1]):
                    for j in range(sh2[2]):
                        if fill[i, j]:
                            full_line_model[:, i, j] = best_model[:, i, j]

                full_line_dv[fill] = best_dv[fill]
                full_line_dv_var[fill] = best_dv_var[fill]

                full_line_sig[fill] = best_sig[fill]
                full_line_sig_var[fill] = best_sig_var[fill]

                full_line_bin[fill] = 1.0

        full_line_dv[full_line_sig <= 0] = np.nan
        full_line_sig[full_line_sig <= 0] = np.nan

        # Reset mask
        self.mask = self.initial_mask & True

        #####
        # Done with bins
        model_fnu = (
            full_line_model
            / (self.spec["to_flam"] / self.sens)[ldata["sl"]][:, None, None]
        )
        resid = (full_sci[ldata["sl"], :, :] - model_fnu) * np.sqrt(
            full_wht[ldata["sl"], :, :]
        )
        err_scale = utils.nmad(
            resid[(full_line_model != 0) & np.isfinite(resid)]
        )
        print("err_scale: ", err_scale)

        result = {
            "redshift": self.redshift,
            "lines": ldata["lines"],
            "sigmas": sigmas,
            "slice": ldata["sl"],
            "nlines": nlines,
            "ncoeffs": ncoeffs,
            "nsl": nsl,
            "err_scale": err_scale,
            "model": full_line_model,
            "coeffs": full_line_coeffs,
            "vcoeffs": full_line_vcoeffs,
            "dv": full_line_dv,
            "dv_var": full_line_dv_var,
            "sig": full_line_sig,
            "sig_var": full_line_sig_var,
            "bin": full_line_bin,
            "bid": full_line_bid,
            "full_sci": full_sci,
            "full_wht": full_wht,
            "grids": grids,
            "bin_factors": bin_factors,
        }

        if make_outputs:
            lfit = LinefitData(self, result, **kwargs)

            result["fig"] = []
            for i in range(lfit.nlines):
                result["fig"].append(
                    lfit.line_figure_2d(line_index=i, **kwargs)
                )

            result["hdu"] = lfit.to_fits_hdu(**kwargs)

        return result

    def xline_fit_pipeline(
        self,
        line_wavelength=None,
        line_name="PaA",
        line_label=None,
        x0=None,
        velocity_range=100,
        recenter_dv=True,
        show_contours=True,
        **kwargs,
    ):
        """ """
        import scipy.ndimage as nd
        from skimage.morphology import isotropic_dilation

        if line_wavelength is None:
            lw, lr = utils.get_line_wavelengths()
            line_wavelength = lw[line_name][0] * (1 + self.redshift) / 1.0e4

        mean_image, mean_weight = self.mean_image(**kwargs)
        weight_mask = mean_weight > 0.1 * np.nanmedian(mean_weight)
        mean_image[~weight_mask] = np.nan

        ldata = self.fit_emission_line(
            line_wavelength=line_wavelength, **kwargs
        )

        max_cont_flux = ldata["max_line_sn"]
        max_line_flux = ldata["max_line_flux"]
        max_line_sn = ldata["max_line_sn"]
        max_line_dv = ldata["max_line_dv"]

        gmask = max_line_sn > 4
        gmask = nd.binary_opening(gmask, iterations=1)
        gmask = isotropic_dilation(gmask, radius=4)

        line_sn_mask = np.ones_like(max_line_sn)
        line_sn_mask[nd.gaussian_filter(max_line_sn, 1) < 3] = np.nan
        line_sn_mask[~gmask] = np.nan

        sh = line_sn_mask.shape

        line_sn_mask[
            max_line_flux
            > np.nanpercentile(max_line_flux[np.isfinite(line_sn_mask)], 98)
            * 3
        ] = np.nan

        fig, axes = plt.subplots(
            1, 3, figsize=(9, 3 * self.aspect), sharex=True, sharey=True
        )

        vmax = 1.0e-3

        perc = np.nanpercentile(max_line_flux, [16, 50, 84])

        norm = simple_norm(
            max_line_flux,
            "log",
            vmin=perc[1] - 2 * (perc[1] - perc[0]),
            vmax=np.minimum(
                perc[1] + 30 * (perc[2] - perc[0]), np.nanmax(max_line_flux)
            ),
            # vmax=np.nanmax(line_flux)*10,
            log_a=1.0e1,
        )

        mean_norm = np.nanpercentile(mean_image[np.isfinite(mean_image)], 99.5)
        axes[0].imshow(
            np.log10(mean_image / mean_norm), vmin=-3, vmax=0.1, cmap="RdGy"
        )

        axes[1].imshow(max_line_flux, norm=norm, cmap="bone_r")

        levels = [3, 5, 10, 20, 40, 80, 160, 320]

        if show_contours:
            for j in [0, 1]:
                axes[j].contour(
                    nd.gaussian_filter(max_line_sn * gmask, 0.5),
                    levels=levels[:1],
                    # colors=['0.6'] * len(levels),
                    colors=[
                        "magenta" for v in np.linspace(0.2, 1, len(levels))
                    ][:1],
                    alpha=0.2,
                )

        ax = axes[2]

        if recenter_dv:
            dv_offset = np.nanmedian(max_line_dv * line_sn_mask)
        else:
            dv_offset = 0.0

        imsh_dv = ax.imshow(
            max_line_dv * line_sn_mask - dv_offset,
            vmin=-velocity_range,
            vmax=velocity_range,
            cmap="RdYlBu_r",
        )

        ax.contour(
            nd.gaussian_filter(max_line_sn * gmask, 0.5),
            levels=levels,
            # colors=['0.6'] * len(levels),
            colors=[
                plt.cm.gray_r(v) for v in np.linspace(0.2, 1, len(levels))
            ],
            alpha=0.8,
        )

        h = self.header
        pscale = np.sqrt(h["CD1_1"] ** 2 + h["CD1_2"] ** 2) * 3600.0
        dxt = 0.5
        ticks = np.arange(-40, 41, dxt) / pscale

        sh = self.shape[1:]

        if x0 is None:
            x0 = (sh[1] / 2, sh[0] / 2)

        xt = ticks + x0[0]
        xtv = (xt > 0.05 * dxt / pscale) & (xt < sh[1] - 0.05 * dxt / pscale)
        yt = ticks + x0[1]
        ytv = (yt > 0.05 * dxt / pscale) & (yt < sh[0] - 0.05 * dxt / pscale)

        for ax in axes:
            ax.grid()
            ax.set_xticks(xt[xtv])
            ax.set_yticks(yt[ytv])
            ax.set_xticklabels([])  # ticks[xtv] * pscale)
            ax.set_yticklabels([])  # (ticks[ytv] * pscale)

        axes[0].text(
            0.5,
            0.98,
            f"{self.outroot}\nz={self.redshift:.4f}",
            fontsize=6,
            ha="center",
            va="top",
            transform=axes[0].transAxes,
            bbox={"fc": "w", "alpha": 0.7, "ec": "None"},
        )

        if line_label is None:
            if line_name not in LINE_LABELS_LATEX:
                print(f"use {line_name} as line label!")
                line_label = line_name + " @ "
            else:
                line_label = LINE_LABELS_LATEX[line_name] + " @ "

        axes[1].text(
            0.5,
            0.98,
            (
                line_label
                + r"$\lambda_\mathrm{obs}~=~x~\mathrm{\mu m}$".replace(
                    "x", f"{line_wavelength:.4f}"
                )
            ),
            fontsize=6,
            ha="center",
            va="top",
            transform=axes[1].transAxes,
            bbox={"fc": "w", "alpha": 0.7, "ec": "None"},
        )

        fig.tight_layout(pad=0.5)
        fig.tight_layout(pad=0.5)

        if recenter_dv:
            cb_label = r"$\Delta v - $ xx [km/s]".replace(
                "xx", f"{dv_offset:.1f}"
            )
        else:
            cb_label = r"$\Delta v$ [km/s]"

        _ = msautils.tight_colorbar(
            imsh_dv,
            fig,
            axes[2],
            sx=0.75,
            sy=0.02,
            loc="uc",
            pad=0.01,
            label=cb_label,
            bbox_kwargs={"fc": "w", "alpha": 0.8, "ec": "None"},
            zorder=100,
            label_format=None,
            labelsize=7,
            colorbar_alpha=None,
        )

        ldata["fig"] = fig
        ldata["dv_offset"] = dv_offset
        ldata["line_name"] = line_name
        ldata["line_label"] = line_label

        return ldata


def marginalize_line_fit_grid(grid, force_chinu=False, **kwargs):
    """ """

    sigmas = np.array([sig for sig in grid])
    dvs = grid[sigmas[0]]["dv_grid"]

    # Weights for trapzezoid integration
    shp = (len(sigmas), len(dvs))

    h_dv = np.ones(shp) * utils.trapz_dx(dvs)[None, :]
    val_dv = np.ones(shp) * dvs[None, :]

    h_sig = np.ones(shp) * utils.trapz_dx(np.log(sigmas))[:, None]
    val_sig = np.ones(shp) * np.log(sigmas)[:, None]

    h2d = h_dv * h_sig

    loss = np.array([grid[sig]["loss"] for sig in sigmas])
    models = np.array([grid[sig]["model"] for sig in sigmas])

    nsl = models.shape[2]

    lmin = np.nanmin(loss, axis=(0, 1))
    lnp = np.exp(-(loss - lmin) / 2 / (lmin / nsl) ** force_chinu)

    lnp_wht = (lnp.T * h2d.T).T
    lnp_norm = lnp_wht.sum(axis=(0, 1))
    lnp_wht /= lnp_norm

    nc = grid[sigmas[0]]["coeffs"].shape[1]

    cc = np.array([grid[sig]["coeffs"] for sig in sigmas])
    vc = np.array([grid[sig]["vcoeffs"] for sig in sigmas])

    # 1st and second moments weighted by lnp
    dv1 = (val_dv.T * lnp_wht.T).T.sum(axis=(0, 1))
    dv2 = ((val_dv[:, :, None, None] - dv1) ** 2 * lnp_wht).sum(axis=(0, 1))
    ds1 = (val_sig.T * lnp_wht.T).T.sum(axis=(0, 1))
    ds2 = ((val_sig[:, :, None, None] - ds1) ** 2 * lnp_wht).sum(axis=(0, 1))

    ccnum = np.array([(cc[:, :, k, :, :] * lnp_wht) for k in range(nc)])
    coeffs1 = np.nansum(ccnum, axis=(1, 2))  # / lnp_wht.sum(axis=(0,1))
    coeffs2 = np.nansum(
        [
            ((cc[:, :, k, :, :] - coeffs1[k, :, :]) ** 2 * lnp_wht)
            for k in range(nc)
        ],
        axis=(1, 2),
    )

    cvnum = np.array([(vc[:, :, k, :, :] * lnp_wht**2) for k in range(nc)])
    vcoeffs = np.nansum(cvnum, axis=(1, 2)) / (lnp_wht**2).sum(axis=(0, 1))

    # model
    mnum = np.array([(models[:, :, k, :, :] * lnp_wht) for k in range(nsl)])
    model1 = np.nansum(mnum, axis=(1, 2))
    model2 = np.nansum(
        [
            ((models[:, :, k, :, :] - model1[k, :, :]) ** 2 * lnp_wht)
            for k in range(nsl)
        ],
        axis=(1, 2),
    )

    cmin = np.nanmin(grid[sigmas[0]]["coeffs"] ** 2, axis=(0, 1))
    mask = cmin > 0

    result = {
        "lnp": lnp,
        "h2d": h2d,
        "lnp_wht": lnp_wht,
        "dv": dv1,
        "dv_var": dv2,
        "ln_sig": ds1,
        "ln_sig_var": ds2,
        "coeffs": coeffs1,
        "coeffs_var": coeffs2,
        "vcoeffs": vcoeffs,
        "model": model1,
        "model_var": model2,
        "mask": mask,
        "sigma_grid": sigmas,
        "dv_grid": dvs,
    }

    return result


class LinefitData:

    def __init__(self, cube, result, vrange=None, **kwargs):
        self.cube = cube
        self.result = result
        self.vrange = vrange

        if "sn_mask" not in self.result:
            self.result["sn_mask"] = np.zeros((self.nlines, *self.shape[1:]))

    @property
    def shape(self):
        return self.cube.shape

    @property
    def nlines(self):
        return self.result["nlines"]

    def to_fits_hdu(self, save=True, **kwargs):
        """ """
        h = self.cube.wcs_header

        h["REDSHIFT"] = self.result["redshift"]

        HDUL = pyfits.HDUList()

        # White light
        sl = self.result["slice"]
        nw = len(self.cube.wave)
        imin = np.maximum(0, sl.start - 512)
        imax = np.minimum(nw - 1, sl.stop + 512)

        wave_range = (self.cube.wave[imin], self.cube.wave[imax])

        h["WLWMIN"] = (wave_range[0], "White light image min wave")
        h["WLWMAX"] = (wave_range[1], "White light image min wave")
        h["BUNIT"] = "microJansky"

        mean_image, mean_weight = self.cube.mean_image(wave_range=wave_range)

        HDUL.append(pyfits.ImageHDU(data=mean_image, header=h, name="WLSCI"))
        HDUL.append(pyfits.ImageHDU(data=mean_weight, header=h, name="WLWHT"))

        h.pop("BUNIT")
        HDUL.append(
            pyfits.ImageHDU(
                data=self.result["bin"].astype(np.uint8), header=h, name="BIN"
            )
        )

        h["BUNIT"] = "km/s"
        HDUL.append(
            pyfits.ImageHDU(
                data=self.result["dv"].astype(np.int16), header=h, name="DV"
            )
        )

        HDUL.append(
            pyfits.ImageHDU(
                data=self.result["sig"].astype(np.uint16),
                header=h,
                name="SIGMA",
            )
        )

        nlines = self.result["nlines"]
        ncoeffs = self.result["ncoeffs"]

        h["BUNIT"] = "1e-20 cgs"
        for i in range(nlines):
            line_name = self.result["lines"][i]

            HDUL.append(
                pyfits.ImageHDU(
                    data=self.result["coeffs"][-nlines + i, :, :],
                    header=h,
                    name=f"{line_name}_SCI".upper(),
                )
            )

            HDUL.append(
                pyfits.ImageHDU(
                    data=self.result["vcoeffs"][-nlines + i, :, :],
                    header=h,
                    name=f"{line_name}_VAR".upper(),
                )
            )

            if "sn_mask" in self.result:
                HDUL.append(
                    pyfits.ImageHDU(
                        data=self.result["sn_mask"][i, :, :].astype(np.uint8),
                        header=h,
                        name=f"{line_name}_MASK".upper(),
                    )
                )

        h.pop("BUNIT")
        for i in range(ncoeffs - nlines):
            HDUL.append(
                pyfits.ImageHDU(
                    data=self.result["coeffs"][i, :, :],
                    header=h,
                    name=f"C{i}_SCI",
                )
            )

            HDUL.append(
                pyfits.ImageHDU(
                    data=self.result["vcoeffs"][i, :, :],
                    header=h,
                    name=f"C{i}_VAR",
                )
            )

        line_name = self.result["lines"][0]

        fits_file = (
            f"{self.cube.outroot}.spec.{line_name.lower()}.fits".replace(
                "oiii.png", "oiii-5007.fits"
            )
        )

        if save:
            msg = f"LinefitData: save {fits_file}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

            HDUL.writeto(fits_file, overwrite=True, output_verify="fix")

        return HDUL

    def line_figure_2d(
        self,
        x0=None,
        line_index=0,
        fig_xsize=12,
        vrange=None,
        recenter=False,
        use_err_scale=False,
        save=True,
        sn_threshold=4,
        discrete_sigma=True,
        show_contours=True,
        **kwargs,
    ):
        """ """
        import scipy.ndimage as nd
        from skimage.morphology import isotropic_dilation
        from astropy.visualization import simple_norm

        tight_colorbar = msautils.tight_colorbar

        line_name = self.result["lines"][line_index]
        nlines = self.result["nlines"]

        full_line_sn = (
            self.result["coeffs"] / np.sqrt(self.result["vcoeffs"])
        )[-nlines:, :, :]

        full_line_flux = self.result["coeffs"][-nlines:, :, :]
        full_line_dv = self.result["dv"]
        full_line_sig = self.result["sig"]
        full_line_bin = self.result["bin"]

        if use_err_scale:
            err_scale = self.result["err_scale"]
        else:
            err_scale = 1.0

        # weight_mask = prof['mean_weight'] > 0.1 * np.nanmedian(prof['mean_weight'])
        sl = self.result["slice"]

        nw = len(self.cube.wave)
        imin = np.maximum(0, sl.start - 512)
        imax = np.minimum(nw - 1, sl.stop + 512)
        mean_image, mean_weight = self.cube.mean_image(
            wave_range=(self.cube.wave[imin], self.cube.wave[imax])
        )
        weight_mask = mean_weight > 0.1 * np.nanmedian(mean_weight)
        if weight_mask.sum() < 0.1 * weight_mask.size:
            weight_mask = np.isfinite(mean_weight)

        max_line_sn = full_line_sn[line_index, :, :] / err_scale * weight_mask
        max_line_flux = full_line_flux[line_index, :, :] * weight_mask
        max_line_dv = full_line_dv * weight_mask
        max_line_err = max_line_flux / max_line_sn

        max_line_sn[~np.isfinite(full_line_dv + full_line_sig)] = np.nan
        max_line_flux[~np.isfinite(full_line_dv + full_line_sig)] = np.nan
        max_line_err[~np.isfinite(full_line_dv + full_line_sig)] = np.nan

        gmask = (max_line_sn > sn_threshold) & np.isfinite(
            full_line_dv + full_line_sig
        )

        gmask = nd.binary_opening(gmask, iterations=1)
        gmask = isotropic_dilation(gmask, radius=4)

        sh = self.cube.shape[1:]

        line_flux_err = max_line_flux / max_line_sn
        # bad = prof['mean_weight'] < 0.1*np.nanmedian(prof['mean_weight'])
        bad = line_flux_err > 5 * np.nanmedian(line_flux_err)
        # max_line_flux[bad] = np.nan

        line_sn_mask = np.ones_like(max_line_sn)
        line_sn_mask[
            nd.gaussian_filter(max_line_sn, 1) < sn_threshold * 3.0 / 4
        ] = np.nan
        line_sn_mask[~gmask] = np.nan
        line_sn_mask[~weight_mask] = np.nan

        print("xxx line_sn_mask", np.isfinite(line_sn_mask).sum())
        if np.isfinite(line_sn_mask).sum() == 0:
            if (full_line_sig > 0).sum() > 0:
                line_sn_mask = np.nan ** (1 - (full_line_sig > 0))
            else:
                line_sn_mask = np.ones_like(line_sn_mask)

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(fig_xsize, fig_xsize / 4 * self.cube.aspect),
            sharex=True,
            sharey=True,
        )

        vmax = 1.0e-3

        perc = np.nanpercentile(max_line_flux * line_sn_mask, [16, 50, 84])

        pixbin = (
            np.sum(self.cube.pixbin[sl, :, :], axis=0) / self.result["nsl"]
        )

        mean_sn = (mean_image - np.nanmedian(mean_image) * 0) * np.nanmedian(
            mean_weight
        )

        inorm = simple_norm(
            mean_sn,
            "log",
            vmin=-20,
            vmax=np.nanpercentile(mean_sn, 99),
            # vmax=np.nanmax(line_flux)*10,
            log_a=1.0e1,
        )

        axes[0].imshow(mean_sn, norm=inorm, cmap="RdGy")

        try:
            norm = simple_norm(
                max_line_flux,
                "log",
                vmin=perc[1] - 2 * (perc[1] - perc[0]),
                vmax=np.minimum(
                    perc[1] + 30 * (perc[2] - perc[0]),
                    np.nanmax(
                        max_line_flux[
                            (full_line_bin <= 1)
                            & (max_line_sn > 4)
                            & (pixbin > 0.95)
                        ]
                    ),
                ),
                log_a=1.0e1,
            )
        except:
            norm = simple_norm(
                max_line_flux,
                "log",
                vmin=-3 * np.nanmedian(max_line_err),
                vmax=50 * np.nanmedian(max_line_err),
                log_a=1.0e1,
            )

        axes[1].imshow(max_line_flux, norm=norm, cmap="bone_r")

        levels = [3, 5, 10, 20, 40, 80, 160, 320]

        if show_contours:
            for j in [0, 1]:
                axes[j].contour(
                    nd.gaussian_filter(max_line_sn * gmask, 0.5),
                    levels=levels[:1],
                    colors=[
                        "magenta" for v in np.linspace(0.2, 1, len(levels))
                    ][:1],
                    alpha=0.2,
                )

        max_line_dv_recenter = max_line_dv * 1
        if recenter:
            max_line_dv_recenter -= np.nanmedian(
                max_line_dv[np.isfinite(line_sn_mask)]
            )

        if vrange is None:
            vrange = self.vrange

        if vrange is None:
            perc = np.nanpercentile(
                max_line_dv_recenter * line_sn_mask, [10, 90]
            )
            # print("xxx", perc)
            vrange = [(np.ceil(np.max(np.abs(perc)) / 50) + 4) * 50] * 2
            vrange[0] *= -1
            self.vrange = vrange

        imsh_dv = axes[2].imshow(
            max_line_dv_recenter * line_sn_mask,
            vmin=vrange[0],
            vmax=vrange[1],
            cmap="RdYlBu_r",
        )

        nsigmas = len(self.result["sigmas"])
        sig_ix = np.interp(
            full_line_sig, self.result["sigmas"], range(nsigmas)
        )

        imsh_sig = axes[3].imshow(
            sig_ix * line_sn_mask,
            vmin=-0.5,
            vmax=nsigmas - 0.5,
            cmap=(
                plt.get_cmap("managua_r", nsigmas)
                if discrete_sigma
                else "managua_r"
            ),
        )

        if show_contours:
            for ax in axes[2:]:
                ax.contour(
                    nd.gaussian_filter(max_line_sn * gmask, 0.5),
                    levels=levels,
                    # colors=['0.6'] * len(levels),
                    colors=[
                        plt.cm.gray_r(v)
                        for v in np.linspace(0.2, 1, len(levels))
                    ],
                    alpha=0.8,
                )

        h = self.cube.header  # cube_hdu['SCI'].header
        pscale = np.sqrt(h["CD1_1"] ** 2 + h["CD1_2"] ** 2) * 3600.0
        dxt = 0.5
        ticks = np.arange(-40, 41, dxt) / pscale

        if x0 is None:
            xc = sh[1] / 2.0
            yc = sh[0] / 2.0
        else:
            xc, yc = x0

        xt = ticks + xc
        xtv = (xt > 0.05 * dxt / pscale) & (xt < sh[1] - 0.95 * dxt / pscale)
        yt = ticks + yc
        ytv = (yt > 0.05 * dxt / pscale) & (yt < sh[0] - 0.95 * dxt / pscale)

        for ax in axes:
            ax.grid()
            ax.set_xticks(xt[xtv])
            ax.set_yticks(yt[ytv])
            ax.set_xticklabels([])  # ticks[xtv] * pscale)
            ax.set_yticklabels([])  # (ticks[ytv] * pscale)

        axes[0].text(
            0.5,
            0.98,
            f"{self.cube.outroot}\nz={self.result['redshift']:.4f}",
            fontsize=6,
            ha="center",
            va="top",
            transform=axes[0].transAxes,
            bbox={"fc": "w", "alpha": 0.7, "ec": "None"},
        )

        if line_name not in LINE_LABELS_LATEX:
            print(f"use {line_name} as line label!")
            line_label = line_name + " @ "
        else:
            line_label = LINE_LABELS_LATEX[line_name] + " @ "

        line_label += r"$\lambda_\mathrm{obs}~=~x~\mathrm{\mu m}$".replace(
            "x",
            f"{LINE_WAVELENGTHS[line_name][0] * (1 + self.result['redshift']) / 1.e4:.4f}",
        )

        axes[1].text(
            0.5,
            0.98,
            line_label,
            fontsize=6,
            ha="center",
            va="top",
            transform=axes[1].transAxes,
            bbox={"fc": "w", "alpha": 0.7, "ec": "None"},
        )

        fig.tight_layout(pad=0.5)
        fig.tight_layout(pad=0.5)

        (_, cb_dv) = tight_colorbar(
            imsh_dv,
            fig,
            axes[2],
            sx=0.85,
            sy=0.02,
            loc="uc",
            pad=0.01,  # location=None,
            label=r"$\Delta v$ [km/s]",
            bbox_kwargs={"fc": "w", "alpha": 0.5, "ec": "None"},
            zorder=100,
            label_format=None,
            labelsize=7,
            colorbar_alpha=None,
        )

        (ax_cb_sig, cb_sig) = tight_colorbar(
            imsh_sig,
            fig,
            axes[3],
            sx=0.85,
            sy=0.02,
            loc="uc",
            pad=0.01,  # location=None,
            label=r"$\sigma$ [km/s]",
            bbox_kwargs={"fc": "w", "alpha": 0.5, "ec": "None"},
            zorder=100,
            label_format=None,
            labelsize=7,
            colorbar_alpha=None,
        )

        cb_sig.set_ticks(range(nsigmas))
        sig_labels = [f"{s:.0f}" for s in self.result["sigmas"]]
        if len(sig_labels) > 6:
            ax_cb_sig.tick_params(labelsize=6)

        if len(sig_labels) > 11:
            for i in range(len(sig_labels))[1::2]:
                sig_labels[i] = ""

        cb_sig.set_ticklabels(sig_labels)

        fig_file = f"{self.cube.outroot}.spec.{line_name.lower()}.png".replace(
            "oiii.png", "oiii-5007.png"
        )

        self.result["sn_mask"][line_index, :, :] = np.isfinite(line_sn_mask)

        if save:
            msg = f"LinefitData: save {fig_file}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
            fig.savefig(fig_file)

        return fig

    def get_line_pixels(
        self, dx_list=[-2, 0, 2], dy_list=[-2, 0, 2], line_index=0, **kwargs
    ):
        """ """
        nlines = self.result["nlines"]

        line_sn = (self.result["coeffs"] / np.sqrt(self.result["vcoeffs"]))[
            -nlines + line_index, :, :
        ]

        if "sn_mask" in self.result:
            line_sn *= self.result["sn_mask"][line_index, :, :]

        i, j = np.unravel_index(np.nanargmax(line_sn), line_sn.shape)
        pix = []
        for di in dy_list:
            for dj in dx_list:
                pix.append([i + di, j + dj])

        return pix

    def line_regions(self, line_index=0, **kwargs):
        """
        Label and compute properties of "islands" in the S/N mask for an emission line
        """
        from skimage.measure import label, regionprops

        labels = label(nd.binary_opening(self.result["sn_mask"][0, :, :]))

        nlines = self.result["nlines"]

        line_sn = (self.result["coeffs"] / np.sqrt(self.result["vcoeffs"]))[
            -nlines + line_index, :, :
        ] * 1

        line_sn[~np.isfinite(line_sn)] = 0
        props = regionprops(labels, intensity_image=line_sn)
        rows = []
        for p in props:
            row = {}
            for k in [
                "label",
                "centroid",
                "centroid_weighted",
                "area",
                "bbox",
                "intensity_max",
                "axis_major_length",
                "axis_minor_length",
                "orientation",
            ]:
                row[k] = getattr(p, k)
            rows.append(row)

        ptab = utils.GTable(rows=rows)
        so = np.argsort(ptab["intensity_max"])[::-1]
        ptab = ptab[so]
        props = [props[j] for j in so]

        return labels, props, ptab

    def line_region_pix(
        self,
        region_index=0,
        line_index=0,
        asteps=[-0.3, 0, 0.3],
        bsteps=[-0.3, 0, 0.3],
        dilate_mask=1,
        first_axis="major",
        plus=False,
        **kwargs,
    ):
        """
        Get pixel list within a line region
        """
        from skimage.morphology import isotropic_dilation

        labels, props, ptab = self.line_regions(
            line_index=line_index, **kwargs
        )
        if labels is None:
            return None, None

        if region_index > len(ptab) - 1:
            return None, None

        row = ptab[region_index]

        arr = []
        arad = row["axis_major_length"] / 2.0
        brad = row["axis_minor_length"] / 2.0

        if first_axis == "major":
            for i in bsteps:
                for j in asteps:
                    if plus & (j != 0) & (i != 0):
                        continue
                    arr.append([j * arad, i * brad])
        else:
            for j in asteps:
                for i in bsteps:
                    if plus & (j != 0) & (i != 0):
                        continue
                    arr.append([j * arad, i * brad])

        mask = labels == row["label"]
        if dilate_mask:
            mask = isotropic_dilation(mask, radius=dilate_mask)

        # print("xxx orientation: ", row['orientation'] / np.pi * 180 - 90)
        orientation_angle = row["orientation"] / np.pi * 180 - 90
        if orientation_angle < -90:
            orientation_angle += 180

        rot = rotation_matrix(orientation_angle)
        coo = (
            np.array(arr).dot(rot) + np.array(row["centroid_weighted"][::-1])
        ).T

        cpix = np.round(coo).astype(int)

        in_mask = np.array([
            mask[cp[1], cp[2]]
            for cp in cpix.T
        ])

        coo = coo[:, in_mask].T
        ind = np.round(coo).astype(int)[:, ::-1]

        return coo, ind

    def plot_pixel_spectra(
        self,
        figsize=(9, 4),
        pix=None,
        pad=32,
        line_index=0,
        renorm=True,
        y_offset=1.2,
        sorty=True,
        cmap=msautils.ClippedColormap(plt.cm.Spectral, vmin=0, vmax=1),
        **kwargs,
    ):
        """ """
        if pix is None:
            coo, pix = self.line_region_pix(line_index=line_index, **kwargs)
            # if pix is not None:
            #     sorty = False

        if pix is None:
            pix = self.get_line_pixels(line_index=line_index, **kwargs)

        if sorty:
            py = [p[0] for p in pix]
            psort = np.argsort(py)
            if y_offset < 0:
                psort = psort[::-1]
        else:
            psort = np.arange(len(pix), dtype=int)

        npix = len(pix)

        sl = self.result["slice"]
        imin = np.maximum(0, sl.start - pad)
        imax = np.minimum(self.cube.shape[0] - 1, sl.stop + pad)
        sl2 = slice(imin, imax)

        nlines = self.result["nlines"]

        line_sn = (self.result["coeffs"] / np.sqrt(self.result["vcoeffs"]))[
            -nlines + line_index, :, :
        ]

        if "sn_mask" in self.result:
            line_sn *= self.result["sn_mask"][line_index, :, :]

        line_sn[line_sn == 0] = np.nan

        sn_mask_im = np.isfinite(line_sn) & (line_sn > 0)
        if sn_mask_im.sum() == 0:
            sn_mask_im = np.nanmax(self.result["coeffs"] ** 0, axis=0) > 0

        xpm = self.cube.xp[sn_mask_im]
        ypm = self.cube.yp[sn_mask_im]
        aspect = (ypm.max() - ypm.min()) / (xpm.max() - xpm.min())

        fig, axes = plt.subplots(1, 2, figsize=figsize, width_ratios=[1, 2])

        inorm = simple_norm(
            line_sn,
            "log",
            vmin=-3,
            vmax=np.nanpercentile(line_sn, 99),
            # vmax=np.nanmax(line_flux)*10,
            log_a=1.0e1,
        )

        axes[0].imshow(line_sn, norm=inorm, cmap="bone_r")

        offset = 0.0

        ax = axes[1]

        w1 = self.cube.wave[sl]
        w2 = self.cube.wave[sl2]

        dw2 = w2.max() - w2.min()

        for ki, k in enumerate(psort):
            (i, j) = pix[k]
            model_ij = self.result["model"][:, i, j] * 1.0
            if renorm:
                mmax = np.nanmax(model_ij)
            else:
                mmax = 1.0

            fnu = (self.result["full_sci"][:, i, j] / self.cube.sens)[sl2]
            flam = fnu * self.cube.spec["to_flam"][sl2]

            # flam[flam < 0] = np.nan
            # model_ij[model_ij < 0] = np.nan

            pl = ax.plot(
                w2, flam / mmax + offset, color="0.5", alpha=0.3, zorder=1
            )

            pl = ax.plot(
                w1,
                model_ij / mmax + offset,
                alpha=0.3,
                zorder=2,
                color=cmap(ki / (npix - 1)),
            )

            axes[0].scatter(
                [j], [i], marker="s", fc=pl[0].get_color(), ec="w", alpha=0.9
            )

            if np.abs(y_offset) > 0:

                label = f"{i:>2}, {j:>2}"
                label += f'  {self.result["dv"][i,j]:>4.0f}  {self.result["sig"][i,j]:>4.0f}'
                ax.text(
                    w2[0] - 0.23 * dw2,
                    model_ij[0] / mmax + offset + 0.1 * y_offset,
                    label,
                    ha="left",
                    va="center",
                    fontsize=6,
                    fontfamily="monospace",
                    color=pl[0].get_color(),
                    bbox={"fc": "w", "alpha": 0.8, "ec": "None"},
                    # backgroundcolor='w',
                )

            offset += y_offset

        ax = axes[0]
        ax.set_ylim(ypm.min() - 4, ypm.max() + 4)
        ax.set_xlim(xpm.min() - 4, xpm.max() + 4)
        ax.tick_params(labelsize=6)

        if renorm:
            axes[1].set_yticklabels([])
            if y_offset > 0:
                axes[1].set_ylim(-0.5, (npix + 0.5) * y_offset)
            else:
                axes[1].set_ylim((npix - 0.5) * y_offset, 1.4)

        if np.abs(y_offset) > 0:
            axes[1].set_xlim(w2[0] - 0.25 * dw2, w2[-1] + 0.05 * dw2)

        axes[1].set_xlabel(r"$\lambda_\mathrm{obs}$")

        xref = []
        for li in self.result["lines"]:
            for lwx in LINE_WAVELENGTHS[li]:
                xref.append(lwx / 1.0e4 * (1 + self.result["redshift"]))

        ylim = axes[1].get_ylim()
        axes[1].vlines(
            [xref],
            *ylim,
            color="magenta",
            linestyle="-",
            alpha=0.1,
            lw=3,
            zorder=-1,
        )
        axes[1].set_ylim(*ylim)

        if self.result["redshift"] > 0.02:

            ax2 = axes[1].twiny()
            ax2.set_xlim(
                *(np.array(axes[1].get_xlim()) / (1 + self.result["redshift"]))
            )
            ax2.set_xlabel(
                r"$\lambda_\mathrm{rest}~~z=xx$".replace(
                    "xx", f"{self.result['redshift']:.4f}"
                )
            )

        for ax in axes:
            ax.grid()

        fig.tight_layout(pad=1)

        return fig
