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

        self.process_pixel_table()

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
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        self.input = jwst.datamodels.open(self.file.replace("_rate", "_cal"))

        if dilate_failed_open:
            self._dilate_failed_open_mask()

        self.roll = -self.input.meta.aperture.position_angle
        self.ra_ref = self.input.meta.wcsinfo.ra_ref
        self.dec_ref = self.input.meta.wcsinfo.dec_ref
        self.target_ra = self.input.meta.target.proposer_ra
        self.target_dec = self.input.meta.target.proposer_dec
        if self.target_ra is None:
            self.target_ra = self.input.meta.target.ra
            self.target_dec = self.input.meta.target.dec

        theta = self.roll / 180 * np.pi
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

    def preprocess(
        self,
        do_flatfield=True,
        do_photom=True,
        extend_wavelengths=True,
        local_sflat=True,
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
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        from jwst.assign_wcs import AssignWcsStep
        from jwst.msaflagopen import MSAFlagOpenStep
        from jwst.flatfield import FlatFieldStep
        from jwst.photom import PhotomStep

        from grizli import jwst_utils
        from .pipeline_extended import assign_wcs_with_extended
        from . import utils as msautils

        do_corr = True
        with pyfits.open(self.file) as im:
            if "ONEFEXP" in im[0].header:
                if im[0].header["ONEFEXP"]:
                    do_corr = False

        if do_corr:
            jwst_utils.exposure_oneoverf_correction(
                self.file,
                erode_mask=False,
                in_place=True,
                axis=0,
                deg_pix=2048,
            )

        # jwst_utils.exposure_oneoverf_correction(
        #     self.file, erode_mask=False, in_place=True, axis=1, deg_pix=2048
        # )

        input = jwst.datamodels.open(self.file)
        _ = msautils.slit_hot_pixels(
            input, verbose=True, max_allowed_flagged=4096 * 8
        )

        if local_sflat:
            inst_ = input.meta.instrument.instance
            sflat_file = "sflat_{grating}-{filter}_{detector}.fits".format(
                **inst_
            ).lower()
            # print("xxx ", sflat_file)

            if not os.path.exists(sflat_file):
                URL_ = (
                    "https://s3.amazonaws.com/msaexp-nirspec/ifu-sflat/"
                    + sflat_file
                )
                msg = f"Download SFLAT file from {URL_}"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)

                sflat_file = download_file(URL_, cache=True)

            if os.path.exists(sflat_file):

                with pyfits.open(sflat_file) as sflat_:

                    outside_sflat = (~np.isfinite(sflat_[0].data)) & (
                        input.dq & 1 == 0
                    )
                    med_outside = np.nanmedian(input.data[outside_sflat])
                    msg = f"{self.file} SFLAT: {sflat_file} med={med_outside:.3f}"
                    utils.log_comment(utils.LOGFILE, msg, verbose=True)

                    input.data -= med_outside

                    input.data /= sflat_[0].data
                    input.err /= sflat_[0].data
                    input.var_poisson /= sflat_[0].data ** 2
                    input.dq |= ((~np.isfinite(input.data)) * 1).astype(
                        input.dq.dtype
                    )

        if extend_wavelengths:
            input_wcs = assign_wcs_with_extended(input)
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
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

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
        BAD_PIXEL_FLAG=msautils.BAD_PIXEL_FLAG,
        image_attrs=["data", "dq", "var_poisson", "var_rnoise", "err"],
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

        theta = self.roll / 180 * np.pi
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        cube = {}

        for c in image_attrs:
            cube[c] = []

        coord_cols = ["slice", "dx", "dy", "lam", "dra", "dde", "xpix", "ypix"]
        for c in coord_cols:
            cube[c] = []

        shapes = []
        ypix, xpix = np.indices(self.input.data.shape, dtype=np.int16)

        for i, l in enumerate(lam_data):
            slx = slice(*x_slice[i])
            sly = slice(*y_slice[i])

            # cut = input_photom.data[sly, slx]
            for c in image_attrs:
                cube[c].append(getattr(self.input, c)[sly, slx].flatten())

            if mask_params is not None:
                msk = np.isfinite(self.input.data[sly, slx])
                msk = nd.binary_closing(msk, iterations=mask_params[0])
                shrink = nd.binary_erosion(
                    msk, structure=np.ones((mask_params[1], 1), dtype=bool)
                )
                cube["dq"][i][~shrink.flatten()] |= 1024

            dra = (
                (coord1_data[i] - ra_ref)
                * np.cos(dec_ref / 180 * np.pi)
                * 3600
            )
            dde = (coord2_data[i] - dec_ref) * 3600
            dx = np.array([dra, dde]).T.dot(rot).T

            sh = coord1_data[i].shape
            shapes.append(sh)
            cube["slice"].append(np.ones(sh, dtype=int).flatten() * i)
            cube["dx"].append(dx[0].flatten())
            cube["dy"].append(dx[1].flatten())
            cube["lam"].append(lam_data[i].flatten())
            cube["dra"].append(dra.flatten())
            cube["dde"].append(dde.flatten())
            cube["xpix"].append(xpix[sly, slx].flatten())
            cube["ypix"].append(ypix[sly, slx].flatten())

        for c in cube:
            cube[c] = np.hstack(cube[c])

        cube["var_total"] = cube["var_poisson"] + cube["var_rnoise"]
        cube = utils.GTable(cube)
        cube.meta["RA_REF"] = ra_ref
        cube.meta["DEC_REF"] = dec_ref
        cube.meta["NSLICE"] = len(shapes)
        cube.meta["FILE"] = self.file

        for meta in [
            self.input.meta.instrument.instance,
            self.input.meta.exposure.instance,
            self.input.meta.pointing.instance,
            self.input.meta.aperture.instance,
            self.input.meta.target.instance,
        ]:
            for k in meta:
                cube.meta[k] = meta[k]

        if BAD_PIXEL_FLAG is not None:
            # valid_dq = (
            #     cube["dq"]
            #     & (jwst.datamodels.dqflags.pixel["MSA_FAILED_OPEN"] | 1)
            # ) == 0
            valid_dq = (cube["dq"] & BAD_PIXEL_FLAG) == 0
            msg = f"{self.file} keep {valid_dq.sum()} ({valid_dq.sum()/len(valid_dq)*100:.1f}%) pixels for bad DQ = {BAD_PIXEL_FLAG}"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

            cube = cube[valid_dq]

        for i in range(len(shapes)):
            cube.meta[f"XSTRT{i}"] = x_slice[i][0]
            cube.meta[f"XSIZE{i}"] = shapes[i][1]
            cube.meta[f"YSTRT{i}"] = y_slice[i][0]
            cube.meta[f"YSIZE{i}"] = shapes[i][0]

        return cube

    def process_pixel_table(self, load_existing=True, **kwargs):
        """ """
        ptab_file = self.file.replace("_rate", "_cal").replace(
            "_cal.fits", "_ptab.fits"
        )
        if os.path.exists(ptab_file) & load_existing:
            msg = f"process_pixel_table: load {ptab_file}"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

            self.ptab = utils.read_catalog(ptab_file)
            self.ptab.meta = meta_lowercase(self.ptab.meta)
            self.target_ra = self.ra_ref = self.ptab.meta["ra_ref"]
            self.target_dec = self.dec_ref = self.ptab.meta["dec_ref"]

        elif self.input is not None:
            self.ptab = self.pixel_table(**kwargs)
            self.ptab.meta = meta_lowercase(self.ptab.meta)
            try:
                pixel_table_background(self.ptab, **kwargs)
            except:
                msg = f"pixel_table_background failed!"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                self.ptab["sky"] = 0

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
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

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


def plot_cube_strips(ptab, figsize=(10, 5), cmap="bone_r"):
    """
    Plot spatial slices of a cube pixel table
    """
    from scipy.spatial import ConvexHull

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    image_attrs = ["data", "dq", "var_poisson", "var_rnoise", "err"]

    for cube in ptab:
        un = utils.Unique(cube["slice"], verbose=False)
        ok = np.isfinite(cube["dx"] + cube["dy"])

        hull = ConvexHull(np.array([cube["dx"][ok], cube["dy"][ok]]).T)
        vert = hull.vertices

        # Slice coordinates
        ax = axes[0]
        pl = ax.plot(
            cube["dx"][ok][vert],
            cube["dy"][ok][vert],
            alpha=0.5,
            zorder=100,
        )

        color = pl[0].get_color()

        for i, v in enumerate(un.values):
            ixi = un[v] & (
                np.abs(cube["lam"] - np.nanmedian(cube["lam"])) < 0.01
            )
            ax.scatter(
                cube["dx"][ixi], cube["dy"][ixi], alpha=0.1, color=color
            )

        # Cube data
        ax = axes[1]

        ax.plot(
            cube["dx"][ok][vert],
            cube["dy"][ok][vert],
            alpha=0.5,
            zorder=100,
            color=color,
        )

        valid_dq = (
            cube["dq"] & (jwst.datamodels.dqflags.pixel["MSA_FAILED_OPEN"] | 1)
        ) == 0

        for c in image_attrs:
            valid_dq &= np.isfinite(cube[c])

        valid_dq &= cube["var_poisson"] > 0
        valid_dq &= cube["var_rnoise"] > 0

        test = np.isfinite(cube["data"])
        test &= np.abs(cube["lam"] - np.nanmedian(cube["lam"])) < 0.02
        test &= valid_dq

        rsize = 0.0
        rx = np.random.rand(test.sum()) * rsize - rsize / 2
        ry = np.random.rand(test.sum()) * rsize - rsize / 2

        ax.scatter(
            cube["dx"][test] + rx,
            cube["dy"][test] + ry,
            c=cube["data"][test],
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


def objfun_scale_rnoise(theta, resid_num, resid_rnoise, resid_poisson):
    """
    objective function for scaling readnoise extension
    """
    norm = scipy.stats.norm(
        scale=np.sqrt(theta * resid_rnoise + resid_poisson)
    )
    lnp = -norm.logpdf(resid_num).sum()
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


def pixel_table_background(
    ptab,
    BAD_PIXEL_FLAG=msautils.BAD_PIXEL_FLAG,
    sky_center=(0, 0),
    sky_annulus=(0.5, 1.0),
    make_plot=True,
    **kwargs,
):
    """
    Extract background spectrum from pixel table cube data
    """
    meta = meta_lowercase(ptab.meta)

    grating = meta["grating"].upper()

    wave_grid = msautils.get_standard_wavelength_grid(grating, sample=1.05)

    if grating == "PRISM":
        nspl = 71
    else:
        nspl = 21

    nw = len(wave_grid)
    xspl = np.arange(0, nw) / nw

    bspl = utils.bspline_templates(
        xspl, df=nspl, minmax=(0, 1), get_matrix=True
    )

    valid_dq = (ptab["dq"] & BAD_PIXEL_FLAG) == 0

    for c in ptab.colnames:
        valid_dq &= np.isfinite(ptab[c])

    # valid_dq &= ptab['var_poisson'] > 0
    valid_dq &= ptab["var_rnoise"] > 0

    poisson_threshold = np.nanpercentile(ptab["var_poisson"][valid_dq], 97)
    rnoise_threshold = np.nanpercentile(ptab["var_rnoise"][valid_dq], 97)

    # print(f"xxx {poisson_threshold} {rnoise_threshold}")
    valid_dq &= ptab["var_poisson"] < 100
    valid_dq &= ptab["var_rnoise"] < 100

    valid_dq &= np.isfinite(ptab["lam"])
    valid_dq &= ptab["lam"] > 0.65

    ptab["valid"] = valid_dq & True

    ###
    if sky_annulus is None:
        coord_test = ptab["valid"] & True
    else:
        dr = np.sqrt(
            (ptab["dx"] - sky_center[0]) ** 2
            + (ptab["dy"] - sky_center[1]) ** 2
        )
        coord_test = ptab["valid"] & (dr > sky_annulus[0])
        coord_test &= ptab["valid"] & (dr < sky_annulus[1])

    if coord_test.sum() > 0:
        test = coord_test
    else:
        test = ptab["valid"] & True

    test &= ptab["data"] > -6 * np.sqrt(ptab["var_rnoise"])
    test &= ptab["data"] < 10

    if grating.endswith("H"):
        test &= ptab["data"] < 20 * np.sqrt(ptab["var_rnoise"])

    if make_plot:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax = axes[0]

    ok = np.isfinite(ptab["lam"][test])

    for _iter in range(3):

        so = np.argsort(ptab["lam"][test][ok])
        xsky = ptab["lam"][test][ok][so]
        sky_med = nd.median_filter(ptab["data"][test][ok][so], 64)
        if make_plot:
            ax.plot(ptab["lam"][test][ok][so], sky_med, alpha=0.5)

        sky_interp = np.interp(ptab["lam"][test], xsky, sky_med)
        ok = np.abs(ptab["data"][test] - sky_interp) < 5 * np.sqrt(
            ptab["var_total"][test]
        )
        ok &= ptab["var_total"][test] < 8 * np.median(
            ptab["var_total"][test][ok]
        )
        msg = f"iter {_iter} {ok.sum()}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

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
            ptab["lam"][test][ok][::skip],
            ptab["data"][test][ok][::skip],
            alpha=0.1,
            zorder=-10,
        )
        ax.set_ylim(-1, 5)

    resid_num = ptab["data"][test][ok] - np.interp(
        ptab["lam"][test][ok], xsky, sky_med
    )
    resid_rnoise = ptab["var_rnoise"][test][ok]
    resid_poisson = ptab["var_poisson"][test][ok]
    resid = resid_num / np.sqrt(resid_rnoise + resid_poisson)

    if make_plot:
        ax = axes[1]
        ax.scatter(
            ptab["lam"][test][ok][::skip],
            resid[::skip],
            c=ptab["dy"][test][ok][::skip],
            alpha=0.1,
            marker=".",
        )

    args = (resid_num, resid_rnoise, resid_poisson)
    res = minimize(
        objfun_scale_rnoise,
        x0=1.0,
        args=(resid_num, resid_rnoise, resid_poisson),
        method="bfgs",
        tol=1.0e-6,
    )
    scale_rnoise = res.x[0]

    msg = f"  uncertainty nmad={utils.nmad(resid):.3f} std={np.std(resid):.3f}  scale_rnoise={np.sqrt(scale_rnoise):.3f}"
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    ptab.meta["rnoise_nmad"] = utils.nmad(resid)
    ptab.meta["rnoise_std"] = np.std(resid)
    ptab.meta["scale_rnoise"] = scale_rnoise

    ptab["var_rnoise"] *= scale_rnoise
    ptab["var_total"] = ptab["var_rnoise"] + ptab["var_poisson"]

    xspl_int = np.interp(ptab["lam"], wave_grid, xspl)

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

    sky = np.interp(ptab["lam"], wave_grid, splm)  #
    # sky = xbspl.dot(coeffs[0])

    # ptab['sky'] = sky #np.interp(ptab['lam'], xsky, sky_med)
    ptab["sky"] = np.interp(ptab["lam"], xsky, sky_med)
    ptab["valid"] &= (ptab["data"] - ptab["sky"]) > -9 * np.sqrt(
        ptab["var_total"]
    )

    ptab["valid"] &= ptab["lam"] > np.nanpercentile(ptab["lam"], 2)
    ptab["valid"] &= ptab["lam"] < np.nanpercentile(ptab["lam"], 98)


def slice_corners(
    input,
    coord_system="skyalign",
    slice_indices=range(30),
    slice_wavelength_range=None,
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

    Notes
    -----
    Returns
    -------
    min and max spatial coordinates and wavelength for slice.

    """
    from jwst.assign_wcs import nirspec
    from jwst.assign_wcs.util import wrap_ra
    from gwcs import wcstools
    from tqdm import tqdm

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


def query_ifu_exposures(
    obsid="04056002001",
    grating="G395H",
    filter=None,
    download=True,
    exposure_type="cal",
    extend_wavelengths=True,
    detectors=None,
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

    query += mastquery.jwst.make_query_filter("grating", values=[grating])
    if filter is not None:
        query += mastquery.jwst.make_query_filter("filter", values=[filter])

    query += mastquery.jwst.make_query_filter("obs_id", text=f"V{obsid}%")

    query += mastquery.jwst.make_query_filter("is_imprt", values=["f"])

    if detectors is None:
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

    query += mastquery.jwst.make_query_filter("detector", values=detectors)

    msg = f"QUERY = {query}"
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

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

    msg = f"Found {len(res)} exposures"
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    if download:
        mast = mastquery.utils.download_from_mast(res)
    else:
        mast = None

    return res, mast


def drizzle_cube_data(
    ptab, wave_sample=1.05, pixel_size=0.1, pixfrac=0.75, side=2.2, **kwargs
):
    """
    Drizzle resample cube
    """
    from tqdm import tqdm

    Nside = int(np.ceil(side / pixel_size))
    xbin = np.arange(-Nside, Nside + 1) * pixel_size - pixel_size / 2
    ybin = xbin * 1

    wave_grid = msautils.get_standard_wavelength_grid(
        ptab.meta["grating"], sample=wave_sample
    )

    wbin = msautils.array_to_bin_edges(wave_grid)

    nx = len(xbin) - 1
    ny = len(ybin) - 1

    num = np.zeros((len(wave_grid), ny, nx))
    den = np.zeros((len(wave_grid), ny, nx))
    vnum = np.zeros((len(wave_grid), ny, nx))

    no_dq = (ptab["dq"] - (ptab["dq"] & 4)) == 0

    for k in tqdm(range(len(wave_grid))):

        sub = (ptab["lam"] > wbin[k]) & (ptab["lam"] < wbin[k + 1])
        sub &= ptab["valid"]
        sub &= no_dq
        if sub.sum() == 0:
            continue

        if 1:
            ysl = ptab["dy"][sub] * 1
            xsl = ptab["dx"][sub] * 1
        else:
            ysl = ptab["dde"][sub] * 1
            xsl = -1 * ptab["dra"][sub]

        data = (ptab["data"] - ptab["sky"])[sub] * 1
        var = (ptab["var_total"])[sub] * 1
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
    pixel_size=0.1,
    pixfrac=0.75,
    side=2.2,
    **kwargs,
):
    """
    Drizzle pixel table into a rectified cube

    Parameters
    ----------
    ptab : `~astropy.table.Table`

    ... TBD

    """

    Nside = int(np.ceil(side / pixel_size))
    xbin = np.arange(-Nside, Nside + 1) * pixel_size - pixel_size / 2
    ybin = xbin * 1

    wave_grid = msautils.get_standard_wavelength_grid(
        ptab.meta["grating"], sample=wave_sample
    )

    wbin = msautils.array_to_bin_edges(wave_grid)

    meta = ptab.meta

    pa_aper = meta["position_angle"]

    hdul = utils.make_wcsheader(
        meta["ra_ref"],
        meta["dec_ref"],
        size=2 * Nside * pixel_size,
        pixscale=pixel_size,
        get_hdu=True,
        theta=pa_aper,
    )
    hdul.header["NAXIS"] = 3

    hdul.header["CUNIT3"] = "Angstrom"

    hdul.header["CD1_1"] *= -1
    hdul.header["CD1_2"] *= -1

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
    outroot=None,
    use_first_center=True,
    files=None,
    make_drizzled=True,
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
        res, mast = query_ifu_exposures(obsid=obsid, grating=grating, **kwargs)
        if res is None:
            return None

        files = [os.path.basename(file) for file in res["dataURI"]]
        files.sort()
    else:
        obsid = files[0][2:13]
        with pyfits.open(files[0]) as im:
            grating = im[0].header["GRATING"]

    msg = f"ifu_pipeline: file={files[0]}  {obsid} {grating}"
    utils.LOGFILE = f"cube-{obsid}-{grating}.log.txt".lower()
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    # Initialize
    cubes = []
    for file in files:
        cube = ExposureCube(file, **kwargs)
        cubes.append(cube)

    # Process pixel tables
    for cube in cubes:
        # Process pixel table
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

    #### TBD: saturated mask
    for i, cube in enumerate(cubes):
        cube.ptab["exposure"] = i

    ptabs = [cube.ptab for cube in cubes]

    try:
        SOURCE = ptabs[0].meta["proposer_name"].lower().replace(" ", "_")
    except:
        SOURCE = "indef"

    if outroot is None:
        outroot = f"cube-{obsid}_{grating}-{ptabs[0].meta['filter']}_{SOURCE}".lower()

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

    if make_drizzled:
        num, den, vnum = drizzle_cube_data(ptab, **kwargs)
        hdul = make_drizzle_hdul(ptab, num, den, vnum, **kwargs)

        for ext in range(3):
            hdul[ext].header["SRCNAME"] = SOURCE
            for i, file_ in enumerate(files):
                hdul[ext].header[f"FILE{i:04d}"] = file_

        for i, file_ in enumerate(files):
            ptab.meta[f"file{i:04d}"] = file_

        cube_file = f"{outroot}.fits"
        msg = f"cube_file: {cube_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        hdul.writeto(cube_file, overwrite=True)
    else:
        hdul = None

    return outroot, cubes, ptab, hdul
