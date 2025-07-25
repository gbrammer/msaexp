import os
import glob
import inspect
import time
from collections import OrderedDict

import yaml

import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage as nd
from scipy.special import huber
from scipy.optimize import minimize

import astropy.io.fits as pyfits

import jwst.datamodels
import jwst

from grizli import utils
from . import utils as msautils

utils.LOGFILE = "/tmp/msaexp_slit_combine.log.txt"
VERBOSE_LOG = True

# Cross-dispersion pixel scale computed from one of the fixed slits
PIX_SCALE = 0.10544

# Default MSA nod offset
MSA_NOD_ARCSEC = 0.529

EVAL_COUNT = 0
CHI2_MASK = None
SKIP_COUNT = 10000
CENTER_WIDTH = 0.1
CENTER_PRIOR = 0.0
SIGMA_PRIOR = 0.6

HUBER_ALPHA = 7

# Mask parameters
PRISM_MAX_VALID = 10000
PRISM_MIN_VALID_SN = -3
# Mask pixels above this percentile of the RN arrays
# RNOISE_THRESHOLD = 95 # version <= 3
RNOISE_THRESHOLD = 99.5

SPLINE_BAR_GRATINGS = [
    "PRISM",
    "G395M",
    "G235M",
    "G140M",
    "G395H",
    "G235H",
    "G140H",
]

WING_SIGMA = 2.0
SCALE_FWHM = 1.0
DEFAULT_WINGS = None
WINGS_XOFF = None

__all__ = [
    "split_visit_groups",
    "SlitGroup",
    "pseudo_drizzle",
    "extract_spectra",
]


def split_visit_groups(
    files,
    join=[0, 3],
    gratings=["PRISM"],
    split_uncover=True,
    **kwargs,
):
    """
    Compute groupings of `SlitModel` files based on exposure, visit, detector,
    slit_id

    Parameters
    ----------
    files : list
        List of `SlitModel` files

    join : list
        Indices of ``files[i].split('[._]') + GRATING`` to join as a group

    gratings : list
        List of NIRSpec gratings to consider

    split_uncover : bool, optional
        Whether to split UNCOVER sub groups, default is True

    Returns
    -------
    groups : dict
        File groups
    """
    keys = []
    all_files = []
    for file in files:
        with pyfits.open(file) as im:
            if im[0].header["GRATING"] not in gratings:
                continue

            fk = "_".join(
                [
                    os.path.basename(file).replace(".", "_").split("_")[i]
                    for i in join
                ]
            )

            # key = f"{fk}-{im[0].header['GRATING']}"
            if "SPAT_NUM" in im[0].header:
                fk += f'_dith{im[0].header["SPAT_NUM"]}'

            key = f"{fk}-{im[0].header['GRATING']}-{im[0].header['FILTER']}"

            keys.append(key.lower())
            all_files.append(file)

    all_files = np.array(all_files)

    keys = np.array(keys)
    un = utils.Unique(keys, verbose=False)
    groups = {}
    for k in np.unique(keys):
        test_field = (un[k].sum() % 6 == 0) & ("jw02561" in files[0])
        test_field |= (un[k].sum() % 4 == 0) & ("jw01810" in files[0])
        test_field |= (un[k].sum() % 4 == 0) & ("jw01324" in files[0])
        test_field |= (un[k].sum() % 6 == 0) & ("jw01324" in files[0])
        test_field |= (un[k].sum() % 6 == 0) & ("jw06368" in files[0])
        test_field &= split_uncover > 0
        test_field |= split_uncover == 16

        if test_field:
            msg = "split_visit_groups: split sub groups (uncover, glass, bluejay, capers) "
            msg += f"{k} N={un[k].sum()}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

            ksp = k.split("-")

            set1 = all_files[un[k]][0::2].tolist()
            set2 = all_files[un[k]][1::2].tolist()

            if (split_uncover == 2) & ("jw02561" in files[0]):
                # outer nods for spatial information
                set1 = set1[1::3] + set1[2::3]
                set2 = set2[1::3] + set2[2::3]
            elif ("jw06368" in files[0]):
                fk = all_files[un[k]].tolist()
                set1 = []
                set2 = []
                for f in fk:
                    if f.split('_')[1] in ['03101','07101','11101']:
                        set1.append(f)
                    else:
                        set2.append(f)

            if len(set1) > 0:
                groups[ksp[0] + "a-" + "-".join(ksp[1:])] = set1
            if len(set2) > 0:
                groups[ksp[0] + "b-" + "-".join(ksp[1:])] = set2

        else:
            groups[k] = np.array(all_files)[un[k]].tolist()

    return groups


LOOKUP_PRF = None
LOOKUP_PRF_ORDER = 1


def slit_prf_fraction(
    wave,
    sigma=0.0,
    x_pos=0.0,
    slit_width=0.2,
    pixel_scale=PIX_SCALE,
    verbose=True,
):
    """
    Rough slit-loss correction given derived source
    width and x_offset shutter centering

    Parameters
    ----------
    wave : array-like, float
        Spectrum wavelengths, microns

    sigma : float
        Derived source width (pixels) in quadtrature with the tabulated
        intrinsic PSF width from `~msaexp.utils.get_nirspec_psf_fwhm`

    x_pos : float
        Shutter-normalized source center in range (-0.5, 0.5)
        (``source_xpos`` in slit metadata)

    slit_width : float
        Slit/shutter width, arcsec

    pixel_scale : float
        NIRSpec pixel scale, arcsec/pix

    Returns
    -------
    prf_frac : array-like
        Wavelength-dependent flux fraction within the shutter

    """
    from msaexp.resample_numba import pixel_integrated_gaussian_numba as PRF

    global SCALE_FWHM
    global LOOKUP_PRF

    # Tabulated PSF FWHM, pix
    psf_fw = msautils.get_nirspec_psf_fwhm(wave) * SCALE_FWHM

    pix_center = np.zeros_like(wave)
    pix_mu = x_pos * slit_width / pixel_scale
    pix_sigma = np.sqrt((psf_fw / 2.35) ** 2 + sigma**2)

    msg = f"slit_prf_fraction: mu = {pix_mu:.2f}, sigma = {sigma:.1f} pix"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    dxpix = slit_width / pixel_scale
    prf_frac = (
        PRF(pix_center, pix_mu, pix_sigma, dx=dxpix, normalization=1) * dxpix
    )

    return prf_frac


# def set_lookup_prf(**kwargs):
#     """
#     Initialize global LOOKUP_PRF
#     """
#     global LOOKUP_PRF
#
#     msg = f"Initialize LookupTablePSF({kwargs})"
#     utils.log_comment(utils.LOGFILE, msg, verbose=True)
#
#     prf = msautils.LookupTablePSF(**kwargs)
#
#     LOOKUP_PRF = prf
#
#     return prf


def objfun_prof_trace(
    theta,
    base_coeffs,
    wave,
    xpix,
    ypix,
    bar,
    yslit0,
    diff,
    vdiff,
    wdiff,
    mask,
    ipos,
    ineg,
    sh,
    fix_sigma,
    force_positive,
    verbose,
    ret,
):
    """
    Objective function for fitting the profile along the trace

    Parameters
    ----------
    theta : array-like
        Array of parameters for the objective function.

    base_coeffs : array-like
        Coefficients for the base trace polynomial.

    wave : array-like
        Array of wavelengths.

    xpix : array-like
        Array of x-pixel positions.

    ypix : array-like
        Array of y-pixel positions.

    yslit0 : array-like
        Array of initial y-slit positions.

    diff : array-like
        A-B nod difference image

    vdiff : array-like
        Propagated variance of the difference image

    wdiff : array-like
        Propagated weight image

    mask : array-like
        Valid pixel mask

    ipos : array-like
        Array of exposure indices corresponding to the positive
        "A" component of the difference

    ineg : array-like
        Array of exposure indices corresponding to the negative
        "B" component of the difference

    sh : tuple
        Shape of the data.

    fix_sigma : float
        Optional value to fix the profile width

    force_positive : bool
        Only consider positive parts of the profile difference image

    verbose : bool
        Flag to enable verbose output.

    ret : int
        Return flag (see returns).

    Returns
    -------
    If ret == 1:
        snum : array-like
            Numerator of the objective function.  The estimated 1D
            spectrum is ``snum/sden``.

        svnum : array-like
            The estimated 1D variance is ``svnum/sden**2``.

        sden : array-like
            Denominator of the objective function.

        smod : array-like
            Profile model of the objective function

        sigma : float
            Value of sigma.

        trace_coeffs : array-like
            Coefficients of the trace polynomial.

        chi2 : float
            Chi-squared value.
    else:
        chi2 : float
            Chi-squared value.

    """

    from msaexp.resample_numba import pixel_integrated_gaussian_numba as PRF

    global EVAL_COUNT
    global CHI2_MASK
    global SKIP_COUNT
    global CENTER_WIDTH
    global CENTER_PRIOR
    global SIGMA_PRIOR
    global SCALE_FWHM
    global WING_SIGMA
    global DEFAULT_WINGS
    global WINGS_XOFF
    global LOOKUP_PRF
    global LOOKUP_PRF_ORDER

    EVAL_COUNT += 1

    wings = DEFAULT_WINGS

    if fix_sigma > 0:
        sigma = fix_sigma / 10.0
        i0 = 0
    elif WINGS_XOFF is not None:
        sigma = theta[0] / 10.0
        wings = theta[1 : len(WINGS_XOFF) + 1]
        i0 = len(WINGS_XOFF) + 1

    elif len(theta) == 4:
        sigma = WING_SIGMA / 10
        wings = np.append(theta[:2], 0)  # [theta[1], 3.]
        i0 = 2
        if DEFAULT_WINGS is not None:
            sigma = theta[0] / 10.0
            wings = DEFAULT_WINGS
            i0 = 1

    elif len(theta) >= 5:
        sigma = WING_SIGMA / 10
        wings = theta[:3]
        i0 = 3

        if DEFAULT_WINGS is not None:
            sigma = theta[0] / 10.0
            wings = DEFAULT_WINGS
            i0 = 1

    else:
        if DEFAULT_WINGS is not None:
            sigma = theta[0] / 10.0
            wings = DEFAULT_WINGS

        sigma = theta[0] / 10.0
        i0 = 1

    # print('xxx', sigma, wings, theta[i0:])
    yslit = yslit0 * 1.0
    for j in np.where(ipos)[0]:
        xj = (xpix[j, :] - sh[1] / 2) / sh[1]
        _ytr = np.polyval(theta[i0:], xj)
        _ytr += np.polyval(base_coeffs, xj)
        yslit[j, :] = ypix[j, :] - _ytr

    psf_fw = msautils.get_nirspec_psf_fwhm(wave) * SCALE_FWHM

    # sig2 = np.sqrt((psf_fw / 2.35)**2 + sigma**2)
    sig2 = np.sqrt((psf_fw / 2.35) ** 2 + sigma**2)

    # wings = (0.05, 2)

    if LOOKUP_PRF is not None:
        ppos = LOOKUP_PRF.evaluate(
            sigma=sigma,
            slit_coords=(wave[ipos, :].flatten(), yslit[ipos, :].flatten()),
            order=LOOKUP_PRF_ORDER,
        )
    else:
        ppos = PRF(
            yslit[ipos, :].flatten(), 0.0, sig2[ipos, :].flatten(), dx=1
        )

    if WINGS_XOFF is not None:
        for wx, wn in zip(WINGS_XOFF, wings):
            ppos += wn * PRF(
                yslit[ipos, :].flatten() + wx,
                0.0,
                sig2[ipos, :].flatten(),
                dx=1,
            )

    elif wings is not None:
        ppos += wings[0] * PRF(
            yslit[ipos, :].flatten() + wings[2],
            0.0,
            sig2[ipos, :].flatten() * wings[1],
            dx=1,
        )

    ppos = ppos.reshape(yslit[ipos, :].shape)
    #####
    #  Don't scale profile by bar because *data* are corrected
    # ppos *= bar[ipos,:]

    if ineg.sum() > 0:
        if LOOKUP_PRF is not None:
            pneg = LOOKUP_PRF.evaluate(
                sigma=sigma,
                slit_coords=(
                    wave[ineg, :].flatten(),
                    yslit[ineg, :].flatten(),
                ),
                order=LOOKUP_PRF_ORDER,
            )
        else:
            pneg = PRF(
                yslit[ineg, :].flatten(), 0.0, sig2[ineg, :].flatten(), dx=1
            )
        if WINGS_XOFF is not None:
            for wx, wn in zip(WINGS_XOFF, wings):
                pneg += wn * PRF(
                    yslit[ineg, :].flatten() + wx,
                    0.0,
                    sig2[ineg, :].flatten(),
                    dx=1,
                )
        elif wings is not None:
            pneg += wings[0] * PRF(
                yslit[ineg, :].flatten() + wings[2],
                0.0,
                sig2[ineg, :].flatten() * wings[1],
                dx=1,
            )

        pneg = pneg.reshape(yslit[ineg, :].shape)
        # pneg *= bar[ineg,:]
    else:
        pneg = np.zeros_like(ppos)

    if 0:
        ppos = np.nansum(ppos, axis=0) / ipos.sum()
        if ineg.sum() > 0:
            pneg = np.nansum(pneg, axis=0) / ineg.sum()
        else:
            pneg = np.zeros_like(ppos)
    else:
        ppos = np.nansum(ppos * mask[ipos, :], axis=0) / np.nansum(
            mask[ipos, :], axis=0
        )
        if ineg.sum() > 0:
            pneg = np.nansum(pneg * mask[ineg, :], axis=0) / np.nansum(
                mask[ineg, :], axis=0
            )
        else:
            pneg = np.zeros_like(ppos)

    pdiff = ppos - pneg

    if (pneg.sum() == 0) & (len(theta) == 1000):
        bkg = theta[2] / 10.0
    else:
        bkg = 0.0

    if force_positive:
        pdiff *= pdiff > 0

    # Remove any masked pixels
    pmask = mask.sum(axis=0) > 0

    snum = np.nansum(
        ((diff - bkg) * pdiff / wdiff * pmask).reshape(sh), axis=0
    )
    svnum = np.nansum(
        (vdiff * pdiff**2 / wdiff**2 * pmask).reshape(sh), axis=0
    )
    sden = np.nansum((pdiff**2 / wdiff * pmask).reshape(sh), axis=0)
    smod = snum / sden * pdiff.reshape(sh)

    chi = (diff - (smod + bkg).flatten()) / np.sqrt(vdiff)

    if 0:
        # two-sided
        # CHI2_MASK = (chi < 40) & (chi > -10)
        # CHI2_MASK &= ((smod+bkg).flatten()/np.sqrt(vdiff) > -10)
        CHI2_MASK = diff / np.sqrt(vdiff) > -10

    elif 0:
        # absolute value
        CHI2_MASK = chi**2 < 40**2
    else:
        # no mask
        CHI2_MASK = np.isfinite(diff)

    ok = np.isfinite(chi)
    chi2 = np.nansum(huber(HUBER_ALPHA, chi[CHI2_MASK & ok]))

    # "prior" on sigma with logistic bounds
    peak = 10000
    chi2 += peak / (1 + np.exp(-10 * (sigma - 1.8)))  # right
    chi2 += peak - peak / (1 + np.exp(-30 * (sigma - 0)))  # left
    chi2 += (sigma - SIGMA_PRIOR) ** 2 / 2 / PIX_SCALE**2
    chi2 += (
        (np.array(theta[i0:]) - CENTER_PRIOR) ** 2 / 2 / CENTER_WIDTH**2
    ).sum()

    if (EVAL_COUNT % SKIP_COUNT == 0) | (ret == 1):
        tval = " ".join([f"{t:6.3f}" for t in theta[i0:]])
        tfix = "*" if i0 == 0 else " "
        msg = f"{EVAL_COUNT:>8} {tfix}sigma={sigma*10:.2f}{tfix}"
        msg += f" [{tval}]  {chi2:.1f}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    if ret == 1:
        trace_coeffs = theta[i0:]
        return snum, svnum, sden, smod, sigma, trace_coeffs, chi2
    else:
        return chi2


class SlitGroup:
    def __init__(
        self,
        files,
        name,
        position_key="y_index",
        diffs=True,
        grating_diffs=True,
        stuck_threshold=0.3,
        hot_cold_kwargs=None,
        bad_shutter_names=None,
        dilate_failed_open=True,
        num_shutters=-1,
        undo_barshadow=2,
        min_bar=0.4,
        bar_corr_mode="wave",
        fix_prism_norm=True,
        extended_calibration_kwargs={"threshold": 0.01},
        with_sflat_correction=True,
        with_extra_dq=True,
        slit_hotpix_kwargs={},
        sky_arrays=None,
        sky_file="read",
        set_background_spectra_kwargs={},
        global_sky_df=7,
        estimate_sky_kwargs=None,
        flag_profile_kwargs=None,
        do_multiple_mask=True,
        flag_trace_kwargs={},
        flag_percentile_kwargs={},
        undo_pathloss=True,
        trace_with_xpos=False,
        trace_with_ypos=False,
        trace_from_yoffset=True,
        with_fs_offset=False,
        fit_shutter_offset_kwargs=None,
        shutter_offset=0.0,
        nod_offset=None,
        pad_border=2,
        weight_type="ivm",
        reference_exposure="auto",
        lookup_prf=None,
        **kwargs,
    ):
        """
        Container for a list of 2D extracted ``SlitModel`` files

        Parameters
        ----------
        files : list
            List of `SlitModel` files

        name : str
            Label for the group

        position_key : str
            Column in the ``info`` table to define the nod positions
                - "y_index" = Rounded y offset
                - "position_number" = dither number
                - "shutter_state" = Shutter state from MPT.  Usually robust, but can
                   get confused when multiple catalog sources fall within the slitlets

        diffs : bool
            Compute nod differences

        grating_diffs : bool
            Force diffs for grating spectra

        stuck_threshold : float
            Parameter for identifying stuck-closed shutters in prism spectra in
            `~msaexp.slit_combine.SlitGroup.mask_stuck_closed_shutters`

        bad_shutter_names : list, None
            List of integer shutter indices (e.g., among ``[-1, 0, 1]`` for a
            3-shutter slitlet) to mask as bad, e.g., from stuck shutters

        dilate_failed_open : bool, int
            Dilate the mask of pixels flagged with ``MSA_FAILED_OPEN``.  If an integer,
            do ``dilate_failed_open`` dilation iterations.

        num_shutters : int
            Manually specify the number of shutters in the slitlet for the bar
            shadow correction.
                - If num_shutters < 0, then compute from
                  ``len(self.info["shutter_state"][0])``
                - If num_shutters = 0, then take the number of offset positions
                  ``self.unp.N``
                - If num_shutters > 0, then use that value

        undo_barshadow : bool, 2
            Undo the ``BarShadow`` correction if an extension found in the
            slit model files.  If ``2``, then apply internal barshadow correction
            with ``bar_corr_mode``.

        min_bar : float
            Minimum acceptable value of the BarShadow reference

        bar_corr_mode : str
            Internal barshadow correction type
                - ``flat``: monochromatic `~msaexp.utils.get_prism_bar_correction`
                - ``wave``: wave-dependent `~msaexp.utils.get_prism_wave_bar_correction`

        fix_prism_norm : bool
            Apply prism normalization correction with
            `~msaexp.utils.get_normalization_correction`.

        extended_calibration_kwargs : dict
            Keyword arguments to `~msaexp.utils.slit_extended_flux_calibration`

        with_sflat_correction : bool
            Additional field-dependent s-flat correction

        with_extra_dq : bool
            With extra DQ mask from `~msaexp.utils.extra_slit_dq_flags`

        sky_arrays : array-like
            Optional sky data (in progress)

        sky_file : str, None
            - Filename of a tabulated global sky
            - `"read"`: try to find a file provided with `msaexp` in
              ``msaexp/data/sky_data``
            - None: ignore

        global_sky_df : int
            If a ``sky_file`` is available and ``estimate_sky_kwargs`` are specified,
            use this as the degrees of freedom in ``estimate_sky_kwargs``

        estimate_sky_kwargs : None, dict
            Arguments to pass to `~msaexp.slit_combine.SlitGroup.estimate_sky` to
            estimate sky directly from the slit data

        flag_percentile_kwargs : None, dict
            Arguments to pass to
            `~msaexp.slit_combine.SlitGroup.flag_percentile_outliers` to
            flag extreme values

        undo_pathloss : bool
            Undo pipeline pathloss correction (should usually be the
            PATHLOSS_UNIFORM correction) if the extensions are found in the slit
            model files

        trace_with_xpos : bool
            Compute traces including the predicted source center x position

        trace_with_ypos : bool
            Compute traces including the predicted source center y position

        trace_from_yoffset : bool
            Compute traces derived from y offsets

        with_fs_offset: bool
            Try to calculate extra fixed slit trace offset.  Seems to be not be needed
            in ``jwst>=1.15``.

        nod_offset : float, None
            Nod offset size (pixels) to use if the slit model traces don't
            already account for it, e.g., in background-indicated slits
            without explicit catalog sources.  If not provided (None), then set
            to `MSA_NOD_ARCSEC / slit_pixel_scale`.

        pad_border : int
            Grow mask around edges of 2D cutouts

        reference_exposure : int, 'auto'
            Define a reference nod position. If ``'auto'``, then will use the
            exposure in the middle of the nod offset distribution

        weight_type : str
            Weighting scheme for 2D resampling
                - ``ivm`` : Use weights from ``var_rnoise``, like `jwst.resample <https://github.com/spacetelescope/jwst/blob/4342988027ee0811b57d3641bda4c8486d7da1f5/jwst/resample/resample_utils.py#L168>`_
                - ``poisson`` : Weight with ``var_poisson``, msaexp extractions v1, v2
                - ``exptime`` : Use ``slit.meta.exposure.exposure_time * mask``
                - ``exptime_bar`` : ``wht = exposure_time * mask / bar``
                - ``mask`` : Just use the bad pixel mask

        pad_border : int
            Grow mask around edges of 2D cutouts

        Attributes
        ----------
        meta : dict
            Metadata about the processing status

        sh : (int, int)
            Dimensions of the 2D slit data

        sci : array-like (float)
            Science data with dimensions ``(N, sh[0]*sh[1])``

        dq : array-like (int)
            DQ bit flags

        mask : array-like (bool)
            Valid data

        var : array-like (float)
            Variance data

        var_rnoise: array-like (float)
            RNOISE variance data

        var_poisson: array-like (float)
            POISSON variance data

        xslit : array-like (float)
            Array of x slit coordinates

        yslit: array-like (float)
            Array of cross-dispersion coordinates. Should be zero along the
            expected center of the (curved) trace

        yslit_orig : array-like (float)
            Copy of ``yslit``, which may be updated with new trace coefficients

        ypix : array-like (float)
            y pixel coordinates

        wave : array-like (float)
            2D wavelengths

        bar : array-like (float)
            The BarShadow correction if found in the `SlitModel` files

        xtr : array-like (float)
            1D x pixel along trace

        ytr : array-like (float)
            1D y trace position

        wtr : array-like (float)
            Wavelength along the trace

        """
        self.name = name

        self.slits = []
        keep_files = []

        if sky_arrays is None:
            self.sky_arrays = None
        else:
            self.sky_arrays = [sky_arrays[0] * 1, sky_arrays[1] * 1]

        if weight_type not in [
            "poisson",
            "mask",
            "ivm",
            "exptime",
            "exptime_bar",
        ]:
            msg = "weight_type {0} not recognized".format(weight_type)
            raise ValueError(msg)

        self.lookup_prf = lookup_prf

        # kwargs to meta dictionary
        self.meta = {
            "diffs": diffs,
            "grating_diffs": grating_diffs,
            "trace_with_xpos": trace_with_xpos,
            "trace_with_ypos": trace_with_ypos,
            "trace_from_yoffset": trace_from_yoffset,
            "shutter_offset": shutter_offset,
            "stuck_threshold": stuck_threshold,
            "bad_shutter_names": bad_shutter_names,
            "dilate_failed_open": dilate_failed_open,
            "num_shutters": num_shutters,
            "undo_barshadow": undo_barshadow,
            "min_bar": min_bar,
            "bar_corr_mode": bar_corr_mode,
            "fix_prism_norm": fix_prism_norm,
            "wrapped_barshadow": False,
            "own_barshadow": False,
            "nod_offset": nod_offset,
            "undo_pathloss": undo_pathloss,
            "reference_exposure": reference_exposure,
            "pad_border": pad_border,
            "position_key": position_key,
            "nhot": 0,
            "ncold": 0,
            "has_sky_arrays": (sky_arrays is not None),
            "weight_type": weight_type,
            "percentile_outliers": 0,
            "sky_file": "N/A",
            "global_sky_df": global_sky_df,
            "with_fs_offset": with_fs_offset,
            "with_sflat_correction": with_sflat_correction,
            "with_extra_dq": with_extra_dq,
        }

        # Comments on meta for header keywords
        self.meta_comment = {
            "diffs": "Calculated with exposure differences",
            "grating_diffs": "Diffs forced for grating",
            "trace_with_xpos": "Trace includes x offset in shutter",
            "trace_with_ypos": "Trace includes y offset in shutter",
            "trace_from_yoffset": "Trace derived from yoffsets",
            "shutter_offset": "Global offset to fixed shutter coordinate",
            "stuck_threshold": "Stuck shutter threshold",
            "dilate_failed_open": "Dilate failed open mask",
            "num_shutters": "Number of shutters used for bar shadow model",
            "undo_barshadow": "Bar shadow update behavior",
            "min_bar": "Minimum allowed bar value",
            "bar_corr_mode": "Bar shadow correction type",
            "fix_prism_norm": "Apply prism scale correction",
            "wrapped_barshadow": "Bar shadow was wrapped for central shutter",
            "own_barshadow": "Internal bar shadow correction applied",
            "nod_offset": "Nod offset size, pixels",
            "undo_pathloss": "Remove pipeline pathloss correction",
            "reference_exposure": "Reference exposure argument",
            "pad_border": "Border padding",
            "position_key": "Method for determining offset groups",
            "nhot": "Number of flagged hot pixels",
            "ncold": "Number of flagged cold pixels",
            "has_sky_arrays": "sky arrays specified",
            "weight_type": "Weighting scheme for 2D combination",
            "percentile_outliers": "Masked pixels from flag_percentile_outliers",
            "sky_file": "Filename of a global sky background table",
            "global_sky_df": "Degrees of freedom of fit with global sky",
            "with_fs_offset": "Extra fixed slit offset",
            "with_sflat_correction": "Field-dependent s-flat correction",
            "with_extra_dq": "Extra DQ pixel mask",
        }

        self.shapes = []
        self.flagged_hot_pixels = []
        for i, file in enumerate(files):
            slit = jwst.datamodels.open(file)

            slit.phot_corr = np.ones_like(slit.data)

            if extended_calibration_kwargs is not None:
                status = msautils.slit_extended_flux_calibration(
                    slit, **extended_calibration_kwargs
                )

            if slit_hotpix_kwargs is not None:
                _ = msautils.slit_hot_pixels(slit, **slit_hotpix_kwargs)
                self.flagged_hot_pixels.append(_)  # dq, count, status
            else:
                self.flagged_hot_pixels.append((None, 0, 0))

            self.slits.append(slit)
            keep_files.append(file)
            self.shapes.append(slit.data.shape)

        self.files = keep_files
        self.info = self.parse_metadata()

        self.calculate_slices()

        self.parse_data(
            with_sflat_correction=with_sflat_correction, with_extra_dq=with_extra_dq
        )

        if sky_file is not None:
            self.get_global_sky(sky_file=sky_file)

        if self.grating.startswith("G"):
            self.meta["diffs"] |= self.meta["grating_diffs"]
            msg = " Disperser {grating} is a grating.  diffs={diffs}"
            msg += " (grating_diffs={grating_diffs})"
            utils.log_comment(
                utils.LOGFILE,
                msg.format(grating=self.grating, **self.meta),
                verbose=VERBOSE_LOG,
            )

        if fix_prism_norm:
            self.apply_normalization_correction()

        if (hot_cold_kwargs is not None) & (self.N > 2):
            nhot, ncold, flag = self.flag_hot_cold_pixels(**hot_cold_kwargs)
            for i in range(self.N):
                self.mask[i, :] &= flag == 0

            self.meta["nhot"] = nhot
            self.meta["ncold"] = ncold

        if (flag_trace_kwargs is not None) & (self.mask.sum() > 100):
            if self.meta["position_key"] != "manual_position":
                self.flag_trace_outliers(**flag_trace_kwargs)

        if (flag_percentile_kwargs is not None) & (self.mask.sum() > 100):
            self.flag_percentile_outliers(**flag_percentile_kwargs)

        if estimate_sky_kwargs is not None:
            try:
                self.estimate_sky(**estimate_sky_kwargs)

                if (flag_trace_kwargs is not None) & (self.mask.sum() > 100):
                    if self.meta["position_key"] != "manual_position":
                        self.flag_trace_outliers(**flag_trace_kwargs)

                if (flag_percentile_kwargs is not None) & (
                    self.mask.sum() > 100
                ):
                    self.flag_percentile_outliers(**flag_percentile_kwargs)

            except ValueError:
                pass

        if flag_profile_kwargs is not None:
            try:
                self.flag_from_profile(**flag_profile_kwargs)
            except ValueError:
                pass

        if (self.N > 2) & (do_multiple_mask):
            bad = ~self.mask
            nbad = bad.sum(axis=0)
            nexp_thresh = 2 * (self.N // 3)
            all_bad = nbad >= nexp_thresh
            all_bad &= nbad < self.N
            msg = f" {'multiple mask':<28}: {all_bad.sum()} pixels in"
            msg += f" >= {nexp_thresh} exposures"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

            for i in range(self.N):
                self.mask[i, :] &= ~all_bad

        if set_background_spectra_kwargs is not None:
            self.set_background_spectra(**set_background_spectra_kwargs)

        if fit_shutter_offset_kwargs is not None:
            self.fit_shutter_offset(**fit_shutter_offset_kwargs)

        self.fit = None

    @property
    def N(self):
        """
        Number of individual SlitModel components
        """
        return len(self.slits)

    @property
    def grating(self):
        """
        Grating used
        """
        return self.info["grating"][0]

    @property
    def filter(self):
        """
        Grating used
        """
        return self.info["filter"][0]

    @property
    def unp(self):
        """
        `grizli.utils.Unique` object for the different nod positions
        """
        return utils.Unique(
            self.info[self.meta["position_key"]], verbose=False
        )

    @property
    def calc_reference_exposure(self):
        """
        Define a reference exposure, usually middle of three nods
        """
        # if reference_exposure in ['auto']:
        #     reference_exposure = 1 if obj.N == 1 else 2 - ('bluejay' in root)
        if self.meta["reference_exposure"] in ["auto"]:
            if self.N < 3:
                ix = 0
            else:
                ix = np.nanargmin(np.abs(self.relative_nod_offset))

            ref_exp = self.info[self.meta["position_key"]][ix]
        else:
            ref_exp = self.meta["reference_exposure"]

        return ref_exp

    @property
    def source_ypixel_position(self):
        """
        Expected relative y pixel location of the source
        """
        for j, slit in enumerate(self.slits[:1]):

            _res = msautils.slit_trace_center(
                slit, with_source_ypos=False, index_offset=0.0
            )

            _xtr, _ytr0, _wtr0, slit_ra, slit_dec = _res

            _res = msautils.slit_trace_center(
                slit, with_source_ypos=True, index_offset=0.0
            )

            _xtr, _ytr1, _wtr1, slit_ra, slit_dec = _res

            # plt.plot(_ytr1 - _ytr0)
            trace_yoffset = np.nanmedian(_ytr1 - _ytr0)
            break

        return trace_yoffset

    @property
    def slit_pixel_scale(self):
        """
        Compute cross dispersion pixel scale from slit WCS

        Returns
        -------
        pix_scale : float
            Cross-dispersion pixel scale ``arcsec / pixel``

        """

        sl = self.slits[0]
        wcs = sl.meta.wcs
        d2s = wcs.get_transform("detector", "world")

        x0 = d2s(self.sh[1] // 2, self.sh[0] // 2)
        x1 = d2s(self.sh[1] // 2, self.sh[0] // 2 + 1)

        dx = np.array(x1) - np.array(x0)
        cosd = np.cos(x0[1] / 180 * np.pi)
        pix_scale = np.sqrt((dx[0] * cosd) ** 2 + dx[1] ** 2) * 3600.0

        return pix_scale

    @property
    def slit_shutter_scale(self):
        """
        Compute cross dispersion pixel scale in shutter coordinates from slit WCS

        Returns
        -------
        pix_scale : float
            Cross-dispersion pixel scale ``shutters / pixel``

        """

        sl = self.slits[0]
        wcs = sl.meta.wcs
        d2s = wcs.get_transform("detector", "slit_frame")

        x0 = d2s(self.sh[1] // 2, self.sh[0] // 2)
        x1 = d2s(self.sh[1] // 2, self.sh[0] // 2 + 1)

        dx = np.array(x1) - np.array(x0)
        pix_scale = np.sqrt(dx[0] ** 2 + dx[1] ** 2)

        return pix_scale

    @property
    def relative_nod_offset(self):
        """
        Compute relative nod offsets from the trace polynomial
        """
        if self.N == 1:
            return np.array([0])

        y0 = np.array([c[-1] for c in self.base_coeffs])
        return y0 - np.median(y0)

    @property
    def fixed_yshutter(self):
        """
        Fixed cross-dispersion shutter coordinate

        Returns
        -------
        shutter_y : array-like
            Cross-dispersion pixel in normalized shutter coordinates:
            ``shutter_y = (slit_frame_y / slit_shutter_scale + shutter_offset ) / 5``

        """
        shutter_y = (
            self.slit_frame_y / self.slit_shutter_scale
            + self.meta["shutter_offset"]
        ) / 5.0

        shutter_y[~self.mask] = np.nan

        return shutter_y

    @property
    def exptime(self):
        """
        Get exposure time of individual slits
        """
        return np.array(
            [slit.meta.exposure.exposure_time for slit in self.slits]
        )

    @property
    def IS_FIXED_SLIT(self):
        """
        Is this a Fixed Slit spectrum?
        """
        return self.info["lamp_mode"][0] == "FIXEDSLIT"

    def slit_metadata(self):
        """
        Make a table of the slit metadata
        """
        rows = []
        pscale = self.slit_pixel_scale * 1
        source_ypixel_position = self.source_ypixel_position * 1

        for i, sl in enumerate(self.slits):
            row = {
                "filename": sl.meta.filename,
                "nx": self.sh[1],
                "ny": self.sh[0],
            }

            for j in range(3):
                row[f"trace_c{j}"] = self.base_coeffs[i][j]

            row["slit_pixel_scale"] = pscale
            row["source_ypixel_position"] = source_ypixel_position

            for att in [
                "is_extended",
                "source_id",
                "source_name",
                "source_ra",
                "source_dec",
                "source_type",
                "source_xpos",
                "source_ypos",
                "shutter_state",
                "shutter_id",
                "slitlet_id",
                "slit_ymin",
                "slit_ymax",
                "quadrant",
                "xcen",
                "ycen",
                "xstart",
                "xsize",
                "ystart",
                "ysize",
            ]:
                row[att] = sl.__getattr__(att)

            inst = sl.meta.instrument.instance
            for k in [
                "detector",
                "grating",
                "filter",
                "msa_metadata_file",
                "msa_configuration_id",
                "msa_metadata_id",
            ]:
                if k in inst:
                    row[k] = inst[k]

            _exp = sl.meta.exposure.instance
            for k in [
                "exposure_time",
                "nframes",
                "ngroups",
                "nints",
                "readpatt",
                "start_time",
            ]:
                if k in _exp:
                    row[k] = _exp[k]

            _point = sl.meta.pointing.instance
            for k in _point:
                row[k] = _point[k]

            _dith = sl.meta.dither.instance
            for k in _dith:
                row[k] = _dith[k]

            rows.append(row)

        tab = utils.GTable(rows)
        return tab

    def parse_metadata(self):
        """
        Generate the `info` metadata attribute from the `slits` data

        Returns
        -------
        info : `~astropy.table.Table`
            Metadata table

        """
        rows = []
        for i, slit in enumerate(self.slits):
            _nbp = self.flagged_hot_pixels[i][1]

            msg = f"{i:>2} {slit.meta.filename} {slit.data.shape}"
            msg += f" {_nbp:>2} flagged hot pixels"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

            row = {
                "filename": slit.meta.filename,
                "visit": slit.meta.filename.split("_")[0],
                "xstart": slit.xstart,
                "ystart": slit.ystart,
                "shape": slit.data.shape,
            }

            md = slit.meta.dither.instance
            for k in md:
                row[k] = md[k]

            mi = slit.meta.instrument.instance
            for k in mi:
                row[k] = mi[k]

            rows.append(row)

        info = utils.GTable(rows=rows)

        # Print message if columns have mask from occasional keywords missing
        # in slit metadata?
        for k in info.colnames:
            if hasattr(info[k], "mask"):
                msg = f"info table: column '{k}' has {info[k].mask.sum()} masked rows"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        info["x_position"] = np.round(info["x_offset"] * 10) / 10.0
        info["y_position"] = np.round(info["y_offset"] * 10) / 10.0
        info["y_index"] = (
            utils.Unique(info["y_position"], verbose=False).indices + 1
        )

        return info

    def get_global_sky(self, sky_file=None):
        """
        Try to read a global sky file from ``msaexp/data/msa_sky``
        """
        if sky_file in [None, "read"]:
            visit = os.path.basename(self.files[0]).split("_")[0]
            sky_file = os.path.join(
                os.path.dirname(__file__),
                "data",
                "msa_sky",
                f"{visit}_sky.csv",
            )

        if os.path.exists(sky_file):
            sky_data = utils.read_catalog(sky_file)
            msg = f" {'get_global_sky':<28}: {os.path.basename(sky_file)} to sky_arrays"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
            self.sky_arrays = (sky_data["wave"], sky_data["flux"])
            self.meta["sky_file"] = os.path.basename(sky_file)
            self.meta["has_sky_arrays"] = True

    def calculate_slices(self):
        """
        Calculate slices to handle unequal cutout sizes

        Returns
        -------
        Adds ``slice`` attribute to each item in the ``slits`` list, where
        ``slice = (slice(ystart, ystart + shape[0]), slice(xstart, xstart + shape[1]))``
        """
        ashapes = np.array(self.shapes)
        ystart = np.max(self.info["ystart"])
        ystop = np.min(self.info["ystart"] + self.info["shape"][:, 0])

        xstart = np.max(self.info["xstart"])
        xstop = np.min(self.info["xstart"] + self.info["shape"][:, 1])

        # Global shape
        self.sh = (ystop - ystart, xstop - xstart)

        y0 = ystart - self.info["ystart"]
        x0 = xstart - self.info["xstart"]
        slices = [
            (slice(yi, yi + self.sh[0]), slice(xi, xi + self.sh[1]))
            for yi, xi in zip(y0, x0)
        ]
        for s, slit in zip(slices, self.slits):
            slit.slice = s

    def parse_data(self, debug=False, with_sflat_correction=True, with_extra_dq=True, **kwargs):
        """
        Read science, variance and trace data from the ``slits`` SlitModel
        files

        """
        import scipy.ndimage as nd

        global PRISM_MAX_VALID, PRISM_MIN_VALID_SN, RNOISE_THRESHOLD

        slits = self.slits

        if self.meta["nod_offset"] is None:
            self.meta["nod_offset"] = MSA_NOD_ARCSEC / self.slit_pixel_scale

        sl = (slice(0, self.sh[0]), slice(0, self.sh[1]))

        # Local sky overlaps
        for j, slit in enumerate(slits):
            if "_raw" in self.files[j]:
                slit_sky_file = os.path.basename(self.files[j]).replace("_raw", "_sky")
                if os.path.exists(slit_sky_file):
                    with pyfits.open(slit_sky_file) as _sky_im:
                        msg = f'parse_data: sky overlap file {slit_sky_file}'
                        utils.log_comment(
                            utils.LOGFILE, msg, verbose=True
                        )

                        slit.sky_overlap = _sky_im[0].data * 1
                        slit.data -= slit.sky_overlap / slit.barshadow
                        # slit.data = (slit.data * slit.barshadow - _sky_im[0].data)

        ##################
        # Extra flat field
        for slit in slits:
            is_ext = msautils.slit_data_from_extended_reference(slit)
            if not is_ext:
                continue

            if slit.meta.exposure.type == "NRS_FIXEDSLIT":
                flat_profile = msautils.fixed_slit_flat_field(
                    slit, apply=True, verbose=VERBOSE_LOG
                )

        sci = np.array([slit.data[slit.slice].flatten() * 1 for slit in slits])
        if self.IS_FIXED_SLIT:
            bar = np.ones_like(sci)
        else:
            try:
                bar = np.array(
                    [
                        slit.barshadow[slit.slice].flatten() * 1
                        for slit in slits
                    ]
                )
            except:
                bar = np.ones_like(sci)

        dq = np.array([slit.dq[slit.slice].flatten() * 1 for slit in slits])

        var_rnoise = np.array(
            [slit.var_rnoise[slit.slice].flatten() * 1 for slit in slits]
        )

        var_poisson = np.array(
            [slit.var_poisson[slit.slice].flatten() * 1 for slit in slits]
        )

        phot_corr = np.array(
            [slit.phot_corr[slit.slice].flatten() * 1 for slit in slits]
        )

        bad = sci == 0
        if debug:
            print("sci == 0 ; bad = ", bad.sum())

        sci[bad] = np.nan
        var_rnoise[bad] = np.nan
        var_poisson[bad] = np.nan
        dq[bad] = 1
        if bar is not None:
            bar[bad] = np.nan

        # 2D
        xslit = []
        ypix = []
        yslit = []
        wave = []
        dwave_dx = []

        # 1D
        xtr = []
        ytr = []
        wtr = []
        slit_frame_y = []

        attr_keys = [
            "source_ra",
            "source_dec",
            "source_xpos",
            "source_ypos",
            "shutter_state",
            "slitlet_id",
        ]

        self.info["shutter_state"] = "xxxxxxxx"

        for k in attr_keys:
            if k not in self.info.colnames:
                self.info[k] = 0.0

        # Will be populated if needed
        sflat_data = None

        for j, slit in enumerate(slits):

            sh = slit.data.shape
            yp, xp = np.indices(sh)

            sl = slit.slice
            _res = msautils.slit_trace_center(
                slit,
                with_source_xpos=False,
                with_source_ypos=self.meta["trace_with_ypos"],
                index_offset=0.0,
            )

            _xtr, _ytr, _wtr, slit_ra, slit_dec = _res

            xslit.append(xp[sl].flatten() - sl[1].start)
            yslit.append((yp[sl] - (_ytr[sl[1]])).flatten() - sl[0].start)
            ypix.append(yp[sl].flatten() - sl[0].start)

            wcs = slit.meta.wcs
            d2w = wcs.get_transform("detector", "world")

            _ypi, _xpi = np.indices(slit.data.shape)
            _ras, _des, _wave = d2w(_xpi, _ypi)

            # Slit frame coordinate defined from first exposure
            if j == 0:
                d2s = wcs.get_transform("detector", "slit_frame")
                _sx, _sy, _slam = np.array(d2s(_xpi[sl], _ypi[sl]))

            if self.meta["trace_with_xpos"] & (slit.source_xpos is not None):
                _xres = msautils.slit_trace_center(
                    slit,
                    with_source_xpos=True,
                    with_source_ypos=self.meta["trace_with_ypos"],
                    index_offset=0.0,
                )
                _xwtr = _xres[2]
                dwave = _xwtr - _wtr
                dwave_step = np.nanpercentile(
                    dwave / np.gradient(_wtr), [5, 50, 95]
                )

                # jwst < 1.15 correction was wrong
                if (jwst.__version__ < "1.15") & (jwst.__version__ > "1.12"):
                    msg = (
                        f"  Flip sign of wavelength correction for"
                        + f" jwst=={jwst.__version__}"
                    )

                    utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
                    dwave *= -1
                    dwave_step *= -1

                # Signs of source_xpos and dwave_step should be the same
                sign = slit.source_xpos * dwave_step[1]
                if sign < 0:
                    dwave *= -1
                    dwave_step *= -1
                    _note = "(flipped) "
                else:
                    _note = ""

                msg = (
                    "   Apply wavelength correction for "
                    f"source_xpos = {slit.source_xpos:.2f}: {_note}"
                    f"dx = {dwave_step[0]:.2f} to {dwave_step[2]:.2f} pixels"
                )

                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

                _wave += np.interp(_xpi, _xtr, dwave)
                _wtr = _xwtr

            xtr.append(_xtr[sl[1]] - sl[1].start)
            ytr.append(_ytr[sl[1]] - sl[0].start)
            wtr.append(_wtr[sl[1]])
            wave.append(_wave[sl].flatten())
            dwave_dx.append(np.gradient(_wave, axis=1)[sl].flatten())

            # SFLAT from shutter names of first exposure
            needs_sflat = (j == 0) & (slit.meta.exposure.type == "NRS_MSASPEC")
            needs_sflat &= msautils.slit_data_from_extended_reference(slit)
            needs_sflat &= with_sflat_correction
            needs_sflat &= slit.meta.instrument.grating.upper() == "PRISM"

            if needs_sflat:
                sflat_data = msautils.msa_slit_sflat(
                    slit,
                    slit_wave=_wave,
                    apply=False,
                    force=False,
                    verbose=VERBOSE_LOG,
                )
                if sflat_data is not None:
                    sflat_data = sflat_data[sl].flatten()

            if sflat_data is not None:
                sci[j, :] /= sflat_data
                var_rnoise[j, :] /= sflat_data**2
                var_poisson[j, :] /= sflat_data**2

            slit_frame_y.append(_sy.flatten())

            for k in attr_keys:
                self.info[k][j] = getattr(slit, k)

        xslit = np.array(xslit)
        yslit = np.array(yslit)
        ypix = np.array(ypix)
        wave = np.array(wave)
        dwave_dx = np.array(dwave_dx)
        slit_frame_y = np.array(slit_frame_y)

        xtr = np.array(xtr)
        ytr = np.array(ytr)
        wtr = np.array(wtr)

        if msautils.BAD_PIXEL_FLAG > 0:
            bad = (dq & msautils.BAD_PIXEL_FLAG) > 0
            if (VERBOSE_LOG & 4):
                msg = f'DEBUG mask: valid N={(~bad).sum():<6} BAD_PIXEL_FLAG'
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=True
                )

        if debug:
            print("msautils.BAD_PIXEL_FLAG ; bad = ", bad.sum(), dq.sum())

        # Extra bad pix
        if with_extra_dq:
            for i, slit in enumerate(slits):
                sl = slit.slice
                dqi, dqf = msautils.extra_slit_dq_flags(
                    slit, verbose=False, # (VERBOSE_LOG > 1)
                )
                bad[i, :] |= dqi[slit.slice].flatten() > 0

            if (VERBOSE_LOG & 4):
                msg = f'DEBUG mask: valid N={(~bad).sum():<6} extra_slit_dq_flags'
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=True
                )

            if debug:
                print("msautils.extra_slit_dq_flags ; bad = ", bad.sum())

        # Dilate stuck open pixels
        if ("MSA_FAILED_OPEN" in msautils.BAD_PIXEL_NAMES) & self.meta[
            "dilate_failed_open"
        ]:
            _bp = jwst.datamodels.dqflags.pixel["MSA_FAILED_OPEN"]
            for i in range(self.N):
                grow_open = nd.binary_dilation(
                    (dq[i].reshape(self.sh) & _bp) > 0,
                    iterations=self.meta["dilate_failed_open"] * 1,
                )
                bad[i] |= grow_open.flatten()

            if (VERBOSE_LOG & 4):
                msg = f'DEBUG mask: valid N={(~bad).sum():<6} dilate_failed_open'
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=True
                )

        # Bad pixels in 2 or more positions should be bad in all
        # if self.N > 2:
        #     nbad = bad.sum(axis=0)
        #     all_bad = nbad >= 2 * (self.N // 3)
        #     for i in range(self.N):
        #         bad[i, :] |= all_bad

        bad |= ~np.isfinite(sci) | (sci == 0)
        bad |= ~np.isfinite(phot_corr)

        if self.grating in ["PRISM"]:
            bad |= sci > PRISM_MAX_VALID
            bad |= sci < PRISM_MIN_VALID_SN * np.sqrt(var_rnoise + var_poisson)

            if (VERBOSE_LOG & 4):
                msg = (
                    f'DEBUG mask: valid N={(~bad).sum():<6} ' +
                    f'prism max {PRISM_MAX_VALID} min {PRISM_MIN_VALID_SN}: '
                )
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=True
                )

        if (~bad).sum() > 0:
            # print('dq before threshold:', bad.sum())
            bad |= sci < -5 * np.sqrt(
                np.nanmedian((var_rnoise + var_poisson)[~bad])
            )

            bad |= var_rnoise > 10 * np.nanpercentile(
                var_rnoise[~bad], RNOISE_THRESHOLD
            )
            # print('dq after threshold:', bad.sum())

            if (VERBOSE_LOG & 4):
                msg = f'DEBUG mask: valid N={(~bad).sum():<6} rnoise {RNOISE_THRESHOLD}'
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=True
                )

        if self.meta["pad_border"] > 0:
            # grow mask around edges
            for i in range(len(slits)):
                ysl_i = wave[i, :].reshape(self.sh)
                msk = nd.binary_dilation(
                    ~np.isfinite(ysl_i), iterations=self.meta["pad_border"]
                )
                bad[i, :] |= (msk).flatten()

        if (VERBOSE_LOG & 4):
            msg = (
                f'DEBUG mask: valid N={(~bad).sum():<6} pad_border '
                + f'{self.meta["pad_border"]}'
            )
            utils.log_comment(
                utils.LOGFILE, msg, verbose=True
            )

        sci[bad] = np.nan
        mask = np.isfinite(sci)
        var_rnoise[~mask] = np.nan
        var_poisson[~mask] = np.nan

        self.sci = sci * phot_corr
        self.phot_corr = phot_corr
        self.dq = dq
        self.mask = mask & True
        self.bkg_mask = mask & True
        self.var_rnoise = var_rnoise * phot_corr**2
        self.var_poisson = var_poisson * phot_corr**2
        self.var_sky = np.zeros_like(sci)

        self.sflat_data = sflat_data
        self.pathloss_corr = np.ones_like(self.sci)

        for j, slit in enumerate(slits):

            sl = slit.slice

            phot_scl = slit.meta.photometry.pixelarea_steradians * 1.0e12
            # phot_scl *= (slit.pathloss_uniform / slit.pathloss_point)[sl].flatten()
            # Remove pathloss correction
            if self.meta["undo_pathloss"]:
                if slit.source_type is None:
                    pl_ext = "PATHLOSS_UN"
                else:
                    pl_ext = "PATHLOSS_PS"

                with pyfits.open(self.files[j]) as sim:
                    if pl_ext in sim:
                        msg = f"   {self.files[j]} source_type={slit.source_type} "
                        msg += f" undo {pl_ext}"
                        path_j_ = (
                            sim[pl_ext].data.astype(sci.dtype)[sl].flatten()
                        )

                        if path_j_.size == 0:
                            path_j_ = np.ones_like(self.pathloss_corr[j, :])

                        phot_scl *= path_j_
                        self.pathloss_corr[j, :] *= path_j_
                        self.meta["removed_pathloss"] = pl_ext
                    else:
                        msg = f"   {self.files[j]} source_type={slit.source_type} "
                        msg += f" no {pl_ext} found"
                        utils.log_comment(
                            utils.LOGFILE, msg, verbose=VERBOSE_LOG
                        )

                    if self.meta["undo_pathloss"] == 2:
                        # Apply point source
                        msg = f"   {self.files[j]} apply PATHLOSS_PS "
                        utils.log_comment(
                            utils.LOGFILE, msg, verbose=VERBOSE_LOG
                        )

                        pl_ps = "PATHLOSS_PS"
                        phot_scl /= (
                            sim[pl_ps].data.astype(sci.dtype)[sl].flatten()
                        )

            self.phot_corr[j, :] *= phot_scl
            self.sci[j, :] *= phot_scl
            self.var_rnoise[j, :] *= phot_scl**2
            self.var_poisson[j, :] *= phot_scl**2

        self.var_total = self.var_rnoise + self.var_poisson + self.var_sky

        self.xslit = xslit
        self.yslit = yslit
        self.yslit_orig = yslit * 1
        self.ypix = ypix
        self.wave = wave
        self.dwave_dx = dwave_dx
        self.slit_frame_y = slit_frame_y

        self.bar = bar

        self.xtr = xtr
        self.ytr = ytr
        self.wtr = wtr

        if (self.info["source_ra"] < 0.0001).sum() == self.N:
            if self.N == -3:
                msg = "Seems to be a background slit.  "
                msg += "Force [0, {0}, -{0}]".format(self.meta["nod_offset"])
                msg += "pix offsets"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

                self.ytr[0, :] -= 1.0
                self.ytr[1, :] += -1 + self.meta["nod_offset"]
                self.ytr[2, :] -= 1 + self.meta["nod_offset"]

                self.info["manual_position"] = [2, 3, 1]
                self.meta["position_key"] = "manual_position"
            else:

                # offsets = self.info["y_position"] - self.info["y_position"][0]
                offsets = self.info["y_offset"] - self.info["y_offset"][0]
                offsets /= self.slit_pixel_scale

                if self.N == 2:
                    offsets -= np.mean(offsets)

                # offsets = np.round(offsets / 5) * 5

                offstr = ", ".join(
                    [f"{_off:5.1f}" for _off in np.unique(offsets)]
                )

                msg = "Seems to be a background slit.  "
                msg += f"Force {offstr} pix offsets"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

                self.info["manual_position"] = [
                    f"{int(o*10)/10.:.1f}" for o in offsets
                ]

                self.meta["position_key"] = "manual_position"

                for i, _off in enumerate(offsets):
                    self.ytr[i, :] += _off  # - 1

        elif self.IS_FIXED_SLIT & (self.meta["with_fs_offset"]):

            _dy = self.info["y_offset"] - np.median(self.info["y_offset"])
            _dy /= self.slit_pixel_scale

            msg = " Fixed slit: "
            _dystr = ", ".join([f"{_dyi:5.2f}" for _dyi in _dy])

            msg += f"force [{_dystr}] pix offsets"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

            for i, _dyi in enumerate(_dy):
                self.ytr[i, :] += 1 + _dyi

        elif self.meta["trace_from_yoffset"]:

            _dy = self.info["y_offset"] - self.info["y_offset"][0]
            _dy /= self.slit_pixel_scale

            msg = " Recomputed offsets slit     : "
            _dystr = ", ".join([f"{_dyi:5.2f}" for _dyi in _dy])

            msg += f"force [{_dystr}] pix offsets"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

            for i, _dyi in enumerate(_dy):
                self.ytr[i, :] = self.ytr[0, :] + _dyi

        self.set_trace_coeffs(degree=2)

        # Calculate "num_shutters" for bar shadow correction
        # if not self.IS_FIXED_SLIT:
        if self.IS_FIXED_SLIT:
            self.meta["num_shutters"] = 0
            self.meta["undo_barshadow"] = False
            self.meta["min_bar"] = None
            self.meta["bad_shutter_names"] = []

        if self.meta["num_shutters"] < 0:
            self.meta["num_shutters"] = len(self.info["shutter_state"][0])
        elif self.meta["num_shutters"] == 0:
            self.meta["num_shutters"] = self.unp.N * 1

        if self.meta["undo_barshadow"] == 2:
            self.apply_spline_bar_correction()
            self.meta["undo_barshadow"] = False

        if self.meta["min_bar"] is not None:
            self.mask &= self.bar > self.meta["min_bar"]

        if self.meta["bad_shutter_names"] is None:
            self.mask_stuck_closed_shutters(
                stuck_threshold=self.meta["stuck_threshold"],
            )
        else:
            self.apply_bad_shutter_mask()

    def flag_hot_cold_pixels(
        self,
        cold_percentile=-0.1,
        hot_percentile=1.5,
        absolute=True,
        min_nexp=-1,
        dilate=None,
    ):
        """
        Flag hot/cold pixels where multiple pixels across exposures are below or above
        global percentiles

        Parameters
        ----------
        cold_percentile : float
            Lower limit for "cold" pixels

        hot_percentile : float
            Upper limit for "hot" pixels

        absolute : bool
            If set, ``cold_percentile`` and ``cold_percentile`` are absolute
            flux densities

        min_nexp : int
            Minimum number of exposures required that exceed the threshold. If provided
            as a negative number, will be interpreted as ``Nflag >= Nexp + min_nexp``.
            If positive, will be treated as an absolute number.

        dilate : int, bool
            If provided, do 2D dilation on the masks

        Returns
        -------
        ncold, nhot : int
            Number of flagged cold, hot pixels

        flag : array-like, int
            Flag array: 1 = cold, 2 = hot
        """
        import scipy.ndimage as nd

        try:
            if absolute:
                cold_level = cold_percentile
                hot_level = hot_percentile
            else:
                cold_level, hot_level = np.nanpercentile(
                    self.data[self.mask],
                    [cold_percentile, hot_percentile],
                )
        except TypeError:
            return 0, 0, np.zeros(self.sh, dtype=bool).flatten()

        if min_nexp < 1:
            Nmin = -min_nexp * self.N // self.unp.N
        else:
            Nmin = min_nexp

        Nmin = np.maximum(Nmin, 2)

        hot_flagged = (self.data > hot_level).sum(axis=0) >= Nmin
        cold_flagged = (self.data < cold_level).sum(axis=0) >= Nmin

        if dilate is not None:
            if dilate > 0:
                hot_flagged = nd.binary_dilation(
                    hot_flagged.reshape(self.sh), iterations=dilate
                ).flatten()

                cold_flagged = nd.binary_dilation(
                    cold_flagged.reshape(self.sh), iterations=dilate
                ).flatten()

        ncold, nhot = cold_flagged.sum(), hot_flagged.sum()

        msg = (
            f" {'flag_hot_cold_pixels':<28}: cold = {ncold}   /   hot = {nhot}"
        )
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        return ncold, nhot, cold_flagged * 1 + hot_flagged * 2

    def set_background_spectra(self, path="", **kwargs):
        """
        Try to read global sky background spectra
        """

        bkg_files = []
        for j, slit in enumerate(self.slits):
            file_root = "_".join(slit.meta.filename.split("_")[:4])
            bkg_file = os.path.join(path, file_root + "_gbkg.fits")
            if os.path.exists(bkg_file):
                bkg_files.append(bkg_file)

        if len(bkg_files) != len(self.slits):
            return None

        self.sky_data = {"use": True, "sky2d": np.zeros_like(self.sci)}

        for j, slit in enumerate(self.slits):
            file_root = "_".join(slit.meta.filename.split("_")[:4])
            bkg_file = file_root + "_gbkg.fits"
            if os.path.exists(bkg_file):
                msg = (
                    f"{__name__} set_background_spectra: read file {bkg_file}"
                )
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

                bkg_ = utils.read_catalog(bkg_file)

                sky_ = np.interp(
                    self.wave[j, :],
                    bkg_["wave"],
                    bkg_["p50"],
                    left=0,
                    right=0.0,
                )

                esky_ = np.interp(
                    self.wave[j, :],
                    bkg_["wave"],
                    bkg_["perr"],
                    left=0,
                    right=0.0,
                )

                if "REFPAREA" in bkg_.meta:
                    ref_pixarea = bkg_.meta["REFPAREA"]
                else:
                    ref_pixarea = 4.88e-13

                area_corr = (
                    slit.meta.photometry.pixelarea_steradians / ref_pixarea
                )
                sky_ *= (
                    self.phot_corr[j, :] / self.pathloss_corr[j, :] / area_corr
                )

                esky_ *= (
                    self.phot_corr[j, :] / self.pathloss_corr[j, :] / area_corr
                )
                vsky_ = esky_**2 + (0.03 * sky_) ** 2
                self.sky_data["sky2d"][j, :] = sky_
                self.var_sky[j, :] = vsky_

        # ## comps
        # if 0:
        #     print('xxx sky pca')
        #     pca = utils.read_catalog("/tmp/prism_bkg_pca.fits")
        #     Asky = np.array([
        #         np.interp(
        #             self.wave.flatten(),
        #             pca['wave'],
        #             c,
        #             left=np.nan, right=np.nan
        #         )
        #         for c in pca['bkg_pca'].T
        #     ])
        #
        #     Asky *= (self.phot_corr / self.pathloss_corr).flatten()
        #
        #     sivar = np.sqrt(self.var_rnoise + self.var_poisson).flatten()
        #     ok = np.isfinite(Asky.sum(axis=0)) & self.mask.flatten()
        #     ok &= sivar > 0
        #     ok &= np.abs(self.slit_frame_y.flatten()) > 0.3
        #
        #     print(f'yyy {Asky.shape} {ok.shape} {sivar.shape} {ok.sum()} {self.slit_frame_y[self.mask].min()}')
        #
        #     pca_coeffs = np.linalg.lstsq(
        #         (Asky[:,ok] / sivar[ok]).T,
        #         self.sci.flatten()[ok] / sivar[ok],
        #         rcond=None
        #     )
        #     sky_ = Asky.T.dot(pca_coeffs[0]).reshape(self.wave.shape)
        #     print(f'yyy {sky_.shape}')
        #     self.sky_data["sky2d"] = sky_

        self.var_total = self.var_rnoise + self.var_poisson + self.var_sky
        self.mask &= np.isfinite(self.var_total) & (self.var_total > 0)

    def estimate_sky(
        self,
        mask_yslit=[[-4.5, 4.5]],
        min_bar=0.95,
        var_percentiles=[-5, -5],
        df=51,
        high_clip=0.8,
        use=True,
        outlier_threshold=7,
        absolute_threshold=0.2,
        make_plot=False,
        grating_limits=msautils.GRATING_LIMITS,
        **kwargs,
    ):
        """
        Estimate sky spectrum by fitting a flexible spline to all available pixels in
        indicated empty shutter areas

        Parameters
        ----------
        mask_yslit : list of (float, float)
            Range of shutter pixels to exclude as coming from the source and/or
            contaminants

        min_bar : float
            Minimum allowed value of the bar obscuration

        var_percentiles : None, (float, float)
            Exclude sky pixels with variances outside of this percentile range.  If
            a negative number is provided, treat as an explicit number of pixels
            to exclude from the low and high sides.

        df : int
            Degrees of freedom of the spline fit.  If ``df = 0``, then just compute
            a scalar normalization factor.  If ``df < 0``, then don't rescale at all.

        use : bool
            Use the resulting sky model for the local sky

        outlier_threshold : float, None
            Mask pixels where the residual w.r.t the sky model is greater than this

        make_plot : bool
            Make a diagnostic plto

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object if ``make_figure`` else None

        Stores sky fit results in ``sky_data`` attribute

        """
        exclude_yslit = np.zeros_like(self.mask, dtype=bool)
        for _yrange in mask_yslit:
            exclude_yslit |= (self.yslit > _yrange[0]) & (
                self.yslit < _yrange[1]
            )

        full_ok_sky = self.mask & ~exclude_yslit & (self.sci < high_clip)
        ok_sky = full_ok_sky & True

        if min_bar is not None:
            ok_sky &= self.bar > min_bar

        if ok_sky.sum() < 64:
            return None

        if var_percentiles is not None:
            var_parray = np.array(var_percentiles)
            absolute_clip = False

            var_p = np.nanpercentile(
                self.var_total[full_ok_sky], np.abs(var_parray)
            )
            so = np.argsort(self.var_total[full_ok_sky])

            if var_parray[0] < 0:
                var_p[0] = self.var_total[full_ok_sky][so][int(-var_parray[0])]
                absolute_clip = True
            if var_parray[1] < 0:
                var_p[1] = self.var_total[full_ok_sky][so][
                    -int(-var_parray[1])
                ]
                absolute_clip = True

            var_clip = (self.var_total > var_p[0]) & (
                self.var_total < var_p[1]
            )
            ok_sky &= var_clip
        else:
            absolute_clip = False

        sky_wave = msautils.get_standard_wavelength_grid(
            self.grating, sample=1.0, grating_limits=grating_limits
        )
        nbin = sky_wave.shape[0]
        xbin = np.interp(self.wave, sky_wave, np.arange(nbin) / nbin)

        if self.meta["sky_file"] not in [None, "N/A"]:
            df_use = self.meta["global_sky_df"]
        else:
            df_use = df

        if df_use <= 0:  # | (self.sky_arrays is not None):
            spl_full = np.ones(xbin.size)[:, None]
        else:
            spl_full = utils.bspline_templates(
                xbin.flatten(),
                df=df_use,
                minmax=(0, 1),
                get_matrix=True,
            )

        if self.sky_arrays is not None:
            _sky_interp = np.interp(
                self.wave.flatten(),
                *self.sky_arrays,
                left=np.nan,
                right=np.nan,
            )
            ok_sky &= np.isfinite(_sky_interp.reshape(ok_sky.shape))
            spl_full = (spl_full.T * _sky_interp).T

        if ok_sky.sum() == 0:
            msg = f" {'estimate_sky':<28}: no valid sky pixels"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
            return None

        ok_skyf = ok_sky.flatten()
        nspl = spl_full[ok_skyf, :].sum(axis=0)
        spl_trim = nspl > 1.0e-4 * nspl.max()
        spl_full = spl_full[:, spl_trim]

        AxT = (spl_full[ok_skyf, :].T / np.sqrt(self.var_rnoise[ok_sky])).T
        yx = (self.sci / np.sqrt(self.var_rnoise))[ok_sky]
        if df_use < 0:  # | (self.sky_arrays is not None):
            sky_coeffs = np.array([1.0])
        else:
            sky_coeffs = np.linalg.lstsq(AxT, yx, rcond=None)[0]

        if absolute_clip & (absolute_threshold is not None):
            # Flag outliers
            sky2d = spl_full.dot(sky_coeffs).reshape(self.sci.shape)
            sky_bad = np.abs(self.sci - sky2d) > absolute_threshold
            var_clip |= sky_bad & ok_sky
            ok_sky &= ~sky_bad
            ok_skyf = ok_sky.flatten()

            if df_use <= 0:
                spl_full = np.ones(xbin.size)[:, None]
            else:
                spl_full = utils.bspline_templates(
                    xbin.flatten(),
                    df=df_use,
                    minmax=(0, 1),
                    get_matrix=True,
                )
            nspl = spl_full[ok_skyf, :].sum(axis=0)
            spl_trim = nspl > 1.0e-4 * nspl.max()
            spl_full = spl_full[:, spl_trim]

            if self.sky_arrays is not None:
                spl_full = (spl_full.T * _sky_interp).T

            AxT = (spl_full[ok_skyf, :].T / np.sqrt(self.var_rnoise[ok_sky])).T
            yx = (self.sci / np.sqrt(self.var_rnoise))[ok_sky]
            if df_use < 0:
                sky_coeffs = np.array([1.0])
            else:
                sky_coeffs = np.linalg.lstsq(AxT, yx, rcond=None)[0]

        if df_use <= 0:
            spl_array = np.ones(nbin)[:, None]
            sky_covar = np.array([1.0])
        else:
            sky_covar = utils.safe_invert(np.dot(AxT.T, AxT))

            spl_array = utils.bspline_templates(
                np.arange(nbin) / nbin,
                df=df_use,
                minmax=(0, 1),
                get_matrix=True,
            )

        spl_array = spl_array[:, spl_trim]
        if self.sky_arrays is not None:
            _sky_interp = np.interp(sky_wave, *self.sky_arrays)
            spl_array = (spl_array.T * _sky_interp).T

        sky_model = spl_array.dot(sky_coeffs)
        sky2d = spl_full.dot(sky_coeffs).reshape(self.sci.shape)

        sky_model[(sky_wave < np.nanmin(self.wave[ok_sky]))] = np.nan
        sky_model[(sky_wave > np.nanmax(self.wave[ok_sky]))] = np.nan

        self.sky_data = {
            "sky_wave": sky_wave,
            "sky_model": sky_model,
            "sky_coeffs": sky_coeffs,
            "sky_covar": sky_covar,
            "spl_trim": spl_trim,
            "df": df_use,
            "sky2d": sky2d,
            "use": use,
            "npix": ok_sky.sum(),
            "mask": ok_sky,
        }

        msg = f" {'estimate_sky':<28}: "
        if outlier_threshold is not None:
            sky_outliers = np.abs(
                self.sci - sky2d
            ) > outlier_threshold * np.sqrt(self.var_total)
            sky_outliers &= full_ok_sky
            if absolute_clip:
                sky_outliers |= full_ok_sky & ~var_clip

            self.mask &= ~sky_outliers

            msg += f"{sky_outliers.sum()} outliers > {outlier_threshold}  / "
        else:
            sky_outliers = None

        msg += f"N={ok_sky.sum()} sky pixels "
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        if make_plot:
            fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            ax = axes[0]
            ax.scatter(
                self.wave[full_ok_sky],
                self.sci[full_ok_sky],
                c=self.yslit[full_ok_sky],
                alpha=0.1,
            )

            if sky_outliers is not None:
                ax.scatter(
                    self.wave[sky_outliers],
                    self.sci[sky_outliers],
                    ec="r",
                    fc="None",
                    s=100,
                    alpha=0.8,
                )

            ax.plot(sky_wave, sky_model, color="magenta", alpha=0.5)
            ax.grid()
            ymax = np.nanmax(sky_model)
            ax.set_ylim(-0.1 * ymax, ymax * 1.1)
            ax.set_ylabel(r"sky $f_\nu$, $\mu$Jy")

            ax = axes[1]
            ax.scatter(
                self.wave[full_ok_sky],
                ((self.sci - sky2d) / np.sqrt(self.var_total))[full_ok_sky],
                c=self.yslit[full_ok_sky],
                alpha=0.1,
            )

            ax.plot(sky_wave, sky_model * 0, color="magenta", alpha=0.5)
            ax.set_ylim(-7, 7)
            ax.grid()
            ax.set_xlabel("wavelength")
            ax.set_ylabel("residual, sigma")
        else:
            fig = None

        return fig

    def fit_shutter_offset(
        self,
        minimizer_kwargs={"method": "bfgs", "tol": 1.0e-5},
        update=True,
        make_plot=False,
        max_sn=50,
        prior=(0, 0.05),
    ):
        """
        Fit for an offset of the `slit_frame_y` coordinate using sky residuals
        """

        if not hasattr(self, "sky_data"):
            msg = f" {'fit_shutter_offset':<28}: no `sky_data` found"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
            return None

        try:
            tfit = self.get_trace_sn()
            sn_value = np.nanpercentile(tfit["sn"], 95)

            if sn_value > max_sn:
                msg = f" {'fit_shutter_offset':<28}: {sn_value:.1f} > {max_sn}"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
                return None
        except:
            msg = f" {'fit_shutter_offset':<28}: error"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

            return None

        # Exclude the outer edge
        mask = np.abs(self.slit_frame_y / self.slit_shutter_scale / 5.0) < 1.4
        mask &= self.mask
        if "mask" in self.sky_data:
            mask &= self.sky_data["mask"]

        var_total = self.var_poisson + self.var_rnoise

        dy_array = []
        chi2_array = []

        def _objfun_shutter(theta):
            """
            objective function for fitting the shutter offset
            """

            dy = theta[0] / 100.0

            shutter_y = (
                self.slit_frame_y / self.slit_shutter_scale + dy
            ) / 5.0

            bar, bar_wrapped = msautils.get_prism_wave_bar_correction(
                shutter_y[mask],
                self.wave[mask],
                num_shutters=self.meta["num_shutters"],
                wrap=False,
            )

            # Compute residuals in uncorrected frame:
            # Residual = sci * bar_correction - sky * new_bar
            resid = (self.sci * self.bar)[mask] - self.sky_data["sky2d"][
                mask
            ] * bar
            resid /= np.sqrt(var_total[mask])

            chi2 = np.nansum(resid**2)
            chi2 += (dy - prior[0]) ** 2 / prior[1] ** 2

            # print(f"xxx shutter offset {dy:.04f}  {chi2:10.2f}")
            dy_array.append(dy)
            chi2_array.append(chi2)

            return chi2

        x0 = self.meta["shutter_offset"] * 100

        res = minimize(_objfun_shutter, x0=x0, **minimizer_kwargs)

        so = np.argsort(dy_array)
        res.samples = np.array(dy_array)[so]
        res.sample_chi2 = np.array(chi2_array)[so]

        if make_plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(res.samples * 100, res.sample_chi2, marker=".")
            yl = ax.get_ylim()
            ax.vlines(res.x[0], *yl, linestyle=":")
            ax.grid()
            ax.set_xlabel("shutter_offset, pix x 100")
            ax.set_ylabel(r"$\chi^2$")

        msg = f" {'fit_shutter_offset':<28}: shutter_offset = {res.x[0]/100:.2f} pixels"
        msg += f" (update={update})"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        if update:
            self.meta["shutter_offset"] = res.x[0] / 100.0
            self.apply_spline_bar_correction()

        return res

    def flag_trace_outliers(
        self, yslit=[-2, 2], filter_size=-2, threshold=1.5
    ):
        """
        Flag pixels outside of the trace that are greater than the maximum inside
        the trace

        Parameters
        ----------
        yslit : [float, float]
            Trace range around the source center

        filter_size : int
            Size of the maximum filter as a function of wavelength.  If negative,
            then the filter size is ``-filter_size * N files``

        threshold : float
            Outlier threshold relative to the trace maximum

        Returns
        -------
        outlier : array-like, None
            Outliers, None if too few pixels found.  Also adds outliers to ``self.mask``
        """
        trace_mask = (self.yslit > yslit[0]) & (self.yslit < yslit[1])

        data = self.data * 1
        in_trace = self.mask & np.isfinite(data) & trace_mask

        if in_trace.sum() < 100:
            msg = f" {'flag_trace_outliers':<28}: too few pixels {in_trace.sum()}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
            return None

        so = np.argsort(self.wave[in_trace])

        fsize = filter_size if filter_size > 0 else -filter_size * self.N
        max_sort = nd.maximum_filter(self.data[in_trace][so], fsize)

        max_interp = np.interp(self.wave, self.wave[in_trace][so], max_sort)

        outlier = (data > max_interp * threshold) & (
            data > 3 * np.sqrt(self.var_total)
        )

        outlier &= ~trace_mask

        self.mask &= ~outlier

        msg = f" {'flag_trace_outliers':<28}: {outlier.sum()} pixels  / yslit = {yslit}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        return outlier

    def flag_percentile_outliers(
        self,
        plevels=[0.95, -4, -0.1],
        scale=2,
        yslit=[-2.0, 2.0],
        dilate=np.ones((3, 3), dtype=int),
        update=True,
        **kwargs,
    ):
        """
        Flag outliers based on a normal distribution.

        Parameters
        ----------
        plevels : [float, float, float]
            Percentile levels (0.0, 1.0).  If the values are < 0, then interpret as an
            absolute number of pixels relative to the total:
            ``plevels[i] = 1 + plevels[i] / (mask.sum() / Nslits)``

        scale : float
            scale factor

        dilate : array, None
            Footprint of binary dilated outlier mask

        update : bool
            Update ``mask`` attribute

        Returns
        -------
        outlier : array-like
            Pixel outliers

        high_level : float
            Threshold level

        """
        from scipy.stats import norm

        trace_mask = (
            self.mask & (self.yslit > yslit[0]) & (self.yslit < yslit[1])
        )
        if trace_mask.sum() < 100:
            msg = f" {'flag_percentile_outliers':<28}: pixels {trace_mask.sum()} < 100"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
            return np.zeros(self.sci.shape, dtype=bool), np.nan

        N_per_slit = trace_mask.sum() / self.N
        plev = np.array(plevels) * 1.0
        for i in range(3):
            if plevels[i] < 0:
                plev[i] = 1.0 + plevels[i] / N_per_slit

        if plev[1] < plev[0]:
            msg = f" {'flag_percentile_outliers':<28}: {plev[1]:.3f} < {plev[0]:.3f}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
            return np.zeros(self.sci.shape, dtype=bool), np.nan

        ppf = norm.ppf(plev)
        pval = np.nanpercentile(self.data[trace_mask], np.array(plev) * 100)

        delta = (pval[1] - pval[0]) / (ppf[1] - ppf[0]) * (ppf[2] - ppf[0])
        high_level = pval[0] + delta * scale

        outlier = self.data > high_level

        msg = f" {'flag_percentile_outliers':<28}: {outlier.sum()} pixels"
        msg += f"  / plevels {plevels} threshold={high_level:.3f}"
        msg += f" (dilate={dilate is not None})"

        if ~np.allclose(np.array(plevels), plev):
            msg += f"\n {'flag_percentile_outliers':<28}: calculated "
            msg += f"{plev[0]:.6f} {plev[1]:.6f} {plev[2]:.6f}"

        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        if outlier.sum() & (dilate is not None):
            if not hasattr(dilate, "shape"):
                dilate = np.ones((3, 3), dtype=int)
            for i in range(self.N):
                outlier[i, :] |= nd.binary_dilation(
                    outlier[i, :].reshape(self.sh),
                    structure=dilate.astype(int),
                ).flatten()

        if update:
            self.mask &= ~outlier
            self.meta["percentile_outliers"] = outlier.sum()

        return outlier, high_level

    def flag_from_profile(
        self,
        grow=2,
        nfilt=-32,
        require_multiple=True,
        make_plot=False,
    ):
        """
        Flag pixel outliers based on the cross-dispersion profile
        - Calculate the p5, p50, and p95 rolling percentiles across the profile
        - Flag high pixels where ``value > p95 + (p95 - p50)*grow``
        - Flag low pixels where ``value < min(p5, 0) - 5 * sigma``

        Parameters
        ----------
        grow : float

        nfilt : int
            Size of the filter window for the profile.  If ``nfilt < 0``, then interpret
            as ``nfilt = self.mask.sum() // -nfilt``, otherwise use directly as the
            filter size

        require_multiple : bool
            Require that flagged pixels appear in multiple exposures

        make_plot : bool
            Make a diagnostic figure

        Returns
        -------
        updates the ``mask`` attribute

        """
        if hasattr(self, "mask_profile"):
            self.mask |= self.mask_profile

        so = np.argsort(self.yslit[self.mask])
        yso = self.yslit[self.mask][so]
        xso = (self.data)[self.mask][so]

        if nfilt < 0:
            # nfilt = self.sh[1] // (-nfilt)
            nfilt = self.mask.sum() // -nfilt

        phi = nd.percentile_filter(xso, 95, nfilt)
        pmi = nd.percentile_filter(xso, 50, nfilt)
        plo = nd.percentile_filter(xso, 5, nfilt)

        hi_thresh = phi + (phi - pmi) * grow
        lo_thresh = np.minimum(plo, 0)

        bad = self.data > np.interp(self.yslit, yso, hi_thresh)
        bad |= self.data < (
            np.interp(self.yslit, yso, lo_thresh) - 5 * np.sqrt(self.var_total)
        )

        if require_multiple:
            if self.N > 2:
                nbad = bad.sum(axis=0)
                bad &= False
                all_bad = nbad >= 2 * (self.N // 3)
                for i in range(self.N):
                    bad[i, :] |= all_bad

        self.mask_profile = bad

        msg = f" {'flag_from_profile':<28}: {bad.sum()} "
        msg += f"({bad.sum() / self.mask.sum() * 100:4.1f}%) pixels"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        if make_plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.scatter(yso, xso, c=self.wave[self.mask][so], alpha=0.1)
            ax.scatter(
                self.yslit[bad],
                self.data[bad],
                ec="r",
                fc="None",
                s=80,
                alpha=0.8,
            )

            ax.plot(yso, hi_thresh, color="r", alpha=0.5)
            ax.plot(yso, lo_thresh, color="r", alpha=0.5)
            ax.plot(yso, np.array([phi, pmi, plo]).T, color="0.5", alpha=0.5)

            ax.set_ylim(
                np.nanmin(lo_thresh), np.nanmax(phi + (phi - pmi) * (grow + 1))
            )
            ax.grid()

        self.mask &= ~bad
        # self.sci[~self.mask] = np.nan

    def set_trace_coeffs(self, degree=2):
        """
        Fit a polynomial to the trace

        Parameters
        ----------
        degree : int
            Polynomial degree of fit to the traces in the ``ytr`` attribute

        Sets the ``base_coeffs`` attribute and initializes ``trace_coeffs``
        with zeros

        """
        coeffs = []
        for i in range(self.N):
            xi = (self.xtr[i, :] - self.sh[1] / 2) / self.sh[1]
            yi = self.ytr[i, :]
            oki = np.isfinite(xi + yi)
            coeffs.append(np.polyfit(xi[oki], yi[oki], degree))

        self.base_coeffs = coeffs
        self.trace_coeffs = [c * 0.0 for c in coeffs]
        self.update_trace_from_coeffs()

    def update_trace_from_coeffs(self):
        """
        Update the ``yslit`` attribute based on the polynomial coefficients in
        the ``base_coeffs`` and ``trace_coeffs`` attributes

        """
        yslit = []
        yshutter = []

        for i in range(self.N):
            xi = (self.xtr[i, :] - self.sh[1] / 2) / self.sh[1]
            _ytr = np.polyval(self.base_coeffs[i], xi)
            if i == 0:
                _ytr0 = _ytr * 1.0

            _ytr += np.polyval(self.trace_coeffs[i], xi)
            yslit.append((self.ypix[i, :].reshape(self.sh) - _ytr).flatten())

            yshutter.append(
                (self.ypix[i, :].reshape(self.sh) - _ytr0).flatten()
            )

        self.yslit = np.array(yslit)
        self.yshutter = np.array(yshutter)

    def apply_normalization_correction(self):
        """
        Apply normalization correction from `~msaexp.utils.get_normalization_correction`

        Only implemented for PRISM for now
        """
        if self.grating not in ["PRISM"]:
            return None

        slit = self.slits[0]
        corr = msautils.get_normalization_correction(
            self.wave,
            slit.quadrant,
            slit.xcen,
            slit.ycen,
            grating=self.grating,
        )

        self.normalization = corr
        self.sci *= corr
        self.var_rnoise *= corr**2
        self.var_poisson *= corr**2

        self.mask &= np.isfinite(corr)

        self.var_total = self.var_poisson + self.var_rnoise + self.var_sky

    def apply_spline_bar_correction(self):
        """
        Own bar shadow correction for PRISM derived from empty background shutters
        and implemented as a flexible bspline

        See `~msaexp.utils.get_prism_bar_correction`.

        Returns
        -------
        Rescales the ``sci`` data array and updates the ``mask``
        """
        global SPLINE_BAR_GRATINGS
        if self.grating.upper() not in SPLINE_BAR_GRATINGS:
            utils.log_comment(
                utils.LOGFILE,
                (
                    " apply_spline_bar_correction : "
                    + f" grating {self.grating.upper()} not in {SPLINE_BAR_GRATINGS}"
                ),
                verbose=VERBOSE_LOG,
            )
            return None

        num_shutters = self.meta["num_shutters"]
        if num_shutters > 3:
            # Force use 3-shutter file
            wrap = True
            num_shutters = 3
        elif num_shutters <= 0:
            wrap = False
            num_shutters = 3
        else:
            wrap = "auto"

        _msg = " (mode='{bar_corr_mode}', wrap={wrap}, num_shutters={num_shutters})"
        utils.log_comment(
            utils.LOGFILE,
            (
                " apply_spline_bar_correction : "
                + _msg.format(wrap=wrap, **self.meta)
            ),
            verbose=VERBOSE_LOG,
        )

        if self.meta["bar_corr_mode"] == "flat":
            bar, bar_wrapped = msautils.get_prism_bar_correction(
                self.fixed_yshutter,
                num_shutters=num_shutters,
                wrap=wrap,
            )
        else:
            bar, bar_wrapped = msautils.get_prism_wave_bar_correction(
                self.fixed_yshutter,
                self.wave,
                num_shutters=num_shutters,
                wrap=wrap,
            )

        self.meta["wrapped_barshadow"] = bar_wrapped
        self.meta["own_barshadow"] = True

        self.sci *= self.bar / bar
        self.var_rnoise *= (self.bar / bar) ** 2
        self.var_poisson *= (self.bar / bar) ** 2
        self.var_total = self.var_poisson + self.var_rnoise + self.var_sky

        self.orig_bar = self.bar * 1
        self.bar = bar * 1
        self.mask &= np.isfinite(self.sci)

    def mask_stuck_closed_shutters(self, stuck_threshold=0.3, min_bar=0.7):
        """
        Identify stuck-closed shutters in prism spectra

        Parameters
        ----------
        stuck_threshold : float
            1. Compute a mask where the bar throughput is greater than ``min_bar``
            2. Compute the median S/N of all pixels in each shutter of the slitlet
            3. If the slitlet is more than one shutter, mask shutters where
               ``sn_shutter < stuck_threshold * max(sn_shutters)``
            4. If the slitlet is a single shutter, mask the shutter if
               the absolute S/N is less than ``stuck_threshold``

        min_bar : float
            Minimum value of the bar shadow mask to treat as valid
            pixels within a shutter

        Returns
        -------
        Updates ``bad_shutter_names`` attribute and runs
        `~msaexp.slit_combine.SlitGroup.apply_bad_shutter_mask`

        """
        if self.grating.upper() != "PRISM":
            self.meta["bad_shutter_names"] = []
            return None

        shutter_y = self.fixed_yshutter

        sn = self.sci / np.sqrt(self.var_total)

        bar_mask = self.mask & (self.bar > min_bar)
        if bar_mask.sum() == 0:
            self.meta["bad_shutter_names"] = []
            return None

        un = utils.Unique(
            np.round(shutter_y[bar_mask]).astype(int), verbose=False
        )
        sn_shutters = np.zeros(un.N, dtype=float)
        for i, v in enumerate(un.values):
            sn_shutters[i] = np.nanmedian(sn[bar_mask][un[v]])

        if un.N == 1:
            bad_shutter = sn_shutters < stuck_threshold
        else:
            bad_shutter = sn_shutters < (stuck_threshold * sn_shutters.max())
            bad_shutter &= un.counts > 0.5 * un.counts.max()

        if bad_shutter.sum() > 0:
            bad_list = [un.values[i] for i in np.where(bad_shutter)[0]]

            self.meta["bad_shutter_names"] = bad_list
        else:
            self.meta["bad_shutter_names"] = []

        self.apply_bad_shutter_mask()

    def apply_bad_shutter_mask(self):
        """
        Mask ``sci`` array for ``bad_shutter_names`` shutters

        """
        if len(self.meta["bad_shutter_names"]) == 0:
            return None

        utils.log_comment(
            utils.LOGFILE,
            f"""!{'apply_bad_shutter_mask':<27}: PRISM stuck bad shutters {self.meta["bad_shutter_names"]}""",
            verbose=VERBOSE_LOG,
        )

        for i in self.meta["bad_shutter_names"]:
            shutter_mask = np.abs(self.fixed_yshutter - i) < 0.6
            self.sci[shutter_mask] = np.nan
            self.mask &= np.isfinite(self.sci)

    @property
    def sky_background(self):
        """
        Optional sky-background data computed from the ``sky_arrays`` or ``sky_data``
        attributes

        Returns
        -------
        sky : array-like
            Sky data with dimensions ``(N, sh[0]*sh[1])``

        """
        if hasattr(self, "sky_data"):
            if self.sky_data["use"]:
                sky = self.sky_data["sky2d"] * 1.0
            else:
                sky = np.zeros_like(self.wave)
        elif self.sky_arrays is not None:
            sky = np.interp(
                self.wave,
                self.sky_arrays[0],
                self.sky_arrays[1],
                left=-1,
                right=-1,
            )

            sky[sky < 0] = np.nan
        else:
            sky = np.zeros_like(self.wave)

        return sky

    @property
    def data(self):
        """
        Evaluate the ``sci`` data including optional ``sky_background`` and
        ``bar`` barshadow attributes

        Returns
        -------
        sci : array-like
            science data with dimensions ``(N, sh[0]*sh[1])``
        """
        sky = self.sky_background

        if self.meta["undo_barshadow"]:
            clean = (self.sci - sky) * self.bar
        else:
            clean = self.sci - sky

        clean[~self.mask] = np.nan
        return clean

    def make_diff_image(self, exp=1, separate=False):
        """
        Make a difference image for an individual exposure group

        Parameters
        ----------
        exp : int
            Exposure group

        separate : bool
            Return separate values of each pair of images, e.g., the ``pos, neg``
            components of ``diff = pos - neg``

        Returns
        -------
        ipos : array-like
            Array indices of the "positive" exposures

        ineg : array-like
            Array indices of the "negative" exposures at the other nod
            positions

        diff : array-like
            Flattened difference image

        bdiff : array-like
            Flattened sky background difference image

        vdiff : array-like
            Flattened variance image

        wdiff : array-like
            Flattened weight image

        """
        ipos = self.unp[exp]

        pos = np.nansum((self.data * self.mask)[ipos, :], axis=0) / np.nansum(
            self.mask[ipos, :], axis=0
        )

        bpos = np.nansum(
            (self.sky_background * self.mask)[ipos, :], axis=0
        ) / np.nansum(self.mask[ipos, :], axis=0)

        vpos = (
            np.nansum((self.var_total * self.mask)[ipos, :], axis=0)
            / np.nansum(self.mask[ipos, :], axis=0) ** 2
        )

        wpos = (
            np.nansum((self.var_rnoise * self.mask)[ipos, :], axis=0)
            / np.nansum(self.mask[ipos, :], axis=0) ** 2
        )

        if self.meta["diffs"]:
            ineg = ~self.unp[exp]

            neg = np.nansum(
                (self.data * self.mask)[ineg, :], axis=0
            ) / np.nansum(self.bkg_mask[ineg, :], axis=0)

            bneg = np.nansum(
                (self.sky_background * self.mask)[ineg, :], axis=0
            ) / np.nansum(self.bkg_mask[ineg, :], axis=0)

            vneg = (
                np.nansum((self.var_total * self.mask)[ineg, :], axis=0)
                / np.nansum(self.bkg_mask[ineg, :], axis=0) ** 2
            )

            wneg = (
                np.nansum((self.var_rnoise * self.mask)[ineg, :], axis=0)
                / np.nansum(self.bkg_mask[ineg, :], axis=0) ** 2
            )
        else:
            ineg = np.zeros(self.N, dtype=bool)
            neg = np.zeros_like(pos)
            bneg = np.zeros_like(bpos)
            vneg = np.zeros_like(vpos)
            wneg = np.zeros_like(wpos)

        if separate:
            return (
                ipos,
                ineg,
                (pos, neg),
                (bpos, bneg),
                (vpos, vneg),
                (wpos, wneg),
            )
        else:
            diff = pos - neg
            bdiff = bpos - bneg
            vdiff = vpos + vneg
            wdiff = wpos + wneg
            return ipos, ineg, diff, bdiff, vdiff, wdiff

    def plot_2d_differences(
        self,
        fit=None,
        clip_sigma=5,
        kws=dict(cmap="bone_r", interpolation="hanning"),
        figsize=(6, 2),
    ):
        """
        Plot the 2D differences between exposures.

        Parameters
        ----------
        fit : dict, optional
            A dictionary containing the fit information for each exposure output from
            `~msaexp.slit_combine.SlitGroup.fit_all_traces`

        clip_sigma : float, optional
            The number of std. deviations to use for clipping the color scale.

        kws : dict, optional
            Additional keyword arguments to be passed to the `imshow` function.

        figsize : tuple, optional
            The size of the figure in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure.

        """
        Ny = self.unp.N
        if fit is None:
            fit = self.fit

        if fit is None:
            Nx = 1
        else:
            Nx = 3

        fig, axes = plt.subplots(
            Ny,
            Nx,
            figsize=(figsize[0] * Nx, figsize[1] * Ny),
            sharex=True,
            sharey=True,
        )

        if Ny == Nx == 1:
            axes = [[axes]]

        ref_exp = self.calc_reference_exposure
        nods = self.relative_nod_offset

        for i, exp in enumerate(self.unp.values):
            ipos, ineg, diff, bdiff, vdiff, wdiff = self.make_diff_image(
                exp=exp
            )

            if fit is not None:
                model = fit[exp]["smod"]
            else:
                model = None

            vmax = clip_sigma * np.nanpercentile(np.sqrt(vdiff), 50)

            kws["vmin"] = -1 * vmax
            kws["vmax"] = 1.5 * vmax

            ax = axes[i][0]
            ax.imshow(
                diff.reshape(self.sh), aspect="auto", origin="lower", **kws
            )
            if i == 0:
                ax.text(
                    0.05,
                    0.95,
                    f"{self.name}",
                    ha="left",
                    va="top",
                    fontsize=10,
                    transform=ax.transAxes,
                    bbox={
                        "fc": "w",
                        "alpha": 0.1,
                        "ec": "None",
                    },
                )

            star = "*" if exp == ref_exp else " "
            spacer = " " * 5
            msg = f"exp={exp}{star}  nod={nods[ipos][0]:5.1f}"
            msg += f"{spacer}Npos = {ipos.sum()} {spacer}Nneg = {ineg.sum()}"

            ax.text(
                0.05,
                0.05,
                msg,
                ha="left",
                va="bottom",
                fontsize=8,
                transform=ax.transAxes,
                bbox={
                    "fc": "w",
                    "alpha": 0.1,
                    "ec": "None",
                },
            )

            if model is not None:
                axes[i][1].imshow(model, aspect="auto", origin="lower", **kws)
                axes[i][2].imshow(
                    diff.reshape(self.sh) - model,
                    aspect="auto",
                    origin="lower",
                    **kws,
                )

            for ax in axes[i]:
                for j in np.where(ipos)[0]:
                    xj = (self.xtr[j, :] - self.sh[1] / 2) / self.sh[1]
                    _ytr = np.polyval(self.trace_coeffs[j], xj)
                    _ytr += np.polyval(self.base_coeffs[j], xj)
                    _ = ax.plot(_ytr, color="tomato", alpha=0.3, lw=2)

                for j in np.where(ineg)[0]:
                    xj = (self.xtr[j, :] - self.sh[1] / 2) / self.sh[1]
                    _ytr = np.polyval(self.trace_coeffs[j], xj)
                    _ytr += np.polyval(self.base_coeffs[j], xj)
                    _ = ax.plot(_ytr, color="wheat", alpha=0.3, lw=2)

                # ax.grid()

        fig.tight_layout(pad=1)
        return fig

    def get_flat_diff_arrays(
        self, apply_mask=True, fit=None, with_pathloss=False, float_dtype=float
    ):
        """
        Make flattened versions of the diff arrays suitable for rebinning

        Parameters
        ----------
        apply_mask : bool
            Only return valid rows where ``self.mask = True``

        fit : dict, optional
            A dictionary containing the fit information for each exposure output from
            `~msaexp.slit_combine.SlitGroup.fit_all_traces`

        with_pathloss : bool
            Derive path loss correction from fitted profile and shutter centering

        float_dtype : type
            Float data type
        Returns
        -------
        tab : `~grizli.utils.GTable`
            Stacked table of the pixel data
        """
        exp_ids = self.unp.values

        if fit is None:
            fit = self.fit

        flat_diff = []
        flat_sky = []
        flat_profile = []
        flat_var_total = []
        flat_var_rnoise = []
        flat_mask = []
        flat_wave = []
        flat_dwave_dx = []
        flat_yslit = []
        flat_yshutter = []
        flat_index = []
        flat_bar = []
        flat_pathloss = []
        flat_normalization = []

        bkg = self.sky_background
        pscale = self.slit_pixel_scale

        header = OrderedDict()
        header["WITHPATH"] = (with_pathloss, "Internal path loss correction")

        for ii, exp in enumerate(exp_ids):
            _ = self.make_diff_image(exp=exp, separate=True)
            (
                ipos,
                ineg,
                (pos, neg),
                (bpos, bneg),
                (vpos, vneg),
                (wpos, wneg),
            ) = _

            ixp = np.where(ipos)[0]
            ixn = np.where(ipos)[0]

            # "negative" background
            if ineg.sum() > 0:
                sky = None
                msk_neg = np.zeros_like(self.mask[0, :])
                for i in ixn:
                    msk_neg |= self.mask[i, :]
            else:
                sky = bkg
                msk_neg = np.ones_like(self.mask[0, :])

            if fit is not None:
                # Normalized profile
                prof = (
                    fit[exp]["smod"] * fit[exp]["sden"] / fit[exp]["snum"]
                ).flatten()
            else:
                prof = None

            for i in ixp:
                flat_diff.append(self.data[i, :] - neg)

                if sky is None:
                    # "sky" is the negative component of the diff
                    flat_sky.append(neg)
                else:
                    # Other sky background, e.g., from estimate_sky
                    flat_sky.append(sky[i, :])

                flat_mask.append(self.mask[i, :] & msk_neg)

                flat_var_total.append(self.var_total[i, :] + vneg)
                flat_var_rnoise.append(self.var_rnoise[i, :] + wneg)

                flat_bar.append(self.bar[i, :])
                flat_wave.append(self.wave[i, :])
                flat_dwave_dx.append(self.dwave_dx[i, :])
                flat_yslit.append(self.yslit[i, :])
                flat_yshutter.append(self.fixed_yshutter[i, :])

                if prof is not None:
                    flat_profile.append(prof)
                else:
                    flat_profile.append(np.ones_like(neg))

                flat_index.append(i * np.ones(neg.shape, dtype=int))

                if hasattr(self, "normalization"):
                    flat_normalization.append(self.normalization[i, :])

                # path loss
                if (
                    with_pathloss
                    & (fit is not None)
                    & (self.lookup_prf is not None)
                ):
                    header[f"XPOS{i}"] = (
                        self.slits[i].source_xpos,
                        f"source_xpos of group {exp}",
                    )

                    # path_i = slit_prf_fraction(
                    #     self.wave[i, :],
                    #     sigma=fit[exp]["sigma"],
                    #     x_pos=self.slits[i].source_xpos,
                    #     slit_width=0.2,
                    #     pixel_scale=pscale,
                    #     verbose=False,
                    # )
                    #
                    # # Relative to centered point source
                    # path_0 = slit_prf_fraction(
                    #     self.wave[i, :],
                    #     sigma=0.01,
                    #     x_pos=0.0,
                    #     slit_width=0.2,
                    #     pixel_scale=pscale,
                    #     verbose=False,
                    # )
                    # path_i /= path_0

                    path_i = self.lookup_prf.path_loss(
                        self.wave[i, :],
                        sigma=fit[exp]["sigma"],
                        x_offset=self.slits[i].source_xpos,
                        order=LOOKUP_PRF_ORDER,
                    )

                else:
                    # print('WithOUT PRF pathloss')
                    path_i = np.ones_like(self.sci[i, :])

                flat_pathloss.append(path_i)

        data = {
            "sci": np.hstack(flat_diff).astype(float_dtype),
            "sky": np.hstack(flat_sky).astype(float_dtype),
            "mask": np.hstack(flat_mask),
            "var_total": np.hstack(flat_var_total).astype(float_dtype),
            "var_rnoise": np.hstack(flat_var_rnoise).astype(float_dtype),
            "wave": np.hstack(flat_wave).astype(float_dtype),
            "dwave_dx": np.hstack(flat_dwave_dx).astype(float_dtype),
            "yslit": np.hstack(flat_yslit).astype(float_dtype),
            "yshutter": np.hstack(flat_yshutter).astype(float_dtype),
            "bar": np.hstack(flat_bar).astype(float_dtype),
            "profile": np.hstack(flat_profile).astype(float_dtype),
            "pathloss": np.hstack(flat_pathloss).astype(float_dtype),
            "exposure_index": np.hstack(flat_index).astype(int),
        }

        if len(flat_normalization) > 0:
            data["normalization"] = np.hstack(flat_normalization).astype(
                float_dtype
            )

        data["mask"] &= np.isfinite(data["sci"])
        for c in data:
            data["mask"] &= np.isfinite(data[c])

        tab = utils.GTable(data)
        if apply_mask:
            tab = tab[tab["mask"]]
            tab.remove_column("mask")

        tab["exptime"] = self.exptime[tab["exposure_index"]].astype(
            float_dtype
        )

        for k in header:
            tab.meta[k] = header[k]

        return tab

    def fit_all_traces(self, niter=3, dchi_threshold=-25, ref_exp=2, **kwargs):
        """
        Fit all traces in the group

        Parameters
        ----------
        niter : int
            Number of iterations for fitting the traces (default: 3)

        dchi_threshold : float
            Threshold value for the change in chi-square to consider a fit
            (default: -25)

        ref_exp : int
            Reference exposure for fitting the traces (default: 2)

        kwargs : dict
            Additional keyword arguments for the fitting process

        Returns
        -------
        fit : dict
            Dictionary containing the fit results for each exposure group

        """
        fit = {}

        if ref_exp is None:
            exp_groups = self.unp.values
        else:
            exp_groups = [ref_exp]
            for p in self.unp.values:
                if p not in exp_groups:
                    exp_groups.append(p)

        if "evaluate" in kwargs:
            force_evaluate = kwargs["evaluate"]
        else:
            force_evaluate = None

        for k in range(niter):
            utils.log_comment(
                utils.LOGFILE,
                f"   fit_all_traces, iter {k}",
                verbose=VERBOSE_LOG,
            )

            for i, exp in enumerate(exp_groups):

                if k > 0:
                    kwargs["x0"] = fit[exp]["theta"]

                if ref_exp is not None:
                    if exp != ref_exp:
                        kwargs["evaluate"] = True
                        kwargs["x0"] = fit[ref_exp]["theta"]
                    else:
                        kwargs["evaluate"] = False
                else:
                    kwargs["evaluate"] = False

                if force_evaluate is not None:
                    kwargs["evaluate"] = force_evaluate

                fit[exp] = self.fit_single_trace(exp=exp, **kwargs)
                dchi = fit[exp]["chi2_fit"] - fit[exp]["chi2_init"]

                msg = f"     Exposure group {exp}   dchi2 = {dchi:9.1f}"

                if (dchi < dchi_threshold) | (kwargs["evaluate"]):
                    msg += "\n"
                    for j in np.where(fit[exp]["ipos"])[0]:
                        self.trace_coeffs[j] = fit[exp]["trace_coeffs"]
                else:
                    msg += "*\n"

                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

            if ref_exp is not None:
                # Match all fits
                for i, exp in enumerate(exp_groups):
                    fit[exp]["theta"] = fit[ref_exp]["theta"]
                    fit[exp]["trace_coeffs"] = fit[ref_exp]["trace_coeffs"]
                    for j in np.where(fit[exp]["ipos"])[0]:
                        self.trace_coeffs[j] = fit[exp]["trace_coeffs"]

            self.update_trace_from_coeffs()

        self.fit = fit

        return fit

    def fit_single_trace(
        self,
        x0=None,
        initial_sigma=3.0,
        exp=1,
        force_positive=True,
        method="powell",
        tol=1.0e-6,
        evaluate=False,
        degree=2,
        sigma_bounds=(1, 20),
        trace_bounds=(-1, 1),
        fix_sigma=-1,
        with_bounds=True,
        verbose=True,
        **kwargs,
    ):
        """
        Fit profile width and trace offset polynomial to one of the nod traces

        Parameters
        ----------
        x0 : array-like, None
            Initial parameter guess

        initial_sigma : float
            Initial profile sigma to use (pixels*10)

        exp : int
            Exposure index (see ``self.unp``)

        force_positive : bool
            Don't consider the negative subtracted parts of the difference
            image

        method, tol : str, float
            Optimization parameters

        evaluate : bool
            Don't fit, just evaluate with given parameters

        degree : int
            Trace offset polynomial degree

        sigma_bounds : (float, float)
            Bounds on profile width

        trace_bounds : (float, float)
            Bounds on trace offset coefficients

        fix_sigma : float
            If ``fix_sigma > 0``, don't fit for the profile width but fix to
            this value

        with_bounds : bool
            Use ``sigma_bounds`` and ``trace_bounds`` in optimization

        verbose : bool
            Status messages

        Returns
        -------
        fit : dict
            Fit results

        """
        from scipy.optimize import minimize

        global EVAL_COUNT
        ipos, ineg, diff, bdiff, vdiff, wdiff = self.make_diff_image(exp=exp)

        base_coeffs = self.base_coeffs[np.where(ipos)[0][0]]

        args = (
            base_coeffs,
            self.wave,
            self.xslit,
            self.ypix,
            self.bar,
            self.yslit,
            diff,
            vdiff,
            wdiff,
            self.mask,
            ipos,
            ineg,
            self.sh,
            fix_sigma,
            force_positive,
            verbose,
            0,
        )
        xargs = (
            base_coeffs,
            self.wave,
            self.xslit,
            self.ypix,
            self.bar,
            self.yslit,
            diff,
            vdiff,
            wdiff,
            self.mask,
            ipos,
            ineg,
            self.sh,
            fix_sigma,
            force_positive,
            verbose,
            1,
        )

        if x0 is None:
            if fix_sigma > 0:
                x0 = np.zeros(degree + 1)
            else:
                x0 = np.append([initial_sigma], np.zeros(degree + 1))

        if with_bounds:
            if fix_sigma > 0:
                bounds = [trace_bounds] * (len(x0))
            else:
                bounds = [sigma_bounds] + [trace_bounds] * (len(x0) - 1)
        else:
            bounds = None

        if evaluate:
            theta = x0
        else:
            EVAL_COUNT = 0

            _res = minimize(
                objfun_prof_trace,
                x0,
                args=args,
                method=method,
                tol=tol,
                bounds=bounds,
            )

            theta = _res.x

        # Initial values
        _ = objfun_prof_trace(x0, *xargs)
        snum, svnum, sden, smod, sigma, trace_coeffs, chi2_init = _

        # Evaluated
        _ = objfun_prof_trace(theta, *xargs)
        snum, svnum, sden, smod, sigma, trace_coeffs, chi2_fit = _

        out = {
            "theta": theta,
            "sigma": sigma,
            "trace_coeffs": trace_coeffs,
            "chi2_init": chi2_init,
            "chi2_fit": chi2_fit,
            "ipos": ipos,
            "ineg": ineg,
            "diff": diff,
            "vdiff": vdiff,
            "wdiff": wdiff,
            "snum": snum,
            "svnum": svnum,
            "sden": sden,
            "smod": smod,
            "force_positive": force_positive,
            "bounds": bounds,
            "method": method,
            "tol": tol,
        }

        return out

    def get_trace_sn(
        self,
        exposure_position="auto",
        theta=[4.0, 0],
        force_positive=True,
        **kwargs,
    ):
        """
        Compute spectrum S/N along the trace

        Parameters
        ----------
        exposure_position : int, 'auto'
            Reference exposure position to use

        theta : array-like
            Default trace profile parameters

        force_positive : bool
            Parameter on `msaexp.slit_combine.fit_single_trace`

        Returns
        -------
        tfit : dict
            Output from `msaexp.slit_combine.fit_single_trace` with an
            additional ``sn`` item

        """
        ref_exp = (
            self.calc_reference_exposure
            if exposure_position in ["auto"]
            else exposure_position
        )

        if ref_exp is None:
            ref_exp = self.unp.values[0]

        tfit = self.fit_single_trace(
            exp=ref_exp,
            x0=np.array(theta),
            fix_sigma=-1,
            evaluate=True,
            force_positive=force_positive,
            verbose=False,
        )

        tfit["sn"] = tfit["snum"] / np.sqrt(tfit["svnum"])

        return tfit

    def fit_params_by_sn(
        self,
        sn_percentile=80,
        sigma_threshold=5,
        degree_sn=[[-1000], [0]],
        **kwargs,
    ):
        """
        Compute trace offset polynomial degree and whether or not to fix the
        profile sigma width as a function of S/N

        Parameters
        ----------
        sn_percentile : float
            Percentile of the 1D S/N array extracted along the trace

        sigma_threshold : float
            Threshold below which the profile width is fixed (``fix_sigma =
            True``)

        degree_sn : [array-like, array-like]
            The two arrays/lists ``x_sn, y_degree = degree_sn`` define the S/N
            thresholds ``x_sn`` below which a polynomial degree ``y_degree``
            is used

        kwargs : dict
            Keyword arguments passed to the ``get_trace_sn`` method

        Returns
        -------
        sn : array-like
            1D S/N along the dispersion axis

        sn_value : float
            ``sn_percentile`` of ``sn`` array

        fix_sigma : bool
            Test whether SN percentile is below ``sigma_threshold``

        interp_degree : int
            The derived polynomial degree given the estimated SN percentile
            ``interp_degree = np.interp(SN[sn_percentile], x_sn, y_degree)``

        """
        tfit = self.get_trace_sn(**kwargs)
        sn_value = np.nanpercentile(tfit["sn"], sn_percentile)

        if not np.isfinite(sn_value):
            return tfit["sn"], sn_value, True, degree_sn[1][0]

        interp_degree = int(
            np.interp(
                sn_value,
                *degree_sn,
                left=degree_sn[1][0],
                right=degree_sn[1][-1],
            )
        )

        fix_sigma = sn_value < sigma_threshold

        msg = f" fit_params_by_sn{' ':>12}: {self.name}"  # {degree_sn[0]} {degree_sn[1]}'
        msg += f"  SN({sn_percentile:.0f}%) = {sn_value:.1f}  fix_sigma={fix_sigma}"
        msg += f"  degree={interp_degree} "
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        return tfit["sn"], sn_value, fix_sigma, interp_degree

    def plot_profile(self, exp=1, ax=None, fit_result=None, ymax=0.2):
        """
        Make a plot of cross-dispersion profile

        Parameters
        ----------
        exp : int
            Exposure index (see ``self.unp``)

        ax : matplotlib.axes.Axes, optional
            Axes object to plot on.
            If not provided, a new figure and axes will be created.

        fit_result : dict, optional
            Fit results from `fit_single_trace` method.
            If provided, the fitted profile will be plotted.

        ymax : float, optional
            Maximum value for the y-axis. Default is 0.2.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the plot.

        """
        ipos, ineg, diff, bdiff, vdiff, wdiff = self.make_diff_image(exp=exp)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.scatter(self.yslit[0, :], diff, alpha=0.1, color="0.5")
        if fit_result is not None:
            _smod = fit_result["smod"]
            ymax = np.nanpercentile(_smod[_smod], 98) * 2.0

            ax.scatter(self.yslit[0, :], _smod, alpha=0.1, color="r")
            ax.vlines(
                fit_result["sigma"], -ymax, ymax, linestyle=":", color="r"
            )

        ax.set_ylim(-0.5 * ymax, ymax)
        ax.grid()
        fig.tight_layout(pad=1)

        return fig


def pseudo_drizzle(
    xpix,
    ypix,
    data,
    var,
    wht,
    xbin,
    ybin,
    arrays=None,
    oversample=4,
    pixfrac=1,
    sample_axes="y",
):
    """
    2D histogram analogous to drizzle.  Oversamples along cross-dispersion axis
    to approximate pixfrac and smooth over pixel aliasing of curved traces.

    Parameters
    ----------
    xpix : array-like
        X pixel positions.

    ypix : array-like
        Y pixel positions.

    data : array-like
        Data values.

    wht : array-like
        Weight values.

    xbin : array-like
        X bin edges.

    ybin : array-like
        Y bin edges.

    arrays : tuple, optional
        Tuple containing the four output arrays that will be updated in place

    oversample : int, optional
        Oversampling factor. Default is 4.

    pixfrac : float, optional
        Pixel fraction. Default is 1.

    sample_axes : str, 'x','y','xy'
        Axes to sample

    Returns
    -------
    num : array-like
        Weighted data numerator.  The full weighted data result is ``num / den``.

    vnum : array-like
        Weighted variance numerator.  The full weighted variance is
        ``vnum / den**2``.

    den : array-like
        Weighted denominator

    ntot : array-like
        Number of exposures that contribute to each output pixel

    """
    from scipy.stats import binned_statistic_2d

    if arrays is None:
        num = np.zeros((len(ybin) - 1, len(xbin) - 1))
        vnum = num * 0.0
        den = num * 0.0
        ntot = num * 0.0
    else:
        num, vnum, den, ntot = arrays

    samples = msautils.pixfrac_steps(oversample, pixfrac)

    gx = np.gradient(xbin)
    gy = np.gradient(ybin)

    if "x" in sample_axes:
        xsamples = samples
    else:
        xsamples = [0.0]

    if "y" in sample_axes:
        ysamples = samples
    else:
        ysamples = [0.0]

    nsamp = len(xsamples) * len(ysamples)

    for xo in xsamples:
        for yo in ysamples:

            ####
            # Total weight
            res = binned_statistic_2d(
                ypix,
                xpix,
                (wht / nsamp),
                statistic="sum",
                bins=(ybin + yo * gy, xbin + xo * gx),
            )
            ok = np.isfinite(res.statistic) & (res.statistic > 0)
            den[ok] += res.statistic[ok]

            ####
            # Weighted data
            res = binned_statistic_2d(
                ypix,
                xpix,
                (data * wht / nsamp),
                statistic="sum",
                bins=(ybin + yo * gy, xbin + xo * gx),
            )
            num[ok] += res.statistic[ok]

            ####
            # Weighted variance
            res = binned_statistic_2d(
                ypix,
                xpix,
                (var * wht**2 / nsamp),
                statistic="sum",
                bins=(ybin + yo * gy, xbin + xo * gx),
            )
            vnum[ok] += res.statistic[ok]

            ####
            # Counts
            res = binned_statistic_2d(
                ypix,
                xpix,
                ((var + wht) > 0) * 1.0,
                statistic="sum",
                bins=(ybin + yo * gy, xbin + xo * gx),
            )
            ntot[ok] += res.statistic[ok] / nsamp

    return (num, vnum, den, ntot)


def pixel_table_to_1d(pixtab, wave_grid, weight=None, y_range=[-3, 3]):
    """
    Optimal extraction from full pixel table

    Parameters
    ----------
    pixtab : `~astropy.table.Table`
        Output of (stacked) pixel tables from
        `~msaexp.slit_combine.SlitGroup.get_flat_diff_arrays`

    wave_grid : array-like
        Array of output wavelengths

    weight : array-like
        Weights, with same size as ``pixtab``.  If not specified, will use
        ``pixtab['var_rnoise']``

    y_range : (float, float)
        Pixel range to extract with respect to the center of the slitlet

    Returns
    -------
    tab : `~astropy.table.Table`
        1D extraction data

    """
    from scipy.stats import binned_statistic

    wave_bins = msautils.array_to_bin_edges(wave_grid)

    if weight is None:
        weight = pixtab["var_rnoise"]

    mask_dy = np.isfinite(
        pixtab["sci"]
        + pixtab["var_rnoise"]
        + pixtab["var_total"]
        + pixtab["profile"]
    )
    if "mask" in pixtab.colnames:
        mask_dy &= pixtab["mask"]

    mask_dy &= (pixtab["yslit"] >= y_range[0]) & (
        pixtab["yslit"] <= y_range[1]
    )

    res = binned_statistic(
        pixtab["wave"][mask_dy],
        (pixtab["profile"])[mask_dy],
        statistic="sum",
        bins=(wave_bins),
    )
    prof_sum = res.statistic

    res = binned_statistic(
        pixtab["wave"][mask_dy],
        (pixtab["sci"] / pixtab["pathloss"])[mask_dy],
        statistic="sum",
        bins=(wave_bins),
    )
    sci_sum = res.statistic

    res = binned_statistic(
        pixtab["wave"][mask_dy],
        (pixtab["var_total"] / pixtab["pathloss"] ** 2)[mask_dy],
        statistic="sum",
        bins=(wave_bins),
    )
    var_sum = res.statistic

    res = binned_statistic(
        pixtab["wave"][mask_dy],
        (pixtab["sci"] * weight * pixtab["profile"])[mask_dy],
        statistic="sum",
        bins=(wave_bins),
    )
    opt_num = res.statistic

    if "normalization" in pixtab.colnames:
        res = binned_statistic(
            pixtab["wave"][mask_dy],
            (
                pixtab["sci"]
                / pixtab["normalization"]
                * weight
                * pixtab["profile"]
            )[mask_dy],
            statistic="sum",
            bins=(wave_bins),
        )
        opt_nonorm = res.statistic
    else:
        opt_nonorm = None

    res = binned_statistic(
        pixtab["wave"][mask_dy],
        (pixtab["sci"] / pixtab["pathloss"] * weight * pixtab["profile"])[
            mask_dy
        ],
        statistic="sum",
        bins=(wave_bins),
    )
    opt_path_num = res.statistic

    res = binned_statistic(
        pixtab["wave"][mask_dy],
        (pixtab["sky"] / pixtab["pathloss"] * weight * pixtab["profile"] ** 2)[
            mask_dy
        ],
        statistic="sum",
        bins=(wave_bins),
    )
    opt_sky_num = res.statistic

    res = binned_statistic(
        pixtab["wave"][mask_dy],
        (weight * pixtab["profile"])[mask_dy],
        statistic="sum",
        bins=(wave_bins),
    )
    opt_sky_den = res.statistic

    res = binned_statistic(
        pixtab["wave"][mask_dy],
        (
            pixtab["var_total"]
            / pixtab["pathloss"] ** 2
            * weight**2
            * pixtab["profile"] ** 2
        )[mask_dy],
        statistic="sum",
        bins=(wave_bins),
    )
    opt_var_num = res.statistic

    res = binned_statistic(
        pixtab["wave"][mask_dy],
        (pixtab["var_total"] ** 0)[mask_dy],
        statistic="sum",
        bins=(wave_bins),
    )
    count = res.statistic

    res = binned_statistic(
        pixtab["wave"][mask_dy],
        (1.0 * weight * pixtab["profile"] ** 2)[mask_dy],
        statistic="sum",
        bins=(wave_bins),
    )
    opt_den = res.statistic

    tab = utils.GTable()
    tab.meta["ymin1d"] = y_range[0], "Start of 1D extraction"
    tab.meta["ymax1d"] = y_range[1], "End of 1D extraction"

    tab["wave"] = wave_grid
    tab["flux"] = opt_path_num / opt_den
    tab["err"] = np.sqrt(opt_var_num / opt_den**2)
    tab["path_corr"] = opt_path_num / opt_num
    tab["sky"] = opt_sky_num / opt_den
    tab["npix"] = count
    tab["npix"].description = "Number of pixels in the 1D bin"

    if opt_nonorm is not None:
        tab["norm_corr"] = opt_num / opt_nonorm
        tab["norm_corr"].description = "Normalization correction"

    tab["flux_sum"] = sci_sum
    tab["flux_sum"].description = (
        f"Summed flux in [{y_range[0]}, {y_range[1]}]"
    )
    tab["profile_sum"] = prof_sum
    tab["var_sum"] = var_sum

    return tab


def obj_header(obj, i0=0, exposure_keys=[]):
    """
    Generate a header from a `msaexp.slit_combine.SlitGroup` object

    Parameters
    ----------
    obj : `msaexp.slit_combine.SlitGroup`
        Data group

    i0 : int
        Start of exposure counter

    Returns
    -------
    header : `astropy.io.fits.Header`
        Merged FITS header from the ``slits`` of ``obj``

    """

    header = pyfits.Header()
    with pyfits.open(obj.files[0]) as im:
        for ext in [0, "SCI"]:
            for k in im[ext].header:
                if k in ("", "COMMENT", "HISTORY"):
                    continue

                header[k] = (im[ext].header[k], im[ext].header.comments[k])

    header["EXPTIME"] = 0.0
    header["NCOMBINE"] = 0
    header["BUNIT"] = "microJansky"

    header["WAVECORR"] = (
        obj.meta["trace_with_xpos"],
        "Wavelength corrected for xpos",
    )

    header["YPIXTRA"] = (
        obj.source_ypixel_position,
        "Y pixel position in 2D cutouts",
    )
    header["YPIXSCL"] = (
        obj.slit_pixel_scale,
        "Cross dispersion pixel scale, arcsec",
    )
    header["CALCREF"] = (
        obj.calc_reference_exposure,
        "Derived reference exposure",
    )

    for k in obj.meta:
        if k == "bad_shutter_names":
            header["NBADSHUT"] = (
                len(obj.meta[k]),
                "Number of flagged bad shutters",
            )
            for i, ki in enumerate(obj.meta[k]):
                header[f"BADSHUT{i}"] = ki, "Bad shutter name"
        else:
            if k in obj.meta_comment:
                key = (obj.meta[k], obj.meta_comment[k])
            else:
                key = obj.meta[k]

            header[k.upper()] = key

    # Exposure time
    for i, sl in enumerate(obj.slits):
        # Shutter quadrant
        if i == 0:
            if sl.quadrant is not None:
                header["SHUTTRQ"] = (
                    sl.quadrant,
                    "MSA quadrant of first slitlet",
                )
                header["SHUTTRX"] = (
                    sl.xcen,
                    "MSA shutter row/xcen of first slitlet",
                )
                header["SHUTTRY"] = (
                    sl.ycen,
                    "MSA shutter col/ycen of first slitlet",
                )
            else:
                header["SHUTTRQ"] = (5, "Pseudo quadrant for fixed slit")
                # header['SHUTTRX'] = (0, 'MSA shutter row/xcen')
                # header['SHUTTRY'] = (0, 'MSA shutter col/ycen')

        header[f"SFILE{i+i0:03d}"] = os.path.basename(sl.meta.filename)
        header["NCOMBINE"] += 1

        fbase = sl.meta.filename.split("_nrs")[0]
        if fbase in exposure_keys:
            # print(f"xxx {fbase} in exposure_keys, don't add to exptime")
            continue
        else:
            # print(f"xxx {fbase} not in exposure_keys {len(exposure_keys)}")
            pass

        exposure_keys.append(fbase)
        header["EXPTIME"] += sl.meta.exposure.effective_exposure_time

    # trace fits
    if obj.fit is not None:
        for ie, exp in enumerate(obj.fit):
            try:
                sigma = obj.fit[exp]["sigma"]
                trace_coeffs = obj.fit[exp]["trace_coeffs"]
            except:
                sigma = 0.0
                trace_coeffs = [0]

            if ie == 0:
                header["SIGMA"] = (
                    sigma,
                    f"Profile width, pixels for group {exp}",
                )

            header[f"EGROUP{ie}"] = (exp, "Exposure group name")
            header[f"SIGMA{ie}"] = (
                sigma,
                f"Profile width, pixels for group {exp}",
            )
            header[f"OFFD{ie}"] = (
                len(trace_coeffs) - 1,
                "Trace offset polynomial degree",
            )
            for i, val in enumerate(trace_coeffs):
                header[f"OFFC{ie}_{i}"] = (
                    val,
                    f"Trace offset polynomial coefficient for group {exp}",
                )

    return header


DRIZZLE_KWS = dict(
    step=1,
    with_pathloss=True,
    wave_sample=1.05,
    ny=13,
    dkws=dict(oversample=16, pixfrac=0.8),
)


def combine_grating_group(
    xobj,
    grating_keys,
    drizzle_kws=DRIZZLE_KWS,
    include_full_pixtab=["PRISM"],
    **kwargs,
):
    """
    Make pseudo-drizzled outputs from a set of `msaexp.slit_combine.SlitGroup`
    objects

    Parameters
    ----------
    xobj : dict
        Set of `msaexp.slit_combine.SlitGroup` objects

    grating_keys : list
        List of keys of ``xobj`` to combine

    drizzle_kws : dict
        Keyword arguments passed to `msaexp.slit_combine.drizzle_grating_group`

    include_full_pixtab : list
        List of dispersers to full pixel table ``PIXTAB`` in output

    Returns
    -------
    hdul : `astropy.io.fits.HDUList`
        FITS HDU list generated from `msaexp.drizzle.make_optimal_extraction`

    """

    import astropy.units as u
    import grizli.utils
    import msaexp.drizzle

    _ = drizzle_grating_group(xobj, grating_keys, **drizzle_kws)
    (
        wave_bin,
        xbin,
        ybin,
        header,
        slit_info,
        pixtab,
        oned,
        arrays,
        barrays,
        parrays,
    ) = _

    num, vnum, den, ntot = arrays
    bnum = barrays[0]
    mnum, _, mden, _ = parrays

    sci2d = num / den
    bkg2d = bnum / den
    # wht2d = den * 1

    var2d = vnum / den**2
    wht2d = 1 / var2d

    pmask = mnum / den > 0
    snum = np.nansum(num * mnum / vnum * pmask, axis=0)
    bnum = np.nansum(bnum * mnum / vnum * pmask, axis=0)
    sden = np.nansum(mnum**2 / vnum * pmask, axis=0)

    smsk = nd.binary_erosion(np.isfinite(snum / sden), iterations=2) * 1.0
    smsk[smsk < 1] = np.nan
    snum *= smsk

    snmask = snum / np.sqrt(sden) > 3
    if snmask.sum() < 10:
        snmask = snum / np.sqrt(sden) > 1

    pdata = np.nansum((num / den) * snmask * den, axis=1)
    pdata /= np.nansum(snmask * den, axis=1)

    pmod = np.nansum(mnum / den * snum / sden * snmask * den, axis=1)
    pmod /= np.nansum(snmask * den, axis=1)

    kwargs = {}

    for k in xobj:
        bkg_offset = int(np.round(xobj[k]["obj"].meta["nod_offset"]))
        break

    _data = msaexp.drizzle.make_optimal_extraction(
        wave_bin,
        sci2d,
        wht2d,
        profile_slice=None,
        prf_center=0.0,
        prf_sigma=header["SIGMA"],
        sigma_bounds=(
            header["SIGMA"] - 0.1,
            header["SIGMA"] + 0.1,
        ),  # (0.5, 2.5),
        center_limit=0.001,
        fit_prf=False,
        fix_center=False,
        fix_sigma=True,
        trim=0,
        bkg_offset=bkg_offset,
        bkg_parity=[1, -1],
        offset_for_chi2=1.0,
        max_wht_percentile=None,
        max_med_wht_factor=10,
        verbose=VERBOSE_LOG,
        find_line_kws={},
        ap_radius=None,
        ap_center=None,
        **kwargs,
    )

    _sci2d, _wht2d, profile2d, spec, prof = _data

    spec["flux"] = snum / sden
    # spec["err"] = 1.0 / np.sqrt(sden)
    spec["err"] = np.sqrt(1.0 / sden)
    spec["flux"].unit = u.microJansky
    spec["err"].unit = u.microJansky
    spec["wave"].unit = u.micron
    spec["sky"] = bnum / sden
    spec["sky"].description = "Weighted sky spectrum"
    spec["sky"].unit = u.microJansky

    # Prefer optimal extraction without resampling
    if oned is not None:
        for c in oned.colnames:
            if c in spec.colnames:
                oned[c].unit = spec[c].unit
                oned[c].description = spec[c].description
                oned[c].format = spec[c].format

            spec[c] = oned[c]

        for k in oned.meta:
            spec.meta[k] = oned.meta[k]

    # Add path_corr column
    # average_path_loss(spec, header=header)

    for c in list(spec.colnames):
        if "aper" in c:
            spec.remove_column(c)

    # spec['flux'][~np.isfinite(smsk)] = 0
    # spec['err'][~np.isfinite(smsk)] = 0

    profile2d = mnum / den  # *snum/sden
    profile2d[~np.isfinite(profile2d)] = 0

    prof["profile"] = pdata
    prof["pfit"] = pmod

    for k in spec.meta:
        header[k] = spec.meta[k]
        pixtab.meta[k] = spec.meta[k]

    msg = "msaexp.drizzle.extract_from_hdul:  Output center = "
    msg += f" {header['PROFCEN']:6.2f}, sigma = {header['PROFSIG']:6.2f}"
    grizli.utils.log_comment(
        grizli.utils.LOGFILE, msg, verbose=VERBOSE_LOG, show_date=False
    )

    hdul = pyfits.HDUList()

    hdul.append(pyfits.BinTableHDU(data=spec, name="SPEC1D"))
    hdul.append(pyfits.ImageHDU(data=sci2d, header=header, name="SCI"))
    hdul.append(pyfits.ImageHDU(data=wht2d, header=header, name="WHT"))
    hdul.append(pyfits.ImageHDU(data=profile2d, header=header, name="PROFILE"))
    hdul.append(pyfits.BinTableHDU(data=prof, name="PROF1D"))
    if np.nanmax(bkg2d) != 0:
        hdul.append(
            pyfits.ImageHDU(data=bkg2d, header=header, name="BACKGROUND")
        )

    if (pixtab is not None) & (header["GRATING"] in include_full_pixtab):
        hdul.append(pyfits.BinTableHDU(data=pixtab, name="PIXTAB"))

    if slit_info is not None:
        hdul.append(pyfits.BinTableHDU(data=slit_info, name="SLITS"))

    for k in hdul["SCI"].header:
        if k not in hdul["SPEC1D"].header:
            hdul["SPEC1D"].header[k] = (
                hdul["SCI"].header[k],
                hdul["SCI"].header.comments[k],
            )

    return hdul


def drizzle_grating_group(
    xobj,
    grating_keys,
    step=1,
    with_pathloss=True,
    wave_sample=1.05,
    grating_limits=msautils.GRATING_LIMITS,
    pixtab_with_mask=True,
    ny=13,
    dkws=dict(oversample=16, pixfrac=0.8, sample_axes="y"),
    y_range=[-3, 3],
    **kwargs,
):
    """
    Run the pseudo-drizzled sampling from a set of
    `msaexp.slit_combine.SlitGroup` objects

    Parameters
    ----------
    xobj : dict
        Set of `msaexp.slit_combine.SlitGroup` objects

    grating_keys : list
        List of keys of ``xobj`` to combine

    step : float
        Cross dispersion step size in units of original pixels

    with_pathloss : bool
        Compute a pathloss correction for each spectrum based on the fitted
        profile width and the (planned) intra-shutter position of a particular
        source

    wave_sample : float
        Wavelength sampling relative to the default grid for a particular
        grating provided by `msaexp.utils.get_standard_wavelength_grid`

    ny : int
        Half-width in pixels of the output grid in the cross-dispersion axis

    dkws : dict
        keyword args passed to `msaexp.slit_combine.pseudo_drizzle`


    Returns
    -------
    wave_bin : array-like
        1D wavelength sample points

    xbin : array-like
        2D x (wavelength) sample points

    ybin : array-like
        2D cross-dispersion sample points

    header : `~astropy.io.fits.Header`
        Header for the combination

    slit_info : `~astropy.table.Table`
        Table of slit metadata

    arrays : (array-like, array-like, array-like)
        2D ``sci_num``, ``var_num`` and ``wht_denom`` arrays of the resampled data. The
        weighted data is ``sci_num / wht_denom``

    parrays : (array-like, array-like, array-like)
        2D `prof2d`, `var2d` and `prof_wht2d` arrays of the profile model resampled
        like the data

    """

    import astropy.io.fits as pyfits
    import astropy.table

    obj = xobj[grating_keys[0]]["obj"]

    header = pyfits.Header()

    wave_bin = msautils.get_standard_wavelength_grid(
        obj.grating,
        sample=wave_sample,
        grating_limits=grating_limits,
    )

    xbin = msautils.array_to_bin_edges(wave_bin)

    ybin = np.arange(-ny, ny + step * 1.01, step) - 0.5

    arrays = None
    barrays = None
    parrays = None

    header = None
    slit_info = []
    tabs = []

    exposure_keys = []
    for k in grating_keys:
        obj = xobj[k]["obj"]
        if header is None:
            header = obj_header(obj, i0=0, exposure_keys=exposure_keys)
        else:
            hi = obj_header(
                obj, i0=header["NCOMBINE"], exposure_keys=exposure_keys
            )
            header["NCOMBINE"] += hi["NCOMBINE"]
            header["EXPTIME"] += hi["EXPTIME"]
            for k in hi:
                if k not in header:
                    # print("new keyword: ", k)
                    header[k] = hi[k]

        tab = obj.get_flat_diff_arrays(
            apply_mask=pixtab_with_mask,
            with_pathloss=with_pathloss,
            # float_dtype=np.float32,
        )
        tabs.append(tab)

        _meta = obj.slit_metadata()
        _meta_nlines = len(_meta)
        for c in list(_meta.colnames):
            if np.isin(_meta[c], [None]).sum() == _meta_nlines:
                _meta.remove_column(c)

        slit_info.append(_meta)

    # Merge slit info tables
    if len(slit_info) > 0:
        slit_info = astropy.table.vstack(slit_info)
    else:
        slit_info = None

    # Do the resampling
    if len(tabs) > 0:
        istart = 0
        for t in tabs:
            t["exposure_index"] += istart
            istart = t["exposure_index"].max() + 1

        pixtab = astropy.table.vstack(tabs)

        for t in tabs:
            for k in t.meta:
                header[k] = t.meta[k]
                pixtab.meta[k] = t.meta[k]

        ############
        # wht = 1.0 / vdiff
        if obj.meta["weight_type"].lower() == "poisson":
            wht = 1.0 / pixtab["var_total"]
        elif obj.meta["weight_type"].lower() == "mask":
            wht = (pixtab["var_total"] > 0.0) * 1.0
        elif obj.meta["weight_type"].lower() == "exptime":
            wht = (pixtab["var_total"] > 0.0) * pixtab["exptime"]
        elif obj.meta["weight_type"].lower() == "exptime_bar":
            wht = (
                (pixtab["var_total"] > 0.0)
                * pixtab["exptime"]
                * pixtab["bar"] ** 2
            )
        elif obj.meta["weight_type"].lower() == "ivm":
            wht = 1.0 / pixtab["var_rnoise"]
        else:
            msg = "weight_type {weight_type} not recognized".format(**obj.meta)
            raise ValueError(msg)

        ok = np.isfinite(
            pixtab["sci"] + pixtab["var_total"] + wht + pixtab["yslit"]
        )
        ok &= (pixtab["var_total"] > 0) & (wht > 0)

        if "mask" in pixtab.colnames:
            ok &= pixtab["mask"]

        ysl = pixtab["yslit"][ok]
        xsl = pixtab["wave"][ok]

        arrays = pseudo_drizzle(
            xsl,
            ysl,
            pixtab["sci"][ok] / pixtab["pathloss"][ok],
            pixtab["var_total"][ok] / pixtab["pathloss"][ok] ** 2,
            wht[ok],
            xbin,
            ybin,
            arrays=arrays,
            **dkws,
        )

        barrays = pseudo_drizzle(
            xsl,
            ysl,
            pixtab["sky"][ok] / pixtab["pathloss"][ok],
            pixtab["var_total"][ok] / pixtab["pathloss"][ok] ** 2,
            wht[ok],
            xbin,
            ybin,
            arrays=barrays,
            **dkws,
        )

        parrays = pseudo_drizzle(
            xsl,
            ysl,
            pixtab["profile"][ok],
            pixtab["var_total"][ok] / pixtab["pathloss"][ok] ** 2,
            wht[ok],
            xbin,
            ybin,
            arrays=parrays,
            **dkws,
        )

        # 1D extraction
        oned = pixel_table_to_1d(pixtab, wave_bin, weight=wht, y_range=y_range)
    else:
        oned = None
        pixtab = None

    return (
        wave_bin,
        xbin,
        ybin,
        header,
        slit_info,
        pixtab,
        oned,
        arrays,
        barrays,
        parrays,
    )


def extract_from_pixtab(
    pixtab,
    step=1,
    grating="PRISM",
    wave_sample=1.05,
    ny=13,
    dkws=dict(oversample=16, pixfrac=0.8, sample_axes="y"),
    y_range=[-3, 3],
    weight_type="ivm",
    grating_limits=msautils.GRATING_LIMITS,
    **kwargs,
):
    """
    Run the pseudo-drizzled sampling from a set of
    `msaexp.slit_combine.SlitGroup` objects

    Parameters
    ----------
    pixtab : table
        Full pixel table

    step : float
        Cross dispersion step size in units of original pixels

    with_pathloss : bool
        Compute a pathloss correction for each spectrum based on the fitted
        profile width and the (planned) intra-shutter position of a particular
        source

    wave_sample : float
        Wavelength sampling relative to the default grid for a particular
        grating provided by `msaexp.utils.get_standard_wavelength_grid`

    ny : int
        Half-width in pixels of the output grid in the cross-dispersion axis

    dkws : dict
        keyword args passed to `msaexp.slit_combine.pseudo_drizzle`


    Returns
    -------
    wave_bin : array-like
        1D wavelength sample points

    xbin : array-like
        2D x (wavelength) sample points

    ybin : array-like
        2D cross-dispersion sample points

    header : `~astropy.io.fits.Header`
        Header for the combination

    slit_info : `~astropy.table.Table`
        Table of slit metadata

    arrays : (array-like, array-like, array-like)
        2D ``sci_num``, ``var_num`` and ``wht_denom`` arrays of the resampled data. The
        weighted data is ``sci_num / wht_denom``

    parrays : (array-like, array-like, array-like)
        2D `prof2d`, `var2d` and `prof_wht2d` arrays of the profile model resampled
        like the data

    """

    import astropy.io.fits as pyfits
    import astropy.table

    wave_bin = msautils.get_standard_wavelength_grid(
        grating, sample=wave_sample, grating_limits=grating_limits
    )

    xbin = msautils.array_to_bin_edges(wave_bin)

    ybin = np.arange(-ny, ny + step * 1.01, step) - 0.5

    arrays = None
    barrays = None
    parrays = None

    ############
    # wht = 1.0 / vdiff
    if weight_type == "poisson":
        wht = 1.0 / pixtab["var_total"]
    elif weight_type == "mask":
        wht = (pixtab["var_total"] > 0.0) * 1.0
    elif weight_type == "exptime":
        wht = (pixtab["var_total"] > 0.0) * pixtab["exptime"]
    elif weight_type == "exptime_bar":
        wht = (
            (pixtab["var_total"] > 0.0)
            * pixtab["exptime"]
            * pixtab["bar"] ** 2
        )
    elif weight_type == "ivm":
        wht = 1.0 / pixtab["var_rnoise"]
    else:
        msg = f"weight_type {weight_type} not recognized"
        raise ValueError(msg)

    ok = np.isfinite(
        pixtab["sci"] + pixtab["var_total"] + wht + pixtab["yslit"]
    )
    ok &= (pixtab["var_total"] > 0) & (wht > 0)

    ysl = pixtab["yslit"][ok]
    xsl = pixtab["wave"][ok]

    arrays = pseudo_drizzle(
        xsl,
        ysl,
        pixtab["sci"][ok] / pixtab["pathloss"][ok],
        pixtab["var_total"][ok] / pixtab["pathloss"][ok] ** 2,
        wht[ok],
        xbin,
        ybin,
        arrays=arrays,
        **dkws,
    )

    barrays = pseudo_drizzle(
        xsl,
        ysl,
        pixtab["sky"][ok] / pixtab["pathloss"][ok],
        pixtab["var_total"][ok] / pixtab["pathloss"][ok] ** 2,
        wht[ok],
        xbin,
        ybin,
        arrays=barrays,
        **dkws,
    )

    parrays = pseudo_drizzle(
        xsl,
        ysl,
        pixtab["profile"][ok],
        pixtab["var_total"][ok] / pixtab["pathloss"][ok] ** 2,
        wht[ok],
        xbin,
        ybin,
        arrays=parrays,
        **dkws,
    )

    # 1D extraction
    oned = pixel_table_to_1d(pixtab, wave_bin, weight=wht, y_range=y_range)

    return (
        wave_bin,
        xbin,
        ybin,
        oned,
        arrays,
        barrays,
        parrays,
    )


FIT_PARAMS_SN_KWARGS = dict(
    sn_percentile=80,
    sigma_threshold=5,
    # degree_sn=[[-10000,10,100000], [0,1,2]],
    degree_sn=[[-10000], [0]],
    verbose=True,
)


def average_path_loss(spec, header=None):
    """
    Get average pathloss correction from spectrum metadata

    Parameters
    ----------
    spec : Table
        Table with metadata including pathloss parameters used above

    header : `~astropy.io.fits.Header`
        Optional FITS header to use instead of ``spec.meta``

    Returns
    -------
    path_corr : Column
        Adds a ``path_corr`` column to ``spec`` that represents the average
        path-loss correction determined from the profile width and x_pos
        centering parameters using `msaexp.slit_combine.slit_prf_fraction`

    """
    if header is None:
        header = spec.meta

    if "WITHPATH" not in header:
        print("WITHPATH keyword not found")
        return False

    if not header["WITHPATH"]:
        print("WITHPATH = False")
        return False

    prf_list = []
    for i in range(100):
        if f"PTHSIG{i}" in header:
            sigma = header[f"PTHSIG{i}"]
            x_pos = header[f"PTHXPO{i}"]
        elif f"SIGMA{i}" in header:
            sigma = header[f"SIGMA{i}"]
            x_pos = header[f"XPOS{i}"]
        else:
            sigma = x_pos = None

        if sigma is not None:
            prf_i = slit_prf_fraction(
                spec["wave"].astype(float),
                sigma=sigma,
                x_pos=x_pos,
                slit_width=0.2,
                pixel_scale=PIX_SCALE,
                verbose=False,
            )

            # Path loss is relative to a centered point source
            prf_0 = slit_prf_fraction(
                spec["wave"].astype(float),
                sigma=0.01,
                x_pos=0.0,
                slit_width=0.2,
                pixel_scale=PIX_SCALE,
                verbose=False,
            )

            prf_list.append(prf_i / prf_0)

    if len(prf_list) > 0:
        print("Added path_corr column to spec")
        spec["path_corr"] = 1.0 / np.nanmean(np.array(prf_list), axis=0)
        spec["path_corr"].format = ".2f"
        spec["path_corr"].description = (
            "Average path loss correction already applied"
        )


def get_spectrum_path_loss(spec):
    """
    Calculate the path loss correction that was applied to a spectrum

    Parameters
    ----------
    spec : Table
        Table with metadata including ``SRCXPOS`` (intra-shutter position,
        arcsec) and ``SIGMA`` (source gaussian width, pixels) keywords

    Returns
    -------
    path_corr : array-like
        Wavelength-dependent path loss correction

    """
    path_loss = slit_prf_fraction(
        spec["wave"].astype(float),
        sigma=spec.meta["SIGMA"],
        x_pos=spec.meta["SRCXPOS"],
        slit_width=0.2,
        pixel_scale=PIX_SCALE,
        verbose=False,
    )

    path_loss_ref = slit_prf_fraction(
        spec["wave"].astype(float),
        sigma=0.01,
        x_pos=0.0,
        slit_width=0.2,
        pixel_scale=PIX_SCALE,
        verbose=False,
    )

    path_loss /= path_loss_ref
    return 1.0 / path_loss


def set_lookup_prf(
    psf_file=None,
    slit_file=None,
    grating="PRISM",
    filter="CLEAR",
    fixed_slit="S200A1",
    version="001",
    lookup_prf_type="merged",
    force_m_gratings=True,
    prism_merged=False,
    **kwargs,
):
    """
    Set lookup table PRF file appropriate for a given grating / filter

    Parameters
    ----------
    psf_file : str, None
        Force filename of the lookup table file

    slit_file : str
        Filename of a slitlet extraction

    grating : str
        Grating name

    filter : str
        Filter name

    fixed_slit : "S200A1", "S1600A1"
        Slit type.  Will use "S200A1" for MSA observations

    version : str
        Version string

    lookup_prf_type : str
        - ``"merged"``: use the grating-merged file
          ``nirspec_merged_{fixed_slit}_exp_psf_lookup_{version}.fits``
        - ``"by_grating"``: use the grating-specific files
          ``nirspec_{grating}_{filter}_{fixed_slit}_exp_psf_lookup_{version}.fits``

    force_m_gratings : bool
        Use "M" versions of files for "H" gratings

    prism_merged : bool
        Always use "merged" for PRISM, e.g., with ``lookup_prf_type="by_grating"``.

    Returns
    -------
    prf : `msaexp.utils.LookupTablePSF`
        PSF object, and updates global ``msaexp.slit_combine.LOOKUP_PRF``` object
    """
    global LOOKUP_PRF

    if slit_file is not None:
        with pyfits.open(slit_file) as im:
            grating = im[0].header["GRATING"]
            filter = im[0].header["FILTER"]
            if im[0].header["EXP_TYPE"] == "NRS_FIXEDSLIT":
                fixed_slit = im[0].header["APERNAME"].split("_")[1].lower()
            else:
                fixed_slit = "s200a1"

    if (grating.upper() == "PRISM") & (prism_merged):
        lookup_prf_type = "merged"

    if lookup_prf_type == "merged":
        key = "merged"
        if fixed_slit in ["s200a2", "s200b1"]:
            fixed_slit = "s200a1"
    else:
        key = f"{grating}_{filter}".lower()

    if psf_file is None:
        psf_file = (
            f"nirspec_{key}_{fixed_slit}_exp_psf_lookup_{version}.fits".lower()
        )

        # One H grating doesn't exist so use M
        # if 'g140h_f070lp' in psf_file:
        #     msg = f"msaexp.slit_combine.set_lookup_prf: g140h_f070lp > g140m_f070lp"
        #     utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
        #     psf_file = psf_file.replace('g140h_f070lp', 'g140m_f070lp')

        if force_m_gratings:
            if "h_f" in psf_file:
                msg = (
                    f"msaexp.slit_combine.set_lookup_prf: force medium grating"
                )
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
                psf_file = psf_file.replace("h_f", "m_f")

    prf = msautils.LookupTablePSF(psf_file=psf_file, **kwargs)

    LOOKUP_PRF = prf

    return prf


def extract_spectra(
    target="1208_5110240",
    root="nirspec",
    path_to_files="./",
    files=None,
    do_gratings=["PRISM", "G395H", "G395M", "G235M", "G140M"],
    join=[0, 3, 5],
    exposure_groups=None,
    split_uncover=True,
    stuck_threshold=0.3,
    valid_frac_threshold=0.1,
    pad_border=2,
    sort_by_sn=False,
    position_key="y_index",
    mask_cross_dispersion=None,
    cross_dispersion_mask_type="trace",
    trace_from_yoffset=False,
    reference_exposure="auto",
    trace_niter=4,
    offset_degree=0,
    degree_kwargs={},
    recenter_all=False,
    free_trace_offset=False,
    nod_offset=None,
    initial_sigma=7,
    fit_type=1,
    initial_theta=None,
    fix_params=False,
    input_fix_sigma=None,
    fit_params_kwargs=None,
    diffs=True,
    undo_pathloss=True,
    undo_barshadow=False,
    sky_arrays=None,
    use_first_sky=False,
    drizzle_kws=DRIZZLE_KWS,
    get_xobj=False,
    trace_with_xpos=False,
    trace_with_ypos="auto",
    get_background=False,
    make_2d_plots=True,
    lookup_prf_type="merged",
    lookup_prf_version="001",
    include_full_pixtab=["PRISM"],
    plot_kws={},
    protect_exception=True,
    **kwargs,
):
    """
    Spectral combination workflow splitting by grating and multiple observations in a particular grating

    Parameters
    ----------
    target : str
        Target name. If no ``files`` specified, will search for the 2D slit
        cutout files with names like ``*phot*{target}.fits``

    root : str
        Output file rootname

    path_to_files : str
        Directory path containing the ``phot`` files

    files : list, None
        Optional explicit list of ``phot`` files to combine

    do_gratings : list
        Gratings to consider

    join : list
        Indices of ``files[i].split('[._]') + GRATING`` to join as a group

    split_uncover : bool
        Split sub-pixel dithers from UNCOVER (GO-2561) when defining exposure groups

    sort_by_sn : bool
        Try to process groups in order of decreasing S/N, i.e., to derive the
        trace offsets in the prism where it will be best defined and propagate
        to other groups with the gratings

    mask_cross_dispersion : None or [int, int]
        Optional cross-dispersion masking, e.g., for stuck-closed shutters or
        multiple sources within a slitlet. The specified values are integer
        indices of the pixel range to mask. See ``cross_dispersion_mask_type``.

    cross_dispersion_mask_type : str
        Type of cross dispersion mask to apply. With ``'trace'``, the masked
        pixels are calculated relative to the (expected) center of the trace,
        and, e.g., ``mask_cross_dispersion = [5,100]`` will mask all pixels 5
        pixels "above" the center of the trace (100 is an arbitrarily large
        number to include all pixels). The mask will shift along with the nod
        offsets.

        With ``fixed``, the mask indices are relative to the trace *in the
        first exposure* and won't shift with the nod offsets. So
        ``mask_cross_dispersion = [-3,3]`` would mask roughly the central
        shutter in all exposures that will contain the source in some
        exposures and not in others. This can be used to try to mitigate some
        stuck-closed shutters, though the how effective it is is still under
        investigation.

    stuck_threshold, pad_border :
        See `~msaexp.slit_combine.SlitGroup`

    trace_from_yoffset, trace_with_xpos, trace_with_ypos :
        See `~msaexp.slit_combine.SlitGroup`

    position_key, reference_exposure, nod_offset, diffs :
        See `~msaexp.slit_combine.SlitGroup`

    undo_pathloss, undo_barshadow :
        See `~msaexp.slit_combine.SlitGroup`

    trace_niter : int, optional
        Number of iterations for the trace fit

    offset_degree : int, optional
        Polynomial offset degree

    degree_kwargs : dict, optional
        Degree keyword arguments

    recenter_all : bool, optional
        Refit for the trace center for all groups.  If False,
        use the center from the first (usually highest S/N prism)
        trace.

    free_trace_offset : bool
        If True, recenter **all** traces within and across groups.

    initial_sigma : float, optional
        Initial sigma value.  This is 10 times the Gaussian sigma
        width, in pixels.

    fit_type : int, optional
        Fit type value

    initial_theta : None, optional
        Initial parameter guesses

    fix_params : bool, optional
        Fix parameters to ``initial_theta``

    input_fix_sigma : None, optional
        Input fix sigma value

    fit_params_kwargs : None, optional
        Fit parameters keyword arguments

    drizzle_kws : dict, optional
        Drizzle keyword arguments

    get_xobj : bool, optional
        Return `~msaexp.slit_combine.SlitGroup` objects along with the
        HDU product

    get_background : bool, optional
        Get background value

    make_2d_plots : bool, optional
        Make 2D plots

    include_full_pixtab : list
        List of dispersers to full pixel table in output

    Returns
    -------
    None : null
      If no valid spectra are found
    hdu : dict
      Dict of `~astropy.io.fits.HDUList` objects for the separate gratings
    xobj : dict
      Dictionary of `~msaexp.slit_combine.SlitGroup` objects if ``get_xobj=True``
    """

    global CENTER_WIDTH, CENTER_PRIOR, SIGMA_PRIOR, MSA_NOD_ARCSEC
    frame = inspect.currentframe()

    # Log function arguments
    utils.LOGFILE = f"{root}_{target}.extract.log"
    args = utils.log_function_arguments(
        utils.LOGFILE,
        frame,
        "slit_combine.extract_spectra",
        ignore=["sky_arrays"],
    )
    if isinstance(args, dict):
        with open(f"{root}_{target}.extract.yml", "w") as fp:
            fp.write(f"# {time.ctime()}\n# {os.getcwd()}\n")
            yaml.dump(args, stream=fp, Dumper=yaml.Dumper)

    if files is None:
        files = glob.glob(os.path.join(path_to_files, f"*phot*{target}.fits"))

    for i in range(len(files))[::-1]:
        if "jw04246003001_03101_00001_nrs2" in files[i]:
            utils.log_comment(
                utils.LOGFILE, f"Exclude {files[i]}", verbose=VERBOSE_LOG
            )
            files.pop(i)
        elif (target == "1210_9849") & ("jw01210001001" in files[i]):
            utils.log_comment(
                utils.LOGFILE, f"Exclude {files[i]}", verbose=VERBOSE_LOG
            )
            files.pop(i)

    files.sort()

    utils.log_comment(
        utils.LOGFILE,
        f"{root}   target: {target}   Files: {len(files)}",
        verbose=VERBOSE_LOG,
    )

    if exposure_groups is None:
        groups = split_visit_groups(
            files, join=join, gratings=do_gratings, split_uncover=split_uncover
        )
    else:
        groups = exposure_groups

    xobj = {}
    for ig, g in enumerate(groups):
        if "xxxprism" in g:
            continue

        if ig == -100:
            continue

        # if "jw02561002001" in g:
        #     continue

        msg = f"\n* Group {g}   "
        msg += f"N={len(groups[g])}\n"
        msg += "=================================="
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        if nod_offset is None:
            if ("glazebrook" in root) | ("suspense" in root):
                # nod_offset = 10
                MSA_NOD_ARCSEC = 0.529 * 2
            else:
                MSA_NOD_ARCSEC = 0.529 * 1

        if trace_with_ypos in ["auto"]:
            trace_with_ypos = ("b" not in target) & (not get_background)
            trace_with_ypos &= "BKG" not in target

        if root.startswith("glazebrook-v"):
            utils.log_comment(
                utils.LOGFILE, "  ! Auto glazebrook", verbose=VERBOSE_LOG
            )
            trace_from_yoffset = True

        elif "maseda" in root:
            utils.log_comment(
                utils.LOGFILE, "  ! Auto maseda", verbose=VERBOSE_LOG
            )
            trace_from_yoffset = True

        elif "smacs0723-ero-v" in root:
            utils.log_comment(
                utils.LOGFILE, "  ! Auto SMACS0723", verbose=VERBOSE_LOG
            )
            trace_from_yoffset = True

        if lookup_prf_type is not None:
            prf = set_lookup_prf(
                slit_file=groups[g][0],
                version=lookup_prf_version,
                lookup_prf_type=lookup_prf_type,
            )
        else:
            prf = None

        if protect_exception:
            try:
                obj = SlitGroup(
                    groups[g],
                    g,
                    position_key=position_key,
                    diffs=diffs,  # (True & (~isinstance(id, str))),
                    stuck_threshold=stuck_threshold,
                    undo_barshadow=undo_barshadow,
                    undo_pathloss=undo_pathloss,
                    sky_arrays=sky_arrays,
                    trace_with_xpos=trace_with_xpos,
                    trace_with_ypos=trace_with_ypos,
                    trace_from_yoffset=trace_from_yoffset,
                    nod_offset=nod_offset,
                    reference_exposure=reference_exposure,
                    pad_border=pad_border,
                    lookup_prf=prf,
                    **kwargs,
                )
            except RuntimeError:
                msg = f"\n    failed RuntimeError\n"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
                continue

            except TypeError:
                msg = f"\n    failed TypeError\n"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
                continue

            except ValueError:
                msg = f"\n    failed ValueError\n"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
                continue
        else:
            obj = SlitGroup(
                groups[g],
                g,
                position_key=position_key,
                diffs=diffs,  # (True & (~isinstance(id, str))),
                stuck_threshold=stuck_threshold,
                undo_barshadow=undo_barshadow,
                undo_pathloss=undo_pathloss,
                sky_arrays=sky_arrays,
                trace_with_xpos=trace_with_xpos,
                trace_with_ypos=trace_with_ypos,
                trace_from_yoffset=trace_from_yoffset,
                nod_offset=nod_offset,
                reference_exposure=reference_exposure,
                pad_border=pad_border,
                lookup_prf=prf,
                **kwargs,
            )

        if 0:
            if (obj.grating not in do_gratings) | (
                obj.sh[1] < 83 * 2 ** (obj.grating not in ["PRISM"])
            ):
                msg = f"\n    skip shape=({obj.sh}) {obj.grating}\n"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
                continue

        if obj.mask.sum() < 256:
            msg = f"\n    skip npix={obj.mask.sum()} shape={obj.sh} "
            msg += (
                f" ({np.nanmin(obj.wave):.2f}, {np.nanmax(obj.wave):.2f} um)"
            )
            msg += f" {obj.grating}\n"

            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)
            continue

        if obj.meta["diffs"]:
            valid_frac = obj.mask.sum() / obj.mask.size

            if obj.N == 1:
                utils.log_comment(
                    utils.LOGFILE,
                    f"\n    skip N=1 {obj.grating}\n",
                    verbose=VERBOSE_LOG,
                )
                continue

            elif len(obj.meta["bad_shutter_names"]) == obj.N:
                utils.log_comment(
                    utils.LOGFILE,
                    f"\n    skip all bad {obj.grating}\n",
                    verbose=VERBOSE_LOG,
                )
                continue

            elif (len(obj.unp.values) == 1) & (obj.meta["diffs"]):
                utils.log_comment(
                    utils.LOGFILE,
                    f"\n    one position {obj.grating}\n",
                    verbose=VERBOSE_LOG,
                )
                continue

            # elif os.path.basename(obj.files[0]).startswith("jw02561002001"):
            #     utils.log_comment(
            #         utils.LOGFILE,
            #         f"\n    uncover {obj.files[0]}\n",
            #         verbose=VERBOSE_LOG,
            #     )
            #     continue

            elif valid_frac < valid_frac_threshold:
                utils.log_comment(
                    utils.LOGFILE,
                    f"\n    valid pixels {valid_frac:.2f} < {valid_frac_threshold}\n",
                    verbose=VERBOSE_LOG,
                )
                continue

            elif (("b" in target) | ("BKG" in target)) & (
                (obj.info["shutter_state"] == "x").sum() > 0
            ):
                utils.log_comment(
                    utils.LOGFILE,
                    "\n    single background shutter\n",
                    verbose=VERBOSE_LOG,
                )
                continue

        ind = None

        if ind is not None:
            obj.xslit = obj.xslit[ind, :]
            obj.yslit = obj.yslit[ind, :]
            obj.ypix = obj.ypix[ind, :]

            obj.xtr = obj.xtr[ind, :]
            obj.ytr = obj.ytr[ind, :]
            obj.wtr = obj.wtr[ind, :]

            obj.wave = obj.wave[ind, :]
            obj.bar = obj.bar[ind, :]

            obj.base_coeffs = [obj.base_coeffs[j] for j in ind]

        xobj[g] = {"obj": obj}

        # if not obj.trace_with_ypos:
        #     CENTER_WIDTH = 2

    if len(xobj) == 0:
        utils.log_comment(
            utils.LOGFILE, "No valid spectra", verbose=VERBOSE_LOG
        )
        return None

    if ("macs0417" in root) & (target == "1208_234"):
        for k in xobj:
            obj = xobj[k]["obj"]
            for j in range(obj.N):
                obj.sci[j, (obj.yslit[j, :] < -8)] = np.nan

            obj.mask &= np.isfinite(obj.sci)

    elif target == "4233_945401":
        for k in xobj:
            obj = xobj[k]["obj"]
            for j in range(obj.N):
                obj.sci[j, (obj.yslit[j, :] > 6)] = np.nan

            obj.mask &= np.isfinite(obj.sci)

    if mask_cross_dispersion is not None:
        msg = f" {'slit_combine':<28}: mask_cross_dispersion {mask_cross_dispersion}"
        msg += f"  cross_dispersion_mask_type={cross_dispersion_mask_type}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        for k in xobj:
            obj = xobj[k]["obj"]
            for j in range(obj.N):
                cross_mask = obj.yslit[j, :] > mask_cross_dispersion[0]
                cross_mask &= obj.yslit[j, :] < mask_cross_dispersion[1]

                if cross_dispersion_mask_type == "bkg":
                    # Just mask background when doing the differences
                    obj.bkg_mask[j, cross_mask] = False

                elif cross_dispersion_mask_type == "fixed":
                    # Relative to first trace
                    cross_mask = obj.yshutter[j, :] > mask_cross_dispersion[0]
                    cross_mask &= obj.yshutter[j, :] < mask_cross_dispersion[1]
                    obj.sci[j, cross_mask] = np.nan

                else:
                    # Mask out all pixels, source and background
                    obj.sci[j, cross_mask] = np.nan

            obj.mask &= np.isfinite(obj.sci)
            obj.bkg_mask &= np.isfinite(obj.sci)

    # Sort by grating
    okeys = []
    xkeys = []
    for k in list(xobj.keys()):
        obj = xobj[k]["obj"]
        okeys.append(f"{k.split('-')[-1]}-{obj.sh[1]}")
        xkeys.append(k)

    if initial_theta is not None:
        if len(initial_theta) > 1:
            fit_params_kwargs["theta"] = initial_theta

    if sort_by_sn:
        # Sort in order of decreasing S/N
        sn_keys = []

        for k in xkeys:
            obj = xobj[k]["obj"]

            _sn, sn_val, _, _ = obj.fit_params_by_sn(**fit_params_kwargs)
            if "prism" in k:
                sn_val *= 2

            if not np.isfinite(sn_val):
                sn_keys.append(-1)
            else:
                sn_keys.append(sn_val)

        so = np.argsort(sn_keys)[::-1]

    else:
        # Sort by the keys favoring largest arrays in the reddest gratings
        so = np.argsort(okeys)[::-1]

    keys = [xkeys[j] for j in so]

    utils.log_comment(utils.LOGFILE, f"\nkeys: {keys}", verbose=VERBOSE_LOG)

    if fit_params_kwargs is not None:
        obj0 = xobj[keys[0]]["obj"]
        _ = obj0.fit_params_by_sn(**fit_params_kwargs)
        _sn, sn_val, do_fix_sigma, offset_degree = _

        if do_fix_sigma:
            # input_fix_sigma = initial_sigma*1
            recenter_all = False
            fix_params = True
            if initial_theta is None:
                initial_theta = np.array([initial_sigma, 0.0])

    if initial_theta is not None:
        CENTER_PRIOR = initial_theta[-1]
        SIGMA_PRIOR = initial_theta[0] / 10.0
    else:
        CENTER_PRIOR = 0.0
        SIGMA_PRIOR = 0.6

    # fix_sigma = None
    if input_fix_sigma is None:
        fix_sigma_across_groups = True
        fix_sigma = -1
    else:
        if input_fix_sigma < 0:
            fix_sigma_across_groups = False
            fix_sigma = -1
        else:
            fix_sigma_across_groups = True
            fix_sigma = input_fix_sigma

    for i, k in enumerate(keys):
        msg = f"\n##### Group #{i+1} / {len(xobj)}: {k} ####\n"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE_LOG)

        obj = xobj[k]["obj"]

        if lookup_prf_type is not None:
            set_lookup_prf(
                slit_file=obj.files[0],
                version=lookup_prf_version,
                lookup_prf_type=lookup_prf_type,
            )

        if free_trace_offset:
            ref_exp = None
        else:
            ref_exp = obj.calc_reference_exposure

        kws = dict(
            niter=trace_niter,
            force_positive=(fit_type == 0),
            degree=offset_degree,
            ref_exp=ref_exp,
            sigma_bounds=(3, 12),
            with_bounds=False,
            trace_bounds=(-1.0, 1.0),
            initial_sigma=initial_sigma,
            x0=initial_theta,
            # evaluate=fix_params,
            method="powell",
            tol=1.0e-8,
        )

        if fix_params:
            kws["evaluate"] = True

        if (i == 0) | (recenter_all):

            if fix_sigma > 0:
                kws["fix_sigma"] = fix_sigma

            tfit = obj.fit_all_traces(**kws)

            theta = tfit[obj.unp.values[0]]["theta"]

            if i == 0:
                if "fix_sigma" in kws:
                    if kws["fix_sigma"] > 0:
                        fix_sigma = kws["fix_sigma"]
                    else:
                        fix_sigma = tfit[obj.unp.values[0]]["sigma"] * 10

                elif fix_sigma_across_groups:
                    theta = theta[1:]
                    fix_sigma = tfit[obj.unp.values[0]]["sigma"] * 10

        else:
            kws["x0"] = theta
            kws["with_bounds"] = False
            kws["evaluate"] = True
            kws["fix_sigma"] = fix_sigma

            tfit = obj.fit_all_traces(**kws)

        xobj[k] = {"obj": obj, "fit": tfit}

    ######
    # List of gratings
    gratings = {}
    max_size = {}

    for k in keys:
        # gr = k.split("-")[-1]
        gr = "-".join(k.split("-")[-2:])
        if gr in gratings:
            gratings[gr].append(k)
            max_size[gr] = np.maximum(max_size[gr], xobj[k]["obj"].sh[1])
        else:
            gratings[gr] = [k]
            max_size[gr] = xobj[k]["obj"].sh[1]

    ######
    # Fit plots
    if make_2d_plots:
        for k in keys:
            obj = xobj[k]["obj"]
            # gr = k.split("-")[-1]
            gr = "-".join(k.split("-")[-2:])

            if (obj.sh[1] < max_size[gr]) & (make_2d_plots > 1):
                continue

            if "fit" in xobj[k]:
                fit = xobj[k]["fit"]
            else:
                fit = None

            fig2d = obj.plot_2d_differences(fit=fit)
            fileroot = f"{root}_{obj.grating}-{obj.filter}_{target}".lower()
            fig2d.savefig(f"{fileroot}.d2d.png")

    # for k in xobj:
    #     xobj[k]["obj"].sky_arrays = None

    utils.log_comment(
        utils.LOGFILE, f"\ngratings: {gratings}", verbose=VERBOSE_LOG
    )

    hdul = {}

    if diffs:
        if len(plot_kws) == 0:
            plot_kws = {}
    else:
        if len(plot_kws) == 0:
            plot_kws = {
                "vmin": -0.1,
                #'ny': 7,
                #'ymax_sigma_scale':5, 'ymax_percentile': 90, 'ymax_scale': 1.5,
            }

    for g in gratings:
        hdul[g] = combine_grating_group(
            xobj,
            gratings[g],
            drizzle_kws=drizzle_kws,
            include_full_pixtab=include_full_pixtab,
        )

        _head = hdul[g][1].header

        specfile = f"{root}_{_head['GRATING']}-{_head['FILTER']}".lower()
        specfile += f"_{_head['SRCNAME']}.spec.fits".lower()

        if True:
            specfile = specfile.replace("background_", "b")

        utils.log_comment(utils.LOGFILE, specfile, verbose=VERBOSE_LOG)
        if "PIXTAB" in hdul[g]:
            ptab = hdul[g].pop("PIXTAB")
            ptabfile = specfile.replace(".spec", ".pixtab")
            utils.log_comment(utils.LOGFILE, ptabfile, verbose=VERBOSE_LOG)
            ptab.writeto(ptabfile, overwrite=True)
        else:
            ptab = None

        hdul[g].writeto(specfile, overwrite=True)
        if ptab is not None:
            hdul[g].append(ptab)

        if g.upper() == "PRISM-CLEAR":
            plot_kws["interpolation"] = "nearest"
        else:
            plot_kws["interpolation"] = "hanning"

        fig = msautils.drizzled_hdu_figure(hdul[g], **plot_kws)
        fig.savefig(specfile.replace(".spec.fits", ".fnu.png"))

        fig = msautils.drizzled_hdu_figure(hdul[g], unit="flam", **plot_kws)
        fig.savefig(specfile.replace(".spec.fits", ".flam.png"))

    # Cleanup
    for k in xobj:
        obj = xobj[k]["obj"]
        for sl in obj.slits:
            sl.close()

    if get_xobj:
        return hdul, xobj
    else:
        del xobj
        return hdul
