"""
Scripts for reducting MIR LRS data with a high-quality PSF model
"""

import os
import glob
import yaml
import logging

import numpy as np

import scipy.ndimage as nd

import astropy.wcs as pywcs
import astropy.io.fits as pyfits
import astropy.units as u

import grizli.jwst_utils
from grizli import utils
from grizli import prep

import jwst.datamodels
import mastquery.jwst

from grizli.aws import visit_processor, db

from ..slit_combine import pseudo_drizzle
from .. import utils as msautils
from .. import resample_numba

CROSS_PSCALE = 0.11057

CRDS_CONTEXT = "jwst_1464.pmap"

LOGGER = logging.getLogger(__name__)

LRS_SLIT_MODEL = None

LRS_REFERENCE_SPECTRUM = None

def load_lrs_badpix():
    """
    Load ``msaexp/data/lrs/miri_lrs_badpix.txt table``
    """
    badpix_file = os.path.join(
        msautils.module_data_path(), "lrs", "miri_lrs_badpix.txt"
    )
    lrs_bpix = utils.read_catalog(badpix_file)
    return lrs_bpix


def load_lrs_slit_model():
    """
    Load ``msaexp/data/lrs_slit_model.fits`` into the global LRS_SLIT_MODEL

    Parameters
    ----------
    None

    Returns
    -------
    LRS_SLIT_MODEL : `~astropy.io.fits.HDUList`
        Slit model HDUList

    """
    global LRS_SLIT_MODEL

    if LRS_SLIT_MODEL is None:
        LRS_SLIT_MODEL = pyfits.open(
            os.path.join(
                msautils.module_data_path(), "lrs", "lrs_slit_model.fits"
            )
        )

    return LRS_SLIT_MODEL


def load_lrs_reference_spectrum(file="jw04533001-mds.miri-deep-survey-alma-10_lrs.spec.fits"):
    """
    Load ``jw04533001-mds.miri-deep-survey-alma-10_lrs.spec.fits`` reference
    spectrum into the global LRS_SLIT_MODEL

    Parameters
    ----------
    None

    Returns
    -------
    LRS_SLIT_MODEL : `~astropy.io.fits.HDUList`
        Slit model HDUList

    """
    import msaexp.spectrum
    
    global LRS_REFERENCE_SPECTRUM

    if LRS_REFERENCE_SPECTRUM is None:
        LRS_REFERENCE_SPECTRUM = msaexp.spectrum.SpectrumSampler(
            os.path.join(
                msautils.module_data_path(),
                "lrs",
                file
            )            
        )


    return LRS_REFERENCE_SPECTRUM


def full_lrs_pipeline(
    prog=5141,
    obs=1,
    prefix=None,
    file_query=None,
    skip_preprocessing=False,
    by_visit=True,
    do_query=True,
    load_datamodels=True,
    sys_err=0.01,
    process_tacq=False,
    run_level2=True,
    skip_photom_step=True,
    do_nod_subtraction=True,
    flatten_bkg_trend=False,
    optimize_trace_niter=3,
    x0=[0.0, 0.0],
    nod_ext="cal",
    prof_kwargs={},
    resample_kwargs={},
    quiet_logs=True,
    cleanup=False,
    **kwargs,
):
    """
    Full MIRI LRS pipeline
    """
    from grizli import jwst_utils

    if quiet_logs:
        jwst_utils.set_quiet_logging(50)

    from jwst.pipeline import Spec2Pipeline, Spec3Pipeline, Image2Pipeline

    global CRDS_CONTEXT

    os.environ["CRDS_CONTEXT"] = CRDS_CONTEXT

    grizli.jwst_utils.set_crds_context()

    update_prefix = prefix is None
    if update_prefix:
        prefix = f"jw{prog:05d}{obs:03d}"

    utils.LOGFILE = f"{prefix}_lrs.log.txt"

    if skip_preprocessing:
        do_query = False
        load_datamodels = False
        process_tacq = False
        run_level2 = False

    if do_query & (file_query is None):
        res = mastquery.jwst.query_jwst(
            instrument="MIR",
            filters=(
                mastquery.jwst.make_program_filter([prog])
                + mastquery.jwst.make_query_filter(
                    "exp_type", values=["MIR_LRS-FIXEDSLIT", "MIR_TACQ"]
                )
            ),
            extensions=["rate", "cal"],
            rates_and_cals=True,
        )

        public = res["publicReleaseDate_mjd"] < utils.nowtime().mjd

        keep = np.array(
            [v.startswith(f"{prog:05d}{obs:03d}") for v in res["visit_id"]]
        )
        res = res[keep]

        is_tacq = res["exp_type"] == "MIR_TACQ"
        if (~is_tacq).sum() > 0:
            target = res["targprop"][~is_tacq][0].lower()
        else:
            target = res["targprop"][0].lower()

        if file_query is None:
            file_query = f"jw{prog:05d}{obs:03d}"

        if update_prefix:
            prefix = f"jw{prog:05d}{obs:03d}-{target}"

        utils.LOGFILE = f"{prefix}_lrs.log.txt"

        msg = f"full_lrs_pipeline: {prefix} {len(res)} files"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        _ = lrs_regions(res, output=f"{prefix}_lrs.reg")

        keep_cols = []
        for c in res.colnames:
            if res[c].dtype not in [object]:
                keep_cols.append(c)

        res[keep_cols].write(f"{prefix}_lrs.mast.fits", overwrite=True)

        rates = [f.replace("_cal", "_rate") for f in res["dataURI"]]

        _ = mastquery.utils.download_from_mast(rates)
    else:
        res = None

    if file_query is None:
        file_query = f"jw{prog:05d}{obs:03d}"

    for d in ["Level2", "Level3"]:
        if not os.path.exists(d):
            os.mkdir(d)

    # files = glob.glob(f"{prefix}*rate.fits")
    if load_datamodels:
        files = glob.glob(f"jw{prog:05d}{obs:03d}*rate.fits")
        files.sort()

        dms = [jwst.datamodels.open(file) for file in files]

        pops = []
        tacq_file = None
        for i in range(len(dms)):
            meta = dms[i].meta
            msg = f"# {i} {meta.filename} {meta.exposure.exposure_time:7.1f}"
            msg += f" {meta.exposure.type}"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

            if dms[i].meta.exposure.type == "MIR_TACQ":
                tacq_file = dms[i].meta.filename
                msg = f"*  tacq_file = '{tacq_file}'"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                pops.append(i)

        for p in pops[::-1]:
            _ = dms.pop(p)
    else:
        tacq_file = None

    if (tacq_file is not None) & (process_tacq):
        process_tacq_file(file=tacq_file, prefix=prefix)

    ### Level 2
    if run_level2:
        Spec2 = Spec2Pipeline(
            save_results=True, output_dir=os.path.join(os.getcwd(), "Level2")
        )
        Spec2.photom.skip = skip_photom_step

        Image2 = Image2Pipeline(
            save_results=True, output_dir=os.path.join(os.getcwd(), "Level2")
        )
        cals = []
        for dm in dms:
            cal_file = f"Level2/{dm.meta.filename}".replace("_rate", "_cal")
            if not os.path.exists(cal_file):
                if dm.meta.exposure.type == "MIR_TACONFIRM":
                    continue
                elif dm.meta.exposure.type == "MIR_TACQ":
                    continue
                else:
                    print(f"Run Spec2Pipeline: {cal_file}")
                    cals.append(Spec2.process(dm))

                c = cals[-1][0]
                c.write(c.meta.filename)

    if do_nod_subtraction:
        s2 = nod_sky_subtraction(
            dir="Level2",
            file_query=file_query,
            ext=nod_ext,
            bkg_threshold=0.3,
            flatten_bkg_trend=flatten_bkg_trend,
        )

    result = {
        "prefix": prefix,
        "mast": res,
        "s2": None,
        "data": None,
        "pres": None,
        "hdu": None,
        "fig": None,
    }

    if s2 is not None:
        dms = []
        for targ in s2:
            print(f"Target: {targ}")
            for d1 in s2[targ]:
                dms += s2[targ][d1]["dm"]

        data = pixtab_from_dms(dms, sys_err=sys_err)

        wcs = data["meta"].pop("wcs")
        with open(f"{prefix}_lrs.yaml", "w") as fp:
            yaml.dump(data["meta"], fp)

        data["meta"]["wcs"] = wcs
        data["meta"]["prefix"] = prefix

        result["data"] = data

        prev = [-99, -99]
        for iter_ in range(optimize_trace_niter):
            msg = f"\n# Optimize trace, iter #{iter_}\n"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

            if np.allclose(prev, x0, rtol=0.001):
                break

            prev = x0
            x0 = simple_trace_fit(data, theta=x0, **prof_kwargs)

        hdu, fig = process_2d_products(data, prefix=prefix, **resample_kwargs)

        for unit in ["flam", "fnu"]:
            fig_ = msautils.drizzled_hdu_figure(hdu, unit=unit)
            fig_.savefig(f"{prefix}_lrs.{unit}.png")

        result["hdu"] = hdu
        result["fig"] = fig

    return result


def pixtab_from_dms(dms, sys_err=0.02, **kwargs):
    """
    Make pixel tables
    """

    if isinstance(dms[0], dict):
        ptabs = dms
        files = []
    else:
        ptabs = []
        files = []
        for dm in dms:
            if "_x02101" not in dm.meta.filename:
                ptabs.append(lrs_pixel_table(dm))
                files.append(dm.meta.filename)

    data = {
        "exposure": [],
        "yp": [],
        "dx": [],
        "wave": [],
        "data": [],
        "err": [],
        "var_rnoise": [],
    }

    for i in range(len(ptabs)):  # [1,3]:
        ptab = ptabs[i]
        ok = np.isfinite(ptab["data"] + ptab["err"]) & ((ptab["dq"] & 1) == 0)
        # ok &= ptab["data"] > low_threshold * ptab["err"]

        if 0:
            bkg = ptab["bkg"]
        else:
            bkg = 0

        data["exposure"].append(np.full(ok.sum(), i))
        data["yp"].append(ptab["yp"][ok])
        data["dx"].append(ptab["dx"][ok] - ptab["x_offset"])
        data["wave"].append(ptab["wave"][ok])
        data["data"].append((ptab["data"] - bkg)[ok])
        data["err"].append(ptab["err"][ok])
        data["var_rnoise"].append(ptab["var_rnoise"][ok])

    for k in data:
        data[k] = np.hstack(data[k])

    data["full_err"] = np.sqrt(
        data["err"] ** 2 + (sys_err * np.maximum(data["data"], 0))
    )

    data["err_rnoise"] = np.sqrt(data["var_rnoise"])

    une = utils.Unique(data["exposure"], verbose=False)

    mask = np.isfinite(data["data"] + data["err"])
    badpix = np.zeros(mask.shape, dtype=bool)

    for exp in une.values:
        test = mask & une[exp]
        so = np.argsort(data["err"][test])
        grad = np.gradient(data["err"][test][so]) / data["err"][test][so]

        bad = grad > 0.04

        if bad.sum() > 100:
            continue

        try:
            bp_thresh = data["err"][test][so][bad].min()
            bpix_i = data["err"][une[exp]] >= bp_thresh
            if bpix_i.sum() > 100:
                continue

            badpix[une[exp]] = bpix_i
            msg = f"Exposure {exp} bad pixels: {badpix[une[exp]].sum()}, threshold = {bp_thresh:.3f}"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)
        except ValueError:
            continue

    mask &= ~badpix
    for k in data:
        data[k] = data[k][mask]

    data["meta"] = ptabs[0]["meta"]
    data["meta"]["sys_err"] = sys_err
    data["meta"]["x_offset"] = [ptab["x_offset"] for ptab in ptabs]
    data["file"] = files

    return data


def get_data_slice(data, slice, **kwargs):
    """
    Get slice of arrays in data dictionary

    Parameters
    ----------
    data : dict
        Data dictionary from `msaexp.cloud.lrs.pixtab_from_dms`

    slice : slice, array-like
        Subset of data array.  Can be anything that broadcasts to
        ``data["data"][slice]``, i.e., a `slice` object or boolean or index
        arrays.

    Returns
    -------
    dslice : dict
        Data dictionary with sliced arrays
    """
    dslice = {}
    for k in [
        "yp",
        "xp",
        "exposure",
        "dx",
        "wave",
        "data",
        "err",
        "var_rnoise",
        "wpix",
        "wpix_norm",
        "group",
        "xdet",
        "ydet",
        "profile",
        "psf_trace",
    ]:
        if k in data:
            dslice[k] = data[k][slice] * 1

    for k in ["ybin", "wbin", "wbin_edge"]:
        if k in data:
            dslice[k] = data[k].copy()

    for k in ["meta", "file"]:
        if k in data:
            dslice[k] = data[k].copy()

    if "group" in data:
        groups = np.unique(data["group"][slice])
        if len(groups) == 1:
            dslice["meta"]["filename"] = data["meta"]["group_names"][groups[0]]

    return dslice


def optimal_extraction(
    data,
    err_data="err",
    weight_type="rnoise",
    data_mask=None,
    prof_threshold=0.01,
    low_threshold=-5,
    recenter=False,
    **kwargs,
):
    """
    Optimal extraction with ``data["profile"]``
    """

    if "wbin" not in data:
        set_bin_arrays(data, **kwargs)

    if weight_type == "rnoise":
        data_wht = 1.0 / data["var_rnoise"]
    elif weight_type == "sys_err":
        data_wht = 1.0 / data["full_err"] ** 2
    else:
        data_wht = 1.0 / data["err"] ** 2

    data["meta"]["weight_type"] = weight_type
    data["meta"]["err_data"] = err_data

    valid = np.isfinite(
        data["wave"]
        + data["dx"]
        + data["data"]
        + data[err_data]
        + data_wht
        + data["profile"]
    )

    valid &= data["data"] > low_threshold * data["err"]

    if data_mask is not None:
        valid &= data_mask

    if recenter:
        dx_mu = data["dx"] - data["psf_trace"]
    else:
        dx_mu = data["dx"] + 0.0

    # 2D drizzle
    num, vnum, den, ntot = pseudo_drizzle(
        data["wave"][valid],
        dx_mu[valid],
        data["data"][valid],
        data[err_data][valid] ** 2,
        data_wht[valid],
        data["wbin_edge"],
        data["ybin"],
    )

    pnum, pvnum, pden, pntot = pseudo_drizzle(
        data["wave"][valid],
        dx_mu[valid],
        data["profile"][valid],
        data[err_data][valid] ** 2,
        data_wht[valid],
        data["wbin_edge"],
        data["ybin"],
    )

    # "den" is summed weight
    opt_num = np.nansum(num / den * pnum / den * den, axis=0)
    opt_den = np.nansum((pnum / den) ** 2 * den, axis=0)

    var = vnum / den**2
    var_num = np.nansum(var * (pnum / den * den) ** 2, axis=0)

    # "optimal" 1D spectrum
    spec = opt_num / opt_den
    spec_var = var_num / opt_den**2

    resid = (num / den - pnum / den * spec) / np.sqrt(var)
    mask2d = np.isfinite(resid + var) & (den > 0) & (var > 0)

    # 1D binned directly from pixel table
    if True:
        from scipy.stats import binned_statistic

        mask = valid & True
        # mask &= data["profile"] > prof_threshold * np.nanmax(data["profile"])

        # print("xxx", valid.sum(), mask.sum(), np.nanmax(data["profile"]))

        snum = binned_statistic(
            data["wave"][mask],
            (data["data"] * data["profile"] * data_wht)[mask],
            statistic="sum",
            bins=data["wbin_edge"],
        )

        sunum = binned_statistic(
            data["wave"][mask],
            (np.ones_like(data["data"]) * data["profile"] * data_wht)[mask],
            statistic="sum",
            bins=data["wbin_edge"],
        )

        sden = binned_statistic(
            data["wave"][mask],
            (data["profile"] ** 2 * data_wht)[mask],
            statistic="sum",
            bins=data["wbin_edge"],
        )

        svnum = binned_statistic(
            data["wave"][mask],
            (data[err_data] * data["profile"] * data_wht)[mask] ** 2,
            statistic="sum",
            bins=data["wbin_edge"],
        )

        opt_bin = snum.statistic / sden.statistic
        unit_bin = sunum.statistic / sden.statistic
        var_bin = svnum.statistic / sden.statistic**2

    # p2d = pnum / pden
    # mask2d &= p2d > prof_threshold * np.nanmax(p2d)
    # mask2d &= num / den > low_threshold * np.sqrt(var)

    data["mask2d"] = mask2d
    data["chi2"] = (resid[mask2d] ** 2).sum()

    data["d2d"] = (num, vnum, den, ntot)
    data["p2d"] = (pnum, pvnum, pden, pntot)
    data["opt_num"] = opt_num
    data["opt_den"] = opt_den
    data["var_num"] = var_num
    data["opt_bin"] = opt_bin
    data["unit_bin"] = unit_bin
    data["var_bin"] = var_bin

    data["spec"] = spec
    data["spec_var"] = spec_var

    data["meta"]["recenter"] = recenter


def set_bin_arrays(
    data, ybin=np.arange(-2.5, 2.5, 0.045), wbin=None, **kwargs
):
    """ """
    data["ybin"] = ybin

    so = np.argsort(data["yp"])

    # wpix = np.arange(data["yp"].min(), data["yp"].max(), 1.0)
    if wbin is None:
        wpix = np.arange(data["yp"].min(), data["yp"].max(), 1.0)
        wbin = np.interp(wpix, data["yp"][so], data["wave"][so])
        wbin = wbin[np.argsort(wbin)]
        wbin = wbin[np.isfinite(wbin)]

    wbin_edge = msautils.array_to_bin_edges(wbin)

    data["ybin"] = ybin

    data["wpix"] = np.interp(data["wave"], wbin, np.arange(len(wbin)))
    data["wpix_norm"] = data["wpix"] * 2 / len(wbin) - 1

    data["wbin"] = wbin
    data["wbin_edge"] = wbin_edge


# def get_profile(
#     data,
#     x0=[0.0, 0.0],
#     trace_offset=0.0,
#     ybin=np.arange(-2.5, 2.5, 0.045),
#     optimize=True,
#     fmin_kwargs=dict(method="powell", tol=1.0e-5, options={"xrtol": 1.0e-4}),
#     **kwargs,
# ):
#     """
#     deprecated, use evaluate_model_psf and/or simple_trace_fit
#     """
#     from scipy.optimize import minimize
#
#     psf_tab = utils.read_catalog("lrs_psf_data_jw06620-obs13.fits")
#
#     set_bin_arrays(data, **kwargs)
#
#     data["dx_trace"] = data["dx"] - trace_offset
#
#     if "wbin" not in data:
#         set_bin_arrays(data, **kwargs)
#
#     mu0 = np.interp(data["wave"], psf_tab["wave"], psf_tab["x0"])
#     psf_width = np.interp(data["wave"], psf_tab["wave"], psf_tab["sigma"])
#
#     num, vnum, den, ntot = pseudo_drizzle(
#         data["wave"],
#         data["dx_trace"],
#         data["data"],
#         data["err"] ** 2,
#         1.0 / data["err"] ** 2,
#         data["wbin_edge"],
#         data["ybin"],
#     )
#
#     def fit_profile(theta, mask2d, spec, ret):
#
#         trace_offset, src_width = theta
#         src_width = np.abs(src_width)
#
#         sig = np.sqrt(psf_width**2 + (src_width / CROSS_PSCALE) ** 2)
#         mu = mu0 + trace_offset / CROSS_PSCALE
#
#         prof = resample_numba.pixel_integrated_gaussian_numba(
#             data["dx_trace"] / CROSS_PSCALE, mu, sig, dx=1.0
#         )
#
#         pnum, pvnum, pden, pntot = pseudo_drizzle(
#             data["wave"],
#             data["dx_trace"],
#             prof,
#             data["err"] ** 2,
#             1.0 / data["err"] ** 2,
#             data["wbin_edge"],
#             data["ybin"],
#         )
#
#         if spec is None:
#             opt_num = np.nansum(num * pnum / den**2 * den, axis=0)
#             opt_den = np.nansum((pnum / den) ** 2 * den, axis=0)
#             spec = opt_num / opt_den
#         else:
#             opt_num = None
#             opt_den = None
#
#         resid = (num / den - pnum / den * spec) * np.sqrt(den)
#
#         if mask2d is None:
#             mask2d = np.isfinite(resid) & (den > 0)
#             mask2d &= (data["wbin"] > 5.5) & (data["ybin"])
#             mask2d &= np.abs(data["ybin"][:-1][:, None]) < 1.0
#
#         chi2 = (resid[mask2d] ** 2).sum()
#         print(theta, chi2)
#
#         if ret == 0:
#             return chi2
#         else:
#             result = {
#                 "theta": theta,
#                 "profile": prof,
#                 "d2d": (num, vnum, den, ntot),
#                 "p2d": (pnum, pvnum, pden, pntot),
#                 "resid": resid,
#                 "mask2d": mask2d,
#                 "chi2": chi2,
#                 "opt_num": opt_num,
#                 "opt_den": opt_den,
#                 "spec": spec,
#                 "trace_offset": trace_offset,
#                 "src_width": src_width,
#             }
#             return result
#
#     init = fit_profile(x0, None, None, 1)
#     if optimize:
#         args = (init["mask2d"], init["spec"], 0)
#         res = minimize(
#             fit_profile,
#             x0=x0,
#             args=args,
#             **fmin_kwargs,
#         )
#
#         final = fit_profile(res.x, None, None, 1)
#     else:
#         final = init
#
#     for k in final:
#         data[k] = final[k]
#
#     return final

CALSPEC_FILE = os.path.join(
    msautils.module_data_path(), "lrs", "calspec_correction_jw06620-obs13.fits"
)


def process_2d_products(
    data,
    calspec_file=CALSPEC_FILE,
    prefix="lrs",
    z=0,
    show_2d_corr=True,
    show_sn_threshold=5,
    vmax=None,
    make_figure=True,
    xlim=None,
    **kwargs,
):
    """
    Combine 2D data

    Parameters
    ----------
    data : dict
        Pixel table data from `msaexp.cloud.lrs.pixtab_from_dms`

    calspec_file : str

    prefix : str

    z : float

    show_2d_corr : bool

    show_sn_threshold : float
        Threshold on median spectrum S/N below which the 2D plot is normalized
        by the overall S/N rather than flux units

    Returns
    -------
    hdul : `astropy.io.fits.HDUList`

    fig : `matplotlib.figure.Figure`

    """
    import matplotlib.pyplot as plt

    if "d2d" not in data:
        # _ = get_profile(data, x0=x0, **kwargs)
        _ = simple_trace_fit(data, **kwargs)

    (num, vnum, den, ntot) = data["d2d"]
    (pnum, pvnum, pden, pntot) = data["p2d"]
    spec = data["spec"]
    ybin = data["ybin"]
    wbin = data["wbin"]

    calsp = utils.read_catalog(calspec_file)
    corr = np.interp(
        data["wbin"], calsp["wave"], calsp["corr"], left=0, right=0
    )

    if show_2d_corr:
        corr2d = np.interp(
            data["wbin"],
            calsp["wave"],
            calsp["corr"] / np.gradient(calsp["wave"]),
            left=0,
            right=0,
        )

    else:
        corr2d = 1.0

    median_sn = np.nanmedian(data["spec"] / np.sqrt(data["spec_var"]))
    if median_sn < show_sn_threshold:
        corr2d = 1.0 / np.sqrt(den)
        corr2d = np.sqrt(data["spec_var"])  # / corr

        if vmax is None:
            vmax = 2 * np.nanpercentile(num / den / corr2d, 95)
            vmax = np.maximum(vmax, 3)

            # print("xxx vmax", vmax)

    elif vmax is None:
        vmax = 5 * np.nanpercentile(num / den / corr2d, 90)

    if make_figure:
        fig, axes = plt.subplots(
            3, 1, figsize=(8, 7), sharex=True, sharey=True
        )

        kws = dict(aspect="auto", cmap="bone", vmin=-0.5 * vmax, vmax=vmax)

        axes[0].imshow(num / den / corr2d, **kws)
        axes[1].imshow((pnum) / pden * spec / corr2d, **kws)
        axes[2].imshow((num / den - pnum / den * spec) / corr2d, **kws)

        for ax in axes:
            ax.grid()

        yt = np.arange(np.ceil(ybin[0]), np.floor(ybin[-1]), 1)

        offsets = np.round(np.array(data["meta"]["x_offset"]) * 10) / 10
        offsets -= offsets[0]
        offsets = np.unique(np.append(offsets, -offsets))

        ax.set_yticks(np.interp(offsets, ybin, np.arange(len(ybin))))
        ax.set_yticklabels([f"{v:.1f}" for v in offsets])

        axes[0].set_ylabel("data, " + r"$\Delta$ [arcsec]")
        axes[1].set_ylabel(r"2D model")
        axes[2].set_ylabel(r"residual")

        axes[1].text(
            0.05,
            0.95,
            "_".join(data["meta"]["filename"].split("_")[:2])
            + "  "
            + data["meta"]["target"]["proposer_name"],
            color="k",
            transform=axes[1].transAxes,
            bbox={"fc": "w", "ec": "None", "alpha": 0.8},
            ha="left",
            va="top",
        )

        zplot = z

        xt = np.arange(5, 14.1) / (1 + zplot)

        if zplot > 6:
            xt = np.arange(0.6, 1.31, 0.1)

        if zplot > 11:
            xt = np.arange(0.4, 0.8, 0.1)

        ax.set_xticks(np.interp(xt, wbin / (1 + zplot), np.arange(len(wbin))))

        if zplot == 0:
            ax.set_xticklabels(xt.astype(int))
            ax.set_xlabel(r"$\lambda_\mathrm{obs}$")
        else:
            ax.set_xticklabels([f"{v:.2f}" for v in xt])
            ax.set_xlabel(r"$\lambda_\mathrm{rest}$" + f"  z={zplot:.4f}")

        ax.set_xlim(0, len(wbin))

        if xlim is not None:
            xlim = np.interp(xlim, wbin, np.arange(len(wbin)))
            ax.set_xlim(*xlim)

        fig.tight_layout(pad=1)
    else:
        fig = None

    tab = utils.GTable()
    tab["wave"] = data["wbin"] * u.micron
    tab["flux"] = data["spec"] / corr * u.microJansky
    # tab["err"] = 1 / np.sqrt(data["opt_den"]) / corr * u.microJansky
    tab["err"] = np.sqrt(data["spec_var"]) / corr * u.microJansky
    tab["flux_bin"] = data["opt_bin"] / corr * u.microJansky
    tab["err_bin"] = np.sqrt(data["var_bin"]) / corr * u.microJansky
    tab["calsp"] = corr / u.microJansky

    tab.meta["grating"] = "LRS"
    tab.meta["filter"] = "P750L"
    tab.meta["instrume"] = "MIRI"
    tab.meta["srcname"] = prefix

    tab.meta["z"] = z
    tab.meta["ytrace"] = np.interp(
        0, data["ybin"], np.arange(len(data["ybin"]))
    )

    ypscale = np.diff(data["ybin"])[0]

    if data["meta"]["recenter"]:
        tab.meta["profcen"] = 0.0
    else:
        tab.meta["profcen"] = data["meta"]["theta"][-2] / ypscale

    tab.meta["profsig"] = data["meta"]["theta"][-1] / ypscale
    tab.meta["pscale"] = ypscale

    tab.meta["ntheta"] = len(data["meta"]["theta"])
    for i, value in enumerate(data["meta"]["theta"]):
        tab.meta[f"theta{i}"] = value

    hdul = pyfits.HDUList(
        [
            pyfits.PrimaryHDU(),
            pyfits.BinTableHDU(tab, name="SPEC1D"),
            pyfits.ImageHDU(data=num / den / corr, name="SCI"),
            pyfits.ImageHDU(data=den**2 / vnum * corr**2, name="WHT"),
            pyfits.ImageHDU(data=pnum / pden, name="PROFILE"),
            pyfits.BinTableHDU(pfit_table(data, **kwargs), name="PROF1D"),
        ]
    )

    for ext in ["SCI", "WHT"]:
        for k in tab.meta:
            hdul[ext].header[k] = tab.meta[k]

    if prefix is not None:
        if fig is not None:
            fig.savefig(f"{prefix}_lrs.2d.png")

        hdul.writeto(f"{prefix}_lrs.spec.fits", overwrite=True)

    return hdul, fig


def pfit_table(data, sn_threshold=20, **kwargs):
    """
    Make a PROF1D table for msaexp.spectrum
    """
    (num, vnum, den, ntot) = data["d2d"]
    (pnum, pvnum, pden, pntot) = data["p2d"]

    spec = data["spec"]
    spec_sn = data["spec"] * np.sqrt(data["opt_den"])

    sub = spec_sn > sn_threshold
    if sub.sum() < 16:
        sub = np.isfinite(spec_sn)

    ptab = utils.GTable()
    ptab["profile"] = np.nansum((num * spec)[:, sub], axis=1) / np.nansum(
        (den * spec**2)[:, sub], axis=1
    )

    ptab["pfit"] = np.nansum((pnum)[:, sub], axis=1) / np.nansum(
        (pden)[:, sub], axis=1
    )

    yp, xp = np.indices(num.shape)
    ptab.meta["profstrt"] = xp[:, sub].min()
    ptab.meta["profstop"] = xp[:, sub].max()

    return ptab


def process_tacq_file(file, prefix="lrs_tacq"):
    """
    Process a TACQ file
    """

    grizli.jwst_utils.initialize_jwst_image(file)
    dm = grizli.jwst_utils.img_with_wcs(file)

    outh, outw = utils.make_maximal_wcs(
        files=[file], pixel_scale=0.11, get_hdu=False
    )

    res = visit_processor.res_query_from_local(files=[file])

    visit_processor.cutout_mosaic(
        rootname=prefix,
        ir_wcs=outw,
        half_optical=False,
        clean_flt=False,
        s3output=None,
        kernel="square",
        pixfrac=1.0,
        res=res,
        jwst_dq_flags=None,
        skip_existing=False,
    )


def lrs_regions(res, output="lrs.reg"):
    """
    make regions for a mastquery table
    """

    import pysiaf
    import jwst.datamodels
    from pysiaf.utils.rotations import attitude

    MIR = pysiaf.Siaf("MIRI")
    slit_ap = MIR["MIRIM_SLIT"]
    illum_ap = MIR["MIRIM_ILLUM"]

    slit_polygons = []
    illum_polygons = []

    with open(output, "w") as fp:

        fp.write("icrs\n")

        for row in res:
            att = attitude(
                slit_ap.V2Ref,
                slit_ap.V3Ref,
                row["targ_ra"],
                row["targ_dec"],
                row["gs_v3_pa"],
            )
            slit_ap.set_attitude_matrix(att)
            illum_ap.set_attitude_matrix(att)

            sr = utils.SRegion(
                np.array(slit_ap.corners("sky", rederive=False))
            )

            label = "{visit_id} {targprop}".format(**row)
            sr.label = label

            slit_polygons += sr.polystr(precision=5)

            sr.ds9_properties = "color=cyan"

            for reg in sr.region:
                fp.write(reg + "\n")

            sr = utils.SRegion(
                np.array(illum_ap.corners("sky", rederive=False))
            )

            sr.label = label
            sr.ds9_properties = "color=magenta"

            for reg in sr.region:
                fp.write(reg + "\n")

            illum_polygons += sr.polystr(precision=5)

    res["slit_footprint"] = slit_polygons
    res["illum_footprint"] = illum_polygons


def nod_sky_subtraction(
    dir="Level2",
    file_query="jw0",
    ext="cal",
    bkg_threshold=0.3,
    flatten_bkg_trend=True,
    **kwargs,
):
    """
    Compute differences of nodded exposures for sky subtraction
    """

    full_file_query = os.path.join(dir, file_query + "*" + ext + ".fits")

    s2_files = glob.glob(full_file_query)

    lrs_bpix = load_lrs_badpix()

    if (ext == "bkg") & (len(s2_files) == 0):
        ext = "cal"
        full_file_query = os.path.join(dir, file_query + "*" + ext + ".fits")
        s2_files = glob.glob(full_file_query)

    if len(s2_files) == 0:
        msg = f"no files found for {full_file_query}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        return None

    s2_files.sort()

    s2 = {}
    for i, file in enumerate(s2_files):

        msg = f"Load datamodel {i:>2}: {file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        dm = jwst.datamodels.open(file)
        if dm.meta.exposure.type == "MIR_TACQ":
            dm.close()
            continue

        # Apply badpix mask
        dm.dq[lrs_bpix["yi"], lrs_bpix["xi"]] |= 1
        dq_mask = (dm.dq & 1) > 0

        # targ, grp, dith, _, _ = file.split('_')

        dith = f"{dm.meta.dither.x_offset:>4.1f}"
        targ = dm.meta.target.proposer_name.lower()

        if targ not in s2:
            s2[targ] = {}

        if dith not in s2[targ]:
            s2[targ][dith] = {"dm": [], "scl": []}

        if flatten_bkg_trend:
            if i == 0:
                bkg_ref = dm.data * np.nan**dq_mask
                bkg_ratio_i = 1.0
            else:
                bkg_ratio_i = np.nanmedian(dm.data / bkg_ref)

            msg = f"flatten_bkg_trend: {file} {bkg_ratio_i:8.3f}"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

        else:
            bkg_ratio_i = 1.0

        s2[targ][dith]["dm"].append(dm)
        s2[targ][dith]["scl"].append(bkg_ratio_i)

    if ext == "bkg":
        return s2

    ###
    for targ in s2:
        msg = f"Target: {targ}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        for dith in s2[targ]:
            msg = f'  dither {dith}: {len(s2[targ][dith]["dm"])} exposures'
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

            s2_i = s2[targ][dith]

            s2_i["avg"] = np.nanmean(
                np.array(
                    [
                        dm.data / scl * np.nan ** (dm.dq & 1)
                        for (dm, scl) in zip(s2_i["dm"], s2_i["scl"])
                    ]
                ),
                axis=0,
            )

            s2_i["var"] = np.nanmean(
                np.array(
                    [
                        (dm.err / scl * np.nan ** (dm.dq & 1)) ** 2
                        for (dm, scl) in zip(s2_i["dm"], s2_i["scl"])
                    ]
                ),
                axis=0,
            )

        for d1 in s2[targ]:
            other_dithers = []
            other_vars = []
            x_offsets = []

            for d2 in s2[targ]:
                x_offset = float(d2) - float(d1)
                if (d1 != d2) & (np.abs(x_offset) > bkg_threshold):
                    x_offsets.append(x_offset)
                    other_dithers.append(s2[targ][d2]["avg"])
                    other_vars.append(s2[targ][d2]["var"])

            if len(other_dithers) == 0:
                other_avg = 0.0
                other_var = 0.0
            else:
                other_avg = np.nanmean(np.array(other_dithers), axis=0)
                other_var = np.nanmean(np.array(other_vars), axis=0)

            s2_i = s2[targ][d1]

            for dm, scl in zip(s2_i["dm"], s2_i["scl"]):
                dm.data = (dm.data / scl - other_avg) * scl
                dm.err = np.sqrt((dm.err / scl) ** 2 + other_var) * scl

                dm.meta.filename = dm.meta.filename.replace("_bkg", "")

                out_file = os.path.join(
                    dir, dm.meta.filename.replace(ext, ext + "_bkg")
                )

                offset_summary = ",".join(
                    [f"{off:>5.1f}" for off in x_offsets]
                )

                msg = f"{out_file}:  src= {d1}  bkg= {offset_summary}"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)

                dm.write(out_file)

    return s2


def lrs_pixel_table(dm, **kwargs):
    """
    Generate a pixel table from LRS data

    Parameters
    ----------
    dm : `ImageModel` from cal file

    Returns
    -------
    pixtab : dict
        Pixel table data
    """
    # d2w = dm.meta.wcs.get_transform("detector", "world")
    d2a = dm.meta.wcs.get_transform("detector", "alpha_beta")

    yp, xp = np.indices(dm.data.shape)
    dx, dy, wave = d2a(xp, yp)

    ok = np.isfinite(dm.data)

    xsl = slice(xp[ok].min(), xp[ok].max())
    ysl = slice(yp[ok].min(), yp[ok].max())

    pixtab = {
        "data": dm.data[ysl, xsl],
        "err": dm.err[ysl, xsl],
        "var_rnoise": dm.var_rnoise[ysl, xsl],
        "var_poisson": dm.var_poisson[ysl, xsl],
        "var_flat": dm.var_flat[ysl, xsl],
        "dq": dm.dq[ysl, xsl],
        "wave": wave[ysl, xsl],
        "dx": dx[ysl, xsl],
        "xp": xp[ysl, xsl],
        "yp": yp[ysl, xsl],
        "slice": (ysl, xsl),
        "shape": dm.data[ysl, xsl].shape,
        "meta": dm.meta.instance,
        "slices": (xsl, ysl),
        "x_offset": dm.meta.dither.x_offset,
        "y_offset": dm.meta.dither.y_offset,
        "to_ujy": 1.0,
    }

    try:
        pixtab["to_ujy"] = 1.0e12 * dm.meta.photometry.pixelarea_steradians
    except TypeError:
        pass

    return pixtab


def plot_lrs_pixel_table(
    ptab, figsize=(6, 6), cmap="magma", wave_power=3, vmax=500
):
    """ """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    kws = dict(
        aspect="auto",
        extent=(
            np.nanmin(ptab["wave"]),
            np.nanmax(ptab["wave"]),
            np.nanmin(ptab["dx"]),
            np.nanmax(ptab["dx"]),
        ),
        cmap=cmap,
    )

    axes[0].imshow(
        np.exp(-np.rot90(ptab["dx"] - ptab["x_offset"], -1) ** 2 / 2 / 0.3**2),
        **kws,
    )
    axes[1].imshow(
        np.rot90(ptab["data"] / (ptab["wave"] / 15) ** wave_power, -1),
        vmin=-0.2 * vmax,
        vmax=vmax,
        **kws,
    )
    fig.text(
        0.5,
        1.0,
        ptab["meta"]["filename"],
        ha="center",
        va="bottom",
        transform=fig.transFigure,
    )

    axes[0].set_ylabel(r"$\Delta x$  [arcsec]")

    for ax in axes:
        ax.grid()
        ax.set_xlabel(r"$\lambda$  $[\mu\mathrm{m}]$")

    fig.set_label("test")

    fig.tight_layout(pad=1)
    return fig


def test_psf_convolution():

    import scipy.ndimage as nd
    import matplotlib.pyplot as plt

    w1, f1 = 0.4, 0.7
    w2, f2 = 1.0, 0.3
    w3 = 0.1

    x = np.linspace(-8, 8, 513)

    y1 = f1 / np.sqrt(2 * np.pi * w1**2) * np.exp(-(x**2) / 2 / w1**2)
    y2 = f2 / np.sqrt(2 * np.pi * w2**2) * np.exp(-(x**2) / 2 / w2**2)

    g3 = 1 / np.sqrt(2 * np.pi * w3**2) * np.exp(-(x**2) / 2 / w3**2)

    wg1 = np.sqrt(w1**2 + w3**2)
    wg2 = np.sqrt(w2**2 + w3**2)

    yg1 = f1 / np.sqrt(2 * np.pi * wg1**2) * np.exp(-(x**2) / 2 / wg1**2)
    yg2 = f2 / np.sqrt(2 * np.pi * wg2**2) * np.exp(-(x**2) / 2 / wg2**2)

    yk = y1 + y2
    yk *= (f1 + f2) / yk.sum()
    cg3 = nd.convolve(g3, yk)

    cg1 = nd.convolve(g3, f1 * y1 / y1.sum())
    cg2 = nd.convolve(g3, f2 * y2 / y2.sum())

    plt.plot(x, yg1 + yg2)
    plt.plot(x, cg3)


def evaluate_model_psf(
    data,
    theta,
    psf_data=None,
    simple_psf=False,
    extended_psf_data=None,
    do_optimal_extraction=False,
    **kwargs,
):
    """ """
    if psf_data is None:
        data_path = msautils.module_data_path()
        psf_tab = utils.read_catalog(
            os.path.join(data_path, "lrs", "lrs_psf_data_jw06620-obs13.fits")
        )

        psf_mu = np.interp(data["wave"], psf_tab["wave"], psf_tab["x0"])
        psf_width = np.interp(data["wave"], psf_tab["wave"], psf_tab["sigma"])
    else:
        psf_mu, psf_width = psf_data

    trace_offset = np.polyval(theta[:-1], data["wpix_norm"])
    mu = psf_mu + trace_offset / CROSS_PSCALE

    src_width = theta[-1]

    result = {
        "theta": theta,
        "psf_mu": psf_mu,
        "psf_trace": mu * CROSS_PSCALE,
        "psf_width": psf_width,
        "trace_offset": trace_offset,
        "src_width": src_width,
        "sigma_scales": None,
        "sigma_coeffs": None,
    }

    if simple_psf:
        sig = np.sqrt(psf_width**2 + (src_width / CROSS_PSCALE) ** 2)
        result["profile"] = resample_numba.pixel_integrated_gaussian_numba(
            data["dx"] / CROSS_PSCALE, mu, sig, dx=1.0
        )
        if do_optimal_extraction:
            data["profile"] = result["profile"]
            optimal_extraction(data, **kwargs)

        return result

    # Wavelength-dependent PSF components
    if extended_psf_data is None:
        data_path = msautils.module_data_path()
        ctab = utils.read_catalog(
            os.path.join(data_path, "lrs", "lrs_psf_data_extended_shift.fits")
        )

        sigma_scales = ctab["psf_scale"][0]
        if "psf_w" in ctab.colnames:
            sigma_widths = ctab["psf_w"][0]
            psf_shifts = ctab["psf_x"][0]
        else:
            sigma_widths = sigma_scales * 0.0
            psf_shifts = sigma_scales * 0.0

        nc = len(sigma_scales)
        sigma_coeffs = np.array(
            [
                np.interp(data["wave"], ctab["wcoeffs"], ctab["coeffs"][:, i])
                for i in range(nc)
            ]
        )
    else:
        (sigma_scales, sigma_widths, psf_shifts, sigma_coeffs) = (
            extended_psf_data
        )

    result["sigma_scales"] = sigma_scales
    result["sigma_coeffs"] = sigma_coeffs
    result["sigma_widths"] = sigma_widths
    result["psf_shifts"] = psf_shifts

    result["profile"] = np.zeros_like(data["dx"])

    for i, scl in enumerate(sigma_scales):
        sig = np.sqrt(
            (psf_width * scl + sigma_widths[i]) ** 2
            + (src_width / CROSS_PSCALE) ** 2
        )

        result["profile"] += (
            resample_numba.pixel_integrated_gaussian_numba(
                data["dx"] / CROSS_PSCALE, mu + psf_shifts[i], sig, dx=1.0
            )
            * sigma_coeffs[i, :]
        )

    if do_optimal_extraction:
        data["profile"] = result["profile"]
        data["trace_offset"] = result["trace_offset"]
        data["psf_trace"] = result["psf_trace"]
        data["meta"]["src_width"] = src_width
        data["meta"]["theta"] = theta

        optimal_extraction(data, **kwargs)

    return result


def fit_shift_and_size(
    data,
    initial_shift=[0.0, 0.0],
    initial_size=0.0,
    fit_shift=True,
    fit_size=True,
    theta_scale=10.0,
    tol=1.0e-4,
    method="powell",
    verbose=True,
    **kwargs,
):
    """
    Fit for trace shift and source size

    Parameters
    ----------
    data : dict
        Pixel table data from `msaexp.cloud.lrs.pixtab_from_dms`

    initial_shift, initial_size : list, float
        Shift coefficients and source size parameters, see
        `msaexp.cloud.lrs.evaluate_model_psf`.

    fit_shift : bool
        Fit for the shift

    fit_size : bool
        Fit for the size

    theta_scale : float
        Scale factors for the fit parameters before passing to
        `scipy.optimize.minimize`

    method : str
        Optimizer method

    tol : float
        Optimizer tolerance

    verbose: bool
        Print status messages

    """
    from scipy.optimize import minimize

    psf_data = evaluate_model_psf(data, [0.0, 0.0], do_optimal_extraction=True)

    psf_dict = (psf_data["psf_mu"], psf_data["psf_width"])

    ext_psf_dict = [
        psf_data[k]
        for k in ["sigma_scales", "sigma_widths", "psf_shifts", "sigma_coeffs"]
    ]

    def optimize_with_shift(xtheta, theta_ref, fit_type):

        if fit_type == 0:
            theta_ = np.append(xtheta, theta_ref[-1:]) / theta_scale
        else:
            theta_ = np.append(theta_ref[:-1], xtheta[0]) / theta_scale

        # Evaluate: theta = [{shift poly coeffs}, size]

        psf_data = evaluate_model_psf(
            data,
            theta_,
            do_optimal_extraction=True,
            psf_data=psf_dict,
            extended_psf_data=ext_psf_dict,
        )

        if verbose:
            tstr = [f"{t:6.3f}" for t in theta_]
            tstr[0] = "(" + tstr[0]
            tstr[-1] = f")  s={tstr[-1]}"
            print(f"  {' '.join(tstr)}  {data['chi2']:10.3f}")

        return data["chi2"]

    initial_theta = np.append(initial_shift, [initial_size]) * theta_scale

    if fit_shift:
        if verbose:
            print("\nFit shift:")

        t0 = np.array(initial_shift) * theta_scale
        args = (initial_theta, 0)

        sres = minimize(
            optimize_with_shift, x0=t0, method=method, tol=tol, args=args
        )

        fit_theta = np.append(sres.x, initial_theta[-1:]) / theta_scale

    else:
        fit_theta = initial_theta / theta_scale

    if fit_size:
        if verbose:
            print("\nFit sigma:")

        args = (fit_theta * theta_scale, 1)
        t0 = fit_theta[-1:] * theta_scale

        pres = minimize(
            optimize_with_shift, x0=t0, method=method, tol=tol, args=args
        )

        fit_theta = np.append(fit_theta[:-1], pres.x / theta_scale)

    # Set data["profile"] with final fit parameters
    psf_data = evaluate_model_psf(data, fit_theta, do_optimal_extraction=True)

    return fit_theta


def profile_pathloss(
    data,
    theta,
    psf_data=None,
    simple_psf=False,
    extended_psf_data=None,
    do_optimal_extraction=False,
    slit_width=0.51,
    **kwargs,
):
    """ """
    from scipy.stats import norm

    if psf_data is None:
        data_path = msautils.module_data_path()
        psf_tab = utils.read_catalog(
            os.path.join(data_path, "lrs", "lrs_psf_data_jw06620-obs13.fits")
        )

        psf_mu = np.interp(data["wave"], psf_tab["wave"], psf_tab["x0"])
        psf_width = np.interp(data["wave"], psf_tab["wave"], psf_tab["sigma"])
    else:
        psf_mu, psf_width = psf_data

    if "wave_scale" not in data:
        data["wave_scale"] = data["wave"] / np.nanmedian(data["wave"])

    trace_offset = np.polyval(theta[:-1], data["wave_scale"])
    mu = psf_mu + trace_offset / CROSS_PSCALE

    src_width = theta[-1]

    result = {
        "theta": theta,
        "psf_mu": psf_mu,
        "psf_width": psf_width,
        "trace_offset": trace_offset,
        "src_width": src_width,
        "sigma_scales": None,
        "sigma_coeffs": None,
    }

    # Wavelength-dependent PSF components
    if extended_psf_data is None:
        data_path = msautils.module_data_path()
        ctab = utils.read_catalog(
            os.path.join(data_path, "lrs", "lrs_psf_data_extended_shift.fits")
        )

        sigma_scales = ctab["psf_scale"][0]
        if "psf_w" in ctab.colnames:
            sigma_widths = ctab["psf_w"][0]
            psf_shifts = ctab["psf_x"][0]
        else:
            sigma_widths = sigma_scales * 0.0
            psf_shifts = sigma_scales * 0.0

        nc = len(sigma_scales)
        sigma_coeffs = np.array(
            [
                np.interp(data["wave"], ctab["wcoeffs"], ctab["coeffs"][:, i])
                for i in range(nc)
            ]
        )
    else:
        (sigma_scales, sigma_widths, psf_shifts, sigma_coeffs) = (
            extended_psf_data
        )

    result["sigma_scales"] = sigma_scales
    result["sigma_coeffs"] = sigma_coeffs
    result["sigma_widths"] = sigma_widths
    result["psf_shifts"] = psf_shifts

    # result["profile"] = np.zeros_like(data["dx"])

    psf_sum = np.zeros_like(data["dx"])
    src_sum = np.zeros_like(data["dx"])

    for i, scl in enumerate(sigma_scales):
        sig = np.sqrt(
            (psf_width * scl + sigma_widths[i]) ** 2
            + (src_width / CROSS_PSCALE) ** 2
        )

        gau_psf = norm(
            loc=psf_shifts[i] * CROSS_PSCALE,
            scale=np.sqrt((psf_width * scl + sigma_widths[i]) ** 2)
            * CROSS_PSCALE,
        )

        psf_sum += (
            gau_psf.cdf(slit_width / 2.0) - gau_psf.cdf(-slit_width / 2.0)
        ) * sigma_coeffs[i, :]

        gau_src = norm(
            loc=psf_shifts[i] * CROSS_PSCALE,
            scale=sig * CROSS_PSCALE,
        )

        src_sum += (
            gau_src.cdf(slit_width / 2.0) - gau_src.cdf(-slit_width / 2.0)
        ) * sigma_coeffs[i, :]

    result["pathloss"] = src_sum / psf_sum

    return result


def simple_trace_fit(
    data,
    theta=[0.0, 0.0],
    optimize=["offset", "sigma"],
    # offset_grid=np.linspace(-1.15, 1.15, 5),
    offsets=np.array([-0.8, -0.05, 0, 0.05, 0.8]),
    simple_psf=False,
    tol=1.0e-3,
    max_iter=10,
    **kwargs,
):
    """
    stepwise trace fit
    """
    import matplotlib.pyplot as plt

    if "ybin" not in data:
        set_bin_arrays(data, **kwargs)

    if optimize in [True]:
        naxes = 2
        optimize = ["offset", "sigma"]
    else:
        naxes = ("offset" in optimize) * 1 + ("sigma" in optimize) * 1

    if naxes > 0:
        fig, axes = plt.subplots(
            1, naxes, figsize=(4 * naxes, 4), sharex=False, sharey=True
        )
        if naxes == 1:
            axes = [axes]
    else:
        fig = None

    if "offset" in optimize:
        chi2_grid = offsets * 0.0

        psf_data = None
        extended_psf_data = None

        offset_grid = offsets + theta[0]

        for i, off in enumerate(offset_grid):
            xpsf = evaluate_model_psf(
                data,
                [off, theta[1]],
                do_optimal_extraction=True,
                simple_psf=simple_psf,
                psf_data=psf_data,
                extended_psf_data=extended_psf_data,
                **kwargs,
            )

            if i == 0:
                psf_data = xpsf["psf_mu"], xpsf["psf_width"]

                extended_psf_data = (
                    xpsf["sigma_scales"],
                    xpsf["sigma_widths"],
                    xpsf["psf_shifts"],
                    xpsf["sigma_coeffs"],
                )

            chi2_grid[i] = data["chi2"]

            msg = f"    offset {off:6.3f}  {chi2_grid[i]:.1f}"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

        axes[0].plot(offset_grid, chi2_grid)
        axes[0].set_ylabel(r"$\chi^2$")
        axes[0].set_xlabel("shift, arcsec")

        j = np.argmin(chi2_grid)
        if j == 0:
            j = 1
        elif j == len(offset_grid) - 1:
            j = len(offset_grid) - 2

        omin = offset_grid[j - 1 : j + 2] * 1
        cmin = chi2_grid[j - 1 : j + 2] * 1

        prev = offset_grid[j]

        for iter_ in range(max_iter):
            c = np.polyfit(omin, cmin, 2)

            best_shift = -c[1] / 2 / c[0]

            xpsf = evaluate_model_psf(
                data,
                [best_shift, theta[1]],
                do_optimal_extraction=True,
                simple_psf=simple_psf,
                psf_data=psf_data,
                extended_psf_data=extended_psf_data,
                **kwargs,
            )

            k = np.argmax(cmin)
            omin[k] = best_shift
            cmin[k] = data["chi2"]
            axes[0].plot(omin, cmin, marker=".")

            dx = best_shift - prev
            prev = best_shift

            msg = f"({iter_}) offset {best_shift:6.3f}  {cmin[k]:.1f}  dx={dx:>9.2e}  c0={c[0]:.1e}"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

            if np.abs(dx) < tol:
                break
    else:
        psf_data = None
        extended_psf_data = None

        xpsf = evaluate_model_psf(
            data,
            theta,
            do_optimal_extraction=True,
            simple_psf=simple_psf,
            psf_data=psf_data,
            extended_psf_data=extended_psf_data,
            **kwargs,
        )

        psf_data = xpsf["psf_mu"], xpsf["psf_width"]

        extended_psf_data = (
            xpsf["sigma_scales"],
            xpsf["sigma_widths"],
            xpsf["psf_shifts"],
            xpsf["sigma_coeffs"],
        )

        best_shift = theta[0]
        cmin = None

    if "sigma" in optimize:

        smin = np.array([-0.05, 0.1, 0.2])
        scmin = np.zeros(3)

        for i, s_i in enumerate(smin):
            xpsf = evaluate_model_psf(
                data,
                [best_shift, s_i],
                do_optimal_extraction=True,
                simple_psf=simple_psf,
                psf_data=psf_data,
                extended_psf_data=extended_psf_data,
                **kwargs,
            )

            scmin[i] = data["chi2"]
            msg = f"       sig {s_i:6.3f}  {scmin[i]:.1f}"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

        axes[-1].plot(smin, scmin)
        axes[-1].set_xlabel(r"$\sigma$, arcsec")

        sprev = smin[np.argmin(scmin)]
        for iter_ in range(max_iter):
            c = np.polyfit(smin, scmin, 2)

            best_sig = -c[1] / 2 / c[0]
            if c[0] < 0:
                smin = np.abs(smin)
                c = np.polyfit(smin, scmin, 2)

                best_sig = -c[1] / 2 / c[0]

            xpsf = evaluate_model_psf(
                data,
                [best_shift, best_sig],
                do_optimal_extraction=True,
                simple_psf=simple_psf,
                psf_data=psf_data,
                extended_psf_data=extended_psf_data,
                **kwargs,
            )

            k = np.argmax(scmin)
            smin[k] = best_sig
            scmin[k] = data["chi2"]
            axes[-1].plot(smin, scmin, marker=".", alpha=0.7)

            ds = best_sig - sprev
            # print("sig refine", iter_, best_sig, sprev, ds, c[0])
            msg = f"({iter_})    sig {best_sig:6.3f}  {scmin[k]:.1f}  ds={ds:>9.2e}  c0={c[0]:.1e}"
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

            sprev = best_sig

            if np.abs(ds) < tol:
                break

        if cmin is not None:
            min_y = np.minimum(cmin.min(), scmin.min())
            max_y = np.minimum(scmin.max(), cmin.min()) + 1000
        else:
            min_y = cmin.min()
            max_y = min_y + 1000

        axes[-1].set_ylim(min_y - 100, max_y + 5000)

    else:
        best_sig = theta[1]

    if fig is not None:
        fig.tight_layout(pad=1)
        for ax in axes:
            ax.grid()

    best_theta = [best_shift, best_sig]

    xpsf = evaluate_model_psf(
        data,
        best_theta,
        do_optimal_extraction=True,
        simple_psf=simple_psf,
        psf_data=psf_data,
        extended_psf_data=extended_psf_data,
        **kwargs,
    )

    return best_theta


def decompose_psf(data):
    """
    Decompose PSF into multiple Gaussian components
    """
    import matplotlib.pyplot as plt
    import eazy.templates
    import msaexp.spectrum

    from importlib import reload
    import lrs_pipeline

    reload(lrs_pipeline)
    res = lrs_pipeline.full_lrs_pipeline(
        prog=6620,
        obs=1,
        resample_kwargs={"sn_threshold": 5},
        prof_kwargs={
            "optimize": True,
            "weight_type": "rnoise",
            "err_data": "err",
        },
        do_query=True,
        skip_preprocessing=True,
        nod_ext="bkg",
        sys_err=0.02,
        prefix="jw06620-obs13",
        file_query="jw0662000[13]",
    )

    data = res["data"]

    data_path = msautils.module_data_path()
    psf_tab = utils.read_catalog(
        os.path.join(data_path, "lrs", "lrs_psf_data_jw06620-obs13.fits")
    )

    trace_offset = data["trace_offset"]

    x0 = np.zeros(2)

    theta = simple_trace_fit(
        data,
        theta=x0,
        simple_psf=False,
        offset_grid=np.linspace(-1.8, 1.8, 8) * 0.001,
        tol=1.0e-4,
    )
    x0 = theta * 1

    psf_mu = np.interp(data["wave"], psf_tab["wave"], psf_tab["x0"])
    psf_width = np.interp(data["wave"], psf_tab["wave"], psf_tab["sigma"])

    mu = psf_mu + trace_offset / CROSS_PSCALE

    sig = np.sqrt(psf_width**2 + (0.0 / CROSS_PSCALE) ** 2)

    prof = resample_numba.pixel_integrated_gaussian_numba(
        data["dx"] / CROSS_PSCALE, mu, sig, dx=1.0
    )

    y = np.interp(data["wave"], data["wbin"], data["spec"])
    wsub = (data["wave"] > 8) & (data["wave"] < 12)

    plt.scatter(
        (data["dx"])[wsub],
        (data["data"] / y)[wsub],
        alpha=0.1,
        c=data["wave"][wsub],
    )

    plt.scatter((data["dx"])[wsub], prof[wsub], alpha=0.1, c="r")

    profs = []
    fs = [0.5, 0.75, 1, 1.5, 2, 3, 4]

    # fs = np.arange(0.5, 4.1, 0.5).tolist()
    fs = np.hstack(
        [np.arange(0.5, 2, 0.25), [2, 2.5]]
    )  # , np.arange(3, 4.1, 1)]).tolist()

    for f in fs:
        prof = resample_numba.pixel_integrated_gaussian_numba(
            data["dx"] / CROSS_PSCALE,
            mu,
            sig * f,
            # sig**0 + f,
            dx=1.0,
        )
        profs.append(prof)

    # theta = simple_trace_fit(
    #     data,
    #     theta=x0,
    #     simple_psf=False,
    #     offset_grid=np.linspace(-1.8, 1.8, 8) * 0.001,
    #     tol=1.0e-4,
    # )
    # x0 = theta * 1
    # y = np.interp(data["wave"], data["wbin"], data["spec"])
    #
    # profs = []
    # fs = np.linspace(-8, 8, 21)
    # for f in fs:
    #     prof = resample_numba.pixel_integrated_gaussian_numba(
    #         data["dx"] / CROSS_PSCALE, mu + f, sig * 0.5, dx=1.0
    #     )
    #     profs.append(prof)

    profs = np.array(profs)

    wsub = (data["wave"] > 8) & (data["wave"] < 10)

    skip = 32
    step = 0.25

    y = np.interp(data["wave"], data["wbin"], data["spec"])

    wmin = 5.0
    sn = data["spec"] / np.sqrt(data["spec_var"]) * (data["wbin"] > wmin)
    cumul_sn = np.cumsum(sn**2)
    cumul_sn /= np.nanmax(cumul_sn)
    sn_steps = np.interp(np.linspace(0, 1, 33), cumul_sn, data["wbin"])
    nstep = len(sn_steps)

    i = -1

    coeffs = []
    wcoeffs = []

    yx = data["data"] / data["err"]

    while i < (nstep - 2):
        i += 1

        # wsub = data["wave"] > data["wbin"][int(i * skip)]
        # wsub &= data["wave"] < data["wbin"][int((i + 1) * skip)]
        wsub = data["wave"] > sn_steps[i]
        wsub &= data["wave"] < sn_steps[i + 1]

        wsub &= np.abs(data["dx"]) < 1

        wsub &= np.isfinite(yx) & (data["wave"] > wmin)

        if wsub.sum() == 0:
            continue

        c = np.linalg.lstsq(
            (profs * y / data["err"])[:, wsub].T, yx[wsub], rcond=None
        )

        coeffs.append(c[0])
        wcoeffs.append(np.median(data["wave"][wsub]))

    coeffs = np.array(coeffs)
    wcoeffs = np.array(wcoeffs)
    csum = coeffs.sum(axis=1)

    y *= np.interp(data["wave"], wcoeffs, csum)

    coeffs = (coeffs.T / csum).T

    if 0:
        ctab = utils.GTable()
        ctab["coeffs"] = coeffs
        ctab["wcoeffs"] = wcoeffs
        ctab["psf_scale"] = np.array(fs)[None, :]
        ctab.write("lrs_psf_data_extended.fits", overwrite=True)

    nc = coeffs.shape[1]
    cw = np.array(
        [np.interp(data["wave"], wcoeffs, coeffs[:, i]) for i in range(nc)]
    )

    full_model = (profs * cw).sum(axis=0)

    # calspec source
    data["profile"] = full_model
    optimal_extraction(data)
    hdu, fig = process_2d_products(
        data,
        prefix=None,
        calspec_file="calspec_correction_ext.fits",
        show_2d_corr=True,
        vmax=10,
    )
    sp = msaexp.spectrum.SpectrumSampler(hdu)

    std = utils.read_catalog("bd60d1753_mod_006.fits")
    stemp = eazy.templates.Template(
        arrays=(std["WAVELENGTH"].value, std["FLUX"].value * 1.0e29)
    )

    resamp = sp.resample_eazy_template(stemp, z=0, fnu=True, scale_disp=1.0)
    cal_corr = data["spec"] / resamp

    calspec_corr = utils.GTable()
    calspec_corr["wave"] = data["wbin"]
    calspec_corr["corr"] = cal_corr * (data["wbin"] > 5)
    calspec_corr.write("calspec_correction_ext_v2.fits", overwrite=True)

    # calspec_corr.write("calspec_correction_ext.fits", overwrite=True) # with extended psf

    ###### Check P330E
    reload(lrs_pipeline)
    res = lrs_pipeline.full_lrs_pipeline(
        prog=1538,
        obs=21,
        resample_kwargs={"sn_threshold": 5},
        prof_kwargs={
            "optimize": True,
            "weight_type": "rnoise",
            "err_data": "err",
        },
        do_query=True,
        skip_preprocessing=True,
        nod_ext="bkg",
        sys_err=0.02,
        # prefix="jw06620-obs13",
        # file_query="jw0662000[13]"
    )
    data = res["data"]

    std = utils.read_catalog(
        "/Users/gbrammer/Research/JWST/Projects/NIRSpec/cal-iras05248/Wavecal2024/p330e_mod_008.fits"
    )
    stemp = eazy.templates.Template(
        arrays=(std["WAVELENGTH"].value, std["FLUX"].value * 1.0e29)
    )

    hdu, fig = process_2d_products(
        data,
        prefix=None,
        calspec_file="calspec_correction_ext_v2.fits",
        show_2d_corr=True,
        vmax=10,
    )
    sp = msaexp.spectrum.SpectrumSampler(hdu)
    resamp = sp.resample_eazy_template(stemp, z=0, fnu=True, scale_disp=1.0)

    ### Check another 1753 visit
    reload(lrs_pipeline)
    res = lrs_pipeline.full_lrs_pipeline(
        prog=1033,
        obs=2,
        resample_kwargs={"sn_threshold": 5},
        prof_kwargs={
            "optimize": True,
            "weight_type": "rnoise",
            "err_data": "err",
        },
        do_query=True,
        skip_preprocessing=True,
        nod_ext="bkg",
        sys_err=0.02,
        # prefix="jw06620-obs13",
        # file_query="jw0662000[13]"
    )
    data = res["data"]

    std = utils.read_catalog("bd60d1753_mod_006.fits")
    stemp = eazy.templates.Template(
        arrays=(std["WAVELENGTH"].value, std["FLUX"].value * 1.0e29)
    )

    hdu, fig = process_2d_products(
        data,
        prefix=None,
        calspec_file="calspec_correction_ext_v2.fits",
        show_2d_corr=True,
        vmax=10,
    )
    sp = msaexp.spectrum.SpectrumSampler(hdu)
    resamp = sp.resample_eazy_template(stemp, z=0, fnu=True, scale_disp=1.0)

    ###############

    wsub = data["wave"] > data["wbin"][int(i * skip)]
    wsub &= data["wave"] < data["wbin"][int((i + 1) * skip)]

    wsub &= np.abs(data["dx"]) < 1

    wsub &= np.isfinite(yx)
    c = np.linalg.lstsq(
        (profs * y / data["err"])[:, wsub].T, yx[wsub], rcond=None
    )

    m = profs[:, wsub].T.dot(c[0])

    plt.scatter(
        (data["dx"])[wsub],
        (data["data"] / y)[wsub],
        alpha=0.1,
        c=data["wave"][wsub],
    )

    plt.scatter((data["dx"])[wsub], m, alpha=0.1, color="r")
    plt.scatter(
        (data["dx"])[wsub], full_model[wsub], alpha=0.1, color="magenta"
    )


####### Tools for slitless extractions
def load_rate_files(files, write_bkg_files=True):
    """
    Load rate files, take diffs to remove background

    Parameters
    ----------
    files : list
        List of ``rate.fits`` or ``rate_bkg.fits`` files. Generally these
        should just be files from a single visit, as differences are taken
        between files at different ``x_offset`` positions to remove the
        background and images from even slightly different epochs generally
        don't subtract perfectly.

    write_bkg_files : bool
        Save ``rate_bkg.fits`` files of the difference products for fast
        reproduction later.

    Returns
    -------
    result : dict
        Loaded image data with keys:
          - ``files`` = list of the original files
          - ``dms`` = list of ``jwst.datamodels.ImageModel`` objects
          - ``sky_arrays`` = ndarray of the sky data
          - ``sky_var`` = ndarray of the sky variance
          - ``wcs_list`` = list of ``astropy.wcs.WCS`` objects for each exposure


    """
    from jwst.assign_wcs import AssignWcsStep
    from grizli import jwst_utils

    dms = [jwst.datamodels.open(file) for file in files]
    dm_data = np.array([dm.data for dm in dms])
    N = len(dms)
    sky_arrays = []
    sky_var = []

    for i in range(N):
        sky = []
        skyv = []
        xi = np.round(dms[i].meta.dither.x_offset * 10)
        for j in range(N):
            if i == j:
                continue

            xj = np.round(dms[j].meta.dither.x_offset * 10)
            if xi != xj:
                sky.append(dms[j].data)
                skyv.append(dms[j].err ** 2)

        utils.log_comment(
            utils.LOGFILE,
            f"i={i}  offset=({dms[i].meta.dither.x_offset:6.2f}, {dms[i].meta.dither.y_offset:6.2f}) (xoff={xi/10:5.1f})  nsky={len(sky)}",
            verbose=True,
        )

        skyn = np.isfinite(sky).sum(axis=0)
        sky_arrays.append(np.nansum(sky, axis=0) / skyn)
        sky_var.append(np.nansum(skyv, axis=0) / skyn**2)

    sky_arrays = np.array(sky_arrays)
    sky_var = np.array(sky_var)

    if "bkg" in files[0]:
        sky_arrays *= 0.0
        sky_var *= 0.0
        # print("background already removed")

    wcs_list = []
    for i, dm in enumerate(dms):
        dm.meta.exposure.type = "mir_image".upper()

        dmc = dm.copy()
        dmc.data -= sky_arrays[i]
        dmc.err = np.sqrt(dmc.err**2 + sky_var[i])

        bkg_file = files[i].replace("_rate", "_rate_bkg")

        dmc.to_fits(bkg_file, overwrite=True)

        dmw = jwst_utils.img_with_wcs(bkg_file, overwrite=True, verbose=False)

        with pyfits.open(bkg_file) as im:
            wcs_list.append(pywcs.WCS(im[1].header, relax=True))

        if not write_bkg_files:
            os.remove(bkg_file)

    for w in wcs_list:
        w.pscale = utils.get_wcs_pscale(w)

    result = {
        "files": files,
        "dms": dms,
        "sky_arrays": sky_arrays,
        "sky_var": sky_var,
        "wcs_list": wcs_list,
    }

    return result


MIRI_APERTURE_PADDING = {
    "MIRIM_SLIT": (1.1, 1.2),
    "MIRIM_ILLUM": (1., 1.01),
    "MIRIM_CORON1140": (1.0, 1.04),
    "MIRIM_CORON1550": (1.0, 1.04),
    "MIRIM_CORON1065": (1.0, 1.04),
    "MIRIM_CORONLYOT": (1.0, 1.04),
}


def miri_illum_window(wcs=None, window_pad=MIRI_APERTURE_PADDING, **kwargs):
    """
    Regions of illuminated portions of the MIRI detector

    Parameters
    ----------
    wcs : None, `astropy.wcs.WCS`
        Optional exposure WCS.  If provided, output ``regions`` will be in
        sky coordinates

    window_pad : dict
        Dilate subregion factors (defaults in
        `msaexp.cloud.lrs.MIRI_APERTURE_PADDING`).

    Returns
    -------
    regions : list
        List of `sregion.SRegion` objects for the ``[MIRIM_ILLUM,
        MIRIM_CORON1140, MIRIM_CORON1550, MIRIM_CORON1065, MIRIM_CORONLYOT``
        apertures

    """
    if 0:
        # Generate baseline regions
        import pysiaf
        import matplotlib.pyplot as plt

        mir = pysiaf.Siaf('MIRI')
        aper = mir["MIRIM_ILLUM"]

        names = ["MIRIM_SLIT", "MIRIM_ILLUM", "MIRIM_CORON1140", "MIRIM_CORON1550", "MIRIM_CORON1065", "MIRIM_CORONLYOT"]

        ref_aper = mir["MIRIM_FULL"]
        for aper in names:
            ap = mir[aper]
            tel = ap.corners("tel", rederive=False)

            try:
                det = ap.corners('det', rederive=False)
            except:
                det = ref_aper.tel_to_det(*tel)
                
            if 1:
                pl = plt.plot(*det, label=aper, alpha=0.3)
        
                sr = utils.SRegion(np.array(det), label=aper, wrap=False, pad=1)
                c = pl[0].get_color()
                sr.add_patch_to_axis(plt.gca(), fc='None', ec=c, alpha=0.1, label=aper)
                print(sr.region[0])
                
            else:
                pl = plt.plot(*tel, label=aper, alpha=0.5)

    regions = []

    polystr = """
polygon(304.767591,298.377327,304.764346,303.026663,347.490111,303.032316,347.482088,298.376952) #  text={MIRIM_SLIT}
polygon(359.497281,0.531945,1027.492269,0.563006,1027.505943,1024.498642,359.502113,1024.488881) #  text={MIRIM_ILLUM}
polygon(11.482517,247.487555,227.504741,247.511862,227.511046,463.514223,11.502512,463.492944) #  text={MIRIM_CORON1140}
polygon(11.495304,468.484722,227.503744,468.506583,227.505201,684.508954,11.506745,684.499635) #  text={MIRIM_CORON1550}
polygon(11.496695,24.587054,227.509750,24.544253,227.510050,240.505483,11.487780,240.481234) #  text={MIRIM_CORON1065}
polygon(5.498274,748.489323,277.496186,748.497263,277.499857,1020.486809,5.487143,1020.495679) #  text={MIRIM_CORONLYOT}
        """

    for poly in polystr.strip().split("\n"):

        lspl = poly.split()
        label = lspl[-1][6:-1]

        if label in window_pad:
            pad_i = window_pad[label]
        else:
            pad_i = 1.

        sr = utils.SRegion(lspl[0], wrap=False, label=label, pad=pad_i)

        if wcs is not None:
            ra, dec = wcs.all_pix2world(*sr.xy[0].T, 0)
            sr = utils.SRegion(np.array([ra, dec]), wrap=False, label=label)

        regions.append(sr)

    return regions


def rate_group_regions(
    rate_groups,
    root="group",
    separate_region_files=False,
    write_file=True,
    make_figure=True,
    figsize=(8, 8),
    **kwargs,
):
    """
    Make region file(s) for the illuminated portions of the loaded exposures

    Parameters
    ----------
    rate_groups : dict
        Output from `~msaexp.cloud.lrs.load_rate_files`

    root : str
        String prefix for output products, i.e., ``{root}_miri_lrs.reg``.

    separate_region_files : bool
        Make separate region files for each group

    make_figure : bool
        Make a summary figure

    figsize : (float, float)
        Figure size

    kwargs : dict
        Keyword arguments passed through to `msaexp.cloud.lrs.miri_illum_window`

    Returns
    -------
    sregions : list
        List of `sregion.SRegion` objects

    fig : figure or None
        Output figure object.  Also saved to ``{root}_miri_lrs.png``.

    """

    import matplotlib.pyplot as plt

    sregions = []

    group_pa = {}

    for group in rate_groups:

        shap = None
        slit = None

        for i, wcs in enumerate(rate_groups[group]["wcs_list"]):
            group_pa[group] = np.arctan2(*wcs.wcs.cd[0, :])

            wcs_regions = miri_illum_window(wcs=wcs, **kwargs)

            for j, sr in enumerate(wcs_regions):

                if j > 0:
                    if shap is None:
                        shap = sr.shapely[0]
                    else:
                        shap = shap.union(sr.shapely[0])
                else:
                    if slit is None:
                        slit = sr.shapely[0]
                    else:
                        slit = slit.union(sr.shapely[0])

        sr = utils.SRegion(shap)
        sr.label = group

        srs = utils.SRegion(slit)
        srs.label = group

        sregions.append(sr)
        sregions.append(srs)

        if separate_region_files:
            with open(f"{group}_miri_lrs.reg", "w") as fp:
                fp.write("icrs\n")
                for reg in sr.region:
                    fp.write(reg + "\n")
                for reg in srs.region:
                    fp.write(reg + "\n")

    if write_file:
        with open(f"{root}_miri_lrs.reg", "w") as fp:
            fp.write("icrs\n")
            for sr in sregions:
                for reg in sr.region:
                    fp.write(reg + "\n")

    if make_figure:

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        label_colors = {}

        for sr in sregions:
            # sr = utils.SRegion(reg.split("#")[0])

            is_slit = sr.sky_area("arcsec2")[0].value < 10

            group = sr.label

            if group in label_colors:
                color = label_colors[group]
            else:
                color = None
            
            if is_slit:

                pl = ax.plot(*sr.xy[0].T, alpha=0.0, color=color)
                color = pl[0].get_color()

                sr.add_patch_to_axis(
                    ax,
                    fc=color,
                    # fc='None',
                    ec=color,
                    alpha=0.5,
                    zorder=1000,
                    label=(group if group not in label_colors else None),
                    # hatch='||||||',
                )
                xy = sr.centroid[0]

                if group not in label_colors:
                    label_colors[group] = color

                    if 0:
                        ax.text(
                            xy[0],
                            xy[1] + 2.0 / 3600,
                            group,
                            color=color,
                            fontsize=7,
                            ha="center",
                            va="bottom",
                            zorder=1001,
                            rotation=(group_pa[group] / np.pi * 180 + 90),
                            rotation_mode="anchor",
                        )
            else:
                for xy in sr.xy:

                    pl = ax.plot(
                        *xy.T, alpha=0.5, color=color,
                        label=(group if group not in label_colors else None)
                    )

                    color = pl[0].get_color()

                    if group not in label_colors:
                        label_colors[group] = color
                    
        ax.set_xlim(*ax.get_xlim()[::-1])
        ax.grid()
        ax.legend(loc="lower right", fontsize=8)

        yl = ax.get_ylim()
        ax.set_aspect(1.0 / np.cos(yl[0] / 180 * np.pi))
        ax.set_title(root)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec.")

        fig.tight_layout(pad=1)

        fig.savefig(f"{root}_miri_lrs.png")

    else:
        fig = None

    return sregions, fig


def rate_group_mosaic(
    rate_groups,
    root="jw09615",
    output_path="./",
    make_separate=True,
    make_combined=True,
    pixel_scale=CROSS_PSCALE,
    kernel="square",
    pixfrac=0.5,
    verbose=True,
):
    """
    Make drizzled mosaics of group exposures

    Parameters
    ----------
    rate_groups : dict
        Output from `~msaexp.cloud.lrs.load_rate_files`

    root : str
        String prefix for output products, i.e., ``{root}_miri_lrs.reg``.

    output_path : str
        Location for output files.  If None, just return arrays.

    make_separate : bool
        Make drizzled combinations separated for each group with their own
        pixel grid.

    make_combined : bool
        Make combination of exposures in all groups

    pixel_scale, kernel, pixfrac : float, str, float
        Drizzle parameters

    Returns
    -------
    products : dict
        Dictionary with keys of the drizzled groups and values of the
        output from `grizli.utils.drizzle_array_groups`.
    
    """
    sci_list = []
    wcs_list = []
    wht_list = []

    products = {}

    for i, group in enumerate(rate_groups):

        wcs_list_i = rate_groups[group]["wcs_list"]
        sci_list_i = []
        wht_list_i = []

        utils.log_comment(
            utils.LOGFILE,
            f"\nGroup mosaic {i+1}/{len(rate_groups)} {group}  N={len(wcs_list_i)} files\n",
            verbose=verbose,
        )

        for dm, sky, skyv in zip(
            rate_groups[group]["dms"],
            rate_groups[group]["sky_arrays"],
            rate_groups[group]["sky_var"],
        ):
            s = dm.data - sky

            wht = 1.0 / (dm.err**2 + skyv) * np.isfinite(s)
            msk = (wht == 0) | (~np.isfinite(wht + s))
            s[msk] = 0
            wht[msk] = 0

            sci_list_i.append(s)
            wht_list_i.append(wht * 1)

        if make_separate:

            outh, outw = utils.make_maximal_wcs(
                wcs_list_i,
                pixel_scale=pixel_scale,
                pad=10,
                get_hdu=False,
                verbose=verbose > 1,
            )

            drz_ = utils.drizzle_array_groups(
                sci_list_i,
                wht_list_i,
                wcs_list_i,
                outputwcs=outw,
                kernel=kernel,
                pixfrac=pixfrac,
                verbose=verbose > 1,
            )

            if output_path is not None:
                out = os.path.join(output_path, f"{group}_miri_lrs_sci.fits")

                utils.log_comment(
                    utils.LOGFILE,
                    f"Write group mosaic {out}",
                    verbose=verbose,
                )

                pyfits.writeto(
                    out,
                    data=drz_[0],
                    header=outh,
                    output_verify="fix",
                    overwrite=True,
                )

                pyfits.writeto(
                    out.replace("_sci.fits", "_wht.fits"),
                    data=drz_[1],
                    header=outh,
                    output_verify="fix",
                    overwrite=True,
                )

            products[group] = drz_

        sci_list.extend(sci_list_i)
        wht_list.extend(wht_list_i)
        wcs_list.extend(wcs_list_i)

    if make_combined:
        utils.log_comment(
            utils.LOGFILE,
            f"\nCombined mosaic {root}  N={len(wcs_list)} files\n",
            verbose=verbose,
        )

        outh, outw = utils.make_maximal_wcs(
            wcs_list,
            pixel_scale=pixel_scale,
            pad=10,
            get_hdu=False,
            verbose=verbose > 1,
        )

        drz_ = utils.drizzle_array_groups(
            sci_list,
            wht_list,
            wcs_list,
            outputwcs=outw,
            kernel=kernel,
            pixfrac=pixfrac,
            verbose=verbose > 1,
        )

        if output_path is not None:
            out = os.path.join(output_path, f"{root}_miri_lrs_sci.fits")

            utils.log_comment(
                utils.LOGFILE,
                f"Write combined mosaic {out}",
                verbose=verbose,
            )

            pyfits.writeto(
                out,
                data=drz_[0],
                header=outh,
                output_verify="fix",
                overwrite=True
            )
        
            pyfits.writeto(
                out.replace("_sci.fits", "_wht.fits"),
                data=drz_[1],
                header=outh,
                output_verify="fix",
                overwrite=True,
            )

        products[root] = drz_

    return products


def lrs_slitless_cutout_offset(xpix, ypix):
    """
    Trace offset as a function of detector pixel position

    Parameters
    ----------
    xpix, ypix : float, float
        Detector coordinates

    Returns
    -------
    dx, dy : float, float
        Pixel offsets for cutout origin relative to ``(xpix, ypix)``

    """
    from astropy.modeling.models import Polynomial2D

    h = pyfits.Header.fromtextfile(
        os.path.join(
            msautils.module_data_path(), "lrs", "lrs_wfss_offset_coeffs.txt"
        )
    )

    if xpix < 250:
        prefix = "DXL"
    else:
        prefix = "DXR"

    dxpoly = Polynomial2D(degree=h[f"{prefix}deg"])

    dxpoly.parameters = np.array(
        [h[f"{prefix}{i}"] for i in range(h[f"{prefix}N"])]
    )

    prefix = "DYR"
    dypoly = Polynomial2D(degree=h[f"{prefix}deg"])
    dypoly.parameters = np.array(
        [h[f"{prefix}{i}"] for i in range(h[f"{prefix}N"])]
    )

    dx = dxpoly(xpix, ypix) - dxpoly(h["REFXPIX"], h["REFYPIX"])
    dy = dypoly(xpix, ypix) - dypoly(h["REFXPIX"], h["REFYPIX"])

    return dx, dy


def get_lrs_slitless_cutout(data, wcs, ra, dec, verbose=False, xoffset=None, pad=512, slit_model=None, check_window=True, **kwargs):
    """
    Get pixel cutouts from data array for a sky position

    Parameters
    ----------
    data : array-like
        MIRI image array

    wcs : `~astropy.wcs.WCS`
        Image WCS

    ra, dec : float
        Sky coordinates

    xoffset : None, float
        Manual x offset

    pad : int
        Array padding for sources near the edges

    slit_model : HDUList
        See `msaexp.cloud.lrs.load_lrs_slit_model`

    check_window : bool
        Check that requested ``(ra, dec)`` is within the detector window from
        ``msaexp.cloud.lrs.miri_illum_window``

    Returns
    -------
    cutout : dict
        Cutout data

    """

    if slit_model is None:
        slit_model = load_lrs_slit_model()

    h = slit_model[0].header
    
    if ra is not None:
        wcs_pix = np.squeeze(wcs.all_world2pix([ra], [dec], 0))

        window_region = ''

        if check_window:
            window_regions = miri_illum_window(wcs=wcs, **kwargs)
            in_window = False
            for reg in window_regions:
                in_window_i = reg.path[0].contains_point((ra, dec))
                if in_window_i:
                    in_window = True
                    window_region = reg.label
                    break

            if not in_window:
                return None

        rdx = np.array(wcs.all_pix2world([wcs_pix[0], wcs_pix[0] + 1], [wcs_pix[1], wcs_pix[1]], 0))
        local_pscale = np.sqrt(((np.diff(rdx, axis=1).T * np.array([np.cos(dec/180*np.pi), 1.]))**2).sum()) * 3600

        corner_dx, corner_dy = lrs_slitless_cutout_offset(*wcs_pix)
        if verbose:
            print(f'   corner offset:  {corner_dx:6.2f} {corner_dy:6.2f}')
        
        x_slice_offset = wcs_pix[0] - h['REFXPIX'] + corner_dx # + 1
        y_slice_offset = wcs_pix[1] - h['REFYPIX'] + corner_dy # + 1
        
    else:
        local_pscale = CROSS_PSCALE
        if xoffset is None:
            x_slice_offset = 0.0
            y_slice_offset = 0.0
        else:
            x_slice_offset = xoffset / CROSS_PSCALE
            y_slice_offset = 0.0
            
        
    x_slice_offset_int = int(np.floor(x_slice_offset))
    y_slice_offset_int = int(np.floor(y_slice_offset))

    if verbose:
        print(f'    pixel offset:  {x_slice_offset:6.2f} {y_slice_offset:6.2f}')
        
    slx = slice(h['SLX0'] + x_slice_offset_int + pad, h['SLX1'] + x_slice_offset_int + pad)
    sly = slice(h['SLY0'] + y_slice_offset_int + pad, h['SLY1'] + y_slice_offset_int + pad)
    dx = slit_model['DX'].data + (x_slice_offset - x_slice_offset_int) * local_pscale
    
    dcut = np.pad(data, pad, constant_values=0)[sly, slx]
    yp, xp = np.indices(dcut.shape)
    
    cutout = {
        'dx': dx,
        'data': dcut,
        'wave': slit_model['wave'].data,
        'slices': (slx, sly),
        'pad': pad,
        'yp': yp,
        'xp': xp,
        'ydet': yp + sly.start - pad,
        'xdet': xp + slx.start - pad,
        'ra': ra,
        'dec': dec,
        'local_pscale': local_pscale,
        'window_region': window_region,
    }

    return cutout


# ref_spec = msaexp.spectrum.SpectrumSampler("jw04533001-mds.miri-deep-survey-alma-10_lrs.spec.fits")

def cutout_pixel_table(ra, dec, rate_groups, slit_model=None, make_figure=False, vmax=1, gridsize=(128, 24), subtract_pedestal=False, sy=1.5, ref_spec=None, check_window=True, verbose=True, **kwargs):
    """
    Get full cutout pixel table
    """
    import matplotlib.pyplot as plt

    N = 0
    for group in rate_groups:
        N += len(rate_groups[group]["dms"])
        
    ptabs = []

    exp_i = -1

    all_files = []
    all_windows = []

    exp_group_idx = {}

    if slit_model is None:
        slit_model = load_lrs_slit_model()
    
    for ig, group in enumerate(rate_groups):
        grp = rate_groups[group]

        for i, dm in enumerate(grp["dms"]):
            exp_i += 1

            exp_group_idx[exp_i] = ig

            data = dm.data - grp["sky_arrays"][i]

            all_files.append(dm.meta.filename)
            
            wcs = grp["wcs_list"][i]

            print(f"{exp_i:>2} {all_files[-1]}")

            cut_kwargs = dict(
                slit_model=slit_model,
                xoffset=dm.meta.dither.x_offset,
                verbose=verbose,
                check_window=check_window,
                **kwargs,
            )

            try:
                # data
                ptab = get_lrs_slitless_cutout(data, wcs, ra, dec, **cut_kwargs)
            except:
                all_windows.append('N/A')
                continue

            if ptab is None:
                all_windows.append('N/A')
                continue

            try:
                if ptab["data"].max() == 0:
                    print("skip")
                    all_windows.append('N/A')
                    continue
            except:
                print("skip")
                all_windows.append('N/A')
                continue
                
            ptab["x_offset"] = 0.0
            ptab["meta"] = dict(dm.meta.instance)
            ptab["to_ujy"] = 1.0

            cut_kwargs["verbose"] = False

            _dq = get_lrs_slitless_cutout(dm.dq, wcs, ra, dec, **cut_kwargs)
            
            ptab["dq"] = _dq["data"]

            _var = get_lrs_slitless_cutout(
                dm.err**2 + grp["sky_var"][i],
                wcs, ra, dec,
                **cut_kwargs,
            )

            ptab["err"] = np.sqrt(_var["data"])

            _var_rn = get_lrs_slitless_cutout(
                dm.var_rnoise,
                wcs, ra, dec,
                **cut_kwargs
            )

            ptab["var_rnoise"] = _var_rn["data"]

            low = ptab["data"] / ptab["err"] < -3

            low = nd.binary_dilation(
                nd.binary_erosion(low, iterations=1),
                structure=np.ones((64, 8), dtype=bool),
            )

            ptab["data"][low] = np.nan
            ptab["data"][ptab["data"] == 0] = np.nan
            
            ptab["exposure"] = np.full(ptab["data"].shape, exp_i)

            if subtract_pedestal:
                pedestal_subarea = np.abs(ptab["dx"]) > 0.8
                pedestal_subarea &= (ptab["wave"] > 6.5) & (ptab["wave"] < 11)
                pedestal_subarea &= (ptab['data'] != 0)
                
                if pedestal_subarea.sum() > 0:
                    pedestal_value = np.nanmedian(
                        ptab["data"][pedestal_subarea]
                    )
                else:
                    pedestal_value = 0.
            else:
                pedestal_value = 0.
                
            ptab["data"] -= pedestal_value
            ptab["meta"]["pedestal"] = pedestal_value

            ptab["file"] = dm.meta.filename
            
            ok = np.isfinite(ptab['data'] + ptab['err'])
            ok &= (ptab['dq'] & 1) == 0
            ok &= (ptab['data'] > -10 * ptab['err'])

            okw = ptab["wave"] < 12

            if (ok & okw).sum() == 0:
                print("  Nvalid = 0, skip")
                all_windows.append('N/A')
                continue
                
            all_windows.append(ptab["window_region"])
            
            ptabs.append(ptab)

    ### Merge pixel tables
    if len(ptabs) == 0:
        return None, None
        
    data = {
        'yp': [],
        'xp': [],
        'ydet': [],
        'xdet': [],
        'exposure': [],
        'dx': [],
        'wave': [],
        'data': [],
        'err': [],
        'var_rnoise': [],
    }
    
    for i in range(len(ptabs)): #[1,3]:
        ptab = ptabs[i]
        ok = np.isfinite(ptab['data'] + ptab['err']) & ((ptab['dq'] & 1) == 0)
        # trim = (ptab['xdet'] > 280) & (ptab["xdet"] < 358)
        # ok &= ~trim
        ok &= ~(ptab['data'] == 0)
        
        if 1:
            ok &= (ptab['data'] > -10*ptab['err'])
    
        if 0:
            bkg = ptab['bkg']
        else:
            bkg = 0
        
        data['yp'].append(ptab['yp'][ok])
        data['xp'].append(ptab['xp'][ok])
        data['ydet'].append(ptab['ydet'][ok])
        data['xdet'].append(ptab['xdet'][ok])
        data['exposure'].append(ptab['exposure'][ok])
        data['dx'].append(ptab['dx'][ok] - ptab['x_offset'])
        data['wave'].append(ptab['wave'][ok])
        data['data'].append((ptab['data'] - bkg)[ok])
        data['err'].append(ptab['err'][ok])
        data['var_rnoise'].append(ptab['var_rnoise'][ok])

    for k in data:
        if k not in ['file']:
            data[k] = np.hstack(data[k])

    if make_figure:
        une = utils.Unique(data["exposure"], verbose=False)
        N = une.N
                
        fig, axes = plt.subplots(N + 1, 1, figsize=(10, (sy * (N+1))), sharex=True, sharey=True)

        kws = dict(vmin=-vmax, vmax=vmax, alpha=0.8, cmap="magma", gridsize=gridsize, reduce_C_function=np.nanmedian)

        for ax_i, exp_i in enumerate(une.values):
            sub = une[exp_i]                 
            axes[ax_i].hexbin(data["wave"][sub], data["dx"][sub], C=data["data"][sub], **kws)

            axes[ax_i].text(
                0.98, 0.08,
                f"group={exp_group_idx[exp_i]} exp={exp_i}  {all_files[exp_i]}",
                fontsize=8,
                ha="right", va="bottom", transform=axes[ax_i].transAxes,
                bbox={"fc": "w", "ec": "None", "alpha": 0.9}
            )

            if all_windows[exp_i] == "MIRIM_SLIT":
                axes[ax_i].text(
                    0.02, 0.08,
                    f"LRS Slit",
                    fontsize=8,
                    ha="left", va="bottom", transform=axes[ax_i].transAxes,
                    bbox={"fc": "w", "ec": "None", "alpha": 0.9}
                )

        axes[N].hexbin(data["wave"], data["dx"], C=data["data"], **kws)
        
        axes[N].text(
            0.98, 0.08,
            f"Stack  {ra:.6f} {dec:.6f}",
            fontsize=8,
            ha="right", va="bottom", transform=axes[N].transAxes,
            bbox={"fc": "w", "ec": "None", "alpha": 0.9}
        )
        
        for ax in axes:
            ax.grid()
            ax.set_ylim(-2.2, 2.2)

        fig.tight_layout(pad=0.5)

    else:
        fig = None

    data["meta"] = ptabs[0]["meta"]
    
    data["meta"]["files"] = all_files
    data["meta"]["windows"] = all_windows

    # data["meta"]["sys_err"] = sys_err
    data["meta"]["x_offset"] = [ptab["x_offset"] for ptab in ptabs]

    data['data'][data['wave'] < 4.92] = np.nan

    pos_key = f"{ra:.6f}_{dec:.6f}"
    data["meta"]["target"]["pos_key"] = pos_key
    data["meta"]["target"]["proposer_name"] = pos_key

    if ref_spec is None:
        ref_spec = load_lrs_reference_spectrum()

    set_bin_arrays(data, wbin=ref_spec["wave"])
    
    return fig, data


def slitless_data_groups(data, exposure_groups, **kwargs):
    """
    Set ``group`` key in ``data`` based on exposure groups in ``config``

    Parameters
    ----------
    data : dict
        Pixel table data

    exposure_groups : dict
        Groups of exposure files

    Returns
    -------
    group_names : list
        List of group names from ``exposure_groups`` keys

    ung : `unik.Unique`
        Unique helper for ``data["group"]``

    """
    data["group"] = (data["wave"] * 0).astype(int)

    group_names = []

    une = utils.Unique(data["exposure"], verbose=False)

    for ig, group in enumerate(exposure_groups):
        for ifile, file in enumerate(exposure_groups[group]):
            if file in data["meta"]["files"]:
                j = data["meta"]["files"].index(file)
                data["group"][une[j]] = ig

                if group not in group_names:
                    group_names.append(group)

    print("\n     Groups\n     ======")
    ung = utils.Unique(np.array(group_names)[data["group"]], **kwargs)

    data["in_slit"] = np.isin(
        np.array(data["meta"]["windows"])[data["exposure"]],
        ["MIRIM_SLIT"]
    )

    in_slit = data["in_slit"].sum()
    if (in_slit > 0) & (in_slit != len(data["in_slit"])):
        msg = '\nWarning: Cutout appears to include both slit and slitless sources!\n' 
        utils.log_comment(
            utils.LOGFILE,
            msg,
            verbose=True,
        )

    data["meta"]["group_names"] = group_names

    return group_names, ung


def slitless_trace_by_group(data, initial_shift=[0., 0.2], initial_size=0.1, min_wave=6, max_wave=11.5, max_dx=1.5, verbose=2, reset_initial=True, fix_first_size=True, fix_first_shift=False, show_each_2d=False, **kwargs):
    """
    Fit trace for each exposure group separately
    """
    ung = utils.Unique(
        np.array(data["meta"]["group_names"])[data["group"]],
        verbose=False
    )

    group_theta = {}
    for j, ig in enumerate(ung.values):

        subset = ung[ig] & (data["wave"] > min_wave)
        subset &= (data["wave"] < max_wave) & (np.abs(data["dx"]) < max_dx)
        subset &= np.isfinite(data["data"])

        if verbose > 0:
            print(f"\nExposure group {ig}  ({j+1} / {ung.N})  npix={subset.sum()}")

        if subset.sum() < 16:
            print("  (skip)")
            continue

        data_i = get_data_slice(data, subset)

        group_theta[ig] = fit_shift_and_size(
            data_i,
            initial_shift=initial_shift,
            initial_size=initial_size,
            verbose=(verbose & 2),
            **kwargs
        )

        group_theta[ig][-1] = np.maximum(group_theta[ig][-1], 0.01)
        
        if (verbose & 1):
            print(f"    result = {group_theta[ig]}")

        if reset_initial:
            initial_shift = group_theta[ig][:-1] * 1
            initial_size = group_theta[ig][-1] * 1

        if show_each_2d:
            _ = process_2d_products(data_i, recenter=True)

        if fix_first_size & (j == 0):
            kwargs["fit_size"] = False
            initial_size = group_theta[ig][-1] * 1

        if fix_first_shift & (j == 0):
            kwargs["fit_shift"] = False
            initial_shift = group_theta[ig][:-1] * 1

    set_full_profile(data, group_theta)
    
    return ung, group_theta


def set_full_profile(data, group_theta):
    """
    Set profiles for all groups with different trace fits
    """
    # ung = utils.Unique(data["group"], verbose=False)
    ung = utils.Unique(
        np.array(data["meta"]["group_names"])[data["group"]], verbose=False
    )

    # Set full profile arrays
    profile = data["wave"] * 0.
    psf_trace = data["wave"] * 0.
    
    for ig in group_theta:
        subset = ung[ig]
        data_i = get_data_slice(data, subset)
        _ = evaluate_model_psf(data_i, group_theta[ig], do_optimal_extraction=True)
        
        profile[subset] = data_i["profile"] * 1
        psf_trace[subset] = data_i["psf_trace"] * 1

    _ = evaluate_model_psf(data, group_theta[ig], do_optimal_extraction=True)
    
    data["profile"] = profile
    data["psf_trace"] = psf_trace


def cafe_pah_templates():
    """
    Generate fitting templates for PAH lines listed by CAFE
    ([Diaz-Santos+2025](https://github.com/GOALS-survey/CAFE))
    
    """
    cafe_pah_info = utils.read_catalog("""# WAVE0     GAMMA        PEAK              COMPLEX
# https://github.com/GOALS-survey/CAFE/blob/master/CAFE/tables/pah_template.txt
        3.289E+00   1.1E-02      0.300               0
        3.403E+00   0.4E-02      0.060               0.1
        3.49E+00   8.4E-02      0.01                 0.2
        5.24E+00   1.1E-02      0.06159884           1.1
        5.33E+00   0.8E-02      0.06159884           1.2
        5.70E+00   3.5E-02      0.07050930           2
        6.029E+00   9.45E-03      0.20                3
        6.2123E+00   1.5E-02      0.77360526          3
        6.2700E+00   4.1E-02      0.77360526          3
        6.69E+00   7.0E-02      0.09797205           4
        7.42E+00  1.26E-01      0.23897256           5
        7.60E+00   4.4E-02      0.89279697           5
        7.85E+00   5.3E-02      0.86156776           5
        8.33E+00   5.0E-02      0.25013986           6
        8.61E+00   3.9E-02      0.64566928           7
        11.02E+00   5.0E-03      0.20000000           9 
       11.242E+00   8.0E-03      1.29167980           9
       11.3572E+00   1.9E-02      1.04019180           9
       11.99E+00   4.5E-02      0.32429740           10
       12.62E+00   4.2E-02      0.84859132           11
       12.69E+00   1.3E-02      0.19136069           11
       13.48E+00   4.0E-02      0.21897772           12
#       14.04E+00   1.6E-02      0.01000000           13
       14.19E+00   2.5E-02      0.17583134           14
#       15.90E+00   2.0E-02      0.01376596           15
#       16.45E+00   1.4E-02      0.42942265           16
#       17.04E+00   6.5E-02      0.63617951           16
#      17.375E+00   1.2E-02      0.29235229           16
#       17.87E+00   1.6E-02      0.25877291           16
#       18.92E+00   1.9E-02      0.31397021           17
    """)

    import eazy.templates

    cafe_pah_info["FWHM"] = cafe_pah_info["GAMMA"] * cafe_pah_info["WAVE0"]

    templ_wave = np.arange(1, 15, 0.003) * 1.e4

    pah_templates = []

    for row in cafe_pah_info:
        utempl = utils.pah_line_template(
            templ_wave,
            center_um=row["WAVE0"],
            fwhm=row["FWHM"]
        )
        
        templ = eazy.templates.Template(
            arrays=(
                utempl.wave,
                utempl.flux / np.trapz(utempl.flux, utempl.wave) * 1.e9
            ),
            name="PAH-{WAVE0:.2f}".format(**row)
        )

        pah_templates.append(templ)

    return pah_templates


MID_IR_LINES = {
    "PaA": 1.87563,
    "PaB": 1.28217,
    "BrA": 4.05228,
    "BrB": 2.62588,
    "SiVII": 6.492,
    "ArII-6.99": 6.985,
    "NeVI": 7.652, 
    "ArIII-8.99": 8.991,
    "SIV": 10.5105,
}

def get_mir_lines(spec, z=0, orders=[1], velocity_sigma=100, scale_disp=1, lines=MID_IR_LINES, **kwargs):
    """
    Return some common mid-IR lines
    """
    wave_range = (
        spec["wave"][spec["valid"]].min(), spec["wave"][spec["valid"]].max()
    )

    line_models = []
    line_names = []
    
    for i, ln in enumerate(lines):
        lwi = lines[ln] * (1 + z)
        if (lwi < wave_range[0]) | (lwi > wave_range[1]):
            continue

        line_i = spec.fast_emission_line(
            lwi, orders=orders,
            velocity_sigma=velocity_sigma,
            scale_disp=scale_disp,
            **kwargs
        ) / spec["to_flam"] / 1.e4

        line_models.append(line_i)
        line_names.append(ln)

    return np.array(line_models), line_names


def fit_mir_templates(spec, z=0, pah_templates=None, line_models=None, continuum_models=None, fitter="nnls", velocity_sigma=100, scale_disp=1, plot_unit="flam", err_col="full_err", write=True, **kwargs):
    """
    Fit mid-IR templates to an LRS spectrum
    """
    from scipy.optimize import nnls

    if z is None:
        z = spec.meta["Z"]

    if pah_templates is None:
        pah_templates = cafe_pah_templates()

    pah_names = [t.name for t in pah_templates]

    if line_models is None:
        line_models, line_names = get_mir_lines(
            spec, z=z,
            velocity_sigma=velocity_sigma,
            scale_disp=scale_disp,
        )

    pah_models = np.array([
        spec.resample_eazy_template(
            templ, z=z, orders=[1],
            velocity_sigma=velocity_sigma,
            scale_disp=scale_disp,
        ) * (1 + z)
        for templ in pah_templates
    ])

    model_list = [pah_models, line_models]

    if continuum_models is not None:
        model_list.extend(continuum_models)

    A = np.vstack(model_list)

    Ax = A / spec[err_col]
    yx = spec["flux"] / spec[err_col]

    ok = np.isfinite(yx)

    if fitter == "nnls":
        lsq = nnls(Ax.T[ok,:], yx[ok])
    else:
        lsq = np.linalg.lstsq(Ax.T[ok,:], yx[ok], rcond=None)

    model = A.T.dot(lsq[0])
    chi = model / spec[err_col] - yx
    chi2 = (chi**2)[ok].sum()

    if plot_unit is not None:
        fig = spec.drizzled_hdu_figure(unit=plot_unit, z=z)

        ax = fig.axes[2]

        to_unit = spec["to_flam"]**(plot_unit == "flam")
        ax.plot(
            np.arange(spec.NSPEC),
            model * to_unit,
            color="magenta", label="PAH model",
            lw=3, alpha=0.3
        )

        _ = ax.plot(
            np.arange(spec.NSPEC),
            (A * to_unit).T * lsq[0],
            alpha=0.3
        )

        if write:
            fig.savefig(f"{spec.meta['SRCNAME']}_lrs.spec.{plot_unit}.png")

    else:
        fig = None

    template_names = pah_names + line_names
    line_fluxes = {
        name: lsq[0][i]*1.e-20 for i, name in enumerate(template_names)
    }

    result = {
        "lsq": lsq,
        "model": model,
        "chi2": chi2,
        "template_models": A,
        "template_names": template_names,
        "line_fluxes": line_fluxes,
        "fig": fig,
    }

    return result


def fit_lrs_redshift(spec, zlim=(0.1, 0.6), dz=0.002, continuum_models=None, pah_templates=None, velocity_sigma=100, scale_disp=1, fitter="nnls", zfig=None, make_plots=True, write=True, **kwargs):
    """
    Fit LRS redshift with PAH templates
    """
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    if pah_templates is None:
        pah_templates = cafe_pah_templates()

    zstep = utils.log_zgrid(zlim, dz)
    chi2 = zstep * 0.
    chi2_min = np.inf
    
    for zi, z in tqdm(enumerate(zstep)):
        result = fit_mir_templates(
            spec,
            z=z,
            pah_templates=pah_templates,
            continuum_models=continuum_models,
            fitter=fitter,
            velocity_sigma=velocity_sigma,
            scale_disp=scale_disp,
            plot_unit=None,
        )
    
        chi2[zi] = result["chi2"]

    zbest = zstep[np.argmin(chi2)]

    result = fit_mir_templates(
        spec,
        z=zbest,
        pah_templates=pah_templates,
        continuum_models=continuum_models,
        fitter=fitter,
        velocity_sigma=velocity_sigma,
        scale_disp=scale_disp,
        plot_unit=("flam" if (make_plots > 1) else None),
    )

    result["zstep"] = zstep
    result["chi2"] = chi2
    result["zbest"] = zbest

    spec.meta['Z'] = result["zbest"]
    for k in result['line_fluxes']:
        spec.meta[f'LINE-{k}'] = result["line_fluxes"][k]

    if write:
        with pyfits.open(
            f"{spec.meta['SRCNAME']}_lrs.spec.fits",
            mode="update"
        ) as im:
            for k in spec.meta:
                if k.lower().endswith("unit"):
                    im[1].header[k] = spec.meta[k].__repr__()
                else:
                    im[1].header[k] = spec.meta[k]

            im.flush()

    if make_plots:
        if zfig is None:
            zfig, ax = plt.subplots(1,1,figsize=(8,4))
            new_fig = True
        else:
            ax = zfig.axes[0]
            new_fig = False

        chi_plot = (chi2 - chi2.min()) / (chi2.min() / spec["valid"].sum())
        ax.plot(zstep, chi_plot, color="0.5", alpha=0.5)
        ax.set_ylim(-0.05 * chi_plot.max(), chi_plot.max())
        
        ylim = ax.get_ylim()
        
        ax.vlines(zbest, *ylim, color="magenta", linestyle="--", alpha=0.5)
        
        ax.text(
            zbest, np.interp(0.98, (0, 1), ylim), f"z = {zbest:.4f}",
            ha="center", va="top",
            bbox={"fc": "w", "ec": "None", "alpha": 0.8}
        )

        ax.text(
            0.02, 0.02,
            r"$\Delta z~/~(1 + z)$ = " + f"{dz:.0e}",
            ha="left", va="bottom",
            transform=ax.transAxes,
            bbox={"fc": "w", "ec": "None", "alpha": 0.8}
        )

        ax.text(
            0.98, 0.02,
            spec.meta["SRCNAME"],
            ha="right", va="bottom",
            transform=ax.transAxes,
            bbox={"fc": "w", "ec": "None", "alpha": 0.8}
        )
        
        ax.grid()
        ax.set_xlabel("z")
        ax.set_ylabel(r"$\Delta \chi^2$")
        ax.set_xlim(*zlim)

        if new_fig:
            zfig.tight_layout(pad=1)

        result["zfig"] = zfig
        
        if write:
            zfig.savefig(f"{spec.meta['SRCNAME']}_lrs.spec.zfit.png")
        
    return result


###########################
##### old below here
def deprecated_lrs_slitless_data_cutout(data, wcs, ra, dec, verbose=False, xoffset=None, pad=512, slit_model=None):
    """ """

    if slit_model is None:
        # slit_model = pyfits.open(
        #     os.path.join(msautils.module_data_path(), "lrs", "lrs_slit_model.fits")
        # )
        slit_model = load_lrs_slit_model()

    h = slit_model[0].header

    if ra is not None:
        wcs_pix = np.squeeze(wcs.all_world2pix([ra], [dec], 0))

        rdx = np.array(
            wcs.all_pix2world(
                [wcs_pix[0], wcs_pix[0] + 1], [wcs_pix[1], wcs_pix[1]], 0
            )
        )
        local_pscale = (
            np.sqrt(
                (
                    (
                        np.diff(rdx, axis=1).T
                        * np.array([np.cos(dec / 180 * np.pi), 1.0])
                    )
                    ** 2
                ).sum()
            )
            * 3600
        )

        corner_dx, corner_dy = lrs_slitless_cutout_offset(*wcs_pix)
        if verbose:
            print(f"   corner offset: {corner_dx:6.2f} {corner_dy:6.2f}")

        x_slice_offset = wcs_pix[0] - h["REFXPIX"] + corner_dx  # + 1
        y_slice_offset = wcs_pix[1] - h["REFYPIX"] + corner_dy  # + 1
    else:
        local_pscale = CROSS_PSCALE
        if xoffset is None:
            x_slice_offset = 0.0
            y_slice_offset = 0.0
        else:
            x_slice_offset = xoffset / CROSS_PSCALE
            y_slice_offset = 0.0

    x_slice_offset_int = int(np.floor(x_slice_offset))
    y_slice_offset_int = int(np.floor(y_slice_offset))

    if verbose:
        print(f"    pixel offset: {x_slice_offset:6.2f} {y_slice_offset:6.2f}")

    slx = slice(
        h["SLX0"] + x_slice_offset_int + pad,
        h["SLX1"] + x_slice_offset_int + pad,
    )
    sly = slice(
        h["SLY0"] + y_slice_offset_int + pad,
        h["SLY1"] + y_slice_offset_int + pad,
    )
    dx = (
        slit_model["DX"].data
        + (x_slice_offset - x_slice_offset_int) * local_pscale
    )

    dcut = np.pad(data, pad, constant_values=0)[sly, slx]
    yp, xp = np.indices(dcut.shape)

    cutout = {
        "dx": dx,
        "data": dcut,
        "wave": slit_model["wave"].data,
        "slices": (slx, sly),
        "yp": yp,
        "xp": xp,
        "ra": ra,
        "dec": dec,
        "local_pscale": local_pscale,
    }

    return cutout


# x_slice_offset, y_slice_offset

