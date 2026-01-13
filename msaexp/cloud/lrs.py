"""
Scripts for reducting MIR LRS data with a high-quality PSF model
"""

import os
import glob
import yaml

import numpy as np

import grizli.jwst_utils
from grizli import utils
from grizli import prep

import jwst.datamodels
import mastquery.jwst

from grizli.aws import visit_processor, db

from ..slit_combine import pseudo_drizzle
from .. import utils as msautils
from .. import resample_numba

import astropy.wcs as pywcs
import astropy.io.fits as pyfits
import astropy.units as u

CROSS_PSCALE = 0.11057

CRDS_CONTEXT = "jwst_1464.pmap"

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

        keep = np.array([v.startswith(f"{prog:05d}{obs:03d}") for v in res["visit_id"]])
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
            dir="Level2", file_query=file_query, ext=nod_ext, bkg_threshold=0.3
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
    else:
        ptabs = []
        for dm in dms:
            if "_x02101" not in dm.meta.filename:
                ptabs.append(lrs_pixel_table(dm))

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

    return data


def optimal_extraction(
    data,
    err_data="err",
    weight_type="rnoise",
    prof_threshold=0.01,
    low_threshold=-5,
    **kwargs,
):
    """
    Optimal extraction with ``data["profile"]``
    """

    # if False:
    #     # Test variance
    #     xx = np.arange(-10, 10.1, 0.1)
    #     p = 1. / np.sqrt(2*np.pi) * np.exp(-xx**2/2)
    #     Nr = 1000
    #     sh = (Nr, len(xx))
    #     var_rn = np.random.rand(*sh) * 0.5 + 1
    #     var_p = np.random.rand(*sh) * 2 + 10
    #     norms = np.zeros(Nr)
    #
    #     norms = np.random.rand(Nr) * 50 + 10
    #     var_p = norms[:,None] * p[None,:]
    #
    #     signal = norms[:,None] * p[None,:]
    #
    #     rvs = signal + np.random.normal(size=sh) * np.sqrt(var_rn + var_p)
    #     var = (var_rn + var_p)
    #     wht = 1.0 / var
    #
    #     psum = (rvs * p * wht).sum(axis=1) / (p**2 * wht).sum(axis=1)
    #     pvar = 1.0 / (p**2 * wht).sum(axis=1)
    #     print(utils.nmad((psum - norms) / np.sqrt(pvar)))
    #
    #     wht = 1.0 / var_rn
    #     psum2 = (rvs * p * wht).sum(axis=1) / (p**2 * wht).sum(axis=1)
    #     pvar2 = (var * (p * wht)**2).sum(axis=1) / (p**2 * wht).sum(axis=1)**2
    #     print(utils.nmad((psum2 - norms) / np.sqrt(pvar2)))

    if "wbin" not in data:
        set_bin_arrays(data, **kwargs)

    if weight_type == "rnoise":
        data_wht = 1.0 / data["var_rnoise"]
    elif weight_type == "sys_err":
        data_wht = 1.0 / data["full_err"] ** 2
    else:
        data_wht = 1.0 / data["err"] ** 2

    if False:
        from scipy.stats import binned_statistic

        data_wht = 1.0 / data["var_rnoise"]

        mask = np.isfinite(data["data"] + data["profile"] + data_wht)

        mask &= data["profile"] > prof_threshold * np.nanmax(data["profile"])
        mask &= data["data"] > -5 * data["err"]

        snum = binned_statistic(
            data["wave"][mask],
            (data["data"] * data["profile"] * data_wht)[mask],
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

        sspec = snum.statistic / sden.statistic
        svar = svnum.statistic / sden.statistic**2

    data["meta"]["weight_type"] = weight_type
    data["meta"]["err_data"] = err_data

    valid = np.isfinite(
        data["wave"] + data["dx"] + data["data"] + data[err_data] + data_wht
    )

    num, vnum, den, ntot = pseudo_drizzle(
        data["wave"][valid],
        data["dx"][valid],
        data["data"][valid],
        data[err_data][valid] ** 2,
        data_wht[valid],
        data["wbin_edge"],
        data["ybin"],
    )

    pnum, pvnum, pden, pntot = pseudo_drizzle(
        data["wave"][valid],
        data["dx"][valid],
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

    spec = opt_num / opt_den
    spec_var = var_num / opt_den**2

    resid = (num / den - pnum / den * spec) / np.sqrt(var)
    mask2d = np.isfinite(resid + var) & (den > 0) & (var > 0)

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
    data["spec"] = spec
    data["spec_var"] = spec_var


def set_bin_arrays(data, ybin=np.arange(-2.5, 2.5, 0.045), wbin=None, **kwargs):
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

CALSPEC_FILE = os.path.join(msautils.module_data_path(), "lrs", "calspec_correction_jw06620-obs13.fits")

def process_2d_products(
    data,
    calspec_file=CALSPEC_FILE,
    prefix="lrs",
    z=0,
    show_2d_corr=True,
    vmax=None,
    **kwargs,
):
    """
    Combine 2D data
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
    corr = np.interp(data["wbin"], calsp["wave"], calsp["corr"], left=0, right=0)

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

    if vmax is None:
        vmax = 5 * np.nanpercentile(num / den / corr2d, 90)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, sharey=True)

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
        color="w",
        transform=axes[1].transAxes,
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

    fig.tight_layout(pad=1)

    tab = utils.GTable()
    tab["wave"] = data["wbin"] * u.micron
    tab["flux"] = data["spec"] / corr * u.microJansky
    # tab["err"] = 1 / np.sqrt(data["opt_den"]) / corr * u.microJansky
    tab["err"] = np.sqrt(data["spec_var"]) / corr * u.microJansky
    tab["calsp"] = corr / u.microJansky

    tab.meta["grating"] = "LRS"
    tab.meta["filter"] = "P750L"
    tab.meta["instrume"] = "MIRI"
    tab.meta["srcname"] = prefix

    tab.meta["z"] = z
    tab.meta["ytrace"] = np.interp(0, data["ybin"], np.arange(len(data["ybin"])))

    ypscale = np.diff(data["ybin"])[0]

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

    ptab["pfit"] = np.nansum((pnum)[:, sub], axis=1) / np.nansum((pden)[:, sub], axis=1)

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

    outh, outw = utils.make_maximal_wcs(files=[file], pixel_scale=0.11, get_hdu=False)

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

            sr = utils.SRegion(np.array(slit_ap.corners("sky", rederive=False)))

            label = "{visit_id} {targprop}".format(**row)
            sr.label = label

            slit_polygons += sr.polystr(precision=5)

            sr.ds9_properties = "color=cyan"

            for reg in sr.region:
                fp.write(reg + "\n")

            sr = utils.SRegion(np.array(illum_ap.corners("sky", rederive=False)))

            sr.label = label
            sr.ds9_properties = "color=magenta"

            for reg in sr.region:
                fp.write(reg + "\n")

            illum_polygons += sr.polystr(precision=5)

    res["slit_footprint"] = slit_polygons
    res["illum_footprint"] = illum_polygons


def nod_sky_subtraction(dir="Level2", file_query="jw0", ext="cal", bkg_threshold=0.3):
    """
    Compute differences of nodded exposures for sky subtraction
    """

    full_file_query = os.path.join(dir, file_query + "*" + ext + ".fits")

    s2_files = glob.glob(full_file_query)

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

        # targ, grp, dith, _, _ = file.split('_')

        dith = f"{dm.meta.dither.x_offset:>4.1f}"
        targ = dm.meta.target.proposer_name.lower()

        if targ not in s2:
            s2[targ] = {}
        if dith not in s2[targ]:
            s2[targ][dith] = {"dm": []}

        s2[targ][dith]["dm"].append(dm)

    if ext == "bkg":
        return s2

    ###
    for targ in s2:
        msg = f"Target: {targ}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        for dith in s2[targ]:
            msg = f'  dither {dith}: {len(s2[targ][dith]["dm"])} exposures'
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

            s2[targ][dith]["avg"] = np.nanmean(
                np.array([dm.data for dm in s2[targ][dith]["dm"]]), axis=0
            )

            s2[targ][dith]["var"] = np.nanmean(
                np.array([dm.err**2 for dm in s2[targ][dith]["dm"]]), axis=0
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

            for dm in s2[targ][d1]["dm"]:
                dm.data -= other_avg
                dm.err += np.sqrt(other_var)

                dm.meta.filename = dm.meta.filename.replace("_bkg", "")
                out_file = os.path.join(
                    dir, dm.meta.filename.replace(ext, ext + "_bkg")
                )

                offset_summary = ",".join([f"{off:>5.1f}" for off in x_offsets])

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


def plot_lrs_pixel_table(ptab, figsize=(6, 6), cmap="magma", wave_power=3, vmax=500):
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
        np.exp(-np.rot90(ptab["dx"] - ptab["x_offset"], -1) ** 2 / 2 / 0.3**2), **kws
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
        psf_tab = utils.read_catalog(os.path.join(data_path, "lrs", "lrs_psf_data_jw06620-obs13.fits"))

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
        ctab = utils.read_catalog(os.path.join(data_path, "lrs", "lrs_psf_data_extended_shift.fits"))

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
        (sigma_scales, sigma_widths, psf_shifts, sigma_coeffs) = extended_psf_data

    result["sigma_scales"] = sigma_scales
    result["sigma_coeffs"] = sigma_coeffs
    result["sigma_widths"] = sigma_widths
    result["psf_shifts"] = psf_shifts

    result["profile"] = np.zeros_like(data["dx"])

    for i, scl in enumerate(sigma_scales):
        sig = np.sqrt(
            (psf_width * scl + sigma_widths[i]) ** 2 + (src_width / CROSS_PSCALE) ** 2
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
        data["meta"]["src_width"] = src_width
        data["meta"]["theta"] = theta

        optimal_extraction(data, **kwargs)

    return result


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
        psf_tab = utils.read_catalog(os.path.join(data_path, "lrs", "lrs_psf_data_jw06620-obs13.fits"))

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
        ctab = utils.read_catalog(os.path.join(data_path, "lrs", "lrs_psf_data_extended_shift.fits"))

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
        (sigma_scales, sigma_widths, psf_shifts, sigma_coeffs) = extended_psf_data

    result["sigma_scales"] = sigma_scales
    result["sigma_coeffs"] = sigma_coeffs
    result["sigma_widths"] = sigma_widths
    result["psf_shifts"] = psf_shifts

    # result["profile"] = np.zeros_like(data["dx"])

    psf_sum = np.zeros_like(data["dx"])
    src_sum = np.zeros_like(data["dx"])

    for i, scl in enumerate(sigma_scales):
        sig = np.sqrt(
            (psf_width * scl + sigma_widths[i]) ** 2 + (src_width / CROSS_PSCALE) ** 2
        )

        gau_psf = norm(
            loc=psf_shifts[i] * CROSS_PSCALE,
            scale=np.sqrt((psf_width * scl + sigma_widths[i]) ** 2) * CROSS_PSCALE,
        )

        psf_sum += (gau_psf.cdf(slit_width / 2.) - gau_psf.cdf(-slit_width / 2.)) * sigma_coeffs[i,:]

        gau_src = norm(
            loc=psf_shifts[i] * CROSS_PSCALE,
            scale=sig * CROSS_PSCALE,
        )

        src_sum += (gau_src.cdf(slit_width / 2.) - gau_src.cdf(-slit_width / 2.)) * sigma_coeffs[i,:]

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
        prof_kwargs={"optimize": True, "weight_type": "rnoise", "err_data": "err"},
        do_query=True,
        skip_preprocessing=True,
        nod_ext="bkg",
        sys_err=0.02,
        prefix="jw06620-obs13",
        file_query="jw0662000[13]",
    )

    data = res["data"]

    data_path = msautils.module_data_path()
    psf_tab = utils.read_catalog(os.path.join(data_path, "lrs", "lrs_psf_data_jw06620-obs13.fits"))

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
        (data["dx"])[wsub], (data["data"] / y)[wsub], alpha=0.1, c=data["wave"][wsub]
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

        c = np.linalg.lstsq((profs * y / data["err"])[:, wsub].T, yx[wsub], rcond=None)

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
    cw = np.array([np.interp(data["wave"], wcoeffs, coeffs[:, i]) for i in range(nc)])

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
        prof_kwargs={"optimize": True, "weight_type": "rnoise", "err_data": "err"},
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
        prof_kwargs={"optimize": True, "weight_type": "rnoise", "err_data": "err"},
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
    c = np.linalg.lstsq((profs * y / data["err"])[:, wsub].T, yx[wsub], rcond=None)

    m = profs[:, wsub].T.dot(c[0])

    plt.scatter(
        (data["dx"])[wsub], (data["data"] / y)[wsub], alpha=0.1, c=data["wave"][wsub]
    )

    plt.scatter((data["dx"])[wsub], m, alpha=0.1, color="r")
    plt.scatter((data["dx"])[wsub], full_model[wsub], alpha=0.1, color="magenta")


####### Tools for slitless extractions

def lrs_slitless_cutout_offset(xpix, ypix):
    """
    Trace offset as a function of detector pixel position
    """
    from astropy.modeling.models import Polynomial2D

    h = pyfits.Header.fromtextfile(
        os.path.join(msautils.module_data_path(), "lrs", "lrs_wfss_offset_coeffs.txt")
    )

    if xpix < 250:
        prefix = 'DXL'
    else:
        prefix = 'DXR'

    dxpoly = Polynomial2D(degree=h[f'{prefix}deg'])
    dxpoly.parameters = np.array([h[f'{prefix}{i}'] for i in range(h[f'{prefix}N'])])

    prefix = 'DYR'
    dypoly = Polynomial2D(degree=h[f'{prefix}deg'])
    dypoly.parameters = np.array([h[f'{prefix}{i}'] for i in range(h[f'{prefix}N'])])

    dx = dxpoly(xpix, ypix) - dxpoly(h['REFXPIX'], h['REFYPIX'])
    dy = dypoly(xpix, ypix) - dypoly(h['REFXPIX'], h['REFYPIX'])

    return dx, dy


def lrs_slitless_data_cutout(data, wcs, ra, dec, verbose=False, xoffset=None, pad=512, slit_model=None):
    """
    """

    if slit_model is None:
        slit_model = pyfits.open(
            os.path.join(msautils.module_data_path(), "lrs", "lrs_slit_model.fits")
        )

    h = slit_model[0].header

    if ra is not None:
        wcs_pix = np.squeeze(wcs.all_world2pix([ra], [dec], 0))

        rdx = np.array(wcs.all_pix2world([wcs_pix[0], wcs_pix[0] + 1], [wcs_pix[1], wcs_pix[1]], 0))
        local_pscale = np.sqrt(((np.diff(rdx, axis=1).T * np.array([np.cos(dec/180*np.pi), 1.]))**2).sum()) * 3600

        corner_dx, corner_dy = lrs_slitless_cutout_offset(*wcs_pix)
        if verbose:
            print(f'corner offset: {corner_dx:6.2f} {corner_dy:6.2f}')

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
        print(f'pixel offset: {x_slice_offset:6.2f} {y_slice_offset:6.2f}')

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
        'yp': yp,
        'xp': xp,
        'ra': ra,
        'dec': dec,
        'local_pscale': local_pscale
    }

    return cutout
    
# x_slice_offset, y_slice_offset