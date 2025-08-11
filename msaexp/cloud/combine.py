"""
Scripts for combining spectra from AWS cloud products
"""

import os
import gc
import glob
import subprocess
import time
import traceback

import yaml

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import astropy.io.fits as pyfits
import jwst.datamodels

from grizli.aws import db
from grizli import utils

from ..slit_combine import extract_spectra
from .. import slit_combine, spectrum
from .. import utils as msautils

__all__ = [
    "fetch_files",
    "get_join_indices",
    "combine_spectra_pipeline",
    "get_extraction_info",
    "handle_spectrum_extraction",
    "get_targets",
]


if 0:
    root = "snh0pe-v4"
    key = "4446_274"
    s3_base = "s3://msaexp-nirspec/extractions"


def fetch_files(
    root="snh0pe-v4",
    key="4446_274",
    s3_base="s3://msaexp-nirspec/extractions",
    get_bkg=True,
    fix_srcname=True,
    shutter=None,
    **kwargs,
):
    """
    Fetch slitlet files and combine spectrum
    """

    if key.startswith("s"):
        slit_files = f"jw*_{key}.fits"
    else:
        if shutter is None:
            slit_files = f"jw*_raw*.{key}.fits"
        else:
            slit_files = f"jw*_raw*.{shutter}.*fits"

    fetch_command = (
        f'aws s3 sync {s3_base}/slitlets/{root}/ ./ --exclude "*" '
        + f'--include "{slit_files}"'
    )

    if root.startswith('jw0'):
        # Fixed slit
        slit_name = key.split('_')[1]
        slit_files = f"{root[:13]}*{slit_name}.fits"
        fetch_command = (
            f'aws s3 sync {s3_base}/{root}/ ./ --exclude "*" '
            + f'--include "{slit_files}"'
        )

    if get_bkg:
        fetch_command += ' --include "*bkg.fits"'

    sync_result = subprocess.run(
        fetch_command, shell=True, capture_output=True
    )

    files = glob.glob(slit_files)
    files.sort()

    if fix_srcname:
        file_groups = {}

        for file in files:
            with pyfits.open(file, mode="update") as im:
                file_src = im[1].header["SRCNAME"]

                group_key = key

                if "SLIT" in im[0].header["APERNAME"]:
                    slit_name = im[1].header["SLTNAME"].upper()
                    im[0].header["FXD_SLIT"] = slit_name
                    im[0].header["APERNAME"] = f"NRS_{slit_name}_SLIT"

                    # Include MSAMETFL index in srcname
                    if "MSAMETFL" in im[0].header:
                        msaspl = im[0].header["MSAMETFL"].split("_")[1]
                        slit_key = f"{msaspl}_{slit_name}".lower()
                    else:
                        slit_key = f"{slit_name}".lower()

                    im[1].header["SRCNAME"] = slit_key

                    msg = f"fetch_files.fix_srcname: {file} => {slit_key}"
                    utils.log_comment(utils.LOGFILE, msg, verbose=True)
                    im.flush()

                    group_key = slit_key

                elif file_src != key:
                    if key.startswith("background"):
                        prog = int(im[0].header["PROGRAM"])
                        key = f"{prog}_b{key.split('_')[1]}"
                        # group_key = "-".join([im[0].header["MSAMETFL"], key])
                        group_key = key

                    msg = (
                        f"fetch_files.fix_srcname: {file} {file_src} => {key}"
                    )
                    utils.log_comment(utils.LOGFILE, msg, verbose=True)
                    im[1].header["SRCNAME"] = key
                    im.flush()

                if group_key not in file_groups:
                    file_groups[group_key] = []

                file_groups[group_key].append(file)
    else:
        file_groups = {key: files}

    return files, file_groups, sync_result


GRATING_LIMITS = {
    "prism": [0.54, 5.51, 0.01],
    "g140m": [0.55, 3.35, 0.00063],
    "g235m": [1.58, 5.3, 0.00106],
    "g395m": [2.68, 5.51, 0.00179],
    "g140h": [0.68, 1.9, 0.000238],
    "g235h": [1.66, 3.17, 0.000396],
    "g395h": [2.83, 5.24, 0.000666],
}

GRATINGS = [k.upper() for k in GRATING_LIMITS]

BAD_PIXEL_NAMES = [
    "DO_NOT_USE",
    # "OTHER_BAD_PIXEL",
    "MSA_FAILED_OPEN",
    # "UNRELIABLE_SLOPE",
    # "UNRELIABLE_BIAS",
    # "NO_SAT_CHECK",
    # "NO_GAIN_VALUE",
    "HOT",
    "DEAD",
    # # "TELEGRAPH",
    # "RC",
    # "LOW_QE",
    "OPEN",
    "ADJ_OPEN",
    "SATURATED",
]

DRIZZLE_KWS = dict(
    step=1,
    with_pathloss=True,
    wave_sample=1.05,
    ny=15,
    dkws=dict(oversample=16, pixfrac=0.8),
    grating_limits=GRATING_LIMITS,
)

FIT_PARAMS_KWARGS = dict(
    sn_percentile=95,
    sigma_threshold=0,
    degree_sn=[[-10000], [0]],
    verbose=True,
)

EXTENDED_CALIBRATION_KWARGS = {
    "threshold": 0.00,
    "fixed_slit_correction": 1,
    "quadrant_correction": True,
}

FLAG_PERCENTILE_KWARGS = dict(
    plevels=[0.95, -4, -0.1],
    yslit=[-2, 2],
    scale=2.0,
    dilate=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # np.ones((3, 3)),
)


def get_join_indices(root):
    """
    Indices for file groupings based on visit / observation / slit ID
    """
    # "join" options for file grouping
    if root in [
        "gds-udeep-lr-v1",
        "xxgds-deep-lr-v1",
        "gds-deep-xmr-v1",
        "bluejay-north-v1",
        "xbluejay-north-alt-v1",
    ]:
        # join = [0,3,5]
        # # join = [0,1,3,5]
        join = [3, 5]
    elif "gds-udeep" in root:
        join = [0, 3, 5]
    elif "gds-deep" in root:
        join = [0, 1, 3, 5]
    elif "jades-gds" in root:
        join = [0, 1, 3, 5]
    elif "uncover" in root:
        join = [0, 3, 5]
    elif "egami" in root:
        join = [3, 5]
    elif "macsj0647-hr" in root:
        join = [3, 5]
        # join = [0,3]
    elif "glazebrook-v" in root:
        join = [0, 3]
    elif "suspense-kriek-v" in root:
        join = [3, 5, 7]
    elif "cecilia" in root:
        join = [0, 1, 3]
    else:
        join = [0, 3, 5]

    return join


def combine_spectra_pipeline(
    root="snh0pe-v4",
    outroot=None,
    key="4446_274",
    files=[],
    output_type="from_root",
    join=None,
    recenter_type=0,
    gratings=GRATINGS,
    bad_pixel_names=BAD_PIXEL_NAMES,
    turn_off_flagging=False,
    sky_diffs=True,
    with_sflat_correction=True,
    sflat_straighten=3,
    initial_theta="auto",
    trace_with_xpos=True,
    fit_params_kwargs=FIT_PARAMS_KWARGS,
    flag_percentile_kwargs=FLAG_PERCENTILE_KWARGS,
    force_diffs=True,
    set_background_spectra_kwargs={},
    extended_calibration_kwargs=EXTENDED_CALIBRATION_KWARGS,
    drizzle_kws=DRIZZLE_KWS,
    plot_kws={},
    flag_trace_kwargs={},
    **kws,
):
    """
    combine spectra
    """

    if outroot in [None]:
        outroot = root

    if len(files) == 0:
        files, file_groups, sync_result = fetch_files(
            root=root, key=key, **kws
        )
    else:
        file_groups = {key: files}

    if len(files) == 0:
        return None

    prf = slit_combine.set_lookup_prf()

    if join is None:
        join = get_join_indices(root)

    ### Do combination

    msautils.SFLAT_STRAIGHTEN = sflat_straighten

    plt.rcParams["scatter.marker"] = "."

    utils.LOGFILE = f"{outroot}_{key}.extract.log"
    utils.log_comment(utils.LOGFILE, "Start", verbose=True, show_date=True)

    utils.log_comment(utils.LOGFILE, f"{len(files)} files", verbose=True)
    for i, f in enumerate(files):
        utils.log_comment(utils.LOGFILE, f"file {i:>3} {f}", verbose=True)

    try:

        if initial_theta in ["auto"]:
            if "b" in key:
                initial_theta = [5.0, -0.5]
            elif "background" in key:
                initial_theta = [5.0, -0.5]
            else:
                initial_theta = None  # [0, 0, 0]

        kwargs = dict(
            path_to_files="./",
            stuck_min_sn=0.9 - 500,
            pad_border=0,
            sort_by_sn=True,
            position_key="y_index",
            reference_exposure="auto",
            offset_degree=1,
            trace_from_yoffset=True,
            initial_sigma=5,
            fit_type=1,
            fix_params=False,
            input_fix_sigma=None,
            undo_barshadow=False,
            undo_pathloss=1,
            split_uncover=1,
            get_xobj=True,
            make_2d_plots=2,
            hot_cold_kwargs=None,
            flag_profile_kwargs=None,
            flag_trace_kwargs=flag_trace_kwargs,
            lookup_prf_type="merged",  # by grating
            weight_type="ivm",
            files=files,
            root=outroot,
            do_gratings=gratings,
            diffs=True,
            join=join,
            trace_with_xpos=trace_with_xpos,
            trace_with_ypos="auto",
            recenter_all=((recenter_type & 1) > 0),
            free_trace_offset=((recenter_type & 2) > 0),
            initial_theta=initial_theta,
            flag_percentile_kwargs=flag_percentile_kwargs,
            fit_params_kwargs=fit_params_kwargs,
            drizzle_kws=drizzle_kws,
            plot_kws=plot_kws,
            mask_cross_dispersion=None,
            cross_dispersion_mask_type="trace",
            bad_shutter_names=None,
            stuck_threshold=0.3,
            with_fs_offset=False,
            fit_shutter_offset_kwargs=None,
            shutter_offset=0.0,
            with_extra_dq=True,
            include_full_pixtab=["PRISM"],
            do_multiple_mask=True,
        )

        for k in kws:
            if k in kwargs:
                msg = f"set keyword arg: {k} = {kws[k]}"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)

                kwargs[k] = kws[k]

        kwargs["fix_prism_norm"] = False

        kwargs["drizzle_kws"]["with_pathloss"] = True

        estimate_sky_kwargs = {
            "make_plot": True,
            "df": 71,
            "outlier_threshold": 7,
            "var_percentiles": [-2, -2],
            "absolute_threshold": 0.2,
            "min_bar": 0.95,
            "mask_yslit": [[-4, 4]],
            "high_clip": 0.9,
        }

        if root.startswith("glazebrook") & ("-v3" in root):
            estimate_sky_kwargs["mask_yslit"] = [[-8, 8]]
            estimate_sky_kwargs["absolute_threshold"] = 0.1
            kwargs["pad_border"] = 2

        kwargs["undo_barshadow"] = 2
        kwargs["min_bar"] = 0.4

        kwargs["fixed_offset"] = 0.0
        kwargs["bar_corr_mode"] = "wave"

        if root.split("-v")[0] in ["macsj0647-hr"]:
            # kwargs['diffs'] = True
            kwargs["diffs"] = False
            # estimate_sky_kwargs = None
            estimate_sky_kwargs["df"] = 15
            kwargs["hot_cold_kwargs"] = None
            # kwargs['join'] = [0,3]

        ####
        # Scale global sky spectrum
        sky_file = f"{root}_prism_sky.csv"

        if os.path.exists(sky_file):

            print(f"\n use {sky_file} for global sky")

            sky_data = utils.read_catalog(sky_file)
            kwargs["sky_arrays"] = sky_data["wave"], sky_data["flux"]

            kwargs["pad_border"] = 1

            estimate_sky_kwargs["df"] = 0
            estimate_sky_kwargs["mask_yslit"] = [[-4, 4]]

            if 1:
                estimate_sky_kwargs["df"] = 5
                estimate_sky_kwargs["mask_yslit"] = [[-6, 6]]
                if root.startswith("glazebrook"):
                    estimate_sky_kwargs["mask_yslit"] = [[-8, 8]]

            # Fixed sky
            if 0:
                estimate_sky_kwargs["df"] = -1
                estimate_sky_kwargs["mask_yslit"] = [[-8, 8]]

            estimate_sky_kwargs["outlier_threshold"] = 7

            if 0:
                estimate_sky_kwargs["outlier_threshold"] = 1000

        kwargs["estimate_sky_kwargs"] = estimate_sky_kwargs
        kwargs["fix_prism_norm"] = True

        if root.split("-v3")[0] in ["macsj0647-single", "whl0137"]:
            kwargs["diffs"] = False
            kwargs["estimate_sky_kwargs"] = None
            kwargs["join"] = [0, 3, 5]
            kwargs["flag_profile_kwargs"] = {"make_plot": True, "grow": 4}
            kwargs["flag_trace_kwargs"] = None
            kwargs["make_2d_plots"] = False
            kwargs["pad_border"] = 0
            kwargs["min_bar"] = 0.1

        slit_combine.BAD_PIXEL_FLAG = 1 | 1024
        for _bp in bad_pixel_names:
            slit_combine.BAD_PIXEL_FLAG |= jwst.datamodels.dqflags.pixel[_bp]

        if 1:
            kwargs["sky_file"] = "read"
            kwargs["global_sky_df"] = 7
            kwargs["estimate_sky_kwargs"]["mask_yslit"] = [[-8, 8]]
            kwargs["estimate_sky_kwargs"]["high_clip"] = 1.5
            kwargs["estimate_sky_kwargs"]["absolute_threshold"] = 1.0

        if root.split("-v")[0] in [
            "cecilia",
            "cosmos-curti",
            "stark-a1703",
            "cosmos-alpha",
            "stark-rxcj2248",
        ]:
            kwargs["recenter_all"] = True
            kwargs["grating_diffs"] = False
            kwargs["diffs"] = True
            # kwargs["include_full_pixtab"] = ["PRISM"]

        if root.split("-v3")[0] in ["abell2744-glass"]:
            kwargs["diffs"] = True
            kwargs["sky_arrays"] = None
            kwargs["estimate_sky_kwargs"] = None
            kwargs["valid_frac_threshold"] = 0.1

            kwargs["dilate_failed_open"] = False

            if key in ["1324_180009", "1324_320027"]:
                kwargs["num_shutters"] = -1
            else:
                kwargs["num_shutters"] = 0

            msautils.BAD_PIXEL_FLAG = 1 | 1024

            if "MSA_FAILED_OPEN" in msautils.BAD_PIXEL_NAMES:
                print("pop failed open dq flag")
                _ = msautils.BAD_PIXEL_NAMES.pop(
                    msautils.BAD_PIXEL_NAMES.index("MSA_FAILED_OPEN")
                )
                msautils._set_bad_pixel_flag()

            for _bp in msautils.BAD_PIXEL_NAMES:
                msautils.BAD_PIXEL_FLAG |= jwst.datamodels.dqflags.pixel[_bp]

            print(msautils.BAD_PIXEL_FLAG)

        if root in [
            "ceers-v3",
            "macs0416-v3",
            "macs1423-v3",
            "abell370-v3",
            "macs1149-v3",
            "gds-egami-ddt-v3",
            "macs0417-v3",
        ]:
            kwargs["diffs"] = False

        if root.startswith("rubies") & ("v3" in root):
            kwargs["diffs"] = False

        if (
            root
            in [
                "jades-gds-wide3-v3",
                "jades-gds-w03-v3",
                "jades-gds-w05-v3",
                "jades-gds-w06-v3",
                "jades-gds-w07-v3",
                "jades-gds-w08-v3",
                "jades-gds-w09-v3",
                "jades-gds05-v3",
                "abell2744-castellano1-v3",
            ]
        ) & (True):
            kwargs["diffs"] = False
            if force_diffs:
                kwargs["diffs"] = True

        if "-nod" in root:
            kwargs["diffs"] = True

        elif root in [
            "xrubies-uds31-v3",
            "xrubies-uds32-v3",
            "xrubies-uds33-v3",
            "uncover-62-v3",
            "uncover-61-v3",
        ]:
            kwargs["diffs"] = True

        if key in ["4233_61167", "4233_61168"]:
            turn_off_flagging = True
            kwargs["grating_diffs"] = True
            kwargs["mask_cross_dispersion"] = [8, 1000]
            kwargs["cross_dispersion_mask_type"] = "trace"
            # kwargs['estimate_sky_kwargs']['df'] = -1

        elif key == "1810_11494":
            kwargs["mask_cross_dispersion"] = [-1, 1000]
            kwargs["cross_dispersion_mask_type"] = "fixed"

        # else:
        #     kwargs["mask_cross_dispersion"] = None
        #     kwargs["cross_dispersion_mask_type"] = "trace"

        if key in ["4233_131457"]:
            kwargs["mask_cross_dispersion"] = [3, 1000]
            kwargs["cross_dispersion_mask_type"] = "fixed"

        if turn_off_flagging:
            utils.log_comment(
                utils.LOGFILE,
                "Turn off bad pixel flagging",
                verbose=True,
                show_date=False,
            )

            kwargs["flag_trace_kwargs"] = None
            kwargs["flag_percentile_kwargs"] = None
            kwargs["flag_profile_kwargs"] = None
            kwargs["estimate_sky_kwargs"]["outlier_threshold"] = 10000000
            kwargs["estimate_sky_kwargs"]["high_clip"] = 100000

        if root == "j0252m0503-hennawi-07-v3":
            kwargs["diffs"] = True

        if root == "macsj0647-hr-v3":
            _files = glob.glob(f"jw04246003001_0310[135]*{key}.fits")
            print(f"{root} manual {len(_files)} files {key}")

            kwargs["files"] = _files
            # kwargs['valid_frac_threshold'] = 0.01
            # kwargs['files'] = None
            kwargs["diffs"] = False
            kwargs["grating_diffs"] = False
            kwargs["make_2d_plots"] = False
            kwargs["trace_with_ypos"] = True
            kwargs["trace_from_yoffset"] = False
            kwargs["sky_file"] = "jw04246003001_sky.csv"
            # kwargs['estimate_sky_kwargs'] = None
            kwargs["global_sky_df"] = 0
            # kwargs['estimate_sky_kwargs']['mask_yslit'] = [[100,-100]]
            kwargs["estimate_sky_kwargs"]["outlier_threshold"] = 7
            # kwargs['drizzle_kws']['wave_sample'] = 0.95
            kwargs["drizzle_kws"]["dkws"]["pixfrac"] = 1.0
            kwargs["estimate_sky_kwargs"]["high_clip"] = 5
            kwargs["flag_profile_kwargs"] = {
                "require_multiple": False,
                "make_plot": False,
            }
            kwargs["initial_theta"] = [4.0, 0]
            kwargs["fix_params"] = True
            kwargs["plot_kws"] = {
                "smooth_sigma": 2,
                "ymax_sigma_scale": 15,
            }

        elif (root == "jades-gds-w09-v3") & (key == "b32"):
            # print('yes')

            _files = glob.glob("*phot*b32.fits")
            _files.sort()
            if 0:
                kwargs["files"] = _files[-3:]  # _files
                kwargs["bad_shutter_names"] = [1]
            else:
                kwargs["files"] = _files[:-3]

        if ("castellano" in root) & ("v3" in root):
            #  kwargs['sky_file'] = 'jw03073010001_sky.csv'
            kwargs["diffs"] = True

        if key in ["3073_22600"]:
            kwargs["mask_cross_dispersion"] = [7, 100]
            kwargs["cross_dispersion_mask_type"] = "trace"
            kwargs["diffs"] = True

        if key in ["3073_22600", "3073_21352", "3073_23628"]:
            kwargs["diffs"] = True

        if "-xoff" in root:
            kwargs["trace_with_xpos"] = True

        if "-v4" in root:
            # kwargs["trace_with_xpos"] = True
            kwargs["fix_prism_norm"] = False
            # kwargs["diffs"] = len(glob.glob("*sky.csv")) == 0
            kwargs["diffs"] = len(glob.glob("*gbkg.fits")) == 0

            if kwargs["diffs"]:
                kwargs["estimate_sky_kwargs"] = None

            # kwargs["include_full_pixtab"] = [
            #     "PRISM",
            #     # "G140M",
            #     # "G235M",
            #     # "G395M",
            # ]

            if force_diffs:
                kwargs["diffs"] = True

        ######################
        kwargs["estimate_sky_kwargs"] = None
        kwargs["sky_file"] = None
        kwargs["fix_prism_norm"] = False
        kwargs["min_extended_calibration"] = None
        kwargs["extended_calibration_kwargs"] = None

        if 0:
            kwargs["mask_cross_dispersion"] = [-100, 2]
            kwargs["cross_dispersion_mask_type"] = "fixed"

        elif key == "3567_66283":
            kwargs["mask_cross_dispersion"] = [-100, -5]
            kwargs["cross_dispersion_mask_type"] = "trace"

        elif key == "4233_120375":
            kwargs["mask_cross_dispersion"] = [2, 100]
            kwargs["cross_dispersion_mask_type"] = "fixed"
            kwargs["initial_theta"] = [1, -3.5]

        elif key == "6368_5643":
            kwargs["mask_cross_dispersion"] = [-100, -2]
            kwargs["cross_dispersion_mask_type"] = "fixed"
            kwargs["initial_theta"] = [1, 0]
            kwargs["stuck_threshold"] = 0.1
            kwargs["bad_shutter_names"] = [-3, -2, 3]

        # if "kriek" in root:
        #     kwargs["recenter_all"] = True

        if "5224_978020" in key:
            kwargs["initial_theta"] = [3, -3.5]

        ####
        ##  Run it
        kwargs["drizzle_kws"]["with_pathloss"] = False

        if output_type == "from_root":
            if "-v3" in root:
                output_type = "v3"
            elif "-v4" in root:
                output_type = "v4"

        if output_type == "v3":
            kwargs["drizzle_kws"]["with_pathloss"] = True

        elif output_type == "raw":

            join_ = []
            for j in join:
                if j > 3:
                    join_.append(j + 2)
                else:
                    join_.append(j)

            kwargs["join"] = join_
            kwargs["extended_calibration_kwargs"] = {"threshold": -99}

        else:
            join_ = []
            for j in join:
                if j > 3:
                    join_.append(j + 2)
                else:
                    join_.append(j)

            kwargs["join"] = join_  # [3, 5, 7]

            kwargs["extended_calibration_kwargs"] = extended_calibration_kwargs
            kwargs["drizzle_kws"]["with_pathloss"] = True
            kwargs["with_sflat_correction"] = with_sflat_correction

            kwargs["diffs"] = sky_diffs > 0
            kwargs["set_background_spectra_kwargs"] = (
                set_background_spectra_kwargs
            )

            kwargs["grating_diffs"] = sky_diffs >= 0

        # # Fixed slit
        # if root.startswith("jw0") & ("_s" in key) & (0):
        #     for g in list(file_groups.keys()):
        #         if len(file_groups[g]) == 5:
        #             spl = g.split("_")
        #             files = [f for f in file_groups[g]]
        #             spl.insert(2, "set1")
        #             g1 = "_".join(spl)
        #             file_groups[g1] = files[0::2]
        #
        #             spl[2] = "set2"
        #             g2 = "_".join(spl)
        #             file_groups[g2] = files[1::2]
        #
        #             _ = file_groups.pop(g)

        for grp in file_groups:
            kwargs["files"] = file_groups[grp]
            hdul, xobj = extract_spectra(target=grp, **kwargs)

    except:
        print("Failed: ", key)
        utils.log_exception(utils.LOGFILE, traceback, verbose=True, mode="a")
        xobj = None

    utils.log_comment(utils.LOGFILE, "Done", verbose=True, show_date=True)

    gc.collect()
    gc.collect()
    gc.collect()

    return xobj


def get_extraction_info(root="snh0pe-v4", outroot=None, key="4446_274"):
    """
    Get columns for nirspec_extractions table
    """
    keys = [
        "SOURCEID",
        "SRCID",
        "SRCNAME",
        "SRCRA",
        "SRCDEC",
        "GRATING",
        "FILTER",
        "EXPTIME",
        "EFFEXPTM",
        "NCOMBINE",
        "FILENAME",
        "MSAMETFL",
        "MSAMETID",
        "MSACONID",
        "PATT_NUM",
        "SLITID",
        "VERSION",
    ]

    if outroot in [None]:
        outroot = root

    files = glob.glob(f"{outroot}*{key}.spec.fits")
    if key.startswith("background"):
        bkey = f"_b" + key.split("_")[1]
        files += glob.glob(f"{outroot}*{bkey}.spec.fits")

    files.sort()

    rows = []
    for file in files:
        with pyfits.open(file) as im:
            row = {"FILE": file}
            for k in keys:
                if k in im[1].header:
                    row[k] = im[1].header[k]

        rows.append(row)

    info = utils.GTable(rows=rows)

    if "NCOMBINE" in info.colnames:
        ren = {
            "NCOMBINE": "NFILES",
            "FILENAME": "FILE1",
            "MSAMETFL": "MSAMET1",
            "MSAMETID": "MSAID1",
            "MSACONID": "MSACNF1",
            "PATT_NUM": "DITHN1",
            "SOURCEID": "SRCID",
            "SLITID": "SLITID1",
            "SLTSTRT1": "XSTART1",
            "SLTSIZE1": "XSIZE1",
            "SLTSTRT2": "YSTART1",
            "SLTSIZE2": "YSIZE1",
            "PA_V3": "PA_V31",
        }

        rev = {}
        for c in ren:
            rev[ren[c]] = c

        for c in ren:
            print("rename ", c, ren[c])
            if c in info.colnames:
                info.rename_column(c, ren[c])

    print(len(info), len(np.unique(info["SRCNAME"])))

    for c in info.colnames:
        info.rename_column(c, c.lower())

    info.rename_column("file1", "dataset")
    info["dataset"] = [f.split("_phot")[0] for f in info["dataset"]]
    info.remove_column("srcname")
    info.rename_column("srcra", "ra")
    info.rename_column("srcdec", "dec")

    for i in range(len(info)):
        try:
            sp = spectrum.read_spectrum(info["file"][i])
            break
        except ValueError:
            print(f"Read {info['file'][i]} failed")
            continue

    info["root"] = root

    info["npix"] = 0
    info["ndet"] = 0
    info["wmin"] = 0.0
    info["wmax"] = 0.0

    info["wmaxsn"] = 0.0

    for perc in [10, 50, 90]:
        info[f"sn{perc}"] = 0.0
        info[f"flux{perc}"] = 0.0
        info[f"err{perc}"] = 0.0

    ikeys = [
        "XSTART1",
        "YSTART1",
        "XSIZE1",
        "YSIZE1",
        "DITHN1",
        "MSAID1",
        "MSACNF1",
        "SLITID1",
    ]
    fkeys = ["SLIT_PA", "PA_V31", "SRCYPIX", "PROFCEN", "PROFSIG"]
    for k in ikeys:
        info[k.lower()] = 0
    for k in fkeys:
        info[k.lower()] = 0.0

    info["valid"] = True

    for i, file in tqdm(enumerate(info["file"])):

        try:
            sp = spectrum.read_spectrum(file)
        except ValueError:
            info["valid"][i] = False
            continue

        sp.valid = sp["valid"]
        det = []
        for k in sp.meta:
            if k.startswith("DETECT"):
                det.append(sp.meta[k])

        info["ndet"][i] = len(np.unique(det))

        if "SLIT_PA" not in sp.meta:
            sp.meta["SLIT_PA"] = 0

        if "SRCYPIX" not in sp.meta:
            sp.meta["SRCYPIX"] = 0

        for k in ikeys + fkeys:
            if k in rev:
                if rev[k] in sp.meta:
                    info[k.lower()][i] = sp.meta[rev[k]]
            else:
                info[k.lower()][i] = sp.meta[k]

        info["npix"][i] = sp.valid.sum()
        if sp.valid.sum() > 1:

            wok = sp["wave"][sp.valid]
            sn = (sp["flux"] / sp["err"])[sp.valid]

            info["wmin"][i] = wok.min()
            info["wmax"][i] = wok.max()
            info["wmaxsn"][i] = wok[np.argmax(sn)]

            snperc = np.percentile(sn, [10, 50, 90])
            eperc = np.percentile(sp["err"][sp.valid], [10, 50, 90])
            fperc = np.percentile(sp["flux"][sp.valid], [10, 50, 90])

            for j, perc in enumerate([10, 50, 90]):
                info[f"sn{perc}"][i] = snperc[j]
                info[f"flux{perc}"][i] = fperc[j]
                info[f"err{perc}"][i] = eperc[j]

    info.remove_column("valid")

    for c in info.colnames:
        if c in ["ra", "dec"]:
            continue

        if info[c].dtype == np.float64:
            info[c] = info[c].astype(np.float32)

    for c in list(info.colnames):
        if c.endswith("1"):
            print(c)
            info.rename_column(c, c[:-1])

    info["ctime"] = [os.path.getmtime(f) for f in info["file"]]

    # Fill missing coordinates
    miss = (info["ra"] == 0) & (info["dec"] == 0)
    if miss.sum() > 0:

        msg = f"Fill shutter coordinates for {miss.sum()} rows"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        SLIT_QUERY = """
        SELECT avg(ra) as ra, avg(dec) as dec,
               array_agg(distinct(srcra, srcdec)) as source_coords,
               count(*)
        FROM nirspec_slits
        WHERE slitlet_id = {slitid} AND msametfl = '{msamet}' AND msametid = {msaid}
        """

        for i in np.where(miss)[0]:
            row_ = info[i]
            slit_rows = db.SQL(SLIT_QUERY.format(**row_))
            if len(slit_rows) == 1:
                info["ra"][i] = slit_rows["ra"][0]
                info["dec"][i] = slit_rows["dec"][0]

    return info


def handle_spectrum_extraction(**event):
    """
    Handler
    """
    defaults = dict(
        root="snh0pe-v4",
        key="4446_274",
        skip_existing=True,
        s3_base="s3://msaexp-nirspec/extractions",
        sync=True,
        clean=True,
        outroot=None,
        rowid=None,
    )

    # Parse yaml args
    if "yaml_kwargs" in event:
        ykws = yaml.load(event["yaml_kwargs"], Loader=yaml.Loader)
        if isinstance(ykws, dict):
            for k in ykws:
                event[k] = ykws[k]

        k_ = event.pop("yaml_kwargs")

    for k in defaults:
        if k not in event:
            print(f"set default {k} = {defaults[k]}")
            event[k] = defaults[k]

    if event["outroot"] in [None, "null"]:
        event["outroot"] = event["root"]

    print(f"Event data:")
    print(yaml.dump(event))

    root = event["root"]
    outroot = event["outroot"]
    key = event["key"]
    skip_existing = event["skip_existing"]
    s3_base = event["s3_base"]
    sync = event["sync"] not in [0, False, "False"]
    sync &= event["rowid"] is not None

    if sync:
        SQL = f"""
    UPDATE nirspec_extractions_helper
    SET status = 1, ctime = {time.time()}
    WHERE rowid = {event["rowid"]} AND key = '{key}'
        """
        print(SQL)
        db.execute(SQL)

    files = glob.glob(f"{outroot}*{key}.spec.fits")
    if key.startswith("background"):
        bkey = f"_b" + key.split("_")[1]
        files += glob.glob(f"{outroot}*{bkey}.spec.fits")

    if (len(files) > 0) & (skip_existing):
        do_extraction = False
    else:
        do_extraction = True

    if do_extraction:
        xobj = combine_spectra_pipeline(**event)
    else:
        xobj = None

    files = glob.glob(f"{outroot}*{key}.spec.fits")
    if key.startswith("background"):
        bkey = f"_b" + key.split("_")[1]
        files += glob.glob(f"{outroot}*{bkey}.spec.fits")

    if len(files) == 0:
        status = 9
        info = None

    else:
        status = 2

        info = get_extraction_info(root=root, outroot=outroot, key=key)

        if sync:
            # Send extraction info
            file_list = ",".join([f"'{file_}'" for file_ in info["file"]])

            SQL_DELETE_FROM_EXTRACTIONS = f"""
    DELETE FROM nirspec_extractions
    WHERE root = '{root}'
    AND file in ({file_list})
            """
            print(SQL_DELETE_FROM_EXTRACTIONS)

            db.execute(SQL_DELETE_FROM_EXTRACTIONS)

            db.send_to_database(
                "nirspec_extractions", info, if_exists="append"
            )

            # Send files
            send_command = (
                f'aws s3 sync ./ {s3_base}/{root}/ --exclude "*" '
                + f'--include "{outroot}*{key}.*" --acl public-read'
            )
            if key.startswith("background"):
                bkey = f"_b" + key.split("_")[1]
                send_command += f' --include "{outroot}*{bkey}.*"'

            print(f"# send to s3: {send_command}")

            sync_result = subprocess.run(
                send_command, shell=True, capture_output=True
            )

            # update status
            SQL = f"""
    UPDATE nirspec_extractions_helper
    SET status = {status}, ctime = {time.time()}
    WHERE rowid = {event["rowid"]} AND key = '{key}'
            """
            print(SQL)
            db.execute(SQL)

    if event["clean"]:
        files = glob.glob(f"{outroot}*{key}.*")
        if key.startswith("background"):
            bkey = f"_b" + key.split("_")[1]
            files += glob.glob(f"{outroot}*{bkey}.*")

        files += glob.glob(f"jw*{key}.*")
        for file in files:
            print(f"rm {file}")
            os.remove(file)

    return xobj, info, status


def get_targets(
    root="snh0pe-v4",
    s3_base="s3://msaexp-nirspec/extractions",
    sync=True,
    status=70,
):
    """ """
    list_command = f"aws s3 ls {s3_base}/slitlets/{root}/"

    print(f"# s3: {list_command}")

    list_result = subprocess.run(list_command, shell=True, capture_output=True)
    rows = list_result.stdout.decode("utf8").strip().split("\n")

    files = []
    for row in rows:
        spl = row.split()
        if len(spl) != 4:
            continue

        file = spl[3]
        is_slitlet = ("phot" in file) | ("raw" in file)
        if is_slitlet & file.startswith("jw") & ("photom" not in file):
            files.append(file)

    if len(files) == 0:
        return None

    files.sort()
    keys = [file.split(".")[-2] for file in files]
    un = utils.Unique(keys, verbose=False)
    un.info(sort_counts=True)

    exist = db.SQL(
        f"""
    SELECT key FROM nirspec_extractions_helper
    where root = '{root}'
    """
    )

    tab = utils.GTable()
    tab["key"] = un.values
    tab["root"] = root
    tab["count"] = un.counts
    tab["status"] = status
    tab["outroot"] = root

    new = ~np.isin(tab["key"], exist["key"])
    tab = tab[new]

    msg = f"{root} {new.sum()} / {len(new)} new keys"
    print(msg)

    if sync & (new.sum() > 0):
        print("send to nirspec_extractions_helper")
        db.send_to_database(
            "nirspec_extractions_helper", tab, if_exists="append"
        )

    return tab
