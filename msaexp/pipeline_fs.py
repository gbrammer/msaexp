"""
Pipeline for processing NIRSpec Fixed-Slit observations
"""

import os
import glob
import inspect
import yaml
import time

import matplotlib.pyplot as plt

from grizli.aws import db
import grizli.utils
from msaexp import pipeline_extended, slit_group, ifu, slit_combine, utils, cloud

QUERY_KWARGS = dict(
    trim_prism_nrs2=True,
    grating=None,
    filter=None,
)


def initialize_fs_helper_table():
    """
    Commands for initializing nirspec_fs_helper DB table
    """
    import astropy.time
    import time

    # https://s3.amazonaws.com/grizli-v2/jwst-public-queries/nirspec_single_object.html?&search=_slit
    summary = grizli.utils.read_catalog("fixed_slit_summary.csv")
    summary["observed_time"] = astropy.time.Time(summary["date_obs"]).unix
    summary["release_time"] = astropy.time.Time(summary["release"]).unix
    summary["status"] = 70

    uni = grizli.utils.Unique(summary["FITS"], verbose=False)
    ind = uni.unique_index()
    summary["obsid"] = [f"{o:011d}" for o in summary["FITS"]]
    summary["version"] = "v4"
    summary["ctime"] = time.time()

    summary = summary[ind][
        "obsid", "version", "observed_time", "release_time", "ctime", "status"
    ]

    QUERY_KWARGS = dict(
        trim_prism_nrs2=True,
        grating=None,
        filter=None,
    )

    summary["query_yaml"] = yaml.dump(QUERY_KWARGS)

    # summary["extract_yaml"] = yaml.dump({'protect_exception': False})

    db.send_to_database("nirspec_fs_helper", summary, if_exists="replace")

    db.execute(
        "ALTER TABLE nirspec_fs_helper ADD COLUMN extract_yaml VARCHAR DEFAULT '';"
    )
    db.execute(
        """
    ALTER TABLE nirspec_fs_helper 
    ADD COLUMN rowid INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY
    """
    )


def run_one_fixed_slit(clean=True):
    """
    Query grizli db for FS data to process
    """
    rows = db.SQL(
        "SELECT * FROM nirspec_fs_helper WHERE status = 0 ORDER BY RANDOM() LIMIT 1"
    )
    if len(rows) == 0:
        print("No rows to run in nirspec_fs_helper with status=0")
        return None

    row = dict(rows[0])
    if row["query_yaml"]:
        row["query_kwargs"] = yaml.load(row["query_yaml"], Loader=yaml.Loader)

    if row["extract_yaml"]:
        row["extract_kwargs"] = yaml.load(row["extract_yaml"], Loader=yaml.Loader)

    row["ctime"] = time.time()
    row["status"] = 1

    db_command = """
    UPDATE nirspec_fs_helper SET status = {status}, ctime = {ctime} WHERE rowid = {rowid}
    """
    db.execute(db_command.format(**row))

    for k in row:
        if "time" in k:
            row[k] = float(row[k])
    row['rowid'] = int(row['rowid'])

    for k in list(row.keys()):
        if 'yaml' in k:
            _ = row.pop(k)

    print(yaml.dump(row))

    try:
        files, s3_root, slit_info, spec_info = reduce_fixed_slit_obsid(**row)
        plt.close("all")

        if len(files) == 0:
            status = 10
        elif slit_info is None:
            status = 9
        elif spec_info is None:
            status = 8
        else:
            status = 2

        if slit_info is not None:
            print(
                f"Send {len(slit_info)} rows to nirspec_cutouts for root = '{s3_root}'"
            )

            db.execute(f"DELETE FROM nirspec_cutouts WHERE root = '{s3_root}'")
            db.send_to_database("nirspec_cutouts", slit_info, if_exists="append")

        if spec_info is not None:
            print(
                f"Send {len(spec_info)} rows to nirspec_extractions for root = '{s3_root}'"
            )

            flist = ",".join(db.quoted_strings(spec_info["file"]))
            db.execute(
                f"DELETE FROM nirspec_extractions WHERE file in ({flist}) AND root = '{s3_root}'"
            )
            db.send_to_database("nirspec_extractions", spec_info, if_exists="append")

        print(f"Sync to s3://msaexp-nirspec/extractions/{s3_root}/")

        os.system(
            f"aws s3 sync {s3_root}_fs/ s3://msaexp-nirspec/extractions/{s3_root}/"
            + f' --exclude "*" --include "jw*_s*[12].fits" '
            + f' --include "{s3_root}_fs.log.txt"'
            + f' --include "{s3_root}*" --acl "public-read"'
        )

        os.system(
            f"aws s3 cp {s3_root}_fs.log.txt s3://msaexp-nirspec/extractions/{s3_root}/"
            " --acl public-read"
        )

    except:
        status = 3
        s3_root = None

    row["ctime"] = time.time()
    row["status"] = status

    db.execute(
        """
        UPDATE nirspec_fs_helper SET status = {status}, ctime = {ctime} WHERE rowid = {rowid}
        """.format(
            **row
        )
    )

    if clean & (s3_root is not None):
        files = glob.glob(f"{s3_root}*txt")
        files += glob.glob(f"{s3_root}_fs/*")
        for file in files:
            print(f"rm {file}")
            os.remove(file)

    return row


def reduce_fixed_slit_obsid(
    obsid="06644003001",
    version="v4",
    query_kwargs=QUERY_KWARGS,
    extract_kwargs={},
    **kwargs,
):
    """
    Full pipeline for NIRSpec Fixed-Slit observations

    Parameters
    ----------
    obsid : str
        Unique JWST observations ID

    version : str
        Version string for data products

    query_kwargs : dict
        Keyword arguments passed to `msaexp.ifu.query_obsid_exposures` for MAST query

    extract_kwargs : dict
        Keyword argments for the spectral combination in `msaexp.slit_combine.extract_spectra`

    Returns
    -------
    files : list
        List of `rate` files

    s3_root : str
        Basename of the output products, ``{obsid}-{version}``
    
    slit_info : table
        Slit cutout info table

    spec_info : table
        Properties of the combined spectrum / spectra

    """
    frame = inspect.currentframe()

    HOME = os.getcwd()

    if os.path.exists("/GrizliImaging"):
        base_path = "/GrizliImaging"
    else:
        base_path = os.getcwd()

    grizli.utils.LOGFILE = os.path.join(base_path, f"jw{obsid}-{version}_fs.log.txt")
    s3_root = f"jw{obsid}-{version}"

    # grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=True)

    args = grizli.utils.log_function_arguments(
        grizli.utils.LOGFILE,
        frame,
        "reduce_fixed_slit_obsid",
        ignore=["sky_arrays", "base_path", "HOME"],
    )

    reduce_path = os.path.join(base_path, f"jw{obsid}-{version}_fs")
    if not os.path.exists(reduce_path):
        os.makedirs(reduce_path)

    os.chdir(reduce_path)

    # Query and download
    for k in ["obsid", "download", "exposure_type", "fixed_slit"]:
        if k in query_kwargs:
            _ = query_kwargs.pop(k)

    res, msg = ifu.query_obsid_exposures(
        obsid=obsid,
        download=True,
        exposure_type="rate",
        fixed_slit=True,
        **query_kwargs,
        # param_ranges={"xoffset": [-0.01, 0.01]},
        # extra_query=extra_query,
    )

    files = []
    for k in msg:
        if ("COMPLETE" in msg[k]) | ("EXISTS" in msg[k]):
            files.append(k)

    if len(files) == 0:
        return files, s3_root, None, None

    # Run pipeline preprocessing
    for rate_file in files[:]:
        # hdul = msautils.resize_subarray_to_full(rate_file)

        slitlet_file = rate_file.replace("_rate.fits", "_slitlet.fits")

        if not os.path.exists(slitlet_file):
            _ = ifu.detector_corrections(rate_file, skip_subarray=False)

            grp = slit_group.NirspecCalibrated(
                rate_file,
                read_slitlet=True,
                make_plot=False,
                area_correction=False,
                prism_threshold=0.999,
                preprocess_kwargs=None,
                mask_zeroth_kwargs=None,
                just_fixed_slit=True,
            )

            grp.write_slitlet_files()

        else:
            print(f"Found {slitlet_file}")

    # Combined spectra
    # slit_files = glob.glob(f"jw{obsid}*{res['apername'][0].split('_')[1].lower()}.fits")
    slit_files = []
    for row in res:
        slit = row["apername"].split("_")[1].lower()
        prefix = "_".join(row["filename"].split("_")[:4])
        slit_files += glob.glob(f"{prefix}*{slit}.fits")

    slit_files.sort()

    exposure_groups = slit_combine.split_visit_groups(
        slit_files, gratings=slit_combine.SPLINE_BAR_GRATINGS, **extract_kwargs
    )

    for g in list(exposure_groups.keys()):
        # print(g, len(exposure_groups[g]))
        if len(exposure_groups[g]) == 5:
            spl = g.split("_")
            files = [f for f in exposure_groups[g]]
            spl.insert(2, "set1")
            g1 = "_".join(spl)
            exposure_groups[g1] = files[0::2]

            spl[2] = "set2"
            g2 = "_".join(spl)
            exposure_groups[g2] = files[1::2]

            _ = exposure_groups.pop(g)

    # kwargs = dict(
    #     files=slit_files,
    #     # initial_theta=[0.5, 0.0, 0.0][:2],
    #     recenter_all=((recenter_type & 1) > 0),
    #     free_trace_offset=((recenter_type & 2) > 0),
    #     # extended_calibration_kwargs=None,
    #     exposure_groups=exposure_groups,
    #     # flag_trace_kwargs={"yslit": [-5, -1]},
    #     # shutter_offset=-3.0,
    #     # initial_theta=[2, 0, 0.], fit_params_kwargs={},
    #     # diffs=False, fix_params=True, initial_theta=[80, 0.0], fit_params_kwargs={},
    #     # fit_params_kwargs={},
    #     # sky_arrays=(sp['wave'], stat.statistic*2),
    #     # estimate_sky_kwargs={"make_plot": True, "high_clip": 100, "df":101, "mask_yslit": [[-4, 4], [12,118]]}, diffs=False,
    #     # diffs=True,
    #     protect_exception=False,
    #     do_gratings=["PRISM", "G395H", "G395M", "G235M", "G140M","G235H","G140H"],
    # )

    if "exposure_groups" not in extract_kwargs:
        extract_kwargs["exposure_groups"] = exposure_groups

    out_root = f"jw{obsid}-{version}"
    if "extended_calibration_kwargs" in extract_kwargs:
        if extract_kwargs["extended_calibration_kwargs"] is None:
            out_root += "_raw"

    extract_kwargs["root"] = out_root

    if "target" not in extract_kwargs:
        extract_kwargs["target"] = "test"

    if "recenter_type" in extract_kwargs:
        recenter_type = extract_kwargs.pop("recenter_type")
        extract_kwargs["recenter_all"] = (recenter_type & 1) > 0
        extract_kwargs["free_trace_offset"] = (recenter_type & 2) > 0

    if "get_sky" in extract_kwargs:
        get_sky = extract_kwargs.pop("get_sky")
        free_sky = dict(
            estimate_sky_kwargs={
                "make_plot": True,
                "high_clip": 100,
                "df": 101,
                "mask_yslit": [[-4, 4], [12, 118]],
            },
            diffs=False,
        )
        for k in free_sky:
            extract_kwargs[k] = free_sky[k]

    _ = slit_combine.extract_spectra(**extract_kwargs)

    all_slit_files = utils.glob_sorted("jw*_s*[12].fits")
    if len(all_slit_files) > 0:
        slit_info = cutout_info(all_slit_files)
        slit_info["root"] = s3_root
    else:
        slit_info = None

    try:
        spec_info = cloud.get_extraction_info(root=out_root, outroot=s3_root, key="")
    except:
        spec_info = None

    os.chdir(base_path)

    return files, s3_root, slit_info, spec_info


def cutout_info(files, clean=False):
    """
    Get information on slit cutouts from a particular NIRSpec exposure
    """
    import os
    import glob

    import numpy as np

    import subprocess
    import astropy.io.fits as pyfits
    import jwst.datamodels

    from grizli.aws import db
    from grizli import utils
    import msaexp.utils as msautils

    cols = db.SQL("select * from nirspec_cutouts limit 1")

    msg = f"{len(files)} slit cutout files"
    grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=True)

    cutout_info = []
    for file in files:
        msg = f"Get slitlet info from {file}"
        grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=True)

        # file_url = f'{s3_base}/slitlets/{root}/{file}'.replace('s3://', 'https://s3.amazonaws.com/')
        with pyfits.open(file) as im:
            hdata = {"file": file}  # , "root": root}
            h0 = im[0].header
            h1 = im[1].header

            for k in cols.colnames:
                for h in [h0, h1]:
                    if k.upper() in h:
                        hdata[k] = h[k.upper()]
                        break

        # Trace information
        with jwst.datamodels.open(file) as dm:
            sh = dm.data.shape
            hdata["quadrant"] = dm.quadrant

            _res = msautils.slit_trace_center(
                dm,
                with_source_xpos=False,
                with_source_ypos=False,
                index_offset=0.0,
            )

        _xtr, _ytr, _wtr, slit_ra, slit_dec = _res

        degree = 2
        xnorm = _xtr - sh[1] / 2  # / sh[1]
        oki = np.isfinite(xnorm + _ytr + _wtr)
        if oki.sum() > 3:
            trace_coeffs = np.polyfit(xnorm[oki], _ytr[oki], degree)

            hdata["trace_c0"] = trace_coeffs[0]
            hdata["trace_c1"] = trace_coeffs[1]
            hdata["trace_c2"] = trace_coeffs[2]

            hdata["x_min"] = int(_xtr[oki].min())
            hdata["x_max"] = int(_xtr[oki].max())

            hdata["wave_min"] = _wtr[oki].min()
            hdata["wave_max"] = _wtr[oki].max()

        else:
            hdata["trace_c0"] = 0.0
            hdata["trace_c1"] = 0.0
            hdata["trace_c2"] = 0.0

            hdata["x_min"] = 0
            hdata["x_max"] = 0

            hdata["wave_min"] = 0.0
            hdata["wave_max"] = 0.0

        cutout_info.append(hdata)

        if clean:
            os.remove(file)

    return utils.GTable(cutout_info)


if __name__ == "__main__":
    import sys

    # print(sys.argv, "--noclean" in sys.argv)
    status = run_one_fixed_slit(clean=("--noclean" not in sys.argv))
