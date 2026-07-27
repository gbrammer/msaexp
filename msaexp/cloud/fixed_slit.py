"""
Pipeline for processing NIRSpec Fixed-Slit observations
"""

import os
import glob
import inspect
import yaml
import time

import matplotlib.pyplot as plt
import numpy as np

from grizli.aws import db
import grizli.utils

from .. import pipeline_extended, slit_group, ifu, slit_combine
from .. import utils as msautils
from . import combine

QUERY_KWARGS = dict(
    trim_prism_nrs2=True,
    grating=None,
    filter=None,
)


def initialize_fs_helper_table():
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


def run_one_fixed_slit(row=None, clean=True, **kwargs):

    if row is None:
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
    row["rowid"] = int(row["rowid"])

    for k in list(row.keys()):
        if "yaml" in k:
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


def get_sky_from_other_slits(files, corr_max=25, df=32, minmax=(0.6, 5.4), sn_clip=[-4, 100], **kwargs):
    """ """
    import jwst.datamodels

    raw_data = []
    raw_err = []
    raw_wave = []

    for f in files:
        with jwst.datamodels.open(f) as dm:
            
            valid = (dm.dq & 1) == 0

            cal_ = msautils.slit_extended_flux_calibration(dm, **kwargs)

            if 0:
                flat_profile = msautils.fixed_slit_flat_field(
                    dm, apply=True, verbose=True
                )

            valid &= dm.phot_corr > 0
            valid &= dm.phot_corr < dm.phot_corr[valid].min() * corr_max
            valid &= np.isfinite((dm.data + dm.err) * dm.phot_corr)

            k = "{grating}-{filter}".format(**dm.meta.instrument.instance)

            sn = dm.data / dm.err
            med_sn = np.nanmedian(sn[valid])
            valid &= (sn > sn_clip[0]) & (sn < sn_clip[1] * med_sn)

            raw_wave.append(dm.wavelength[valid])
            raw_data.append((dm.data * dm.phot_corr)[valid])
            raw_err.append((dm.err * dm.phot_corr)[valid])

    raw_wave = np.hstack(raw_wave)
    raw_data = np.hstack(raw_data)
    raw_err = np.hstack(raw_err)

    bspl = grizli.utils.bspline_templates(
        raw_wave, df=df, minmax=minmax, get_matrix=True
    )

    mwave = np.linspace(*minmax, 1024)
    mbspl = grizli.utils.bspline_templates(mwave, df=df, minmax=minmax, get_matrix=True)

    c = np.linalg.lstsq((bspl.T / raw_err).T, (raw_data / raw_err), rcond=None)
    sky_model = mbspl.dot(c[0])
    sky_mask = mwave > 0
    sky_mask[np.interp(raw_wave, mwave, np.arange(1024)).astype(int)] = False

    return mwave, np.nan**sky_mask * sky_model * 0.5


def reduce_fixed_slit_obsid(
    obsid="06644003001",
    version="v4",
    query_kwargs=QUERY_KWARGS,
    extract_kwargs={},
    sky_kwargs=dict(corr_max=25, df=32, minmax=(0.6, 5.4), sn_clip=[-4, 100]),
    global_sky=False,
    **kwargs,
):
    """
    Full pipeline for FS
    """
    frame = inspect.currentframe()

    HOME = os.getcwd()

    if os.path.exists("/GrizliImaging"):
        base = "/GrizliImaging"
    else:
        base = os.getcwd()

    grizli.utils.LOGFILE = os.path.join(base, f"jw{obsid}-{version}_fs.log.txt")
    s3_root = f"jw{obsid}-{version}"

    args = grizli.utils.log_function_arguments(
        grizli.utils.LOGFILE,
        frame,
        "reduce_fixed_slit_obsid",
        ignore=["sky_arrays", "base", "HOME"],
    )

    reduce_path = os.path.join(base, f"jw{obsid}-{version}_fs")
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
    other_slit_files = []

    other_s200 = {
        "s200a1": "s200a2",
        "s200a2": "s200a1",
    }

    for row in res:
        slit = row["apername"].split("_")[1].lower()
        prefix = "_".join(row["filename"].split("_")[:4])
        files_i = glob.glob(f"{prefix}*{slit}.fits")
        slit_files += files_i
        if slit in other_s200:
            other_slit_files += [f.replace(slit, other_s200[slit]) for f in files_i]

    slit_files.sort()
    other_slit_files.sort()

    exposure_groups = slit_combine.split_visit_groups(
        slit_files, join=[0, 1, 3, 5], gratings=slit_combine.SPLINE_BAR_GRATINGS
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

    if len(other_slit_files) > 0:
        other_exposure_groups = slit_combine.split_visit_groups(
            other_slit_files,
            join=[0, 1, 3, 5],
            gratings=slit_combine.SPLINE_BAR_GRATINGS,
        )

        for g in list(other_exposure_groups.keys()):
            # print(g, len(exposure_groups[g]))
            if len(other_exposure_groups[g]) == 5:
                spl = g.split("_")
                files = [f for f in other_exposure_groups[g]]
                spl.insert(2, "set1")
                g1 = "_".join(spl)
                other_exposure_groups[g1] = files[0::2]

                spl[2] = "set2"
                g2 = "_".join(spl)
                other_exposure_groups[g2] = files[1::2]

                _ = other_exposure_groups.pop(g)

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

    if (len(other_slit_files) > 0):
        sky_arrays = get_sky_from_other_slits(
            other_slit_files, **sky_kwargs
        )
        if global_sky:
            extract_kwargs['sky_arrays'] = sky_arrays

    for g in exposure_groups:
        if len(exposure_groups[g]) == 1:
            if "diffs" not in extract_kwargs:
                extract_kwargs["diffs"] = False
            if "grating_diffs" not in extract_kwargs:
                extract_kwargs["grating_diffs"] = False

            msg = "group {g} has N=1 exposure, diffs={diffs} grating_diffs={grating_diffs}".format(
                g=g, **extract_kwargs
            )
            grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=True)

            extract_kwargs["make_2d_plots"] = False

            if (len(other_slit_files) > 0) & (0):
                extract_kwargs["sky_arrays"] = sky_arrays

    if "drizzle_kws" not in extract_kwargs:
        extract_kwargs["drizzle_kws"] = {
            'step': 1,
            'with_pathloss': True,
            'wave_sample': 1.05,
            'ny': 30,
            'dkws': {'oversample': 16, 'pixfrac': 0.8}
        }
 
    _ = slit_combine.extract_spectra(**extract_kwargs)

    # if len(other_slit_files) > 0:
    #     extract_kwargs["exposure_groups"] = other_exposure_groups
    #     _ = slit_combine.extract_spectra(**extract_kwargs)

    all_slit_files = msautils.glob_sorted("jw*_s*[12].fits")
    if len(all_slit_files) > 0:
        slit_info = cutout_info(all_slit_files)
        slit_info["root"] = s3_root
    else:
        slit_info = None

    try:
        spec_info = combine.get_extraction_info(
            root=out_root, outroot=s3_root, key=""
        )
    except:
        spec_info = None

    os.chdir(HOME)

    return files, s3_root, slit_info, spec_info


def cutout_info(files, clean=False):
    """
    Get information on slit cutouts from a particular NIRSpec exposure
    """
    import os
    import glob

    import subprocess
    import astropy.io.fits as pyfits
    import jwst.datamodels

    from grizli.aws import db

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

    return grizli.utils.GTable(cutout_info)


def exposure_slit_footprints(file, make_region_file=True):
    """
    """
    import pysiaf
    from pysiaf.utils import rotations
    
    import astropy.io.fits as pyfits
    
    import grizli.utils
    
    with pyfits.open(file) as im:
        h0 = im[0].header
        h1 = im[1].header

    nrs = pysiaf.Siaf('NIRSpec')

    ap = nrs[h0['APERNAME']]
    
    att = rotations.attitude(
        ap.V2Ref, ap.V3Ref, h1['RA_REF'], h1['DEC_REF'], h1['ROLL_REF']
    )

    slits = """
    NRS_S200A1_SLIT
    NRS_S200A2_SLIT
    NRS_S400A1_SLIT
    NRS_S1600A1_SLIT
    NRS_S200B1_SLIT
    """.strip().split("\n")
    
    regions = []
    
    for slit in slits:
        ap = nrs[slit.strip()]
        ap.set_attitude_matrix(att)
        sr = grizli.utils.SRegion(np.array(ap.corners('sky')))
        sr.label = slit.strip()
        regions.append(sr)

    if make_region_file:
        with open(file.replace('.fits', '.reg'), 'w') as fp:
            fp.write("icrs\n")
            for reg in regions:
                fp.write(reg.region[0] + "\n")

    return regions


if __name__ == "__main__":
    import sys

    # print(sys.argv, "--noclean" in sys.argv)
    run_one_fixed_slit(clean=("--noclean" not in sys.argv))
