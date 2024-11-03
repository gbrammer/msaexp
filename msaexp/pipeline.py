"""
Manual extractions of NIRSpec MSA spectra
"""

import os
import glob
import time
import traceback
from collections import OrderedDict

from tqdm import tqdm
import yaml

import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as pyfits
import astropy.units as u
from astropy.units.equivalencies import spectral_density

from grizli import utils, prep, jwst_utils

from . import utils as msautils
from . import msa

utils.set_warnings()

FLAM_UNIT = 1.0e-19 * u.erg / u.second / u.cm**2 / u.Angstrom
FNU_UNIT = u.microJansky

GRATINGS = ["prism", "g140m", "g140h", "g235m", "g235h", "g395m", "g395h"]
FILTERS = ["clear", "f070lp", "f100lp", "f170lp", "f290lp"]
ACQ_FILTERS = ["f140x", "f110w"]
DETECTORS = ["nrs1", "nrs2"]


__all__ = ["query_program", "exposure_groups", "NirspecPipeline"]


def query_program(
    prog=2767,
    download=True,
    detectors=DETECTORS,
    gratings=GRATINGS,
    filters=FILTERS,
    extensions=["s2d"],
    product="rate",
    extra_filters=[],
    levels=["2", "2a", "2b"],
):
    """
    Query and download MSA exposures for a given program from MAST

    Parameters
    ----------
    prog : int
        Program ID

    download : bool
        Download results

    detectors: list
        List of detectors to consider ('nrs1','nrs2')

    gratings : list
        List of gratings to consider ('prism', 'g140m', 'g140h', 'f235m',
        'g235h','g395m','g395h')

    filters : list
        List of filters to consider ('clear', 'f070lp', 'f100lp', 'f170lp',
        'f290lp')

    extensions : list
        File extensions to query.  ``s2d`` should have a one-to-one mapping
        with the level-1 countrate ``rate`` images, which are what we're after

    product : str
        MAST product to download, 'rate' or 'cal'

    extra_filters : list
        Additional query filters from, e.g., `mastquery.jwst.make_query_filter`

    levels : list
        List of ``productLevel`` entries to include in the query

    Returns
    -------
    res : `~astropy.table.Table`
        Query result

    """
    import mastquery.jwst
    import mastquery.utils

    query = []
    query += mastquery.jwst.make_query_filter("productLevel", values=levels)
    query += mastquery.jwst.make_program_filter([prog])

    if detectors is not None:
        query += mastquery.jwst.make_query_filter("detector", values=detectors)

    if gratings is not None:
        query += mastquery.jwst.make_query_filter("grating", values=gratings)

    if filters is not None:
        query += mastquery.jwst.make_query_filter("filter", values=filters)

    res = mastquery.jwst.query_jwst(
        instrument="NRS",
        filters=query + extra_filters,
        extensions=extensions,
        rates_and_cals=False,
    )

    if len(res) == 0:
        print("Nothing found.")
        return None

    # Unique rows
    rates = []
    unique_indices = []

    for i, uri in enumerate(res["dataURI"]):
        ui = uri.replace("s2d", product)
        for e in extensions:
            ui = ui.replace(e, product)

        if ui not in rates:
            unique_indices.append(i)

        rates.append(ui)

    res.remove_column("dataURI")
    res["dataURI"] = rates

    res = res[unique_indices]

    skip = np.in1d(res["msametfl"], [None])
    skip &= ~np.in1d(res["exp_type"], ["NRS_FIXEDSLIT"])

    if skip.sum() > 0:
        print(f"Remove {skip.sum()} rows with msametfl=None")
        res = res[~skip]

    skip = np.in1d(res["filter"], ["OPAQUE"])
    if skip.sum() > 0:
        print(f"Remove {skip.sum()} rows with filter=OPAQUE")
        res = res[~skip]

    if download:
        mastquery.utils.download_from_mast(rates[0:])

        download_msa_meta_files()

    return res


def download_msa_meta_files(files=None, do_download=True):
    """
    Download ``MSAMETFL`` files indicated in header keywords

    Parameters
    ----------
    files : list, optional
        List of files to consider. If not provided, it will search for all
        "*rate.fits" and "*cal.fits" files in the current directory.

    do_download : bool, optional
        Flag indicating whether to download the files. Default is True.

    Returns
    -------
    msa : list
        List of MSA files downloaded from MAST.

    """
    import mastquery.utils

    if files is None:
        files = glob.glob("*rate.fits")
        files += glob.glob("*cal.fits")
        files.sort()

    msa = []
    for file in files:
        with pyfits.open(file) as im:
            if "MSAMETFL" not in im[0].header:
                continue

            msa_file = im[0].header["MSAMETFL"]
            if not os.path.exists(msa_file):
                msa.append(f"mast:JWST/product/{msa_file}")

    if (len(msa) > 0) & (do_download):
        mastquery.utils.download_from_mast(msa)

    return msa


def exposure_groups(files=None, split_groups=True, verbose=True):
    """
    Group files by MSAMETFL, grating, filter, detector

    Parameters
    ----------
    files : list, None
        Explicit list of ``rate.fits`` files to consider.  Otherwise, `glob` in
        working directory

    split_groups : bool
        Split MSA exposures by both ``MSAMETFL`` and ``ACT_ID``, where the
        latter helps to group exposures in sets of 3 nodded files.

    verbose : bool
        Status messages

    Returns
    -------
    groups : dict
        Dictionary of exposure groups

    """

    if files is None:
        files = glob.glob("*rate.fits")
        files.sort()

    hkeys = [
        "filter",
        "grating",
        "effexptm",
        "detector",
        "msametfl",
        "targprop",
        "exp_type",
        "act_id",
    ]
    rows = []

    for file in files:
        with pyfits.open(file) as im:
            row = [file]
            for k in hkeys:
                if k in im[0].header:
                    row.append(im[0].header[k])
                else:
                    row.append(None)

            rows.append(row)

    tab = utils.GTable(names=["file"] + hkeys, rows=rows)
    keys = []
    for row in tab:
        if row["exp_type"] == "NRS_MSASPEC":
            keystr = "{msametfl}-{filter}-{grating}-{detector}"
            if split_groups:
                keystr = "{msametfl}-{act_id}-{filter}-{grating}-{detector}"

        else:
            keystr = "{targprop}-{filter}-{grating}-{detector}"

        key = keystr.format(**row)
        key = key.replace("_msa.fits", "").replace("_", "-")
        keys.append(key.lower())

    tab["key"] = keys

    un = utils.Unique(tab["key"], verbose=verbose)

    groups = OrderedDict()
    for v in un.values:
        groups[v] = [f for f in tab["file"][un[v]]]

    return groups


def primary_sources_by_group(groups):
    """
    Get list of sources from the MSA metadata files where
    ``primary_source == 'Y'`` and the source is listed in all exposures of the
    group

    Parameters
    ----------
    groups : dict
        Exposure grouping as returned by `msaexp.pipeline.exposure_groups`

    Returns
    -------
    src_list : dict
        List of ``source_id`` values by from ``groups``

    """
    src_list = {}
    for mode in groups:
        with pyfits.open(groups[mode][0]) as im:
            metf = msa.MSAMetafile(im[0].header["MSAMETFL"])
            sub = metf.shutter_table["primary_source"] == "Y"
            sub &= (
                metf.shutter_table["msa_metadata_id"]
                == im[0].header["MSAMETID"]
            )

            all_src = metf.shutter_table["source_id"][sub]

            un = utils.Unique(all_src, verbose=False)

            source_ids = np.array(un.values)[
                np.array(un.counts) == len(groups[mode])
            ]

            print(f"{mode} N={len(source_ids)} primary sources")

            src_list[mode] = source_ids

    return src_list


class SlitData:
    """
    Container for a list of SlitModel objects read from saved files
    """

    def __init__(
        self,
        file="jw02756001001_03101_00001_nrs1_rate.fits",
        step="phot",
        read=False,
        indices=None,
        targets=None,
    ):
        """
        Load slitlet objects from stored data files

        Parameters
        ----------
        file : str
            Parent exposure file

        step : str
            Calibration pipeline processing step of the output slitlet file

        read : bool
            Don't just find the filenames but also read the data

        indices : list, None
            Optional slit indices of specific files

        targets : list, None
            Optional target names of specific individual sources

        Attributes
        ----------
        slits : list
            List of `jwst.datamodels.SlitModel` objects

        files : list
            Filenames of slitlet data
        """

        self.slits = []

        if indices is not None:
            self.files = []
            for i in indices:
                fr = file.replace("rate.fits", f"{step}.{i:03d}.*.fits")
                fr = fr.replace("cal.fits", f"{step}.{i:03d}.*.fits")
                self.files += glob.glob(fr)
        elif targets is not None:
            self.files = []
            for target in targets:
                fr = file.replace("rate.fits", f"{step}.*{target}.fits")
                fr = fr.replace("cal.fits", f"{step}.*{target}.fits")
                self.files += glob.glob(fr)
        else:
            if file.endswith("_rate.fits"):
                fr = file.replace("rate.fits", f"{step}.*.fits")
            else:
                fr = file.replace("cal.fits", f"{step}.*.fits")

            self.files = glob.glob(fr)
            self.files.sort()

        if read:
            self.read_data()

    @property
    def N(self):
        """
        Number of slitlets
        """
        return len(self.files)

    def read_data(self, verbose=True):
        """
        Read files into SlitModel objects in `slits` attribute

        Parameters
        ----------
        verbose : bool, optional
            Prints a statement per read file if True (default).
        """
        from jwst.datamodels import SlitModel

        for file in self.files:
            self.slits.append(SlitModel(file))
            msg = f"msaexp.read_data: {file} {self.slits[-1].source_name}"
            utils.log_comment(
                utils.LOGFILE, msg, verbose=verbose, show_date=False
            )


def exposure_oneoverf(file, fix_rows=False, skip_completed=True, **kwargs):
    """
    Remove column-averaged 1/f striping

    Parameters
    ----------
    file : str
        Exposure (rate.fits) filename

    fix_rows : bool
        Apply 1/f correction to detector rows, as well as columns

    skip_completed : bool
        Skip steps that have already been completed

    Returns
    -------
    status : bool
        False if ``skip_completed`` and ``ONEFEXP`` keyword found, True if
        executed without exception.

    """
    with pyfits.open(file) as im:
        if "ONEFEXP" in im[0].header:
            if im[0].header["ONEFEXP"] & skip_completed:
                return False

    jwst_utils.exposure_oneoverf_correction(
        file, erode_mask=False, in_place=True, axis=0, deg_pix=256
    )

    if fix_rows:
        jwst_utils.exposure_oneoverf_correction(
            file, erode_mask=False, in_place=True, axis=1, deg_pix=2048
        )

    return True


def exposure_detector_effects(
    file, fix_rows=False, scale_rnoise=True, skip_completed=True, **kwargs
):
    """
    Remove 1/f striping, bias pedestal offset and rescale RNOISE extension

    Parameters
    ----------
    file : str
        Exposure (rate.fits) filename

    scale_rnoise : bool
        Calculate the RNOISE scaling

    skip_completed : bool
        Skip steps that have already been completed

    Returns
    -------
    status : bool
        False if ``skip_completed`` and ``ONEFEXP`` keyword found, True if
        executed without exception.

    """

    status = exposure_oneoverf(
        file, fix_rows=fix_rows, skip_completed=skip_completed, **kwargs
    )

    with pyfits.open(file, mode="update") as im:
        # bias
        dq = (im["DQ"].data & 1025) == 0

        if im[0].header["DETECTOR"] == "NRS2":
            dq[:, :1400] = False
        else:
            dq[:, 1400:] = False

        if ("MASKBIAS" in im[0].header) & skip_completed:
            bias_level = im[0].header["MASKBIAS"]
            msg = f"msaexp.preprocess : {file}  bias offset ="
            msg += f" {bias_level:7.3f} (from MASKBIAS)"
            utils.log_comment(
                utils.LOGFILE, msg, verbose=True, show_date=False
            )
        else:
            bias_level = np.nanmedian(im["SCI"].data[dq])
            msg = f"msaexp.preprocess : {file}  bias offset ="
            msg += f" {bias_level:7.3f}"
            utils.log_comment(
                utils.LOGFILE, msg, verbose=True, show_date=False
            )

            im["SCI"].data -= bias_level
            im[0].header["MASKBIAS"] = bias_level, "Bias level"
            im[0].header["MASKNPIX"] = (
                dq.sum(),
                "Number of pixels used for bias level",
            )

        if scale_rnoise:

            if ("SCLREADN" in im[0].header) & skip_completed:
                rms = im[0].header["SCLREADN"]
                msg = f"msaexp.preprocess : {file}    rms scale ="
                msg += f" {rms:>7.2f} (from SCLREADN)"
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=True, show_date=False
                )
            else:
                resid = im["SCI"].data / np.sqrt(im["VAR_RNOISE"].data)
                rms = utils.nmad(resid[dq])
                msg = f"msaexp.preprocess : {file}    rms scale ="
                msg += f"{rms:>7.2f}"
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=True, show_date=False
                )

                im[0].header["SCLREADN"] = rms, "RNOISE Scale factor"

                im["VAR_RNOISE"].data *= rms**2

                im[0].header["SCLRNPIX"] = (
                    dq.sum(),
                    "Number of pixels used for rnoise scale",
                )

        im.flush()

    return True


class NirspecPipeline:
    def __init__(
        self,
        mode="jw02767005001-02-clear-prism-nrs1",
        files=None,
        verbose=True,
        source_ids=None,
        slitlet_ids=None,
        pad=0,
        positive_ids=False,
        primary_sources=True,
    ):
        """
        Container class for NIRSpec data, generally in groups split by
        grating/filter/detector

        Parameters
        ----------
        mode : str
            Group / mode name, i.e., in the groups computed in
            `msaexp.pipeline.exposure_groups`

        files : list
            Explicit list of exposure or slitlet files

        verbose : bool
            Print status messages

        source_ids : list
            Specific list of source_id to trim from the MSA metadata file

        slitlet_ids: list
            Specific list of slitlet_id to trim from the MSA metadata file

        pad : int
            Number of dummy slits to pad the open slitlets

        positive_ids : bool
            If true, ignore background slits with source_id values <= 0

        primary_sources : bool
            Only extract sources with ``primary_source='Y'`` in the MSA
            metadata file.

        Attributes
        ----------
        mode : str
            Group name

        files : list
            List of exposure (``rate.fits``) filenames

        pipe : dict
            Dictionary with data from the various calibration pipeline
            products.  The final flux-calibrated data should generally be in
            ``pipe['phot']``.

        last_step : str
            The last step of the calibration pipeline that was run,
            e.g., 'phot'

        slitlets : dict
            Slitlet metadata

        msametfl : str
            Filename of the MSAMETFL metadata file

        msa : `msaexp.msa.MSAMetafile`
            MSA metadata object, perhaps that has been modified by the
            parameters ``source_ids``, ``slitlet_ids`` and ``pad`` above

        """
        from .msa import pad_msa_metafile, MSAMetafile

        self.mode = mode
        utils.LOGFILE = self.mode + ".log.txt"

        if files is None:
            groups = exposure_groups(verbose=verbose)
            if mode in groups:
                self.files = groups[mode]
            else:
                self.files = []
        else:
            self.files = files

        msg = f"msaexp.NirspecPipeline: Initialize {mode}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose, show_date=True)

        for file in self.files:
            msg = f"msaexp.NirspecPipeline: {file}"
            utils.log_comment(
                utils.LOGFILE, msg, verbose=verbose, show_date=False
            )

        self.pipe = OrderedDict()
        self.slitlets = OrderedDict()

        self.last_step = None

        self.msametfl = None
        self.msa = None

        self.init_source_ids = source_ids
        self.init_slitlet_ids = slitlet_ids

        if len(self.files) > 0:
            if os.path.exists(self.files[0]) & (~self.is_fixed_slit):
                with pyfits.open(self.files[0]) as im:
                    if "MSAMETFL" not in im[0].header:
                        msametfl = None
                    else:
                        msametfl = im[0].header["MSAMETFL"]

                if not os.path.exists(msametfl):
                    msametfl = None

                _do_pad = (pad > 0) | (source_ids is not None) | positive_ids
                _do_pad |= slitlet_ids is not None
                if _do_pad:
                    if (msametfl is not None) & (os.path.exists(msametfl)):
                        msametfl = pad_msa_metafile(
                            msametfl,
                            pad=pad,
                            positive_ids=positive_ids,
                            source_ids=source_ids,
                            slitlet_ids=slitlet_ids,
                            primary_sources=primary_sources,
                        )

                self.msametfl = msametfl
                msg = f"msaexp.NirspecPipeline: mode={mode}"
                msg += f" exp_type={self.exp_type}  msametfl={msametfl}"
                print(msg)

                _do_regions = (self.msametfl is not None) & (
                    source_ids is None
                )
                _do_regions &= slitlet_ids is None
                if _do_regions:
                    self.msa = MSAMetafile(self.msametfl)
                    with open(
                        self.msametfl.replace(".fits", ".reg"), "w"
                    ) as fp:
                        fp.write(
                            self.msa.regions_from_metafile(
                                as_string=True, with_bars=True
                            )
                        )

    @property
    def exp_type(self):
        """
        Get data EXP_TYPE

        Returns
        -------
        expt : str
            ``EXP_TYPE`` keyword from the first file in the file list.  Returns
            an empty string ``''`` if the file not found or of the keyword not
            found in the header.

        """
        expt = ""

        if len(self.files) > 0:
            if os.path.exists(self.files[0]):
                with pyfits.open(self.files[0]) as im:
                    if "EXP_TYPE" in im[0].header:
                        expt = im[0].header["EXP_TYPE"]

        return expt

    @property
    def is_fixed_slit(self):
        """
        Are data in fixed-slit mode with ``EXP_TYPE == 'NRS_FIXEDSLIT'``
        """
        return self.exp_type == "NRS_FIXEDSLIT"

    @property
    def grating(self):
        """
        Grating name, e.g., 'prism' from ``mode`` string
        """
        return "-".join(self.mode.split("-")[-3:-1])

    @property
    def detector(self):
        """
        Detector name, e.g., 'nrs1' from ``mode`` string
        """
        return self.mode.split("-")[-1]

    @property
    def N(self):
        """
        Number of *exposures* for this group
        """
        return len(self.files)

    @property
    def targets(self):
        """
        Reformatted target names for background and negative source names

        - ``background_{i}`` to ``b{i}``
        - ``xxx_-{i}`` to ``xxx_m{i}``

        """
        return list(self.slitlets.keys())

    def slit_index(self, key):
        """
        Index of ``key`` in ``self.slitlets``.

        Parameters
        ----------
        key : str
            The key to search for in ``self.slitlets``.

        Returns
        -------
        int or None
            The index of ``key`` in ``self.slitlets`` if it exists,
            otherwise None.

        """
        if key not in self.slitlets:
            return None
        else:
            return self.targets.index(key)

    def initialize_from_cals(self, key="phot", verbose=True):
        """
        Initialize processing object from cal.fits products

        Parameters
        ----------
        key : str
            The key to identify the calibration product to load.
            Default is "phot".

        verbose : bool, optional
            If True, print verbose output. Default is True.

        Returns
        -------
        None

        """
        import jwst.datamodels

        self.pipe[key] = []
        for file in self.files:
            msg = (
                f"msaexp.initialize_from_cals : load {file} as MultiSlitModel"
            )
            utils.log_comment(
                utils.LOGFILE, msg, verbose=verbose, show_date=True
            )
            self.pipe[key].append(jwst.datamodels.MultiSlitModel(file))

        self.last_step = key

    def preprocess(
        self,
        set_context=True,
        fix_rows=False,
        scale_rnoise=True,
        skip_completed=True,
        **kwargs,
    ):
        """
        Run grizli exposure-level preprocessing

        1. Snowball masking
        2. Apply 1/f correction
        3. Median "bias" removal
        4. Rescale RNOISE

        Parameters
        ----------
        set_context : bool
            Set the `CRDS_CTX` based on the keyword in the exposure files

        fix_rows : bool
            Apply 1/f correction to detector rows, as well as columns

        scale_rnoise : bool
            Calculate rescaling of the ``VAR_RNOISE`` data extension based on
            pixel statistics

        skip_completed : bool
            Skip steps that have already been completed

        Returns
        -------
        status : bool
            True if completed OK

        """

        if set_context:
            # Set CRDS_CTX to match the exposures
            if (os.getenv("CRDS_CTX") is None) | (set_context > 1):
                with pyfits.open(self.files[0]) as im:
                    _ctx = im[0].header["CRDS_CTX"]

                msg = f"msaexp.preprocess : set CRDS_CTX={_ctx}"
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=True, show_date=True
                )

                os.environ["CRDS_CTX"] = _ctx

        # Extra mask for snowballs
        prep.mask_snowballs(
            {"product": self.mode, "files": self.files},
            mask_bit=1024,
            instruments=["NIRSPEC"],
            snowball_erode=8,
            snowball_dilate=24,
        )

        # 1/f, bias & rnoise
        for file in self.files:
            exposure_detector_effects(
                file,
                fix_rows=fix_rows,
                scale_rnoise=scale_rnoise,
                skip_completed=skip_completed,
            )

        return True

    def run_jwst_pipeline(
        self, verbose=True, run_flag_open=True, run_bar_shadow=True, **kwargs
    ):
        """
        Steps taken from https://github.com/spacetelescope/jwebbinar_prep/blob/main/spec_mode/spec_mode_stage_2.ipynb

        See also https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html#calwebb-spec2

        - `AssignWcs`:  initialize WCS and populate slit bounding_box data
        - `Extract2dStep`:  identify slits and set slit WCS
        - `FlatFieldStep`:  slit-level flat field
        - `PathLossStep`:  NIRSpec path loss
        - `BarShadowStep`:  Bar shadow correction
        - `PhotomStep`:  Photometric calibration

        Parameters
        ----------
        verbose : bool
            Printing status messages

        run_flag_open : bool
            Run `jwst.msaflagopen.MSAFlagOpenStep` after `AssignWcsStep`

        run_bar_shadow : bool
            Run `jwst.barshadow.BarShadowStep` after `PathLossStep`

        Returns
        -------
        status : bool
            True if completed successfully

        """
        # AssignWcs
        import jwst.datamodels
        from jwst.assign_wcs import AssignWcsStep
        from jwst.msaflagopen import MSAFlagOpenStep
        from jwst.extract_2d import Extract2dStep
        from jwst.flatfield import FlatFieldStep
        from jwst.pathloss import PathLossStep
        from jwst.barshadow import BarShadowStep
        from jwst.photom import PhotomStep

        if "wcs" not in self.pipe:
            wstep = AssignWcsStep()

            wcs = []
            for file in self.files:
                with pyfits.open(file) as hdu:
                    hdu[0].header["MSAMETFL"] = self.msametfl
                    wcs.append(wstep.call(jwst.datamodels.ImageModel(hdu)))

            self.pipe["wcs"] = wcs

            # self.pipe['wcs'] = [wstep.call(jwst.datamodels.ImageModel(f))
            #                     for f in self.files]

        self.last_step = "wcs"

        # step = ImprintStep()
        # pipe['imp'] = [step.call(obj) for obj in pipe[last]]
        # last = 'imp'

        if ("open" not in self.pipe) & run_flag_open & (~self.is_fixed_slit):
            step = MSAFlagOpenStep()
            self.pipe["open"] = []

            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f"msaexp.jwst.MSAFlagOpenStep: {self.files[i]}"
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=verbose, show_date=True
                )

                self.pipe["open"].append(step.call(obj))

            self.last_step = "open"

        if "2d" not in self.pipe:
            step2d = Extract2dStep()
            self.pipe["2d"] = []

            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f"msaexp.jwst.Extract2dStep: {self.files[i]}"
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=verbose, show_date=True
                )

                self.pipe["2d"].append(step2d.call(obj))

        self.last_step = "2d"

        if "flat" not in self.pipe:
            flat_step = FlatFieldStep()
            self.pipe["flat"] = []

            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f"msaexp.jwst.FlatFieldStep: {self.files[i]}"
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=verbose, show_date=True
                )

                # Update metadata for fixed slit
                if self.is_fixed_slit:
                    for _slit in obj.slits:
                        msautils.update_slit_metadata(_slit)

                self.pipe["flat"].append(flat_step.call(obj))

        self.last_step = "flat"

        if "path" not in self.pipe:
            path_step = PathLossStep()
            self.pipe["path"] = []

            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f"msaexp.jwst.PathLossStep: {self.files[i]}"
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=verbose, show_date=True
                )

                self.pipe["path"].append(path_step.call(obj))

        self.last_step = "path"

        if run_bar_shadow & (~self.is_fixed_slit):
            bar_step = BarShadowStep()
            self.pipe["bar"] = []

            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f"msaexp.jwst.BarShadowStep: {self.files[i]}"
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=verbose, show_date=True
                )

                self.pipe["bar"].append(bar_step.call(obj))

            self.last_step = "bar"

        if "phot" not in self.pipe:
            phot_step = PhotomStep()
            self.pipe["phot"] = []

            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f"msaexp.jwst.PhotomStep: {self.files[i]}"
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=verbose, show_date=True
                )

                self.pipe["phot"].append(phot_step.call(obj))

        self.last_step = "phot"

        return True

    def save_slit_data(self, step="phot", verbose=True):
        """
        Save slit data to FITS

        Parameters
        ----------
        step : str
            The step of the pipeline from which the slit data is saved.

        verbose : bool, optional
            If True, print verbose output. Default is True.

        Returns
        -------
        bool
            True if the slit data is saved successfully.

        """
        from jwst.datamodels import SlitModel

        for j in range(self.N):
            for _name in self.slitlets:

                i = self.slitlets[_name]["slit_index"]
                si = self.slitlets[_name]["slitlet_id"]

                slit_file = self.files[j].replace(
                    "rate.fits", f"{step}.{si:03d}.{_name}.fits"
                )

                slit_file = slit_file.replace(
                    "cal.fits", f"{step}.{si:03d}.{_name}.fits"
                )

                msg = f"msaexp.save_slit_data: {slit_file} "
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=verbose, show_date=False
                )

                try:
                    dm = SlitModel(self.pipe[step][j].slits[i].instance)
                    dm.write(slit_file, overwrite=True)
                except:
                    utils.log_exception(
                        utils.LOGFILE, traceback, verbose=verbose
                    )

        return True

    def initialize_slit_metadata(
        self, use_yaml=True, yoffset=0, prof_sigma=1.8 / 2.35, skip=[]
    ):
        """
        Initialize the `slitlets` metadata dictionary

        Parameters
        ----------
        use_yaml : bool
            Read data from stored yaml file

        yoffset : float
            Default cross-dispersion offset for source centering

        prof_sigma : float
            Default profile width

        skip : list
            Exposures to skip

        Returns
        -------
        slitlets : dict
            Slitlet metadata

        """

        slitlets = OrderedDict()

        if self.last_step not in ["2d", "flat", "path", "phot"]:
            return slitlets

        msg = "# slit_index slitlet_id  source_name  source_ra  source_dec"
        msg += f"\n# {self.mode}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True, show_date=False)

        # Load saved data
        yaml_file = f"{self.mode}.slits.yaml"
        if os.path.exists(yaml_file) & use_yaml:
            msg = f"# Get slitlet data from {yaml_file}"
            utils.log_comment(
                utils.LOGFILE, msg, verbose=True, show_date=False
            )

            with open(yaml_file) as fp:
                yaml_data = yaml.load(fp, Loader=yaml.Loader)
        else:
            yaml_data = {}

        for i in range(len(self.pipe[self.last_step][0].slits)):
            bg = []
            src = 0
            meta = None

            for ii, o in enumerate(self.pipe[self.last_step]):
                _slit = o.slits[i]
                if _slit.source_name is None:
                    # Missing info?
                    src = ii
                    meta = _slit.instance
                    for k in [
                        "source_name",
                        "source_ra",
                        "source_dec",
                        "source_alias",
                        "source_id",
                        "source_type",
                        "source_xpos",
                        "source_ypos",
                    ]:

                        if k not in meta:
                            meta[k] = getattr(_slit, k)

                elif _slit.source_name.startswith("background"):
                    bg.append(ii)
                    bgmeta = _slit.instance
                else:
                    src = ii
                    meta = _slit.instance

            if meta is None:
                meta = bgmeta
                meta["slit_index"] = i
                meta["is_background"] = True
            else:
                meta["slit_index"] = i
                meta["is_background"] = False

            meta["bkg_index"] = bg
            meta["src_index"] = src
            meta["slit_index"] = i
            meta["prof_sigma"] = prof_sigma
            meta["yoffset"] = yoffset
            meta["skip"] = skip
            meta["redshift"] = None

            # is_fixed = meta['meta']['instrument']['lamp_mode'] == 'FIXEDSLIT'

            if self.is_fixed_slit & (meta["source_name"] is None):
                # Set source info for fixed slit targets
                targ = meta["meta"]["target"]
                _name = f"{targ['proposer_name'].lower()}_{_slit.name}".lower()
                meta["source_name"] = _name
                meta["source_ra"] = targ["ra"]
                meta["source_dec"] = targ["dec"]

            if "slitlet_id" not in meta:
                meta["slitlet_id"] = 9999

            _name = msautils.rename_source(meta["source_name"])
            if _name in yaml_data:
                for k in meta:
                    if k in yaml_data[_name]:
                        meta[k] = yaml_data[_name][k]

            # Make sure index is set
            meta["slit_index"] = i

            msg = "{slit_index:>4}  {slitlet_id:>4} {source_name:>12} "
            msg += " {source_ra:.6f} {source_dec:.6f}"
            utils.log_comment(
                utils.LOGFILE,
                msg.format(**meta),
                verbose=True,
                show_date=False,
            )

            # _name =  meta['source_name'].replace('background_','b')
            # _name = _name.replace('_-','_m')
            slitlets[_name] = {}
            for k in meta:
                slitlets[_name][k] = meta[k]

        return slitlets

    def slit_source_regions(self, color="magenta"):
        """
        Make region file of source positions

        **Deprecated**, see `~msaexp.msa.MSAMetafile.regions_from_metafile`.

        """
        regfile = f"{self.mode}.reg"
        with open(regfile, "w") as fp:
            fp.write(f"# {self.mode}\n")
            fp.write(f"global color={color}\nicrs\n")

            for s in self.slitlets:
                row = self.slitlets[s]
                if row["source_ra"] != 0:
                    ss = 'circle({source_ra:.6f},{source_dec:.6f},0.3") # '
                    ss += " text={{{source_name}}}\n"
                    fp.write(ss.format(**row))

        return regfile

    def set_background_slits(self, find_by_id=False):
        """
        Initialize elements in ``self.pipe['bkg']`` for background-subtracted
        slitlets

        Parameters
        ----------
        find_by_id : bool, optional
            If True, find background slits by source ID.
            If False, find background slits by slitlet ID.
            Default is False.

        Returns
        -------
        bool
            True if the initialization is successful.

        """
        # Get from slitlet_ids
        # indices = [self.slitlets[k]['slitlet_id'] for k in self.slitlets]
        if find_by_id:
            targets = [
                slit.source_id for slit in self.pipe[self.last_step][0].slits
            ]

            _data = self.load_slit_data(
                step=self.last_step, targets=targets, indices=None
            )
        else:
            indices = [
                slit.slitlet_id for slit in self.pipe[self.last_step][0].slits
            ]

            _data = self.load_slit_data(
                step=self.last_step, targets=None, indices=indices
            )

        if _data is None:
            targets = [
                slit.source_id for slit in self.pipe[self.last_step][0].slits
            ]

            _data = self.load_slit_data(
                step=self.last_step, targets=targets, indices=None
            )

        self.pipe["bkg"] = _data

        for j in range(self.N):
            for s in self.pipe["bkg"][j].slits:
                s.has_background = False

        return True

    def fit_profile(
        self,
        key,
        yoffset=None,
        prof_sigma=None,
        bounds=[(-5, 5), (1.4 / 2.35, 3.5 / 2.35)],
        min_delta=100,
        use_huber=True,
        verbose=True,
        **kwargs,
    ):
        """
        Fit for profile width and offset

        Parameters
        ----------
        key : str
            The key of the slitlet to fit the profile for.

        yoffset : float, optional
            The initial guess for the cross-dispersion offset. Default is None.

        prof_sigma : float, optional
            The initial guess for the profile width. Default is None.

        bounds : list, optional
            The bounds for the fitting parameters.
            Default is [(-5, 5), (1.4 / 2.35, 3.5 / 2.35)].

        min_delta : float, optional
            The minimum change in chi-square required to perform the fit.
            Default is 100.

        use_huber : bool, optional
            If True, use Huber loss function for fitting. Default is True.

        verbose : bool, optional
            If True, print verbose output. Default is True.

        Returns
        -------
        tuple
            A tuple containing the fitted profile width and offset.

        """
        from photutils.psf import IntegratedGaussianPRF
        from scipy.optimize import minimize

        prf = IntegratedGaussianPRF(sigma=2.8 / 2.35, x_0=0, y_0=0)

        if key.startswith("background"):
            bounds[0] = (-8, 8)

        _slit_data = self.extract_spectrum(
            key,
            fit_profile_params={},
            flux_unit=FLAM_UNIT,
            pstep=1,
            get_slit_data=True,
            yoffset=yoffset,
            prof_sigma=prof_sigma,
            show_sn=True,
            verbose=False,
        )

        _slit, _clean, _ivar, prof, y1, _wcs, chi, bad = _slit_data

        slitlet = self.slitlets[key]

        sh = _clean.shape
        yp, xp = np.indices(sh)

        _res = msautils.slit_trace_center(
            _slit, with_source_ypos=True, index_offset=0.5
        )

        xd, yd, _w, _, _ = _res

        ytr = slitlet["ytrace"] * 2 - yd

        def _objfun_fit_profile(params, data, ret):
            """
            Loss function for fitting profile parameters.

            Parameters
            ----------
            params : array-like
                The fitting parameters yoffset and prof_sigma
                (see 'fit_profile').

            data : tuple
                A tuple containing the data needed for the fitting.
                - xx0 : unused
                - yp : y pixel
                - ytr : trace
                - sh : shape
                _clean : cleaned array
                _ivar : inverse variance weight
                bad : mask
            ret : int
                The return value (see 'Returns').

            Returns
            -------
            array-like or float
                The desired output based on the value of `ret`:
                - 0: return the full chi array
                - 1: return the chi squared value
                - 2: return the loss value
                - other: return the fitted profile

            """
            from scipy.special import huber

            yoff = params[0]
            prf.sigma = params[1]

            xx0, yp, ytr, sh, _clean, _ivar, bad = data
            prof = prf(xx0, (yp - ytr - yoff).flatten()).reshape(sh)

            _wht = (prof**2 * _ivar).sum(axis=0)
            y1 = (_clean * prof * _ivar).sum(axis=0) / _wht
            dqi = (~bad) & (_wht > 0)

            # _sys = 1/(1/_ivar + (0.02*_clean)**2)

            y1[y1 < 0] = 0

            chi = (_clean - prof * y1) * np.sqrt(_ivar)
            # chi[prof*y1*np.sqrt(_ivar) < -5] *= 10
            chi2 = (chi[dqi] ** 2).sum()

            if ret == 0:
                return chi
            elif ret == 1:
                # print(params, chi2)
                return chi2
            elif ret == 2:
                loss = huber(3, chi)[dqi].sum()
                # print(params, loss)
                # loss +=  (y1 < 0).sum() - y1[np.isfinite(y1)].max()
                return loss
            else:
                return y1 * prof

        xx0 = yp.flatten() * 0.0
        data = (xx0, yp, ytr, sh, _clean, _ivar, bad)

        # compute dchi2 / dy and only do the fit if this is
        # greater than some threshold min_delta
        x0 = [slitlet["yoffset"] * 1.0, slitlet["prof_sigma"] * 1.0]
        x1 = [slitlet["yoffset"] * 1.0 + 1, slitlet["prof_sigma"] * 1.0]
        chi0 = _objfun_fit_profile(x0, data, 1 + use_huber)
        chi1 = _objfun_fit_profile(x1, data, 1 + use_huber)

        d0 = np.abs(chi1 - chi0)

        if d0 > min_delta:
            _res = minimize(
                _objfun_fit_profile,
                x0,
                args=(data, 1 + use_huber),
                method="slsqp",
                bounds=bounds,
                jac="2-point",
                options={"direc": np.eye(2, 2) * np.array([0.5, 0.2])},
            )

            dx = chi0 - _res.fun

            # print('xxx', x0, chi0, x1, chi1, _res.fun, _res)

            msg = "msaexp.fit_profile:     "
            msg += f" {key:<20}  (dchi2 = {d0:8.1f})"
            msg += f" yoffset = {_res.x[0]:.2f}  prof_sigma = {_res.x[1]:.2f}"
            msg += f" dchi2 = {dx:8.1f}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

            return _res.x, _res

        else:
            msg = "msaexp.fit_profile:     "
            msg += f" {key:<20}  (dchi2 = {d0:8.1f} <"
            msg += f" {min_delta} - skip)  yoffset = {x0[0]:.2f} "
            msg += f" prof_sigma = {x0[1]:.2f}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            return x0, None

    def get_background_slits(
        self, key, step="bkg", check_background=True, **kwargs
    ):
        """
        Get background-subtracted slitlets

        Parameters
        ----------
        key : str
            The key identifier of the slitlet

        step : str, optional
            The step in the pipeline to get the slitlets from. Default is "bkg"

        check_background : bool, optional
            If True, check if the background subtraction has been performed.
            Default is True.

        Returns
        -------
        slits : list
            List of `jwst.datamodels.slit.SlitModel` objects

        """
        if step not in self.pipe:
            return None

        if key not in self.slitlets:
            return None

        slits = []

        i = self.slit_index(key)  # slitlet['slit_index']

        for j in range(self.N):
            bsl = self.pipe[step][j].slits[i]
            if check_background:
                if hasattr(bsl, "has_background"):
                    if bsl.has_background:
                        slits.append(bsl)
            else:
                slits.append(bsl)

        return slits

    def drizzle_2d(self, key, drizzle_params={}, **kwargs):
        """
        Not used

        Drizzle the 2D spectra for a given slitlet.

        Parameters
        ----------
        key : str
            The key of the slitlet to drizzle the spectra for.

        drizzle_params : dict, optional
            Additional parameters to pass to the `ResampleSpecData` class.

        Returns
        -------
        `jwst.datamodels.ModelContainer`
            The drizzled 2D spectra.

        """
        from jwst.datamodels import ModelContainer
        from jwst.resample.resample_spec import ResampleSpecData

        slits = self.get_background_slits(key, **kwargs)
        if slits in [None, []]:
            return None

        bcont = ModelContainer()
        for s in slits:
            bcont.append(s)

        step = ResampleSpecData(bcont, **drizzle_params)
        result = step.do_drizzle()

        return result

    def get_slit_traces(self, verbose=True):
        """
        Set center of slit traces in `slitlets`.

        Parameters
        ----------
        verbose : bool, optional
            If True, print verbose output. Default is True.

        Returns
        -------
        None

        """

        msg = "msaexp.get_slit_traces: Run"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose, show_date=True)

        for key in self.slitlets:
            i = self.slit_index(key)
            dith_ref = 1000
            for j in range(self.N):
                slit = self.pipe[self.last_step][j].slits[i]
                dith = slit.meta.dither.instance

                # if dith['position_number'] == 1:
                # find lowest position number
                if dith["position_number"] < dith_ref:

                    dith_ref = dith["position_number"]
                    jref = j

                    _res = msautils.slit_trace_center(
                        slit, with_source_ypos=True, index_offset=0.5
                    )

                    xtr, ytr, wtr, slit_ra, slit_dec = _res

                    self.slitlets[key]["xtrace"] = xtr
                    self.slitlets[key]["ytrace"] = ytr
                    self.slitlets[key]["wtrace"] = wtr

                    self.slitlets[key]["slit_ra"] = slit_ra
                    self.slitlets[key]["slit_dec"] = slit_dec

                    # break

            msg = "msaexp.get_slit_traces: "

            msg += f"Trace set at index {jref} for {key}"

            utils.log_comment(
                utils.LOGFILE, msg, verbose=verbose, show_date=False
            )

    def extract_spectrum(
        self,
        key,
        slit_key=None,
        prof_sigma=None,
        fit_profile_params={"min_delta": 100},
        pstep=1.0,
        show_sn=True,
        flux_unit=FNU_UNIT,
        vmax=0.2,
        yoffset=None,
        skip=None,
        bad_dq_bits=(1 | 1024),
        clip_sigma=-4,
        ntrim=5,
        get_slit_data=False,
        verbose=False,
        center2d=False,
        trace_sign=1,
        min_dyoffset=0.2,
        **kwargs):
        """
        Main function for extracting 2D/1D spectra from individual slitlets

        Parameters
        ----------

        key: str
            The key of the slitlet to extract the spectrum from.

        slit_key: str
            The key of the slit to use for extraction.
            If None, the last step in the pipeline will be used.

        prof_sigma: float
            The sigma parameter for the IntegratedGaussianPRF.
            If None, the value from the slitlet will be used.

        fit_profile_params: dict
            Additional parameters for fitting the profile.
            Default is {"min_delta": 100}.

        pstep: float
            The step size for the running median. Default is 1.0.

        show_sn: bool
            Whether to show the signal-to-noise ratio. Default is True.

        flux_unit: str
            The unit of the flux. Default is FNU_UNIT.

        vmax: float
            The maximum value for the plot. Default is 0.2.

        yoffset: float
            The y offset for the slit.
            If None, the value from the slitlet will be used.

        skip: list
            A list of indices to skip. Default is None.

        bad_dq_bits: int
            The bad dq bits to mask. Default is (1 | 1024).

        clip_sigma: float
            The sigma value for clipping. Default is -4.

        ntrim: int
            The number of points to trim. Default is 5.

        get_slit_data: bool
            Whether to get the slit data. Default is False.

        verbose: bool
            Whether to print verbose output. Default is False.

        center2d: bool
            Whether to center the 2D spectrum. Default is False.

        trace_sign: int
            The sign of the trace. Default is 1.

        min_dyoffset: float
            The minimum value for the y offset. Default is 0.2.

        Returns
        -------
        If 'key' is not found in 'slitlets', returns None.

        Else:

        slitlet: object
            The slitlet object.

        tabs: list
            A list of tables containing the extracted spectra.

        full_tab: object
            The combined table of all extracted spectra.

        fig: object
            The `matplotlib` figure object.

        """
        from photutils.psf import IntegratedGaussianPRF
        import eazy.utils

        if key not in self.slitlets:
            print(f"{self.mode}: {key} not found in slitlets")

            return None

        if fit_profile_params:
            try:
                _xprof, _res = self.fit_profile(key, **fit_profile_params)
            except TypeError:
                _res = None

            if _res is not None:
                if _res.success:
                    yoffset = None
                    prof_sigma = None
                    self.slitlets[key]["yoffset"] = float(_res.x[0])
                    self.slitlets[key]["prof_sigma"] = float(_res.x[1])

        slitlet = self.slitlets[key]
        # slitlet['bkg_index'], slitlet['src_index'], slitlet['slit_index']

        i = self.slit_index(key)  # slitlet['slit_index']

        for j in range(self.N):
            if "bkg" in self.pipe:
                self.pipe["bkg"][j].slits[i].has_background = False

        if slit_key is None:
            slit_key = self.last_step

        if yoffset is None:
            yoffset = slitlet["yoffset"]
        else:
            slitlet["yoffset"] = yoffset

        if prof_sigma is None:
            prof_sigma = slitlet["prof_sigma"]
        else:
            slitlet["prof_sigma"] = prof_sigma

        prf = IntegratedGaussianPRF(
            sigma=prof_sigma,
            x_0=0.0,
            y_0=0.0,
            flux=prof_sigma * np.sqrt(2 * np.pi),
        )

        if skip is None:
            skip = slitlet["skip"]
        else:
            slitlet["skip"] = skip

        pipe = self.pipe[slit_key]

        if not get_slit_data:
            heights = [1] * self.N + [2]
            fig, axes = plt.subplots(
                self.N + 1,
                1,
                figsize=(12, self.N + 2),
                sharex=True,
                sharey=False,
                gridspec_kw={"height_ratios": heights},
            )

            a1 = axes[-1]

        sci_slit = pipe[slitlet["src_index"]].slits[i]

        x = np.arange(sci_slit.data.shape[1])
        rs, ds, ws = sci_slit.meta.wcs.forward_transform(x, x * 0 + 5)

        tabs = []
        yavj = None

        all_sci = []
        all_ivar = []
        all_prof = []

        for ip in range(self.N):
            if ip in skip:
                continue

            # First exposure
            _slit = pipe[ip].slits[i]

            if "bkg" in self.pipe:
                _bkg_slit = self.pipe["bkg"][ip].slits[i]
            else:
                _bkg_slit = None

            sh = _slit.data.shape

            _dq = (pipe[ip].slits[i].dq & bad_dq_bits) == 0
            _sci = pipe[ip].slits[i].data
            _ivar = 1 / pipe[ip].slits[i].err ** 2
            _dq &= _sci * np.sqrt(_ivar) > clip_sigma
            _sci[~_dq] = 0
            _ivar[~_dq] = 0

            _bkg = np.zeros(_sci.shape)
            _bkgn = np.zeros(_sci.shape, dtype=int)

            _wcs = _slit.meta.wcs
            d2w = _wcs.get_transform("detector", "world")

            yp, xp = np.indices(sh)

            _res = msautils.slit_trace_center(
                _slit, with_source_ypos=True, index_offset=0.5
            )

            xd, yd, _w, _, _ = _res

            xtr = xd
            if trace_sign > 0:
                ytr = slitlet["ytrace"] * 2 - yd + yoffset
            else:
                ytr = yd + yoffset

            _ras, _des, ws = d2w(xtr, ytr)

            for j in range(self.N):
                if j == ip:
                    continue

                _sbg = pipe[j].slits[i]
                dy_off = _sbg.meta.dither.y_offset - _slit.meta.dither.y_offset

                if np.abs(dy_off) < min_dyoffset:
                    continue

                _dq = (_sbg.dq & bad_dq_bits) == 0
                try:
                    _bkg[_dq] += _sbg.data[_dq]
                    _bkgn[_dq] += 1

                except ValueError:
                    continue

            # before this was in the loop.
            # The if statement preserves the indentation
            if True:
                if (np.nanmax(_bkg) == 0) & (not get_slit_data):
                    axi = axes[ip]
                    axi.imshow(
                        _sci * 0.0, vmin=-0.05, vmax=0.3, origin="lower"
                    )

                    continue

                y0 = np.nanmean(ytr)
                if not np.isfinite(y0):
                    continue

                if np.isfinite(ws).sum() == 0:
                    continue

                _clean = _sci - _bkg / _bkgn

                bad = ~np.isfinite(_clean)
                bad |= _bkg == 0
                bad |= _sci == 0

                _ivar[bad] = 0
                _clean[bad] = 0

                if _bkg_slit is not None:
                    _bkg_slit.data = _clean * 1
                    _bkg_slit.dq = (bad * 1).astype(_bkg_slit.dq.dtype)
                    _bkg_slit.has_background = True
                    _bkg_slit._bkg = _bkg
                    _bkg_slit._bkgn = _bkgn

                prof = prf(yp.flatten() * 0, (yp - ytr).flatten()).reshape(sh)

                if 1:
                    _wht = (prof**2 * _ivar).sum(axis=0)
                    y1 = (_clean * prof * _ivar).sum(axis=0) / _wht
                    _err = 1 / np.sqrt(_wht) * np.sqrt(3.0 / 2)
                else:
                    _err = y1 * 0.1

                if get_slit_data:
                    chi = (_clean - prof * y1) * np.sqrt(_ivar)
                    return _slit, _clean, _ivar, prof, y1, _wcs, chi, bad

                axi = axes[ip]

                _running = eazy.utils.running_median(
                    (yp - ytr)[~bad],
                    (prof * y1)[~bad],
                    bins=np.arange(-5, 5.01, pstep),
                )
                pscl = np.nanmax(_running[1])
                if pscl < 0:
                    pscl = 0
                else:
                    pscl = 1 / pscl

                _run_flux = np.maximum(_running[1], 0)
                axi.step(
                    x.max() + _run_flux * pscl * 0.08 * xtr.max(),
                    _running[0] * 2 + np.nanmedian(ytr),
                    color="r",
                    alpha=0.5,
                    zorder=1000,
                )

                _running = eazy.utils.running_median(
                    (yp - ytr)[~bad],
                    _clean[~bad],
                    bins=np.arange(-5, 5.01, pstep),
                )
                _run_flux = np.maximum(_running[1], 0)
                axi.step(
                    x.max() + _run_flux * pscl * 0.08 * xtr.max(),
                    _running[0] * 2 + np.nanmedian(ytr),
                    color="k",
                    alpha=0.5,
                    zorder=1000,
                )

                # aa.set_xlim(-5,5)

                if slit_key in ["flat", "path"]:
                    scl = 1e12
                else:
                    scl = 1.0

                if slit_key == "phot":
                    pixel_area = _slit.meta.photometry.pixelarea_steradians
                    _ukw = {}
                    _ukw["equivalencies"] = spectral_density(ws * u.micron)

                    eflam = (_err * u.MJy * pixel_area).to(flux_unit, **_ukw)
                    flam = (y1 * u.MJy * pixel_area).to(flux_unit, **_ukw)

                    y1 = flam.value
                    _err = eflam.value

                if yavj is None:
                    yavj = y1 * 1
                    xy0 = ws[sh[1] // 2]
                    shx = None
                    dx = 0.0
                else:
                    dx = (ws[sh[1] // 2] - xy0) / np.diff(ws)[sh[1] // 2]
                    try:
                        shx = int(np.round(dx))
                        if verbose:
                            print("x shift: ", dx, shx)
                        yavj += np.roll(y1, shx)
                    except ValueError:
                        print("x shift err")
                        shx = None

                if shx is not None:
                    dxp = shx
                    if verbose:
                        print("Roll 2d", shx)

                    all_sci.append(np.roll(_clean, shx, axis=1))
                    all_ivar.append(np.roll(_ivar, shx, axis=1))
                    all_prof.append(np.roll(prof, shx, axis=1))

                else:
                    dxp = 0
                    all_sci.append(_clean)
                    all_ivar.append(_ivar)
                    all_prof.append(prof)

                a1.plot(xtr + dxp, y1 * scl, color="k", alpha=0.2)
                a1.plot(xtr + dxp, _err * scl, color="pink", alpha=0.2)

                tab = utils.GTable()
                tab["wave"] = ws * 1
                tab["wave"].unit = u.micron

                tab["flux"] = y1
                tab["err"] = _err
                tab["xtrace"] = xtr
                tab["ytrace"] = ytr

                if slit_key == "phot":
                    tab["flux"].unit = flux_unit
                    tab["err"].unit = flux_unit

                tabs.append(tab)

                vmax = np.clip(
                    1.5 * np.nanpercentile(_clean[_ivar > 0], 95), 0.02, 0.5
                )
                if show_sn is None:
                    axi.imshow(
                        _clean,
                        vmin=-vmax / 2 / scl,
                        vmax=vmax / scl,
                        origin="lower",
                    )
                else:
                    if show_sn in [True, 1]:
                        vmax = np.clip(
                            1.5
                            * np.nanpercentile(
                                (_clean * np.sqrt(_ivar))[_ivar > 0], 95
                            ),
                            5,
                            30,
                        )
                        vmin = -1
                    else:
                        vmin, vmax = show_sn[:2]

                    axi.imshow(
                        _clean * np.sqrt(_ivar),
                        vmin=vmin,
                        vmax=vmax,
                        origin="lower",
                    )

                axi.plot(xtr, ytr + 2, color="w", alpha=0.5)
                axi.plot(xtr, ytr - 2, color="w", alpha=0.5)

                if center2d:
                    axi.set_ylim(y0 - 8, y0 + 8)

        if yavj is None:
            plt.close(fig)
            return None, None, None, None

        yavj /= self.N

        # Combined optimal extraction
        sall_prof = np.vstack(all_prof)
        sall_sci = np.vstack(all_sci)
        sall_ivar = np.vstack(all_ivar)
        sall_ivar[sall_sci * np.sqrt(sall_ivar) < clip_sigma] = 0

        all_num = sall_prof * sall_sci * sall_ivar
        all_den = (sall_prof**2 * sall_ivar).sum(axis=0)
        all_flux = all_num.sum(axis=0) / all_den
        all_err = 1 / np.sqrt(all_den)

        _wave = tabs[0]["wave"]

        if slit_key == "phot":
            pixel_area = _slit.meta.photometry.pixelarea_steradians
            _ukw = {}
            _ukw["equivalencies"] = spectral_density(_wave.value * u.micron)

            all_err = (all_err * u.MJy * pixel_area).to(flux_unit, **_ukw)
            all_flux = (all_flux * u.MJy * pixel_area).to(flux_unit, **_ukw)

        if ntrim > 0:
            _ok = np.isfinite(all_err + all_flux) & (all_err > 0)
            if _ok.sum() > ntrim * 2:
                _oki = np.where(_ok)[0]
                all_flux[_oki[-ntrim:]] = np.nan
                all_err[_oki[-ntrim:]] = np.nan
                all_flux[_oki[:ntrim]] = np.nan
                all_err[_oki[:ntrim]] = np.nan

        full_tab = utils.GTable()
        full_tab["wave"] = _wave
        full_tab["flux"] = all_flux
        full_tab["err"] = all_err

        if slit_key == "phot":
            full_tab["flux"].unit = flux_unit
            full_tab["err"].unit = flux_unit

        for t in tabs:
            t.meta["prof_sigma"] = prof_sigma
            t.meta["yoffset"] = yoffset
            t.meta["bad_dq_bits"] = bad_dq_bits
            t.meta["clip_sigma"] = clip_sigma

            for k in slitlet:
                if k.startswith("source"):
                    t.meta[k] = slitlet[k]

            for _m in [_slit.meta.instrument, _slit.meta.exposure]:
                _mi = _m.instance
                for k in _mi:
                    t.meta[k] = _mi[k]

        for k in tabs[0].meta:
            full_tab.meta[k] = tabs[0].meta[k]

        full_tab.meta["exptime"] = 0.0
        full_tab.meta["ncombined"] = 0
        for t in tabs:
            full_tab.meta["exptime"] += t.meta["effective_exposure_time"]
            full_tab.meta["ncombined"] += 1

        wx = np.append(np.array([0.7, 0.8]), np.arange(1, 5.6, 0.5))

        wx = wx[(wx > np.nanmin(_wave)) & (wx < np.nanmax(_wave))]
        if len(wx) < 2:
            wx = np.arange(0.6, 5.6, 0.1)
            wx = wx[(wx > np.nanmin(_wave)) & (wx < np.nanmax(_wave))]

        xx = np.interp(wx, _wave, tabs[0]["xtrace"])
        for i, ax in enumerate(axes[:-1]):
            ax.set_aspect("auto")
            ax.set_yticklabels([])

        for i, ax in enumerate(axes):

            ax.set_xticks(xx)
            # ax.set_xlim(0, _bkg.shape[1]-1)
            ax.set_xlim(0, _bkg.shape[1] * 1.08)
            if i == self.N:
                ax.set_xticklabels([f"{wi:.1f}" for wi in wx])
            else:
                ax.set_xticklabels([])

        # axes[1].set_ylabel(mode + key)
        a1.text(
            0.02,
            0.95,
            self.mode,
            va="top",
            ha="left",
            transform=a1.transAxes,
            fontsize=8,
            bbox={"fc": "w", "ec": "None", "alpha": 0.7},
        )

        a1.text(
            0.98,
            0.95,
            key,
            va="top",
            ha="right",
            transform=a1.transAxes,
            fontsize=8,
            bbox={"fc": "w", "ec": "None", "alpha": 0.7},
        )

        ymax = 2.2 * np.nanpercentile(full_tab["flux"], 98)
        a1.plot(
            tabs[0]["xtrace"], full_tab["flux"] * scl, color="k", alpha=0.8
        )
        a1.grid()

        a1.set_ylim(-0.15 * ymax, ymax)
        a1.set_xlabel(r"$\lambda$, observed $\mu\mathrm{m}$")

        if slit_key == "phot":
            a1.set_ylabel(full_tab["flux"].unit.to_string(format="latex"))

        # timestamp
        fig.text(
            0.015 * 12.0 / 12,
            0.02,
            f"dy={yoffset:.2f} sig={prof_sigma:.2f}",
            ha="left",
            va="bottom",
            transform=fig.transFigure,
            fontsize=8,
        )

        fig.text(
            1 - 0.015 * 12.0 / 12,
            0.02,
            time.ctime(),
            ha="right",
            va="bottom",
            transform=fig.transFigure,
            fontsize=6,
        )

        fig.tight_layout(pad=0.5)

        return slitlet, tabs, full_tab, fig

    def extract_all_slits(self, keys=None, verbose=True, close=True, **kwargs):
        """
        Extract all spectra and make diagnostic figures

        Parameters
        ----------
        keys : list, optional
            List of keys corresponding to the slitlets to extract spectra from.
            If not provided, spectra will be extracted from all slitlets.
        verbose : bool, optional
            Print status messages. Default is True.
        close : bool, optional
            Close all figures after saving. Default is True.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the
            `extract_spectrum` method.

        Returns
        -------
        None

        """
        slitlets = self.slitlets

        if keys is None:
            keys = list(slitlets.keys())

        for i, key in enumerate(keys):
            out = f"{self.mode}-{key}.spec"

            try:
                _ext = self.extract_spectrum(key, **kwargs)

                slitlet, tabs, full_tab, fig = _ext

                fig.savefig(out + ".png")
                full_tab.write(out + ".fits", overwrite=True)
                msg = f"msaexp.extract_spectrum: {key}"
                utils.log_comment(
                    utils.LOGFILE, msg, verbose=verbose, show_date=False
                )

                if close:
                    plt.close("all")

            except:
                print(f"{self.mode} {key} Failed")
                utils.log_exception(out + ".failed", traceback, verbose=True)
                if close:
                    plt.close("all")

    def get_slit_polygons(self, include_yoffset=False):
        """
        Get slit polygon regions using slit wcs

        **Deprecated**, use `~msaexp.msa.MSAMetafile.regions_from_metafile`.

        """
        from tqdm import tqdm

        slit_key = self.last_step
        pipe = self.pipe[slit_key]

        regs = []
        for j in range(self.N):
            regs.append([])

        for key in tqdm(self.slitlets):
            slitlet = self.slitlets[key]
            # slitlet['bkg_index'], slitlet['src_index'], slitlet['slit_index']

            i = self.slit_index(key)  # slitlet['slit_index']

            yoffset = slitlet["yoffset"]

            for j in range(self.N):
                _slit = pipe[j].slits[i]
                _wcs = _slit.meta.wcs
                sh = _slit.data.shape

                # _dq = (pipe[j].slits[i].dq & bad_dq_bits) == 0
                # _sci = pipe[j].slits[i].data

                yp, xp = np.indices(sh)
                x0 = np.ones(sh[0]) * sh[1] / 2.0
                y0 = np.arange(sh[0])
                r0, d0, w0 = _wcs.forward_transform(x0, y0)

                tr = _wcs.get_transform(_wcs.slit_frame, _wcs.world)

                rl, dl, _ = tr(-0.5, 0, 1)
                rr, dr, _ = tr(0.5, 0, 1)

                # yoffset along slit, as 0.1" pixels along to 0.46" slits
                if include_yoffset:
                    ro, do, _ = tr(-0.5, 0.1 / 0.46 * yoffset, 1)
                    rs = ro - rl
                    ds = do - dl
                else:
                    rs = ds = 0.0

                rw = rr - rl
                dw = dr - dl

                ok = np.isfinite(d0)

                xy = np.array(
                    [
                        np.append(r0[ok] + rs, r0[ok][::-1] + rs + rw),
                        np.append(d0[ok] + ds, d0[ok][::-1] + ds + dw),
                    ]
                )

                sr = utils.SRegion(xy)
                _name = slitlet["source_name"]

                if "_-" in _name:
                    sr.ds9_properties = "color=yellow"
                elif "background" in _name:
                    sr.ds9_properties = "color=white"
                else:
                    sr.ds9_properties = "color=green"

                if j == 0:
                    sr.label = _name
                    sr.ds9_properties += " width=2"

                regs[j].append(sr)

        _slitreg = f"{self.mode}.slits.reg"
        print(_slitreg)
        with open(_slitreg, "w") as fp:
            for j in range(self.N):
                fp.write("icrs\n")
                for sr in regs[j]:
                    fp.write(sr.region[0] + "\n")

    def load_slit_data(
        self, step="phot", verbose=True, indices=None, targets=None
    ):
        """
        Load slitlet data from saved files.  This script runs
        `msaexp.pipeline.SlitData` for each exposure file in the group.

        Parameters
        ----------
        step : str
            Calibration pipeline processing step

        verbose : bool
            Print status messages

        indices : list, None
            Optional slit indices of specific files

        targets : list, None
            Optional target names of specific individual sources

        Returns
        -------
        slit_lists : list
            List of `msaexp.pipeline.SlitData` objects containing the loaded
            slitlet data.
        """
        slit_lists = [
            SlitData(
                file, step=step, read=False, indices=indices, targets=targets
            )
            for file in self.files
        ]

        counts = [sl.N for sl in slit_lists]
        if (counts[0] > 0) & (np.allclose(counts, np.min(counts))):
            for sl in slit_lists:
                sl.read_data(verbose=verbose)

            return slit_lists

        else:
            return None

    def parse_slit_info(self, write=True):
        """
        Parse information from / to ``{mode}.slits.yaml`` file.

        Parameters
        ----------
        write : bool, optional
            Whether to write the parsed information back to the YAML file.
            Default is True.

        Returns
        -------
        info : dict
            A dictionary containing the parsed information from the YAML file.

        """
        import yaml

        keys = [
            "source_name",
            "source_ra",
            "source_dec",
            "skip",
            "yoffset",
            "prof_sigma",
            "redshift",
            "is_background",
            "slit_index",
            "src_index",
            "bkg_index",
            "slit_ra",
            "slit_dec",
        ]

        yaml_file = f"{self.mode}.slits.yaml"
        if os.path.exists(yaml_file):
            with open(yaml_file) as fp:
                info = yaml.load(fp, Loader=yaml.Loader)
        else:
            info = {}

        for _src in self.slitlets:
            s = self.slitlets[_src]
            info[_src] = {}
            for k in keys:
                if k in s:
                    info[_src][k] = s[k]

            if len(info[_src]["skip"]) == 0:
                info[_src]["skip"] = []

        if write:
            with open(yaml_file, "w") as fp:
                yaml.dump(info, stream=fp)

            print(yaml_file)

        return info

    def full_pipeline(
        self,
        load_saved="phot",
        run_preprocess=True,
        run_extractions=True,
        indices=None,
        targets=None,
        initialize_bkg=True,
        make_regions=True,
        use_yaml_metadata=True,
        **kwargs,
    ):
        """
        Run all steps through extractions

        Parameters
        ----------
        load_saved : str, optional
            The calibration pipeline processing step to load saved data from.
            If provided, the pipeline will skip the preprocessing
            and JWST pipeline steps.
            Default is None.

        run_preprocess : bool, optional
            Whether to run the preprocessing step. Default is True.

        run_extractions : bool, optional
            Whether to run the extraction step. Default is True.

        indices : list, optional
            List of slit indices of specific files to process. Default is None.

        targets : list, optional
            List of target names of specific individual sources to process.
            Default is None.

        initialize_bkg : bool, optional
            Whether to initialize the background slits. Default is True.

        make_regions : bool, optional
            Whether to create slit source regions. Default is True.

        use_yaml_metadata : bool, optional
            Whether to use YAML metadata for initializing slit metadata.
            Default is True.

        **kwargs : dict, optional
            Additional keyword arguments to pass to the preprocessing and
            extraction steps.

        Returns
        -------
        None

        """

        if load_saved is not None:
            if load_saved in self.pipe:
                status = self.pipe[load_saved]
            else:
                status = self.load_slit_data(
                    step=load_saved, indices=indices, targets=targets
                )
        else:
            status = None

        if status is not None:
            # Have loaded saved data
            make_regions = False
            self.pipe[load_saved] = status
            self.last_step = load_saved
        elif targets is not None:
            print(f"Targets {targets} not found")
            return True

        else:
            if self.files[0].endswith("_cal.fits"):
                self.initialize_from_cals()
            else:
                if run_preprocess:
                    self.preprocess(**kwargs)

                self.run_jwst_pipeline(**kwargs)

        self.slitlets = self.initialize_slit_metadata(
            use_yaml=use_yaml_metadata
        )

        self.get_slit_traces()

        if run_extractions:
            self.extract_all_slits(**kwargs)

        if make_regions:
            self.slit_source_regions()

        self.parse_slit_info(write=True)

        if status is None:
            self.save_slit_data()

        if initialize_bkg:
            print("Set background slits")
            self.set_background_slits()


def make_summary_tables(root="msaexp", zout=None):
    """
    Make a summary table of all extracted sources

    Parameters
    ----------
    root : str
        The root directory where the data is stored. Default is "msaexp".

    zout : astropy.table.Table
        Optional table containing photometric redshift information. Default is None.

    Returns
    -------
    tabs : list
        List of astropy tables containing the extracted source information.

    full : astropy.table.Table

        Combined astropy table containing all the extracted source information.

    """
    import yaml
    import astropy.table

    groups = exposure_groups()

    tabs = []
    for mode in groups:
        # mode = 'jw02767005001-02-clear-prism-nrs1'

        yaml_file = f"{mode}.slits.yaml"
        if not os.path.exists(yaml_file):
            print(f"Skip {yaml_file}")
            continue

        with open(yaml_file) as fp:
            yaml_data = yaml.load(fp, Loader=yaml.Loader)

        cols = []
        rows = []
        for k in yaml_data:
            row = []
            for c in [
                "source_name",
                "source_ra",
                "source_dec",
                "yoffset",
                "prof_sigma",
                "redshift",
                "is_background",
            ]:
                if c in ["skip"]:
                    continue
                if c not in cols:
                    cols.append(c)

                row.append(yaml_data[k][c])

            rows.append(row)

        tab = utils.GTable(names=cols, rows=rows)
        tab.rename_column("source_ra", "ra")
        tab.rename_column("source_dec", "dec")
        bad = np.in1d(tab["redshift"], [None])
        tab["z"] = -1.0
        tab["z"][~bad] = tab["redshift"][~bad]
        tab["mode"] = " ".join(mode.split("-")[-3:-1])
        tab["detector"] = mode.split("-")[-1]

        tab["group"] = mode

        tab.remove_column("redshift")
        tab.write(f"{mode}.info.csv", overwrite=True)

        tab["wmin"] = 0.0
        tab["wmax"] = 0.0

        tab["oiii_sn"] = -100.0
        tab["ha_sn"] = -100.0
        tab["max_cont"] = -100.0
        tab["dof"] = 0
        tab["dchi2"] = -100.0
        tab["bic_diff"] = -100.0

        # Redshift output
        for i, s in tqdm(enumerate(tab["source_name"])):
            yy = f"{mode}-{s}.spec.yaml"
            if os.path.exists(yy):
                with open(yy) as fp:
                    zfit = yaml.load(fp, Loader=yaml.Loader)

                for k in ["z", "dof", "wmin", "wmax", "dchi2"]:
                    if k in zfit:
                        tab[k][i] = zfit[k]

                oiii_key = None
                if "spl_coeffs" in zfit:

                    max_spl = -100
                    nline = 0
                    ncont = 0

                    for k in zfit["spl_coeffs"]:
                        if k.startswith("bspl") & (
                            zfit["spl_coeffs"][k][1] > 0
                        ):
                            _coeff = zfit["spl_coeffs"][k]
                            max_spl = np.maximum(
                                max_spl, _coeff[0] / _coeff[1]
                            )
                            ncont += 1

                        elif k.startswith("line"):
                            nline += 1

                    tab["max_cont"][i] = max_spl

                    bic_cont = (
                        np.log(zfit["dof"]) * ncont + zfit["spl_cont_chi2"]
                    )
                    bic_line = (
                        np.log(zfit["dof"]) * (ncont + nline)
                        + zfit["spl_full_chi2"]
                    )

                    tab["bic_diff"][i] = bic_cont - bic_line

                    if "line Ha" in zfit["spl_coeffs"]:
                        _coeff = zfit["spl_coeffs"]["line Ha"]
                        if _coeff[1] > 0:
                            tab["ha_sn"][i] = _coeff[0] / _coeff[1]

                    for k in ["line OIII-5007", "line OIII"]:
                        if k in zfit["spl_coeffs"]:
                            oiii_key = k

                    if oiii_key is not None:
                        _coeff = zfit["spl_coeffs"][oiii_key]
                        if _coeff[1] > 0:
                            tab["oiii_sn"][i] = _coeff[0] / _coeff[1]

        tabs.append(tab)

    full = utils.GTable(astropy.table.vstack(tabs))
    ok = np.isfinite(full["ra"] + full["dec"])
    full = full[ok]

    full["ra"].format = ".7f"
    full["dec"].format = ".7f"
    full["yoffset"].format = ".2f"
    full["prof_sigma"].format = ".2f"
    full["z"].format = ".4f"
    full["oiii_sn"].format = ".1f"
    full["ha_sn"].format = ".1f"
    full["max_cont"].format = ".1f"
    full["dchi2"].format = ".1f"
    full["dof"].format = ".0f"
    full["bic_diff"].format = ".1f"
    full["wmin"].format = ".1f"
    full["wmax"].format = ".1f"

    if zout is not None:
        idx, dr = zout.match_to_catalog_sky(full)
        hasm = dr.value < 0.3
        if root == "uds":
            hasm = dr.value < 0.4

        full["z_phot"] = -1.0
        full["z_phot"][hasm] = zout["z_phot"][idx][hasm]

        full["z_spec"] = -1.0
        full["z_spec"][hasm] = zout["z_spec"][idx][hasm]

        full["z_phot"].format = ".2f"
        full["z_spec"].format = ".3f"

        full["phot_id"] = -1
        full["phot_id"][hasm] = zout["id"][idx][hasm]

    url = '<a href="{m}-{name}.spec.fits">'
    url += '<img src="{m}-{name}.spec.png" height=200px>'
    url += "</a>"

    churl = '<a href="{m}-{name}.spec.fits">'
    churl += '<img src="{m}-{name}.spec.chi2.png" height=200px>'
    churl += "</a>"

    furl = '<a href="{m}-{name}.spec.fits">'
    furl += '<img src="{m}-{name}.spec.zfit.png" height=200px>'
    furl += "</a>"

    full["spec"] = [
        url.format(m=m, name=name)
        for m, name in zip(full["group"], full["source_name"])
    ]

    full["chi2"] = [
        churl.format(m=m, name=name)
        for m, name in zip(full["group"], full["source_name"])
    ]

    full["zfit"] = [
        furl.format(m=m, name=name)
        for m, name in zip(full["group"], full["source_name"])
    ]

    full.write(f"{root}_nirspec.csv", overwrite=True)
    full.write_sortable_html(
        f"{root}_nirspec.html",
        max_lines=10000,
        filter_columns=[
            "ra",
            "dec",
            "z_phot",
            "wmin",
            "wmax",
            "z",
            "dof",
            "bic_diff",
            "dchi2",
            "oiii_sn",
            "ha_sn",
            "max_cont",
            "z_spec",
            "yoffset",
            "prof_sigma",
        ],
        localhost=False,
    )

    print(f"Created {root}_nirspec.html {root}_nirspec.csv")
    return tabs, full
