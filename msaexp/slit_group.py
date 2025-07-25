import numpy as np
import astropy.io.fits as pyfits
import os
import glob
import msaexp.utils as msautils
from msaexp import pipeline_extended

from tqdm import tqdm

import jwst.datamodels
from grizli import utils
import matplotlib.pyplot as plt

from .msa import slit_best_source_alias

# Standard error on the median
SE_MEDIAN = 1.2533

from msaexp.pipeline_extended import (
    extend_wavelengthrange,
    step_reference_files,
    EXTENDED_RANGES,
)

ranges = EXTENDED_RANGES
ranges["CLEAR_PRISM"] = [0.5, 5.7]
ranges["F290LP_G395M"] = [2.5, 5.6]

VERBOSITY = True


def load_used_flat_models(
    input_model=None, photom_file="jw06648015001_01203_00001_nrs2_photom.fits"
):
    """
    Return a dictionary of the flat references that were used with a particular file

    Parameters
    ---------
    input_model : `jwst.datamodels.DataModel`
        Preloaded data model

    photom_file : str
        Calibrated data filename to open

    Returns
    -------
    reference_file_models : dict
        Dictionary with reference file model objects

    """
    from jwst.flatfield import FlatFieldStep

    opened = False
    if input_model is None:
        input_model = jwst.datamodels.open(photom_file)
        opened = True

    refs = input_model.meta.ref_file.instance

    kws = {}
    for k in ["fflat", "sflat", "dflat"]:
        if "_ext" in refs[k]["name"]:
            kws[f"override_{k}"] = refs[k]["name"]

    step = FlatFieldStep(**kws)

    reference_file_models = step._get_references(
        input_model, input_model.meta.exposure.type
    )

    if opened:
        input_model.close()

    return reference_file_models


def get_slit_flat(slit, flat_models):
    """
    Compute the flat field reference data for a particular slit object

    Parameters
    ---------
    slit : `jwst.datamodels.SlitModel`
        Slit object

    flat_models : dict
        Dictionary from `load_used_flat_models`

    Returns
    -------
    slit_flat : `jwst.datamodels.SlitModel`
        Flat data model
    """
    from jwst.flatfield.flat_field import flat_for_nirspec_slit

    exposure_type = slit.meta.exposure.type

    if exposure_type == "NRS_MSASPEC":
        slit_nt = slit  # includes quadrant info
    else:
        slit_nt = None

    slit_flat = flat_for_nirspec_slit(
        slit,
        flat_models["fflat"],
        flat_models["sflat"],
        flat_models["dflat"],
        1,
        exposure_type,
        slit_nt,
        slit.meta.subarray,
        use_wavecorr=None,
    )
    return slit_flat


class MultiSlitGroup:
    def __init__(
        self, file="jw02750002001_03101_00001_nrs1_photom.fits", **kwargs
    ):
        """ """
        self.file = file

        self.hdu = pyfits.open(file)

        self.source_name = []
        self.slit_id = []
        self.extver = []

        for i, ext in enumerate(self.hdu):
            h = ext.header
            if "EXTNAME" not in h:
                continue

            if h["EXTNAME"] == "SCI":
                self.source_name.append(h["SRCNAME"])
                self.slit_id.append(h["SLITID"])
                self.extver.append(h["EXTVER"])

    def get_slit(self, idx=1, source_name=None, slit_id=None):
        """ """
        if source_name is not None:
            if source_name in self.source_name:
                idx = self.extver[self.source_name.index(source_name)]
            else:
                return None

        if slit_id is not None:
            if slit_id in self.slit_id:
                idx = self.extver[self.slit_id.index(slit_id)]
            else:
                return None

        ext = [
            "SCI",
            "DQ",
            "ERR",
            "WAVELENGTH",
            "BARSHADOW",
            "VAR_POISSON",
            "VAR_RNOISE",
            "VAR_FLAT",
            "PATHLOSS_PS",
            "PATHLOSS_UN",
        ]

        hdul = [self.hdu[0]]
        for e in ext:
            if (e, idx) in self.hdu:
                hdul.append(self.hdu[e, idx].copy())

        hdul = pyfits.HDUList(hdul)

        # hdul = pyfits.HDUList(
        #     [self.hdu[0]] +
        #     [self.hdu[e, idx].copy() for e in ext]
        # )

        for i in range(len(hdul) - 1):
            hdul[i + 1].header["EXTVER"] = 1

        dm = jwst.datamodels.open(hdul)
        return dm.slits[0]


class NirspecCalibrated:
    def __init__(
        self,
        file="jw02750002001_03101_00001_nrs1_rate.fits",
        pedestal=0.0,
        read_slitlet=False,
        write_results=True,
        pixel_area_ref=4.88e-13,
        just_fixed_slit=False,
        **kwargs,
    ):

        self.file = file

        self.pixel_area_ref = pixel_area_ref
        self.just_fixed_slit = just_fixed_slit

        with pyfits.open(file) as hdu:
            self.rate_data = hdu["SCI"].data
            self.h0 = hdu[0].header
            self.h1 = hdu["SCI"].header

        slitlet_file = self.file.replace("_rate.fits", "_slitlet.fits")
        if os.path.exists(slitlet_file) & (read_slitlet):
            self.read_from_slitlet_file()
        else:
            self.run_pipeline(**kwargs)
            self.fs_extractor = MultiSlitGroup(self.fs_photom_file)

            if just_fixed_slit:
                self.extractor = self.fs_extractor
            else:
                self.extractor = MultiSlitGroup(self.photom_file)

            self.slits_to_full_frame(**kwargs)

            if write_results:
                hdul = self.write_full_file()

        self.set_mask(**kwargs)
        self.mask_stuck_closed(**kwargs)

        self.pedestal = pedestal
        self.global_flat = np.ones((2048, 2048))
        self.FLAT_ON = 1
        self.PIXEL_AREA_EXP = 0

        self.set_pixel_area()

    @property
    def grating(self):
        return self.h0["GRATING"]

    @property
    def filter(self):
        return self.h0["FILTER"]

    @property
    def detector(self):
        return self.h0["DETECTOR"]

    def __getitem__(self, key):
        """
        Get item from ``full`` dictionary
        """
        if key == "mask":
            return self.mask
        elif key == "bkg":
            return self.full["data"] - self.pedestal * self.flat_guess
        elif key == "corr":
            return (
                (self.full["data"] - self.pedestal * self.flat_guess)
                * 1.0
                / self.full["bar"]
                / self.global_flat**self.FLAT_ON
                * self.pixel_area**self.PIXEL_AREA_EXP
            )
        elif key == "corr_err":
            return self.full["err"] / self.full["bar"] / self.global_flat
        elif key == "corr_rnoise":
            p = self.FLAT_ON * 2
            pa = self.PIXEL_AREA_EXP * 2
            return (
                self.full["var_rnoise"]
                / self.full["bar"] ** 2
                / self.global_flat**p
                * self.pixel_area**pa
            )
        elif key == "corr_poisson":
            p = self.FLAT_ON * 2
            pa = self.PIXEL_AREA_EXP * 2
            return (
                self.full["var_poisson"]
                / self.full["bar"] ** 2
                / self.global_flat**p
                * self.pixel_area**pa
            )
        elif key in self.full:
            return self.full[key]
        else:
            print(f"{key} not found")
            return None

    def run_pipeline(self, ranges=ranges, preprocess_kwargs={}, **kwargs):
        file = self.file

        if self.just_fixed_slit:
            self.photom_file = None
            photom = None
        else:
            self.photom_file = file.replace("rate.fits", "photom.fits")

            if os.path.exists(self.photom_file):
                print("Read ", self.photom_file)
                photom = jwst.datamodels.open(self.photom_file)
                print(f"{len(photom.slits)} slits")
            else:
                photom = pipeline_extended.run_pipeline(
                    file,
                    slit_index=0,
                    all_slits=True,
                    write_output=False,
                    set_log=True,
                    skip_existing_log=False,
                    undo_flat=True,
                    preprocess_kwargs=preprocess_kwargs,
                    ranges=ranges,
                    make_trace_figures=False,
                    run_pathloss=False,
                    **kwargs,
                )

                print("write file")
                photom.write(self.photom_file)

        self.fs_photom_file = file.replace("rate.fits", "fs_photom.fits")

        if os.path.exists(self.fs_photom_file):
            print("Read ", self.fs_photom_file)
            fs_photom = jwst.datamodels.open(self.fs_photom_file)
        else:
            _file = file

            with pyfits.open(_file, mode="update") as _im:
                ORIG_EXPTYPE = _im[0].header["EXP_TYPE"]
                if ORIG_EXPTYPE != "NRS_FIXEDSLIT":
                    print(f"Set {_file} MSA > FIXEDSLIT keywords")
                    _im[0].header["EXP_TYPE"] = "NRS_FIXEDSLIT"
                    _im[0].header["APERNAME"] = "NRS_S200A2_SLIT"
                    _im[0].header["OPMODE"] = "FIXEDSLIT"
                    _im[0].header["FXD_SLIT"] = "S200A2"
                    _im.flush()

            fs_photom = pipeline_extended.run_pipeline(
                _file,
                slit_index=0,
                all_slits=True,
                write_output=False,
                set_log=True,
                skip_existing_log=False,
                undo_flat=True,
                preprocess_kwargs=preprocess_kwargs,
                ranges=ranges,
                make_trace_figures=False,
                run_pathloss=False,
                **kwargs,
            )

            with pyfits.open(_file, mode="update") as _im:
                if ORIG_EXPTYPE == "NRS_MSASPEC":
                    print(f"Reset {_file} FIXEDSLIT > MSA keywords")
                    _im[0].header["EXP_TYPE"] = "NRS_MSASPEC"
                    _im[0].header["APERNAME"] = "NRS_FULL_MSA"
                    _im[0].header["OPMODE"] = "MSASPEC"
                    _im[0].header.pop("FXD_SLIT")
                    _im.flush()

            fs_photom.write(self.fs_photom_file)

        self.photom = photom
        self.fs_photom = fs_photom

        for slit in self.fs_photom.slits:
            flat_profile = msautils.fixed_slit_flat_field(slit, apply=True)

    def write_slitlet_files(
        self,
    ):
        """
        Write individual slitlet files
        """
        if not hasattr(self, "photom"):
            msg = "failed: photom and fs_photom SlitGroups missing"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)
            return None

        for i, slit_obj in enumerate([self.photom, self.fs_photom]):
            if self.just_fixed_slit & (i == 0):
                continue
            # flat_reference_files = load_used_flat_models(slit_obj)
            ref_files = slit_obj.meta.ref_file.instance

            for _slit in slit_obj.slits:
                catalog_name_ = _slit.meta.target.catalog_name
                if catalog_name_ is not None:
                    targ_ = catalog_name_.replace(" ", "-").replace("_", "-")
                else:
                    targ_ = "cat"

                inst_key = f"{_slit.meta.instrument.filter}_{_slit.meta.instrument.grating}"

                root_ = self.file.split("_rate")[0]
                if _slit.meta.exposure.type == "NRS_FIXEDSLIT":
                    if targ_ == "":
                        targ_ = "indef"

                    slit_prefix_ = f"{root_}_{targ_}_{inst_key}_{_slit.name}"
                else:
                    if "_ext" in ref_files["fflat"]["name"]:
                        plabel = "raw"
                    else:
                        plabel = "phot"

                    try:
                        source_alias = slit_best_source_alias(
                            _slit,
                            require_primary=False,
                            which="min",
                            verbose=False,
                        )
                    except:
                        source_alias = _slit.source_name

                    slit_prefix_ = (
                        f"{root_}_{inst_key}_{plabel}."
                        + f"{_slit.name}.{source_alias}"
                    )

                slit_file = slit_prefix_.lower() + ".fits"

                msg = f"slit_group.MultiSlitGroup: write slitlet {slit_file}"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSITY)

                slit_model = jwst.datamodels.SlitModel(_slit.instance)
                slit_model.write(slit_file, overwrite=True)

                with pyfits.open(slit_file, mode="update") as im:
                    im[0].header["NOFLAT"] = (
                        True,
                        "Dummy flat field with extended wavelength references",
                    )

                    for ftype in ["fflat", "sflat", "dflat", "photom"]:
                        im[0].header["R_" + ftype.upper()] = ref_files[ftype][
                            "name"
                        ]

                    im.flush()

    def slits_to_full_frame(
        self, area_correction=False, shutter_pad=0.5, **kwargs
    ):
        """
        tbd
        """
        sh = (2048, 2048)
        full = {
            "data": np.zeros(sh, dtype=np.float32),
            "err": np.zeros(sh, dtype=np.float32),
            "var_rnoise": np.zeros(sh, dtype=np.float32),
            "var_poisson": np.zeros(sh, dtype=np.float32),
            "dq": np.zeros(sh, dtype=np.uint32),
            "yslit": np.zeros(sh, dtype=np.float32),
            "wave": np.zeros(sh, dtype=np.float32),
            "bar": np.zeros(sh, dtype=np.float32),
            "nexp": np.zeros(sh, dtype=int),
            "num_shutters": np.zeros(sh, dtype=int),
            "exp_index": np.zeros(sh, dtype=int),
            "slit": np.zeros((3, 2048, 2048), dtype=int),
            "slit_flat": np.zeros(sh, dtype=np.float32),
        }

        self.with_pixelarea = area_correction

        counter = 0

        self.slit_counter = []
        self.slit_names = []
        self.slit_slices = []
        self.slit_shutter = []
        self.pixel_areas = []

        for i, slit_group in enumerate([self.fs_photom, self.photom]):
            if self.just_fixed_slit & (i == 1):
                continue

            flat_models = load_used_flat_models(slit_group)

            progress = tqdm(enumerate(slit_group.slits))
            for i, slit in progress:
                counter += 1
                progress.set_description(
                    f"Process slit #{counter} {slit.name}"
                )

                try:
                    slit_data = msautils.get_slit_data(slit)
                except ValueError:
                    continue

                if slit_data["num_shutters"] > 3:
                    print(i, slit_data["num_shutters"])

                if slit_data["num_shutters"] > 8:
                    continue

                shutter_center = slit_data["shutter_y"] - np.nanmedian(
                    slit_data["shutter_y"], axis=0
                )

                # ymax = np.floor(np.nanmax(np.abs(shutter_center)))
                try:
                    yhi = np.nanmax(shutter_center[np.isfinite(slit.data)])
                    ylo = np.nanmin(shutter_center[np.isfinite(slit.data)])
                except:
                    continue

                edge_mask = shutter_center > (ylo + shutter_pad)
                edge_mask &= shutter_center < (yhi - shutter_pad)
                edge_mask &= slit_data["bar"] > 0
                edge_mask &= np.isfinite(slit.data + slit_data["wave"])

                MSA_FAILED_OPEN = jwst.datamodels.dqflags.pixel[
                    "MSA_FAILED_OPEN"
                ]
                stuck_open = ((slit.dq & MSA_FAILED_OPEN) > 0) & np.isfinite(
                    slit.data
                )

                if stuck_open.sum() > 0:
                    print(
                        i,
                        counter,
                        slit.name,
                        stuck_open.sum(),
                        "Has failed open shutters",
                    )

                slit_data["wave"][~edge_mask] = 0
                slit_data["bar"][~edge_mask] = 0

                sly, slx = slit_data["sly"], slit_data["slx"]
                full["wave"][sly, slx] += slit_data["wave"]
                full["bar"][sly, slx] += slit_data["bar"]
                full["nexp"][sly, slx] += slit_data["bar"] > 0
                full["num_shutters"][sly, slx] += (
                    slit_data["num_shutters"] * edge_mask
                )
                full["exp_index"][sly, slx] += counter * edge_mask

                self.slit_counter.append(counter)
                self.slit_names.append(slit.name.strip())
                self.slit_slices.append((sly, slx))
                self.slit_shutter.append((slit.quadrant, slit.xcen, slit.ycen))
                self.pixel_areas.append(
                    slit.meta.photometry.pixelarea_steradians
                )

                if slit.name in [
                    "S200A1",
                    "S200A2",
                    "S200B1",
                    "S400A1",
                    "S1600A1",
                ]:
                    full["slit"][0, sly, slx] += 5 * edge_mask
                else:
                    full["slit"][0, sly, slx] += slit.quadrant * edge_mask
                    full["slit"][1, sly, slx] += slit.xcen * edge_mask
                    full["slit"][2, sly, slx] += slit.ycen * edge_mask

                fslit = slit  # photom.slits[i]
                fslit_data = fslit.data * 1
                if hasattr(fslit, "barshadow"):
                    if fslit.barshadow.shape[0] == 0:
                        fslit_bar = np.ones_like(fslit_data)
                    else:
                        fslit_bar = fslit.barshadow * 1
                else:
                    fslit_bar = np.ones_like(fslit_data)

                if fslit.source_type is None:
                    path_data = fslit.pathloss_uniform
                else:
                    path_data = fslit.pathloss_point

                if path_data.shape[0] == 0:
                    fslit_path = np.ones_like(fslit_data)
                else:
                    fslit_path = path_data * 1

                fslit_rnoise = fslit.var_rnoise * 1
                fslit_poisson = fslit.var_poisson * 1
                fslit_msk = edge_mask & np.isfinite(fslit_data)

                fslit_data[~fslit_msk] = 0.0
                fslit_rnoise[~fslit_msk] = 0.0
                fslit_poisson[~fslit_msk] = 0.0
                fslit_bar[~fslit_msk] = 0
                fslit_path[~fslit_msk] = 1

                fslit_dq = fslit.dq * fslit_msk
                slit_flat = get_slit_flat(fslit, flat_models)
                fslit_flat = slit_flat.data
                fslit_flat[~fslit_msk] = 0

                yslit_i = slit_data["yslit"]
                yslit_i[~fslit_msk] = 0
                # yslit_i[~np.isfinite(yslit_i)]
                if area_correction:
                    fslit_scale = (
                        fslit.meta.photometry.pixelarea_steradians * 1.0e12
                    )
                else:
                    fslit_scale = 1.0

                fslit_scale = fslit_scale * fslit_bar * fslit_path

                full["data"][sly, slx] += fslit_data * fslit_scale

                full["dq"][sly, slx] |= fslit_dq
                full["var_rnoise"][sly, slx] += fslit_rnoise * fslit_scale**2
                full["var_poisson"][sly, slx] += fslit_poisson * fslit_scale**2
                full["yslit"][sly, slx] += yslit_i
                full["slit_flat"][sly, slx] += fslit_flat

        self.full = full
        self.flat_guess = self.full["data"] / self.rate_data

        self.mask = np.isfinite(full["data"]) & (full["data"] != 0)
        self.stuck_mask = self.mask * 1 > 10

    def read_from_slitlet_file(self, **kwargs):
        """ """
        slitlet_file = self.file.replace("_rate.fits", "_slitlet.fits")
        print(f"Read from {slitlet_file}")

        full = {}
        with pyfits.open(slitlet_file) as hdul:
            for ext in hdul[1:]:
                full[ext.header["EXTNAME"].lower()] = hdul[ext].data * 1

            nslits = hdul[0].header["NSLITS"]
            self.slit_counter = []
            self.slit_names = []
            self.slit_slices = []
            self.slit_shutter = []
            self.pixel_areas = []

            h0 = hdul[0].header

            for k in h0:
                if k.startswith("SLIT"):
                    c = int(k[4:])
                    self.slit_counter.append(c)
                    self.slit_names.append(h0[k])

                    sly = slice(h0[f"SY0_{c:04d}"], h0[f"SY1_{c:04d}"])
                    slx = slice(h0[f"SX0_{c:04d}"], h0[f"SX1_{c:04d}"])
                    self.slit_slices.append((sly, slx))

                    self.slit_shutter.append(
                        (
                            h0[f"SHUQ{c:04d}"],
                            h0[f"SHUX{c:04d}"],
                            h0[f"SHUY{c:04d}"],
                        )
                    )

                    self.pixel_areas.append(h0[f"PIXA{c:04d}"])

        self.full = full

        self.flat_guess = self.full["data"] / self.rate_data

        self.mask = np.isfinite(full["data"]) & (full["data"] != 0)
        self.stuck_mask = self.mask * 1 > 10

    def set_pixel_area(self):
        """ """
        sq, sx, sy = self["slit"]
        msk = np.where(((self["nexp"] == 1) & (sq > 0) & (sq < 6)).flatten())[
            0
        ]

        pixel_area = np.zeros((2048, 2048), dtype=np.float32).flatten()

        un = utils.Unique(self["exp_index"].flatten()[msk], verbose=False)
        for i, area in zip(self.slit_counter, self.pixel_areas):
            if i in un.values:
                pixel_area[msk[un[i]]] = area / self.pixel_area_ref

        self.pixel_area = pixel_area.reshape((2048, 2048))

    def set_global_flat(
        self,
        use_local=True,
        skip_quads=True,
        ratio_smooth_params=(3.1, 0.7, 0.8),
        prefix="sflat_lamp_spl_coeffs",
    ):
        """
        tbd
        """
        sq, sx, sy = self["slit"]
        msk = (self["nexp"] == 1) & (sq > 0) & (sq < 5)

        mskf = msk.flatten()
        sqf = sq.flatten()
        sxf = sx.flatten()
        syf = sy.flatten()

        unq = utils.Unique(sqf[mskf], verbose=False)

        full_wave = self["wave"].flatten()

        global_flat = np.ones((2048, 2048)).flatten()

        for qi in unq.values:
            if skip_quads:
                if "nrs1" in self.file:
                    if qi in [1, 2]:
                        continue
                else:
                    if qi in [3, 4]:
                        continue

            if use_local:
                flat_file = (
                    # f"flat_coeffs_{self.h0['DETECTOR']}_q{qi}.fits".lower()
                    f"{prefix}_q{qi}.fits".lower()
                )
            else:
                flat_file = os.path.join(
                    os.path.dirname(msautils.__file__),
                    "data/extended_sensitivity",
                    f"sflat_spl_coeffs_prism_q{qi}.fits".lower(),
                )

            if not os.path.exists(flat_file):
                print(f"file {flat_file} not found")
                continue

            ftab = utils.read_catalog(flat_file)
            msg = f"Compute flat for quadrant {qi} with {flat_file}"
            if "MTIME" in ftab.meta:
                msg += f" (mtime: {ftab.meta['MTIME']})"

            print(msg)

            xlim = (ftab.meta["XMIN"], ftab.meta["XMAX"])
            ylim = (ftab.meta["YMIN"], ftab.meta["YMAX"])

            xg = np.linspace(*xlim, xlim[1] - xlim[0])
            yg = np.linspace(*ylim, ylim[1] - ylim[0])
            xg, yg = np.meshgrid(xg, yg)

            gxbspl = utils.bspline_templates(
                xg.flatten(), df=ftab.meta["DFX"], minmax=xlim, get_matrix=True
            )
            gybspl = utils.bspline_templates(
                yg.flatten(), df=ftab.meta["DFY"], minmax=ylim, get_matrix=True
            )

            gbspl = np.vstack([gxbspl.T * row for row in gybspl.T]).T

            gmodels = []
            for c in ftab["coeffs"]:
                gmodels.append(gbspl.dot(c).reshape(xg.shape))

            gmodels = np.array(gmodels)
            SFLAT_STRAIGHTEN = 0
            if SFLAT_STRAIGHTEN > 0:
                print("Straighten")
                wsub = (ftab["wave"] > 0.8) & (ftab["wave"] < 5.25)
                med = np.nanmedian(np.nanmedian(gmodels, axis=1), axis=1)
                wsub &= med > 0
                c = np.polyfit(
                    ftab["wave"][wsub], med[wsub], SFLAT_STRAIGHTEN - 1
                )
                cfit = np.polyval(c, ftab["wave"])
                gmodels = (gmodels.T / cfit).T

            qix = np.where(mskf)[0][unq[qi]]
            unx = utils.Unique(sxf[qix], verbose=False)

            Nf = len(ftab)

            for xi in unx.values:
                unxi = unx[xi]
                uny = utils.Unique(syf[qix][unxi], verbose=False)
                for yi in uny.values:
                    # print(qi, xi, yi)
                    ixi = qix[unxi][uny[yi]]
                    flat_ = gmodels[:, yi, xi]
                    fok = flat_ > 0
                    # global_flat[ixi] = np.interp(
                    #     full_wave[ixi], ftab["wave"][fok], flat_[fok]
                    # )
                    cspl = utils.bspline_templates(
                        np.interp(
                            full_wave[ixi], ftab["wave"], np.linspace(0, 1, Nf)
                        ),
                        df=Nf,
                        minmax=(0, 1),
                        get_matrix=True,
                    )

                    # Flatten it out in the middle
                    # wg = np.abs(full_wave[ixi] - 3.1)
                    # yg = (1 - np.exp(-(wg**2) / 2 / 0.7**2)) ** 1 * 0.8 + 0.2
                    # spl_flat = (cspl.dot(flat_) - 1) * yg + 1
                    wsm = np.abs(full_wave[ixi] - ratio_smooth_params[0])
                    ysm = (
                        1 - np.exp(-(wsm**2) / 2 / ratio_smooth_params[1] ** 2)
                    ) ** 1 * ratio_smooth_params[2] + (
                        1 - ratio_smooth_params[2]
                    )
                    spl_flat = (cspl.dot(flat_) - 1) * ysm + 1

                    global_flat[ixi] = spl_flat

        global_flat[global_flat <= 0] = 1.0

        self.global_flat = global_flat.reshape((2048, 2048))

    def get_global_background(self, sample=1.0, min_yslit=5, suffix="gbkg"):
        """
        Get background spectrum from all available MSA shutters
        """
        if np.nanmax(self.global_flat) == 1.0:
            self.set_global_flat(
                use_local=False, ratio_smooth_params=(3.1, 0.7, 0.8)
            )

        self.set_mask(min_yslit=min_yslit, verbose=False)

        quad = self["slit"][0, :, :]

        slit_selection = (quad > 0) & (quad < 5)
        if "nrs1" in self.file:
            slit_selection = np.isin(quad, [3, 4])
        else:
            slit_selection = np.isin(quad, [1, 2])

        self.global_background_spectrum = self.percentile_bins(
            sample=sample, extra=slit_selection
        )

        if ("rate.fits" in self.file) & (suffix is not None):
            ref_file = self.file.replace("rate.fits", f"{suffix}.fits")
            print(ref_file)
            self.global_background_spectrum.write(ref_file, overwrite=True)

        return self.global_background_spectrum

    def write_full_file(self, overwrite=True, **kwargs):
        """
        TBD
        """

        hdul = [pyfits.PrimaryHDU(header=self.h0.copy())]

        hdul[0].header["WITHPIXA"] = self.with_pixelarea

        hdul[0].header["NSLITS"] = len(self.slit_counter)
        for i, name_, (sly, slx), shutter_, area_ in zip(
            self.slit_counter,
            self.slit_names,
            self.slit_slices,
            self.slit_shutter,
            self.pixel_areas,
        ):
            hdul[0].header[f"SLIT{i:04d}"] = name_
            hdul[0].header[f"SX0_{i:04d}"] = (slx.start, "Slice x start")
            hdul[0].header[f"SX1_{i:04d}"] = (slx.stop, "Slice x stop")
            hdul[0].header[f"SY0_{i:04d}"] = (sly.start, "Slice y start")
            hdul[0].header[f"SY1_{i:04d}"] = (sly.stop, "Slice y stop")

            sq, sx, sy = shutter_
            hdul[0].header[f"SHUQ{i:04d}"] = (sq, "Shutter quadrant")
            hdul[0].header[f"SHUX{i:04d}"] = (sx, "Shutter xcen")
            hdul[0].header[f"SHUY{i:04d}"] = (sy, "Shutter ycen")

            hdul[0].header[f"PIXA{i:04d}"] = (area_, "PIXAR_SR at shutter")

        # hdul[0].header["PEDESTAL"] = self.pedestal

        for k in self.full:
            hdul.append(pyfits.ImageHDU(self.full[k], name=k))

        for k in self.h1:
            if k in ["COMMENT", ""]:
                continue
            if k not in hdul[1].header:
                hdul[1].header[k] = self.h1[k], self.h1.comments[k]

        hdul = pyfits.HDUList(hdul)

        slitlet_file = self.file.replace("_rate.fits", "_slitlet.fits")
        print(f"Write {slitlet_file}")

        if os.path.exists(slitlet_file):
            if overwrite:
                hdul.writeto(slitlet_file, overwrite=overwrite)
        else:
            hdul.writeto(slitlet_file, overwrite=overwrite)

        return hdul

    def set_mask(
        self,
        min_yslit=3.0,
        bar_threshold=0.35,
        low_sigma=-3,
        keep_fixed_slits=["S200A1", "S200A2", "S200B1"],
        verbose=True,
        **kwargs,
    ):
        """
        tbd
        """
        yp, xp = np.indices(self.rate_data.shape)

        msk = self["nexp"] == 1
        msk &= self["bar"] > bar_threshold
        msk &= self["exp_index"] > 0

        mask_sources = self["num_shutters"] > 2
        mask_sources &= np.abs(self["yslit"]) < min_yslit
        msk &= ~mask_sources

        fs_ind = []
        for i, sname in enumerate(self.slit_names):
            if not sname.startswith("S"):
                break
            elif sname not in keep_fixed_slits:
                ind = self.slit_counter[self.slit_names.index(sname)]
                if verbose:
                    print(f"exclude {sname} index={ind}")
                fs_ind.append(ind)

        msk &= ~np.isin(self["exp_index"], fs_ind)
        msk &= self["data"] > low_sigma * np.sqrt(self.full["var_rnoise"])

        self.mask = msk

    def mask_stuck_closed(self, prism_threshold=0.28, plot=False, **kwargs):
        """ """
        low = self.mask & (self["data"] < 2 * np.sqrt(self["var_rnoise"]))
        ind = utils.Unique(self["exp_index"][low], verbose=False)
        all_ind = utils.Unique(self["exp_index"][self.mask], verbose=False)

        low_fraction = []
        for i, v in enumerate(ind.values):
            low_fraction.append(
                ind.counts[i] / all_ind.counts[all_ind.values.index(v)]
            )

        if self.h0["GRATING"] == "PRISM":
            bad_slits = np.array(ind.values)[
                np.array(low_fraction) > prism_threshold
            ]
        else:
            bad_slits = np.array(ind.values)[np.array(low_fraction) > 0.99]

        self.bad_slits = bad_slits.tolist()

        print("bad slits (stuck closed?): ", bad_slits)
        self.stuck_mask = np.isin(self["exp_index"], bad_slits)

        if plot:
            plt.scatter(
                ind.values,
                low_fraction,
                c=np.isin(ind.values, bad_slits),
                vmin=0,
                vmax=1.5,
            )

    def plot_slit(self, counter=1, slit_name=None, aspect="auto", **kwargs):
        """ """

        idx = self.slit_counter.index(counter)

        if slit_name is not None:
            idx = self.slit_names.index(str(slit_name))

        fig, axes = plt.subplots(
            3, 1, figsize=(10, 6), sharex=True, sharey=True
        )

        sly, slx = self.slit_slices[idx]
        axes[0].imshow(self["data"][sly, slx], aspect=aspect, **kwargs)
        axes[1].imshow(self["corr"][sly, slx], aspect=aspect, **kwargs)
        axes[2].imshow(
            self["corr"][sly, slx] * self.mask[sly, slx],
            aspect=aspect,
            **kwargs,
        )

        axes[0].set_ylabel("data")
        axes[1].set_ylabel("bar")
        axes[2].set_ylabel("masked")

        axes[0].set_title(
            f'slit #{self.slit_counter[idx]}: "{self.slit_names[idx]}"'
        )

        fig.tight_layout(pad=1)

    def percentile_bins(
        self,
        sample=0.5,
        pvals=[5, 16, 50, 84, 95],
        extra=None,
        key="corr",
        column_prefix="p",
        **kwargs,
    ):
        """ """

        wgrid = msautils.get_standard_wavelength_grid(
            sample=sample,
            grating=self.h0["GRATING"],
        )
        wbins = msautils.array_to_bin_edges(wgrid)
        nbin = len(wgrid)
        xgrid = np.linspace(0, 1, nbin)

        mask = self.mask & (~self.stuck_mask)
        if extra is not None:
            mask &= extra

        full_wave = self["wave"]
        # so = full_wave[mask]
        corr = self[key]

        perc_data = []
        npix = []

        xwave = full_wave[mask]
        xcorr = corr[mask]

        for i in range(nbin):
            xsub = (xwave >= wbins[i]) & (xwave < wbins[i + 1])
            if xsub.sum() == 0:
                perc_data.append(np.zeros(len(pvals)))
                npix.append(0)
            else:
                perc_data.append(np.percentile(xcorr[xsub], pvals))
                npix.append(xsub.sum())

        perc_data = np.array(perc_data)

        tab = utils.GTable(
            perc_data, names=[f"{column_prefix}{v}" for v in pvals]
        )
        tab["wave"] = wgrid
        tab["grid"] = xgrid
        tab[f"{column_prefix}npix"] = npix
        tab[f"{column_prefix}err"] = (
            (perc_data[:, 2] - perc_data[:, 1]) / np.sqrt(npix) * SE_MEDIAN
        )
        tab.meta["sample"] = (sample, "Wavelength sample factor")
        tab.meta["datakey"] = (key, "Data type")
        tab.meta["REFPAREA"] = (
            self.pixel_area_ref,
            "Reference pixel area steradians",
        )

        return tab

    def percentile_table(
        self,
        sample=0.5,
        pvals=[5, 16, 50, 84, 95],
        make_plot=False,
        extra=None,
        **kwargs,
    ):
        """ """

        tab = self.percentile_bins(sample=sample, pvals=pvals, extra=None)

        yp, xp = np.indices((2048, 2048))

        btab = self.percentile_bins(
            sample=sample, pvals=pvals, extra=(yp < 1000), column_prefix="b"
        )

        ttab = self.percentile_bins(
            sample=sample, pvals=pvals, extra=(yp > 1100), column_prefix="t"
        )

        for t in [btab, ttab]:
            for c in t.colnames:
                if c not in tab.colnames:
                    tab[c] = t[c]

        full_wave = self["wave"]

        full_corr_med = np.interp(full_wave, tab["wave"], tab["p50"])
        full_corr_lo = np.interp(full_wave, tab["wave"], tab[f"p{pvals[0]}"])

        self.full_corr_med = full_corr_med
        self.full_corr_lo = full_corr_lo

        threshold_mask = self["corr"] < (
            full_corr_med + (full_corr_med - full_corr_lo) * 2
        )

        threshold_mask &= self["corr"] > (
            full_corr_med - (full_corr_med - full_corr_lo) * 1.5
        )

        self.percentiles = tab

        self.threshold_mask = threshold_mask

        return tab


def run_all():

    step = 1
    det = 1

    import glob
    from importlib import reload
    import slit_group

    reload(slit_group)
    import matplotlib.pyplot as plt

    plt.rcParams["backend"] = "tkagg"
    plt.ioff()

    files = glob.glob(f"*nrs{det}_rate.fits")
    files.sort()

    reload(slit_group)

    for file in files[::step]:
        grp = slit_group.NirspecCalibrated(
            file, read_slitlet=True, make_plot=False, area_correction=False
        )
