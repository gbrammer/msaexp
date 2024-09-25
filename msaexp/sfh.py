"""
Routines for estimating stellar population parameters from spectrum fits
"""

import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

try:
    from urllib3.exceptions import HTTPError
except ImportError:
    from urllib.error import HTTPError

from scipy.stats import t as student_t, norm
from scipy.optimize import nnls, minimize

from astropy.cosmology import WMAP9
import astropy.units as u

from grizli import utils
import eazy

from . import spectrum

COSMOLOGY = WMAP9

# Rough scale factors used for photo-z calculations,
# multiplied to the tabulated {filter}_tot_1 photometry
# measurements
PHOTOZ_ZP = {
    "f606w": 0.88,
    "f814w": 0.834,
    "f105w": 0.9628,
    "f125w": 0.979,
    "f140w": 0.9959,
    "f160w": 1.0057,
    "f090w": 0.8902054758355986,
    "f115w": 0.8603698910303057,
    "f150w": 0.8685858987386202,
    "f200w": 0.9083862482988716,
    "f277w": 1.0,
    "f356w": 1.0812199194444387,
    "f410m": 1.1306452082799334,
    "f444w": 1.163794484414118,
}

EAZY_FILTER_NUMBERS = {
    "f606w": 236,
    "f814w": 239,
    "f105w": 202,
    "f125w": 203,
    "f140w": 204,
    "f160w": 205,
    "f090w": 363,
    "f115w": 364,
    "f150w": 365,
    "f200w": 366,
    "f277w": 375,
    "f356w": 376,
    "f410m": 383,
    "f444w": 377,
}

PHOTOMETRIC_FILTERS = [
    "f606w",
    "f814w",
    # "f105w","f125w","f140w","f160w",
    "f090w",
    "f115w",
    "f150w",
    "f200w",
    "f277w",
    "f356w",
    "f410m",
    "f444w",
]

VERBOSE = True


class SpectrumPhotometry:
    RES = eazy.filters.FilterFile(path=None)

    def __init__(
        self,
        file,
        sys_err=0.03,
        escale_kwargs=dict(fit_sys_err=False, update=False),
        undo_path_corr=True,
        undo_norm_corr=True,
        use_filters=PHOTOMETRIC_FILTERS,
        redshift="query",
        **kwargs,
    ):
        """
        Object for handling combined spectrum + photometry fits
        """
        self.use_filters = use_filters
        self.file = file

        if redshift == "query":
            self.get_best_redshift(**kwargs)
        else:
            self.redshift = redshift
            self.redshift_type = "input"

        self.photom = None
        self.filters = None
        self.filter_ix = {}

        self.initial_scale_coeffs = [1.0]
        self.sys_err = sys_err

        self.fetch_photometry(sys_err=sys_err, **kwargs)

        try:
            self.spec = spectrum.SpectrumSampler(
                file, sys_err=sys_err, **kwargs
            )

            self.spec.spec["orig_flux"] = self.spec.spec["flux"]

            if ("path_corr" in self.spec.spec.colnames) & (undo_path_corr):
                msg = f"SpectrumPhotometry: {self.basename}  Undo path_corr"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

                self.spec.spec["flux"] /= self.spec.spec["path_corr"]
                self.spec.spec["err"] /= self.spec.spec["path_corr"]

            if ("norm_corr" in self.spec.spec.colnames) & (undo_norm_corr):
                msg = f"SpectrumPhotometry: {self.basename}  Undo norm_corr"
                utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

                self.spec.spec["flux"] /= self.spec.spec["norm_corr"]
                self.spec.spec["err"] /= self.spec.spec["norm_corr"]

            if escale_kwargs is not None:
                if self.redshift_type in ("input", "db"):
                    escale_kwargs["z"] = self.redshift

                espec, escale, esys, eres = spectrum.calc_uncertainty_scale(
                    file, sys_err=sys_err, **escale_kwargs
                )
                self.spec.spec["escale"] = escale
                spectrum.set_spec_sys_err(self.spec.spec, sys_err=esys)
                self.sys_err = esys

        except (FileNotFoundError, HTTPError):
            msg = f"SpectrumPhotometry: ! {self.file}  File/URL not found"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

            self.spec = None

        self.calib_poly = None
        if (self.spec is not None) & (len(self.filter_ix) > 0):
            _ = self.scale_spectrum_to_photometry(**kwargs)

        self.info()

    def info(self):
        """
        Print some summary information
        """
        _cstr = ",".join([f"{c:6.2f}" for c in self.initial_scale_coeffs])

        msg = f"""
# file = {self.file}
# z = {self.redshift:.4f} ({self.redshift_type})
# photometry in ({' '.join(self.filter_ix.keys())})
# scale coeffs {_cstr}
# photom_weight {self.photom_weight:.2f}
"""
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

    @property
    def basename(self):
        return os.path.basename(self.file)

    @property
    def NFILT(self):
        return self.photom.shape[0]

    def add_figure_label(self, **kwargs):
        """
        Add filename and timestamp to figure objects
        """
        pass

    def get_best_redshift(self, with_db=False, **kwargs):
        """
        Query best redshift for a particular spectrum
        """
        if with_db:
            res = db.SQL(
                f"""
            SELECT un.*, um.file
            FROM nirspec_unique un NATURAL JOIN nirspec_unique_match um
            WHERE um.file = '{os.path.basename(self.file)}'
            """
            )

        else:
            URL = (
                "https://grizli-cutout.herokuapp.com/nirspec_file_redshift?file_spec="
                + os.path.basename(self.file)
            )
            res = utils.read_catalog(URL, format="csv")

        self.zquery = res
        if len(res) == 1:
            self.redshift = res["z"][0]
            self.redshift_type = "db"

            msg = f"SpectrumPhotometry: {self.basename}  Set z = {self.redshift:.4f}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)
        else:
            msg = f"SpectrumPhotometry: {self.basename}  No redshift found, set z = 0"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

            self.redshift = 0.0
            self.redshift_type = "db_err"

    def fetch_photometry(self, i=None, with_db=False, **kwargs):
        """
        Get photometry data from the DB / API
        """
        if with_db:
            self.query = db.SQL(
                f"""
SELECT mat.file_spec, dra, ddec, dr, ph.*
FROM grizli_photometry ph, nirspec_phot_match mat
WHERE
mat.file_spec = '{os.path.basename(self.file)}'
AND mat.file_phot = ph.file_phot
AND mat.id_phot =  ph.id_phot
AND mat.dr < 0.5
ORDER BY file_zout DESC
"""
            )
            msg = f"SpectrumPhotometry: {self.basename}  query DB"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

        else:
            URL = (
                "https://grizli-cutout.herokuapp.com/grizli_photometry?file_spec="
                + os.path.basename(self.file)
            )

            msg = f"SpectrumPhotometry: {self.basename}  query {URL}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

            self.query = utils.read_catalog(URL, format="csv")

        if len(self.query) == 0:
            msg = f"SpectrumPhotometry: ! {self.basename}  No photometry found"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)
            return None

        if i is None:
            i = 0
            # most recent jwst catalog
            for i, row in enumerate(self.query):
                if row["dr"] > 0.8:
                    continue
                elif ("3dhst" in row["file_zout"]) & (len(self.query) > 1):
                    continue
                else:
                    break

        self.set_photometry(i=i, **kwargs)
        self.EBV, self.F99 = self.get_mw_alambda()

    def set_photometry(self, i=0, sys_err=0.03, use_aper=False, **kwargs):
        """
        Set photometry data from a row in the query
        """
        row = self.query[i]

        msg = (
            "SpectrumPhotometry: "
            + '{file_spec}  {file_zout} {id_phot} dr={dr:.1f}" z_phot={z_phot:.3f}'
        )
        utils.log_comment(utils.LOGFILE, msg.format(**row), verbose=VERBOSE)

        if self.redshift == 0:
            msg = (
                f"SpectrumPhotometry: {self.basename}"
                + f"  Set redshift from z_phot = {row['z_phot']:.3f}"
            )
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)
            self.redshift = row["z_phot"]
            self.redshift_type = "z_phot"

        filters = []
        photom = []

        self.filter_ix = {}
        count = 0

        for c in row.colnames:
            if "_tot_1" in c:
                filt = c.split("_tot")[0]

                if filt not in EAZY_FILTER_NUMBERS:
                    continue

                elif filt not in self.use_filters:
                    continue

                e_i = row[c.replace("_tot", "_etot")]
                # print(filt, e_i, hasattr(e_i, 'mask'))
                if e_i <= 0:
                    continue
                elif hasattr(e_i, "mask"):
                    if e_i.mask:
                        continue

                if filt in PHOTOZ_ZP:
                    zp_i = PHOTOZ_ZP[filt]
                else:
                    zp_i = 1.0

                # photom.append([row[c], row[c.replace('_tot', '_etot')]])
                if use_aper:
                    photom.append(
                        [
                            row[c.replace("_tot", "_flux_aper")] * zp_i,
                            row[c.replace("_tot", "_fluxerr_aper")] * zp_i,
                        ]
                    )
                else:
                    photom.append(
                        [row[c] * zp_i, row[c.replace("_tot", "_etot")] * zp_i]
                    )

                filters.append(self.RES[EAZY_FILTER_NUMBERS[filt]])
                self.filter_ix[filt] = count
                count += 1

        # print("xxx", self.filter_ix)

        photom = np.array(photom)
        pivot = np.array([f.pivot * 1 for f in filters]) / 1.0e4

        self.filters = filters
        self.photom = photom
        self.photom_err = np.sqrt(
            photom[:, 1] ** 2 + (sys_err * photom[:, 0]) ** 2
        )
        self.photom_weight = 1.0

        self.pivot = pivot

    def scale_spectrum_to_photometry(self, degree=0, min_frac=0.1):
        """
        Calculate polynomial to scale spectrum to photometry
        """

        # Integrate spectrum through filter bandpasses
        rows = []
        for filt in self.filter_ix:
            fix = self.filter_ix[filt]
            npix, ff, fflux, ferr, ffull = spectrum.integrate_spectrum_filter(
                self.spec.spec, self.filters[fix]
            )
            rows.append(
                [
                    filt,
                    self.pivot[fix],
                    ff,
                    *self.photom[fix],
                    self.photom_err[fix],
                    fflux,
                    ferr,
                    ffull,
                ]
            )

        scl = utils.GTable(
            names=[
                "filter",
                "pivot",
                "fcover",
                "photom_flux",
                "photom_err",
                "photom_full_err",
                "spec_flux",
                "spec_err",
                "spec_full_err",
            ],
            rows=rows,
        )

        # Calculate scaling
        fr = scl["photom_flux"] / scl["spec_flux"]
        fw = fr**2 * (
            (scl["photom_full_err"] / scl["photom_flux"]) ** 2
            + (scl["spec_full_err"] / scl["spec_flux"]) ** 2
        )
        ok = scl["fcover"] > 0.1
        if ok.sum() == 0:
            self.initial_scale_coeffs = [1.0]
            self.spec.meta["scale_degree"] = 0
            self.spec.meta["scale_coeff0"] = 1.0

            msg = (
                f"SpectrumPhotometry: ! {self.basename}"
                + "  No filters with sufficient overlap"
            )
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

            return 1.0

        self.initial_scale_coeffs = np.polyfit(
            self.pivot[ok], fr[ok], degree, w=1 / np.sqrt(fw[ok])
        )

        scale_phot = np.polyval(self.initial_scale_coeffs, self.pivot)
        scale_spec = np.polyval(self.initial_scale_coeffs, self.spec["wave"])
        # scale = (fr / fw)[ok].sum() / (1.0 / fw)[ok].sum()

        self.photom_weight = np.sqrt(
            (scl["photom_full_err"][ok] ** 2).sum()
            / ((scale_phot * scl["spec_full_err"])[ok] ** 2).sum()
        )

        # self.spec.meta["scale_filter"] = filter
        self.spec.meta["scale_degree"] = 0
        for i, c in enumerate(self.initial_scale_coeffs):
            self.spec.meta[f"scale_coeff{i}"] = c

        _cstr = ",".join([f"{c:6.2f}" for c in self.initial_scale_coeffs])
        _filt_list = ", ".join(scl["filter"][ok].tolist())

        msg = f"""
SpectrumPhotometry: {self.basename}  scale spectrum by ({_cstr}) to match photometry in ({_filt_list})
SpectrumPhotometry: {self.basename}  photom_weight {self.photom_weight:.2f}
"""
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

        if "scale_to_photometry" in self.spec.spec.colnames:
            self.spec.spec["scale_to_photometry"] *= scale_spec
        else:
            self.spec.spec["scale_to_photometry"] = scale_spec

        for c in ["flux", "err", "full_err"]:
            self.spec.spec[c] *= scale_spec

        return self.initial_scale_coeffs

    def plot(
        self,
        figsize=(10, 5),
        ax=None,
        flam=-2,
        scale_spec=1.0,
        fit_args=None,
        fit_results=None,
        show_fit_components=False,
        fit_color="tomato",
        z=0,
        **kwargs,
    ):
        """
        Plot spectrum, photometry and optional fit results
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None

        if flam != 0:
            pnorm = self.pivot**flam
            snorm = self.spec["wave"] ** flam
        else:
            pnorm = snorm = 1

        ax.errorbar(
            self.pivot / (1 + z),
            self.photom[:, 0] * pnorm,
            self.photom_err * pnorm,
            alpha=0.5,
            linestyle="None",
            marker="o",
            color="k",
        )
        model_spec = None

        if fit_args is not None:
            c, chi2, model_phot, comps_spec, model_spec, scale_spec = (
                self.fit_templates(*fit_args)
            )

        if fit_results is not None:
            c, chi2, model_phot, comps_spec, model_spec, scale_spec = (
                fit_results
            )

        if model_spec is not None:
            _templates, _z = fit_args[0:2]
            _mass, _sfr_continuum, _sfr_halpha = template_mass_sfr(
                _templates, c[0], z=_z
            )
            
            total_mass = np.log10(_mass.sum())
            total_halpha = _sfr_halpha.sum()
            
            if 1:
                # Sum of first two bins
                dt = np.array([0.019999  , 0.015     , 0.03298023, 0.06405724, 0.12441788,
                    0.24165589, 0.46936636, 0.91164663, 1.77068419, 3.43918614,
                    6.67990456])
            
                _sfr_continuum = _mass[:2].sum() / (dt[:2].sum()*1.e9)
            else:
                _sfr_continuum = _sfr_continuum.sum()
            
            self.spec.meta["sfh_mass"] = total_mass
            self.spec.meta["sfh_sfr_continuum"] = _sfr_continuum
            self.spec.meta["sfh_sfr_halpha"] = total_halpha

            if np.isfinite(total_mass) & (total_mass > 0) & (fig is not None):
                ax.text(
                    0.05,
                    0.02,
                    (
                        r"$\log_{10} M/M_\odot = $"
                        + f"{total_mass:.2f}"
                        + "\n"
                        + r"SFR(cont) = "
                        + f"{_sfr_continuum:.1f} "
                        + r" $M_\odot / \mathrm{yr}$"
                        + "\n"
                        + r"SFR(H$\alpha$) = "
                        + f"{total_halpha:.1f} "
                        + r" $M_\odot / \mathrm{yr}$"
                    ),
                    ha="left",
                    va="bottom",
                    transform=ax.transAxes,
                    bbox={"fc": "w", "ec": "None", "alpha": 0.7},
                    fontsize=7,
                    zorder=10000,
                )

            ax.scatter(
                self.pivot / (1 + z),
                model_phot * pnorm,
                fc="None",
                ec=fit_color,
                marker="s",
                s=80,
                zorder=100,
            )
            v = self.spec.valid
            ax.plot(
                self.spec["wave"][v] / (1 + z),
                (model_spec * snorm)[v],
                color=fit_color,
                alpha=0.5,
                zorder=100,
                label="model",
            )
            if show_fit_components:
                ax.plot(
                    self.spec["wave"][v] / (1 + z),
                    (comps_spec.T * snorm).T[v, :],
                    alpha=0.3,
                    zorder=80,
                )

            ymax = (model_spec * snorm)[v].max()
            ax.plot(
                self.spec["wave"][v] / (1 + z),
                scale_spec[v] / np.median(scale_spec[v]) * 0.5 * ymax,
                color="olive",
                alpha=0.2,
                lw=1,
                label="scale_spec",
            )
            ax.plot(
                self.spec["wave"][v] / (1 + z),
                scale_spec[v] ** 0 * 0.5 * ymax,
                color="olive",
                alpha=0.3,
                lw=1,
                ls=":",
            )

            ax.set_ylim(-0.1 * ymax, 1.1 * ymax)

        ax.plot(
            self.spec["wave"][self.spec.valid] / (1 + z),
            (self.spec["flux"] * snorm * scale_spec)[self.spec.valid],
            alpha=0.4,
            color="0.5",
        )

        ax.grid()
        if z > 0:
            ax.set_xlabel(
                r"$\lambda_\mathrm{rest}$ [$\,\mu\mathrm{m}\,$] (z="
                + f"{z:.3f}"
                + ")"
            )
        else:
            ax.set_xlabel(r"$\lambda_\mathrm{obs}$ [$\,\mu\mathrm{m}\,$]")

        if flam != 0:
            if flam == -2:
                _funit = r"$f_\lambda$"
            else:
                _funit = r"scaled $f_\nu$"

            ax.set_ylabel(
                _funit
                + r" [$\,\mu\mathrm{Jy}\,\cdot\,\mu^{"
                + f"{flam}"
                + r"}\,$]"
            )
        else:
            ax.set_ylabel(r"$f_\nu$ [$\,\mu\mathrm{Jy}\,$]")

        if fig is not None:
            fig.tight_layout(pad=1)

            utils.figure_timestamp(
                fig,
                text_prefix=os.path.basename(self.file).split(".spec")[0]
                + "\n",
                fontsize=6,
            )

        return fig, ax

    def set_calibration_polynomials(
        self,
        nspline=5,
        polynomial_order=None,
        poly_func="chebyshev",
        both_signs=False,
        wave_pixels=True,
    ):
        """
        Set calibration polynomials

        Parameters
        ----------

        """
        if polynomial_order is not None:

            if wave_pixels:
                poly_x = np.linspace(
                    self.spec["wave"].min(),
                    self.spec["wave"].max(),
                    len(self.spec["wave"]),
                )
            else:
                poly_x = self.spec["wave"]

            if poly_func == "chebyshev":
                poly = utils.cheb_templates(poly_x, order=polynomial_order)
            else:
                poly = utils.polynomial_templates(
                    poly_x,
                    ref_wave=np.nanmedian(self.spec["wave"][self.spec.valid]),
                    order=polynomial_order,
                )

            spec_poly = []
            for t in poly:
                spec_poly.append(poly[t].flux)
                # if both_signs:
                #     spec_poly.append(-1 * poly[t].flux)

            self.calib_poly = np.array(spec_poly)
        else:
            self.calib_poly = self.spec.bspline_array(nspline=nspline)

    def fit_templates(
        self,
        templates,
        z,
        scale_disp=1.5,
        velocity_sigma=100,
        fast_fit=True,
        get_design=False,
        fitter="nnls",
        eweight=1.0,
        relative_correction=False,
        redden=1.0,
        get_args=False,
    ):
        """
        Combined template + photometry fit
        """

        from scipy.optimize import nnls

        if get_args:
            return (
                templates,
                z,
                scale_disp,
                velocity_sigma,
                fast_fit,
                get_design,
                fitter,
                eweight,
                relative_correction,
                redden,
            )

        if self.calib_poly is None:
            spl = np.ones((1, len(self.spec["wave"])))
        else:
            spl = self.calib_poly

        spsh = spl.shape
        Nt = len(templates)

        with_igm = z > (self.spec["wave"][self.spec.valid].min() / 0.1216 - 1)

        if fast_fit:
            tflux, spt = integrate_filters_templates(
                self.filters,
                templates,
                z=z,
                spec=self.spec,
                redden=redden,
                scale_disp=scale_disp,
                velocity_sigma=velocity_sigma,
                with_igm=with_igm,
                F99=self.F99,
            )

        else:
            tflux = np.array(
                [
                    t.integrate_filter_list(
                        self.filters, z=z, flam=False, include_igm=with_igm
                    )
                    for t in templates
                ]
            )

            spt = np.array(
                [
                    self.spec.resample_eazy_template(
                        t,
                        z=z,
                        scale_disp=scale_disp,
                        velocity_sigma=velocity_sigma,
                        fnu=True,
                        with_igm=with_igm,
                    )
                    for t in templates
                ]
            )

        valid = self.spec.valid & True

        # Design matrix
        Nv = valid.sum()

        # Both signs on correction function for 'nnls'
        if fitter == "nnls_both":
            A = np.zeros((Nt + spsh[0] * 2, Nv + self.NFILT))
            A[:Nt, : self.NFILT] = tflux * 1
            A[:Nt, self.NFILT : self.NFILT + Nv] += spt[:, valid]
            A[Nt : Nt + spsh[0], self.NFILT : self.NFILT + Nv] = (
                -1 * (spl * self.spec["flux"])[:, valid]
            )
            A[Nt + spsh[0] :, self.NFILT : self.NFILT + Nv] = (
                1 * (spl * self.spec["flux"])[:, valid]
            )
        else:
            A = np.zeros((Nt + spsh[0], Nv + self.NFILT))
            A[:Nt, : self.NFILT] = tflux * 1
            A[:Nt, self.NFILT : self.NFILT + Nv] += spt[:, valid]
            A[Nt:, self.NFILT : self.NFILT + Nv] = (
                -1 * (spl * self.spec["flux"])[:, valid]
            )

        # Weighted by uncertainties
        Ax = A * 1
        Ax[:, : self.NFILT] *= self.photom_weight / self.photom_err
        Ax[:, self.NFILT : self.NFILT + Nv] /= self.spec["full_err"][
            valid
        ]  # * self.photom_weight

        if relative_correction:
            yx = np.append(
                self.photom[:, 0] * self.photom_weight / self.photom_err,
                (self.spec["flux"] / self.spec["full_err"])[valid],
            )
        else:
            yx = np.append(
                self.photom[:, 0] * self.photom_weight / self.photom_err,
                np.zeros(valid.sum()),
            )

        if get_design:
            return {
                "Nt": Nt,
                "A": A.T,
                "Ax": Ax.T,
                "yx": yx,
                "template_filters": tflux,
                "template_spec": spt,
                "spl_spec": spl,
                "valid": valid,
            }

        if fitter in ["nnls", "nnls_both"]:
            c = nnls(Ax.T, yx)
        else:
            c = np.linalg.lstsq(Ax.T, yx)

        chi = Ax.T.dot(c[0]) - yx
        chi2 = (chi**2).sum()

        model_phot = tflux.T.dot(c[0][:Nt])
        model_spec = spt.T.dot(c[0][:Nt])
        comps_spec = spt.T * c[0][:Nt]

        if A.shape[0] == Nt + spsh[0]:
            spl_spec = spl.T.dot(c[0][Nt:])
        else:
            spl_spec = spl.T.dot(c[0][Nt : Nt + spsh[0]]) - spl.T.dot(
                c[0][Nt + spsh[0] :]
            )

        if relative_correction:
            spl_spec = 1 + spl_spec

        return c, chi2, model_phot, comps_spec, model_spec, spl_spec

    def get_mw_alambda(self, Rv=3.1):
        """
        MW extinction curve
        """
        import eazy.utils

        fields = utils.read_catalog(
            """file_zout,count,ra,dec,EBV
    abell2744clu-grizli-v7.0-fix.eazypy.zout.fits,47888,3.5420553334483373,-30.370833030335604,0.0111
    abell2744clu-grizli-v7.1-fix.eazypy.zout.fits,54702,3.5418470307908065,-30.373627555310886,0.0111
    abell2744clu-grizli-v7.2-fix.eazypy.zout.fits,66172,3.539443035847806,-30.376123118841438,0.0111
    abell2744par-grizli-v7.0-fix.eazypy.zout.fits,26544,3.612553440565225,-30.473878599733652,0.0109
    abell370-full-grizli-v7.1-fix.eazypy.zout.fits,24957,40.02169201920347,-1.6353102162712103,0.0263
    abells1063-grizli-v7.0-fix.eazypy.zout.fits,1621,342.1839715617437,-44.531184752886624,0.0106
    aegis-3dhst-v4.1.zout.fits,41200,214.94339136431478,52.89583084495243,0.0062
    ceers-full-grizli-v7.0-fix.eazypy.zout.fits,76300,214.92239371475372,52.87323832533617,0.0061
    ceers-full-grizli-v7.2-fix.eazypy.zout.fits,76637,214.9219422202121,52.87305707120073,0.0061
    elgordo-grizli-v7.0-fix.eazypy.zout.fits,6775,15.735479872799319,-49.23736080804439,0.0086
    gdn-grizli-v7.0-fix.eazypy.zout.fits,37890,189.21284014236502,62.244830960160236,0.0105
    gdn-grizli-v7.3-fix.eazypy.zout.fits,70421,189.21248165594375,62.229156662691295,0.0105
    gds-grizli-v7.0-fix.eazypy.zout.fits,52427,53.143897854381265,-27.796587275230348,0.0068
    gds-grizli-v7.2-fix.eazypy.zout.fits,70357,53.13274259729532,-27.807580000139787,0.0066
    gds-sw-grizli-v7.0-fix.eazypy.zout.fits,71703,53.1037394892602,-27.863567797471585,0.0065
    goodsn-3dhst-v4.1.zout.fits,38279,189.2245628155191,62.23691450133485,0.0105
    j112716p4228-grizli-v7.0-fix.eazypy.zout.fits,2997,171.81664913408176,42.46241265674058,0.0165
    macs0416-clu-grizli-v7.0-fix.eazypy.zout.fits,9608,64.05839452797258,-24.089349632772095,0.0351
    macs0417-full-grizli-v7.2-fix.eazypy.zout.fits,17318,64.3864258510793,-11.850465599295434,0.0332
    macs0647-grizli-v7.0-fix.eazypy.zout.fits,7956,101.95958050521024,70.22835515966803,0.0939
    macs1149-clu-grizli-v7.0-fix.eazypy.zout.fits,17537,177.39567957687845,22.338789514664434,0.0201
    macs1423-full-grizli-v7.2-fix.eazypy.zout.fits,17471,215.90790345715138,24.124783752454018,0.0197
    mrg0138-grizli-v7.2-fix.eazypy.zout.fits,2066,24.524240500579797,-21.930739762027482,0.0138
    ngdeep-grizli-v7.2-fix.eazypy.zout.fits,25259,53.26650307267384,-27.851868092132094,0.0062
    primer-cosmos-east-grizli-v7.0-fix.eazypy.zout.fits,59680,150.15432668112422,2.3250990860881062,0.0143
    primer-cosmos-west-grizli-v7.0-fix.eazypy.zout.fits,59114,150.09271442012331,2.2869259920960583,0.0155
    primer-uds-north-grizli-v7.0-fix.eazypy.zout.fits,73227,34.37100590761816,-5.147697566420972,0.0199
    primer-uds-south-grizli-v7.0-fix.eazypy.zout.fits,70325,34.376375172875285,-5.254176471433919,0.02
    rxcj0600-grizli-v7.2-fix.eazypy.zout.fits,6819,90.0441426082566,-20.1545462883024,0.0436
    rxj2129-grizli-v7.0-fix.eazypy.zout.fits,3295,322.4082804314248,0.09505525882316537,0.0346
    smacs0723-grizli-v7.0-fix.eazypy.zout.fits,8388,110.75718752896465,-73.46730417937614,0.1909
    sunrise-grizli-v7.0-fix.eazypy.zout.fits,10194,24.355194560433564,-8.453755068096676,0.0286
    ulasj1342-grizli-v7.2-fix.eazypy.zout.fits,4592,205.53291057447814,9.494873844701457,0.0207""",
            format="csv",
        )

        file_zout = self.query["file_zout"][0]
        ix = fields["file_zout"] == file_zout
        if ix.sum() == 0:
            EBV = eazy.utils.get_mw_dust(
                self.query["ra"][0], self.query["dec"][0]
            )
        else:
            EBV = fields["EBV"][ix][0]

        F99 = eazy.utils.GalacticExtinction(EBV=EBV, Rv=Rv)
        return EBV, F99

    def fit_redshift_dust_grid(
        self,
        templates,
        dust_model="calzetti",
        Alam=None,
        start_zwidth=0.01,
        Avs=[0.0, 0.4, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
        nsteps=16,
        **kwargs,
    ):
        """
        Fit redshift and dust on a grid
        """
        from . import resample_numba

        twave = templates[0].wave / 1.0e4
        if Alam is None:
            if dust_model == "calzetti":
                Alam = resample_numba.calzetti2000_alambda(twave, 0)
            elif dust_model == "salim":
                Alam = resample_numba.salim2018_fit_alambda(twave, 0, 0)
            elif dust_model == "smc":
                Alam = resample_numba.smc_alambda(twave, 0)

        z = self.redshift
        zgrid = utils.log_zgrid(
            [z - start_zwidth * (1 + z), z + start_zwidth * (1 + z)],
            start_zwidth / nsteps,
        )

        cc = np.meshgrid(Avs, zgrid)[0] * 0

        fig = plt.figure(figsize=(6, 4))
        axes = []

        dw, dwp = 0.45, 0.012
        axes.append(fig.add_axes((0.1, 0.1, dw, 0.88)))
        axes.append(fig.add_axes((0.1 + dw + dwp, 0.1, dw, 0.88)))

        colors = ["0.5", "orange", "red", "tomato"]

        # First pass
        for j, Avi in tqdm(enumerate(Avs)):
            for i, zi in enumerate(zgrid):
                _ = self.fit_templates(
                    templates, zi, redden=10 ** (-0.4 * Avi * Alam), **kwargs
                )
                c, cc[i, j], model_phot, comps_spec, model_spec, spl_spec = _

            axes[0].plot(zgrid, cc[:, j], alpha=0.2, color=colors[0])

        for ci in cc:
            axes[1].plot(Avs, ci, alpha=0.2, color=colors[0])

        ix = np.argmin(cc.flatten())
        ib, jb = np.unravel_index(ix, cc.shape)

        zb = zgrid[ib]
        Avb = Avs[jb]

        msg = f"fit_redshift_dust_grid: iter {0}  z={zb:.3f}  Av={Avb:.2f}"
        utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

        # Iterate
        for ii, dust_iter in enumerate(range(2)):

            ib = int(np.clip(ib, 3, len(zgrid) - 4))

            # zgrid = utils.log_zgrid([zgrid[ib-2], zgrid[ib+2]], 0.0002)
            if ii > 0:
                zgrid = np.linspace(zgrid[ib - 3], zgrid[ib + 3], nsteps)

            jb = int(np.clip(jb, 1, len(Avs) - 2))

            Avs = np.linspace(Avs[jb - 1], Avs[jb + 1], nsteps)
            # print(Avs)

            # print(len(zgrid))
            cc = np.meshgrid(Avs, zgrid)[0] * 0

            for j, Avi in tqdm(enumerate(Avs)):
                for i, zi in enumerate(zgrid):
                    _ = self.fit_templates(
                        templates,
                        zi,
                        redden=10 ** (-0.4 * Avi * Alam),
                        **kwargs,
                    )
                    (
                        c,
                        cc[i, j],
                        model_phot,
                        comps_spec,
                        model_spec,
                        spl_spec,
                    ) = _

                axes[0].plot(
                    zgrid, cc[:, j], alpha=0.2, color=colors[dust_iter + 1]
                )

            for ci in cc:
                axes[1].plot(Avs, ci, alpha=0.2, color=colors[dust_iter + 1])

            ix = np.argmin(cc.flatten())
            ib, jb = np.unravel_index(ix, cc.shape)

            zb = zgrid[ib]
            Avb = Avs[jb]

            # print(dust_iter, zb, Avb)
            msg = f"fit_redshift_dust_grid: iter {dust_iter+1}  "
            msg += f"z={zb:.3f}  Av={Avb:.2f}"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

        # Finish plot
        ymi = cc.min()
        axes[0].set_ylim(ymi - 10, ymi + 100)
        axes[0].set_xlabel("z")
        axes[1].set_xlabel("Av")
        axes[1].set_xlim(Avb - 0.5, Avb + 0.5)
        axes[1].set_yticklabels([])
        axes[0].set_ylabel(r"$\chi^2$")

        for ax in axes[:2]:
            ax.grid()
            ax.set_ylim(ymi - 10, ymi + 100)

        redden = 10 ** (-0.4 * Avb * Alam)
        ax = axes[0]

        ax.text(
            0.06,
            0.98,
            os.path.basename(self.file).split(".spec")[0],
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=6,
            bbox={"fc": "w", "ec": "None"},
        )

        ax.text(
            0.06,
            0.03,
            f"z = {zb:.4f}",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            fontsize=7,
            bbox={"fc": "w", "ec": "None"},
        )

        ax = axes[1]
        ax.text(
            0.06,
            0.03,
            r"$A_V$" f" = {Avb:.2f}",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            fontsize=7,
            bbox={"fc": "w", "ec": "None"},
        )

        fig.tight_layout(pad=1)

        res = {
            "best_z": zb,
            "best_Av": Avb,
            "best_redden": 10 ** (-0.4 * Avb * Alam),
            "dust_model": dust_model,
            "Alam": Alam,
        }

        return fig, res

    def get_continuity_inputs(self, zb, templates, redden, **kwargs):
        """ """
        ####
        # Arguments for the fit
        xargs = self.fit_templates(
            templates,
            zb,
            get_design=True,
            redden=redden,
            get_args=True,
            **kwargs,
        )

        X = self.fit_templates(*xargs)

        Nspl = X["spl_spec"].shape[0]

        # Nonzero templates
        X["ok"] = X["A"].sum(axis=0) != 0
        X["oNt"] = X["Nt"] - (~X["ok"]).sum()
        X["oA"] = X["A"][:, X["ok"]]
        X["oAx"] = X["Ax"][:, X["ok"]]
        X["otemplate_filters"] = X["template_filters"][X["ok"][:-Nspl], :]
        X["otemplate_spec"] = X["template_spec"][X["ok"][:-Nspl], :]
        X["NFILT"] = X["otemplate_filters"].shape[1]
        X["valid"] = self.spec["valid"]

        X["Nv"] = X["valid"].sum()

        # parameter names
        pnames = []
        for j in np.where(X["ok"][:-Nspl])[0]:
            pname = (
                (templates)[j]
                .name.split("_av")[0]
                .split("logz")[-1]
                .replace("tage", "t")
                .split(".fits")[0]
                .replace("nebular_continuum_H", "recomb")
            )
            if "_" in pname:
                pname = "_".join(pname.split("_")[1:])

            pnames.append(pname)

        for j in range(Nspl):
            pnames.append(f"poly {j}")

        return X, pnames

    def fit_continuity_sfh(
        self,
        zb,
        templates,
        redden,
        Nc,
        cont_prior=student_t(df=2, scale=0.3),
        res=None,
        tol=1.e-6,
        **kwargs,
    ):
        """
        fit SFH with continuity prior
        """
    
        #####
        # Priors and loss functions
        X, pnames = self.get_continuity_inputs(zb, templates, redden, **kwargs)

        phot_loss = norm(
            loc=self.photom[:, 0],
            scale=self.photom_err / self.photom_weight ** (1),
        )

        spec_loss = norm(
            loc=self.spec["flux"][X["valid"]],
            scale=self.spec["full_err"][X["valid"]],
        )

        x0 = nnls(X["oAx"], X["yx"])[0]
        x0c = x0[:Nc] * 1
        xi = np.arange(Nc)
        x0c[x0c == 0] = np.interp(xi[x0c == 0], xi[x0c != 0], x0c[x0c != 0])
        x0[:Nc] = np.log10(x0c)
        x0[:Nc] = np.log10(x0c) * 0 + np.median(np.log10(x0c))

        if res is None:
            res = minimize(
                self.objfun_continuity,
                x0,
                method="bfgs",
                tol=tol,
                args=(
                    X,
                    Nc,
                    cont_prior,
                    phot_loss,
                    spec_loss,
                    0,
                ),
            )

        rfit = self.objfun_continuity(
            res.x, X, Nc, cont_prior, phot_loss, spec_loss, 1
        )

        fit_results = (
            [rfit["full_coeffs"]],
            0,
            rfit["model_phot"],
            rfit["components_spec"],
            rfit["model_spec"],
            rfit["spl_corr"],
        )

        return res, fit_results, pnames

    @staticmethod
    def objfun_continuity(thet, X, Nc, cont_prior, phot_loss, spec_loss, ret):
        """
        Objective function for continuity SFH
        """
        theta = np.array(thet)

        linear = theta * 1.0
        sfr_coeffs = theta[:Nc]
        dlog_sfr = np.diff(sfr_coeffs)

        lnp_prior = cont_prior.logpdf(dlog_sfr).sum()

        linear[:Nc] = 10**sfr_coeffs
        if ret == 2:
            print(theta, linear)

        if 0:
            resid = X["oAx"].dot(linear) - X["yx"]
            lnp_data = data_loss.logpdf(resid).sum()
            lnp_total = lnp_prior + lnp_data
        else:
            model_phot = X["oA"][: X["NFILT"], :].dot(linear)
            model_spec = X["otemplate_spec"].T.dot(linear[: X["oNt"]])
            spl_corr = X["spl_spec"].T.dot(theta[X["oNt"] :])

            lnp_photom = phot_loss.logpdf(model_phot).sum()

            lnp_spec = spec_loss.logpdf(
                (model_spec / spl_corr)[X["valid"]]
            ).sum()

            lnp_total = lnp_prior + lnp_photom + lnp_spec
            if ret == 2:
                print(lnp_prior, lnp_photom, lnp_spec)

            if ret == 1:
                full_coeffs = np.zeros(X["A"].shape[1])
                full_coeffs[X["ok"]] += linear

                components_spec = X["template_spec"].T * full_coeffs[: X["Nt"]]

                return {
                    "full_coeffs": full_coeffs,
                    "coeffs": linear,
                    "model_phot": model_phot,
                    "model_spec": model_spec,
                    "components_spec": components_spec,
                    "spl_corr": spl_corr,
                    "lnp_prior": lnp_prior,
                    "lnp_photom": lnp_photom,
                    "lnp_spec": lnp_spec,
                }

        if (ret & 4) > 0:
            return Tensor([-lnp_total / 2])

        return -1 * lnp_total / 2

    @staticmethod
    def objfun_hyper_av(theta, args):
        """
        Objective function for fitting Av with continuity SFH
        """
        zb, cont_prior, templates, twave, Alam, basis, self = args
        Avi = theta[0]/10
    
        redden_i = 10 ** (-0.4 * Avi * Alam)
    
        res, fit_results, pnames = self.fit_continuity_sfh(
            zb,
            templates,
            redden_i,
            basis.tstep_Nc,
            cont_prior=cont_prior,
            # **kwargs,
        )
        print(f"    Optimize Av for continuity prior: {Avi:.3f}  {res.fun:.3f}")
    
        return res.fun
    
    def continuity_sfh_pipeline(
        self,
        calib_kwargs={},
        basis_kwargs={},
        dust_kwargs={},
        optimize_av_sfh=False,
        scale_disp=1.3,
        cont_prior=student_t(df=2, scale=0.3),
        make_corner=True,
        with_quasar=False,
        **kwargs,
    ):
        """
        Run the full continuity fit
        """
        import scipy.stats
        import cov_corner

        self.set_calibration_polynomials(**calib_kwargs)

        self.basis = BasisTemplates(z=self.redshift, **basis_kwargs)

        # for _iter in range(2):
        #     _ = spectrum.plot_spectrum(
        #         inp=self.spec,
        #         z=self.redshift,
        #         # eazy_templates=(self.basis.tstep_Av_grid + self.basis.extra_lines),
        #         sys_err=self.sys_err,
        #         scale_uncertainty_kwargs={},
        #         return_fit_results=True,
        #     )
        
        textra = [t for t in self.basis.extra_lines]

        if (self.spec["wave"] / (1 + self.redshift)).max() > 3.4:
            msg = f"SpectrumPhotometry: {self.basename}  Include PAH template"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)            
            textra += [self.basis.pah]

        if with_quasar:
            msg = f"SpectrumPhotometry: {self.basename}  Include QSO template"
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)
            textra += [self.basis.qso]

        templates = self.basis.tstep + textra

        dust_fig, dust_fit = self.fit_redshift_dust_grid(
            # self.basis.tstep, #templates,
            templates,
            **dust_kwargs
        )
        dust_fig.savefig(self.basename.replace('.spec.fits', '.grid.png'))

        zb = dust_fit["best_z"]
        Avb = dust_fit["best_Av"]
        redden = dust_fit["best_redden"]

        args = self.fit_templates(
            templates, zb, scale_disp, redden=redden, get_args=True
        )

        fig, ax = self.plot(
            flam=-2, fit_args=args, show_fit_components=True, z=zb
        )
        fig.savefig(self.basename.replace('.spec.fits', '.sfh.flam.png'))

        fig, ax = self.plot(
            flam=0, fit_args=args, show_fit_components=True, z=zb
        )
        fig.savefig(self.basename.replace('.spec.fits', '.sfh.fnu.png'))

        _ = self.fit_templates(*args)
        c0 = _[0]
        
        if optimize_av_sfh:
            twave = templates[0].wave / 1.0e4
            Alam = dust_fit["Alam"]
            _args = zb, cont_prior, templates, twave, Alam, self.basis, self

            hyper_res = minimize(
                self.objfun_hyper_av,
                [Avb*10],
                method='powell',
                tol=1.e-4,
                args=(_args,),
                options={'maxfev': 16}
            )

            Avb = hyper_res.x[0] / 10.
            msg = f'continuity-optimized Av: {Avb:.2f}'
            utils.log_comment(utils.LOGFILE, msg, verbose=VERBOSE)

            redden = 10 ** (-0.4 * Avb * Alam)
            
        res, fit_results, pnames = self.fit_continuity_sfh(
            zb,
            templates,
            redden,
            self.basis.tstep_Nc,
            cont_prior=cont_prior,
            # **kwargs,
        )
        
        fig = self.basis.plot_sfh(zb, c0[0], res=res, file=self.file)
        fig.savefig(self.basename.replace('.spec.fits', '.sfh.png'))

        xargs = self.fit_templates(
            templates, zb, scale_disp, redden=None, get_args=True
        )[:-1]

        fig, ax = self.plot(
            flam=-2,
            fit_args=(xargs) + (redden,),
            fit_results=fit_results,
            show_fit_components=True,
            z=zb,
        )
        fig.savefig(self.basename.replace('.spec.fits', '.csfh.flam.png'))

        fig, ax = self.plot(
            flam=0,
            fit_args=(xargs) + (redden,),
            fit_results=fit_results,
            show_fit_components=True,
            z=zb,
        )
        fig.savefig(self.basename.replace('.spec.fits', '.csfh.fnu.png'))

        # Add to spectrum outputs
        # _ = self.fit_templates(*args)
        c, chi2, model_phot, comps_spec, model_spec, spl_spec = fit_results
        continuum_model = comps_spec[:, :self.basis.tstep_Nc].sum(axis=1)

        self.spec.spec["continuum_model"] = continuum_model
        self.spec.spec["sfh_full_model"] = model_spec

        if self.basis.lines_mode in ("default", "separate"):
            
            tcont = self.basis.tstep[:-1] + [self.basis.neb_cont] + textra
            
            _res, _fit_results, pnames = self.fit_continuity_sfh(
                zb,
                tcont,
                redden,
                self.basis.tstep_Nc,
                res=res,
                cont_prior=cont_prior,
                # **kwargs,
            )

            _comps_spec = _fit_results[3]
            continuum_model = _comps_spec[:, :self.basis.tstep_Nc].sum(axis=1)
            continuum_model += _comps_spec[:, self.basis.tstep_Nc+1]
            
            self.spec.spec["neb_continuum"] = (
                self.spec.spec["continuum_model"] - continuum_model
            )
            
            self.spec.spec["continuum_model"] = continuum_model
            
        self.spec.spec["sfh_deredden"] = 1.0 / np.interp(
            self.spec.spec["wave"] / (1 + zb),
            self.basis.tstep[0].wave / 1.0e4,
            redden,
        )

        if "scale_to_photometry" in self.spec.spec.colnames:
            self.spec.spec["scale_to_photometry"] *= spl_spec
        else:
            self.spec.spec["scale_to_photometry"] = spl_spec

        self.spec.spec["flux"] *= spl_spec
        self.spec.spec["err"] *= spl_spec

        spectrum.set_spec_sys_err(self.spec.spec, sys_err=self.sys_err)

        self.spec.meta["grid_Av"] = Avb
        self.spec.meta["grid_z"] = zb
        self.spec.meta["sfh_logz"] = self.basis.logz

        ######
        # Corner plot
        if make_corner:
            Nspl = self.calib_poly.shape[0]
            hsub = np.zeros(len(res.x), dtype=bool)
            hsub[: self.basis.tstep_Nc] = True
            hsub[-Nspl:] = True
            pn = [pnames[j] for j in np.where(hsub)[0]]

            rnorm = scipy.stats.multivariate_normal(
                mean=res.x[hsub], cov=res.hess_inv[hsub, :][:, hsub]
            )

            cfig = cov_corner.cov_corner(rnorm.cov, rnorm.mean, pn)

            utils.figure_timestamp(
                cfig,
                text_prefix=os.path.basename(self.file).split(".spec")[0] + "\n",
                fontsize=6,
                x=0.97,
                y=0.97,
                ha="right",
                va="top",
            )

            cfig.savefig(self.basename.replace('.spec.fits', '.sfh.corner.png'))

        ######
        # Update spectrum
        
        fig, sp, data = spectrum.plot_spectrum(
            inp=self.spec,
            z=zb, #self.redshift,
            sys_err=self.sys_err,
            scale_disp=scale_disp,
            vel_width=100,
            trim_negative=False,
        )

        sp.write(self.basename.replace('.spec.fits','.spec.sfh.fits'), overwrite=True)
        
        return templates


def integrate_filters_templates(
    filters,
    templates,
    z=8,
    spec=None,
    scale_disp=1.3,
    velocity_sigma=100,
    with_igm=True,
    F99=None,
    redden=1.0,
):
    """
    Integrate filters through templates and resample to spectra

    Parameters
    ----------
    filters : list
        List of N `eazy.filters.FilterDefinition` objects

    templates : list
        List of M `eazy.templates.Template` objects *resampled to same wavelength grid*

    z : float
        Redshift

    spec : `msaexp.specrum.SpectrumSampler` object

    scale_disp, velocity_sigma : float
        Resolution / dispersion for resampled spectrum

    with_igm : bool
        Include IGM absorption

    F99 : `eazy.utils.GalacticExtinction`
        Object for applying MW extinction

    redden : float, array-like
        Array to multiply to template fluxes, e.g., for dust reddening

    Returns
    -------
    filter_fluxes : (M, N) array
        Integrated flux densities

    spec_fluxes : (M, Nspec) array, None
        Resampled templates if ``spec`` provided
    """
    from .resample_numba import (
        compute_igm,
        integrate_filter,
        resample_template_numba,
    )

    if with_igm:
        igmz = compute_igm(z, templates[0].wave * (1 + z))
    else:
        igmz = 1.0

    igmz = igmz * redden

    if F99 is not None:
        igmz = igmz * 10 ** (-0.4 * F99(templates[0].wave * (1 + z)))

    filter_fluxes = []
    templ_wrest = templates[0].wave
    templ_wobs = templ_wrest * (1 + z)

    for t in templates:
        flux_i = [
            integrate_filter(
                filter.wave,
                filter.throughput,
                filter.norm,
                templ_wobs,
                t.fnu * igmz,
            )
            for filter in filters
        ]
        filter_fluxes.append(flux_i)

    if spec is not None:
        spec_fluxes = []
        for t in templates:
            if hasattr(t, "velocity_sigma"):
                v = t.velocity_sigma
            else:
                v = velocity_sigma

            res = resample_template_numba(
                spec.spec_wobs,
                spec.spec_R_fwhm * scale_disp,
                templ_wobs / 1.0e4,
                t.fnu * igmz,
                velocity_sigma=v,
            )
            spec_fluxes.append(res)

        spec_fluxes = np.array(spec_fluxes)
    else:
        spec_fluxes = None

    return np.array(filter_fluxes), spec_fluxes


def fsps_template_obsframe_scale(z=1.0, unit=u.microJansky, **kwargs):
    """ """
    global COSMOLOGY
    from astropy.constants import L_sun

    dL = COSMOLOGY.luminosity_distance(z).to(u.cm)
    obs_unit = (1 * L_sun / u.Hz / (4 * np.pi * dL**2)).to(unit) * (1 + z)
    return obs_unit.value


def template_mass_sfr(templates, coeffs, z=1.0):
    """
    Extract mass, sfh tags from eazy FSPS templates
    """
    tstep_masses = []
    tstep_sfr_continuum = []

    for t in templates:
        if "stellar_mass" in t.meta:
            tstep_masses.append(t.meta["stellar_mass"])
        else:
            tstep_masses.append(0.0)

        if "SFR" in t.meta:
            tstep_sfr_continuum.append(t.meta["SFR"])
        else:
            tstep_sfr_continuum.append(0.0)

    tstep_masses = np.array(tstep_masses)
    tstep_sfr_halpha = np.array(
        [1.0 if "nebular_continuum_H" in t.name else 0.0 for t in templates]
    )

    zscale = fsps_template_obsframe_scale(z=z)

    mass = coeffs[: len(templates)] * tstep_masses / zscale
    sfr_continuum = coeffs[: len(templates)] * tstep_sfr_continuum / zscale
    sfr_halpha = coeffs[: len(templates)] * tstep_sfr_halpha / zscale

    return mass, sfr_continuum, sfr_halpha


DEFAULT_AVS = -2.5 * np.log10(np.linspace(0.01, 0.999, 7))[::-1]

EXTRA_EMISSION_LINES = [
    "MgII",
    "OII",
    "OIII",
    "NII",
    "SII",
    "HeI-1083",
    "FeII-11128",
    "PII-11886",
    "FeII-12570",
    "FeII-16440",
    "FeII-16877",
    "FeII-17418",
    "FeII-18362",
]


class BasisTemplates:
    def __init__(self, **kwargs):
        self.load_basis_templates(**kwargs)

    def load_basis_templates(
        self,
        z=0,
        log_zsol=0.0,
        ncomps=6,
        spec_R=1800,
        trim_age=1,
        lines_mode="separate",
        av_grid=DEFAULT_AVS,
        extra_line_names=EXTRA_EMISSION_LINES,
        velocity_sigma=100,
        broad_sigma=2500,
    ):
        """
        Load SFH basis templates

        Parameters
        ----------
        z : float
            Redshift for limiting age of oldest template

        log_zsol : float
            Will load templates from log Z/Zsun = [0.25, 0.0, -0.3, -0.6, -1.0, -2.0]
            closest to this target metallicity.

        ncomps : [6, 10]
            Which set of components to use

        spec_R : float
            Target spectral resolution

        trim_age : int
            Trim to templates younger than the age of the universe.  If > 1, allow
            additional (trim_age - 1) bins older than the age of the universe at ``z``

        lines_mode : str
            Handling of emission line templates

        av_grid : array-like
            Apply dust attenuation at a precomputed grid of Av

        Returns
        -------
        res : dict
            Template and metadata dictionary

        """
        import eazy
        import glob

        from .resample_numba import (
            calzetti2000_alambda,
            sample_gaussian_line_numba,
        )

        tlog = 10 ** np.arange(
            np.log10(100), np.log10(5.3e4), 1.0 / spec_R / np.log(10)
        )

        if ncomps == 6:
            logz_grid = [0.25, 0.0, -0.3, -0.6, -1.0, -2.0]
        else:
            logz_grid = [0.0, -0.3, -0.6, -1.0, -2.0]

        zix = np.argmin(np.abs(np.array(logz_grid) - log_zsol))
        logz = logz_grid[zix]

        if ncomps == 6:
            sfh_tsteps = [
                0.000001,
                0.02,
                0.035,
                0.15587477,
                0.69419844,
                3.09165785,
                13.76889912,
            ]
            all_bfiles = glob.glob(f"basis_templates/*logz{logz:.2f}*")
            all_bfiles.sort()

            old = all_bfiles.pop(6)
        else:
            ncomps = 10
            sfh_tsteps = [
                0.000001,
                0.02,
                0.035,
                0.06798023,
                0.13203747,
                0.25645535,
                0.49811124,
                0.9674776,
                1.87912423,
                3.64980842,
                7.08899456,
                13.76889912,
            ]
            all_bfiles = glob.glob(f"basis_templates_10comps/*logz{logz:.2f}*")
            all_bfiles.sort()

            old = all_bfiles.pop(10)

        all_bfiles.sort()

        ###################
        # Separate line + continuum

        tage = COSMOLOGY.age(z).to(u.Gyr).value
        ix = np.where(tage > np.array(sfh_tsteps))[0][-1] + trim_age

        if lines_mode == "f_escape":
            # template with lines and nolines, i.e., escape fraction
            bfiles = all_bfiles[0:1] + all_bfiles[2:-2] + [old]
            bfiles = bfiles[: ix + 1]
        elif lines_mode == "fsps":
            # template with lines from original FSPS
            bfiles = all_bfiles[0:1] + all_bfiles[3:-2] + [old]
            bfiles = bfiles[:ix]
        else:
            bfiles = (all_bfiles[2:-2] + [old])[:ix] + all_bfiles[-2:]

        ###################
        msg = [f"load_basis_templates: {ncomps} components, logz = {logz}"]
        msg += [f"load_basis_templates: z={z:.3f} t={tage:.2} Gyr"]
        msg += [f"load_basis_templates: lines_mode = '{lines_mode}'", ""]
        for i, t in enumerate(bfiles):
            msg += [f"load_basis_templates: {i:>2} {os.path.basename(t)}"]

        utils.log_comment(utils.LOGFILE, "\n".join(msg), verbose=VERBOSE)

        tstep = []
        tstep_Av = []

        for bf in bfiles:
            bi = eazy.templates.Template(bf)
            bi.resample(tlog, in_place=True)

            Alam = calzetti2000_alambda(bi.wave / 1.0e4, 0)

            # Av grid
            for i, Av in enumerate(av_grid):
                red = 10 ** (-0.4 * Av * Alam)
                ri = eazy.templates.Template(
                    arrays=(bi.wave, bi.flux.flatten() * red),
                    name=bi.name.replace("av0.00", f"av{Av:.2f}"),
                )
                ri.meta = bi.meta
                ri.redden_array = red
                ri.redden_Av = Av

                if i == 0:
                    tstep.append(ri)

                tstep_Av.append(ri)

        # Neb continuum only
        neb_cont = eazy.templates.Template(all_bfiles[1])
        neb_cont.resample(tlog, in_place=True)

        bnoline = eazy.templates.Template(all_bfiles[2])
        bnoline.resample(tlog, in_place=True)

        line_only = eazy.templates.Template(all_bfiles[-1])
        line_only.name = neb_cont.name.split("_")[2] + "_just_lines"
        line_only.resample(tlog, in_place=True)

        neb_cont.flux -= bnoline.flux
        neb_cont.name = neb_cont.name.split("_")[2] + "_neb_cont_only"
        neb_cont.fnu = neb_cont.flux_fnu() * 1

        line_only.flux -= neb_cont.flux
        line_only.fnu = line_only.flux_fnu() * 1

        broad_lines = eazy.templates.Template(
            arrays=(line_only.wave, line_only.flux_flam()),
            name="broad_rec_lines",
        )
        broad_lines.velocity_sigma = broad_sigma
        broad_lines.fnu = broad_lines.flux_fnu() * 1

        # if (1) & (self.spec['wave'].max() / (1+z) > 3.3):
        ##
        # PAH template
        _temp = utils.pah33(tlog)
        tpf = None
        tpn = {
            "line PAH-3.29": 1.0,
            "line PAH-3.40": 0.15,
            "line PAH-3.47": 0.5,
        }

        for t in _temp:
            # print(t)
            if "x3.47" in t:
                continue

            _tp = _temp[t]
            if tpf is None:
                tpf = _tp.flux * 0

            tpf += _tp.flux * tpn[t]

        pah = eazy.templates.Template(
            name="line PAH-3um", arrays=(_tp.wave, tpf)
        )

        # Set fnu helper attribute
        for t in (
            tstep + tstep_Av + [neb_cont, bnoline, line_only, broad_lines, pah]
        ):
            t.fnu = t.flux_fnu()

        # Extra emission line components
        lw, lr = utils.get_line_wavelengths()
        extra_lines = []
        for ln in extra_line_names:
            lf = tlog * 0.0
            lrsum = 0.
            
            for lwi, lri in zip(lw[ln], lr[ln]):
                lf += sample_gaussian_line_numba(
                    tlog,
                    tlog**0 + spec_R,
                    lwi,
                    line_flux=lri * 1.e10, # Put scale closer to templates
                    velocity_sigma=velocity_sigma,
                )
                lrsum += lri

            lf /= lrsum

            extra_lines.append(
                eazy.templates.Template(arrays=(tlog, lf), name="line " + ln)
            )

            extra_lines[-1].fnu = extra_lines[-1].flux_fnu()

        # Quasar template
        qso = eazy.templates.Template(
            "/Users/gbrammer/Research/JWST/Projects/QsoTempl/qsogen/temple_qsogen_z2.fits"
        )
        qso.resample(tlog, in_place=True)
        qso.fnu = qso.flux_fnu() * 1

        # Output dict
        self.logz = logz
        self.z = z
        self.tage = tage
        self.ncomps = ncomps
        self.spec_R = spec_R
        self.lines_mode = lines_mode
        self.sfh_tsteps = sfh_tsteps
        self.tstep = tstep
        self.tstep_Av_grid = tstep_Av
        self.neb_cont = neb_cont
        self.noline = bnoline
        self.line_only = line_only
        self.broad_lines = broad_lines
        self.pah = pah
        self.qso = qso
        self.extra_lines = extra_lines

        btemp = tstep
        btemp_masses = []
        for t in btemp:
            if "stellar_mass" in t.meta:
                btemp_masses.append(t.meta["stellar_mass"])
            else:
                btemp_masses.append(0.0)

        btemp_masses = np.array(btemp_masses)
        btemp_sfr_halpha = np.array(
            [1.0 if "nebular_continuum_H" in t.name else 0.0 for t in btemp]
        )

        Ncontinuum = 0
        for t in btemp:
            if t.name.startswith("fsps_"):
                Ncontinuum += 1

        self.tstep_mass = btemp_masses
        self.tstep_sfh_halpha = btemp_sfr_halpha
        self.tstep_Nc = Ncontinuum
    
    @staticmethod
    def effective_Av(self, coeffs_fit, templates_av):
        """
        Compute effective Av from template coeffs

        Parameters
        ----------
        coeffs_fit : array-like
            Fit coefficients from fit with ``tstep_Av_grid`` templates

        templates_av : list
            List of templates

        Returns
        -------
        (wave, effective_redden) : array-like
            Effective attenuation curve

        effective_av : float
            Effective Av of the combination
    
        """

        coeffs_av = coeffs_fit[:len(templates_av)]
    
        fnu_array = np.array([t.fnu for t in templates_av])

        redden_array = np.array([
            t.redden_array if hasattr(t, 'redden_array') else t.wave**0
            for t in templates_av
        ])

        template_vflux = [np.interp(5500., t.wave, t.fnu) for t in templates_av]

        as_fit = (fnu_array**1).T.dot(coeffs_av)
        de_redden = (fnu_array**1 / redden_array**1).T.dot(coeffs_av)

        effective_redden = as_fit / de_redden
    
        tau = np.array([
            10**(-0.4*t.redden_Av) if hasattr(t, 'redden_Av') else 1.
            for t in templates_av
        ])
        
        tau_fit = (
            (template_vflux * coeffs_av).sum()
            / (template_vflux / tau * coeffs_av).sum()
        )

        effective_av = -2.5*np.log10(tau_fit)
    
        print('Effective Av: ', effective_av)
    
        # w = templates_av[0].wave
        # plt.plot(w, effective_redden)
        # # plt.plot(w, de_redden, alpha=0.5)
        # plt.scatter(5500, 10**(-0.4*effective_av), color='r', zorder=100)
        # plt.ylim(0, 1.05)
        # plt.plot(w, 10**(-0.4 * Alam * effective_av), color='0.5', alpha=0.5)
    
        return (w, effective_redden), effective_av

    def plot_sfh(self, zb, coeffs, file=None, res=None, **kwargs):
        """
        Plot log and linear SFH
        """
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))

        # log
        _ = self.sub_plot_sfh(
            zb,
            coeffs,
            ax=axes[0],
            ylog=True,
            min_threshold=1.0e-2,
            as_age=False,
            res=res,
        )

        # linear
        _ = self.sub_plot_sfh(
            zb,
            coeffs,
            ax=axes[1],
            ylog=False,
            min_threshold=1.0e-2,
            as_age=True,
            res=res,
        )

        fig.tight_layout(pad=0.1)

        if file is not None:
            utils.figure_timestamp(
                fig,
                text_prefix=os.path.basename(file).split(".spec")[0] + "\n",
                fontsize=6,
                x=0.97,
                y=0.5,
            )

        return fig

    def sub_plot_sfh(
        self,
        zb,
        coeffs,
        ax=None,
        res=None,
        draws=1000,
        figsize=(7, 4),
        ylog=False,
        min_threshold=1.0e-3,
        as_age=False,
        **kwargs,
    ):
        """ """

        import scipy.stats

        # from astropy.cosmology import WMAP9
        tage = COSMOLOGY.age(zb).to("Gyr").value

        sfh_tsteps = self.sfh_tsteps
        Nc = self.tstep_Nc

        zscale = fsps_template_obsframe_scale(zb)
        if hasattr(zscale, "value"):
            zscale = zscale.value

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None

        if ylog:
            sfh_y = np.log10(
                np.maximum(
                    coeffs[:Nc],
                    min_threshold * coeffs[:Nc].max(),
                )
                / zscale
            )
            ymin = sfh_y.min()
        else:
            sfh_y = coeffs[:Nc] / zscale
            ymin = 1.0e-10

        if res is None:
            for i in range(len(sfh_y)):
                ax.fill_between(
                    sfh_tsteps[i : i + 2],
                    [ymin, ymin],
                    sfh_y[i] * np.ones(1),
                    alpha=0.5,
                )

        else:
            for i in range(len(sfh_y)):
                ax.fill_between(
                    sfh_tsteps[i : i + 2],
                    [ymin, ymin],
                    sfh_y[i] * np.ones(1),
                    alpha=0.3,
                    fc="None",
                    hatch="///",
                )

            hsub = np.zeros(len(res.x), dtype=bool)
            hsub[:Nc] = True

            rnorm = scipy.stats.multivariate_normal(
                mean=res.x[hsub], cov=res.hess_inv[hsub, :][:, hsub]
            )
            rvs = rnorm.rvs(draws)

            if ylog:
                sfh_y_fit = res.x[:Nc] - np.log10(zscale)
                linear_rvs = rvs
                rvs_sfh = linear_rvs - np.log10(zscale)
            else:
                sfh_y_fit = 10 ** res.x[:Nc] / zscale
                linear_rvs = 10**rvs
                rvs_sfh = linear_rvs / zscale

            for i in range(len(sfh_y)):
                ax.fill_between(
                    sfh_tsteps[i : i + 2],
                    [ymin, ymin],
                    sfh_y_fit[i] * np.ones(1),
                    alpha=0.3,
                )

            for i in range(Nc):
                ax.errorbar(
                    np.maximum(
                        (sfh_tsteps[:-1] + np.diff(sfh_tsteps) / 2)[:Nc][i],
                        0.015,
                    ),
                    rvs_sfh.mean(axis=0)[i],
                    rvs_sfh.std(axis=0)[i],
                    marker="o",
                )

        if as_age:
            ax.set_xlim(-0.03 * tage, tage)
            ax.set_xlabel(r"age $t$ [Gyr]")
            tstep = 0.5
            if tage < 1.4:
                tstep = 0.2
            if tage < 0.7:
                tstep = 0.1

            xv = np.arange(0, tage, tstep)
            xt = np.interp(xv, [0, tage], [tage, 0])
            ax.set_xticks(xt)
            ax.set_xticklabels([f"{v:.1f}" for v in xv])

        else:
            ax.semilogx()
            ax.set_xlim(0.01, 14)
            ax.set_xlabel(r"SFH $t$")

        if ylog:
            ax.set_ylabel(r"$\log_{10}$ SFR [ $M_\odot$ / yr ]")
            ax.set_ylim(ymin, sfh_y.max() + 0.5)
        else:
            ax.set_ylabel(r"SFR [ $M_\odot$ / yr ]")
            ax.set_ylim(0.001, 1.15 * sfh_y.max())

        yma = ax.get_ylim()[1]
        if as_age:
            ax.vlines(0, -1e10, 1e10, linestyle=":", color="0.5")
            ax.text(
                0.0,
                yma,
                f"z = {zb:.3f}" + "\n" + f"{tage:.1f} Gyr",
                ha="left",
                va="top",
                fontsize=6,
                bbox={"fc": "w", "ec": "None"},
            )

        else:
            ax.text(
                tage,
                yma,
                f"z = {zb:.3f}" + "\n" + f"{tage:.1f} Gyr",
                ha="right",
                va="top",
                fontsize=6,
                bbox={"fc": "w", "ec": "None"},
            )

            ax.vlines(tage, -1e10, 1e10, linestyle=":", color="0.5")
            tv = [0.01, 0.1, 1]  # , 10]
            tn = ["10 Myr", "100 Myr", "1 Gyr"]  # , '10 Gyr']
            # ins = np.where(np.array(tv) > tage)[0][0]

            ax.set_xlim(ax.get_xlim()[0], tage)

            # tv.append(tage)
            # tn.append('') #f'{tage:.1f}')

            ax.set_xticks(tv)
            ax.set_xticklabels(tn)

        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.grid()

        if fig is not None:
            fig.tight_layout(pad=1)

            utils.figure_timestamp(
                fig,
                text_prefix=os.path.basename(self.file).split(".spec")[0]
                + "\n",
                fontsize=6,
                x=0.97,
                y=0.03,
            )

        return fig
