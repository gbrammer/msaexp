"""
Fits, etc. to extracted spectra
"""

import os
import time
import warnings
import inspect
import yaml

import numpy as np

import scipy.ndimage as nd
from scipy.optimize import nnls

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec

import astropy.io.fits as pyfits

from grizli import utils

utils.set_warnings()


def float_representer(dumper, value):
    """
    Representer function for converting a float value to a YAML scalar
    with six decimal places (``{value:.6f}``).

    Parameters
    ----------
    dumper : yaml.Dumper
        The YAML dumper object.

    value : float
        The float value to be represented.

    Returns
    -------
    yaml.ScalarNode:
        The YAML scalar node representing the float value.

    """
    text = "{0:.6f}".format(value)
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


yaml.add_representer(float, float_representer)

try:
    import eazy.templates

    wave = np.exp(np.arange(np.log(2.4), np.log(4.5), 1.0 / 4000)) * 1.0e4
    _temp = utils.pah33(wave)
    PAH_TEMPLATES = {}
    for t in _temp:
        if "3.47" in t:
            continue

        _tp = _temp[t]
        PAH_TEMPLATES[t] = eazy.templates.Template(
            name=t, arrays=(_tp.wave, _tp.flux)
        )
except:
    print("Failed to initialize PAH_TEMPLATES")
    PAH_TEMPLATES = {}

import astropy.units as u

import eazy.igm

from . import drizzle
from . import utils as msautils

try:
    from .resample_numba import compute_igm

    IGM_FUNC = compute_igm
except ImportError:
    igm = eazy.igm.Inoue14()
    IGM_FUNC = igm.full_IGM

try:
    from .resample_numba import (
        resample_template_numba as RESAMPLE_FUNC,
    )
    from .resample_numba import (
        sample_gaussian_line_numba as SAMPLE_LINE_FUNC,
    )
except ImportError:
    from .resample import resample_template as RESAMPLE_FUNC
    from .resample import sample_gaussian_line as SAMPLE_LINE_FUNC

SCALE_UNCERTAINTY = 1.0

FFTSMOOTH = False

__all__ = [
    "fit_redshift",
    "fit_redshift_grid",
    "plot_spectrum",
    "read_spectrum",
    "calc_uncertainty_scale",
    "resample_bagpipes_model",
    "SpectrumSampler",
]


def resample_bagpipes_model(
    model_galaxy,
    model_comp=None,
    spec_wavs=None,
    R_curve=None,
    nsig=5,
    scale_disp=1.3,
    wave_scale=1,
):
    """

    Parameters
    ----------
    model_galaxy : `bagpipes.models.model_galaxy.model_galaxy`

    model_comp : dict
        Model components dictionary.  If specified, run
        ``model_galaxy.update(model_comp)``, otherwise get from
        ``model_galaxy.model_comp``

    spec_wavs : array-like, None
        Spectrum wavelengths, Angstroms. If not specified, try to use
        ``model_galaxy.spec_wavs``

    R_curve : array-like, None
        Spectral resolution FWHM curve.  If not specified, try to use
        ``model_comp["R_curve"][:,1]``.

    nsig : int
        Number of sigmas for resample.

    wave_scale : float
        Scalar multipled to model wavelengths, i.e., for additional spectral orders

    Returns
    -------
    spectrum : array-like
        Resampled.  If ``model_galaxy.spec_wavs`` is found and has same length
        as ``spec_wavs``, also set ``model_galaxy.spectrum = spectrum`` along with the
        wavelength array.

        The units of ``spectrum`` returned by the function are always microJansky, but
        the units of ``model_galaxy.spectrum`` are set as appropriate given
        ``model_galaxy.spec_units``.

    """
    from .resample_numba import resample_template_numba

    if model_comp is not None:
        model_galaxy.update(model_comp)

    model_comp = model_galaxy.model_comp

    zplusone = model_comp["redshift"] + 1.0

    if "veldisp" in model_comp:
        velocity_sigma = model_comp["veldisp"]
    else:
        velocity_sigma = 0.0

    redshifted_wavs = zplusone * model_galaxy.wavelengths

    if spec_wavs is None:
        spec_wavs = model_galaxy.spec_wavs

    if R_curve is None:
        R_curve = model_comp["R_curve"][:, 1] * scale_disp

    fluxes = resample_template_numba(
        spec_wavs,
        R_curve,
        redshifted_wavs * wave_scale,
        model_galaxy.spectrum_full,
        velocity_sigma=velocity_sigma,
        nsig=nsig,
        fill_value=0.0,
    )

    to_mujy = 10**-29 * 2.9979 * 10**18 / spec_wavs**2

    if model_galaxy.spec_wavs is not None:
        if len(model_galaxy.spec_wavs) == len(spec_wavs):
            if model_galaxy.spec_units == "mujy":
                spectrum = np.c_[spec_wavs, fluxes / to_mujy]
            else:
                spectrum = np.c_[spec_wavs, fluxes]

            model_galaxy.spectrum = spectrum

    return fluxes / to_mujy


class SpectrumSampler(object):

    spec = {}
    spec_wobs = None
    spec_R_fwhm = None
    valid = None
    wave_step = None

    def __init__(
        self,
        spec_input,
        oversample_kwargs=dict(factor=5, pad=12),
        with_sensitivity=True,
        **kwargs,
    ):
        """
        Helper functions for sampling templates onto the wavelength grid
        of an observed spectrum

        Parameters
        ----------
        spec_input : str, `~astropy.io.fits.HDUList`
            - `str` : spectrum filename, usually `[root].spec.fits`
            - `~astropy.io.fits.HDUList` : FITS data

        with_sensitivity : bool
            Initialize sensitivity curves for including multiple spectral orders
            in the model generation.

        Attributes
        ----------

        spec : `~astropy.table.Table`
            1D spectrum table from the `SPEC1D HDU of ``file``

        meta : dict
            Metadata from ``spec.meta``

        spec_wobs : array-like
            Observed wavelengths, microns

        spec_R_fwhm : array-like
            Tabulated spectral resolution `R = lambda / dlambda`, assumed to be
            defined as FWHM

        valid : array-like
            Boolean array of valid 1D data

        """
        self.initialize_spec(spec_input, **kwargs)

        self.NSPEC = len(self.spec)

        self.oversamp_wobs = self.oversampled_wavelengths(**oversample_kwargs)

        self.initialize_emission_line()

        self.sensitivity_file = None
        self.sensitivity = {1: 1.0}
        self.inv_sensitivity = 1.0

        for i in range(1, 4):
            self.sensitivity[i + 1] = None

        if with_sensitivity:
            self.load_sensitivity_curve(**kwargs)

    def __getitem__(self, key):
        """
        Return column of the `spec` table
        """
        return self.spec[key]

    @property
    def meta(self):
        """
        Metadata of `spec` table
        """
        return self.spec.meta

    def initialize_emission_line(self, nsamp=64):
        """
        Initialize emission line

        Parameters
        ----------
        nsamp : int
            Number of samples for the emission line.
            Default = 64
        """
        self.xline = (
            np.linspace(-nsamp, nsamp, 2 * nsamp + 1) / nsamp * 0.1 + 1
        )
        self.yline = self.xline * 0.0
        self.yline[nsamp] = 1
        self.yline /= np.trapz(self.yline, self.xline)

    def initialize_spec(self, spec_input, **kwargs):
        """
        Read spectrum data from file and initialize attributes

        Parameters
        ----------
        spec_input : str
            Filename, usually `[root].spec.fits`

        kwargs : dict
            Keyword arguments passed to `~msaexp.spectrum.read_spectrum`

        """
        self.spec_input = spec_input
        if isinstance(spec_input, str):
            self.file = spec_input
        else:
            self.file = None

        self.spec = read_spectrum(spec_input, **kwargs)

        self.spec_wobs = self.spec["wave"].astype(np.float32)
        self.spec_R_fwhm = self.spec["R"].astype(np.float32)

        if "wave_step" in self.spec.colnames:
            self.wave_step = self.spec["wave_step"].astype(np.float32)

        self.valid = np.isfinite(self.spec["flux"] / self.spec["full_err"])
        if "valid" in self.spec.colnames:
            self.valid &= self.spec["valid"]

    def oversampled_wavelengths(self, factor=5, pad=12):
        """
        Generate a wavelength grid that oversamples the spectrum wavelengths

        Parameters
        ----------
        factor : int
            Oversampling factor

        Returns
        -------
        waves : array-like
            Oversampled wavelengths
        """
        dx = np.gradient(self.spec["wave"])
        xin = np.linspace(0, 1, self.NSPEC)
        xout = np.linspace(0, 1, (self.NSPEC) * factor)
        dxout = np.interp(xout, xin, dx) / factor

        waves = (
            self.spec["wave"][0] - (dx[0] + dxout[0]) / 2 + np.cumsum(dxout)
        )
        if pad > 0:
            waves = np.pad(
                waves,
                pad * factor,
                mode="linear_ramp",
                end_values=(
                    waves[0] - pad * factor * dxout[0],
                    waves[-1] + pad * factor * dxout[-1],
                ),
            )

        return waves

    def resample_eazy_template(
        self,
        template,
        z=0,
        scale_disp=1.0,
        velocity_sigma=100.0,
        fnu=True,
        nsig=4,
        with_igm=False,
        orders=[1, 2, 3, 4],
        verbose=False,
        **kwargs,
    ):
        """
        Smooth and resample an `eazy.templates.Template` object onto the
        observed wavelength grid of a spectrum

        Parameters
        ----------
        template : `eazy.templates.Template`
            Template object

        z : float
            Redshift

        scale_disp : float
            Factor multiplied to the tabulated spectral resolution before
            sampling

        velocity_sigma : float
            Gaussian velocity broadening factor, km/s

        fnu : bool
            Return resampled template in f-nu flux densities

        nsig : int
            Number of standard deviations to sample for the convolution

        orders : list
            List of spectral orders to include if the sensitivity curves have been
            read along with the spectrum.

        Returns
        -------
        res : array-like
            Template flux density smoothed and resampled at the spectrum
            wavelengths

        """

        templ_wobs = template.wave.astype(np.float32) * (1 + z) / 1.0e4
        if fnu:
            templ_flux = template.flux_fnu(z=z).astype(np.float32)
        else:
            templ_flux = template.flux_flam(z=z).astype(np.float32)

        if with_igm:
            igmz = IGM_FUNC(z, templ_wobs * 1.0e4)
        else:
            # Turn off
            igmz = 1.0

        res = np.zeros_like(self.spec_wobs)

        for order in orders:
            if order not in self.sensitivity:
                msg = f"resample_eazy_template: order {order} not found in sensitivity"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                continue
            elif self.sensitivity[order] is None:
                continue

            res_i = RESAMPLE_FUNC(
                self.spec_wobs,
                self.spec_R_fwhm * scale_disp * order,
                templ_wobs * order,
                templ_flux * igmz,
                velocity_sigma=velocity_sigma,
                nsig=nsig,
            )

            if order != 1:
                res += res_i * self.sensitivity[order] * self.inv_sensitivity
            else:
                res += res_i

        return res

    def emission_line(
        self,
        line_um,
        line_flux=1,
        scale_disp=1.0,
        velocity_sigma=100.0,
        nsig=4,
        **kwargs,
    ):
        """
        Make an emission line template - *deprecated in favor of*
        `~msaexp.spectrum.SpectrumSampler.fast_emission_line`

        Parameters
        ----------
        line_um : float
            Line center, microns

        line_flux : float
            Line normalization

        scale_disp : float
            Factor by which to scale the tabulated resolution FWHM curve

        velocity_sigma : float
            Velocity sigma width in km/s

        nsig : int
            Number of sigmas of the convolution kernel to sample

        Returns
        -------
        res : array-like
            Gaussian emission line sampled at the spectrum wavelengths
        """
        res = RESAMPLE_FUNC(
            self.spec_wobs,
            self.spec_R_fwhm * scale_disp,
            self.xline * line_um,
            self.yline,
            velocity_sigma=velocity_sigma,
            nsig=nsig,
        )

        return res * line_flux / line_um

    def fast_emission_line(
        self,
        line_um,
        line_flux=1,
        scale_disp=1.0,
        velocity_sigma=100.0,
        orders=[1, 2, 3, 4],
        lorentz=False,
        verbose=False,
    ):
        """
        Make an emission line template with numerically correct pixel
        integration function

        Parameters
        ----------
        line_um : float
            Line center, microns

        line_flux : float
            Line normalization

        scale_disp : float
            Factor by which to scale the tabulated resolution FWHM curve

        velocity_sigma : float
            Velocity sigma width in km/s

        orders : list
            List of spectral orders to include if the sensitivity curves have been
            read along with the spectrum.

        lorentz : bool
            Generate a Lorentzian function instead of a Gaussian

        Returns
        -------
        res : array-like
            Gaussian emission line sampled at the spectrum wavelengths
        """
        res = np.zeros_like(self.spec_wobs)

        for order in orders:
            if order not in self.sensitivity:
                msg = f"resample_eazy_template: order {order} not found in sensitivity"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                continue
            elif self.sensitivity[order] is None:
                continue

            res_i = SAMPLE_LINE_FUNC(
                self.spec_wobs,
                self.spec_R_fwhm * scale_disp * order,
                line_um * order,
                dx=self.wave_step,
                line_flux=line_flux,
                velocity_sigma=velocity_sigma,
                lorentz=lorentz,
            )

            if order != 1:
                res += res_i * self.sensitivity[order] * self.inv_sensitivity
            else:
                res += res_i

        return res

    def bspline_array(
        self,
        nspline=13,
        remap_arrays=None,
        minmax=None,
        log=False,
        by_wavelength=False,
        get_matrix=True,
        orders=[1, 2, 3, 4],
    ):
        """
        Initialize bspline templates for continuum fits

        Parameters
        ----------
        nspline : int
            Number of spline functions to sample across the wavelength range

        log : bool
            Sample in log(wavelength)

        by_wavelength : bool
            If True, sample bspline functions across the wavelength range.
            If False, sample bspline functions across the index range.

        get_matrix : bool
            If True, return array data. Otherwise, return template objects.

        Returns
        -------
        bspl : array-like
            bspline data, depending on ``get_matrix``

        """
        if by_wavelength:
            bspl = utils.bspline_templates(
                wave=self.spec_wobs * 1.0e4,
                degree=3,
                df=nspline,
                log=log,
                get_matrix=get_matrix,
                minmax=minmax,
            )
        else:
            if remap_arrays is None:
                xvalue = np.arange(len(self.spec_wobs))
            else:
                xvalue = np.interp(
                    self.spec_wobs, *remap_arrays, left=0.0, right=0.0
                )

            bspl = utils.bspline_templates(
                wave=xvalue,
                degree=3,
                df=nspline,
                log=log,
                get_matrix=get_matrix,
                minmax=minmax,
            )

        if 1 not in orders:
            if get_matrix:
                bspl1 = bspl * 1
            else:
                bspl1 = {}
                for t in bspl:
                    bspl1[t] = bspl[t].flux * 1
        else:
            bspl1 = None

        for order in orders:
            if order == 1:
                # Computed above
                continue
            elif order not in self.sensitivity:
                msg = f"resample_eazy_template: order {order} not found in sensitivity"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                continue
            elif self.sensitivity[order] is None:
                continue

            if get_matrix:
                for i in range(nspline):
                    extra = np.interp(
                        self.spec_wobs,
                        self.spec_wobs * order,
                        bspl[:, i],
                        left=0,
                        right=0,
                    )
                    extra *= self.sensitivity[order] * self.inv_sensitivity
                    extra[~np.isfinite(extra)] = 0
                    bspl[:, i] += extra
            else:
                for t in bspl:
                    extra = np.interp(
                        self.spec_wobs,
                        self.spec_wobs * order,
                        bspl[t].flux,
                        left=0,
                        right=0,
                    )
                    extra *= self.sensitivity[order] * self.inv_sensitivity
                    extra[~np.isfinite(extra)] = 0
                    bspl[t].flux += extra

        if 1 not in orders:
            if get_matrix:
                bspl = bspl - bspl1
            else:
                for t in bspl:
                    bspl[t].flux -= bspl1[t].flux

        del bspl1

        if get_matrix:
            bspl = bspl.T
        else:
            for t in bspl:
                bspl[t].wave = self.spec_wobs

        return bspl

    def fit_single_template(
        self,
        template,
        z=0.0,
        spl=None,
        spline_type='multiply',
        lsq=np.linalg.lstsq,
        lsq_kwargs=dict(rcond=None),
        loss=None,
        **kwargs,
    ):
        """
        Resample and fit a template to the spectrum

        Parameters
        ----------
        template : `eazy.templates.Template`
            High-resolution, rest-frame template

        z : float
            Redshift

        spl : (N, M) array
            Optional spline array

        lsq, lsq_kwargs : func, dict
            Least-squares optimization function, keyword args

        loss : func
            Loss function, e.g, `scip.stats.norm(scale=spec['full_err'][spec.valid])`

        kwargs : dict
            keyword arguments passed to
            `~msaexp.spectrum.SpectrumSampler.resample_eazy_template`, e.g.,
            "scale_disp", "velocity_sigma"

        Returns
        -------
        result : dict
            Fit results
            - ``model``: best-fit model, (M,) array
            - ``A``: design matrix, (N, M) array
            - ``coeffs``: least-squares coefficients, (N,) array
            - ``lnp`` : ``loss.logpdf(residual)`` (M,) array
        """
        from scipy.stats import norm

        res = self.resample_eazy_template(template, z=z, **kwargs)

        if spl is not None:
            if spline_type == 'multiply':
                res = res * spl
            else:
                res = np.vstack([res, spl])
        else:
            res = res[None, :]

        A = res / self.spec["full_err"]
        b = self.spec["flux"] / self.spec["full_err"]
        coeffs, _r, rank, s = lsq(
            A[:, self.valid].T, b[self.valid], **lsq_kwargs
        )

        model = res.T.dot(coeffs)
        residual = self.spec["flux"] - model

        if loss is None:
            loss = norm(loc=0, scale=self.spec["full_err"][self.valid])

        result = {
            "model": model,
            "A": res,
            "coeffs": coeffs,
            "lnp": loss.logpdf(residual[self.valid]),
        }

        return result

    def redo_1d_extraction(self, **kwargs):
        """
        Redo 1D extraction from 2D arrays with
        `~msaexp.drizzle.make_optimal_extraction`

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to
            `~msaexp.drizzle.make_optimal_extraction`

        Returns
        -------
        output : `~msaexp.spectrum.SpectrumSampler`
            A new `~msaexp.spectrum.SpectrumSampler` object

        Examples
        --------

        .. plot::
            :include-source:

            # Compare 1D extractions
            from msaexp import spectrum
            import matplotlib.pyplot as plt

            sp = spectrum.SpectrumSampler('https://s3.amazonaws.com/msaexp-nirspec/extractions/ceers-ddt-v3/ceers-ddt-v3_prism-clear_2750_1598.spec.fits')

            fig, axes = plt.subplots(2, 1, figsize=(8, 5),
                                     sharex=True, sharey=True)

            # Boxcar extraction, center pixel +/- 2 pix
            ax = axes[0]
            new = sp.redo_1d_extraction(ap_radius=2, bkg_offset=-6)

            ax.plot(sp['wave'], sp['flux'], alpha=0.5,
                    label='Original optimal extraction')
            ax.plot(new['wave'], new['aper_flux'], alpha=0.5,
                    label='Boxcar, y = 23 ± 2')

            ax.grid()
            ax.legend()

            # Extractions above and below the center
            ax = axes[1]
            low = sp.redo_1d_extraction(ap_center=21, ap_radius=1)
            hi = sp.redo_1d_extraction(ap_center=25, ap_radius=1)

            ax.plot(low['wave'], low['aper_flux']*1.5, alpha=0.5,
                    label='Below, y = 21 ± 1', color='b')
            ax.plot(hi['wave'], hi['aper_flux']*3, alpha=0.5,
                    label='Above, y = 25 ± 1', color='r')

            ax.set_xlim(0.9, 5.3)
            ax.grid()
            ax.legend()

            ax.set_xlabel(r'$\lambda$')
            for ax in axes:
                ax.set_ylabel(r'$\mu\mathrm{Jy}$')

            fig.tight_layout(pad=1)
        """

        if isinstance(self.spec_input, pyfits.HDUList):
            out_hdul = drizzle.extract_from_hdul(self.spec_input, **kwargs)
        else:
            with pyfits.open(self.file) as hdul:
                out_hdul = drizzle.extract_from_hdul(hdul, **kwargs)

        output = SpectrumSampler(out_hdul)

        return output

    def drizzled_hdu_figure(self, **kwargs):
        """
        Run `~msaexp.utils.drizzled_hdu_figure` on array data

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to `~msaexp.utils.drizzled_hdu_figure`

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Spectrum figure

        """
        if isinstance(self.spec_input, pyfits.HDUList):
            fig = msautils.drizzled_hdu_figure(self.spec_input, **kwargs)
        else:
            with pyfits.open(self.file) as hdul:
                fig = msautils.drizzled_hdu_figure(hdul, **kwargs)

        return fig

    def resample_bagpipes_model(
        self,
        model_galaxy,
        model_comp=None,
        nsig=5,
        scale_disp=1.3,
        orders=[1, 2, 3, 4],
    ):
        """
        Resample a `bagpipes` model to the wavelength grid of the spectrum.

        See ``msaexp.spectrum.resample_bagpipes_model``.
        """

        if "scale_disp" in model_comp:
            scale_disp = model_comp["scale_disp"]

        spectrum = np.zeros_like(self.spec_wobs)

        for order in orders:
            if order not in self.sensitivity:
                msg = f"resample_eazy_template: order {order} not found in sensitivity"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                continue
            elif self.sensitivity[order] is None:
                continue

            spectrum_i = resample_bagpipes_model(
                model_galaxy,
                model_comp=model_comp,
                spec_wavs=self.spec_wobs * 1.0e4,
                R_curve=self.spec["R"] * scale_disp * order,
                nsig=nsig,
                wave_scale=order,
            )
            spectrum += (
                spectrum_i * self.sensitivity[order] * self.inv_sensitivity
            )

        return spectrum

    def igm_absorption(self, z, scale_tau=1.0, scale_disp=1.3):
        r"""
        `Inoue+ (2014) <https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.1805I>`_ IGM
        absorption

        Parameters
        ----------
        z : float
            Redshift

        scale_tau : float
            Factor to scale $\tau_\mathrm{IGM}$

        scale_disp : float
            Dispersion R rescaling

        Returns
        -------
        igm : array-like
            IGM model transmission at observed-frame wavelengths
        """
        from .resample_numba import compute_igm, resample_template_numba

        igm_raw = IGM_FUNC(z, self.oversamp_wobs * 1.0e4)

        igm = np.ones(self.NSPEC)

        igm = RESAMPLE_FUNC(
            self.spec_wobs,
            self.spec["R"] * scale_disp,
            self.oversamp_wobs,
            igm_raw,
            velocity_sigma=0.0,
            wave_min=0.0800 * (1 + z),
            wave_max=0.1350 * (1 + z),
            left=0.0,
            right=1.0,
        )

        return igm

    def load_sensitivity_curve(
        self,
        sens_file=None,
        prefix="msaexp_sensitivity",
        version="001",
        file_template="{prefix}_{grating}_{filter}_{version}.fits",
        verbose=False,
        **kwargs,
    ):
        """ """

        if sens_file is None:
            sens_file = file_template.format(
                prefix=prefix,
                filter=self.meta["FILTER"],
                grating=self.meta["GRATING"],
                version=version,
            ).lower()

        # paths to search
        paths = [
            "",
            os.path.join(
                os.path.dirname(__file__), "data/extended_sensitivity"
            ),
        ]

        file_path = None
        for path in paths:
            if os.path.exists(os.path.join(path, sens_file)):
                file_path = path
                break

        if file_path is None:
            return None

        msg = f"load_sensitivity_curve: {sens_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        sens_data = utils.read_catalog(os.path.join(file_path, sens_file))
        self.sensitivity_file = sens_file

        self.sensitivity_correction = 1.0
        self.sensitivity_correction_type = None

        if "nrs1_s200a1" in sens_data.colnames:
            if "APERNAME" in self.meta:
                if "_SLIT" in self.meta["APERNAME"]:
                    self.sensitivity_correction = (
                        1.0 / sens_data["nrs1_s200a1"]
                    )
                    self.sensitivity_correction_type = "nrs1_s200a1"

                if 'S400A1_SLIT' in self.meta['APERNAME']:
                    s400_file = os.path.join(file_path, "sensitivity_ratio_s400a1_s200a1_001.yaml")
                    # print("xxx", s400_file)
                    if os.path.exists(s400_file):
                        with open(s400_file) as fp:
                            s400 = yaml.load(fp, Loader=yaml.Loader)
                            df = len(s400['s400a1_s200a1_coefs'])
                            wpr = msautils.get_standard_wavelength_grid(grating='PRISM', sample=1.0)
                            spx = np.interp(
                                # self.spec['wave'],
                                sens_data["wavelength"],
                                wpr,
                                np.arange(len(wpr))/len(wpr)
                            )
                            bspl = utils.bspline_templates(
                                spx, df=df, minmax=(0, 1), get_matrix=True
                            )

                            self.sensitivity_correction *= bspl.dot(s400['s400a1_s200a1_coefs'])
                            self.sensitivity_correction_type = "nrs1_s200a1_s400a1"

        self.sensitivity[1] = np.interp(
            self.spec["wave"],
            sens_data["wavelength"],
            sens_data["sensitivity"] * self.sensitivity_correction,
            left=0.0,
            right=0.0,
        )

        # Precompute inv_sensitivity = 1 / sensitivity[1] to avoid divide by zero
        self.inv_sensitivity = 1.0 / self.sensitivity[1]
        self.inv_sensitivity[~np.isfinite(self.inv_sensitivity)] = 0.0

        for order in range(2, 5):
            if f"sensitivity_{order}" in sens_data.colnames:
                sens_i = np.interp(
                    self.spec["wave"],
                    sens_data["wavelength"] * order,
                    sens_data[f"sensitivity_{order}"]
                    * self.sensitivity_correction,
                    left=0.0,
                    right=0.0,
                )
                if np.nanmax(sens_i) == 0:
                    self.sensitivity[order] = None
                else:
                    self.sensitivity[order] = sens_i

            else:
                self.sensitivity[order] = None


def smooth_template_disp_eazy(
    templ, wobs_um, disp, z, velocity_fwhm=80, scale_disp=1.3, flambda=True
):
    """
    Smooth a template with a wavelength-dependent dispersion function.

    *NB:* Not identical to the preferred
    `~msaexp.spectrum.SpectrumSampler.resample_eazy_template`

    Parameters
    ----------
    templ : `eazy.template.Template`
        Template object

    wobs_um : array-like
        Target observed-frame wavelengths, microns

    disp : table
        NIRSpec dispersion table with columns ``WAVELENGTH``, ``R``

    z : float
        Target redshift

    velocity_fwhm : float
        Velocity dispersion FWHM, km/s

    scale_disp : float
        Scale factor applied to ``disp['R']``

    flambda : bool
        Return smoothed template in units of f_lambda or f_nu.

    Returns
    -------
    tsmooth : array-like
        Template convolved with spectral resolution + velocity dispersion.
        Same length as `wobs_um`

    """
    dv = np.sqrt(velocity_fwhm**2 + (3.0e5 / disp["R"] / scale_disp) ** 2)
    disp_ang = disp["WAVELENGTH"] * 1.0e4
    dlam_ang = disp_ang * dv / 3.0e5 / 2.35

    def _lsf(wave):
        return np.interp(
            wave,
            disp_ang,
            dlam_ang,
            left=dlam_ang[0],
            right=dlam_ang[-1],
        )

    if hasattr(wobs_um, "value"):
        wobs_ang = wobs_um.value * 1.0e4
    else:
        wobs_ang = wobs_um * 1.0e4

    flux_model = templ.to_observed_frame(
        z=z,
        lsf_func=_lsf,
        clip_wavelengths=None,
        wavelengths=wobs_ang,
        smoothspec_kwargs={"fftsmooth": FFTSMOOTH},
    )

    if flambda:
        flux_model = np.squeeze(flux_model.flux_flam())
    else:
        flux_model = np.squeeze(flux_model.flux_fnu())

    return flux_model


def smooth_template_disp_sedpy(
    templ,
    wobs_um,
    disp,
    z,
    velocity_fwhm=80,
    scale_disp=1.3,
    flambda=True,
    with_igm=True,
):
    """
    Smooth a template with a wavelength-dependent dispersion function using
    the `sedpy`/`prospector` LSF smoothing function

    Parameters
    ----------
    templ : `eazy.template.Template`
        Template object

    wobs_um : array-like
        Target observed-frame wavelengths, microns

    disp : table
        NIRSpec dispersion table with columns ``WAVELENGTH``, ``R``

    z : float
        Target redshift

    velocity_fwhm : float
        Velocity dispersion FWHM, km/s

    scale_disp : float
        Scale factor applied to ``disp['R']``

    flambda : bool
        Return smoothed template in units of f_lambda or f_nu.

    with_igm : bool
        Apply the intergalactic medium (IGM) absorption to smoothed template

    Returns
    -------
    tsmooth : array-like
        Template convolved with spectral resolution + velocity dispersion.
        Same length as `wobs_um`
    """
    from sedpy.smoothing import smoothspec

    wobs = templ.wave * (1 + z)
    trim = wobs > wobs_um[0] * 1.0e4 * 0.95
    trim &= wobs < wobs_um[-1] * 1.0e4 * 1.05

    if flambda:
        fobs = templ.flux_flam(z=z)  # [wclip]
    else:
        fobs = templ.flux_fnu(z=z)  # [wclip]

    if with_igm:
        fobs *= templ.igm_absorption(z)

    wobs = wobs[trim]
    fobs = fobs[trim]

    R = (
        np.interp(
            wobs,
            disp["WAVELENGTH"] * 1.0e4,
            disp["R"],
            left=disp["R"][0],
            right=disp["R"][-1],
        )
        * scale_disp
    )

    dv = np.sqrt(velocity_fwhm**2 + (3.0e5 / R) ** 2)
    dlam_ang = wobs * dv / 3.0e5 / 2.35

    def _lsf(wave):
        return np.interp(wave, wobs, dlam_ang)

    tsmooth = smoothspec(
        wobs,
        fobs,
        smoothtype="lsf",
        lsf=_lsf,
        outwave=wobs_um * 1.0e4,
        fftsmooth=FFTSMOOTH,
    )

    return tsmooth


def smooth_template_disp(
    templ,
    wobs_um,
    disp,
    z,
    velocity_fwhm=80,
    scale_disp=1.3,
    flambda=True,
    with_igm=True,
):
    """
    Smooth a template with a wavelength-dependent dispersion function

    Parameters
    ----------
    templ : `eazy.template.Template`
        Template object

    wobs_um : array-like
        Target observed-frame wavelengths, microns

    disp : table
        NIRSpec dispersion table with columns ``WAVELENGTH``, ``R``

    z : float
        Target redshift

    velocity_fwhm : float
        Velocity dispersion FWHM, km/s

    scale_disp : float
        Scale factor applied to ``disp['R']``

    flambda : bool
        Return smoothed template in units of f_lambda or f_nu.

    with_igm : bool
        Apply the intergalactic medium (IGM) absorption to the  template

    Returns
    -------
    tsmooth : array-like
        Template convolved with spectral resolution + velocity dispersion.
        Same length as `wobs_um`
    """

    wobs = templ.wave * (1 + z) / 1.0e4
    if flambda:
        fobs = templ.flux_flam(z=z)  # [wclip]
    else:
        fobs = templ.flux_fnu(z=z)  # [wclip]

    if with_igm:
        fobs *= templ.igm_absorption(z)

    disp_r = np.interp(wobs, disp["WAVELENGTH"], disp["R"]) * scale_disp
    fwhm_um = np.sqrt(
        (wobs / disp_r) ** 2 + (velocity_fwhm / 3.0e5 * wobs) ** 2
    )
    sig_um = np.maximum(fwhm_um / 2.35, 0.5 * np.gradient(wobs))

    x = wobs_um[:, np.newaxis] - wobs[np.newaxis, :]
    gaussian_kernel = (
        1.0 / np.sqrt(2 * np.pi * sig_um**2) * np.exp(-(x**2) / 2 / sig_um**2)
    )
    tsmooth = np.trapz(gaussian_kernel * fobs, x=wobs, axis=1)

    return tsmooth


def fit_redshift(
    file="jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits",
    z0=[0.2, 10],
    zstep=None,
    eazy_templates=None,
    nspline=None,
    scale_disp=1.3,
    vel_width=100,
    Rline=None,
    is_prism=False,
    use_full_dispersion=False,
    ranges=None,
    sys_err=0.02,
    **kwargs,
):
    """
    Fit spectrum for the redshift

    Parameters
    ----------
    file : str
        Spectrum filename

    z0 : (float, float)
        Redshift range

    zstep : (float, float)
        Step sizes in `dz/(1+z)`

    eazy_templates : list, None
        List of `eazy.templates.Template` objects.  If not provided, just use
        dummy spline continuum and emission line templates

    nspline : int
        Number of splines to use for dummy continuum

    scale_disp : float
        Scale factor of nominal dispersion files, i.e., `scale_disp > 1`
        *increases* the spectral resolution

    vel_width : float
        Velocity width the emission line templates

    Rline : float
        Original spectral resolution used to sample the line templates

    is_prism : bool
        Is the spectrum from the prism?

    use_full_dispersion : bool
        Convolve `eazy_templates` with the full wavelength-dependent
        dispersion function

    ranges : list of tuples
        Wavelength ranges for the subplots

    sys_err : float
        Systematic uncertainty added in quadrature with nominal uncertainties

    Returns
    -------
    fig : Figure
        Diagnostic figure

    sp : `~astropy.table.Table`
        A copy of the 1D spectrum as fit with additional columns describing the
        best-fit templates

    data : dict
        Fit metadata

    """
    frame = inspect.currentframe()

    if "spec.fits" in file:
        froot = file.split(".spec.fits")[0]
    else:
        froot = file.split(".fits")[0]

    # Log function arguments
    utils.LOGFILE = f"{froot}.zfit.log"
    if os.path.exists(utils.LOGFILE):
        os.remove(utils.LOGFILE)

    # Log arguments
    args = utils.log_function_arguments(
        utils.LOGFILE,
        frame,
        "spectrum.fit_redshift",
        ignore=["eazy_templates"],
    )
    if isinstance(args, dict):
        with open(f"{froot}.zfit.call.yml", "w") as fp:
            fp.write(f"# {time.ctime()}\n# {os.getcwd()}\n")
            if eazy_templates is not None:
                for i, t in enumerate(eazy_templates):
                    msg = f'# eazy_templates[{i}] = "{t.__str__()}"'
                    utils.log_comment(utils.LOGFILE, msg, verbose=False)
                    fp.write(msg + "\n")

            yaml.dump(args, stream=fp, Dumper=yaml.Dumper)

    # is_prism |= ('clear' in file)
    spec = read_spectrum(file, sys_err=sys_err, **kwargs)
    is_prism |= spec.grating in ["prism"]

    if zstep is None:
        if is_prism:
            step0 = 0.002
            step1 = 0.0001
        else:
            step0 = 0.001
            step1 = 0.00002
    else:
        step0, step1 = zstep

    if Rline is None:
        if is_prism:
            Rline = 1000
        else:
            Rline = 5000

    # First pass
    zgrid = utils.log_zgrid(z0, step0)
    zg0, chi0 = fit_redshift_grid(
        file,
        zgrid=zgrid,
        line_complexes=False,
        vel_width=vel_width,
        scale_disp=scale_disp,
        eazy_templates=eazy_templates,
        Rline=Rline,
        use_full_dispersion=use_full_dispersion,
        sys_err=sys_err,
        **kwargs,
    )

    zbest0 = zg0[np.argmin(chi0)]

    # Second pass
    zgrid = utils.log_zgrid(
        zbest0 + np.array([-0.005, 0.005]) * (1 + zbest0), step1
    )

    zg1, chi1 = fit_redshift_grid(
        file,
        zgrid=zgrid,
        line_complexes=False,
        vel_width=vel_width,
        scale_disp=scale_disp,
        eazy_templates=eazy_templates,
        Rline=Rline,
        use_full_dispersion=use_full_dispersion,
        sys_err=sys_err,
        **kwargs,
    )

    zbest = zg1[np.argmin(chi1)]

    fz, az = plt.subplots(1, 1, figsize=(6, 4))
    az.plot(zg0, chi0)
    az.plot(zg1, chi1)

    az.set_ylim(chi1.min() - 50, chi1.min() + 10**2)
    az.grid()
    az.set_xlabel("redshift")
    az.set_ylabel(r"$\chi^2$")
    az.set_title(os.path.basename(file))

    fz.tight_layout(pad=1)

    fz.savefig(froot + ".chi2.png")

    if is_prism:
        if ranges is None:
            ranges = [(3427, 5308), (6250, 9700)]
        if nspline is None:
            nspline = 41
    else:
        if ranges is None:
            ranges = [(3680, 4400), (4861 - 50, 5008 + 50), (6490, 6760)]
        if nspline is None:
            nspline = 23

    fig, sp, data = plot_spectrum(
        file,
        z=zbest,
        show_cont=True,
        draws=100,
        nspline=nspline,
        figsize=(16, 8),
        vel_width=vel_width,
        ranges=ranges,
        Rline=Rline,
        scale_disp=scale_disp,
        eazy_templates=eazy_templates,
        use_full_dispersion=use_full_dispersion,
        sys_err=sys_err,
        **kwargs,
    )

    if eazy_templates is not None:
        spl_fig, sp2, spl_data = plot_spectrum(
            file,
            z=zbest,
            show_cont=True,
            draws=100,
            nspline=nspline,
            figsize=(16, 8),
            vel_width=vel_width,
            ranges=ranges,
            Rline=Rline,
            scale_disp=scale_disp,
            eazy_templates=None,
            use_full_dispersion=use_full_dispersion,
            sys_err=sys_err,
            **kwargs,
        )

        for k in [
            "coeffs",
            "covar",
            "model",
            "mline",
            "fullchi2",
            "contchi2",
            "eqwidth",
        ]:
            if k in spl_data:
                data[f"spl_{k}"] = spl_data[k]

        spl_fig.savefig(froot + ".spl.png")

        sp["spl_model"] = sp2["model"]

    sp["wave"].unit = u.micron
    sp["flux"].unit = u.microJansky

    sp.write(froot + ".spec.zfit.fits", overwrite=True)

    zdata = {}
    zdata["zg0"] = zg0.tolist()
    zdata["chi0"] = chi0.tolist()
    zdata["zg1"] = zg1.tolist()
    zdata["chi1"] = chi1.tolist()

    data["dchi2"] = float(np.nanmedian(chi0) - np.nanmin(chi0))

    for k in ["templates", "spl_covar", "covar"]:
        if k in data:
            _ = data.pop(k)

    with open(froot + ".zfit.yaml", "w") as fp:
        yaml.dump(zdata, stream=fp)

    with open(froot + ".yaml", "w") as fp:
        yaml.dump(data, stream=fp)

    fig.savefig(froot + ".zfit.png")

    return fig, sp, data


H_RECOMBINATION_LINES = [
    "Ha+NII",
    "Ha",
    "Hb",
    "Hg",
    "Hd",
    "PaA",
    "PaB",
    "PaG",
    "PaD",
    "Pa8",
    "BrA",
    "BrB",
    "BrG",
    "BrD",
]

EXTRA_NIR_LINES = [
    "FeII-11128",
    "PII-11886",
    "FeII-12570",
    "FeII-16440",
    "FeII-16877",
    "FeII-17418",
    "FeII-18362",
]


def make_templates(
    sampler,
    z,
    bspl={},
    eazy_templates=None,
    vel_width=100,
    broad_width=4000,
    broad_lines=[],
    scale_disp=1.3,
    grating="prism",
    halpha_prism=["Ha+NII"],
    oiii=["OIII"],
    o4363=[],
    sii=["SII"],
    with_pah=True,
    extra_lines=EXTRA_NIR_LINES,
    apply_igm=True,
    **kwargs,
):
    """
    Generate fitting templates

    sampler : `~msaexp.spectrum.SpectrumSampler`
        Spectrum object with wavelength arrays and metadata

    z : float
        Redshift

    bspl : dict
        Spline templates for dummy continuum

    eazy_templates : list
        Optional list of `eazy.templates.Template` template objects to use in
        place of the spline + line templates

    vel_width : float
        Velocity width of the individual emission line templates, km/s

    broad_width : float
        Velocity width of the broad emission line templates, km/s

    broad_lines : list
        List of line template names to be modeled as broad lines

    scale_disp : float
        Scaling factor for the tabulated dispersion

    grating : str
        Grating used for the observation

    halpha_prism : ['Ha+NII'], ['Ha','NII']
        Line template names to use for Halpha and [NII], i.e., ``['Ha+NII']``
        fits with a fixed line ratio and `['Ha','NII']` fits them separately
        but with a fixed line ratio 6548:6584 = 1:3

    oiii : ['OIII'], ['OIII-4959','OIII-5007']
        Similar for [OIII]4959+5007, ``['OIII']`` fits as a doublet with fixed
        ratio 4959:5007 = 1:2.98 and ``['OIII-4949', 'OIII-5007']`` fits them
        independently.

    o4363 : [] or ['OIII-4363']
        Include [OIII]4363 as a separate template

    sii : ['SII'], ['SII-6717','SII-6731']
        How to fit the [SII] doublet

    with_pah : bool
        Whether to include PAH templates in the fitting

    extra_lines : list
        List of extra emission lines to fit

    Returns
    -------
    templates : list
        List of the computed template objects

    tline : array
        Boolean list of which templates are line components

    _A : (NT, NWAVE) array
        Design matrix of templates interpolated at ``wobs``

    """
    from grizli import utils

    wobs = sampler.spec_wobs

    wmask = sampler.valid

    wmin = wobs[wmask].min()
    wmax = wobs[wmask].max()

    templates = []
    tline = []

    if eazy_templates is None:
        lw, lr = utils.get_line_wavelengths()

        _A = [bspl * 1]
        for i in range(bspl.shape[0]):
            templates.append(f"spl {i}")
            tline.append(False)

        # templates = {}
        # for k in bspl:
        #    templates[k] = bspl[k]

        # templates = {}
        if grating in ["prism"]:
            hlines = ["Hb", "Hg", "Hd"]

            if z > 4:
                oiii = ["OIII-4959", "OIII-5007"]
                hene = ["HeII-4687", "NeIII-3867", "HeI-3889"]
                o4363 = ["OIII-4363"]

            else:
                # oiii = ['OIII']
                hene = ["HeI-3889"]
                # o4363 = []

            # sii = ['SII']
            # sii = ['SII-6717', 'SII-6731']

            hlines += halpha_prism + ["NeIII-3968"]
            fuv = ["OIII-1663"]
            oii_7320 = ["OII-7325"]
            extra = []

        else:
            hlines = ["Hb", "Hg", "Hd", "H8", "H9", "H10", "H11", "H12"]

            hene = ["HeII-4687", "NeIII-3867"]
            o4363 = ["OIII-4363"]
            oiii = ["OIII-4959", "OIII-5007"]
            sii = ["SII-6717", "SII-6731"]
            hlines += ["Ha", "NII-6549", "NII-6584"]
            hlines += ["H7", "NeIII-3968"]
            fuv = ["OIII-1663", "HeII-1640", "CIV-1549"]
            oii_7320 = ["OII-7323", "OII-7332"]

            extra = ["HeI-6680", "SIII-6314"]

        line_names = []
        line_waves = []

        for li in [
            *hlines,
            *oiii,
            *o4363,
            "OII",
            *hene,
            *sii,
            *oii_7320,
            "ArIII-7138",
            "ArIII-7753",
            "SIII-9068",
            "SIII-9531",
            "OI-6302",
            "PaD",
            "PaG",
            "PaB",
            "PaA",
            "HeI-1083",
            "CI-9850",
            # "SiVI-19634",
            "BrA",
            "BrB",
            "BrG",
            "BrD",
            "BrE",
            "BrF",
            "PfB",
            "PfG",
            "PfD",
            "PfE",
            "Pa8",
            "Pa9",
            "Pa10",
            "HeI-5877",
            *fuv,
            "CIII-1906",
            "NIII-1750",
            "Lya",
            "MgII",
            "NeV-3346",
            "NeVI-3426",
            "HeI-7065",
            "HeI-8446",
            *extra,
            *extra_lines,
        ]:

            if li not in lw:
                continue

            lwi = lw[li][0] * (1 + z)

            if lwi < wmin * 1.0e4:
                continue

            if lwi > wmax * 1.0e4:
                continue

            line_names.append(li)
            line_waves.append(lwi)

        so = np.argsort(line_waves)
        line_waves = np.array(line_waves)[so]

        for iline in so:
            li = line_names[iline]
            lwi = lw[li][0] * (1 + z)

            if lwi < wmin * 1.0e4:
                continue

            if lwi > wmax * 1.0e4:
                continue

            # print(l, lwi, disp_r)

            name = f"line {li}"

            for i, (lwi0, lri) in enumerate(zip(lw[li], lr[li])):
                lwi = lwi0 * (1 + z) / 1.0e4
                if li in broad_lines:
                    vel_i = broad_width
                else:
                    vel_i = vel_width

                line_i = sampler.fast_emission_line(
                    lwi,
                    line_flux=lri / np.sum(lr[li]),
                    scale_disp=scale_disp,
                    velocity_sigma=vel_i,
                )
                if i == 0:
                    line_0 = line_i
                else:
                    line_0 += line_i

            _A.append(line_0 / 1.0e4)
            templates.append(name)
            tline.append(True)

        if with_pah:
            xpah = 3.3 * (1 + z)
            if ((xpah > wmin) & (xpah < wmax)) | (0):
                for t in PAH_TEMPLATES:
                    tp = PAH_TEMPLATES[t]
                    tflam = sampler.resample_eazy_template(
                        tp,
                        z=z,
                        velocity_sigma=vel_width,
                        scale_disp=scale_disp,
                        fnu=False,
                    )

                    _A.append(tflam)

                    templates.append(t)
                    tline.append(True)

        _A = np.vstack(_A)

        if apply_igm:
            igmz = IGM_FUNC(z, wobs.value * 1.0e4)
            _A *= np.maximum(igmz, 0.001)

    else:
        if isinstance(eazy_templates[0], dict) & (len(eazy_templates) == 2):
            # lw, lr dicts
            lw, lr = eazy_templates

            _A = [bspl * 1]
            for i in range(bspl.shape[0]):
                templates.append(f"spl {i}")
                tline.append(False)

            for li in lw:
                name = f"line {li}"

                line_0 = None

                for i, (lwi0, lri) in enumerate(zip(lw[li], lr[li])):
                    lwi = lwi0 * (1 + z) / 1.0e4

                    if lwi < wmin:
                        continue

                    elif lwi > wmax:
                        continue

                    if li in broad_lines:
                        vel_i = broad_width
                    else:
                        vel_i = vel_width

                    line_i = sampler.fast_emission_line(
                        lwi,
                        line_flux=lri / np.sum(lr[li]),
                        scale_disp=scale_disp,
                        velocity_sigma=vel_i,
                    )
                    if line_0 is None:
                        line_0 = line_i
                    else:
                        line_0 += line_i

                if line_0 is not None:
                    _A.append(line_0 / 1.0e4)
                    templates.append(name)
                    tline.append(True)

            _A = np.vstack(_A)

            if apply_igm:
                igmz = IGM_FUNC(z, wobs.value * 1.0e4)
                _A *= np.maximum(igmz, 0.001)

        elif len(eazy_templates) == 1:
            # Scale single template by spline
            t = eazy_templates[0]

            for i in range(bspl.shape[0]):
                templates.append(f"{t.name} spl {i}")
                tline.append(False)

            tflam = sampler.resample_eazy_template(
                t,
                z=z,
                velocity_sigma=vel_width,
                scale_disp=scale_disp,
                fnu=False,
            )

            _A = np.vstack([bspl * tflam])

            if apply_igm:
                igmz = IGM_FUNC(z, wobs.value * 1.0e4)
                _A *= np.maximum(igmz, 0.001)

        else:
            templates = []
            tline = []

            _A = []
            for i, t in enumerate(eazy_templates):
                tflam = sampler.resample_eazy_template(
                    t,
                    z=z,
                    velocity_sigma=vel_width,
                    scale_disp=scale_disp,
                    fnu=False,
                )

                _A.append(tflam)

                templates.append(t.name)
                tline.append(False)

            _A = np.vstack(_A)
            if apply_igm:
                igmz = IGM_FUNC(z, wobs.value * 1.0e4)
                _A *= np.maximum(igmz, 0.001)

    return templates, np.array(tline), _A


def fit_redshift_grid(
    file="jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits",
    zgrid=None,
    vel_width=100,
    scale_disp=1.3,
    nspline=27,
    eazy_templates=None,
    use_full_dispersion=True,
    use_aper_columns=False,
    **kwargs,
):
    """
    Fit redshifts on a grid

    Parameters
    ----------
    zgrid : array
        Redshifts to fit

    others : see `~msaexp.spectrum.fit_redshift`

    Returns
    -------
    zgrid : array
        Copy of `zgrid`

    chi2 : array
        Chi-squared of the template fits at redshifts from `zgrid`

    """
    from tqdm import tqdm

    sampler = SpectrumSampler(file, **kwargs)
    spec = sampler.spec

    if (use_aper_columns > 0) & ("aper_flux" in spec.colnames):
        if ("aper_corr" in spec.colnames) & (use_aper_columns > 1):
            ap_corr = spec["aper_corr"] * 1
        else:
            ap_corr = 1

        flam = spec["aper_flux"] * spec["to_flam"] * ap_corr
        eflam = spec["aper_full_err"] * spec["to_flam"] * ap_corr
    else:
        flam = spec["flux"] * spec["to_flam"]
        eflam = spec["full_err"] * spec["to_flam"]

    mask = spec["valid"]

    flam[~mask] = np.nan
    eflam[~mask] = np.nan

    bspl = sampler.bspline_array(nspline=nspline, get_matrix=True)

    chi2 = zgrid * 0.0

    for iz, z in tqdm(enumerate(zgrid)):

        templates, tline, _A = make_templates(
            sampler,
            z,
            bspl=bspl,
            eazy_templates=eazy_templates,
            vel_width=vel_width,
            scale_disp=scale_disp,
            use_full_dispersion=use_full_dispersion,
            grating=spec.grating,
            **kwargs,
        )

        okt = _A[:, mask].sum(axis=1) != 0

        _Ax = _A[okt, :] / eflam
        _yx = flam / eflam

        try:
            if eazy_templates is None:
                _x = np.linalg.lstsq(_Ax[:, mask].T, _yx[mask], rcond=None)
            else:
                _x = nnls(_Ax[:, mask].T, _yx[mask])
        except RuntimeError:
            x = [np.zeros(okt.sum())]

        coeffs = np.zeros(_A.shape[0])
        coeffs[okt] = _x[0]
        _model = _A.T.dot(coeffs)

        chi = (flam - _model) / eflam

        chi2_i = (chi[mask] ** 2).sum()
        # print(z, chi2_i)
        chi2[iz] = chi2_i

    return zgrid, chi2


def calc_uncertainty_scale(
    file=None,
    data=None,
    order=0,
    initial_mask=(0.2, 5),
    sys_err=0.02,
    fit_sys_err=False,
    method="bfgs",
    student_df=10,
    update=True,
    verbose=True,
    **kwargs,
):
    """
    Compute a polynomial scaling of the spectrum uncertainties. The procedure
    is to fit for coefficients of a polynomial multiplied to the `err` array
    of the spectrum such that `(flux - model)/(err*scl)` residuals are `N(0,1)`

    Parameters
    ----------
    file : str
        Spectrum filename

    data : tuple
        Precomputed outputs from `~msaexp.spectrum.plot_spectrum`

    order : int
        Degree of the correction polynomial

    initial_mask : (float, float)
        Masking for the fit initialization. First parameter is zeroth-order
        uncertainty scaling and the second parameter is the mask threshold of
        the residuals

    sys_err : float
        Systematic component of the uncertainties

    fit_sys_err : bool
        Fit for adjusted ``sys_err`` parameter

    method : str
        Optimization method for `scipy.optimize.minimize`

    student_df : int, None
        If specified, calculate log likelihood of a `scipy.stats.t Student-t
        distribution with ``df=student_df``. Otherwise, calculate
        log-likelihood of the `scipy.stats.normal` distribution.

    update : bool
        Update the global `msaexp.spectrum.SCALE_UNCERTAINTY` array with the
        fit result

    verbose : bool
        Print status messages. If ``verbose > 1`` will also print status at
        each step of the optimization.

    kwargs : dict
        Keyword arguments for `~msaexp.spectrum.plot_spectrum` if `data` not
        specified

    Returns
    -------
    spec : `~astropy.table.Table`
        The spectrum as fit

    escale : array
        The wavelength-dependent scaling of the uncertainties

    sys_err : float
        The systematic uncertainty used, fixed or adjusted depending on
        ``fit_sys_err``

    res : object
        Output from `scipy.optimize.minimize`

    """
    from scipy.stats import norm
    from scipy.stats import t as student

    from scipy.optimize import minimize

    global SCALE_UNCERTAINTY
    SCALE_UNCERTAINTY = 1.0

    if data is None:
        spec, spl = plot_spectrum(
            inp=file,
            eazy_templates=None,
            get_spl_templates=True,
            sys_err=sys_err,
            get_init_data=True,
            **kwargs,
        )
    else:
        spec, spl = data

    _err = spec["err"]

    ok = (_err > 0) & (spec["flux"] != 0)
    ok &= np.isfinite(_err + spec["flux"])

    def objfun_scale_uncertainties(c, ok, ret):
        """
        Objective function for scaling uncertainties
        (Nested method in `calc_uncertainty_scale`).

        Parameters
        ----------
        c : list
            List of coefficients for scaling uncertainties.

        ok : array
            Boolean array indicating which elements to include.

        ret : int
            Return type indicator. If 1, return sys_err, full_err, and model.

        Returns
        ----------
        chi2 : float
            The calculated chi-squared value.

        """

        if fit_sys_err:
            _sys_err = c[0] / 100
            _coeffs = c[1:]
        else:
            _coeffs = c[:]
            _sys_err = sys_err

        err = 10 ** np.polyval(_coeffs, spec["wave"] - 3) * _err
        if "escale" in spec.colnames:
            err *= spec["escale"]

        _full_err = np.sqrt(
            err**2 + np.maximum(_sys_err * spec["flux"], 0) ** 2
        )

        _Ax = spl / _full_err
        _yx = spec["flux"] / _full_err
        _x = np.linalg.lstsq(_Ax[:, ok].T, _yx[ok], rcond=None)
        _model = spl.T.dot(_x[0])

        if ret == 1:
            return _sys_err, _full_err, _model

        if student_df is None:
            _lnp = norm.logpdf(
                (spec["flux"] - _model)[ok],
                loc=_model[ok] * 0.0,
                scale=_full_err[ok],
            ).sum()
            chi2 = -1 * _lnp / 2
        else:
            _lnp = student.logpdf(
                (spec["flux"] - _model)[ok],
                student_df,
                loc=_model[ok] * 0.0,
                scale=_full_err[ok],
            ).sum()
            chi2 = -1 * _lnp / 2

        if 0:
            _resid = (spec["flux"] - _model)[ok] / _full_err[ok]
            chi2 = np.log(utils.nmad(_resid)) ** 2

        cstr = " ".join([f"{ci:6.2f}" for ci in c])
        msg = f"{cstr}: {chi2:.6e}"
        utils.log_comment(utils.LOGFILE, msg, verbose=(verbose > 1))

        return chi2

    if fit_sys_err:
        c0 = np.zeros(order + 2)
        c0[0] = sys_err * 100
    else:
        c0 = np.zeros(order + 1)

    if initial_mask is not None:
        c0[-1] = initial_mask[0]
        _sys, _efit, _model = objfun_scale_uncertainties(c0, ok, 1)
        bad = np.abs((spec["flux"] - _model) / _efit) > initial_mask[1]

        msg = "calc_uncertainty_scale: "
        msg += f"Mask additional {(bad & ok).sum()} pixels"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        ok &= ~bad

    res = minimize(objfun_scale_uncertainties, c0, method=method, args=(ok, 0))
    _sys, _efit, _model = objfun_scale_uncertainties(res.x, ok, 1)
    spec.meta["calc_syse"] = _sys
    spec["calc_err"] = _efit
    spec["calc_model"] = _model
    spec["calc_valid"] = ok

    if fit_sys_err:
        sys_err, _coeffs = res.x[0] / 100.0, res.x[1:]
    else:
        _coeffs = res.x[:]

    _resid = (spec["flux"] - _model) / _efit
    msg = f"calc_uncertainty_scale: sys_err = {sys_err:.4f}"
    msg += f"\ncalc_uncertainty_scale: coeffs = {_coeffs}"
    msg += f"\ncalc_uncertainty_scale: NMAD = {utils.nmad(_resid[ok]):.3f}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    if update:

        msg = f"calc_uncertainty_scale: Set SCALE_UNCERTAINTY: {_coeffs}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        SCALE_UNCERTAINTY = _coeffs

    return spec, 10 ** np.polyval(_coeffs, spec["wave"] - 3), sys_err, res


def setup_spectrum(file, **kwargs):
    """
    Deprecated, use `~msaexp.spectrum.read_spectrum`
    """
    return read_spectrum(file, **kwargs)


def set_spec_sys_err(spec, sys_err=0.02):
    """
    Set `full_err` columns including a systematic component

        >>> full_err**2 = (err * escale)**2 + (sys_err * maximum(flux,0))**2

    """
    spec["full_err"] = np.sqrt(
        (spec["err"] * spec["escale"]) ** 2
        + np.maximum(sys_err * spec["flux"], 0) ** 2
    )

    if "aper_err" in spec.colnames:
        spec["aper_full_err"] = np.sqrt(
            (spec["aper_err"] * spec["escale"]) ** 2
            + np.maximum(sys_err * spec["aper_flux"], 0) ** 2
        )

    spec.meta["sys_err"] = sys_err


def read_spectrum(
    inp,
    spectrum_extension="SPEC1D",
    sys_err=0.02,
    err_mask=(5, 0.5),
    err_median_filter=[11, 0.2],
    **kwargs,
):
    """
    Read a spectrum and apply flux and/or uncertainty scaling

    Flux scaling `corr` is applied if there are `POLY[i]` keywords in the
    spectrum metadata, with

    .. code-block:: python
        :dedent:

        >>> coeffs = [header[f'POLY{i}'] for i in range(order+1)]
        >>> corr = np.polyval(coeffs, np.log(spec['wave']*1.e4))

    Parameters
    ----------
    inp : str or `~astropy.io.fits.HDUList`
        Fits filename of a file that includes a `~astropy.io.fits.BinTableHDU`
        table of an extracted spectrum. Alternatively, can be an
        `~astropy.io.fits.HDUList` itself

    spectrum_extension : str
        Extension name of 1D spectrum in file or HDUList input

    sys_err : float
        Systematic uncertainty added in quadrature with `err` array

    err_mask : float, float or None
        Mask pixels where ``err < np.percentile(err[err > 0],
        err_mask[0])*err_mask[1]``

    err_median_filter : int, float or None
        Mask pixels where ``err < nd.median_filter(err,
        err_median_filter[0])*err_median_filter[1]``

    Returns
    -------
    spec : `~astropy.table.Table`
        Spectrum table.  Existing columns in `file` should be

            - ``wave`` : observed-frame wavelength, microns
            - ``flux`` : flux density, `~astropy.units.microJansky`
            - ``err`` : Uncertainty on ```flux```

        Columns calculated here are

            - ``corr`` : flux scaling
            - ``escale`` : extra scaling of uncertainties
            - ``full_err`` : Full uncertainty including `sys_err`
            - ``R`` : spectral resolution
            - ``valid`` : Data are valid

    """
    global SCALE_UNCERTAINTY

    import scipy.ndimage as nd

    if isinstance(inp, str):
        if "fits" in inp:
            with pyfits.open(inp) as hdul:
                if spectrum_extension in hdul:
                    spec = utils.read_catalog(hdul[spectrum_extension])
                else:
                    spec = utils.read_catalog(inp)
        else:
            spec = utils.read_catalog(inp)

    elif isinstance(inp, pyfits.HDUList):
        if spectrum_extension in inp:
            spec = utils.read_catalog(inp[spectrum_extension])
        else:
            msg = f"{spectrum_extension} extension not found in HDUList input"
            raise ValueError(msg)

    elif hasattr(inp, "colnames"):
        spec = utils.GTable()
        for c in inp.colnames:
            spec[c] = inp[c]

        for k in inp.meta:
            spec.meta[k] = inp.meta[k]

    else:
        spec = utils.read_catalog(inp)

    if "POLY0" in spec.meta:
        pc = []
        for pi in range(10):
            if f"POLY{pi}" in spec.meta:
                pc.append(spec.meta[f"POLY{pi}"])

        corr = np.polyval(pc, np.log(spec["wave"] * 1.0e4))
        spec["flux"] *= corr
        spec["err"] *= corr
        spec["corr"] = corr
    else:
        spec["corr"] = 1.0

    if "escale" not in spec.colnames:
        if hasattr(SCALE_UNCERTAINTY, "__len__"):
            if len(SCALE_UNCERTAINTY) < 6:
                spec["escale"] = 10 ** np.polyval(
                    SCALE_UNCERTAINTY, spec["wave"]
                )
            elif len(SCALE_UNCERTAINTY) == len(spec):
                spec["escale"] = SCALE_UNCERTAINTY
        else:
            spec["escale"] = SCALE_UNCERTAINTY
            # print('xx scale scalar', SCALE_UNCERTAINTY)

    for c in ["flux", "err"]:
        if hasattr(spec[c], "filled"):
            spec[c] = spec[c].filled(0)

    valid = np.isfinite(spec["flux"] + spec["err"])
    valid &= spec["err"] > 0
    valid &= spec["flux"] != 0

    if (err_mask is not None) & (valid.sum() > 0):
        _min_err = (
            np.nanpercentile(spec["err"][valid], err_mask[0]) * err_mask[1]
        )
        valid &= spec["err"] > _min_err

    if (err_median_filter is not None) & (valid.sum() > 0):
        med = nd.median_filter(
            spec["err"][valid].astype(float), err_median_filter[0]
        )
        medi = np.interp(
            spec["wave"], spec["wave"][valid], med, left=0, right=0
        )
        valid &= spec["err"] > err_median_filter[1] * medi

    set_spec_sys_err(spec, sys_err=sys_err)

    spec["full_err"][~valid] = 0
    spec["flux"][~valid] = 0.0
    spec["err"][~valid] = 0.0

    spec["valid"] = valid

    grating = spec.meta["GRATING"].lower()
    _filter = spec.meta["FILTER"].lower()

    if "R" not in spec.colnames:
        R_fwhm = msautils.get_default_resolution_curve(
            grating=grating,
            wave=spec['wave'],
            # grating_degree=2 default poly fit for gratings
            **kwargs
        )
        spec["R"] = R_fwhm
        spec["R"].description = "Spectral resolution from tabulated curves"

    spec.grating = grating
    spec.filter = _filter

    flam_unit = 1.0e-20 * u.erg / u.second / u.cm**2 / u.Angstrom

    um = spec["wave"].unit
    if um is None:
        um = u.micron

    spec.equiv = u.spectral_density(spec["wave"].data * um)

    spec["to_flam"] = (
        (1 * spec["flux"].unit).to(flam_unit, equivalencies=spec.equiv).value
    )
    spec.meta["flamunit"] = flam_unit.unit

    spec.meta["fluxunit"] = spec["flux"].unit
    spec.meta["waveunit"] = spec["wave"].unit

    spec["wave"] = spec["wave"].value
    spec["flux"] = spec["flux"].value
    spec["err"] = spec["err"].value

    return spec


def plot_spectrum(
    inp="jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits",
    z=9.505,
    vel_width=100,
    scale_disp=1.3,
    nspline=27,
    bspl=None,
    show_cont=True,
    draws=100,
    figsize=(16, 8),
    ranges=[(3650, 4980)],
    full_log=False,
    eazy_templates=None,
    use_full_dispersion=True,
    get_init_data=False,
    scale_uncertainty_kwargs=None,
    plot_unit=None,
    trim_negative=True,
    sys_err=0.02,
    return_fit_results=False,
    use_aper_columns=False,
    label=None,
    **kwargs,
):
    """
    Fit templates to a spectrum and make a diagnostic figure

    Parameters
    ----------
    inp : str or `~astropy.io.fits.HDUList` or `~msaexp.spectrum.SpectrumSampler`
        Input spectrum file path or HDUList object or SpectrumSampler object.
        Default is ``jw02767005001-02-clear-prism-nrs2-2767_11027.spec.fits``.

    z : float, optional
        Redshift value. Default is 9.505.

    vel_width : int, optional
        Velocity width, km/s. Default is 100.

    scale_disp : float, optional
        Dispersion scale factor. Default is 1.3.

    nspline : int, optional
        Number of spline continuum components. Default is 27.

    show_cont : bool, optional
        Whether to show continuum. Default is True.

    draws : int, optional
        Number of draws from the fit covariance to plot. Default is 100.

    figsize : tuple, optional
        Figure size. Default is (16, 8).

    ranges : list of tuples, optional
        List of wavelength ranges to plot in zoom panels. Default is [(3650, 4980)].

    full_log : bool, optional
        Whether to use full log. Default is False.

    eazy_templates : None or str, optional
        Eazy templates to use for the fit. Default is None.

    use_full_dispersion : bool, optional
        Whether to use full dispersion. Default is True.

    get_init_data : bool, optional
        If specified, just return the `~msaexp.spectrum.SpectrumSampler` object and
        the design matrix array ``A``

    scale_uncertainty_kwargs : None or dict, optional
        Scale uncertainty keyword arguments. Default is None.

    plot_unit : None, str, `~astropy.units.Unit`, optional
        Plot unit. Default is None.

    sys_err : float, optional
        Systematic error added in quadrature with the spectrum
        uncertainties. Default is 0.02.

    return_fit_results : bool
        Just return the fit results without making a plot -
        ``templates, coeffs, flam, eflam, _model, mask, full_chi2``

    use_aper_columns : bool, optional
        Whether to use aperture columns in the spectrum table.
        Default is False.

    label : None or str, optional
        Label to add to the figure. Default is None.

    kwargs : dict, optional
        Additional keyword arguments passed to `~msaexp.spectrum.SpectrumSampler`
        and  `~msaexp.spectrum.make_templates`

    Returns
    -------
    If ``return_fit_results = True``, returns a tuple containing the fit results:
    ``templates, coeffs, flam, eflam, _model, mask, full_chi2``.
    Otherwise, returns ``None``.

    """
    global SCALE_UNCERTAINTY

    lw, lr = utils.get_line_wavelengths()

    if isinstance(inp, str):
        sampler = SpectrumSampler(inp, sys_err=sys_err, **kwargs)
        file = inp
    elif isinstance(inp, pyfits.HDUList):
        sampler = SpectrumSampler(inp, sys_err=sys_err, **kwargs)
        file = None
    else:
        file = None
        sampler = inp

    if (label is None) & (file is not None):
        label = os.path.basename(file)

    spec = sampler.spec

    if (use_aper_columns > 0) & ("aper_flux" in spec.colnames):
        if ("aper_corr" in spec.colnames) & (use_aper_columns > 1):
            ap_corr = spec["aper_corr"] * 1
        else:
            ap_corr = 1

        flux_column = "aper_flux"
        err_column = "aper_full_err"

        # flam = spec['aper_flux']*spec['to_flam']*ap_corr
        # eflam = spec['aper_full_err']*spec['to_flam']*ap_corr
    else:

        flux_column = "flux"
        err_column = "full_err"

        # flam = spec['flux']*spec['to_flam']
        # eflam = spec['full_err']*spec['to_flam']
        ap_corr = 1.0

    flam = spec[flux_column] * spec["to_flam"] * ap_corr
    eflam = spec[err_column] * spec["to_flam"] * ap_corr

    wrest = spec["wave"] / (1 + z) * 1.0e4
    wobs = spec["wave"]
    mask = spec["valid"]

    flam[~mask] = np.nan
    eflam[~mask] = np.nan

    if bspl is None:
        if "continuum_model" in spec.colnames:
            bspl = spec["continuum_model"][None, :] * spec["to_flam"] * ap_corr
            apply_igm = False
        else:
            bspl = sampler.bspline_array(nspline=nspline, get_matrix=True)
            apply_igm = True

    templates, tline, _A = make_templates(
        sampler,
        z,
        bspl=bspl,
        eazy_templates=eazy_templates,
        vel_width=vel_width,
        scale_disp=scale_disp,
        use_full_dispersion=use_full_dispersion,
        grating=spec.grating,
        apply_igm=apply_igm,
        **kwargs,
    )

    if get_init_data:
        return spec, _A

    if scale_uncertainty_kwargs is not None:
        _, escl, _sys_err, _ = calc_uncertainty_scale(
            file=None,
            data=(spec, _A),
            sys_err=sys_err,
            **scale_uncertainty_kwargs,
        )
        # eflam *= escl
        spec["escale"] *= escl
        set_spec_sys_err(spec, sys_err=_sys_err)
        eflam = spec[err_column] * spec["to_flam"] * ap_corr

    okt = _A[:, mask].sum(axis=1) != 0

    _Ax = _A[okt, :] / eflam
    _yx = flam / eflam

    if eazy_templates is None:
        _x = np.linalg.lstsq(_Ax[:, mask].T, _yx[mask], rcond=None)
    else:
        _x = nnls(_Ax[:, mask].T, _yx[mask])

    coeffs = np.zeros(_A.shape[0])
    coeffs[okt] = _x[0]

    _model = _A.T.dot(coeffs)
    _mline = _A.T.dot(coeffs * tline)
    _mcont = _model - _mline

    full_chi2 = ((flam - _model) ** 2 / eflam**2)[mask].sum()
    cont_chi2 = ((flam - _mcont) ** 2 / eflam**2)[mask].sum()

    if return_fit_results:
        return templates, coeffs, flam, eflam, _model, mask, full_chi2

    try:
        oktemp = okt & (coeffs != 0)

        AxT = (_A[oktemp, :] / eflam)[:, mask].T

        covar_i = utils.safe_invert(np.dot(AxT.T, AxT))
        covar = utils.fill_masked_covar(covar_i, oktemp)
        covard = np.sqrt(covar.diagonal())

        has_covar = True
    except:
        has_covar = False
        covard = coeffs * 0.0
        N = len(templates)
        covar = np.eye(N, N)

    msg = "\n# line flux err\n# flux x 10^-20 erg/s/cm2\n"
    if label is not None:
        msg += f"# {label}\n"

    msg += f"# z = {z:.5f}\n# {time.ctime()}\n"

    cdict = {}
    eqwidth = {}

    for i, t in enumerate(templates):
        cdict[t] = [float(coeffs[i]), float(covard[i])]
        if t.startswith("line "):
            lk = t.split()[-1]

            # Equivalent width:
            # coeffs, line fluxes are in units of 1e-20 erg/s/cm2
            # _mcont, continuum model is in units of 1-e20 erg/s/cm2/A
            # so observed-frame equivalent width is roughly
            # eqwi = coeffs[i] / _mcont[ wave_obs[i] ]

            if lk in lw:
                lwi = lw[lk][0] * (1 + z) / 1.0e4
                continuum_i = np.interp(lwi, spec["wave"], _mcont)
                eqwi = coeffs[i] / continuum_i
            else:
                eqwi = np.nan

            eqwidth[t] = [float(eqwi)]

            msg += f"{t:>20}   {coeffs[i]:8.1f} ± {covard[i]:8.1f}"
            msg += f" (EW={eqwi:9.1f})\n"

    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    if "srcra" not in spec.meta:
        spec.meta["srcra"] = 0.0
        spec.meta["srcdec"] = 0.0
        spec.meta["srcname"] = "unknown"

    spec["model"] = _model / spec["to_flam"]
    spec["mline"] = _mline / spec["to_flam"]

    data = {
        "z": float(z),
        "file": file,
        "label": label,
        "ra": float(spec.meta["srcra"]),
        "dec": float(spec.meta["srcdec"]),
        "name": str(spec.meta["srcname"]),
        "wmin": float(spec["wave"][mask].min()),
        "wmax": float(spec["wave"][mask].max()),
        "coeffs": cdict,
        "covar": covar.tolist(),
        "wave": [float(m) for m in spec["wave"]],
        "flux": [float(m) for m in spec["flux"]],
        "err": [float(m) for m in spec["err"]],
        "escale": [float(m) for m in spec["escale"]],
        "model": [float(m) for m in _model / spec["to_flam"]],
        "mline": [float(m) for m in _mline / spec["to_flam"]],
        "templates": templates,
        "dof": int(mask.sum()),
        "fullchi2": float(full_chi2),
        "contchi2": float(cont_chi2),
        "eqwidth": eqwidth,
    }

    for k in ["z", "wmin", "wmax", "dof", "fullchi2", "contchi2"]:
        spec.meta[k] = data[k]

    # fig, axes = plt.subplots(len(ranges)+1,1,figsize=figsize)
    if len(ranges) > 0:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(2, len(ranges), figure=fig)
        axes = []
        for i, _ra in enumerate(ranges):
            axes.append(fig.add_subplot(gs[0, i]))

        axes.append(fig.add_subplot(gs[1, :]))

    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]

    _Acont = (_A.T * coeffs)[mask, :][:, :nspline]
    if trim_negative:
        _Acont[_Acont < 0.001 * _Acont.max()] = np.nan

    if (draws is not None) & has_covar:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu = np.random.multivariate_normal(
                coeffs[oktemp], covar_i, size=draws
            )

        # print('draws', draws, mu.shape, _A.shape)
        mdraws = _A[oktemp, :].T.dot(mu.T)
    else:
        mdraws = None

    if plot_unit is not None:
        unit_conv = (
            (1 * spec.meta["flamunit"])
            .to(plot_unit, equivalencies=spec.equiv)
            .value
        )
    else:
        unit_conv = np.ones(len(wobs))

    for ax in axes:
        if 1:
            ax.errorbar(
                wobs,
                flam * unit_conv,
                eflam * unit_conv,
                marker="None",
                linestyle="None",
                alpha=0.5,
                color="k",
                ecolor="k",
                zorder=100,
            )

        ax.step(
            wobs, flam * unit_conv, color="k", where="mid", lw=1, alpha=0.8
        )
        # ax.set_xlim(3500, 5100)

        # ax.plot(_[1]['templz']/(1+z), _[1]['templf'])

        ax.step(
            wobs[mask],
            (_mcont * unit_conv)[mask],
            color="pink",
            alpha=0.8,
            where="mid",
        )
        ax.step(
            wobs[mask],
            (_model * unit_conv)[mask],
            color="r",
            alpha=0.8,
            where="mid",
        )

        cc = utils.MPL_COLORS
        for w, c in zip(
            [3727, 4980, 6565, 9070, 9530, 1.094e4, 1.282e4, 1.875e4],
            [
                cc["purple"],
                cc["b"],
                cc["g"],
                "darkred",
                "darkred",
                cc["pink"],
                cc["pink"],
                cc["pink"],
            ],
        ):
            wz = w * (1 + z) / 1.0e4
            dw = 70 * (1 + z) / 1.0e4
            ax.fill_between(
                [wz - dw, wz + dw],
                [0, 0],
                [100, 100],
                color=c,
                alpha=0.07,
                zorder=-100,
            )

        if mdraws is not None:
            ax.step(
                wobs[mask],
                (mdraws.T * unit_conv).T[mask, :],
                color="r",
                alpha=np.maximum(1.0 / draws, 0.02),
                zorder=-100,
                where="mid",
            )

        if show_cont:
            ax.plot(
                wobs[mask],
                (_Acont.T * unit_conv[mask]).T,
                color="olive",
                alpha=0.3,
            )

        ax.fill_between(
            ax.get_xlim(),
            [-100, -100],
            [0, 0],
            color="0.8",
            alpha=0.5,
            zorder=-1,
        )

        ax.fill_betweenx(
            [0, 100],
            [0, 0],
            [1215.67 * (1 + z) / 1.0e4] * 2,
            color=utils.MPL_COLORS["orange"],
            alpha=0.2,
            zorder=-1,
        )

        ax.grid()

    # axes[0].set_xlim(1000, 2500)
    # ym = 0.15; axes[0].set_ylim(-0.1*ym, ym)

    for i, r in enumerate(ranges):
        axes[i].set_xlim(*[ri * (1 + z) / 1.0e4 for ri in r])
        # print('xxx', r)

    if spec.filter == "clear":
        axes[-1].set_xlim(0.6, 5.54)
        axes[-1].xaxis.set_minor_locator(MultipleLocator(0.1))
        axes[-1].xaxis.set_major_locator(MultipleLocator(0.5))
    elif spec.filter == "f070lp":
        axes[-1].set_xlim(0.65, 1.31)
        axes[-1].xaxis.set_minor_locator(MultipleLocator(0.02))
    elif spec.filter == "f100lp":
        axes[-1].set_xlim(0.92, 1.91)
        axes[-1].xaxis.set_minor_locator(MultipleLocator(0.02))
        axes[-1].xaxis.set_major_locator(MultipleLocator(0.1))
    elif spec.filter == "f170lp":
        axes[-1].set_xlim(1.65, 4.21)
    elif spec.filter == "f290lp":
        axes[-1].set_xlim(2.89, 5.31)
    else:
        axes[-1].set_xlim(wrest[mask].min(), wrest[mask].max())

    axes[-1].set_xlabel(f"obs wavelenth, z = {z:.5f}")

    # axes[0].set_title(os.path.basename(file))

    for ax in axes:
        xl = ax.get_xlim()
        ok = wobs > xl[0]
        ok &= wobs < xl[1]
        ok &= np.abs(wrest - 5008) > 100
        ok &= np.abs(wrest - 6564) > 100
        ok &= mask
        if ok.sum() == 0:
            ax.set_visible(False)
            continue

        ymax = np.maximum(
            (_model * unit_conv)[ok].max(),
            10 * np.median((eflam * unit_conv)[ok]),
        )

        ymin = np.minimum(-0.1 * ymax, -3 * np.median((eflam * unit_conv)[ok]))
        ax.set_ylim(ymin, ymax * 1.3)
        # print(xl, ymax)

    if ok.sum() > 0:
        if (np.nanmax((flam / eflam)[ok]) > 20) & (full_log):
            ax.set_ylim(0.005 * ymax, ymax * 5)
            ax.semilogy()

    if len(axes) > 0:
        gs.tight_layout(fig, pad=0.8)
    else:
        fig.tight_layout(pad=0.8)

    if label is not None:
        fig.text(
            0.015 * 12.0 / 12,
            0.005,
            f"{label}",
            ha="left",
            va="bottom",
            transform=fig.transFigure,
            fontsize=8,
        )

    fig.text(
        1 - 0.015 * 12.0 / 12,
        0.005,
        time.ctime(),
        ha="right",
        va="bottom",
        transform=fig.transFigure,
        fontsize=6,
    )

    return fig, spec, data


DEFAULT_FNUMBERS = [
    239,
    205,  # F814W, F160W
    362,
    363,
    364,
    365,
    366,
    370,
    371,
    375,
    376,
    377,  # NRC BB
    379,
    380,
    381,
    382,
    383,
    384,
    385,
    386,  # NRC MB
]

DEFAULT_REST_FNUMBERS = [
    120,
    121,  # GALEX
    218,
    219,
    270,
    271,
    272,
    274,  # FUV
    153,
    154,
    155,  # UBV
    156,
    157,
    158,
    159,
    160,  # SDSS ugriz
    161,
    162,
    163,  # 2MASS JHK
    414,
    415,
    416,  # ugi Antwi-Danso 2022
]

DEFAULT_SCALE_KWARGS = dict(
    order=0, sys_err=0.02, nspline=31, scale_disp=1.3, vel_width=100
)

BETA_KWS = dict(
    wrange=((0.14, 0.1860), (0.1955, 0.2580)),
    ref_wave=0.1550,
    dla_wrange=(0.1180, 0.1350),
    fit_restart=False,
    make_plot=False,
)


def do_integrate_filters(
    file,
    z=0,
    RES=None,
    fnumbers=DEFAULT_FNUMBERS,
    rest_fnumbers=DEFAULT_REST_FNUMBERS,
    scale_kwargs=DEFAULT_SCALE_KWARGS,
    beta_kwargs=BETA_KWS,
):
    """
    Integrate a spectrum through a list of filter bandpasses

    Parameters
    ----------
    file : str
        Spectrum filename

    z : float
        Redshift

    RES : `eazy.filters.FilterFile`
        Container of filter bandpasses

    fnumbers : list
        List of observed-frame ``f_numbers``

    rest_fnumbers : list
        List of rest-frame ``f_numbers`` to evaluate at ``z``

    scale_kwargs : dict, None
        If provided, initialize the spectrum by first passing through
        `~msaexp.spectrum.calc_uncertainty_scale`

    beta_kwargs : dict
        Compute rest-frame UV slope and DLA equivalent width with
        `~msaexp.spectrum.measure_uv_slope`
    Returns
    -------
    fdict : dict
        "horizontal" dictionary with keys for each separate filter

    sed : `~astropy.table.Table`
        "vertical" table of the integrated flux densities

    """
    import eazy.filters

    global SCALE_UNCERTAINTY
    SCALE_UNCERTAINTY = 1.0

    if RES is None:
        RES = eazy.filters.FilterFile(path=None)

    if scale_kwargs is not None:
        # Initialize spectrum rescaling uncertainties
        spec, escl, _sys_err, _ = calc_uncertainty_scale(
            file, z=z, **scale_kwargs
        )
        spec["escale"] *= escl
        set_spec_sys_err(spec, sys_err=_sys_err)

    else:
        # Just read the catalog
        spec = utils.read_catalog(file)
        if "escale" not in spec.colnames:
            spec["escale"] = 1.0

    rows = []
    fdict = {"file": file, "z": z, "escale": np.nanmedian(spec["escale"])}

    # Observed-frame filters
    for fn in fnumbers:
        obs_flux = integrate_spectrum_filter(spec, RES[fn], z=0)
        rows.append(
            [RES[fn].name, RES[fn].pivot / 1.0e4, fn, 0] + list(obs_flux)
        )

        for c, v in zip(
            ["valid", "frac", "flux", "err", "full_err"], obs_flux
        ):
            fdict[f"obs_{fn}_{c}"] = v

    for fn in rest_fnumbers:
        rest_flux = integrate_spectrum_filter(spec, RES[fn], z=z)
        rows.append(
            [RES[fn].name, RES[fn].pivot * (1 + z) / 1.0e4, fn, z]
            + list(rest_flux)
        )

        for c, v in zip(
            ["valid", "frac", "flux", "err", "full_err"], rest_flux
        ):
            fdict[f"rest_{fn}_{c}"] = v

    sed = utils.GTable(
        names=[
            "name",
            "pivot",
            "f_number",
            "z",
            "valid",
            "frac",
            "flux",
            "err",
            "full_err",
        ],
        rows=rows,
    )

    if beta_kwargs is not None:
        bet = measure_uv_slope(spec, z, **beta_kwargs)
        # beta, beta_unc, ndla, dla_value, dla_unc, _fig = _

        cov = bet.pop("beta_cov")

        for k in bet:
            fdict[k] = bet[k]

        for i in range(2):
            for j in range(2):
                fdict[f"beta_cov_{i}{j}"] = cov[i][j]

        # fdict['dla_beta'] = beta
        # fdict['dla_beta_unc'] = beta_unc
        # fdict['ndla'] = ndla
        # fdict['dla_value'] = dla_value
        # fdict['dla_unc'] = dla_unc

    return fdict, sed


def integrate_spectrum_filter(spec, filt, z=0, filter_fraction_threshold=0.1):
    """Integrate spectrum data through a filter bandpass

    Parameters
    ----------
    spec : `~astropy.table.Table`
        Spectrum data with columns ``wave`` (microns), ``flux`` and ``err``
        [``full_err``] (fnu) and ``valid``

    filt : `~eazy.filters.FilterDefinition`
        Filter bandpass object

    z : float
        Redshift

    filter_fraction_threshold : float
        Minimum allowed ``filter_fraction``, i.e. for filters that overlap
        with the observed spectrum

    Returns
    -------
    npix : int
        Number of "valid" pixels

    filter_fraction : float
        Fraction of the integrated bandpass that falls on valid spectral
        wavelength bins

    filter_flux : float
        Integrated flux density, in units of ``spec['flux']``

    filter_err : float
        Propagated uncertainty from ``spec['err']``

    filter_full_err : float
        Propagated uncertianty from ``spec['full_err']``, if available.
        Returns -1 if not available.

    """

    # Interpolate bandpass to wavelength grid
    filter_int = np.interp(
        spec["wave"],
        filt.wave / 1.0e4 * (1 + z),
        filt.throughput,
        left=0.0,
        right=0.0,
    )

    if "valid" in spec.colnames:
        valid = spec["valid"]
    else:
        valid = (spec["err"] > 0) & np.isfinite(spec["err"] + spec["flux"])

    if valid.sum() < 2:
        return (valid.sum(), 0.0, 0.0, -1.0, -1.0)

    # Full filter normalization
    filt_norm_full = np.trapz(
        filt.throughput / (filt.wave / 1.0e4), filt.wave / 1.0e4
    )

    # Normalization of filter sampled by the spectrum
    filt_norm = np.trapz(
        (filter_int / spec["wave"])[valid], spec["wave"][valid]
    )

    filter_fraction = filt_norm / filt_norm_full

    if filter_fraction < filter_fraction_threshold:
        return (valid.sum(), filter_fraction, 0.0, -1.0, -1.0)

    #######
    # Integrals

    # Trapezoid rule steps
    trapz_dx = utils.trapz_dx(spec["wave"])

    # Integrated flux
    fnu_flux = (
        (filter_int / filt_norm * trapz_dx * spec["flux"] / spec["wave"])[
            valid
        ]
    ).sum()

    # Propagated uncertainty
    fnu_err = np.sqrt(
        (
            (filter_int / filt_norm * trapz_dx * spec["err"] / spec["wave"])[
                valid
            ]
            ** 2
        ).sum()
    )

    if "full_err" in spec.colnames:
        fnu_full_err = np.sqrt(
            (
                (
                    filter_int
                    / filt_norm
                    * trapz_dx
                    * spec["full_err"]
                    / spec["wave"]
                )[valid]
                ** 2
            ).sum()
        )
    else:
        fnu_full_err = -1.0

    return valid.sum(), filter_fraction, fnu_flux, fnu_err, fnu_full_err


def measure_uv_slope(
    spec,
    z,
    wrange=((0.14, 0.1860), (0.1955, 0.2580)),
    ref_wave=0.1550,
    dla_wrange=(0.1180, 0.1350),
    fit_restart=False,
    make_plot=False,
):
    """
    Measure UV slope beta and absolute magnitude

    Parameters
    ----------
    spec : `~astropy.table.Table`
        Spectrum data with columns ``wave`` (microns), ``flux`` and ``err``
        [``full_err``] (fnu) and ``valid``

    z : float
        Redshift

    wrange : list of (float, float)
        Wavelength ranges for the UV slope fit, microns. The default excludes
        potential contribution from CIII]1909.

    ref_wave : float
        Reference wavelength, microns

    dla_wrange : (float, float)
        Wavelength range for the DLA equivalent width parameter, which is the
        integral of ``(1 - Fobs/Fcont)`` near Ly-alpha where ``Fobs`` is the
        observed spectrum and ``Fcont`` is the continuum spectrum extrapolated
        with the derived UV slope beta

    fit_restart : bool
        Redo fit after perturbing initial parameters. Seems to help make the
        fit covariance more reasonable in some cases.

    make_plot : bool
        Make a diagnostic plot

    Returns
    -------
    res : dict
        - ``beta`` = Derived UV slope ``flam = lam**beta``
        - ``beta_ref_flux`` = Flux density at redshifted ``ref_wave``
        - ``beta_cov`` = 2x2 covariance matrix for the fit for ``beta``,
           ``flux_ref``
        - ``beta_npix`` = Number of wavelength pixels satisfying ``wrange``
        - ``beta_wlo, beta_whi`` = Minimum and maximum of rest-frame
            wavelengths within ``wrange``
        - ``beta_nmad`` = NMAD of the power-law beta fit
        - ``dla_npix`` = Number of spectral bins for DLA parameter estimate
        - ``dla_value`` = DLA equivalent width, Angstroms
        - ``dla_unc`` = Uncertainty on the DLA EQW
        - ``fig`` = `matplotlib.Figure` object if requested with make_plot
    """
    from scipy.optimize import minimize

    # Force fit beta slope to linear flux densities
    fit_linear = True

    wrest = spec["wave"] / (1 + z)

    blim = np.zeros(len(spec), dtype=bool)
    for wr in wrange:
        blim |= (wrest >= wr[0]) & (wrest <= wr[1])

    pos = spec["flux"].data > 0
    blim &= spec["flux"] > -spec["full_err"]

    npix = blim.sum()

    if npix < 3:
        cov = np.ones((2, 2)) * np.nan
        res = {
            "beta": np.nan,
            "beta_ref_flux": np.nan,
            "beta_cov": np.ones((2, 2)) * np.nan,
            "beta_npix": npix,
            "beta_wlo": np.nan,
            "beta_whi": np.nan,
        }
        return res

    wlo = wrest[blim].min()
    whi = wrest[blim].max()
    scale_err = 1.0

    # Objective function for beta fit
    def _objfun_beta(theta, x, flux, err, ret):
        m = theta[1] * x ** (theta[0] + 2)
        if ret == 1:
            return m

        chi2 = ((flux - m) ** 2 / err**2).sum()
        # Penalize very blue
        if theta[0] < -2:
            chi2 += (theta[0] - -2) ** 2 / 2 / 1.0**2

        return chi2

    # Guess from log-linear fit mag = beta * ln(wave) + mag(ref_wave)
    lnf = 23.9 - 2.5 * np.log10((spec["flux"].data))
    lnfe = (
        2.5 / np.log(10) * spec["full_err"] / scale_err / spec["flux"]
    ).data
    x = np.log(wrest / ref_wave)
    A = np.array([-x, x**0])
    AxT = (A / lnfe).T
    c = np.linalg.lstsq(
        AxT[blim & pos, :], (lnf / lnfe)[blim & pos], rcond=None
    )

    beta = c[0][0] - 2
    ref_flux = 10 ** (-0.4 * (c[0][1] - 23.9))
    theta = c[0]

    np.random.seed(fit_restart * 1)

    nmad = -1

    if fit_linear:
        # Now fit in linear flux densities
        x0 = np.array([c[0][0] - 2, 10 ** (-0.4 * (c[0][1] - 23.9))])
        xargs = (
            wrest[blim] / ref_wave,
            spec["flux"].data[blim],
            spec["full_err"].data[blim],
        )

        _x = minimize(_objfun_beta, x0=x0, method="bfgs", args=(*xargs, 0))

        best_model = _objfun_beta(_x.x, *xargs, 1)
        resid = (xargs[1] - best_model) / xargs[2]

        # Rescale  Uncertainties
        nmad = utils.nmad(resid)

        if fit_restart > 0:
            # Restart
            print("nmad: ", nmad, fit_restart)
            xargs = (
                wrest[blim] / ref_wave,
                spec["flux"].data[blim],
                spec["full_err"].data[blim] * nmad,
            )

            if fit_restart > 1:
                _x = minimize(
                    _objfun_beta,
                    x0=_x.x + np.random.normal(size=2) * np.array([0.1, 0.03]),
                    method="bfgs",
                    args=(*xargs, 0),
                )  # , tol=1.e-6)
            else:
                _x = minimize(
                    _objfun_beta, x0=x0, method="bfgs", args=(*xargs, 0)
                )  # , tol=1.e-6)

        theta = _x.x
        beta, ref_flux = theta

        # Covariance is hess_inv from the BFGS optimization
        cov = _x.hess_inv
    else:
        cov = utils.safe_invert(
            np.dot(AxT[blim & pos, :].T, AxT[blim & pos, :])
        )
        nmad = -1

    res = {
        "beta": beta,
        "beta_ref_flux": ref_flux,
        "beta_cov": cov,
        "beta_npix": npix,
        "beta_wlo": wlo,
        "beta_whi": whi,
        "beta_nmad": nmad,
    }

    # DLA equivalent width parameter
    if fit_linear:

        xdla = (
            (wrest >= dla_wrange[0])
            & (wrest <= dla_wrange[1])
            & (spec["valid"])
        )
        res["dla_npix"] = xdla.sum()

        if xdla.sum() < 3:
            dla_value = -1
            dla_unc = -1

            res["dla_value"] = -1
            res["dla_unc"] = -1
        else:
            draws = np.random.multivariate_normal(theta, cov, 1000)
            xargs = (
                wrest[xdla] / ref_wave,
                spec["flux"].data[xdla],
                spec["full_err"].data[xdla],
            )

            dla_continuum = _objfun_beta(_x.x, *xargs, 1)
            dla_draws = np.array(
                [_objfun_beta(xi, *xargs, 1) for xi in draws]
            ).T
            dla_continuum_unc = np.std(dla_draws, axis=1)

            dx = utils.trapz_dx(wrest[xdla]) * 1.0e4
            sx = spec[xdla]
            ydata = 1 - sx["flux"] / dla_continuum
            vdata = (sx["flux"] / dla_continuum) ** 2 * (
                (sx["full_err"] / sx["flux"]) ** 2
                + (dla_continuum_unc / dla_continuum) ** 2
            )

            # Do trapezoid rule integration and propagation of uncertainty
            dla_value = (ydata * dx).sum()
            dla_unc = np.sqrt((vdata * dx**2).sum())

            res["dla_value"] = dla_value
            res["dla_unc"] = dla_unc
    else:
        res["dla_npix"] = 0
        res["dla_value"] = dla_value = -1
        res["dla_unc"] = dla_unc = -1

    # Make a diagnostic plot
    if make_plot:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        draws = np.random.multivariate_normal(theta, cov, 100)

        if fit_linear:
            xargs = (
                wrest / ref_wave,
                spec["flux"].data,
                spec["full_err"].data,
            )
            best_model = _objfun_beta(_x.x, *xargs, 1)
            flux_draws = np.array(
                [_objfun_beta(xi, *xargs, 1) for xi in draws]
            ).T

            mag = 23.9 - 2.5 * np.log10(ref_flux)
            emag = 2.5 / np.log(10) * np.sqrt(cov[1][1]) / ref_flux

        else:
            best_model = 10 ** (-0.4 * (A.T.dot(c[0]) - 23.9))
            flux_draws = 10 ** (-0.4 * (A.T.dot(draws.T) - 23.9))

            mag = c[0][1]
            emag = np.sqrt(cov[1][1])

        ok = ~blim
        ax.errorbar(
            wrest[ok],
            spec["flux"][ok],
            spec["full_err"][ok] / scale_err,
            color="k",
            marker=".",
            linestyle="None",
            alpha=0.5,
        )

        ax.errorbar(
            wrest[blim],
            spec["flux"][blim],
            spec["full_err"][blim] / scale_err,
            color="magenta",
            marker="o",
            linestyle="None",
            alpha=0.5,
        )

        # plot labels
        lab = r"$\beta = bbv \pm bbe$"
        lab += "\n" + r"$m_{rr} = mmv \pm mme$"

        lab = lab.replace("bbv", f"{beta:5.2f}")
        lab = lab.replace("bbe", f"{np.sqrt(cov[0][0]):5.2f}")
        lab = lab.replace("rr", f"{ref_wave*1e4:.0f}")
        lab = lab.replace("mmv", f"{mag:5.2f}")
        lab = lab.replace("mme", f"{emag:4.2f}")

        ax.plot(wrest, best_model, color="tomato", label=lab)

        if dla_value < 0:
            y0 = 2 * np.interp(0.1216, wrest[xdla], dla_continuum)
        else:
            y0 = 0.0

        ax.vlines(
            0.1216 + np.array([-1, 1]) * dla_value / 2.0e4,
            y0,
            np.interp(0.1216, wrest[xdla], dla_continuum),
            color="orange",
            label=f"EW_DLA = {dla_value:.1f} ({dla_unc:.1f})",
        )

        _ = ax.plot(wrest, flux_draws, color="pink", alpha=0.1)
        ax.set_xlim(0.1, 0.39)
        ymax = flux_draws[blim, :].max() + spec["full_err"][blim].max()
        for wr in wrange:
            ax.vlines(
                wr, -0.5 * ymax, 10 * ymax, color="purple", lw=3, alpha=0.4
            )

        leg = ax.legend(loc="upper right")
        leg.set_title(f"{spec.meta['SRCNAME']}\nz = {z:.4f}")

        ax.set_ylim(-0.5 * ymax, 2 * ymax)

        ax.errorbar(
            wrest[xdla], dla_continuum, dla_continuum_unc, color="orange"
        )

        ax.set_xlabel(r"$\lambda_\mathrm{rest}$")
        ax.set_ylabel(r"$f_\nu$")
        ax.grid()

        fig.tight_layout(pad=1)
        res["fig"] = fig

    return res
