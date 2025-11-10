"""
Tools for working with emission lines
"""

import os

import numpy as np

import astropy.constants
import astropy.units as u

from grizli import utils

from .general import module_data_path

__all__ = ["LineList", "MolecularHydrogen"]


class LineList(object):

    WHITE_BBOX = {
        "fc": "w",
        "ec": "None",
        "alpha": 1.0,
        "boxstyle": "square,pad=0.3",
    }

    CLEAN_CENTER_KWARGS = {
        "ha": "center",
        "bbox": WHITE_BBOX,
        "x_pad": 0.0,
    }

    LEFT_OFFSET_KWARGS = {
        "ha": "right",
        "x_pad": -0.005,
    }

    RIGHT_OFFSET_KWARGS = {
        "ha": "left",
        "x_pad": 0.005,
    }

    def __init__(self):
        """
        Tools for working with a table of lines

        Examples
        --------

        .. plot::
            :include-source:

            from msaexp.utils import LineList

            ll = LineList()
            fig = ll.demo()

        """
        self.load_data()

    def load_data(self, fill_color="0.2"):
        """
        Load the data from ``msaexp/data/linelist.ecsv``
        """
        linelist_file = os.path.join(module_data_path(), "linelist.ecsv")

        self.data = utils.read_catalog(linelist_file, format="ascii.ecsv")
        self.data["color"] = self.data["color"].filled(fill_color)
        self.N = len(self.data)

    def select_lines(
        self,
        z=0,
        wave_range=[0, np.inf],
        prefixes=[],
        priorities=[4, 5],
        floor=True,
        **kwargs,
    ):
        """
        Get selection of lines within a wavelength range

        Parameters
        ----------
        z : float
            Redshift

        wave_range : [float, float]
            Observed-frame wavelength range.  The script tries to automatically
            distinguish between wavelength units of Angstroms or microns assuming
            the latter if ``wave_range[1] < 500``

        prefixes : list
            List of strings compared to the ``name`` column in the linelist using
            the test ``name.startswith(prefix)``

        priorities : list
            List of linelist priorities to include

        floor : bool
            If true, ignore decimal part of the linelist priorities and just compare
            the integer values

        Returns
        -------
        selected : boolean array
            Selection array on the listlist table that satisfy the selection
            criteria

        x_scale : float
            Scale factor needed to put the "wavelength" column of the linelist
            in the frame of ``wave_range``

        """
        x_scale = 1 + z
        if wave_range[1] < 500:
            # axis is in probably in microns
            x_scale /= 1.0e4

        selected = np.zeros(self.N, dtype=bool)

        for i, row in enumerate(self.data):
            selected[i] = (row["wavelength"][0] > wave_range[0] / x_scale) & (
                row["wavelength"][0] < wave_range[1] / x_scale
            )
            if round:
                selected[i] &= np.floor(row["priority"]) in priorities
            else:
                selected[i] &= row["priority"] in priorities

            if len(prefixes) > 0:
                in_prefixes = False
                for prefix in prefixes:
                    if row["name"].startswith(prefix):
                        in_prefixes = True
                        break

                selected[i] &= in_prefixes

        return selected, x_scale

    def add_to_axis(
        self,
        ax,
        bottom=0,
        top=1.0,
        x_pad=0,
        wfunc=np.min,
        y_label=0.98,
        priorities=[3, 4, 5],
        label_priorities=None,
        label_prefixes=[],
        y_normalized=True,
        rotation=90,
        ha="center",
        va="top",
        lw=1.5,
        ls="-",
        fontsize=7,
        alpha=0.8,
        bbox={"fc": "w", "ec": "None", "alpha": 1.0},
        zorder=1,
        label_zorder=2,
        **kwargs,
    ):
        """
        Add vlines and labels to axis

        Parameters
        ----------
        ax : `matplotlib` axis
            Axis to draw labels in

        bottom : float, (array-like, array-like)
            Lower limit of vlines to draw.  If two arrays are provided, will
            interpolate the value ``value = np.interp(wave, *bottom)``

        top : float, (array-like, array-like)
            Upper limit of the vlines, with same interpolation options as ``bottom``

        x_pad : float
            relative padding for text labels

        wfunc : func
            Function to apply to the potential multiple entries for line complexes
            to specify where to draw the labels

        y_label : float
            Vertical location of the line text labels

        priorities : list
            List of priorities to include from the linelist table

        label_priorities : list, None
            List of priorities to include with text labels.  If not provided, will be
            the same as ``priorities``

        label_prefixes : list
            Prefix filtering for lines with text labels

        y_normalized : bool
            If True, the ``bottom``, ``top`` and ``y_label`` values are calculated
            in the normalized frame of the plot extent, i.e., with 0 at the bottom
            and 1 at the top.  Not used if values are interpolated.

        lw, ls : float, string
            Linewidth and linestyle properties for the vlines

        rotation, ha, va, fontsize, alpha, bbox, zorder :
            Text label properties

        Returns
        -------
        data : table
            View of the linelist table with the selected lines

        """

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if label_priorities is None:
            label_priorities = priorities

        selected, x_scale = self.select_lines(
            wave_range=xlim, priorities=priorities, **kwargs
        )

        if selected.sum() == 0:
            return None

        is_xlog = ax.xaxis.get_scale() == "log"
        is_ylog = ax.yaxis.get_scale() == "log"

        for row in self.data[selected]:
            wave_i = np.array(row["wavelength"]) * x_scale
            if hasattr(bottom, "__len__"):
                vlo = np.interp(wave_i, *bottom)
            elif y_normalized:
                if is_ylog:
                    vlo = 10 ** np.interp(bottom, [0, 1], np.log10(ylim))
                else:
                    vlo = np.interp(bottom, [0, 1], ylim)
            else:
                vlo = bottom

            if hasattr(top, "__len__"):
                vhi = np.interp(wave_i, *top)
            elif y_normalized:
                if is_ylog:
                    vhi = 10 ** np.interp(top, [0, 1], np.log10(ylim))
                else:
                    vhi = np.interp(top, [0, 1], ylim)
            else:
                vhi = top

            ax.vlines(
                wave_i,
                vlo,
                vhi,
                color=row["color"],
                ls=ls,
                lw=lw,
                alpha=alpha,
                zorder=zorder,
            )

            if (y_label is not None) & (row["priority"] in label_priorities):
                if len(label_prefixes) > 0:
                    in_prefixes = False
                    for prefix in label_prefixes:
                        if row["name"].startswith(prefix):
                            in_prefixes = True
                            break
                    if not in_prefixes:
                        continue

                wl = wfunc(wave_i)

                if hasattr(y_label, "__len__"):
                    yl = np.interp(wl, *y_label)
                elif y_normalized:
                    if is_ylog:
                        yl = 10 ** np.interp(y_label, [0, 1], np.log10(ylim))
                    else:
                        yl = np.interp(y_label, [0, 1], ylim)
                else:
                    yl = y_label

                if is_xlog:
                    dx = wl * (np.exp(x_pad * np.log(xlim[1] / xlim[0])) - 1)
                else:
                    dx = x_pad * (xlim[1] - xlim[0])

                ax.text(
                    wl + dx,
                    yl,
                    row["tex"],
                    color=row["color"],
                    va=va,
                    ha=ha,
                    alpha=alpha,
                    fontsize=fontsize,
                    bbox=bbox,
                    zorder=label_zorder,
                    rotation=rotation,
                )

        return self.data[selected]

    def demo(self, xlim=[0.35, 0.51]):
        """
        Show a demonstration of various plot options
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(7, 1, figsize=(8, 8))
        for ax in axes:
            ax.set_ylim(-0.5, 2.5)

        kws = dict(fontsize=8, bbox=self.WHITE_BBOX, ha="left", va="bottom")
        lxy = (0.02, 0.09)

        axes[0].set_xlim(xlim[0] * 1.0e4, xlim[1] * 1.0e4)
        self.add_to_axis(axes[0])
        axes[0].text(*lxy, "defaults", transform=axes[0].transAxes, **kws)

        for ax in axes[1:]:
            ax.set_xlim(xlim)

        self.add_to_axis(
            axes[1], prefixes=["O", "Pa"], va="bottom", y_normalized=False
        )
        axes[1].text(
            *lxy,
            'prefixes=["O","Pa"], y_normalized=False',
            transform=axes[1].transAxes,
            **kws,
        )

        self.add_to_axis(
            axes[2],
            priorities=[3, 4, 5],
            label_prefixes=["Ne", "H"],
            **self.LEFT_OFFSET_KWARGS,
        )
        axes[2].text(
            *lxy,
            'priorities=[3,4,5], label_prefixes=["Ne","H"], **LEFT_OFFSET_KWARGS',
            transform=axes[2].transAxes,
            **kws,
        )

        axes[3].set_ylim(0.1, 10)
        axes[3].semilogy()
        self.add_to_axis(
            axes[3], priorities=[2, 3, 4, 5], label_priorities=[4, 5]
        )
        axes[3].text(
            *lxy,
            "priorities=[2,3,4,5], label_priorities=[4,5], semilogy",
            transform=axes[3].transAxes,
            **kws,
        )

        axes[4].set_ylim(0.1, 10)
        axes[4].loglog()
        self.add_to_axis(
            axes[4], priorities=[3, 4, 5], y_label=0.5, va="center"
        )
        axes[4].text(
            *lxy,
            'loglog, y_label=0.5, va="center"',
            transform=axes[4].transAxes,
            **kws,
        )

        xi = np.linspace(-1, 1, 128)
        yi = np.cos(xi * 4 * np.pi)
        xp = np.interp(xi, [-1, 1], xlim)
        axes[5].plot(xp, yi - 1, color="0.8")
        axes[5].plot(xp, yi * 0.2 + 1, color="0.8")
        self.add_to_axis(
            axes[5],
            bottom=(xp, yi - 1),
            top=(xp, yi * 0.2 + 1),
            priorities=[3, 4, 5],
        )
        axes[5].text(
            *lxy,
            "interpolated bottom, top",
            transform=axes[5].transAxes,
            **kws,
        )

        axes[6].plot(xp, yi * 0.2 + 1, color="0.8")
        self.add_to_axis(
            axes[6],
            y_label=(xp, yi * 0.2 + 1),
            priorities=[3, 4, 5],
            va="center",
            bbox=self.WHITE_BBOX,
            fontsize=5,
        )
        axes[6].text(
            *lxy,
            'interpolated y_label, va="center", white bbox, fontsize=5',
            transform=axes[6].transAxes,
            **kws,
        )

        fig.tight_layout(pad=1)

        return fig

    def spectrum_line_models(
        self, spec, z=0, selected=None, fnu=True, **kwargs
    ):
        """
        Make line models `msaexp.spectrum.SpectrumSampler.fast_emission_line`

        Parameters
        ----------
        spec : `~msaexp.spectrum.SpectrumSampler`
            Spectrum object

        z : float
            Redshift

        selected : None, array
            Line selection array.  If not provided, will pull from
            `LineList.select_lines`.

        fnu : bool
            Return in fnu units

        kwargs : dict
            Keyword arguments passed to `LineList.select_lines` and
            `msaexp.spectrum.SpectrumSampler.fast_emission_line`

        Returns
        -------
        line_templates : array-like (Nlines, Nspec)
            Sampled line models

        line_names : list
            String line names

        """

        if selected is None:
            valid_wave = spec["wave"][spec["valid"]]
            wave_range = [valid_wave.min(), valid_wave.max()]
            selected, x_scale = self.select_lines(
                wave_range=wave_range, z=z, **kwargs
            )

        line_names = []
        line_templates = []
        for i, row in enumerate(self.data[selected]):
            line_i = np.zeros_like(spec["flux"])
            for lw, lr in zip(row["wavelength"], row["ratio"]):
                line_i += spec.fast_emission_line(
                    lw * (1 + z) / 1.0e4,
                    line_flux=lr,
                    **kwargs,
                )

            if fnu:
                line_i /= spec["to_flam"]

            line_templates.append(line_i)
            line_names.append(row["name"])

        return np.array(line_templates), line_names


class MolecularHydrogen:

    # Reference transition
    reference = ("1-0", "S(1)")

    separate_ZT = False

    def __init__(self, **kwargs):
        """
        Tools for working with molecular hydrogen lines using the helpful
        relations from Aditya Togi and J. D. T. Smith 2016 ApJ, 830, 18

        Optically thin flux:
        
        .. math::
        
            F_j = h \nu A N_{j+2} \Omega / (4 \pi)$

        Line flux F as a function of temperature, T (excitation diagram):

        .. math::
          
            N_{j+2} = g_j / Z(T) \exp{-E / kT}
            
            Z(T) = \sum g_j \exp{-E / kT}
        
            F_j(T) \propto N_{j+2} A / \lambda

        Total number, mass:
        
        .. math::
            
            n_\mathrm{tot} &= N_\mathrm{tot} \Omega d^2 \\
            
                           &= 4 \pi d^2 \sum F_j / (h \nu A)

        Line data from the Gemini compilation at
        https://www.gemini.edu/observing/resources/near-ir-resources/spectroscopy/important-h2-lines
        """
        
        self.load_data()

        self.initialize_flux_table()
        self.unv = utils.Unique(self.data["vib"], verbose=False)

    def load_data(self):
        """
        Read the data table and set the units on some columns
        """
        self.data = utils.read_catalog(
            os.path.join(module_data_path(), "h2_linelist.txt")
        )

        self.data["wave"].unit = u.micron
        self.data["A"].unit = 1.e-7 * u.second**-1
        self.data["j"] = [int(r[2:-1]) for r in self.data["rot"]]
        self.data["OSQ"] = [r[0] for r in self.data["rot"]]

    def initialize_flux_table(self):
        """
        Initialize ``flux`` table
        """
        self.flux = self.data["wave",]
        self.flux["name"] = self.line_names()
        self.flux["flux"] = 0.0
        self.flux["Nj"] = 0.0
        self.flux["mask"] = True
        self.flux.meta["T"] = (0, "Temperature")
        self.Tflux = None

    def line_names(self, prefix=r"H$_2$"):
        """
        LaTeX line names
        """
        return np.array(
            [
                prefix + f'{row["vib"]:^5} {row["rot"]}'
                for row in self.data["vib", "rot"]
            ]
        )

    @property
    def nu(self):
        """
        Transition frequency
        """
        return (astropy.constants.c / self.data["wave"]).to(u.Hz)

    @property
    def h_nu_A(self):
        """
        precompute :math:`h \nu  A` (erg / second)
        """
        return (
            astropy.constants.h * self.nu * self.data["A"]
        ).to(u.erg / u.second)

    @property
    def N(self):
        """
        Length of data table
        """
        return len(self.data)

    @property
    def ix(self):
        """
        Index of the reference transition (usually "1-0 S(1)")
        in the data table
        """
        return np.where(
            (self.data["vib"] == self.reference[0])
            & (self.data["rot"] == self.reference[1])
        )[0][0]

    @staticmethod
    def temperature_powerlaw(n=4.5, Tl=50, Tu=2000, nsteps=16):
        """
        Generate a powerlaw temperature distribution :math:`dN = T^{-n} dT`
        """
        tgrid = np.linspace(Tl, Tu, nsteps)
        Nt = tgrid**-n * (n-1) / (Tl**(1-n) - Tu**(1-n))
        return tgrid, Nt

    def ZT(self, T=1000.0):
        """
        Compute :math:`Z(T) = \sum g_j \exp{-E / kT}`
        """
        # ZT = np.zeros(self.N)
        # for v in self.unv.values:
        #     un_i = self.unv[v]
        #     ZT[un_i] = np.sum(
        #         (self.data["gJ"] * np.exp(-self.data["Eupper"] / T))[un_i]
        #     )

        # ZT = np.zeros_like(self.data["wave"])
        # for v in self.unv.values:
        #     un_i = self.unv[v]
        #     for i in [0,1]:
        #         sub = un_i & (self.data["j"] % 2 == i)
        #
        #         ZT[sub] = np.sum(
        #             (self.data["gJ"] * np.exp(-self.data["Eupper"] / T))[sub]
        #         )

        if self.separate_ZT:
            ZT = np.zeros_like(self.data["wave"])
            for i in [0,1]:
                sub = self.data["j"] % 2 == i
                ZT[sub] = np.sum(
                    (self.data["gJ"] * np.exp(-self.data["Eupper"] / T))[sub]
                )
        else:
            ZT = np.sum(
                (self.data["gJ"] * np.exp(-self.data["Eupper"] / T))
            )

        return ZT

    def Nj(self, T=1000.0):
        """
        Compute number density: :math:`N_j = g_j / Z(T) \exp{-E / kT}`
        """
        if hasattr(T, "__len__"):
            # Temperature distribution
            norm = np.trapz(T[1], T[0])
            Tx = T[0]
            Ty = T[1] / norm
            dT = utils.trapz_dx(Tx)
        else:
            Tx = [T]
            Ty = [1.0]
            dT = [1.0]

        Nj = np.zeros(self.N)
        Nt = len(Tx)

        for i in range(Nt):
            Nj += (
                dT[i]
                * Ty[i]
                * self.data["gJ"]
                / self.ZT(Tx[i])
                * np.exp(-self.data["Eupper"] / Tx[i])
            )

        return Nj

    def line_flux(self, T=1000.0, **kwargs):
        """
        Line flux :math:`F_j = h \nu A N_{j+2} \Omega / (4 \pi)`
        """
        Nj = self.Nj(T) * u.cm**-2

        Fj = self.h_nu_A * Nj

        return Nj, Fj.to(u.erg / u.second / u.cm**2)


    def update_flux_table(
        self, T=1000.0, min_line_fraction=0.05, wave_range=(1.6, 2.6), **kwargs
    ):
        """
        Get a list of lines and ratios
        """
        Nj, Fj = self.line_flux(T, **kwargs)

        keep = (
            (Fj > min_line_fraction * Fj[self.ix])
            & (self.data["wave"] > wave_range[0])
            & (self.data["wave"] < wave_range[1])
        )

        self.flux["flux"] = Fj
        self.flux["Nj"] = Nj

        self.flux["mask"] = keep

        self.flux.meta["ref_flux"] = Fj[self.ix]

        self.Tflux = T

    def msaexp_model(
        self,
        spec,
        z=0.0,
        update_flux=True,
        wave_range=None,
        fnu=True,
        **kwargs,
    ):
        """
        Make an MSAEXP model
        """
        if update_flux:
            if wave_range is None:
                v = spec["valid"]

                wave_range = (
                    spec["wave"][v].min() / (1 + z),
                    spec["wave"][v].max() / (1 + z),
                )

            self.update_flux_table(wave_range=wave_range, **kwargs)

        model = np.zeros_like(spec["flux"])

        for row in self.flux[self.flux["mask"]]:
            model += spec.fast_emission_line(
                row["wave"] * (1 + z),
                line_flux=row["flux"] / self.flux.meta["ref_flux"],
                **kwargs,
            )

        if fnu:
            model /= spec["to_flam"]

        return model

    def h2_mass(self, line_flux=1.0e-20 * u.erg / u.second / u.cm**2, ix=None, z=1.01, n=4.5, Tl=50, Tu=4000):
        """
        Still figuring out unit conversions....
        """
        from astropy.cosmology import WMAP9

        dL = WMAP9.luminosity_distance(z)  # .to(u.cm)

        tgrid, Nt = self.temperature_powerlaw(n=n, Tl=Tl, Tu=Tu, nsteps=512)

        Nj = self.Nj(T=(tgrid, Nt))
        
        if ix is None:
            ix = self.ix
        
        Ntot = line_flux / self.h_nu_A[ix] / Nj[ix+2] * dL.to(u.cm)**2
        
        Mtot = (Ntot * 2 * astropy.constants.m_p).to(u.Msun)
