"""
Tools for working with emission lines
"""

import os

import numpy as np

import astropy.constants
import astropy.units as u

from grizli import utils

from .general import module_data_path

__all__ = ["MolecularHydrogen"]


class MolecularHydrogen:

    reference = ("1-0", "S(1)")

    def __init__(self, **kwargs):
        """
        Tools for working with molecular hydrogen lines using the helpful
        relations from Aditya Togi and J. D. T. Smith 2016 ApJ, 830, 18

        Optically thin flux:
          $F_j = h \nu A N_{j+2} \Omega / (4 \pi)$

        Line flux F as a function of temperature, T (excitation diagram):

          $N_{j+2} = g_j / Z(T) \exp{-E / kT}$
          $Z(T) = \Sum g_j \exp{-E / kT}$
          $F_j(T) \propto N_{j+2} A / \lambda$

        Total number, mass:
            $n_\mathrm{tot} = N_\mathrm{tot} \Omega d^2$
            $     = 4 \pi d^2 \Sum F_j / (h \nu A)$

        Line data from the Gemini compilation at
        https://www.gemini.edu/observing/resources/near-ir-resources/spectroscopy/important-h2-lines
        """
        self.data = utils.read_catalog(
            os.path.join(module_data_path(), "h2_lines.txt")
        )
        self.initialize_flux_table()
        self.unv = utils.Unique(self.data["vib"], verbose=False)

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

    def line_names(self, prefix=r"H$_2$"):
        return np.array(
            [
                prefix + f'{row["vib"]:^5} {row["rot"]}'
                for row in self.data["vib", "rot"]
            ]
        )

    @property
    def wave(self):
        return self.data["wave"] * u.micron

    @property
    def nu(self):
        """
        Frequency
        """
        return (astropy.constants.c / self.wave).to(u.Hz)

    @property
    def h_nu_A(self):
        """
        $h * \nu * A$
        """
        return (
            astropy.constants.h * self.nu * self.data["A"] * u.second**-1
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

    def ZT(self, T=1000.0):
        """
        $Z(T) = \Sum g_j \exp{-E / kT}$
        """
        # ZT = np.zeros(self.N)
        # for v in self.unv.values:
        #     un_i = self.unv[v]
        #     ZT[un_i] = np.sum(
        #         (self.data["gJ"] * np.exp(-self.data["Eupper"] / T))[un_i]
        #     )
        ZT = np.sum((self.data["gJ"] * np.exp(-self.data["Eupper"] / T)))

        return ZT

    def Nj(self, T=1000.0):
        """
        Compute number density $N_{j+2} = g_j / Z(T) \exp{-E / kT}$
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
        $F_j = h \nu A N_{j+2} \Omega / (4 \pi)$
        """
        Nj = self.Nj(T) * u.cm**-2

        Fj = (
            astropy.constants.h * self.nu * self.data["A"] * u.second**-1 * Nj
        )  # / (4 * np.pi * u.sr)

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

    def h2_mass(self, ref_flux=1.0e-20 * u.erg / u.second / u.cm**2, z=1.01):
        """ """
        from astropy.cosmology import WMAP9

        dL = WMAP9.luminosity_distance(z)  # .to(u.cm)

        Ntot = (
            4
            * np.pi
            * dL**2
            * ref_flux  # / self.flux["flux"][self.ix]
            # * self.flux["flux"][self.ix]
            / self.h_nu_A[self.ix]
        )

        Mtot = (Ntot * 2 * astropy.constants.m_p).to(u.Msun)
