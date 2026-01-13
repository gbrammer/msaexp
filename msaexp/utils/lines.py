"""
Tools for working with emission lines
"""

import os

import numpy as np

import astropy.constants
import astropy.units as u

from grizli import utils

from .general import module_data_path

__all__ = ["LineList", "MolecularHydrogen", "line_flam_to_fnu"]

def arabic_to_roman(label, count=-1):
    """
    Convert string with an arabic number to Roman numeral, e.g., "Ca2" -> "CaII"
    """
    roman = ['','I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV','XV']
    out = label + ''
    for a in range(1,len(roman)):
        out = out.replace(f"{a}", roman[a], count)

    return out


def line_flam_to_fnu(
    wave=2.12 * (1 + 1.0132),
    flam_unit=1.0e-20 * u.erg / u.second / u.cm**2 / u.Angstrom,
):
    """
    Convert from flambda cgs to microJansky
    """

    to_fnu = (
        (1 * flam_unit)
        .to(u.microJansky, equivalencies=u.spectral_density(wave * u.micron))
        .value
    )

    return to_fnu


def axis_renorm_x(ax, value):
    """Transform value to normalized coordinate"""
    if ax.xaxis.get_scale() == "log":
        return np.interp(np.log(value), np.log(ax.get_xlim()), [0,1])
    else:
        return np.interp(value, ax.get_xlim(), [0,1])


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

    def __init__(self, data=None):
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
        if data is None:
            self.load_data()
        else:
            self.data = data

    def load_data(self, fill_color="0.2"):
        """
        Load the data from ``msaexp/data/linelist.ecsv``
        """
        linelist_file = os.path.join(module_data_path(), "linelist.ecsv")

        self.data = utils.read_catalog(linelist_file, format="ascii.ecsv")
        self.data["color"] = self.data["color"].filled(fill_color)
        #self.N = len(self.data)

    @staticmethod
    def from_table(tab, wave_column="wavelength", to_angstroms=1.0, color="0.5", priority=5, prefix="", ratio_column=None):
        """
        Generate from normal table with a list of line wavelengths
        """
        data = utils.GTable()
        for c in tab.colnames:
            data[c] = tab[c]

        data["wavelength"] = [[w*to_angstroms] for w in tab[wave_column]]

        if ratio_column is None:
            data["ratio"] = [[1.0] for w in tab[wave_column]]
        else:
            data["ratio"] = [[r] for r in tab[ratio_column]]

        if "color" not in data.colnames:
            data["color"] = color

        if "name" not in data.colnames:
            data["name"] = [
                f"{prefix}{w*to_angstroms:.0f}" for w in tab[wave_column]
            ]

        if "label" not in data.colnames:
            data["label"] = data["name"]

        if "priority" not in data.colnames:
            data["priority"] = priority

        return LineList(data=data)

    @property
    def N(self):
        return len(self.data)

    def select_lines(
        self,
        z=0,
        wave_range=[0, np.inf],
        prefixes=[],
        exclude_prefixes=[],
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

            if len(exclude_prefixes) > 0:
                in_excluded = False
                for prefix in exclude_prefixes:
                    if row["name"].startswith(prefix):
                        in_excluded = True
                        break

                selected[i] &= ~in_excluded

        return selected, x_scale

    def add_to_axis(
        self,
        ax,
        wave_pixels=None,
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

        if wave_pixels is not None:
            wpix = np.arange(len(wave_pixels))
            xlim = np.interp(xlim, wpix, wave_pixels)

        if label_priorities is None:
            label_priorities = priorities

        label_priorities_str = ' '.join([f'{p:.1f}' for p in label_priorities])

        selected, x_scale = self.select_lines(
            wave_range=xlim, priorities=priorities, **kwargs
        )

        if selected.sum() == 0:
            return None

        is_xlog = ax.xaxis.get_scale() == "log"
        ax_yscale = ax.yaxis.get_scale()
        is_ylog = ax_yscale == "log"

        if y_normalized:
            tr = ax.transAxes
            y_normalized = False
        else:
            tr = None

        for row in self.data[selected]:
            wave_i = np.array(row["wavelength"]) * x_scale
            p_i = row["priority"]

            if hasattr(bottom, "upper") & (bottom == "priority"):
                vlo = p_i / 5.
            elif hasattr(bottom, "__len__"):
                vlo = np.interp(wave_i, *bottom)
            else:
                vlo = bottom

            if (y_normalized):
                if is_ylog:
                    vlo = 10 ** np.interp(vlo, [0, 1], np.log10(ylim))
                elif ax_yscale == "asinh":
                    vlo = np.sinh(np.interp(vlo, [0, 1], np.arcsinh(ylim)))
                else:
                    vlo = np.interp(vlo, [0, 1], ylim)
            else:
                vlo = vlo

            if hasattr(top, "upper") & (top == "priority"):
                vhi = p_i / 5.
            elif hasattr(top, "__len__"):
                vhi = np.interp(wave_i, *top)
            else:
                vhi = top

            if y_normalized:
                if is_ylog:
                    vhi = 10 ** np.interp(vhi, [0, 1], np.log10(ylim))
                elif ax_yscale == "asinh":
                    vhi = np.sinh(np.interp(vhi, [0, 1], np.arcsinh(ylim)))
                else:
                    vhi = np.interp(vhi, [0, 1], ylim)
            else:
                vhi = vhi

            if wave_pixels is not None:
                x_line = np.interp(wave_i, wave_pixels, wpix)
            else:
                x_line = wave_i

            if tr is not None:
                x_line = axis_renorm_x(ax, x_line)

            ax.vlines(
                x_line,
                vlo,
                vhi,
                color=row["color"],
                ls=ls,
                lw=lw,
                alpha=alpha,
                zorder=zorder,
                transform=tr,
            )

            if (y_label is not None) & (f'{p_i:.1f}' in label_priorities_str):
                if len(label_prefixes) > 0:
                    in_prefixes = False
                    for prefix in label_prefixes:
                        if row["name"].startswith(prefix):
                            in_prefixes = True
                            break
                    if not in_prefixes:
                        continue

                wl = wfunc(wave_i)

                if hasattr(y_label, "upper") & (y_label == "priority"):
                    yl = p_i / 5. * 0.95
                elif hasattr(y_label, "__len__"):
                    yl = np.interp(wl, *y_label)
                else:
                    yl = y_label

                if y_normalized:
                    if is_ylog:
                        yl = 10 ** np.interp(yl, [0, 1], np.log10(ylim))
                    elif ax_yscale == "asinh":
                        yl = np.sinh(np.interp(yl, [0, 1], np.arcsinh(ylim)))
                    else:
                        yl = np.interp(yl, [0, 1], ylim)
                else:
                    yl = yl

                if tr is not None:
                    dx = 0
                elif is_xlog:
                    dx = wl * (np.exp(x_pad * np.log(xlim[1] / xlim[0])) - 1)
                else:
                    dx = x_pad * (xlim[1] - xlim[0])

                if wave_pixels is not None:
                    x_text = np.interp(wl + dx, wave_pixels, wpix)
                else:
                    x_text = wl + dx

                if tr is not None:
                    x_text = axis_renorm_x(ax, x_text) + x_pad

                ax.text(
                    x_text,
                    yl,
                    row["label"],
                    color=row["color"],
                    va=va,
                    ha=ha,
                    alpha=alpha,
                    fontsize=fontsize,
                    bbox=bbox,
                    zorder=label_zorder,
                    rotation=rotation,
                    transform=tr,
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

    def spec_model(
        self, spec, z=0, selected=None, fnu=True, use_ratios=True, **kwargs
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

        use_ratios : bool
            Use line ratio values.  If False, normalize line flux to unity.

        kwargs : dict
            Keyword arguments passed to `LineList.select_lines` and
            `msaexp.spectrum.SpectrumSampler.fast_emission_line`

        Returns
        -------
        result : dict
            Model results:

            * ``templates``: ``(Nline, Nwave)`` array. of line templates

            * ``names``: line names

            * ``index``: indices of the selected lines from the full linelist

        """

        if selected is None:
            valid_wave = spec["wave"][spec["valid"]]
            wave_range = [valid_wave.min(), valid_wave.max()]
            selected, x_scale = self.select_lines(
                wave_range=wave_range, z=z, **kwargs
            )

        if selected.sum() == 0:
            return np.zeros((1, spec.NSPEC)), ["empty"]

        line_names = []
        line_templates = []
        for i, row in enumerate(self.data[selected]):
            line_i = np.zeros_like(spec["flux"])
            for lw, lr in zip(row["wavelength"], row["ratio"]):

                if use_ratios:
                    lri = lr
                else:
                    lri = 1.0

                line_i += spec.fast_emission_line(
                    lw * (1 + z) / 1.0e4,
                    line_flux=lri,
                    **kwargs,
                )

            line_templates.append(line_i)
            line_names.append(row["name"])

        line_templates = np.array(line_templates)

        if fnu:
            line_templates *= 1.0e-4 / spec["to_flam"]

        result = {
            "templates": line_templates,
            "names": line_names,
            "index": np.where(selected)[0],
        }

        return result


def get_atom_colors(my_colors={}):
    """
    """
    from matplotlib.colors import to_rgb, CSS4_COLORS
    colors = {}

    # S
    c_i = np.array(to_rgb(CSS4_COLORS["gold"]))
    colors["S2"] = c_i * 0.6
    colors["S3"] = c_i * 0.8
    colors["S4"] = c_i * 0.9

    c_i = np.array(to_rgb(CSS4_COLORS["darkorchid"]))
    colors["Ar3"] =  c_i
    for i in range(7):
        colors[f"Ar{i+1}"] = c_i * np.interp(i, [0, 6], [0.6, 0.9])

    # C
    # for i in range(4):
    #     colors[f"C{i+1}"] = np.ones(3) * np.interp(i, [0, 3], [0.3, 0.7])

    c_i = np.array(to_rgb(CSS4_COLORS["goldenrod"]))
    for i in range(4):
        colors[f"C{i+1}"] = c_i * 0.7

    # colors["Fe2"] = np.array([63.14, 61.57, 58.04])/100.
    colors["Fe2"] = np.array(to_rgb(CSS4_COLORS["saddlebrown"]))
    for i in range(2, 11):
        colors[f"Fe{i+1}"] = colors["Fe2"] * np.interp(i, [2, 10], [0.5, 0.8])

    # O
    c_i = np.array(to_rgb(CSS4_COLORS["yellowgreen"]))
    for i in range(5):
        colors[f"O{i+1}"] = np.array(c_i) * np.interp(i, [0, 5], [0.6, 0.9])

    for i in range(5, 7):
        colors[f"O{i+1}"] = colors["O3"]

    # Ne
    c_i = np.array(to_rgb(CSS4_COLORS["orange"]))
    for i in range(2, 7):
        colors[f"Ne{i}"] = np.array(c_i) * np.interp(i, [2, 6], [0.6, 0.9])

    # Mg
    c_i = np.array(to_rgb(CSS4_COLORS["thistle"]))
    for i in range(1, 9):
        colors[f"Mg{i}"] = c_i * 0.7

    # N
    c_i = np.array(to_rgb(CSS4_COLORS["steelblue"]))
    for i in range(1, 5):
        colors[f"N{i}"] = np.array(c_i) * np.interp(i, [1, 5], [0.6, 0.9])

    # Si
    c_i = np.array(to_rgb(CSS4_COLORS["green"]))
    for i in range(1, 10):
        colors[f"Si{i}"] = np.array(c_i) * c_i * (0.8 - 0.2 * (i % 2))

    c_i = np.array(to_rgb(CSS4_COLORS["orangered"]))
    for i in range(1, 10):
        colors[f"Ca{i}"] = np.array(c_i) * c_i * (0.8 - 0.2 * (i % 2))

    c_i = np.array(to_rgb(CSS4_COLORS["yellowgreen"]))
    for i in range(1, 10):
        colors[f"Cl{i}"] = np.array(c_i) * c_i * (0.8 - 0.2 * (i % 2))

    # Al, Ni
    c_i = np.array(to_rgb(CSS4_COLORS["slategray"]))
    for i in range(10):
        colors[f"Ca{i+1}"] = c_i
        colors[f"Al{i+1}"] = c_i * 0.8
        colors[f"Ni{i+1}"] = c_i * 0.6
        colors[f"Na{i+1}"] = c_i * 0.6
        colors[f"K{i+1}"] = c_i * 0.5

    for i in range(1, 11):
        colors[f"P{i}"] = np.array(to_rgb(CSS4_COLORS["peru"]))

    c_i = np.array(to_rgb(CSS4_COLORS["tomato"]))

    for i, v in enumerate(['H','Pa','Br','Pf','Hu']):
        colors["H1_" + v] = c_i * (0.8 - 0.3 * (i % 2))

    c_i = np.array(to_rgb(CSS4_COLORS["teal"]))
    colors["He1"] = c_i * 0.8
    colors["He2"] = c_i * 0.95


    for c in my_colors:
        if hasattr(my_colors[c], 'upper'):
            colors[c] = to_rgb(CSS4_COLORS[my_colors[c]])
        else:
            colors[c] = my_colors[c]

    return colors


class FullLineList(LineList):
    column = None
    threshold = 1.e-3
    with_fe2 = False
    with_extra = True
    emis_columns = []
    merged_lines = {}

    def __init__(self, use_pumped_iron=True, **kwargs):
        """
        Use a linelist derived from PyNeb with ionic line ratios as a function of temperature
        and density
        """
        import astropy.table

        self.pyneb = utils.read_catalog(
            os.path.join(module_data_path(), "lines_pyneb_emissivity.rst"),
            format="ascii.rst"
        )

        self.available_columns = []
        for c in self.pyneb.colnames:
            if c.startswith('t'):
                if (c[4] == "n"):
                    self.available_columns.append(c)

        if use_pumped_iron:
            fe2 = utils.read_catalog(
                os.path.join(module_data_path(), "lines_pumped_iron.rst"),
                format="ascii.rst"
            )
            for c in self.available_columns:
                fe2[c] = fe2["flux"]

            fe2.remove_column("flux")

            pop = self.pyneb["atom"] == "Fe2"

            self.pyneb = astropy.table.vstack([self.pyneb[~pop], fe2])

        self.extra = utils.read_catalog(
            os.path.join(module_data_path(), "lines_supplemental.rst"),
            format="ascii.rst"
        )

        self.pyneb["pyneb"] = True
        self.extra["pyneb"] = False

        self.set_colors(**kwargs)
        self.set_linelist(**kwargs)


    def set_colors(self, my_colors={}, default=(0.2, 0.2, 0.2), **kwargs):
        """
        Set colors for line labels
        """
        from matplotlib.colors import to_rgb, CSS4_COLORS

        if hasattr(default, "upper"):
            self.pyneb["color"] = [to_rgb(CSS4_COLORS[default])] * len(self.pyneb)
            self.extra["color"] = [to_rgb(CSS4_COLORS[default])] * len(self.extra)
        else:
            self.pyneb["color"] = [default] * len(self.pyneb)
            self.extra["color"] = [default] * len(self.extra)

        colors = get_atom_colors()

        for t in [self.pyneb, self.extra]:
            ind_h1 = np.where(t["atom"] == "H1")[0]

            for c in colors:
                if c.startswith("H1_"):
                    for j in ind_h1:
                        prefix = f"{t['atom'][j]}_{t['name'][j]}"
                        if prefix.startswith(c):
                            t["color"][j] = colors[c]
                else:
                    test = t["atom"] == c
                    if test.sum() > 0:
                        for j in np.where(test)[0]:
                            t["color"][j] = colors[c]

    def calculate_merged_lines(self, threshold=0.01, skip_atoms=["H1", "Fe2"], verbose=True):
        """
        Look for lines with ratios insensitive to temperature / density
        """
        line_grid = np.array([self.pyneb[c] for c in self.available_columns])
        merged_lines = {}

        una = utils.Unique(self.pyneb["atom"], verbose=False)

        for ia, atom in enumerate(una.values):
            if atom in skip_atoms:
                continue

            Na = una.counts[ia]
            if Na > 0:
                ind = np.where(una[atom])[0]
                lr = line_grid[:, ind]
                lrm = lr.max(axis=0)
                so = np.argsort(lrm)[::-1]

                ks = []

                for si in range(Na-1):
                    for sj in range(si+1, Na):
                        i = so[si]
                        j = so[sj]

                        if lrm[j] < 8e-3:
                            continue

                        lr_ij = lr[:, j] / lr[:, i]
                        ok_ij = np.isfinite(lr_ij)
                        std = np.std(lr_ij[ok_ij])
                        med = np.median(lr_ij[ok_ij])

                        ki = ind[i]
                        kj = ind[j]

                        if std / med < threshold:
                            if ki in merged_lines:
                                merged_lines[ki].append(kj)
                            else:
                                used = False
                                for k in ks:
                                    if ki in merged_lines[k]:
                                        merged_lines[k].append(kj)
                                        used = True

                                if not used:
                                    merged_lines[ki] = [kj]
                                    ks.append(ki)

            # Print a summary
            if (len(ks) > 0) & verbose:
                print(f"\nAtom: {atom}")
                for k in ks:
                    print(f"  {self.pyneb['name'][k]}")
                    for kj in merged_lines[k]:
                        lr_ij = line_grid[:, kj] / line_grid[:, k]
                        lr_ijs = " ".join([f"{v:6.3f}" for v in lr_ij])

                        print(f"      {self.pyneb['name'][kj]}: {lr_ijs}")

        return merged_lines

    def set_linelist(self, column="t1.0n3", threshold=1.e-3, with_fe2=False, with_extra=True, **kwargs):
        """
        """
        import astropy.table

        keep = self.pyneb["wavelength"] > 0
        if column is not None:
            keep &= self.pyneb[column] > threshold
        if not with_fe2:
            keep &= ~(self.pyneb["atom"] == "Fe2")

        self.column = column
        self.threshold = threshold
        self.with_fe2 = with_fe2
        self.with_extra = with_extra

        pn_data_ = LineList.from_table(
            self.pyneb[keep],
            ratio_column=column,
            to_angstroms=1.e4
        ).data

        if with_extra:
            data_ = [
                pn_data_,
                LineList.from_table(self.extra, to_angstroms=1.e4).data
            ]
            data_ = astropy.table.vstack(data_)
        else:
            data_ = pn_data_

        # Merged lines
        if column is not None:
            kfull = np.zeros(len(self.pyneb), dtype=int) - 1
            kfull[keep] = np.arange(keep.sum(), dtype=int)

            lw = [w.tolist() for w in data_["wavelength"]]
            lr = [r.tolist() for r in data_["ratio"]]

            pop = np.zeros(len(data_), dtype=bool)
            merged_k = []

            for k in self.merged_lines:
                ks = np.array([k] + self.merged_lines[k])
                keep_k = keep[ks]
                if keep_k.sum() < 2:
                    continue

                ks = ks[keep_k]
                lr_k = self.pyneb[column][ks]
                so = np.argsort(lr_k)[::-1]

                kf = kfull[ks[so]].tolist()

                data_["priority"][kf[0]] = self.pyneb["priority"][ks].max()
                for kf_i in kf[1:]:
                    lw[kf[0]] += lw[kf_i]
                    lr[kf[0]] += lr[kf_i]

                merged_k.append(kf[0])

                pop[kf[1:]] = True

            data_["wavelength"] = lw
            data_["ratio"] = lr
            data_["multiple"] = False
            data_["multiple"][merged_k] = True

            data_ = data_[~pop]

        self.data = data_


    def atom_spec_model(self, spec, **kwargs):
        """
        """
        kwargs["use_ratios"] = True

        res = self.spec_model(spec, **kwargs)

        pyn = self.data["pyneb"][res["index"]]

        if pyn.sum() == 0:
            return res

        ind_pyn = np.where(pyn)[0]

        una = utils.Unique(self.data["atom"][res["index"]][pyn], verbose=False)

        templates = []
        names = []
        index = []

        for atom in una.values:
            for i, j in enumerate(ind_pyn[una[atom]]):
                if i == 0:
                    model_j = res["templates"][j,:]
                    index.append(j)
                else:
                    model_j += res["templates"][j,:]

            templates.append(model_j)
            names.append(f"{atom}_series")

        extra = ~pyn
        if extra.sum() > 0:
            for j in np.where(extra)[0]:
                templates.append(res["templates"][j,:])
                names.append(res["names"][j])
                index.append(res["index"][j])

        result = {
            "templates": np.array(templates),
            "names": names,
            "index": index,
            "color": self.data["color"][index],
        }

        return result


def init_pumped_fe2():
    """
    Process the Fe2 table from Sigut & Pradhan 2003

    https://iopscience.iop.org/article/10.1086/345498/fulltext/

    """
    import pyneb as pn
    
    fe2 = utils.read_catalog(
        "https://iopscience.iop.org/article/10.1086/345498/fulltext/datafile3.txt?doi=10.1086/345498", format="cds"
    )
    fe2["Flux"] = fe2["Flux"].astype(float) / fe2["Flux"].max()
    fe2["atom"] = "Fe2"
    fe2["wavelength"] = pn.utils.physics.airtovac(fe2["Wave"]) / 1.e4
    fe2["name"] = [f"Fe2_{w*1.e4:.0f}" for w in fe2["wavelength"]]
    fe2["label"] = [r"FeII$_{~xx}$".replace("xx", f"{w*1.e4:.0f}") for w in fe2["wavelength"]]
    fe2["priority"] = 5 + np.round(np.maximum(np.log10(fe2["Flux"]), -4)).astype(int)

    fe2.remove_column("Wave")
    fe2.rename_column("Flux","flux")

    fe2["wavelength"].format = ".7f"
    fe2["flux"].format = ".3f"

    fe2.write("/tmp/lines_pumped_iron.rst", format="ascii.rst", overwrite=True)

class MolecularHydrogen:

    # Reference transition
    reference = ("1-0", "S(1)")

    # Do Z(T) sum separate by odd / even j
    separate_ZT = False

    def __init__(self, **kwargs):
        """
        Tools for working with molecular hydrogen lines using the helpful
        relations from Aditya Togi and J. D. T. Smith 2016 ApJ, 830, 18

        Optically thin flux:

        .. math::

            F_j = h \\nu A N_{j+2} \\Omega / 4\\pi

        Line flux F as a function of temperature, T (excitation diagram):

        .. math::

            N_{j+2} = \\frac{g_{j+2}}{Z(T)} e^{-E_\mathrm{upper} / kT}

            Z(T) = \\sum g_j e^{-E_j / kT}

            F_j \\propto \\frac{N_{j+2} A}{\\lambda}

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
        self.data["A"].unit = 1.0e-7 * u.second**-1
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
        precompute :math:`h \\cdot \\nu \\cdot A` (erg / second)
        """
        return (astropy.constants.h * self.nu * self.data["A"]).to(
            u.erg / u.second
        )

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
        Nt = tgrid**-n * (n - 1) / (Tl ** (1 - n) - Tu ** (1 - n))
        return tgrid, Nt

    def ZT(self, T=1000.0):
        """
        Compute :math:`Z(T) = \\sum g_j e^{-E_j / kT}`
        """
        if self.separate_ZT:
            # Seems like it should be this
            ZT = np.zeros_like(self.data["wave"])
            for i in [0, 1]:
                sub = self.data["j"] % 2 == i
                ZT[sub] = np.sum(
                    (self.data["gJ"] * np.exp(-self.data["Eupper"] / T))[sub]
                )
        else:
            # But this needed to make the ratios agree with the gemini table
            ZT = np.sum((self.data["gJ"] * np.exp(-self.data["Eupper"] / T)))

        return ZT

    def Nj(self, T=1000.0):
        """
        Compute number density: :math:`N_j = \\frac{g_j}{Z(T)}~e^{-E_j / kT}`
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
        Line flux :math:`F_j = h \\nu A N_{j+2} \\Omega / 4\\pi`
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

    def to_linelist(self):
        """
        Reformat into a `LineList` object
        """
        rows = []
        for j in np.where(self.flux["mask"])[0]:
            name = "{vib} {rot}".format(**self.data[j])
            rows.append({
                "priority": 5 if "(1)" in name else 2.1,
                "name": name,
                "label": self.flux["name"][j],
                "wavelength": [self.flux["wave"][j] * 1.e4],
                "ratio": [self.flux["flux"][j] / self.flux["flux"][self.ix]],
                "color": "brown",
            })

        return LineList(data=utils.GTable(rows=rows))

    def spec_model(
        self,
        spec,
        z=0.0,
        update_flux=True,
        wave_range=None,
        fnu=True,
        single=True,
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

        if single:
            model = np.zeros_like(spec["flux"])
        else:
            models = []

        if fnu:
            scale = 1.0e-4 / spec["to_flam"]
        else:
            scale = 1.0

        for row in self.flux[self.flux["mask"]]:

            if single:
                flux_i = row["flux"] / self.flux.meta["ref_flux"]
            else:
                flux_i = 1.0

            model_i = spec.fast_emission_line(
                row["wave"] * (1 + z),
                line_flux=flux_i,
                **kwargs,
            )

            if single:
                model += model_i
            else:
                models.append(model_i)

        if single:
            return model * scale
        else:
            return np.array(models) * scale

    def h2_mass(
        self,
        line_flux=1.0e-20 * u.erg / u.second / u.cm**2,
        transition=("1-0", "S(1)"),
        z=1.01,
        **kwargs,
    ):
        """
        Still figuring out unit conversions....
        """
        from astropy.cosmology import WMAP9

        dL = WMAP9.luminosity_distance(z).to(u.cm)

        # tgrid, Nt = self.temperature_powerlaw(n=n, Tl=Tl, Tu=Tu, nsteps=512)

        # Nj = self.Nj(T=(tgrid, Nt))
        Nj = self.flux["Nj"]

        if transition is None:
            ix = self.ix
        else:
            ix = np.where(
                (self.data["vib"] == transition[0])
                & (self.data["rot"] == transition[1])
            )[0][0]

            transition = (
                self.data["vib"][ix],
                self.data["rot"][ix],
            )

        Ntot = line_flux / self.h_nu_A[ix] / Nj[ix] * dL.to(u.cm) ** 2
        Mtot = (Ntot * 2 * astropy.constants.m_p).to(u.Msun)

        result = {
            "line_flux": line_flux,
            "z": z,
            "dL": dL,
            "transition": transition,
            "ix": ix,
            "wave": self.flux["wave"][ix],
            "name": self.line_names()[ix],
            "mass": Mtot,
            "log_mass": np.log10(Mtot.value),
        }

        return result
