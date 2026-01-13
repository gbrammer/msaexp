import numpy as np
import matplotlib.colorizer

class ClippedColormap(matplotlib.colorizer.Colorizer):
    
    max_lightness = 90
    colorbar = None
    N = 0

    def __init__(self, cmap='Spectral', max_lightness=90, norm=None, vmin=None, vmax=None, **kwargs):
        """
        Color map with Lightness clipped to a maximum value

        Parameters
        ----------
        cmap : `matplotlib` colormap
            Colormap object, e.g., `matplotlib.cm.Spectral`.

        max_lightness : float
            Maximum lightness to allow (0 - 100)

        """
        import matplotlib.colorizer
        
        self.input_cmap = cmap
        self._cmap = matplotlib.colorizer.Colorizer(cmap=cmap, norm=norm)
        if (vmin is not None) | (vmax is not None):
            self.set_clim(vmin=vmin, vmax=vmax)

        self.max_lightness = max_lightness


    @staticmethod
    def get_color_lightness(rgba, **kwargs):
        """
        Get color Lightness
        """
        import numpy as np
        from colorspacious import cspace_converter

        arr = np.array(rgba)
        ndim = arr.ndim * 1

        if ndim == 1:
            arr = arr[np.newaxis, np.newaxis, :]
        elif ndim == 2:
            arr = arr[np.newaxis, :, :]

        lab = cspace_converter("sRGB1", "CAM02-UCS")(arr[:, :, :3])
        return lab[:, :, 0]

    @staticmethod
    def clip_color_lightness(rgba, max_lightness=85, **kwargs):
        """
        Clip color to maximum lightness
    
        Conversion from https://matplotlib.org/stable/users/explain/colors/colormaps.html
        
        """
        import numpy as np
        from colorspacious import cspace_converter

        arr = np.array(rgba)
        ndim = arr.ndim * 1

        if ndim == 1:
            arr = arr[np.newaxis, np.newaxis, :]
        elif ndim == 2:
            arr = arr[np.newaxis, :, :]
            
        lab = cspace_converter("sRGB1", "CAM02-UCS")(arr[:, :, :3])
        if max_lightness > 0:
            lab[:, :, 0] = np.minimum(lab[:, :, 0], max_lightness)
        else:
            lab[:, :, 0] = -max_lightness

        # Transform back
        arr[:, :, :3] = cspace_converter("CAM02-UCS", "sRGB1")(lab) #), rgba[-1])
        if ndim == 1:
            arr = arr[0, 0, :]
        elif ndim == 2:
            arr = arr[0, :, :]

        return arr

    def to_rgba(self, x, autoscale=False, vmin=None, vmax=None, **kwargs):
        """
        Override the to_rgba method with clipped lightness values
        """
        self._cmap.colorbar = self.colorbar

        if autoscale:
            self.autoscale(x)
        elif vmin is not None:
            self.set_clim(vmin=vmin, vmax=vmax)

        rgba = self._cmap.to_rgba(x, **kwargs)
        return self.clip_color_lightness(rgba, max_lightness=self.max_lightness)

    def __call__(self, x, **kwargs):
        return self.to_rgba(x, **kwargs)

    @property
    def autoscale(self):
        return self._cmap.autoscale

    @property
    def autoscale_None(self):
        return self._cmap.autoscale_None

    @property
    def callbacks(self):
        return self._cmap.callbacks

    @property
    def changed(self):
        return self._cmap.changed

    @property
    def clip(self):
        return self._cmap.clip

    @property
    def get_clim(self):
        return self._cmap.get_clim

    @property
    def norm(self):
        return self._cmap.norm

    @property
    def set_clim(self):
        return self._cmap.set_clim

    @property
    def stale(self):
        return self._cmap.stale

    @property
    def vmin(self):
        return self._cmap.vmin

    @property
    def vmax(self):
        return self._cmap.vmax

    def __repr__(self):
        """
        """
        return f"ClippedColormap: {self.input_cmap} max_lightness={self.max_lightness}"


def tight_colorbar(
    color_result, fig, ax, sx=0.325, sy=0.03, loc='ul', pad=0.01, location=None,
    label=None, bbox_kwargs=None, zorder=100, label_format=None, labelsize=8,
    colorbar_alpha=None,
    **kwargs
):
    """
    Insert a colorbar at the corners/edges of a plot axis

    Parameters
    ----------
    color_result : object
        Something that can generate a colorbar, e.g., `color_result = ax.scatter(x, y, c=z, vmin=0, vmax=1, cmap='viridis')`.

    fig : `matplotlib.figure.Figure`
        Figure object

    ax : `matplotlib.axes.Axes`
        Axis to define the positioning of the colorbar

    sx, sy : float
        x and y size of the colorbar.  If ``sx < sy``, will draw the colorbar oriented vertically.  The larger of the two inputs is taken
        as the color bar length and has units of the size of ``ax``.  The smaller value is the color bar width and has units of the full
        figure size along that dimension.  For example, the defaults of ``sx = 0.325`` and ``sy = 0.03`` will result in a horizonal colorbar
        about 1/3 of the width of the plot axis and 3% of the full figure size.

    loc : str
        Two-character string specifying the vertical (``u``pper, ``c``enter, ``l``ower) and horizontal
        (``l``eft, ``c``enter, ``r``ight) location of the anchor of the colorbar axis. The default ``ul`` is the "upper left" corner.
        
    pad : float
        Padding to add relative to the ``loc`` position.  

    location : str
        Label location to override the internal calculation ("top", "bottom" , "left", "right").
        
    label : str
        Label of the colorbar

    bbox_kwargs : dict
        Keyword arguments of a box to draw around the colorbar and its associated labels, e.g., to visually raise it over overlapping background
        points in the plot axis.

    zorder : number
        zorder of the colorbar axis added to the figure

    label_format : str
        String formatting of the colorbar tick labels, e.g., "%.1f".

    labelsize : float
        Font size of the colorbar labels

    colorbar_alpha : float
        Transparency to use for the colorbar itself, which can be different than the transparency used for the data plotted in ``color_result``.

    Returns
    -------
    cax : `matplotlib.axes.Axes`
        Colorbar axis

    cb : `matplotlib.colorbar.Colorbar`
        Colorbar object
    
    """
    from matplotlib.patches import Rectangle

    bbox = ax.get_position().bounds
    asp = bbox[2] / bbox[3]

    figh = fig.get_figheight()
    figw = fig.get_figwidth()
    fig_asp = figw / figh
    # asp *= fig_asp

    sxa = sx * bbox[2]
    sya = sy * bbox[3]

    # # Don't understand these, but they make for standard sizes
    # scale_width = (asp * 2)**-0.5 * (fig_asp / 1.6)**0.5 * (np.exp(np.abs(np.log(asp)))/2)**0.5
    # if fig_asp < 1:
    #     scale_width /= fig_asp**2

    if sx > sy:
        orientation = 'horizontal'
        # sya *= asp * fig_asp * scale_width
        sya = sy * fig_asp**0.5
        # sya = sy * bbox[3] * asp * fig_asp
        
    else:
        orientation = 'vertical'
        # sxa *= scale_width
        sxa = sx * fig_asp**-0.5
        # sxa = bbox[2] * sx
        
    padx = bbox[2] * pad
    pady = bbox[3] * pad * asp * fig_asp

    if loc[0] == 'u':
        cax_y = (bbox[1] + bbox[3]) - sya - pady
    elif loc[0] == 'c':
        cax_y = (bbox[1] + bbox[3] * 0.5) - sya / 2
    else:
        cax_y = (bbox[1] + pady)

    if loc[1] == 'r':
        cax_x = (bbox[0] + bbox[2]) - sxa - padx
    elif loc[1] == 'c':
        cax_x = (bbox[0] + bbox[2] * 0.5) - sxa / 2
    else:
        cax_x = (bbox[0] + padx)

    cax_extent = (
        cax_x,
        cax_y,
        sxa,
        sya
    )

    # Label location depending on ``loc`` anchor
    locs = {
        "ul": {"horizontal": "bottom", "vertical": "right"},
        "uc": {"horizontal": "bottom", "vertical": "right"},
        "ur": {"horizontal": "bottom", "vertical": "left"},
        "cl": {"horizontal": "bottom", "vertical": "right"},
        "cc": {"horizontal": "bottom", "vertical": "right"},
        "cr": {"horizontal": "bottom", "vertical": "left"},
        "ll": {"horizontal": "top", "vertical": "right"},
        "lc": {"horizontal": "top", "vertical": "right"},
        "lr": {"horizontal": "top", "vertical": "left"},
    }

    if location is None:
        if loc not in locs:
            msg = f"{loc} is invalid, must be u/c/l + l/c/r, e.g., 'ul'"
            raise ValueError(msg)
                
        location = locs[loc][orientation]
    
    cax = fig.add_axes(
        cax_extent,
        zorder=zorder,
    )

    cax.tick_params(labelsize=labelsize)

    cb = fig.colorbar(
        color_result,
        cax=cax,
        orientation=orientation,
        location=location,
        format=label_format,
        **kwargs
    )
    
    if colorbar_alpha is not None:
        cb.solids.set_alpha(colorbar_alpha)

    if label is not None:
        cl = cb.set_label(label, size=labelsize)

    if bbox_kwargs is not None:
        ext = cax.get_tightbbox()
        ext = ext.transformed(fig.transFigure.inverted())

        if "zorder" not in bbox_kwargs:
            bbox_kwargs["zorder"] = zorder - 1
            
        rect = Rectangle(
            [ext.xmin, ext.ymin], ext.width, ext.height,
            transform=fig.transFigure,
            **bbox_kwargs
        )
        
        fig.patches.append(rect)

    return (cax, cb)


def square_axes(fig, ax, aspect=1):
    """
    Shrink axis to force a specified axis ratio
    """
    figh = fig.get_figheight()
    figw = fig.get_figwidth()
    fig_asp = figw / figh
    bounds = ax.get_position().bounds

    sx = bounds[2]
    sy = bounds[3]

    new_sy = bounds[2] * aspect * fig_asp
    if new_sy < sy:
        ax.set_position((bounds[0], bounds[1], bounds[2], new_sy))
    else:
        ax.set_position((bounds[0], bounds[1], bounds[3] / aspect / fig_asp, bounds[3]))

