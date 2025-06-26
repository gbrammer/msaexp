"""
Helper scripts for dealing with MSA metadata files (``MSAMETFL``)
"""

import os
import numpy as np
import astropy.io.fits as pyfits

__all__ = [
    "regions_from_metafile",
    "regions_from_fits",
    "MSAMetafile",
    "slit_best_source_alias",
]


def pad_msa_metafile(
    metafile,
    pad=0,
    source_ids=None,
    slitlet_ids=None,
    positive_ids=False,
    prefix="src_",
    verbose=True,
    primary_sources=True,
    merge_first=True,
    **kwargs,
):
    """
    Pad a MSAMETFL with dummy slits and trim to a subset of source_ids

    Parameters
    ----------
    metafile : str
        Filename of the MSA metadata file (``MSAMETFL``)

    pad : int
        Padding of dummy slits

    source_ids : list, None
        List of source_id values to keep

    slitlet_ids : list, None
        List of slitlet_id values to keep

    positive_ids : bool
        If no ``source_ids`` provided, generate sources with `source_id > 0`

    prefix : str
        Prefix of new file to create (``prefix + metafile``)

    verbose : bool
        Print verbose output

    primary_sources : bool
        Include only primary sources

    Returns
    -------

    output_file : str
        Filename of new table

    """
    from astropy.table import Table
    import grizli.utils

    msa = MSAMetafile(metafile)

    if merge_first:
        msa.merge_source_ids(verbose=verbose, **kwargs)

    all_ids = np.unique(msa.shutter_table["source_id"])

    if source_ids is None:
        if slitlet_ids is not None:
            six = np.in1d(msa.shutter_table["slitlet_id"], slitlet_ids)

            if six.sum() == 0:
                msg = f"msaexp.utils.pad_msa_metafile: {slitlet_ids} not found"
                msg += " in {metafile} slitlet_id"
                raise ValueError(msg)

            source_ids = np.unique(msa.shutter_table["source_id"][six])

            msg = "msaexp.utils.pad_msa_metafile: Trim slitlet_ids in "
            msg += f"{metafile} to "
            msg += f"{list(slitlet_ids)} (N={len(source_ids)} source_ids)\n"
            grizli.utils.log_comment(
                grizli.utils.LOGFILE, msg, verbose=True, show_date=True
            )

        else:
            if positive_ids:
                source_ids = all_ids[all_ids > 0]
            else:
                source_ids = all_ids[all_ids != 0]

            six = np.in1d(msa.shutter_table["source_id"], source_ids)
            if primary_sources:
                six &= msa.shutter_table["primary_source"] == "Y"

            if six.sum() == 0:
                msg = f"msaexp.utils.pad_msa_metafile: {source_ids} not found"
                msg += f" in {metafile}.  Available ids are {list(all_ids)}"
                raise ValueError(msg)
    else:
        six = np.in1d(msa.shutter_table["source_id"], source_ids)
        if six.sum() == 0:
            msg = f"msaexp.utils.pad_msa_metafile: {source_ids} not found"
            msg += f" in {metafile}.  Available ids are {list(all_ids)}"
            raise ValueError(msg)

    this_source_ids = np.unique(msa.shutter_table["source_id"][six])

    slitlets = np.unique(msa.shutter_table["slitlet_id"][six])
    im = pyfits.open(metafile)

    shut = Table(im["SHUTTER_INFO"].data)
    shut = shut[np.in1d(shut["slitlet_id"], slitlets)]

    # Add a shutter on either side
    row = {}
    for k in shut.colnames:
        row[k] = shut[k][0]

    row["shutter_state"] = "CLOSED"
    row["background"] = "Y"
    row["estimated_source_in_shutter_x"] = np.nan
    row["estimated_source_in_shutter_y"] = np.nan
    row["primary_source"] = "N"

    new_rows = []

    # Add padding
    for src_id in this_source_ids:
        six = shut["source_id"] == src_id

        for mid in np.unique(shut["msa_metadata_id"][six]):
            mix = (shut["msa_metadata_id"] == mid) & (six)
            quad = shut["shutter_quadrant"][mix][0]
            slit_id = shut["slitlet_id"][mix][0]
            slix = shut["msa_metadata_id"] == mid
            slix &= shut["slitlet_id"] == slit_id

            shutters = np.unique(shut["shutter_column"][slix])
            for eid in np.unique(shut["dither_point_index"][slix]):
                for p in range(pad):
                    for s in [
                        shutters.min() - (p + 1),
                        shutters.max() + (p + 1),
                    ]:

                        row["msa_metadata_id"] = mid
                        row["dither_point_index"] = eid
                        row["shutter_column"] = s
                        row["shutter_quadrant"] = quad
                        row["slitlet_id"] = slit_id
                        row["source_id"] = src_id

                        new_row = {}
                        for k in row:
                            new_row[k] = row[k]

                        new_rows.append(new_row)

    for row in new_rows:
        shut.add_row(row)

    src = Table(im["SOURCE_INFO"].data)
    src = src[np.in1d(src["source_id"], source_ids)]

    hdus = {
        "SHUTTER_INFO": pyfits.BinTableHDU(shut),
        "SOURCE_INFO": pyfits.BinTableHDU(src),
    }

    for e in hdus:
        for k in im[e].header:
            if k not in hdus[e].header:
                # print(e, k, im[e].header[k])
                hdus[e].header[k] = im[e].header[k]
        im[e] = hdus[e]

    output_file = prefix + os.path.basename(metafile)

    im.writeto(output_file, overwrite=True)
    im.close()

    if verbose:
        msg = (
            f"msaexp.utils.pad_msa_metafile: Trim source_id in {metafile} to "
        )
        msg += f"{list(this_source_ids)}\n"
        msg += f"msaexp.utils.pad_msa_metafile: pad = {pad}"
        grizli.utils.log_comment(
            grizli.utils.LOGFILE, msg, verbose=True, show_date=True
        )

    return output_file


def get_msa_source_ids(rate_file, merge_first=True):
    """
    Get source_ids from the MSAMETFL associated with an exposure

    Parameters
    ----------
    rate_file : str
        Exposure filename

    merge_first : bool
        Run `~msaexp.msa.MSAMetafile.merge_source_ids` before computing unique source
        ids.

    Returns
    -------
    source_ids : list
        List of source ids in the mask configuration

    """

    with pyfits.open(rate_file) as im:
        if im[0].header["EXP_TYPE"] != "NRS_MSASPEC":
            return []

        metfl = im[0].header["MSAMETFL"]
        msa = MSAMetafile(metfl)
        if merge_first:
            msa.merge_source_ids()

        conf = im[0].header["MSACONID"]
        msaid = im[0].header["MSAMETID"]
        expid = im[0].header["PATT_NUM"]

    shu = msa.shutter_table

    rows = shu["msa_metadata_id"] == msaid
    rows &= shu["dither_point_index"] == expid

    source_ids = np.unique(shu[rows]["source_id"]).tolist()

    return source_ids


def regions_from_metafile(metafile, **kwargs):
    """
    Wrapper around `msaexp.msa.MSAMetafile.regions_from_metafile`

    Parameters
    ----------
    metafile : str
        Name of a MSAMETFL metadata file

    kwargs : dict
        Keyword arguments are passed through to
        `~msaexp.msa.MSAMetafile.regions_from_metafile`.

    Returns
    -------
    regions : str, list
        Output from `~msaexp.msa.MSAMetafile.regions_from_metafile`

    """
    metf = MSAMetafile(metafile)
    regions = metf.regions_from_metafile(**kwargs)

    return regions


def regions_from_fits(file, **kwargs):
    """
    Wrapper around `msaexp.msa.MSAMetafile.regions_from_metafile`

    Parameters
    ----------
    file : str
        Exposure filename, e.g., `..._rate.fits`.  The `dither_point_index` and
        `msa_metadata_id` will be determined from the file header

    kwargs : dict
        Keyword arguments are passed through to
        `~msaexp.msa.MSAMetafile.regions_from_metafile`.

    Returns
    -------
    regions : str, list
        Output from `~msaexp.msa.MSAMetafile.regions_from_metafile`

    """
    with pyfits.open(file) as im:
        metafile = im[0].header["MSAMETFL"]
        metaid = im[0].header["MSAMETID"]
        dither_point = im[0].header["PATT_NUM"]

    metf = MSAMetafile(metafile)
    regions = metf.regions_from_metafile(
        msa_metadata_id=metaid, dither_point_index=dither_point, **kwargs
    )
    return regions


class MSAMetafile:

    query_product_level = ["2b"]

    def __init__(self, filename):
        """
        Helper for parsing MSAMETFL metadata files

        Parameters
        ----------
        filename : str
            Filename of an `_msa.fits` metadata file or a FITS file with a
            keyword `MSAMETFL` in the primary header, e.g., a `_rate.fits`
            file.

        Attributes
        ----------
        filename : str
            Input filename

        metafile : str
            Filename of the MSAMETFL, either ``filename`` itself or derived
            from it

        shutter_table : `~astropy.table.Table`
            Table of shutter metadata

        src_table : `~astropy.table.Table`
            Table of source information

        mast : `~astropy.table.Table`, None
            Result of `~msaexp.msa.MSAMetafile.query_mast_exposures`

        Examples
        --------

        .. plot::
            :include-source:

            ### Make a plot with slitlets

            import numpy as np
            import matplotlib.pyplot as plt
            from msaexp import msa

            uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
            meta = msa.MSAMetafile(uri+'jw02756001001_01_msa.fits')

            fig, axes = plt.subplots(1, 3, figsize=(9, 2.6),
                                     sharex=True, sharey=True)
            cosd = np.cos(np.median(meta.src_table['dec'])/180*np.pi)

            # Show offset slitlets from three dithered exposures
            for i in [0,1,2]:
                ax = axes[i]
                ax.scatter(meta.src_table['ra'], meta.src_table['dec'],
                           marker='.', color='k', alpha=0.5)
                slits = meta.regions_from_metafile(dither_point_index=i+1,
                                                   as_string=False,
                                                   with_bars=True)
                for s in slits:
                    if s.meta['is_source']:
                        if s.meta['source_id'] in [110003, 410044, 410045]:
                            ax.text(s.meta['ra'] - 0.8/3600, s.meta['dec'],
                                    s.meta['source_id'],
                                    fontsize=7, ha='left', va='center')
                        fc = '0.5'
                    else:
                        fc = 'pink'

                    for patch in s.get_patch(fc=fc, ec='None', alpha=0.8,
                                             zorder=100):
                        ax.add_patch(patch)

                ax.set_aspect(1./cosd)
                ax.set_xlim(3.5936537138517317, 3.588363444812261)
                ax.set_ylim(-30.39750646306242, -30.394291511397544)

                ax.grid()
                ax.set_title(f'Dither point #{i+1}')

            x0 = np.mean(ax.get_xlim())
            ax.set_xticks(np.array([-5, 0, 5])/3600./cosd + x0)
            ax.set_xticklabels(['+5"', 'R.A.', '-5"'])

            y0 = np.mean(ax.get_ylim())
            ax.set_yticks(np.array([-5, 0, 5])/3600. + y0)
            axes[0].set_yticklabels(['-5"', 'Dec.', '+5"'])
            axes[1].scatter(x0, y0, marker='x', c='b')
            axes[1].text(0.5, 0.45, f'({x0:.6f}, {y0:.6f})',
                         ha='left', va='top',
                         transform=axes[1].transAxes, fontsize=6,
                         color='b')

            fig.tight_layout(pad=0.5)


        """
        from astropy.table import Table

        self.filename = filename

        if filename.endswith("_msa.fits"):
            self.metafile = filename
        else:
            with pyfits.open(filename) as _im:
                if "MSAMETFL" not in _im[0].header:
                    raise ValueError(
                        f"{filename}[0].header does not have MSAMETFL keyword"
                    )
                else:
                    self.metafile = _im[0].header["MSAMETFL"]

        with pyfits.open(self.metafile) as im:
            src = Table(im["SOURCE_INFO"].data)
            shut = Table(im["SHUTTER_INFO"].data)

        # Merge src and shutter tables
        shut_ix, src_ix = [], []

        for i, sid in enumerate(shut["source_id"]):
            if sid == 0:
                continue
            elif sid in src["source_id"]:
                shut_ix.append(i)
                src_ix.append(np.where(src["source_id"] == sid)[0][0])

        for c in ["ra", "dec", "program"]:
            shut[c] = src[c][0] * 0
            shut[c][shut_ix] = src[c][src_ix]

        self.shutter_table = shut
        self.src_table = src
        self.mast = None

    @property
    def metadata_id_list(self):
        """
        Returns
        -------
        ids : list
            list of metadata_id from ``shutter_table``

        """
        ids = list(np.unique(self.shutter_table["msa_metadata_id"]))
        return ids

    @property
    def metadata_id_unique(self):
        """
        Returns
        -------
        un : `grizli.utils.Unique`
             Unique of metadata_id from ``shutter_table``
        """
        import grizli.utils

        return grizli.utils.Unique(self.shutter_table["msa_metadata_id"])

    @property
    def key_pairs(self):
        """
        List of unique ``msa_metadata_id, dither_point_index`` pairs
        from the ``shutter_table``

        Returns
        -------
        keys : list
            List of key pairs
        """
        keys = []
        for row in self.shutter_table:
            key = row["msa_metadata_id"], row["dither_point_index"]
            if key not in keys:
                keys.append(key)

        return keys

    @property
    def mast_key_pairs(self):
        """
        List of unique ``msametid, exposure`` pairs from the ``mast``
        metadata table

        Returns
        -------
        keys : list
            List of key pairs
        """
        mast = self.query_mast_exposures(force=False)

        keys = []
        for row in mast:
            key = row["msametid"], row["patt_num"]
            if key not in keys:
                keys.append(key)

        return keys

    def merge_source_ids(self, sort_count=-1, verbose=True, **kwargs):
        """
        Merge cases where a particular slitlet_id has multiple source_ids

        Parameters
        ----------
        sort_count : int
            Direction in which to prefer among the multiple ``source_id`` values
            sorted by their count.

        Returns
        -------
        ``shutter_table`` attribute updated in place

        """
        from grizli import utils

        shu = self.shutter_table
        conf_ids = np.unique(shu["msa_metadata_id"])

        for msaid in conf_ids:
            rows = shu["msa_metadata_id"] == msaid
            row_ix = np.where(rows)[0]

            slits = utils.Unique(shu[rows]["slitlet_id"], verbose=False)

            rename_slits = {}

            for sl in slits.values:
                nexp = len(
                    np.unique(shu[rows][slits[sl]]["dither_point_index"])
                )
                nrows = slits[sl].sum()

                source_ids = utils.Unique(
                    shu[rows][slits[sl]]["source_id"], verbose=False
                )
                test = np.array(source_ids.values) > 0
                # All individual exposures or all rows
                test &= (source_ids.counts != nexp) & (
                    source_ids.counts != nrows
                )

                if test.sum() > 0:
                    if sort_count is not None:
                        so = np.argsort(source_ids.counts)[::sort_count]
                    else:
                        so = range(source_ids.N)

                    id_sorted = []
                    for j in so:
                        if source_ids.values[j] > 0:
                            id_sorted.append(source_ids.values[j])

                    if len(id_sorted) > 0:
                        rename_slits[sl] = id_sorted

            for sl in rename_slits:
                msg = f"{self.filename} {msaid:>3} : "
                msg += (
                    f"merge source_ids for slitlet {sl:>3} {rename_slits[sl]}"
                )
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

                shu["source_id"][row_ix[slits[sl]]] = rename_slits[sl][0]

        return None

    def query_mast_exposures(self, force=False, **kwargs):
        """
        Query MAST database for exposures for this MSA file

        Parameters
        ----------
        force : bool
            Running the query with this method stores the result in the `mast`
            attribute.  If the attribute is `None` or if ``force==True`` then
            run/redo the query.

        Returns
        -------
        mast : `~astropy.table.Table`
            Query results from `mastquery.jwst.query_jwst`.

        """
        from mastquery.jwst import make_query_filter, query_jwst

        if (self.mast is not None) & (not force):
            return self.mast

        filters = []

        filters += make_query_filter(
            "grating",
            values=[
                "PRISM",
                "G140M",
                "G235M",
                "G395M",
                "G140H",
                "G235H",
                "G395H",
            ],
        )
        filters += make_query_filter("effexptm", range=[30, 5.0e5])
        filters += make_query_filter(
            "productLevel", values=self.query_product_level
        )
        filters += make_query_filter(
            "apername",
            values=[
                "NRS_FULL_MSA",
                "NRS_FIELD1_MSA4",
                "NRS_FIELD2_MSA4",
                "NRS_FULL_MSA1",
            ],
        )
        filters += make_query_filter(
            "category", values=["COM", "DD", "ERS", "GTO", "GO", "CAL"]
        )

        filters += make_query_filter("detector", values=["NRS1"])
        filters += make_query_filter(
            "msametfl", values=[os.path.basename(self.filename)]
        )

        mast = query_jwst(
            instrument="NRS",
            filters=filters,
            columns="*",
            rates_and_cals=True,
            extensions=["rate", "cal"],
        )

        self.mast = mast

        return mast

    def get_transforms(
        self,
        dither_point_index=None,
        msa_metadata_id=None,
        fit_degree=2,
        verbose=False,
        **kwargs,
    ):
        """
        Fit for `~astropy.modeling.models.Polynomial2D` transforms between
        slit ``(row, col)`` and ``(ra, dec)``.

        Parameters
        ----------
        dither_point_index : int, None
            Dither index in ``shutter_table``

        msa_metadata_id : int, None
            Metadata id in ``shutter_table``

        fit_degree : int
            Polynomial degree

        verbose : bool
            Print status messages

        Returns
        -------
        dither_match : bool array
            Boolean mask of ``shutter_table`` matching ``dither_point_index``

        meta_match : bool array
            Boolean mask of ``shutter_table`` matching ``msa_metadata_id``

        coeffs : dict
            `~astropy.modeling.models.Polynomial2D` transformations to sky
            coordinates in each of 4 MSA quadrants

            >>> quadrant = 1
            >>> pra, pdec = coeffs[quadrant]
            >>> ra = pra(shutter_row, shutter_column)
            >>> dec = pdec(shutter_row, shutter_column)

        inv_coeffs : dict
            Inverse `~astropy.modeling.models.Polynomial2D` transformations
            from sky to shutters:

            >>> quadrant = 1
            >>> prow, pcol = inv_coeffs[quadrant]
            >>> shutter_row = prow(ra, dec)
            >>> shutter_column = pcol(ra, dec)

        """
        from astropy.modeling.models import Polynomial2D
        from astropy.modeling.fitting import LinearLSQFitter

        p2 = Polynomial2D(degree=fit_degree)
        fitter = LinearLSQFitter()

        _shut = self.shutter_table

        if dither_point_index is None:
            dither_point_index = _shut["dither_point_index"].min()
        if msa_metadata_id is None:
            msa_metadata_id = _shut["msa_metadata_id"].min()

        dither_match = _shut["dither_point_index"] == dither_point_index
        meta_match = _shut["msa_metadata_id"] == msa_metadata_id
        exp = dither_match & meta_match

        has_offset = np.isfinite(_shut["estimated_source_in_shutter_x"])
        has_offset &= np.isfinite(_shut["estimated_source_in_shutter_y"])

        is_src = (_shut["source_id"] > 0) & (has_offset)
        si = _shut[exp & is_src]

        if len(si) == 0:
            is_src = has_offset
            si = _shut[exp & is_src]

        # Fit for transformations
        coeffs = {}
        inv_coeffs = {}

        if verbose:
            output = f"# msametfl = {self.metafile}\n"
            output += f"# dither_point_index = {dither_point_index}\n"
            output += f"# msa_metadata_id = {msa_metadata_id}"
            print(output)

        for qi in np.unique(si["shutter_quadrant"]):
            q = si["shutter_quadrant"] == qi
            # print(qi, q.sum())

            row = si["shutter_row"] + (
                si["estimated_source_in_shutter_x"] - 0.5
            )
            col = si["shutter_column"] + (
                si["estimated_source_in_shutter_y"] - 0.5
            )

            pra = fitter(p2, row[q], col[q], si["ra"][q])
            pdec = fitter(p2, row[q], col[q], si["dec"][q])

            # RMS of the fit
            xra = pra(row[q], col[q])
            xdec = pdec(row[q], col[q])
            dra = (
                (si["ra"][q] - xra)
                * np.cos(si["dec"][q] / 180 * np.pi)
                * 3600
                * 1000
            )
            dde = (si["dec"][q] - xdec) * 3600 * 1000
            pra.rms = np.std(dra)
            pdec.rms = np.std(dde)
            pra.N = q.sum()

            if verbose:
                print(
                    f"# Q{qi} N={q.sum()}  "
                    + f"rms= {pra.rms:.1f}, {pdec.rms:.1f} mas"
                )

            coeffs[qi] = pra, pdec

            prow = fitter(p2, si["ra"][q], si["dec"][q], row[q])
            pcol = fitter(p2, si["ra"][q], si["dec"][q], col[q])
            inv_coeffs[qi] = prow, pcol

        return dither_match, meta_match, coeffs, inv_coeffs

    def regions_from_metafile(self, as_string=False, with_bars=True, **kwargs):
        """
        Get slit footprints in sky coords

        Parameters
        ----------
        as_string : bool
            Return regions as DS9 region strings

        with_bars : bool
            Account for bar vignetting

        kwargs : dict
            Keyword arguments passed to `msaexp.msa.MSAMetafile.get_transforms`

        Returns
        -------
            String or a list of `grizli.utils.SRegion` objects, depending on
            ``as_string``

        Examples
        --------
        .. code-block:: python
            :dedent:

            >>> from msaexp import msa
            >>> uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
            >>> meta = msa.MSAMetafile(uri+'jw02756001001_01_msa.fits')
            >>> regs = meta.regions_from_metafile(as_string=True, with_bars=True)
            >>> print(regs)
            # msametfl = https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/jw02756001001_01_msa.fits
            # dither_point_index = 1
            # msa_metadata_id = 1
            # Q1 N=13  rms= 0.7, 0.4 mas
            # Q2 N=29  rms= 1.3, 1.3 mas
            # Q3 N=11  rms= 1.3, 0.3 mas
            # Q4 N=27  rms= 1.2, 0.7 mas
            icrs
            polygon(3.623046,-30.427251,3.622983,-30.427262,3.622951,-30.427136,3.623014,-30.427125) # color=lightblue
            circle(3.6229814, -30.4270337, 0.2") # color=cyan text={160159}
            polygon(3.623009,-30.427106,3.622946,-30.427117,3.622915,-30.426991,3.622978,-30.426980) # color=cyan
            polygon(3.622973,-30.426960,3.622910,-30.426971,3.622878,-30.426845,3.622941,-30.426834) # color=lightblue
            polygon(3.613902,-30.392060,3.613840,-30.392071,3.613809,-30.391948,3.613871,-30.391936) # color=lightblue
            circle(3.6137989, -30.3918859, 0.2") # color=cyan text={160321}
            polygon(3.613867,-30.391918,3.613804,-30.391929,3.613774,-30.391805,3.613836,-30.391794) # color=cyan
            polygon(3.613831,-30.391775,3.613769,-30.391786,3.613738,-30.391663,3.613800,-30.391652) # color=lightblue
            polygon(3.610960,-30.384123,3.610897,-30.384134,3.610867,-30.384011,3.610929,-30.384000) # color=lightblue
            ...

        """
        import grizli.utils

        dith, metid, coeffs, inv_coeffs = self.get_transforms(**kwargs)

        exp = dith & metid
        _shut = self.shutter_table

        has_offset = np.isfinite(_shut["estimated_source_in_shutter_x"])
        has_offset &= np.isfinite(_shut["estimated_source_in_shutter_y"])

        # Regions for a particular exposure
        se = _shut[exp]

        sx = (np.array([-0.5, 0.5, 0.5, -0.5])) * (
            1 - 0.07 / 0.27 * with_bars / 2
        )
        sy = (np.array([-0.5, -0.5, 0.5, 0.5])) * (
            1 - 0.07 / 0.53 * with_bars / 2
        )

        row = se["shutter_row"]
        col = se["shutter_column"]

        regions = []

        for i in range(len(se)):

            if se["shutter_quadrant"][i] not in coeffs:
                continue

            pra, pdec = coeffs[se["shutter_quadrant"][i]]
            sra = pra(row[i] + sx, col[i] + sy)
            sdec = pdec(row[i] + sx, col[i] + sy)

            sr = grizli.utils.SRegion(np.array([sra, sdec]), wrap=False)
            sr.meta = {}
            for k in [
                "program",
                "source_id",
                "ra",
                "dec",
                "slitlet_id",
                "shutter_quadrant",
                "shutter_row",
                "shutter_column",
                "estimated_source_in_shutter_x",
                "estimated_source_in_shutter_y",
            ]:
                sr.meta[k] = se[k][i]

            sr.meta["is_source"] = np.isfinite(
                se["estimated_source_in_shutter_x"][i]
            )

            if sr.meta["is_source"]:
                sr.ds9_properties = "color=cyan"
            else:
                sr.ds9_properties = "color=lightblue"

            regions.append(sr)

        if as_string:
            output = f"# msametfl = {self.metafile}\n"

            di = _shut["dither_point_index"][dith][0]
            output += f"# dither_point_index = {di}\n"
            mi = _shut["msa_metadata_id"][metid][0]
            output += f"# msa_metadata_id = {mi}\n"

            for qi in coeffs:
                pra, pdec = coeffs[qi]
                output += f"# Q{qi} N={pra.N}  "
                output += f"rms= {pra.rms:.1f}, {pdec.rms:.1f} mas\n"

            output += "icrs\n"
            for sr in regions:
                m = sr.meta
                if m["is_source"]:
                    output += f"circle({m['ra']:.7f}, {m['dec']:.7f}, 0.2\")"
                    output += f" # color=cyan text=xx{m['source_id']}yy\n"

                for r in sr.region:
                    output += r + "\n"

            output = output.replace("xx", "{").replace("yy", "}")

        else:
            output = regions

        return output

    def plot_slitlet(
        self,
        source_id=110003,
        dither_point_index=1,
        msa_metadata_id=None,
        cutout_size=1.5,
        step=None,
        rgb_filters=None,
        rgb_scale=5,
        rgb_invert=False,
        figsize=(4, 4),
        ax=None,
        add_labels=True,
        set_axis_labels=True,
    ):
        """
        Make a plot showing a slitlet

        Parameters
        ----------
        source_id : int
            Source id, must be in ``src_table``

        dither_point_index : int
            Dither to show

        msa_metadata_id : int
            Optional specified ``msa_metadata_id`` in ``shutter_table``

        cutout_size : float
            Cutout half-width, arcsec

        step : int
            Place to mark axis labels, defaults to ``floor(cutout_size)``

        rgb_filters : list, None
            List of filters to use for an RGB cutout.  Will be grayscale if
            just one item specified.

        rgb_scale : float
            Scaling of the image thumbnail if ``rgb_filters`` specified

        rgb_invert : bool
            Invert color map if ``rgb_filters`` specified

        figsize : tuple
            Size if generating a new figure

        ax : `~matplotlib.axes._subplots.AxesSubplot`, None
            Plot axis

        add_labels : bool
            Add plot labels

        set_axis_labels : bool
            Set axis labels

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure object if generating a new figure, None otherwise

        ax : `~matplotlib.axes._subplots.AxesSubplot`
            Plot axes

        Examples
        --------
        .. plot::
            :include-source:

            # Simple figure

            import matplotlib.pyplot as plt
            from msaexp import msa

            uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
            meta = msa.MSAMetafile(uri+'jw02756001001_01_msa.fits')

            fig, ax = plt.subplots(1,1,figsize=(8,8))
            _ = meta.plot_slitlet(source_id=110003, cutout_size=12.5,
                                  rgb_filters=None, ax=ax)

            fig.tight_layout(pad=1.0)
            fig.show()

        .. plot::
            :include-source:

            # With RGB cutout

            import matplotlib.pyplot as plt
            from msaexp import msa

            uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
            meta = msa.MSAMetafile(uri+'jw02756001001_01_msa.fits')

            fig, ax = plt.subplots(1,1,figsize=(4,4))

            filters = ['f200w-clear','f150w-clear','f115w-clear']
            _ = meta.plot_slitlet(source_id=110003, cutout_size=1.5,
                                  rgb_filters=filters, ax=ax)

            fig.tight_layout(pad=1.0)
            fig.show()

        .. plot::
            :include-source:

            # With grayscale cutout

            import matplotlib.pyplot as plt
            from msaexp import msa

            uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
            meta = msa.MSAMetafile(uri+'jw02756001001_01_msa.fits')

            fig, ax = plt.subplots(1,1,figsize=(4,4))

            filters = ['f160w']
            _ = meta.plot_slitlet(source_id=110003, cutout_size=1.5,
                                  rgb_filters=filters, ax=ax, rgb_invert=True)

            fig.tight_layout(pad=1.0)
            fig.show()

        """
        import numpy as np
        import matplotlib.pyplot as plt
        import PIL
        from urllib.request import urlopen

        if (rgb_filters is not None) & (cutout_size > 20):
            raise ValueError('Maximum size of 20" with the thumbnail')

        if source_id not in self.src_table["source_id"]:
            print(f"{source_id} not in src_table for {self.metafile}")
            return None

        ix = np.where(self.src_table["source_id"] == source_id)[0][0]
        ra = self.src_table["ra"][ix]
        dec = self.src_table["dec"][ix]
        cosd = np.cos(dec / 180 * np.pi)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None

        # cutout_size = 2.5 # arcsec
        if rgb_filters is not None:
            url = (
                f"https://grizli-cutout.herokuapp.com/thumb?coords={ra},{dec}"
            )
            url += "&filters=" + ",".join(rgb_filters)
            url += f"&size={cutout_size}&scl={rgb_scale}&invert={rgb_invert}"

            # url = cutout.format(ra=ra, dec=dec, cutout_size=cutout_size)

            if rgb_invert:
                src_color = "k"
            else:
                src_color = "w"

            try:
                rgb = np.array(PIL.Image.open(urlopen(url)))
                # rgb = np.roll(np.roll(rgb, 2, axis=0), -2, axis=1)
                pscale = np.round(2 * cutout_size / rgb.shape[0] / 0.05) * 0.05
                thumb_size = rgb.shape[0] / 2.0 * pscale
                extent = (
                    ra + thumb_size / 3600 / cosd,
                    ra - thumb_size / 3600.0 / cosd,
                    dec - thumb_size / 3600.0,
                    dec + thumb_size / 3600.0,
                )

                ax.imshow(
                    np.flip(rgb, axis=0),
                    origin="lower",
                    extent=extent,
                    interpolation="Nearest",
                )
            except:
                src_color = "k"
        else:
            extent = (
                ra + cutout_size / 3600 / cosd,
                ra - cutout_size / 3600.0 / cosd,
                dec - cutout_size / 3600.0,
                dec + cutout_size / 3600.0,
            )

            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            src_color = "k"

        # ax.set_aspect(cosd)

        ax.scatter(
            self.src_table["ra"],
            self.src_table["dec"],
            marker="o",
            fc="None",
            ec=src_color,
            alpha=0.5,
        )

        slits = self.regions_from_metafile(
            dither_point_index=dither_point_index,
            as_string=False,
            with_bars=True,
            msa_metadata_id=msa_metadata_id,
        )
        for s in slits:
            if s.meta["is_source"]:
                kws = dict(color=src_color, alpha=0.8, zorder=100)
            else:
                kws = dict(color="0.7", alpha=0.8, zorder=100)

            ax.plot(*np.vstack([s.xy[0], s.xy[0][:1, :]]).T, **kws)

        if step is None:
            step = int(np.floor(cutout_size))

        xt = np.array([-step, 0, step]) / 3600.0 / cosd + ra
        yt = np.array([-step, 0, step]) / 3600.0 + dec

        ax.set_xticks(xt)
        ax.set_yticks(yt)

        if set_axis_labels:
            ax.set_yticklabels([f'-{step}"', "Dec.", f'+{step}"'])
            ax.set_xticklabels([f'+{step}"', "R.A.", f'-{step}"'])

        ax.set_xlim(ra + np.array([cutout_size, -cutout_size]) / 3600.0 / cosd)
        ax.set_ylim(dec + np.array([-cutout_size, cutout_size]) / 3600.0)

        ax.set_aspect(1.0 / cosd)
        ax.grid()

        if add_labels:
            ax.text(
                0.03,
                0.07,
                f"Dither #{dither_point_index}",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                color=src_color,
                fontsize=8,
            )
            ax.text(
                0.03,
                0.03,
                f"{os.path.basename(self.metafile)}",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                color=src_color,
                fontsize=8,
            )
            ax.text(
                0.97,
                0.07,
                f"{source_id}",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                color=src_color,
                fontsize=8,
            )
            ax.text(
                0.97,
                0.03,
                f"({ra:.6f}, {dec:.6f})",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                color=src_color,
                fontsize=8,
            )

        if fig is not None:
            fig.tight_layout(pad=1)

        return fig, ax

    def make_summary_table(
        self,
        msa_metadata_id=None,
        image_path="slit_images",
        write_tables=True,
        **kwargs,
    ):
        """
        Make a summary table for all sources in the mask

        Parameters
        ----------
        msa_metadata_id : int, None
            Metadata id in ``shutter_table``

        image_path : str
            Path for slitlet thumbnail images with filename derived from
            `self.metafile`.

        write_tables : bool
            Write FITS and HTML versions of the summary table

        kwargs : dict
            Arguments passed to `~msaexp.msa.MSAMetafile.plot_slitlet` if
            ``image_path`` specified

        Returns
        -------
        tab : `~astropy.table.Table`
            Summary table with slit information.

        Examples
        --------

        .. code-block:: python
            :dedent:

            >>> from msaexp import msa
            >>> uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
            >>> meta = msa.MSAMetafile(uri+'jw02756001001_01_msa.fits')
            >>> res = meta.make_summary_table(msa_metadata_id=None,
                                              image_path=None,
                                              write_tables=False)
            >>> print(res[-10:])
            source_id    ra       dec     nexp Exp1 Exp2 Exp3
            --------- -------- ---------- ---- ---- ---- ----
               320023 3.610363 -30.414991    3  -o-  o--  --o
               320029 3.557964 -30.426137    3  -o-  o--  --o
               320035 3.616975 -30.419344    3  -o-  o--  --o
               340975 3.576616 -30.401801    3  ---  ---  ---
               410005 3.604646 -30.392461    3  ---  ---  ---
               410044 3.592863 -30.396336    3  -o-  o--  --o
               410045 3.592619 -30.397096    3  -o-  o--  --o
               410067 3.571049 -30.388132    3  -o-  o--  --o
               500002 3.589697 -30.398156    2  --o  -o-
               500003 3.591399 -30.401982    3  -o-  o--  --o

        """
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        import grizli.utils

        if msa_metadata_id is None:
            msa_metadata_id = self.shutter_table["msa_metadata_id"].min()

        _mset = self.shutter_table["msa_metadata_id"] == msa_metadata_id
        shut = self.shutter_table[_mset]

        sources = grizli.utils.Unique(shut["source_id"], verbose=False)

        root = os.path.basename(self.metafile).split("_msa.fits")[0]

        tab = grizli.utils.GTable()
        tab["source_id"] = sources.values
        tab["ra"] = -1.0
        tab["dec"] = -1.0
        tab["ra"][sources.indices] = shut["ra"]
        tab["dec"][sources.indices] = shut["dec"]

        tab["ra"].format = ".6f"
        tab["dec"].format = ".6f"
        tab["ra"].description = "Target R.A. (degrees)"
        tab["dec"].description = "Target Dec. (degrees)"

        tab.meta["root"] = root
        tab.meta["msa_metadata_id"] = msa_metadata_id

        exps = np.unique(shut["dither_point_index"])

        tab["nexp"] = 0
        for exp in exps:
            exp_ids = shut["source_id"][shut["dither_point_index"] == exp]
            tab["nexp"] += np.in1d(tab["source_id"], exp_ids)

            slitlets = []
            for s in tab["source_id"]:
                if s in exp_ids:
                    ix = sources[s] & (shut["dither_point_index"] == exp)
                    so = np.argsort(
                        shut["shutter_column", "primary_source"][ix]
                    )
                    ss = ""
                    # ss = f'{ix.sum()} '
                    for p in shut["primary_source"][ix][so]:
                        if p == "Y":
                            ss += "o"
                        else:
                            ss += "-"
                    slitlets.append(ss)
                else:
                    slitlets.append("")

            tab[f"Exp{exp}"] = slitlets

        mroot = f"{root}_{msa_metadata_id}"

        if image_path is not None:
            slit_images = []
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            print(f"Make {len(tab)} slit thumbnail images:")

            for src, ra in tqdm(zip(tab["source_id"], tab["ra"])):
                slit_image = os.path.join(
                    image_path, f"{mroot}_{src}_slit.png"
                )
                slit_image = slit_image.replace("_-", "_m")
                slit_images.append(slit_image)

                if os.path.exists(slit_image):
                    continue
                elif ra <= 0.00001:
                    continue

                # Make slit cutout
                dith = shut["dither_point_index"][sources[src]].min()
                fig, ax = self.plot_slitlet(
                    source_id=src,
                    dither_point_index=dith,
                    msa_metadata_id=msa_metadata_id,
                    **kwargs,
                )

                fig.savefig(slit_image)
                plt.close(fig)

                if 0:
                    kwargs = dict(
                        cutout_size=1.5,
                        step=None,
                        rgb_filters=[
                            "f200w-clear",
                            "f150w-clear",
                            "f115w-clear",
                        ],
                        rgb_scale=5,
                        rgb_invert=False,
                        figsize=(4, 4),
                        ax=None,
                        add_labels=True,
                        set_axis_labels=True,
                    )

            tab["thumb"] = [
                f'<img src="{im}" height=200px />' for im in slit_images
            ]

        if write_tables:
            tab.write_sortable_html(
                mroot + "_slits.html",
                max_lines=5000,
                filter_columns=["ra", "dec", "source_id"],
                localhost=False,
                use_json=False,
            )

            tab.write(mroot + "_slits.fits", overwrite=True)
        return tab

    def get_siaf_transforms(
        self,
        prefix=(
            "https://github.com/spacetelescope/pysiaf/raw/"
            + "master/pysiaf/source_data/NIRSpec/delivery/"
            + "test_data/apertures_testData/"
        ),
        check_rms=True,
    ):
        """
        Read shutter (i,j) > (v2,v3) transformations from the files at
        https://github.com/spacetelescope/pysiaf/tree/master/pysiaf/source_data/NIRSpec/delivery/test_data/apertures_testData

        Parameters
        ----------
        prefix : str, optional
            URL prefix for the files containing the transformations.

        check_rms : bool, optional
            If True, calculate and print the root mean square (RMS) of the
            transformations.

        Returns
        -------

        transforms : dict

            A dictionary containing the (i,j) > (v2,v3) transformations for each
            quadrant of the MSA.

        """
        from astropy.modeling.models import Polynomial2D
        from astropy.modeling.fitting import LinearLSQFitter
        import grizli.utils

        poly = Polynomial2D(degree=3)

        transforms = {}

        for quadrant in [1, 2, 3, 4]:
            ref_file = os.path.join(
                prefix, f"sky_fpa_projectionMSA_Q{quadrant}.fits"
            )
            fpa = grizli.utils.read_catalog(ref_file)
            # i, j transformations
            ij_to_v2 = LinearLSQFitter()(
                poly, fpa["I"] * 1.0, fpa["J"] * 1.0, fpa["XPOSSKY"] * 3600
            )
            ij_to_v3 = LinearLSQFitter()(
                poly, fpa["I"] * 1.0, fpa["J"] * 1.0, fpa["YPOSSKY"] * 3600
            )
            transforms[quadrant] = (ij_to_v2, ij_to_v3)

            if check_rms:
                ijx = ij_to_v2(fpa["I"] * 1.0, fpa["J"] * 1.0)
                ijy = ij_to_v3(fpa["I"] * 1.0, fpa["J"] * 1.0)
                stdx = np.std(ijx - fpa["XPOSSKY"] * 3600)
                stdy = np.std(ijy - fpa["YPOSSKY"] * 3600)

                print(f"Quadrant {quadrant} rms = {stdx:.2e} {stdy:.2e}")

        return transforms

    def get_exposure_info(
        self, msa_metadata_id=1, dither_point_index=1, **kwargs
    ):
        """
        Get MAST keywords for a particular exposure

        Parameters
        ----------
        msa_metadata_id, dither_point_index : int
            Exposure definition

        Returns
        -------
        row : `~astropy.table.row.Row`
            Row of the ``mast`` info table for a particular exposure

        """
        mast = self.query_mast_exposures(**kwargs)

        ix = mast["msametfl"] == os.path.basename(self.filename)
        ix &= mast["msametid"] == msa_metadata_id
        ix &= mast["patt_num"] == dither_point_index
        if ix.sum() == 0:
            msg = f"msametid = {msa_metadata_id}, "
            msg += f"exposure = {dither_point_index}"
            msg += " not found in MAST table"
            raise ValueError(msg)

        row = mast[ix][0]

        return row

    def get_siaf_aperture(
        self,
        msa_metadata_id=1,
        dither_point_index=1,
        pa_offset=-0.1124,
        ra_ref=None,
        dec_ref=None,
        roll_ref=None,
        use_ref_columns=True,
        **kwargs,
    ):
        """
        Generate a `pysiaf` aperture object based on pointing information

        Parameters
        ----------
        msa_metadata_id, dither_point_index : int
            Exposure definition

        pa_offset : float
            Empirical offset added to ``gs_v3_pa`` from the MAST query to match
            ``ROLL_REF`` in the science headers

        ra_ref, dec_ref, roll_ref : None, float
            Specify a reference parameters aperture attitude, e.g., taken from
            the ``ROLL_REF`` science header keywords

        use_ref_columns : bool
            Use "ref" columns in `mast` metadata table if found, e.g.,
            generated from `~msaexp.msa.MSAMetafile.fit_mast_pointing_offset`

        Returns
        -------
        ra, dec, roll : float
            The V2/V3 reference ra, dec, roll used for the aperture attitude

        ap : `pysiaf.aperture.NirspecAperture`
            Aperture object with attitude set based on database pointing
            keywords and with various coordinate transformation methods

        """
        import pysiaf
        from pysiaf.utils import rotations

        # Define SIAF aperture
        # ---------------------
        instrument = "NIRSPEC"

        siaf = pysiaf.siaf.Siaf(instrument)
        ap = siaf["NRS_FULL_MSA"]

        # Input is fully specified
        # -------------------------
        if (
            (ra_ref is not None)
            & (dec_ref is not None)
            & (roll_ref is not None)
        ):
            att = rotations.attitude(
                ap.V2Ref, ap.V3Ref, ra_ref, dec_ref, roll_ref
            )

            ap.set_attitude_matrix(att)

            return ra_ref, dec_ref, roll_ref, ap

        # Get pointing information from MAST query
        # -----------------------------------------
        row = self.get_exposure_info(
            msa_metadata_id=msa_metadata_id,
            dither_point_index=dither_point_index,
            **kwargs,
        )

        if ("roll_ref" in row.colnames) & use_ref_columns:
            ra_ref = row["ra_ref"]
            dec_ref = row["dec_ref"]
            roll_ref = row["roll_ref"]

            att = rotations.attitude(
                ap.V2Ref, ap.V3Ref, ra_ref, dec_ref, roll_ref
            )

            ap.set_attitude_matrix(att)

            return ra_ref, dec_ref, roll_ref, ap

        if roll_ref is None:
            roll = row["gs_v3_pa"] + pa_offset
        else:
            roll = roll_ref

        att = rotations.attitude(
            ap.V2Ref, ap.V3Ref, row["targ_ra"], row["targ_dec"], roll
        )

        ap.set_attitude_matrix(att)

        # Apply xoffset, yoffset defined in Ideal (idl) coordinates
        # ----------------------------------------------------------
        tel = ap.idl_to_tel(-row["xoffset"], -row["yoffset"])
        offset_rd = ap.tel_to_sky(*tel)

        # Recompute aperture with offset
        # -------------------------------
        siaf = pysiaf.siaf.Siaf(instrument)
        ap = siaf["NRS_FULL_MSA"]

        att = rotations.attitude(ap.V2Ref, ap.V3Ref, *offset_rd, roll)

        ap.set_attitude_matrix(att)
        return *offset_rd, roll, ap

    def fit_mast_pointing_offset(
        self,
        iterations=3,
        with_ransac=True,
        verbose=True,
        apply=True,
        init_ref_columns=False,
        **kwargs,
    ):
        """
        Fit for offsets to the pointing attitude derived from the MAST metadata

        Parameters
        ----------
        iterations : int
            Number of fitting iterations

        with_ransac : bool
            Refine transformation with `skimage.measure` inliers

        verbose : bool
            Print messages

        apply : bool
            Add updated pointing parameters to `ref` columns in `mast` metadata

        init_ref_columns : bool
            Initialize using "ref" columns in the MAST table

        Returns
        -------
        res : `~astropy.table.Table`
            Table summarizing fit results

        """
        import skimage.transform
        from skimage.measure import ransac
        import grizli.utils

        transform = skimage.transform.EuclideanTransform

        coeffs = load_siaf_shutter_transforms()

        _shut = self.shutter_table
        has_offset = np.isfinite(_shut["estimated_source_in_shutter_x"])
        has_offset &= np.isfinite(_shut["estimated_source_in_shutter_y"])

        is_src = (_shut["source_id"] > 0) & (has_offset)

        rows = []

        msg = "{0} {1:>.0f} {2:>.0f} {2:.6f} {3:.6f} {4:.3f}  {5:>5.0f}  "
        msg += "{6:.6f} {7:.6f} {8:.3f}"

        for key in self.mast_key_pairs:
            msa_metadata_id, dither_point_index = key

            _ = self.get_siaf_aperture(
                msa_metadata_id=msa_metadata_id,
                dither_point_index=dither_point_index,
                pa_offset=0.0,
                use_ref_columns=init_ref_columns,
                **kwargs,
            )
            _ra, _dec, _roll, ap = _

            xrow = [
                msa_metadata_id,
                dither_point_index,
                _ra * 1.0,
                _dec * 1,
                _roll * 1,
            ]

            exp = self.shutter_table["msa_metadata_id"] == msa_metadata_id
            exp &= (
                self.shutter_table["dither_point_index"] == dither_point_index
            )

            # _shut[is_src]['msa_metadata_id','dither_point_index']

            Nsrc = (exp & is_src).sum()

            if Nsrc < 4:
                xrow += [Nsrc, np.nan, np.nan, np.nan]
                rows.append(xrow)
                if (verbose & 4) > 0:
                    print(msg.format(*xrow))

                continue

            se = _shut[exp & is_src]

            row = se["shutter_row"] + (
                se["estimated_source_in_shutter_x"] - 0.5
            )
            col = se["shutter_column"] + (
                se["estimated_source_in_shutter_y"] - 0.5
            )

            ra, dec = se["ra"], se["dec"]

            # Shutter to V2,V3
            v2 = []
            v3 = []
            ind = []

            for i in range(len(se)):
                if se["shutter_quadrant"][i] not in coeffs:
                    v2.append(0)
                    v3.append(0)
                    continue

                ij_to_v2, ij_to_v3 = coeffs[se["shutter_quadrant"][i]]
                v2.append(ij_to_v2(row[i], col[i]))
                v3.append(ij_to_v3(row[i], col[i]))
                ind.append(i)

            #########
            # Fit for the transform

            # APT target coordinates
            input = np.array([[ra[i], dec[i]] for i in ind])
            x0 = np.mean(input, axis=0)
            cosd = np.array(
                [[np.cos(inp[1] / 180 * np.pi), 1] for inp in input]
            )

            init_output = np.array([ap.tel_to_sky(v2[i], v3[i]) for i in ind])

            for _iter in range(iterations):
                output = np.array([ap.tel_to_sky(v2[i], v3[i]) for i in ind])

                tf = transform()
                tf.estimate((output - x0) * cosd, (input - x0) * cosd)

                if with_ransac:
                    # Fit with inliers / outliers
                    tf, inliers = ransac(
                        [(output - x0) * cosd, (input - x0) * cosd],
                        transform,
                        min_samples=3,
                        residual_threshold=3,
                        max_trials=100,
                    )

                _ = self.get_siaf_aperture(
                    msa_metadata_id=msa_metadata_id,
                    dither_point_index=dither_point_index,
                    ra_ref=_ra + tf.translation[0] / cosd[0][0],
                    dec_ref=_dec + tf.translation[1],
                    roll_ref=_roll - tf.rotation / np.pi * 180,
                    **kwargs,
                )
                _ra, _dec, _roll, ap = _

            if (verbose & 2) > 0:
                resid_in = (init_output - x0) * cosd - (input - x0) * cosd
                resid_out = (output - x0) * cosd - (input - x0) * cosd
                print(
                    "rms: in ({0:.2f} {1:.2f}) out ({2:.2f}, {3:.2f}) mas".format(
                        *np.std(resid_in, axis=0) * 3600.0 * 1000,
                        *np.std(resid_out, axis=0) * 3600 * 1000.0,
                    )
                )

            xrow += [Nsrc, _ra * 1, _dec * 1, _roll * 1]
            if (verbose & 4) > 0:
                print(msg.format(*xrow))

            rows.append(xrow)

        res = grizli.utils.GTable(
            rows=rows,
            names=[
                "id",
                "dith",
                "ra0",
                "dec0",
                "roll0",
                "Nsrc",
                "ra",
                "dec",
                "roll",
            ],
        )

        cosd = np.cos(res["dec0"] / 180 * np.pi)
        res["dra"] = (res["ra"] - res["ra0"]) * cosd * 3600
        res["ddec"] = (res["dec"] - res["dec0"]) * 3600
        res["droll"] = res["roll"] - res["roll0"]

        if apply & (res["Nsrc"].max() > 4):
            dra = np.nanmedian(res["dra"] / 3600 / cosd)
            dde = np.nanmedian(res["ddec"] / 3600)
            dro = np.nanmedian(res["droll"])

            res["ra_ref"] = res["ra0"] + dra
            res["dec_ref"] = res["dec0"] + dde
            res["roll_ref"] = res["roll0"] + dro

            if not init_ref_columns:
                self.mast["ra_ref"] = 0.0
                self.mast["dec_ref"] = 0.0
                self.mast["roll_ref"] = 0.0

            for i, (id, dith) in enumerate(
                zip(self.mast["msametid"], self.mast["patt_num"])
            ):
                _ = self.get_siaf_aperture(
                    msa_metadata_id=id,
                    dither_point_index=dith,
                    pa_offset=0.0,
                    use_ref_columns=init_ref_columns,
                    **kwargs,
                )

                _ra, _dec, _roll, ap = _
                # print('xxx', _ra, _dec, _roll)

                ix = np.where(
                    (self.mast["msametid"] == id)
                    & (self.mast["patt_num"] == dith)
                )[0][0]

                self.mast["ra_ref"][ix] = _ra + dra
                self.mast["dec_ref"][ix] = _dec + dde
                self.mast["roll_ref"][ix] = _roll + dro

                # for c in ['ra_ref','dec_ref','roll_ref']:
                #     self.mast[c][ix] = res[c][i]

            if verbose:
                print(
                    f"Apply offset to metadata: translation= {dra*3600:6.3f}"
                    + f"  {dde*3600:6.3f}  rotation= {dro:>.4f}"
                )

        return res

    def regions_from_metafile_siaf(
        self,
        as_string=True,
        with_bars=True,
        msa_metadata_id=1,
        dither_point_index=1,
        shutter_table=None,
        meta_keys=[
            "program",
            "pi_name",
            "proptarg",
            "filename",
            "filter",
            "grating",
            "expstart",
            "effexptm",
            "nod_type",
            "final_0x0_EOSA",
        ],
        source_color="cyan",
        other_color="lightblue",
        verbose=False,
        **kwargs,
    ):
        """
        MSA shutter regions using pointing info and SIAF shutter
        transformations

        Parameters
        ----------
        as_string : bool
            Return regions as DS9 region strings

        with_bars : bool
            Account for bar vignetting

        msa_metadata_id, dither_point_index : int
            Exposure definition

        meta_keys : list of str, optional
            List of metadata keys to include in the output

        verbose : bool, optional
            Print verbose messages

        Returns
        -------
            String or a list of `grizli.utils.SRegion` objects, depending on
            ``as_string``

        """
        import grizli.utils

        # Exposure metadata
        meta = self.get_exposure_info(
            msa_metadata_id=msa_metadata_id,
            dither_point_index=dither_point_index,
            **kwargs,
        )

        if verbose:
            msg = "Generate regions from {msametfl} id = {msametid:>3}"
            msg += " exposure = {exposure}"
            msg += " (filename = {filename})"
            print(msg.format(**meta))

        # Slitlet transforms (i,j) > (v2,v3)
        # ----------------------------------
        coeffs = load_siaf_shutter_transforms()

        # Aperture with pointing information
        # -----------------------------------
        _ra, _dec, _roll, ap = self.get_siaf_aperture(
            msa_metadata_id=msa_metadata_id,
            dither_point_index=dither_point_index,
            **kwargs,
        )

        # Which exposure?
        # ----------------
        exp = self.shutter_table["msa_metadata_id"] == msa_metadata_id
        exp &= self.shutter_table["dither_point_index"] == dither_point_index

        _shut = self.shutter_table

        has_offset = np.isfinite(_shut["estimated_source_in_shutter_x"])
        has_offset &= np.isfinite(_shut["estimated_source_in_shutter_y"])

        # Regions for a particular exposure
        if shutter_table is None:
            se = _shut[exp]
        else:
            se = shutter_table

        sx = (np.array([-0.5, 0.5, 0.5, -0.5])) * (
            1 - 0.07 / 0.27 * with_bars / 2
        )
        sy = (np.array([-0.5, -0.5, 0.5, 0.5])) * (
            1 - 0.07 / 0.53 * with_bars / 2
        )

        row = se["shutter_row"]
        col = se["shutter_column"]

        regions = []

        for i in range(len(se)):

            if se["shutter_quadrant"][i] not in coeffs:
                continue

            ij_to_v2, ij_to_v3 = coeffs[se["shutter_quadrant"][i]]
            v2 = ij_to_v2(row[i] + sx, col[i] + sy)
            v3 = ij_to_v3(row[i] + sx, col[i] + sy)

            sra, sdec = ap.tel_to_sky(v2, v3)

            # sra = pra(row[i] + sx, col[i]+sy)
            # sdec = pdec(row[i] + sx, col[i]+sy)

            sr = grizli.utils.SRegion(np.array([sra, sdec]), wrap=False)
            sr.meta = {}
            for k in [
                "program",
                "source_id",
                "ra",
                "dec",
                "slitlet_id",
                "shutter_quadrant",
                "shutter_row",
                "shutter_column",
                "estimated_source_in_shutter_x",
                "estimated_source_in_shutter_y",
            ]:
                if k in se.colnames:
                    sr.meta[k] = se[k][i]

            if "estimated_source_in_shutter_x" in sr.meta:
                sr.meta["is_source"] = np.isfinite(
                    sr.meta["estimated_source_in_shutter_x"]
                )
            else:
                sr.meta["is_source"] = False

            if sr.meta["is_source"]:
                sr.ds9_properties = "color=" + source_color
            else:
                sr.ds9_properties = "color=" + other_color

            regions.append(sr)

        if as_string:
            output = f"# msametfl = {self.metafile}\n"

            di = dither_point_index
            output += f"# dither_point_index = {di}\n"
            mi = msa_metadata_id
            output += f"# msa_metadata_id = {mi}\n"

            for k in meta_keys:
                if k in meta.colnames:
                    output += f"# {k} = {meta[k]}\n"

            output += "icrs\n"
            for sr in regions:
                m = sr.meta
                if m["is_source"]:
                    output += f"circle({m['ra']:.7f}, {m['dec']:.7f}, 0.2\")"
                    output += f" # color=cyan text=xx{m['source_id']}yy\n"

                for r in sr.region:
                    output += r + "\n"

            output = output.replace("xx", "{").replace("yy", "}")

        else:
            output = regions

        return output

    def all_regions_from_metafile_siaf(self, **kwargs):
        """
        Run `~msaexp.msa.MSAMetafile.regions_from_metafile_siaf`
        for all exposures

        Parameters
        ----------
        kwargs : dict
            Passed to `~msaexp.msa.MSAMetafile.regions_from_metafile_siaf`

        Returns
        -------
        output : list, str
            Depending on ``as_string`` input keyword

        """
        output = None
        for key in self.mast_key_pairs:
            msa_metadata_id, dither_point_index = key
            _out = self.regions_from_metafile_siaf(
                msa_metadata_id=msa_metadata_id,
                dither_point_index=dither_point_index,
                **kwargs,
            )
            if output is None:
                output = _out
            else:
                output += _out

        return output

    def write(self, prefix="edit_", overwrite=True, verbose=True):
        """
        Write a new metadata file
        """
        import grizli.utils

        im = pyfits.open(self.filename)

        shut = self.shutter_table
        src = self.src_table

        hdus = {
            "SHUTTER_INFO": pyfits.BinTableHDU(shut),
            "SOURCE_INFO": pyfits.BinTableHDU(src),
        }

        for e in hdus:
            for k in im[e].header:
                if k not in hdus[e].header:
                    # print(e, k, im[e].header[k])
                    hdus[e].header[k] = im[e].header[k]
            im[e] = hdus[e]

        output_file = prefix + self.filename

        msg = f"msaexp.msa: write from {self.filename} to {output_file}"
        grizli.utils.log_comment(
            grizli.utils.LOGFILE, msg, verbose=verbose, show_date=True
        )

        im.writeto(output_file, overwrite=overwrite)
        im.close()


PYSIAF_GITHUB = "https://github.com/spacetelescope/pysiaf/raw/main/pysiaf/source_data/NIRSpec/delivery/test_data/apertures_testData/"


def fit_siaf_shutter_transforms(
    prefix=PYSIAF_GITHUB, degree=3, inverse_degree=2, check_rms=True
):
    """
    Fit shutter (i,j) > (v2,v3) transformations from the files at
    https://github.com/spacetelescope/pysiaf/tree/main/pysiaf/source_data/NIRSpec/delivery/test_data/apertures_testData

    Parameters
    ----------
    prefix : str, optional
        The URL prefix for the files containing the transformations.
    degree : int, optional
        The degree of the polynomial fit for the transformations. Default is 3.
    inverse_degree : int, optional
        The degree of the polynomial fit for the inverse transformations.
        Default is 2.
    check_rms : bool, optional
        Whether to calculate and print the RMS of the transformations.
        Default is True.

    Returns
    -------
    transforms : dict
        A dictionary containing the fitted transformations. The keys are:

        - "degree": the degree of the polynomial fit

        - "coeffs": a nested dictionary containing the coefficients of the
            transformations for each quadrant

        - "irange": a nested dictionary containing the range of valid "i"
            values for each quadrant

        - "jrange": a nested dictionary containing the range of valid "j"
            values for each quadrant

        - "inverse_degree": the degree of the polynomial fit for the inverse
            transformations

        - "inverse": a nested dictionary containing the coefficients of the
            inverse transformations for each quadrant

        - "rms" (optional): a nested dictionary containing the RMS of the
            transformations for each quadrant

    """
    from astropy.modeling.models import Polynomial2D
    from astropy.modeling.fitting import LinearLSQFitter
    import grizli.utils

    def _model_to_dict(model):
        """ """
        params = {}
        for k, v in zip(model.param_names, model.parameters):
            params[k] = float(v)
        return params

    poly = Polynomial2D(degree=degree)
    pinv = Polynomial2D(degree=inverse_degree)

    transforms = {
        "degree": degree,
        "coeffs": {},
        "irange": {},
        "jrange": {},
        "inverse_degree": inverse_degree,
        "inverse": {},
    }

    if check_rms:
        transforms["rms"] = {}

    for quadrant in [1, 2, 3, 4]:
        ref_file = os.path.join(
            prefix, f"sky_fpa_projectionMSA_Q{quadrant}.fits"
        )
        fpa = grizli.utils.read_catalog(ref_file)

        transforms["irange"][quadrant] = [
            int(fpa["I"].min()),
            int(fpa["I"].max()),
        ]
        transforms["jrange"][quadrant] = [
            int(fpa["J"].min()),
            int(fpa["J"].max()),
        ]

        # Transformation (i,j) to (v2,v3)
        ij_to_v2 = LinearLSQFitter()(
            poly, fpa["I"] * 1.0, fpa["J"] * 1.0, fpa["XPOSSKY"] * 3600
        )

        ij_to_v3 = LinearLSQFitter()(
            poly, fpa["I"] * 1.0, fpa["J"] * 1.0, fpa["YPOSSKY"] * 3600
        )

        v2d = _model_to_dict(ij_to_v2)
        v3d = _model_to_dict(ij_to_v3)

        transforms["coeffs"][quadrant] = {"ij_to_v2": v2d, "ij_to_v3": v3d}

        # Inverse transformation v2,v3 to i,j
        v23_to_i = LinearLSQFitter()(
            pinv, fpa["XPOSSKY"] * 3600, fpa["YPOSSKY"] * 3600, fpa["I"] * 1.0
        )

        v23_to_j = LinearLSQFitter()(
            pinv, fpa["XPOSSKY"] * 3600, fpa["YPOSSKY"] * 3600, fpa["J"] * 1.0
        )

        icoeffs = _model_to_dict(v23_to_i)
        jcoeffs = _model_to_dict(v23_to_j)

        transforms["inverse"][quadrant] = {
            "v23_to_i": icoeffs,
            "v23_to_j": jcoeffs,
        }

        if check_rms:
            ijx = ij_to_v2(fpa["I"] * 1.0, fpa["J"] * 1.0)
            ijy = ij_to_v3(fpa["I"] * 1.0, fpa["J"] * 1.0)
            stdx = np.std(ijx - fpa["XPOSSKY"] * 3600)
            stdy = np.std(ijy - fpa["YPOSSKY"] * 3600)

            transforms["rms"][quadrant] = [float(stdx), float(stdy)]

            msg = f"Q{quadrant} rms = {stdx:.2e} {stdy:.2e}"
            msg += "    arcsec (i,j > v2,v3)"

            ix = v23_to_i(fpa["XPOSSKY"] * 3600.0, fpa["YPOSSKY"] * 3600.0)

            jx = v23_to_j(fpa["XPOSSKY"] * 3600.0, fpa["YPOSSKY"] * 3600.0)

            stdi = np.std(ix - fpa["I"] * 1.0)
            stdj = np.std(jx - fpa["J"] * 1.0)

            msg += (
                "\n"
                + f"         {stdi:.2e} {stdj:.2e}   shutter (v2,v3 > i,j)"
            )

            print(msg)

    if False:
        import yaml

        header = """# 2D polynomal fits to the tables at https://github.com/spacetelescope/pysiaf/tree/main/pysiaf/source_data/NIRSpec/delivery/test_data/apertures_testData
#
# The keys of the "coeffs" dict are the MSA quadrant numbers: (1,2,3,4)
# and the values are astropy.modeling.models.Polynomial2D(degree) coefficients
# for the transformation shutter (row=i,col=j) > v2 and (row=i,col=j) > v3
#
# The coefficients of the "inverse" dict are the transformations
# (v2,v3) > (row=i) and (v2,v3) > (col=j)
#
# RMS of the transformations:
#   Q1 rms = 1.76e-04 1.80e-04    arcsec (i,j > v2,v3)
#            6.93e-03 1.70e-03   shutter (v2,v3 > i,j)
#   Q2 rms = 1.55e-04 1.88e-04    arcsec (i,j > v2,v3)
#            8.96e-03 2.99e-03   shutter (v2,v3 > i,j)
#   Q3 rms = 1.44e-04 2.56e-04    arcsec (i,j > v2,v3)
#            6.66e-03 2.10e-03   shutter (v2,v3 > i,j)
#   Q4 rms = 6.72e-05 2.51e-04    arcsec (i,j > v2,v3)
#            9.91e-03 2.62e-03   shutter (v2,v3 > i,j)
        """
        with open("/tmp/nirspec_msa_transforms.yaml", "w") as fp:
            fp.write(header)
            yaml.dump(transforms, fp)

    return transforms


def load_siaf_shutter_transforms():
    """
    Read MSA shutter transforms (i,j) > (v2,v3) from file created with
    `msaexp.msa.fit_siaf_shutter_transforms`

    Returns
    -------
    tr : dict
        Transform coefficients by quadrant

    """
    import yaml
    from astropy.modeling.models import Polynomial2D

    tfile = os.path.join(
        os.path.dirname(__file__), "data/nirspec_msa_transforms.yaml"
    )

    with open(tfile) as fp:
        traw = yaml.load(fp, Loader=yaml.Loader)

    degree = traw["degree"]
    tr = {}
    for k in traw["coeffs"]:
        tr[k] = (
            Polynomial2D(degree, **traw["coeffs"][k]["ij_to_v2"]),
            Polynomial2D(degree, **traw["coeffs"][k]["ij_to_v3"]),
        )

    return tr


def load_siaf_inverse_shutter_transforms():
    """
    Read MSA shutter transforms (v2,v3) > (i,j) from file created with
    `msaexp.msa.fit_siaf_shutter_transforms`

    Returns
    -------
    tr : dict
        Nested dictionary with keys
          - ``inverse``: inverse transform by quadrant
          - ``irange``: Range of valid ``i``
          - ``jrange``: Range of valid ``j``

    """
    import yaml
    from astropy.modeling.models import Polynomial2D

    tfile = os.path.join(
        os.path.dirname(__file__), "data/nirspec_msa_transforms.yaml"
    )

    with open(tfile) as fp:
        traw = yaml.load(fp, Loader=yaml.Loader)

    degree = traw["inverse_degree"]
    tr = {
        "inverse": {},
        "irange": traw["irange"],
        "jrange": traw["jrange"],
    }

    for k in traw["inverse"]:
        tr["inverse"][k] = (
            Polynomial2D(degree, **traw["inverse"][k]["v23_to_i"]),
            Polynomial2D(degree, **traw["inverse"][k]["v23_to_j"]),
        )

    return tr


def slit_best_source_alias(
    slit, require_primary=False, which="min", verbose=False
):
    """
    Get the "best" source id from the MSA metadata file associated with a particular
    slit extraction.

    Parameters
    ----------
    slit : str, `stdatamodels.jwst.datamodels.slit.SlitModel`
        Slit object or filename of one

    require_primary : bool
        Include a test on "primary_source = Y" in the shutter table.

    which : 'most', 'min', 'max'
        Criterion for picking the "best" source_id among multiple source ids assigned
        for a particular ``slitlet_id``.

    Returns
    -------
    alias : str
        Source alias, either ``background_{id}``if no non-zero source ids found or
        ``{program}_{id}`` if a primary source found with a particular slitlet_id.

    """
    import jwst.datamodels
    import grizli.utils

    if which not in ["most", "min", "max"]:
        msg = f"slit_best_source_alias: 'which' must be 'most_common', 'min', 'max'"
        raise ValueError(msg)

    if isinstance(slit, str):
        slit = jwst.datamodels.open(slit)
        opened = True
    else:
        opened = False

    msa = MSAMetafile(slit.meta.instrument.msa_metadata_file)
    shu = msa.shutter_table
    rows = shu["msa_metadata_id"] == slit.meta.instrument.msa_metadata_id
    rows &= shu["slitlet_id"] == slit.slitlet_id
    rows &= shu["source_id"] > 0

    test = rows

    if require_primary:
        primary = rows & (shu["primary_source"] == "Y")
        if primary.sum() > 0:
            test = primary

    if test.sum() == 0:
        alias = f"background_{slit.slitlet_id}"
        return alias

    source_ids = grizli.utils.Unique(
        shu["source_id"][test].tolist(), verbose=False
    )
    if which == "most":
        best_id = np.array(source_ids.values)[np.argmax(source_ids.counts)]
    elif which == "min":
        best_id = np.min(source_ids.values)
    elif which == "max":
        best_id = np.max(source_ids.values)

    prog = int(slit.meta.observation.program_number)

    alias = f"{prog}_{best_id}"

    msg = f"slit_best_source_alias: {alias} chosen among {source_ids.values}"
    grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=verbose)

    if opened:
        slit.close()

    return alias


def msa_shutter_catalog(ra, dec, pointing=None, ap=None, inv=None):
    """
    Compute shutter centering for a list of input coordinates

    Parameters
    ----------
    ra, dec : array-like
        Arrays of RA, Dec.  in decimal degrees

    pointing : None, (RA_REF, DEC_REF, ROLL_REF)
        Pointing definition

    ap : `pysiaf.aperture.NirspecAperture`
        Aperture object with attitude set based on database pointing keywords
        and with various coordinate transformation methods

    inv : dict
        Inverse shutter transformations from
        `msaexp.msa.load_siaf_inverse_shutter_transforms`

    Returns
    -------
    tab : `astropy.table.table`
        Table with shutter information

    """
    import pysiaf
    from pysiaf.utils import rotations
    from grizli import utils

    if pointing is not None:
        siaf = pysiaf.siaf.Siaf("NIRSPEC")
        ap = siaf["NRS_FULL_MSA"]
        att = rotations.attitude(ap.V2Ref, ap.V3Ref, *pointing)

        ap.set_attitude_matrix(att)

    if ap is None:
        raise ValueError("Either pointing or ap must be provided")

    if inv is None:
        inv = load_siaf_inverse_shutter_transforms()

    tab = utils.GTable()
    tab["ra"] = ra
    tab["dec"] = dec
    tab["quadrant"] = -1
    tab["row_i"] = np.nan
    tab["col_j"] = np.nan

    v2, v3 = ap.sky_to_tel(ra, dec)

    for q in [1, 2, 3, 4]:
        tr = inv["inverse"][q]
        iq = tr[0](v2, v3) + 0.5
        jq = tr[1](v2, v3) + 0.5

        inti = np.floor(iq).astype(int)
        intj = np.floor(jq).astype(int)

        clip = (inti >= inv["irange"][q][0]) & (inti <= inv["irange"][q][1])
        clip &= (intj >= inv["jrange"][q][0]) & (intj <= inv["jrange"][q][1])

        if clip.sum() > 0:
            tab["quadrant"][clip] = q
            tab["row_i"][clip] = iq[clip]
            tab["col_j"][clip] = jq[clip]

    tab["di"] = tab["row_i"] - np.floor(tab["row_i"]) - 0.5
    tab["dj"] = tab["col_j"] - np.floor(tab["col_j"]) - 0.5

    return tab


# Offsets for MSA shutter quadrants
DXCEN = 450
DYCEN = 231.7


def set_msa_axis_ticks(ax=None, which="xy", dxcen=450, dycen=231.7):
    """
    Draw axis ticks for a plot of the MSA shutters
    """
    import matplotlib.pyplot as plt

    global DXCEN, DYCEN

    if ax is None:
        ax = plt.gca()

    xlim = (377 + DXCEN + 20, -20)
    ylim = (171 + DYCEN + 10, -10)

    if "x" in which:
        xt = [1, 90, 180, 270, 377]
        xtv = xt + [xi + DXCEN for xi in xt]
        ax.set_xticks(xtv)
        ax.set_xticklabels(xt * 2)
        ax.set_xlabel("xcen")
        ax.set_xlim(*xlim)

    if "y" in which:
        xt = [1, 60, 120, 171]
        xtv = xt + [xi + DYCEN for xi in xt]
        ax.set_yticks(xtv)
        ax.set_yticklabels(xt * 2)
        ax.set_ylabel("ycen")
        ax.set_ylim(*ylim)

    return ax


def plot_msa_shutters(xcen, ycen, quadrant, c=None, ax=None, **kwargs):
    """
    Make a plot of a list of MSA shutters
    """
    import matplotlib.pyplot as plt

    global DXCEN, DYCEN

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        newplot = True
    else:
        newplot = False

    if c is None:
        c = quadrant

    ax.scatter(
        xcen + np.isin(quadrant, [3, 4]) * DXCEN,
        ycen + np.isin(quadrant, [2, 4]) * DYCEN,
        c=c,
        **kwargs,
    )

    if newplot:
        set_msa_axis_ticks(ax, "xy")
        return fig, ax
    else:
        return None, ax


def diffraction_spikes_on_msa(
    ra, dec, attitude_matrix, aper=None, buffer_arcsec=None
):
    """
    Generate estimated regions of diffraction spikes for a bright star

    Parameters
    ----------
    ra, dec: float
        Coordinates of the source creating the spikes in decimal degrees

    aper : `pysiaf.Aperture`, None
        SIAF aperture in which to compute the transformations.  If None, then use
        NIRSpec NRS_FULL_MSA.

    attitude_matrix : array
        Spacecraft attitude matrix from `pysiaf.rotations.attitude`

    buffer_arcsec : float, None
        Optional buffer to add to the spike regions

    Returns
    -------
    spike_regions : list
        List of `~grizli.utils.SRegion` objects for the spike polygons
    """
    import pysiaf
    import grizli.utils

    # Rough polygons of the diffraction spikes in relative "tel" coordinates
    regions_v23 = """
    ((0.69,-1.27),(-0.69,-1.25),(-0.33,-38.09),(0.72,-38.68))
    ((0.75,1.24),(1.43,0.04),(40.86,22.74),(39.10,23.29),(3.88,3.07))
    ((4.30,3.31),(4.30,3.31),(4.30,3.31),(4.30,3.31))
    ((-0.58,-1.15),(-1.26,0.05),(-34.79,-19.34),(-33.77,-20.18))
    ((-0.76,1.08),(0.61,1.07),(0.10,36.74),(-0.58,36.97))
    ((1.30,0.00),(0.61,-1.20),(41.75,-24.18),(42.99,-23.48))
    ((-1.45,-0.07),(-0.74,1.11),(-34.94,21.31),(-35.74,20.20))
    ((-1.64,-0.23),(-1.64,0.29),(-11.00,0.14),(-11.02,-0.18),(-4.41,-0.14))
    ((1.36,0.31),(1.35,-0.21),(10.71,-0.11),(10.73,0.21))
    """.strip().split()

    if aper is None:
        nrs = pysiaf.Siaf("NIRSPEC")
        aper = nrs["NRS_FULL_MSA"]

    aper.set_attitude_matrix(attitude_matrix)

    # ra_src, dec_src = 150.0670825, 2.2730196
    v23_src = aper.sky_to_tel(ra, dec)

    spike_regions = []

    for region in regions_v23:
        sr0 = grizli.utils.SRegion(region, wrap=False)

        reg_v23 = np.array(sr0.xy[0]) + np.array(v23_src)
        reg_sky = np.array(aper.tel_to_sky(*(reg_v23.T)))

        sr = grizli.utils.SRegion(reg_sky)
        if buffer_arcsec is not None:
            sr = grizli.utils.SRegion(
                sr.shapely[0].buffer(buffer_arcsec / 3600.0)
            )

        spike_regions.append(sr)

    return spike_regions


def read_apt_shutter_csv(file):
    """
    Read an APT MSA shutter CSV file into a table

    Parameters
    ----------
    file : str
        Filename of the shutter CSV file exported from APT

    Returns
    -------
    tab, otab : `~grizli.utils.GTable`
        Tables for closed and open shutters

    """
    import grizli.utils

    with open(file) as fp:
        lines = fp.readlines()[1:]

    data = np.array([np.cast[int](line.strip().split(",")) for line in lines])

    q = data * 0
    q[:365, :171] = 1
    q[365:, :171] = 3
    q[:365, 171:] = 2
    q[365:, 171:] = 4

    row, col = np.indices(data.shape)
    row = (row % 365) + 1
    col = (col % 171) + 1

    closed = data == 1

    tab = grizli.utils.GTable()
    tab["shutter_quadrant"] = q[closed]
    tab["shutter_row"] = row[closed]
    tab["shutter_column"] = col[closed]

    otab = grizli.utils.GTable()
    otab["shutter_quadrant"] = q[~closed]
    otab["shutter_row"] = row[~closed]
    otab["shutter_column"] = col[~closed]

    return tab, otab
