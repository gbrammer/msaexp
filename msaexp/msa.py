"""
Helper scripts for dealing with MSA metadata files (``MSAMETFL``)
"""
import os
import numpy as np
import astropy.io.fits as pyfits

__all__ = ["regions_from_metafile", "regions_from_fits",
           "MSAMetafile"]


def pad_msa_metafile(metafile, pad=1, source_ids=None, positive_ids=False, prefix='src_', verbose=True):
    """
    Pad a MSAMETFL with dummy slits and trim to a subset of source_ids
    
    Parameters
    ----------
    metafile : str
        Filename of the MSA metadata file (``MSAMETFL``)
    
    pad : int
        Padding of dummy slits
    
    source_ids : list, None
        List of source_id 
    
    positive_ids : bool
        If no ``source_ids`` provided, generate sources with `source_id > 0`
    
    prefix : str
        Prefix of new file to create (``prefix + metafile``)
    
    Returns
    -------
    
    output_file : str
        Filename of new table
    
    """
    from astropy.table import Table
    import grizli.utils
    
    msa = MSAMetafile(metafile)

    all_ids = np.unique(msa.shutter_table['source_id'])
    
    if source_ids is None:
        if positive_ids:
            source_ids = all_ids[all_ids > 0]
        else:
            source_ids = all_ids[all_ids != 0]

    six = np.in1d(msa.shutter_table['source_id'], source_ids)

    if six.sum() == 0:
        msg = f'msaexp.utils.pad_msa_metafile: {source_ids} not found in {metafile}.'
        msg += f'  Available ids are {list(all_ids)}'
        raise ValueError(msg)
    
    slitlets = np.unique(msa.shutter_table['slitlet_id'][six])
    im = pyfits.open(metafile)
    
    shut = Table(im['SHUTTER_INFO'].data)
    shut = shut[np.in1d(shut['slitlet_id'], slitlets)]

    # Add a shutter on either side
    row = {}
    for k in shut.colnames:
        row[k] = shut[k][0]

    row['shutter_state'] = 'CLOSED'
    row['background'] = 'Y'
    row['estimated_source_in_shutter_x'] = np.nan
    row['estimated_source_in_shutter_y'] = np.nan
    row['primary_source'] = 'N'
    
    new_rows = []

    for src_id in source_ids:
        six = shut['source_id'] == src_id
    
        for mid in np.unique(shut['msa_metadata_id'][six]):
            mix = (shut['msa_metadata_id'] == mid) & (six)
            quad = shut['shutter_quadrant'][mix][0]
            slit_id = shut['slitlet_id'][mix][0]
            shutters = np.unique(shut['shutter_column'][mix])
            for eid in np.unique(shut['dither_point_index'][mix]):
                for p in range(pad):
                    for s in [shutters.min()-(p+1), shutters.max()+(p+1)]:

                        row['msa_metadata_id'] = mid
                        row['dither_point_index'] = eid
                        row['shutter_column'] = s
                        row['shutter_quadrant'] = quad
                        row['slitlet_id'] = slit_id
                        row['source_id'] = src_id
                        
                        new_row = {}
                        for k in row:
                            new_row[k] = row[k]
                        
                        new_rows.append(new_row)
                
    for row in new_rows:
        shut.add_row(row)
    
    src = Table(im['SOURCE_INFO'].data)
    src = src[np.in1d(src['source_id'], source_ids)]

    hdus = {'SHUTTER_INFO': pyfits.BinTableHDU(shut), 
            'SOURCE_INFO': pyfits.BinTableHDU(src)}

    for e in hdus:
        for k in im[e].header:
            if k not in hdus[e].header:
                #print(e, k, im[e].header[k])
                hdus[e].header[k] = im[e].header[k]
        im[e] = hdus[e]
    
    output_file = prefix + metafile
    
    im.writeto(output_file, overwrite=True)
    im.close()
    
    if verbose:
        msg = f'msaexp.utils.pad_msa_metafile: Trim {metafile} to {list(source_ids)}\n'
        msg += f'msaexp.utils.pad_msa_metafile: pad = {pad}'
        grizli.utils.log_comment(grizli.utils.LOGFILE, msg, verbose=True, 
                                 show_date=True)

    return output_file


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
        metafile = im[0].header['MSAMETFL']
        metaid = im[0].header['MSAMETID']
        dither_point = im[0].header['PATT_NUM']
        
    metf = MSAMetafile(metafile)
    regions = metf.regions_from_metafile(msa_metadata_id=metaid,
                                         dither_point_index=dither_point,
                                         **kwargs)
    return regions


class MSAMetafile():
    def __init__(self, metafile):
        """
        Helper for parsing MSAMETFL metadata files
        
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
            
            fig, axes = plt.subplots(1,3,figsize=(9,2.6), sharex=True, sharey=True)
            cosd = np.cos(np.median(meta.src_table['dec'])/180*np.pi)
        
            for i in [0,1,2]:
                ax = axes[i]
                ax.scatter(meta.src_table['ra'], meta.src_table['dec'],
                           marker='.', color='k', alpha=0.5)
                slits = meta.regions_from_metafile(dither_point_index=i+1,
                                                   as_string=False, with_bars=True)
                for s in slits:
                    if s.meta['is_source']:
                        if s.meta['source_id'] in [110003, 410044, 410045]:
                            ax.text(s.meta['ra'] - 0.8/3600, s.meta['dec'],
                                    s.meta['source_id'],
                                    fontsize=7, ha='left', va='center')
                        fc = '0.5'
                    else:
                        fc = 'pink'
                
                    for patch in s.get_patch(fc=fc, ec='None', alpha=0.8, zorder=100):
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
            axes[1].text(0.5, 0.45, f'({x0:.6f}, {y0:.6f})', ha='left', va='top',
                         transform=axes[1].transAxes, fontsize=6,
                         color='b')
            
            fig.tight_layout(pad=0.5)
        
        
        Attributes
        ----------
        metafile : str
            Filename
        
        shutter_table : `~astropy.table.Table`
            Table of shutter metadata
        
        src_table : `~astropy.table.Table`
            Table of source information

        """
        from astropy.table import Table

        self.metafile = metafile
        
        with pyfits.open(metafile) as im:
            src = Table(im['SOURCE_INFO'].data)
            shut = Table(im['SHUTTER_INFO'].data)
    
        # Merge src and shutter tables
        shut_ix, src_ix = [], []

        for i, sid in enumerate(shut['source_id']):
            if sid == 0:
                continue
            elif sid in src['source_id']:
                shut_ix.append(i)
                src_ix.append(np.where(src['source_id'] == sid)[0][0])

        for c in ['ra','dec','program']:
            shut[c] = src[c][0]*0
            shut[c][shut_ix] = src[c][src_ix]
        
        self.shutter_table = shut
        self.src_table = src


    def get_transforms(self, dither_point_index=None, msa_metadata_id=None, fit_degree=2, verbose=False, **kwargs):
        """
        Fit for `~astropy.modeling.models.Polynomial2D` transforms between slit ``(row, col)``
        and ``(ra, dec)``.
        
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
            `~astropy.modeling.models.Polynomial2D` transformations to sky coordinates in 
            each of 4 MSA quadrants
            
            >>> quadrant = 1
            >>> pra, pdec = coeffs[quadrant]
            >>> ra = pra(shutter_row, shutter_column)
            >>> dec = pdec(shutter_row, shutter_column)
        
        inv_coeffs : dict
            Inverse `~astropy.modeling.models.Polynomial2D` transformations from sky to 
            shutters:
        
            >>> quadrant = 1
            >>> prow, pcol = inv_coeffs[quadrant]
            >>> shutter_row = prow(ra, dec)
            >>> shutter_column = pcol(ra, dec)
        
        """
        from astropy.modeling.models import Polynomial2D
        from astropy.modeling.fitting import LinearLSQFitter
        import grizli.utils

        p2 = Polynomial2D(degree=fit_degree)
        fitter = LinearLSQFitter()
            
        if dither_point_index is None:
            dither_point_index = self.shutter_table['dither_point_index'].min()
        if msa_metadata_id is None:
            msa_metadata_id = self.shutter_table['msa_metadata_id'].min()
        
        dither_match = (self.shutter_table['dither_point_index'] == dither_point_index) 
        meta_match = (self.shutter_table['msa_metadata_id'] == msa_metadata_id)
        exp = dither_match & meta_match
    
        has_offset = np.isfinite(self.shutter_table['estimated_source_in_shutter_x'])
        has_offset &= np.isfinite(self.shutter_table['estimated_source_in_shutter_y'])
    
        is_src = (self.shutter_table['source_id'] > 0) & (has_offset)
        si = self.shutter_table[exp & is_src]
    
        # Fit for transformations
        coeffs = {}
        inv_coeffs = {}
        
        if verbose:
            output = f'# msametfl = {self.metafile}\n'
            output += f'# dither_point_index = {dither_point_index}\n'
            output += f'# msa_metadata_id = {msa_metadata_id}'
            print(output)
            
        for qi in np.unique(si['shutter_quadrant']):
            q = si['shutter_quadrant'] == qi
            # print(qi, q.sum())
        
            row = si['shutter_row'] + si['estimated_source_in_shutter_x']
            col = si['shutter_column'] + si['estimated_source_in_shutter_y']

            pra = fitter(p2, row[q], col[q], si['ra'][q])
            pdec = fitter(p2, row[q], col[q], si['dec'][q])
            
            # RMS of the fit
            xra = pra(row[q], col[q])
            xdec = pdec(row[q], col[q])
            dra = (si['ra'][q]-xra)*np.cos(si['dec'][q]/180*np.pi)*3600*1000
            dde = (si['dec'][q]-xdec)*3600*1000
            pra.rms = np.std(dra)
            pdec.rms = np.std(dde)
            pra.N = q.sum()
            
            if verbose:
                print(f'# Q{qi} N={q.sum()}  rms= {pra.rms:.1f}, {pdec.rms:.1f} mas')
                
            coeffs[qi] = pra, pdec

            prow = fitter(p2, si['ra'][q], si['dec'][q], row[q])
            pcol = fitter(p2, si['ra'][q], si['dec'][q], col[q])
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
            String or a list of `grizli.utils.SRegion` objects, depending on ``as_string``
        
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
    
        has_offset = np.isfinite(self.shutter_table['estimated_source_in_shutter_x'])
        has_offset &= np.isfinite(self.shutter_table['estimated_source_in_shutter_y'])
    
        is_src = (self.shutter_table['source_id'] > 0) & (has_offset)
        si = self.shutter_table[exp & is_src]
    
        # Regions for a particular exposure
        se = self.shutter_table[exp]
    
        sx = (np.array([-0.5, 0.5, 0.5, -0.5]))*(1-0.07/0.27*with_bars/2) + 0.5
        sy = (np.array([-0.5, -0.5, 0.5, 0.5]))*(1-0.07/0.53*with_bars/2) + 0.5

        row = se['shutter_row'] #+ se['estimated_source_in_shutter_x']
        col = se['shutter_column'] #+ se['estimated_source_in_shutter_y']
        ra, dec = se['ra'], se['dec']
    
        regions = []
    
        for i in range(len(se)):
            
            if se['shutter_quadrant'][i] not in coeffs:
                continue
                
            pra, pdec = coeffs[se['shutter_quadrant'][i]]
            sra = pra(row[i] + sx, col[i]+sy)
            sdec = pdec(row[i] + sx, col[i]+sy)

            sr = grizli.utils.SRegion(np.array([sra, sdec]), wrap=False)
            sr.meta = {}
            for k in ['program', 'source_id', 'ra', 'dec', 
                      'slitlet_id', 'shutter_quadrant', 'shutter_row', 'shutter_column',
                      'estimated_source_in_shutter_x', 'estimated_source_in_shutter_y']:
                sr.meta[k] = se[k][i]
        
            sr.meta['is_source'] = np.isfinite(se['estimated_source_in_shutter_x'][i])
        
            if sr.meta['is_source']:
                sr.ds9_properties = "color=cyan"
            else:
                sr.ds9_properties = "color=lightblue"
        
            regions.append(sr)
        
        if as_string:
            output = f'# msametfl = {self.metafile}\n'
            
            di = self.shutter_table['dither_point_index'][dith][0]
            output += f'# dither_point_index = {di}\n'
            mi = self.shutter_table['msa_metadata_id'][metid][0]
            output += f'# msa_metadata_id = {mi}\n'
            
            for qi in coeffs:
                pra, pdec = coeffs[qi]
                output += f'# Q{qi} N={pra.N}  rms= {pra.rms:.1f}, {pdec.rms:.1f} mas\n'
            
            output += 'icrs\n'
            for sr in regions:
                m = sr.meta
                if m['is_source']:
                    output += f"circle({m['ra']:.7f}, {m['dec']:.7f}, 0.2\")"
                    output += f" # color=cyan text=xx{m['source_id']}yy\n"
                
                for r in sr.region:
                    output += r + '\n'
            
            output = output.replace('xx','{').replace('yy', '}')
            
        else:
            output = regions
            
        return output
    
    
    def plot_slitlet(self, source_id=110003, cutout_size=1.5, step=None, rgb_filters=None, figsize=(4,4), ax=None, add_labels=True, dither_point_index=1, msa_metadata_id=None):
        """
        Make a plot showing a slitlet
        
        Parameters
        ----------
        source_id : int
            Source id, must be in ``src_table``
        
        cutout_size : float
            Cutout half-width, arcsec
        
        step : int
            Place to mark axis labels, defaults to ``floor(cutout_size)``
        
        rgb_filters : list, None
            List of filters to use for an RGB cutout
        
        figsize : tuple
            Size if generating a new figure
        
        ax : `~matplotlib.axes._subplots.AxesSubplot`, None
            Plot axis
        
        add_labels : bool
            Add plot labels
        
        dither_point_index : int
            Dither to show
        
        msa_metadata_id : int
            Optional specified ``msa_metadata_id`` in ``shutter_table``
        
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
        
        
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import PIL
        from urllib.request import urlopen
        
        if (rgb_filters is not None) & (cutout_size > 20):
            raise ValueError('Maximum size of 20" with the thumbnail')
            
        if source_id not in self.src_table['source_id']:
            print(f'{source_id} not in src_table for {self.metafile}')
            return None
        
        ix = np.where(self.src_table['source_id'] == source_id)[0][0]
        ra = self.src_table['ra'][ix]
        dec = self.src_table['dec'][ix]
        cosd = np.cos(dec/180*np.pi)

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=figsize)
        else:
            fig = None
        
        extent = (ra + cutout_size/3600/cosd, ra - cutout_size/3600./cosd,
                  dec - cutout_size/3600., dec + cutout_size/3600.)
        
        #cutout_size = 2.5 # arcsec
        if rgb_filters is not  None:
            cutout = 'https://grizli-cutout.herokuapp.com/thumb?coords={ra},{dec}'
            cutout += '&filters=' + ','.join(rgb_filters)
            cutout += '&size={cutout_size}&scl=5'
            url = cutout.format(ra=ra, dec=dec, cutout_size=cutout_size)
        
            rgb = np.array(PIL.Image.open(urlopen(url)))
            ax.imshow(rgb, origin='lower', extent=extent, interpolation='Nearest')
            src_color = 'w'
        else:
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            src_color = 'k'
    
        #ax.set_aspect(cosd)
    
        ax.scatter(self.src_table['ra'], self.src_table['dec'],
                   marker='o', fc='None', ec=src_color, alpha=0.5)
        
        slits = self.regions_from_metafile(dither_point_index=dither_point_index,
                                           as_string=False,
                                           with_bars=True,
                                           msa_metadata_id=msa_metadata_id)
        for s in slits:
            if s.meta['is_source']:
                kws = dict(color=src_color, alpha=0.8, zorder=100)
            else:
                kws = dict(color='0.7', alpha=0.8, zorder=100)
            
            ax.plot(*np.vstack([s.xy[0], s.xy[0][:1,:]]).T, **kws)
        
        if step is None:
            step = int(np.floor(cutout_size))
        
        xt = np.array([-step, 0, step])/3600./cosd + ra
        yt = np.array([-step, 0, step])/3600. + dec
        
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        
        if fig is not None:
            ax.set_yticklabels([f'-{step}"', 'Dec.', f'+{step}"'])
            ax.set_xticklabels([f'+{step}"', 'R.A.', f'-{step}"'])
        
        ax.set_xlim(ra + np.array([cutout_size,-cutout_size])/3600./cosd)
        ax.set_ylim(dec + np.array([-cutout_size,cutout_size])/3600.)
        
        ax.set_aspect(1./cosd)
        ax.grid()
        
        if add_labels:
            ax.text(0.03, 0.07, f'Dither #{dither_point_index}',
                    ha='left', va='bottom',
                    transform=ax.transAxes, color=src_color, fontsize=8)
            ax.text(0.03, 0.03, f'{os.path.basename(self.metafile)}',
                    ha='left', va='bottom',
                    transform=ax.transAxes, color=src_color, fontsize=8)
            ax.text(0.97, 0.07, f'{source_id}',
                    ha='right', va='bottom',
                    transform=ax.transAxes, color=src_color, fontsize=8)
            ax.text(0.97, 0.03, f'({ra:.6f}, {dec:.6f})',
                    ha='right', va='bottom',
                    transform=ax.transAxes, color=src_color, fontsize=8)
                    
        #fig.tight_layout(pad=1)
        
        return fig, ax
        