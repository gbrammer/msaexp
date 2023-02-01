"""
Helper scripts for dealing with MSA metadata files (``MSAMETFL``)
"""
import os
import numpy as np
import astropy.io.fits as pyfits


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

    new_rows = []

    for src_id in source_ids:
        six = shut['source_id'] == src_id
        shutters = np.unique(shut['shutter_column'][six])
        quad = shut['shutter_quadrant'][six][0]
        slit_id = shut['slitlet_id'][six][0]
    
        for mid in np.unique(shut['msa_metadata_id'][six]):
            mix = (shut['msa_metadata_id'] == mid) & (six)
            for eid in np.unique(shut['dither_point_index'][mix]):
                for p in range(pad):
                    for s in [shutters.min()-(p+1), shutters.max()+(p+1)]:
                        row['msa_metadata_id'] = mid
                        row['dither_point_index'] = eid
                        row['shutter_column'] = s
                        row['shutter_quadrant'] = quad
                        row['slitlet_id'] = slit_id
                        row['source_id'] = src_id
                        nrow = {}
                        for k in row:
                            nrow[k] = row[k]
                        new_rows.append(nrow)
                
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
    """
    metf = MSAMetafile(metafile)
    regions = metf.regions_from_metafile(**kwargs)
    return regions


def regions_from_fits(file, **kwargs):
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
        """
        import grizli.utils

        self.metafile = metafile
        
        with pyfits.open(metafile) as im:
            src = grizli.utils.GTable(im['SOURCE_INFO'].data)
            shut = grizli.utils.GTable(im['SHUTTER_INFO'].data)
    
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
        Fit for transforms between slit (row, col) and (ra, dec)
        
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
        
        dith = (self.shutter_table['dither_point_index'] == dither_point_index) 
        metid = (self.shutter_table['msa_metadata_id'] == msa_metadata_id)
        exp = dith & metid
    
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
        
        return dith, metid, coeffs, inv_coeffs


    def regions_from_metafile(self, as_string=False, with_bars=True, **kwargs):
        """
        Get slit footprints in sky coords
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
    
        sx = (np.array([-0.5, 0.5, 0.5, -0.5]))*(1-0.07/0.27*with_bars) + 0.5
        sy = (np.array([-0.5, -0.5, 0.5, 0.5]))*(1-0.07/0.53*with_bars) + 0.5

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
