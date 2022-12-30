"""
Manual extractions of NIRSpec MSA spectra
"""

import os
import glob
import time
import traceback
from collections import OrderedDict

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as pyfits
import astropy.units as u
from astropy.units.equivalencies import spectral_density
from itertools import permutations

import jwst

from grizli import utils, prep, jwst_utils
utils.set_warnings()

FLAM_UNIT = 1.e-19*u.erg/u.second/u.cm**2/u.Angstrom
FNU_UNIT = u.microJansky

GRATINGS = ['prism','g140m','g140h','f235m','g235h','g395m','g395h']
FILTERS = ['clear', 'f070lp','f100lp','f170lp','f290lp']
ACQ_FILTERS = ['f140x','f110w']
DETECTORS = ['nrs1','nrs2']

def query_program(prog=2767, download=True, detectors=DETECTORS, gratings=GRATINGS, filters=FILTERS, extensions=['s2d']):
    """
    Query and download MSA exposures for a given program
    """
    import mastquery.jwst
    import mastquery.utils
    
    query = []
    query += mastquery.jwst.make_query_filter('productLevel', 
                                                values=['2','2a','2b'])
    query += mastquery.jwst.make_program_filter([prog])
    
    if detectors is not None:
        query += mastquery.jwst.make_query_filter('detector', values=detectors)
    
    if gratings is not None:
        query += mastquery.jwst.make_query_filter('grating', values=gratings)
    
    if filters is not None:
        query += mastquery.jwst.make_query_filter('filter', values=filters)
        
    res = mastquery.jwst.query_jwst(instrument='NRS',
                                    filters=query, 
                                    extensions=extensions,
                                    rates_and_cals=False)
    
    if len(res) == 0:
        print('Nothing found.')
        return None
        
    # Unique rows
    rates = []
    unique_indices = []
    
    for i, u in enumerate(res['dataURI']):
        ui = u.replace('s2d','rate')
        for e in extensions:
            ui = ui.replace(e, 'rate')
        
        if ui not in rates:
            unique_indices.append(i)
        
        rates.append(ui)
    
    res.remove_column('dataURI')
    res['dataURI'] = rates
    
    res = res[unique_indices]

    skip = np.in1d(res['msametfl'], [None])
    if skip.sum() > 0:
        print(f'Remove {skip.sum()} rows with msametfl=None')
        res = res[~skip]
        
    skip = np.in1d(res['filter'], ['OPAQUE'])
    if skip.sum() > 0:
        print(f'Remove {skip.sum()} rows with filter=OPAQUE')
        res = res[~skip]
        
    if download:
        mastquery.utils.download_from_mast(rates[0:])
        
        download_msa_meta_files()
    
    return res


def download_msa_meta_files():
    """
    """
    import mastquery.utils

    files = glob.glob('*rate.fits')
    msa = []
    for file in files:
        with pyfits.open(file) as im:
            msa_file = im[0].header['MSAMETFL']
            if not os.path.exists(msa_file):
                msa.append(f'mast:JWST/product/{msa_file}')

    if len(msa) > 0:
        mastquery.utils.download_from_mast(msa)


def exposure_groups(path='./', verbose=True):
    """
    Group by MSAMETFL, grating, filter, detector
    """

    files = glob.glob('*rate.fits')
    files.sort()
    keys = ['filter','grating','effexptm','detector', 'msametfl']
    rows = []
    for file in files:
        with pyfits.open(file) as im:
            row = [file] + [im[0].header[k] for k in keys]
            rows.append(row)

    tab = utils.GTable(names=['file']+keys, rows=rows)
    keystr = "{msametfl}-{filter}-{grating}-{detector}"
    tab['key'] = [keystr.format(**row).lower() for row in tab]
    tab['key'] = [k.replace('_msa.fits','').replace('_','-')
                  for k in tab['key']]

    un = utils.Unique(tab['key'], verbose=verbose)

    groups = OrderedDict()
    for v in un.values:
        groups[v] = [f for f in tab['file'][un[v]]]
    
    return groups


class SlitData():
    """
    Container for a list of SlitModel objects read from saved files
    """
    def __init__(self, file='jw02756001001_03101_00001_nrs1_rate.fits', step='phot', verbose=True, read=False, indices=None, targets=None):
        """
        Load saved slits
        """

        self.slits = []
        
        if indices is not None:
            self.files = []
            for i in indices:
                self.files += glob.glob(file.replace('rate.fits',
                                        f'{step}.{i:03d}.*.fits'))
        elif targets is not None:
            self.files = []
            for target in targets:
                self.files += glob.glob(file.replace('rate.fits',
                                        f'{step}.*{target}.fits'))
        else:
            self.files = glob.glob(file.replace('rate.fits', f'{step}.*.fits'))
                
        self.files.sort()
        
        if read:
            self.read_data()


    @property
    def N(self):
        return len(self.files)


    def read_data(self, verbose=True):
        """
        Read files into SlitModel objects
        """
        from jwst.datamodels import SlitModel

        for file in self.files:
            self.slits.append(SlitModel(file))
            msg = f'msaexp.read_data: {file} {self.slits[-1].source_name}'
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose, 
                              show_date=False)


class NirspecPipeline():
    def __init__(self, mode='jw02767005001-02-clear-prism-nrs1', files=None, verbose=True):
        """
        """
        self.mode = mode
        utils.LOGFILE = self.mode + '.log.txt'
        
        if files is None:
            groups = exposure_groups(verbose=False)
            if mode in groups:
                self.files = groups[mode]
            else:
                self.files = []
        else:
            self.files = files
        
        msg = f'msaexp.NirspecPipeline: Initialize {mode}'
        utils.log_comment(utils.LOGFILE, msg, verbose=True, 
                              show_date=True)
        
        for file in self.files:
            msg = f'msaexp.NirspecPipeline: {file}'
            utils.log_comment(utils.LOGFILE, msg, verbose=True, 
                              show_date=False)
                              
        self.pipe = OrderedDict()
        self.slitlets = OrderedDict()
        
        self.last_step = None


    @property
    def grating(self):
        return '-'.join(self.mode.split('-')[-3:-1])


    @property
    def detector(self):
        return self.mode.split('-')[-1]


    @property
    def N(self):
        return len(self.files)


    @property
    def targets(self):
        """
        Reformatted target name:
        
        background_{i} > b{i}
        xxx_-{i} > xxx_m{i}
        
        """
        return list(self.slitlets.keys())


    def slit_index(self, key):
        """
        Index of ``key`` in ``self.slitlets``
        """
        if key not in self.slitlets:
            return None
        else:
            return self.targets.index(key)


    def preprocess(self, set_context=True):
        """
        Run grizli exposure-level preprocessing
        
        1. snowball masking
        2. 1/f correction
        3. median "bias" removal
        
        """
        
        if set_context:
            # Set CRDS_CTX to match the exposures
            if (os.getenv('CRDS_CTX') is None) | (set_context > 1):
                with pyfits.open(self.files[0]) as im:
                    _ctx = im[0].header["CRDS_CTX"]
                    
                msg = f'msaexp.preprocess : set CRDS_CTX={_ctx}'
                utils.log_comment(utils.LOGFILE, msg, verbose=True, 
                                  show_date=True)
                
                os.environ['CRDS_CTX'] = _ctx
                
        # Extra mask for snowballs
        prep.mask_snowballs({'product':self.mode, 'files':self.files},
                             mask_bit=1024, instruments=['NIRSPEC'], 
                             snowball_erode=8, snowball_dilate=24) 

        # 1/f correction
        for file in self.files:
            jwst_utils.exposure_oneoverf_correction(file, erode_mask=False, 
                                    in_place=True, axis=0,
                                    deg_pix=256)
        
        # bias
        for file in self.files:
            with pyfits.open(file, mode='update') as im:
                dq = (im['DQ'].data & 1025) == 0
                bias_level = np.nanmedian(im['SCI'].data[dq])
                msg = f'msaexp.preprocess : bias level {file} ='
                msg += f' {bias_level:.4f}'
                utils.log_comment(utils.LOGFILE, msg, verbose=True, 
                                  show_date=True)
            
                im['SCI'].data -= bias_level
                im.flush()
            
        return True
    
    
    def run_jwst_pipeline(self, verbose=True):
        """
        Steps taken from https://github.com/spacetelescope/jwebbinar_prep/blob/main/spec_mode/spec_mode_stage_2.ipynb

        See also https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html#calwebb-spec2

        AssignWcs - initialize WCS and populate slit bounding_box data
        Extract2dStep - identify slits and set slit WCS
        FlatFieldStep - slit-level flat field
        PathLossStep - NIRSpec path loss
        PhotomStep - Photometric calibration
        """
        # AssignWcs
        import jwst.datamodels
        from jwst.assign_wcs import AssignWcsStep, nirspec
        from jwst.imprint import ImprintStep
        from jwst.msaflagopen import MSAFlagOpenStep
        from jwst.extract_2d import Extract2dStep
        from jwst.flatfield import FlatFieldStep
        from jwst.pathloss import PathLossStep
        from jwst.barshadow import BarShadowStep
        from jwst.photom import PhotomStep

        if 'wcs' not in self.pipe:
            wstep = AssignWcsStep()
            self.pipe['wcs'] = [wstep.call(jwst.datamodels.ImageModel(f))
                                for f in self.files]
        self.last_step = 'wcs'

        # step = ImprintStep()
        # pipe['imp'] = [step.call(obj) for obj in pipe[last]]
        # last = 'imp'

        if 'open' not in self.pipe:
            step = MSAFlagOpenStep()
            self.pipe['open'] = []
            
            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f'msaexp.jwst.MSAFlagOpenStep: {self.files[i]}'
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose, 
                                  show_date=True)
                
                self.pipe['open'].append(step.call(obj))
                
        self.last_step = 'open'

        if '2d' not in self.pipe:
            step2d = Extract2dStep()
            self.pipe['2d'] = []
            
            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f'msaexp.jwst.Extract2dStep: {self.files[i]}'
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose, 
                                  show_date=True)
                
                self.pipe['2d'].append(step2d.call(obj))
        
        self.last_step = '2d'

        if 'flat' not in self.pipe:
            flat_step = FlatFieldStep()
            self.pipe['flat'] = []
            
            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f'msaexp.jwst.FlatFieldStep: {self.files[i]}'
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose, 
                                  show_date=True)
                
                self.pipe['flat'].append(flat_step.call(obj))
                       
        self.last_step = 'flat'

        if 'path' not in self.pipe:
            path_step = PathLossStep()
            self.pipe['path'] = []
            
            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f'msaexp.jwst.PathLossStep: {self.files[i]}'
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose, 
                                  show_date=True)
                
                self.pipe['path'].append(path_step.call(obj))
        
        self.last_step = 'path'

        # Skip BarShadow
        
        if 'phot' not in self.pipe:
            phot_step = PhotomStep()
            self.pipe['phot'] = []
            
            for i, obj in enumerate(self.pipe[self.last_step]):
                msg = f'msaexp.jwst.PhotomStep: {self.files[i]}'
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose, 
                                  show_date=True)
                
                self.pipe['phot'].append(phot_step.call(obj))
        
        self.last_step = 'phot'
        
        return True


    def save_slit_data(self, step='phot', verbose=True):
        """
        Save slit data to FITS
        """
        from jwst.datamodels import SlitModel
        
        for j in range(self.N):
            for _name in self.slitlets:
                
                i = self.slitlets[_name]['slit_index']
                
                slit_file = self.files[j].replace('rate.fits',
                                                f'{step}.{i:03d}.{_name}.fits')
                                                  
                msg = f'msaexp.save_slit_data: {slit_file} '
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose, 
                                  show_date=False)

                try:
                    dm = SlitModel(self.pipe[step][j].slits[i].instance)
                    dm.write(slit_file, overwrite=True)
                except:
                    utils.log_comment(utils.LOGFILE, 'Failed',
                                      verbose=verbose)
                
        return True


    def initialize_slit_metadata(self, use_yaml=True, yoffset=0, prof_sigma=1.8/2.35, skip=[]):
        """
        """
        import yaml
        from . import utils as msautils
        
        slitlets = OrderedDict()
        
        if self.last_step not in ['2d','flat','path','phot']:
            return slitlets
            
        msg = '# slit_index slitlet_id  source_name  source_ra  source_dec'
        msg += f'\n# {self.mode}'
        utils.log_comment(utils.LOGFILE, msg, verbose=True, show_date=False)
        
        # Load saved data
        yaml_file = f'{self.mode}.slits.yaml'
        if os.path.exists(yaml_file) & use_yaml:
            msg = f'# Get slitlet data from {yaml_file}'
            utils.log_comment(utils.LOGFILE, msg, verbose=True, 
                              show_date=False)
            
            with open(yaml_file) as fp:
                yaml_data = yaml.load(fp, Loader=yaml.Loader)
        else:
            yaml_data = {}
                
        for i in range(len(self.pipe[self.last_step][0].slits)):
            bg = []
            src = 0
            meta = None

            for ii, o in enumerate(self.pipe[self.last_step]):
                _slit = o.slits[i]
                if _slit.source_name.startswith('background'):
                    bg.append(ii)
                    bgmeta = _slit.instance
                else:
                    src = ii
                    meta = _slit.instance

            if meta is None:
                meta = bgmeta
                meta['slit_index'] = i
                meta['is_background'] = True
            else:
                meta['slit_index'] = i    
                meta['is_background'] = False
            
            meta['bkg_index'] = bg
            meta['src_index'] = src
            meta['slit_index'] = i    
            meta['prof_sigma'] = prof_sigma
            meta['yoffset'] = yoffset
            meta['skip'] = skip
            meta['redshift'] = None

            _name = msautils.rename_source(meta['source_name'])
            if _name in yaml_data:
                for k in meta:
                    if k in yaml_data[_name]:
                        meta[k] = yaml_data[_name][k]
            
            msg = '{slit_index:>4}  {slitlet_id:>4} {source_name:>12} '
            msg += ' {source_ra:.6f} {source_dec:.6f}'
            utils.log_comment(utils.LOGFILE, msg.format(**meta), 
                              verbose=True, show_date=False)
            
            #_name =  meta['source_name'].replace('background_','b')
            #_name = _name.replace('_-','_m')
            slitlets[_name] = {}
            for k in meta:
                slitlets[_name][k] = meta[k]

        return slitlets


    def slit_source_regions(self, color='magenta'):
        """
        Make region file of source positions
        """
        regfile = f'{self.mode}.reg'
        with open(regfile, 'w') as fp:
            fp.write(f'# {self.mode}\n')
            fp.write(f'global color={color}\nicrs\n')

            for s in self.slitlets:
                row = self.slitlets[s]
                if row['source_ra'] != 0:
                    ss = 'circle({source_ra:.6f},{source_dec:.6f},0.3") # '
                    ss += ' text={{{source_name}}}\n'
                    fp.write(ss.format(**row))
                
        return regfile


    def set_background_slits(self):
        """
        """
        from tqdm import tqdm
        from jwst.datamodels import SlitModel
        
        self.pipe['bkg'] = self.load_slit_data(step=self.last_step,
                                               targets=self.targets)
        
        # self.pipe['bkg'] = []
        #
        # for j in tqdm(range(self.N)):
        #     self.pipe['bkg'][j].slits = []
        #     for i in range(len(self.pipe['phot'][j].slits)):
        #         s = self.pipe['phot'][j].slits[i].copy()
        #         s.has_background = False
        #         self.pipe['bkg'][j].slits.append(s)
                                                   
        for j in range(self.N):
            for s in self.pipe['bkg'][j].slits:
                s.has_background = False
                
        return True


    def fit_profile(self, key, yoffset=None, prof_sigma=None, bounds=[(-5,5), (1.4/2.35, 3.5/2.35)], min_delta=100, use_huber=True, verbose=True, **kwargs):
        """
        Fit for profile width and offset
        """
        from photutils.psf import IntegratedGaussianPRF
        from scipy.optimize import minimize
        from . import utils as msautils
        
        prf = IntegratedGaussianPRF(sigma=2.8/2.35, x_0=0, y_0=0)
        
        if key.startswith('background'):
            bounds[0] = (-8,8)
            
        # print('xxx', bounds)
        
        _slit_data = self.extract_spectrum(key, 
                                          fit_profile_params={},
                                      flux_unit=FLAM_UNIT,
                                      pstep=1, get_slit_data=True,
                                      yoffset=yoffset, prof_sigma=prof_sigma,
                                      show_sn=True, verbose=False)
        
        _slit, _clean, _ivar, prof, y1, _wcs, chi, bad = _slit_data
        
        slitlet = self.slitlets[key]
               
        sh = _clean.shape
        yp, xp = np.indices(sh)
        x = xp[0,:]
        _dq = ~bad
        # _ra = slitlet['source_ra']
        # _dec = slitlet['source_dec']
        #
        # rs, ds, ws = _wcs.forward_transform(x,
        #                              x*0+np.nanmean(yp[_dq]))
        # if _ra <= 0:
        #     x0 = np.nanmean(xp[_dq])
        #     y0 = np.nanmean(yp[_dq])
        #     _ra, _dec, ws = _wcs.forward_transform(x0, y0)
        #
        # xtr, ytr = _wcs.backward_transform(_ra, _dec, ws)
        # yeval = np.nanmean(ytr)
        # dytest = [0]
        # for _dy in range(sh[0]):
        #     dytest.extend([-_dy, _dy])
        #
        # for dy in dytest:
        #     rs, ds, ws = _wcs.forward_transform(x,
        #                                         x*0+yeval+dy)
        #     if (~np.isfinite(ws)).sum() == 0:
        #         break
        #
        # xtr, ytr = _wcs.backward_transform(_ra, _dec, ws)
        # ytr -= 0.5
        _res = msautils.slit_trace_center(_slit, 
                                  with_source_ypos=True, 
                                  index_offset=0.5)
    
        xd, yd, _w, _, _ = _res
        
        xtr = xd
        ytr = slitlet['ytrace']*2 - yd
        
        def _objfun_fit_profile(params, data, ret):
            """
            Loss function for fitting profile parameters

            params: yoffset, prof_sigma
            """
            from scipy.special import huber

            yoff = params[0]
            prf.sigma = params[1]

            xx0, yp, ytr, sh, _clean, _ivar, bad = data
            prof = prf(xx0, (yp - ytr - yoff).flatten()).reshape(sh)

            _wht = (prof**2*_ivar).sum(axis=0)
            y1 = (_clean*prof*_ivar).sum(axis=0) / _wht
            dqi = (~bad) & (_wht > 0)
            
            #_sys = 1/(1/_ivar + (0.02*_clean)**2)
            
            y1[y1 < 0] = 0
            
            chi = (_clean - prof*y1)*np.sqrt(_ivar)
            #chi[prof*y1*np.sqrt(_ivar) < -5] *= 10
            chi2 = (chi[dqi]**2).sum()

            if ret == 0:
                return chi
            elif ret == 1:
                #print(params, chi2)
                return chi2
            elif ret == 2:
                loss = huber(3, chi)[dqi].sum()
                #print(params, loss)
                #loss +=  (y1 < 0).sum() - y1[np.isfinite(y1)].max()
                return loss
            else:
                return y1*prof
        
        xx0 = yp.flatten()*0.
        data = (xx0, yp, ytr, sh, _clean, _ivar, bad)
        
        # compute dchi2 / dy and only do the fit if this is 
        # greater than some threshold min_delta
        x0 = [slitlet['yoffset']*1., slitlet['prof_sigma']*1.]
        x1 = [slitlet['yoffset']*1.+1, slitlet['prof_sigma']*1.]
        chi0 = _objfun_fit_profile(x0, data, 1+use_huber)
        chi1 = _objfun_fit_profile(x1, data, 1+use_huber)
                
        d0 = np.abs(chi1-chi0)

        if d0 > min_delta:
            _res = minimize(_objfun_fit_profile, x0,
                            args=(data, 1+use_huber),
                            method='slsqp', 
                            bounds=bounds,
                            jac='2-point',
                        options={'direc':np.eye(2,2)*np.array([0.5, 0.2])})

            dx = chi0 - _res.fun
            
            # print('xxx', x0, chi0, x1, chi1, _res.fun, _res)
            
            msg = f'msaexp.fit_profile:     '
            msg += f' {key:<20}  (dchi2 = {d0:8.1f})'
            msg += f' yoffset = {_res.x[0]:.2f}  prof_sigma = {_res.x[1]:.2f}'
            msg += f' dchi2 = {dx:8.1f}'
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

            return _res.x, _res

        else:
            msg = f'msaexp.fit_profile:     '
            msg += f' {key:<20}  (dchi2 = {d0:8.1f} <'
            msg += f' {min_delta} - skip)  yoffset = {x0[0]:.2f} '
            msg += f' prof_sigma = {x0[1]:.2f}'
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            return x0, None


    def get_background_slits(self, key, step='bkg', check_background=True, **kwargs):
        """
        Get background-subtracted slitlets
        
        Returns
        -------
        slits : list
            List of `jwst.datamodels.slit.SlitModel` objects
            
        """
        if step not in self.pipe:
            return None
                
        if key not in self.slitlets:
            return None
        
        slits = [] 
            
        slitlet = self.slitlets[key]
        i = self.slit_index(key) #slitlet['slit_index']
        
        for j in range(self.N):
            bsl = self.pipe[step][j].slits[i]
            if check_background:
                if hasattr(bsl, 'has_background'):
                    if bsl.has_background:
                        slits.append(bsl)
            else:
                slits.append(bsl)
        
        return slits


    def drizzle_2d(self, key, drizzle_params={}, **kwargs):
        """
        """
        from jwst.datamodels import SlitModel, ModelContainer
        from jwst.resample.resample_spec import ResampleSpecData

        slits = self.get_background_slits(key, **kwargs)
        if slits in [None, []]:
            return None
        
        bcont = ModelContainer()
        for s in slits:
            bcont.append(s)
            
        step = ResampleSpecData(bcont, **drizzle_params)
        result = step.do_drizzle()
        
        return result


    def get_slit_traces(self, verbose=True):
        """
        Set center of slit traces in `slitlets`
        """
        from . import utils as msautils
        
        msg = f'msaexp.get_slit_traces: Run'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose, 
                          show_date=True)
        
        for key in self.slitlets:
            i = self.slit_index(key)
            dith_ref = 1000
            for j in range(self.N):
                slit = self.pipe[self.last_step][j].slits[i]
                dith = slit.meta.dither.instance
                
                #if dith['position_number'] == 1:
                # find lowest position number
                if dith['position_number'] < dith_ref:
                    
                    dith_ref = dith['position_number']
                    jref = j
                    
                    _res = msautils.slit_trace_center(slit, 
                                       with_source_ypos=True, index_offset=0.5)
                    
                    xtr, ytr, wtr, slit_ra, slit_dec = _res
                    
                    self.slitlets[key]['xtrace'] = xtr
                    self.slitlets[key]['ytrace'] = ytr
                    self.slitlets[key]['wtrace'] = wtr
                    
                    self.slitlets[key]['slit_ra'] = slit_ra
                    self.slitlets[key]['slit_dec'] = slit_dec
                    
                    #break
            
            msg = 'msaexp.get_slit_traces: '
            
            #if j == self.N-1:
            #    msg += f'! no index position found for {key}'
            #else:
            msg += f'Trace set at index {jref} for {key}'

            utils.log_comment(utils.LOGFILE, msg, verbose=verbose, 
                              show_date=False)


    def extract_spectrum(self, key, slit_key=None, prof_sigma=None, fit_profile_params={'min_delta':100}, pstep=1.0, show_sn=True, flux_unit=FNU_UNIT, vmax=0.2, yoffset=None, skip=None, bad_dq_bits=(1 | 1024), clip_sigma=-4, ntrim=5, get_slit_data=False, verbose=False, center2d=False, trace_sign=1, min_dyoffset=0.2, **kwargs):
        """
        Extract 2D spectrum
        """                
        from gwcs import wcstools
        from photutils.psf import IntegratedGaussianPRF
        import eazy.utils
        
        from . import utils as msautils
        
        if key not in self.slitlets:
            print(f'{key} not found in slitlets')

            return None
        
        if fit_profile_params:
            try:
                _xprof, _res = self.fit_profile(key, **fit_profile_params)
            except TypeError:
                _res = None
                
            if _res is not None:
                if _res.success:
                    yoffset = None
                    prof_sigma = None
                    self.slitlets[key]['yoffset'] = float(_res.x[0])
                    self.slitlets[key]['prof_sigma'] = float(_res.x[1])
                
        slitlet = self.slitlets[key]
        #slitlet['bkg_index'], slitlet['src_index'], slitlet['slit_index']

        i = self.slit_index(key) #slitlet['slit_index']
        
        for j in range(self.N):
            if 'bkg' in self.pipe:
                self.pipe['bkg'][j].slits[i].has_background = False
            
        if slit_key is None:
            slit_key = self.last_step

        if yoffset is None:
            yoffset = slitlet['yoffset']
        else:
            slitlet['yoffset'] = yoffset

        if prof_sigma is None:
            prof_sigma = slitlet['prof_sigma']
        else:
            slitlet['prof_sigma'] = prof_sigma
                
        prf = IntegratedGaussianPRF(sigma=prof_sigma,
                                    x_0=0.0,
                                    y_0=0.0, 
                                    flux=prof_sigma*np.sqrt(2*np.pi))
        
        if skip is None:
            skip = slitlet['skip']
        else:
            slitlet['skip'] = skip

        pipe = self.pipe[slit_key]
        
        if not get_slit_data:
            heights = [1]*self.N + [2]
            fig, axes = plt.subplots(self.N+1, 1,
                                 figsize=(12,self.N+2),
                                 sharex=True, sharey=False,
                                 gridspec_kw={'height_ratios':heights})

            a1 = axes[-1]

        sci_slit = pipe[slitlet['src_index']].slits[i]

        x = np.arange(sci_slit.data.shape[1])
        rs, ds, ws = sci_slit.meta.wcs.forward_transform(x, x*0+5)

        tabs = []
        yavj = None

        all_sci = []
        all_ivar = []
        all_prof = []

        _ra = None
        xref = yref = None
        
        #for ip, p in enumerate(permutations(range(self.N), self.N)):
        for ip in range(self.N):
            if ip in skip:
                continue
                
            # First exposure
            _slit = pipe[ip].slits[i]
                        
            if 'bkg' in self.pipe:
                _bkg_slit = self.pipe['bkg'][ip].slits[i]
            else:
                _bkg_slit = None
                
            sh = _slit.data.shape

            _dq = (pipe[ip].slits[i].dq & bad_dq_bits) == 0
            _sci = pipe[ip].slits[i].data
            _ivar = 1/pipe[ip].slits[i].err**2
            _dq &= _sci*np.sqrt(_ivar) > clip_sigma
            _sci[~_dq] = 0
            _ivar[~_dq] = 0

            _bkg = _sci*0.
            _bkgn = _sci*0.

            _wcs = _slit.meta.wcs
            d2w = _wcs.get_transform('detector', 'world')

            yp, xp = np.indices(sh)
            
            _res = msautils.slit_trace_center(_slit, 
                                      with_source_ypos=True, 
                                      index_offset=0.5)
        
            xd, yd, _w, _, _ = _res
            
            xtr = xd
            if trace_sign > 0:
                ytr = slitlet['ytrace']*2 - yd + yoffset
            else:
                ytr = yd + yoffset
            
            _ras, _des, ws = d2w(xtr, ytr)                 
            
            for j in range(self.N):
                if j == ip:
                    continue
                    
                _sbg = pipe[j].slits[i]
                dy_off = (_sbg.meta.dither.y_offset -
                          _slit.meta.dither.y_offset)
                          
                if np.abs(dy_off) < min_dyoffset:
                    continue
                    
                _dq = (_sbg.dq & bad_dq_bits) == 0
                try:
                    _bkg += _sbg.data*_dq
                    _bkgn += _dq
                except ValueError:
                    #print('background failed', j)
                    continue

            # _sci = None
            # if ip % (self.N-1) == 0:
            #     if (ip//(self.N-1)) in skip:
            #         continue
            #
            #     axi = axes[ip//(self.N-1)]
            #     for j in p:
            #         if _sci is None:
                    #     _slit = pipe[j].slits[i]
                    #
                    #     if 'bkg' in self.pipe:
                    #         _bkg_slit = self.pipe['bkg'][j].slits[i]
                    #     else:
                    #         _bkg_slit = None
                    #
                    #     sh = _slit.data.shape
                    #
                    #     _dq = (pipe[j].slits[i].dq & bad_dq_bits) == 0
                    #     _sci = pipe[j].slits[i].data
                    #     _ivar = 1/pipe[j].slits[i].err**2
                    #     _dq &= _sci*np.sqrt(_ivar) > clip_sigma
                    #     _sci[~_dq] = 0
                    #     _ivar[~_dq] = 0
                    #
                    #     _bkg = _sci*0.
                    #     _bkgn = _sci*0.
                    #
                    #     _wcs = _slit.meta.wcs
                    #     d2w = _wcs.get_transform('detector', 'world')
                    #
                    #     yp, xp = np.indices(sh)
                    #
                    #     _res = msautils.slit_trace_center(_slit,
                    #                               with_source_ypos=True,
                    #                               index_offset=0.5)
                    #
                    #     xd, yd, _w, _, _ = _res
                    #
                    #     xtr = xd
                    #     if trace_sign > 0:
                    #         ytr = slitlet['ytrace']*2 - yd + yoffset
                    #     else:
                    #         ytr = yd + yoffset
                    #
                    #     # # Trace along expected source position
                    #     # s2d = _wcs.get_transform('slit_frame', 'detector')
                    #     # d2s = _wcs.get_transform('detector', 'slit_frame')
                    #     # s2w = _wcs.get_transform('slit_frame', 'world')
                    #     # d2w = _wcs.get_transform('detector', 'world')
                    #     #
                    #     # bbox = _wcs.bounding_box
                    #     # grid = wcstools.grid_from_bounding_box(bbox)
                    #     # _, sy, slam = np.array(d2s(*grid))
                    #     #
                    #     # smi = np.nanmin(sy)
                    #     # sma = np.nanmax(sy)
                    #     #
                    #     # x = np.arange(sh[1], dtype=float)
                    #     # xd, yd = x, x*0.+sh[0]//2
                    #     #
                    #     # for _iter in range(3):
                    #     #     xs, ys, ls = d2s(xd, yd)
                    #     #     xd, yd = s2d(x*0, x*0. + _slit.source_ypos, ls)
                    #     #     yd = np.interp(x, xd, yd)
                    #     #     xd = x
                    #     #
                    #     # dith = _slit.meta.dither.instance
                    #     # if dith['position_number'] == 1:
                    #     #     # Center world coord in slit
                    #     #     rs, ds, _ = s2w(0, _slit.source_ypos, np.median(ls))
                    #     #     slitlet['slit_ra'] = rs
                    #     #     slitlet['slit_dec'] = ds
                    #     #
                    #     #     xref = xd*1.
                    #     #     yref = yd*1.
                    #     #
                    #     #     xtr = x*1
                    #     #     ytr = yref + yoffset + 0.5
                    #     #
                    #     # else:
                    #     #
                    #     #     ytr = yref*2 - yd + yoffset + 0.5
                    #
                    #     _ras, _des, ws = d2w(xtr, ytr)
                    #
                    #     # rs, ds, ws = _wcs.forward_transform(x,
                    #     #                              x*0+np.nanmean(yp[_dq]))
                    #     # if _ra is None:
                    #     #     if (slitlet['source_ra'] > 0):
                    #     #         _ra = slitlet['source_ra']
                    #     #         _dec = slitlet['source_dec']
                    #     #     else:
                    #     #         x0 = np.nanmean(xp[_dq])
                    #     #         y0 = np.nanmean(yp[_dq])
                    #     #         _ra, _dec, w0 = _wcs.forward_transform(x0, y0)
                    #     #         msg = f'msaexp.extract_spectrum: Set '
                    #     #         msg += f' source_ra/source_dec: '
                    #     #         msg += f'{_ra}, {_dec}'
                    #     #         utils.log_comment(utils.LOGFILE, msg,
                    #     #                           verbose=True)
                    #     #
                    #     #         slitlet['source_ra'] = _ra
                    #     #         slitlet['source_dec'] = _dec
                    #     #
                    #     # xtr, ytr = _wcs.backward_transform(_ra, _dec, ws)
                    #     # yeval = np.nanmean(ytr)
                    #     #
                    #     # dytest = [0]
                    #     # for _dy in range(sh[0]):
                    #     #     dytest.extend([-_dy, _dy])
                    #     #
                    #     # for dy in dytest:
                    #     #     rs, ds, ws = _wcs.forward_transform(x,
                    #     #                                         x*0+yeval+dy)
                    #     #     if verbose:
                    #     #         print('dy: ', (~np.isfinite(ws)).sum(), dy)
                    #     #
                    #     #     if (~np.isfinite(ws)).sum() == 0:
                    #     #         if verbose:
                    #     #             print('y center: ', dy)
                    #     #         break
                    #     #
                    #     # xtr, ytr = _wcs.backward_transform(_ra, _dec, ws)
                    #     # ytr -= 0.5
                    #     # ytr += yoffset
                    #     # xtr = x*1
                    #     #
                    #     # # Todo - calculate ws again with offset?
                    #     # rx, dx, ws = _wcs.forward_transform(xtr, ytr)
                    #
                    # else:
                    #     _sbg = pipe[j].slits[i]
                    #     dy_off = (_sbg.meta.dither.y_offset -
                    #               _slit.meta.dither.y_offset)
                    #
                    #     if np.abs(dy_off) < min_dyoffset:
                    #         continue
                    #
                    #     _dq = (_sbg.dq & bad_dq_bits) == 0
                    #     try:
                    #         _bkg += _sbg.data*_dq
                    #         _bkgn += _dq
                    #     except ValueError:
                    #         #print('background failed', j)
                    #         continue

                if (np.nanmax(_bkg) == 0) & (not get_slit_data):
                    axi = axes[ip]
                    axi.imshow(_sci*0., vmin=-0.05, vmax=0.3,
                                       origin='lower')

                    continue

                y0 = np.nanmean(ytr)
                if not np.isfinite(y0):
                    continue

                if np.isfinite(ws).sum() == 0:
                    continue

                _bkgn[_bkgn == 0] = 1
                _clean = _sci - _bkg/_bkgn
                                
                bad = ~np.isfinite(_clean)
                # bad |= _clean*np.sqrt(_ivar) < clip_sigma
                bad |= _bkg == 0
                bad |= _sci == 0

                _ivar[bad] = 0
                _clean[bad] = 0
                
                if _bkg_slit is not None:
                    _bkg_slit.data = _clean*1
                    _bkg_slit.dq = (bad*1).astype(_bkg_slit.dq.dtype)
                    _bkg_slit.has_background = True
                    
                prof = prf(yp.flatten()*0, (yp - ytr).flatten()).reshape(sh)
                        
                if 1:
                    _wht = (prof**2*_ivar).sum(axis=0)
                    y1 = (_clean*prof*_ivar).sum(axis=0) / _wht
                    _err = 1/np.sqrt(_wht)*np.sqrt(3./2)
                else:
                    _err = y1*0.1
                
                if get_slit_data:
                    chi = (_clean - prof*y1)*np.sqrt(_ivar)
                    return _slit, _clean, _ivar, prof, y1, _wcs, chi, bad
                
                axi = axes[ip]
                
                _running = eazy.utils.running_median((yp-ytr)[~bad],
                                                     (prof*y1)[~bad], 
                                                bins=np.arange(-5,5.01,pstep))
                pscl = np.nanmax(_running[1])
                if pscl < 0:
                    pscl = 0
                else:
                    pscl = 1/pscl
                    
                _run_flux = np.maximum(_running[1], 0)
                axi.step(x.max() + _run_flux*pscl*0.08*xtr.max(),
                                 _running[0]*2 + np.nanmedian(ytr),
                                 color='r', alpha=0.5, zorder=1000)

                _running = eazy.utils.running_median((yp-ytr)[~bad],
                                                     _clean[~bad],
                                                bins=np.arange(-5,5.01,pstep))
                _run_flux = np.maximum(_running[1], 0)
                axi.step(x.max() + _run_flux*pscl*0.08*xtr.max(),
                                 _running[0]*2 + np.nanmedian(ytr), color='k',
                                 alpha=0.5 ,zorder=1000)
            
                #aa.set_xlim(-5,5)

                if slit_key in ['flat','path']:
                    scl = 1e12
                else:
                    scl = 1.

                if slit_key == 'phot':
                    pixel_area = _slit.meta.photometry.pixelarea_steradians
                    _ukw = {}      
                    _ukw['equivalencies'] = spectral_density(ws*u.micron)

                    eflam = (_err*u.MJy*pixel_area).to(flux_unit, **_ukw)
                    flam = (y1*u.MJy*pixel_area).to(flux_unit, **_ukw)

                    y1 = flam.value
                    _err = eflam.value

                # print('yavg', yavj is None)

                if yavj is None:
                    yavj = y1*1
                    xy0 = ws[sh[1]//2]
                    shx = None
                    dx = 0.
                else:
                    dx = (ws[sh[1]//2] - xy0) / np.diff(ws)[sh[1]//2]
                    try:
                        shx = int(np.round(dx))
                        if verbose:
                            print('x shift: ', dx, shx)
                        yavj += np.roll(y1, shx)
                    except ValueError:
                        print('x shift err')
                        shx = None

                if shx is not None:
                    dxp = shx
                    if verbose:
                        print('Roll 2d', shx)

                    all_sci.append(np.roll(_clean, shx, axis=1))
                    all_ivar.append(np.roll(_ivar, shx, axis=1))
                    all_prof.append(np.roll(prof, shx, axis=1))

                else:
                    dxp = 0
                    all_sci.append(_clean)
                    all_ivar.append(_ivar)
                    all_prof.append(prof)

                a1.plot(xtr+dxp, y1*scl, color='k', alpha=0.2)
                a1.plot(xtr+dxp, _err*scl, color='pink', alpha=0.2)

                tab = utils.GTable()
                tab['wave'] = ws*1
                tab['wave'].unit = u.micron

                tab['flux'] = y1
                tab['err'] = _err
                tab['xtrace'] = xtr
                tab['ytrace'] = ytr

                if slit_key == 'phot':
                    tab['flux'].unit = flux_unit
                    tab['err'].unit = flux_unit

                tabs.append(tab)

                vmax = np.clip(1.5*np.nanpercentile(_clean[_ivar > 0], 95), 0.02, 0.5)
                if show_sn is None:
                    axi.imshow(_clean,
                                   vmin=-vmax/2/scl, vmax=vmax/scl,
                                   origin='lower')
                else:
                    if show_sn in [True, 1]:
                        vmax = np.clip(1.5*np.nanpercentile((_clean*np.sqrt(_ivar))[_ivar > 0], 95), 5, 30)
                        vmin = -1
                    else:
                        vmin, vmax = show_sn[:2]

                    axi.imshow(_clean*np.sqrt(_ivar),
                                   vmin=vmin, vmax=vmax,
                                   origin='lower')

                axi.plot(xtr, ytr+2, color='w', alpha=0.5)
                axi.plot(xtr, ytr-2, color='w', alpha=0.5)

                if center2d:
                    axi.set_ylim(y0-8, y0+8)

        if yavj is None:
            plt.close(fig)
            return None, None, None, None
        
        yavj /= self.N

        # Combined optimal extraction
        sall_prof = np.vstack(all_prof)
        sall_sci = np.vstack(all_sci)
        sall_ivar = np.vstack(all_ivar)
        sall_ivar[sall_sci*np.sqrt(sall_ivar) < clip_sigma] = 0
        
        all_num = (sall_prof*sall_sci*sall_ivar)
        all_den = (sall_prof**2*sall_ivar).sum(axis=0)
        all_flux = all_num.sum(axis=0) / all_den
        all_err = 1/np.sqrt(all_den)

        _wave = tabs[0]['wave']

        if slit_key == 'phot':
            pixel_area = _slit.meta.photometry.pixelarea_steradians
            _ukw = {}      
            _ukw['equivalencies'] = spectral_density(_wave.value*u.micron)

            all_err = (all_err*u.MJy*pixel_area).to(flux_unit, **_ukw)
            all_flux = (all_flux*u.MJy*pixel_area).to(flux_unit, **_ukw)
        
        if ntrim > 0:
            _ok = np.isfinite(all_err+all_flux) & (all_err > 0)
            if _ok.sum() > ntrim*2:
                _oki = np.where(_ok)[0]
                all_flux[_oki[-ntrim:]] = np.nan
                all_err[_oki[-ntrim:]] = np.nan
                all_flux[_oki[:ntrim]] = np.nan
                all_err[_oki[:ntrim]] = np.nan
            
        full_tab = utils.GTable()
        full_tab['wave'] = _wave
        full_tab['flux'] = all_flux
        full_tab['err'] = all_err

        if slit_key == 'phot':
            full_tab['flux'].unit = flux_unit
            full_tab['err'].unit = flux_unit

        for t in tabs:
            t.meta['prof_sigma'] = prof_sigma
            t.meta['yoffset'] = yoffset
            t.meta['bad_dq_bits'] = bad_dq_bits
            t.meta['clip_sigma'] = clip_sigma

            for k in slitlet:
                if k.startswith('source'):
                    t.meta[k] = slitlet[k]

            for _m in [_slit.meta.instrument, _slit.meta.exposure]:
                _mi =  _m.instance
                for k in _mi:
                    t.meta[k] = _mi[k]

        for k in tabs[0].meta:
            full_tab.meta[k] = tabs[0].meta[k]

        full_tab.meta['exptime'] = 0.
        full_tab.meta['ncombined'] = 0
        for t in tabs:
            full_tab.meta['exptime'] += t.meta['effective_exposure_time']
            full_tab.meta['ncombined'] += 1

        wx = np.append(np.array([0.7, 0.8]), np.arange(1, 5.6, 0.5))
    
        wx = wx[(wx > np.nanmin(_wave)) & (wx < np.nanmax(_wave))]
        if len(wx) < 2:
            wx = np.arange(0.6, 5.6, 0.1)    
            wx = wx[(wx > np.nanmin(_wave)) & (wx < np.nanmax(_wave))]

        xx = np.interp(wx, _wave, tabs[0]['xtrace'])
        for i, ax in enumerate(axes[:-1]):
            ax.set_aspect('auto')
            ax.set_yticklabels([])

        for i, ax in enumerate(axes):

            ax.set_xticks(xx)
            #ax.set_xlim(0, _bkg.shape[1]-1)
            ax.set_xlim(0, _bkg.shape[1]*1.08)
            if i == self.N:
                ax.set_xticklabels([f'{wi:.1f}' for wi in wx])
            else:
                ax.set_xticklabels([])

        #axes[1].set_ylabel(mode + key)
        a1.text(0.02, 0.95, self.mode, va='top', ha='left',
                transform=a1.transAxes, fontsize=8,
                bbox={'fc':'w', 'ec':'None', 'alpha':0.7})

        a1.text(0.98, 0.95, key, va='top', ha='right',
                transform=a1.transAxes, fontsize=8,
                bbox={'fc':'w', 'ec':'None', 'alpha':0.7})

        ymax = 2.2*np.nanpercentile(full_tab['flux'], 98)
        a1.plot(tabs[0]['xtrace'], full_tab['flux']*scl, color='k', alpha=0.8)
        a1.grid()

        a1.set_ylim(-0.15*ymax, ymax)
        a1.set_xlabel(r'$\lambda$, observed $\mu\mathrm{m}$')

        if slit_key == 'phot':
            a1.set_ylabel(full_tab['flux'].unit.to_string(format="latex"))

        # timestamp
        fig.text(0.015*12./12, 0.02, f'dy={yoffset:.2f} sig={prof_sigma:.2f}',
                 ha='left', va='bottom',
                 transform=fig.transFigure, fontsize=8)
        
        fig.text(1-0.015*12./12, 0.02, time.ctime(),
                 ha='right', va='bottom',
                 transform=fig.transFigure, fontsize=6)

        fig.tight_layout(pad=0.5)

        return slitlet, tabs, full_tab, fig


    def extract_all_slits(self, keys=None, verbose=True, close=True, **kwargs):
        """
        Extract all spectra and make diagnostic figures
        """
        slitlets = self.slitlets
    
        if keys is None:
            keys = list(slitlets.keys())
        
        for i, key in enumerate(keys):
            out = f'{self.mode}-{key}.spec'

            try:
                _ext = self.extract_spectrum(key, **kwargs)

                slitlet, tabs, full_tab, fig = _ext

                fig.savefig(out+'.png')
                full_tab.write(out+'.fits', overwrite=True)
                msg = f'msaexp.extract_spectrum: {key}'
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose, 
                                  show_date=False)
                                  
                if close:
                    plt.close('all')

            except:
                print(f'{self.mode} {key} Failed')
                utils.log_exception(out+'.failed', traceback, verbose=True)
                if close:
                    plt.close('all')

    def get_slit_polygons(self, include_yoffset=False):
        """
        Get slit polygon regions using slit wcs
        """
        from tqdm import tqdm
        slit_key = self.last_step
        pipe = self.pipe[slit_key]
        
        regs = []
        for j in range(self.N):
            regs.append([])
        
        for key in tqdm(self.slitlets):
            slitlet = self.slitlets[key]
            #slitlet['bkg_index'], slitlet['src_index'], slitlet['slit_index']

            i = self.slit_index(key) # slitlet['slit_index']

            yoffset = slitlet['yoffset']
            
            for j in range(self.N):
                _slit = pipe[j].slits[i]
                _wcs = _slit.meta.wcs
                sh = _slit.data.shape
                
                #_dq = (pipe[j].slits[i].dq & bad_dq_bits) == 0
                #_sci = pipe[j].slits[i].data
                
                yp, xp = np.indices(sh)
                x = np.arange(sh[1])
                x0 = np.ones(sh[0])*sh[1]/2.
                y0 = np.arange(sh[0])
                r0, d0, w0 = _wcs.forward_transform(x0,  y0)
                
                # 
                tr = _wcs.get_transform(_wcs.slit_frame, _wcs.world)
                ysl = np.linspace(slitlet['slit_ymin'],
                                  slitlet['slit_ymax'], 8)
                
                rl, dl, _ = tr(-0.5, 0, 1)
                rr, dr, _ = tr(0.5, 0, 1)
                
                #ro, do, _ = tr(-0.5, 0.1/0.46*yoffset + slitlet['source_ypos'], 1)
                # yoffset along slit, as 0.1" pixels along to 0.46" slits
                if include_yoffset:
                    ro, do, _ = tr(-0.5, 0.1/0.46*yoffset, 1)
                    rs = ro-rl
                    ds = do-dl
                else:
                    rs = ds = 0.
                    
                rw = rr-rl
                dw = dr-dl
                
                ok = np.isfinite(d0)
                
                xy = np.array([np.append(r0[ok]+rs, r0[ok][::-1]+rs+rw),
                               np.append(d0[ok]+ds, d0[ok][::-1]+ds+dw)])
                               
                sr = utils.SRegion(xy)
                _name = slitlet['source_name']
                
                if '_-' in _name:
                    sr.ds9_properties = 'color=yellow'
                elif 'background' in _name:
                    sr.ds9_properties = 'color=white'
                else:
                    sr.ds9_properties = 'color=green'

                if j == 0:
                    sr.label = _name
                    sr.ds9_properties += ' width=2'
                
                regs[j].append(sr)
        
        _slitreg = f'{self.mode}.slits.reg'
        print(_slitreg)
        with open(_slitreg, 'w') as fp:
            for j in range(self.N):
                    fp.write('icrs\n')
                    for sr in regs[j]:
                        fp.write(sr.region[0]+'\n')


    def load_slit_data(self, step='phot', verbose=True, indices=None, targets=None):
        """
        Load slit data from files
        """
        slit_lists = [SlitData(file, step=step, read=False,
                               indices=indices, targets=targets)
                      for file in self.files]

        counts = [sl.N for sl in slit_lists]
        if (counts[0] > 0) & (np.allclose(counts, np.min(counts))):
            for sl in slit_lists:
                sl.read_data(verbose=verbose)

            return slit_lists

        else:
            return None


    def parse_slit_info(self, write=True):
        """
        """
        import yaml

        keys = ['source_name','source_ra','source_dec','skip',
                'yoffset','prof_sigma','redshift','is_background',
                'slit_index', 'src_index', 'bkg_index',
                'slit_ra', 'slit_dec']
                
        yaml_file = f'{self.mode}.slits.yaml'
        if os.path.exists(yaml_file):
            with open(yaml_file) as fp:
                info = yaml.load(fp, Loader=yaml.Loader)
        else:
            info = {}
    
        for _src in self.slitlets:
            s = self.slitlets[_src]
            info[_src] = {}
            for k in keys:
                if k in s:
                    info[_src][k] = s[k]
        
            if len(info[_src]['skip']) == 0:
                info[_src]['skip'] = []
    
        if write:
            with open(yaml_file,'w') as fp:
                yaml.dump(info, stream=fp)
        
            print(yaml_file)
        
        return info


    def full_pipeline(self, load_saved='phot', run_extractions=True, indices=None, targets=None, initialize_bkg=True, make_regions=True, **kwargs):
        """
        Run all steps through extractions
        """
                
        if load_saved is not None:
            if load_saved in self.pipe:
                status = self.pipe[load_saved]
            else:
                status = self.load_slit_data(step=load_saved, indices=indices,
                                             targets=targets)
        else:
            status = None
        
        if status is not None:
            # Have loaded saved data
            make_regions = False
            self.pipe[load_saved] = status
            self.last_step = load_saved
        elif targets is not None:
            print(f'Targets {targets} not found')
            return True
            
        else:
            self.preprocess()
            self.run_jwst_pipeline()
        
        self.slitlets = self.initialize_slit_metadata()
        
        self.get_slit_traces()
        
        if run_extractions:
            self.extract_all_slits(**kwargs)
        
        if make_regions:
            self.slit_source_regions()
        
        self.parse_slit_info(write=True)
        
        if status is None:
            self.save_slit_data()
        
        if initialize_bkg:
            print('Set background slits')
            self.set_background_slits()


def make_summary_tables(root='msaexp', zout=None):
    """
    """
    import yaml
    import astropy.table
    
    groups = exposure_groups()

    tabs = []
    for mode in groups:
        # mode = 'jw02767005001-02-clear-prism-nrs1'

        yaml_file = f'{mode}.slits.yaml'
        if not os.path.exists(yaml_file):
            print(f'Skip {yaml_file}')
            continue

        with open(yaml_file) as fp:
            yaml_data = yaml.load(fp, Loader=yaml.Loader)

        cols = []
        rows = []
        for k in yaml_data:
            row = []
            for c in ['source_name','source_ra','source_dec','yoffset',
                      'prof_sigma','redshift','is_background']:
                if c in ['skip']:
                    continue
                if c not in cols:
                    cols.append(c)

                row.append(yaml_data[k][c])

            rows.append(row)

        tab = utils.GTable(names=cols, rows=rows)
        tab.rename_column('source_ra','ra')
        tab.rename_column('source_dec','dec')
        bad = np.in1d(tab['redshift'], [None])
        tab['z'] = -1.
        tab['z'][~bad] = tab['redshift'][~bad]
        tab['mode'] = ' '.join(mode.split('-')[-3:-1])
        tab['detector'] = mode.split('-')[-1]

        tab['group'] = mode

        tab.remove_column('redshift')
        tab.write(f'{mode}.info.csv', overwrite=True)
        
        tab['wmin'] = 0.
        tab['wmax'] = 0.
        
        tab['oiii_sn'] = -100.
        tab['ha_sn'] = -100.
        tab['max_cont'] = -100.
        tab['dof'] = 0
        tab['dchi2'] = -100.
        tab['bic_diff'] = -100.
        
        # Redshift output
        for i, s in tqdm(enumerate(tab['source_name'])):
            yy = f'{mode}-{s}.spec.yaml'
            if os.path.exists(yy):
                with open(yy) as fp:
                    zfit = yaml.load(fp, Loader=yaml.Loader)
                
                for k in ['z','dof','wmin','wmax','dchi2']:
                    if k in zfit:
                        tab[k][i] = zfit[k]
                    
                oiii_key = None
                if 'spl_coeffs' in zfit:
                    
                    max_spl = -100
                    nline = 0
                    ncont = 0
                    
                    for k in zfit['spl_coeffs']:
                        if k.startswith('bspl') & (zfit['spl_coeffs'][k][1] > 0):
                            _coeff = zfit['spl_coeffs'][k]
                            max_spl = np.maximum(max_spl, _coeff[0]/_coeff[1])
                            ncont += 1
                            
                        elif k.startswith('line'):
                            nline += 1
                            
                    tab['max_cont'][i] = max_spl
                    
                    bic_cont = np.log(zfit['dof'])*ncont + zfit['spl_cont_chi2']
                    bic_line = np.log(zfit['dof'])*(ncont+nline) + zfit['spl_full_chi2']
                    
                    tab['bic_diff'][i] = bic_cont - bic_line

                    if 'line Ha' in zfit['spl_coeffs']:
                        _coeff = zfit['spl_coeffs']['line Ha']
                        if _coeff[1] > 0:
                            tab['ha_sn'][i] = _coeff[0]/_coeff[1]
                        
                    for k in ['line OIII-5007', 'line OIII']:
                        if k in zfit['spl_coeffs']:
                            oiii_key = k
                
                    if oiii_key is not None:
                        _coeff = zfit['spl_coeffs'][oiii_key]
                        if _coeff[1] > 0:
                            tab['oiii_sn'][i] = _coeff[0]/_coeff[1]
                    

        tabs.append(tab)

    full = utils.GTable(astropy.table.vstack(tabs))
    ok = np.isfinite(full['ra'] + full['dec'])
    full = full[ok]
    
    full['ra'].format = '.7f'
    full['dec'].format = '.7f'
    full['yoffset'].format = '.2f'
    full['prof_sigma'].format = '.2f'
    full['z'].format = '.4f'
    full['oiii_sn'].format = '.1f'
    full['ha_sn'].format = '.1f'
    full['max_cont'].format = '.1f'
    full['dchi2'].format = '.1f'
    full['dof'].format = '.0f'
    full['bic_diff'].format = '.1f'
    full['wmin'].format = '.1f'
    full['wmax'].format = '.1f'
    
    if zout is not None:
        idx, dr = zout.match_to_catalog_sky(full)
        hasm = dr.value < 0.3
        if root == 'uds':
            hasm = dr.value < 0.4

        full['z_phot'] = -1.0
        full['z_phot'][hasm] = zout['z_phot'][idx][hasm]

        full['z_spec'] = -1.0
        full['z_spec'][hasm] = zout['z_spec'][idx][hasm]

        full['z_phot'].format = '.2f'
        full['z_spec'].format = '.3f'

        full['phot_id'] = -1
        full['phot_id'][hasm] = zout['id'][idx][hasm]
    
    url = '<a href="{m}-{name}.spec.fits">'
    url += '<img src="{m}-{name}.spec.png" height=200px>'
    url += '</a>'

    churl = '<a href="{m}-{name}.spec.fits">'
    churl += '<img src="{m}-{name}.spec.chi2.png" height=200px>'
    churl += '</a>'

    furl = '<a href="{m}-{name}.spec.fits">'
    furl += '<img src="{m}-{name}.spec.zfit.png" height=200px>'
    furl += '</a>'
    
    full['spec'] = [url.format(m=m, name=name)
                    for m, name in zip(full['group'], full['source_name'])]

    full['chi2'] = [churl.format(m=m, name=name)
                    for m, name in zip(full['group'], full['source_name'])]
    
    full['zfit'] = [furl.format(m=m, name=name)
                    for m, name in zip(full['group'], full['source_name'])]
    
    full.write(f'{root}_nirspec.csv', overwrite=True)
    full.write_sortable_html(f'{root}_nirspec.html',
                             max_lines=10000, 
                             filter_columns=['ra','dec','z_phot', 
                                             'wmin', 'wmax',
                                             'z', 'dof', 'bic_diff', 'dchi2',
                                             'oiii_sn', 'ha_sn', 'max_cont',
                                             'z_spec','yoffset','prof_sigma'],
                             localhost=False)

    print(f'Created {root}_nirspec.html {root}_nirspec.csv')
    return tabs, full
        
