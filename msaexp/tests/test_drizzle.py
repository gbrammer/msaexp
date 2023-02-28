import os
import copy
import numpy as np

from .. import utils

def data_path():
    return os.path.join(os.path.dirname(__file__), 'data')

def test_combine():
    """
    Test drizzle combination from slitlet
    """
    import matplotlib.pyplot as plt
    
    from .. import drizzle as msadrizzle
    
    os.chdir(data_path())
    
    drizzle_kws = dict(center_on_source=False,  # Center output source as defined in MSA file.
                       fix_slope=None,          # Cross-dispersion scale dslit/dpix
                       force_nypix=31,         # Y size of output array
                       bkg_offset=6,           # Number of pixels to roll for background sub
                       bkg_parity=[-1,1],      # Roll directions for background, i.e., -1 rolls 
                       log_step=False,         # Log wavelength steps
                       outlier_threshold=10,   # Outlier rejection threshold
                       err_threshold=1000,     # Reject pixels in a slit where err > err_threshold*median(err)
                       show_drizzled=True,     # Figures
                       show_slits=True,
                       imshow_kws=dict(vmin=-0.05, vmax=0.4, aspect='auto', cmap='cubehelix_r'),
                       sn_threshold=-np.inf,   # Reject pixels where S/N < sn_threshold.  
                       bar_threshold=0.0,      # Mask pixels where barshadow array less than this value
                      )
    
    DRIZZLE_PARAMS = copy.deepcopy(msadrizzle.DRIZZLE_PARAMS)
    DRIZZLE_PARAMS['kernel'] = 'square'
    DRIZZLE_PARAMS['pixfrac'] = 1.0
    
    outroot = 'test-driz'
    target = '933'
    
    _ = msadrizzle.drizzle_slitlets(target,
                                    output=outroot,
                                    # files=slit_files[:],
                                    **drizzle_kws,
                                    )
    
    figs, hdu_data, wavedata, all_slits, drz_data = _
    
    # Fixed "slope"
    drizzle_kws['fix_slope'] = 0.2
    outroot = 'test-driz-slope'
    _ = msadrizzle.drizzle_slitlets(target,
                                    output=outroot,
                                    # files=slit_files[:],
                                    **drizzle_kws,
                                    )
    
    figs, hdu_data, wavedata, all_slits, drz_data = _

    # Center on source
    drizzle_kws['center_on_source'] = True
    outroot = 'test-driz-center'
    _ = msadrizzle.drizzle_slitlets(target,
                                    output=outroot,
                                    # files=slit_files[:],
                                    **drizzle_kws,
                                    )
    
    figs, hdu_data, wavedata, all_slits, drz_data = _
    
    # 1D extraction
    extract_kws = dict(prf_sigma=1.2, fix_sigma=False,
                       prf_center=None, fix_center=False,
                       center_limit=3,
                       verbose=True,
                       profile_slice=None
                      )
    
    grating = 'prism-clear'
    hdul = hdu_data[grating]

    outhdu = msadrizzle.extract_from_hdul(hdul, **extract_kws)
    
    file = f'{outroot}_{target}.spec.fits'

    outhdu.writeto(file, overwrite=True)

    # Make figures
    fig = utils.drizzled_hdu_figure(outhdu, unit='fnu')
    fig.savefig(f'{outroot}-{grating}_{target}.fnu.png')

    fig = utils.drizzled_hdu_figure(outhdu, unit='flam')
    fig.savefig(f'{outroot}-{grating}_{target}.flam.png')
    
    plt.close('all')
    