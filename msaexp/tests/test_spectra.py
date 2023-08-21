"""
Test spectrum extractions and fits
"""
import os

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

import astropy.io.fits as pyfits

from .. import utils, spectrum

eazy_templates = None

# SPECTRUM_FILE = f'ceers-prism.1345_933.v0.spec.fits'
SPECTRUM_FILE = 'test-driz-center-bkg_933.spec.fits'
# SPECTRUM_FILE = 'test-driz-center_933.spec.fits'

def data_path():
    return os.path.join(os.path.dirname(__file__), 'data')

def test_load_templates():
    
    import eazy
    global eazy_templates
    
    os.chdir(data_path())
    
    current_path = os.getcwd()
    
    path = os.path.join(os.path.dirname(eazy.__file__), 'data/')
    if not os.path.exists(os.path.join(path, 'templates')):
        eazy.fetch_eazy_photoz()
    
    os.chdir(current_path)
    os.chdir(data_path())
    
    if not os.path.exists('templates'):
        eazy.symlink_eazy_inputs()

    _param = 'templates/sfhz/carnall_sfhz_13.param'
    eazy_templates = eazy.templates.read_templates_file(_param)


def test_fit_redshift():
    """
    Redshift fit with spline + line templates
    """
    global eazy_templates
    
    os.chdir(data_path())
    
    spectrum.FFTSMOOTH = True
        
    kws = dict(eazy_templates=None,
               scale_disp=1.0,
               nspline=33, 
               Rline=2000, 
               use_full_dispersion=False,
               vel_width=100,
               )
    
    z=4.2341
    z0 = [4.1, 4.4]
    
    fig, spec, zfit = spectrum.plot_spectrum(SPECTRUM_FILE,
                                             z=z,
                                             **kws)
    
    fig.savefig(SPECTRUM_FILE.replace('spec.fits', 'spec.spl.png'))
    
    assert('z' in zfit)
    
    assert(np.allclose(zfit['z'], z, rtol=0.01))
    
    assert('coeffs' in zfit)
    if 'line OIII' in zfit['coeffs']:
        assert(np.allclose(zfit['coeffs']['line OIII'],
              [2386.17, 35.93], rtol=0.5))
    elif 'line OIII-5007' in zfit['coeffs']:
        assert(np.allclose(zfit['coeffs']['line OIII-5007'],
              [1753.03, 35.952727162], rtol=0.5))
        
    if eazy_templates is not None:
        kws['eazy_templates'] = eazy_templates
        kws['use_full_dispersion'] = False
        fig, spec, zfit = spectrum.fit_redshift(SPECTRUM_FILE,
                              z0=z0,
                              is_prism=True,
                              **kws)
        
        plt.close('all')
        assert('z' in zfit)
    
        assert(np.allclose(zfit['z'], z, rtol=0.01))
    
        assert('coeffs' in zfit)
        assert(np.allclose(zfit['coeffs']['4590.fits'],
                           #[127.2, 3.418],
                           [120.2, 2.5], # With SpectrumSampler fits
                           rtol=0.5))
        
        #### use_full_dispersion is deprecated now using SpectrumSampler
        
        # # With dispersion
        # kws['use_full_dispersion'] = True
        # fig, spec, zfit = spectrum.fit_redshift(f'ceers-prism.1345_933.v0.spec.fits',
        #                       z0=z0,
        #                       is_prism=True,
        #                       **kws)
        #
        # plt.close('all')
        # assert('z' in zfit)
        #
        # assert(np.allclose(zfit['z'], z, rtol=0.01))
        #
        # assert('coeffs' in zfit)
        # assert(np.allclose(zfit['coeffs']['4590.fits'],
        #                    [75.95547, 3.7042],
        #                    rtol=0.5))


def test_sampler_object():
    """
    Test the spectrum.SpectrumSampler methods
    """
    
    os.chdir(data_path())
    
    spec = spectrum.SpectrumSampler(SPECTRUM_FILE)
    sampler_checks(spec)
    
    new = spec.redo_1d_extraction()
    sampler_checks(new)
    
    # Initialized from HDUList
    with pyfits.open(SPECTRUM_FILE) as hdul:
        spec = spectrum.SpectrumSampler(hdul)
        sampler_checks(spec)


def sampler_checks(spec):
    
    assert(np.allclose(spec.valid.sum(), 327, atol=5))
    
    # emission line
    z = 4.2341
    line_um = 3727.*(1+z)/1.e4
    
    for s in [1, 1.3, 1.8, 2.]:
        for v in [50, 100, 300, 500, 1000]:
            kws = dict(scale_disp=s, velocity_sigma=v)

            gau = spec.emission_line(line_um, line_flux=1, **kws)
            assert(np.allclose(np.trapz(gau, spec.spec_wobs), 1., rtol=5.e-2))

            gau2 = spec.fast_emission_line(line_um, line_flux=1, **kws)
            assert(np.allclose(np.trapz(gau2, spec.spec_wobs), 1., rtol=1.e-3))

