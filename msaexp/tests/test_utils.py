import os
import numpy as np
from scipy.stats import uniform

from .. import utils


def test_import():
    """ """
    import msaexp.pipeline
    import msaexp.spectrum
    import msaexp.utils
    import msaexp.slit_combine
    import msaexp.drizzle


def test_wavelength_grids():

    import grizli.utils

    grizli.utils.set_warnings()

    for gr in utils.GRATING_LIMITS:
        grid = utils.get_standard_wavelength_grid(gr)
        grid = utils.get_standard_wavelength_grid(gr, log_step=True)

    grid = utils.get_standard_wavelength_grid("prism", free_prism=False)


def test_fwhm():

    import numpy as np

    wave = np.linspace(0.8, 5.0, 128)
    psf_fwhm = utils.get_nirspec_psf_fwhm(wave)

    assert psf_fwhm.min() > 0.2
    assert psf_fwhm.max() < 7


def test_bar_correction():

    scaled_yshutter = np.array([0.0, 0.5, 2.0])

    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert not wrapped
    assert bar[0] > 0.95
    assert bar[1] < 0.5
    assert bar[2] < 0.1

    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=True,
        bar_data=None,
    )

    assert not wrapped
    assert bar[0] > 0.95
    assert bar[1] < 0.5
    assert np.isnan(bar[2])

    # num_shutters
    for s in [1, 2, 3]:
        bar, wrapped = utils.get_prism_bar_correction(
            scaled_yshutter,
            num_shutters=s,
            wrap="auto",
            wrap_pad=0.2,
            mask=True,
            bar_data=None,
        )
        assert bar[0] > 0.95
        assert bar[1] < 0.5
        assert np.isnan(bar[2])

    scaled_yshutter = np.array([-3.0, 0.0, 0.5, 2.0, 3.0])
    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert wrapped
    assert bar[0] == bar[1] == bar[3]

    scaled_yshutter = np.array([-3.0, 0.0, 0.5, 2.0, 3.0])
    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap=False,
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert not wrapped
    assert bar[0] == 0

    scaled_yshutter = np.array([0.0, 0.5, 2.0, 3.0])
    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap=True,
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert wrapped
    assert bar[0] == bar[2]

    # Shapes
    scaled_yshutter = np.array([-2.0, 0.0, 0.5, 2.0]).reshape((2, 2))

    bar, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter,
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert not wrapped
    assert bar.shape == (2, 2)

    bar_flat, wrapped = utils.get_prism_bar_correction(
        scaled_yshutter.flatten(),
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=False,
        bar_data=None,
    )

    assert np.allclose(bar.flatten(), bar_flat)

    assert np.allclose(
        bar,
        bar_flat.reshape((2, 2)),
    )

    # Wavelength
    scaled_yshutter = np.array([0.0, 0.5, 2.0])
    wavelengths = np.array([1.0, 2.0, 3.0])

    bar, wrapped = utils.get_prism_wave_bar_correction(
        scaled_yshutter,
        wavelengths,
        num_shutters=3,
        wrap="auto",
        wrap_pad=0.2,
        mask=True,
        bar_data=None,
    )

    assert bar[0] > 0.95
    assert bar[1] < 0.5
    assert np.isnan(bar[2])

    # Bigger 2D array
    sh = (21, 435)

    np.random.seed(1)

    scaled_yshutter = uniform.rvs(loc=-1.5, scale=3, size=sh)
    wavelengths = uniform.rvs(loc=0.8, scale=5.3 - 0.8, size=sh)

    for s in [1, 2, 3]:
        bar, wrapped = utils.get_prism_bar_correction(
            scaled_yshutter,
            num_shutters=3,
            wrap="auto",
            wrap_pad=0.2,
            mask=False,
            bar_data=None,
        )

    assert bar.shape == sh
    assert bar.max() < 1.05
    assert bar.min() < 0.4

    for s in [1, 2, 3]:
        bar, wrapped = utils.get_prism_wave_bar_correction(
            scaled_yshutter,
            wavelengths,
            num_shutters=3,
            wrap="auto",
            wrap_pad=0.2,
            mask=False,
            bar_data=None,
        )

    assert bar.shape == sh
    assert bar.max() < 1.05
    assert bar.min() < 0.4


def test_normalization():

    sh = (21, 435)

    np.random.seed(1)

    wavelengths = uniform.rvs(loc=0.8, scale=5.3 - 0.8, size=sh)

    for q in [1, 2, 3, 4]:
        corr = utils.get_normalization_correction(
            wavelengths,
            q,
            180,
            65,
            grating="PRISM",
            verbose=True,
        )

        assert corr.shape == sh
        assert corr.min() > 0.9
        assert corr.max() < 1.15

    for grating in ["G140M", "G235M", "G395M"]:
        corr = utils.get_normalization_correction(
            wavelengths,
            q,
            180,
            65,
            grating="G235M",
            verbose=True,
        )

        assert np.allclose(corr, 1.0)


def test_badpix():

    utils.cache_badpix_arrays()

    for detector in ["NRS1", "NRS2"]:
        assert detector in utils.MSAEXP_BADPIX

        assert len(utils.MSAEXP_BADPIX[detector]) == 3
        assert utils.MSAEXP_BADPIX[detector][0].shape == (2048, 2048)
        assert "yaml" in utils.MSAEXP_BADPIX[detector][2]


def test_array_bins():

    step = 2
    array = np.arange(0, 11, step)
    bins = utils.array_to_bin_edges(array)

    assert len(bins) == len(array) + 1
    assert np.allclose(bins[:-1], array - step / 2.0)
    assert np.allclose(bins[1:], array + step / 2.0)
    assert np.allclose(np.diff(bins), step)


def test_pixfrac():

    oversample = 8
    pixfrac = 0.5

    steps = utils.pixfrac_steps(oversample, pixfrac)

    assert len(steps) == oversample
    assert steps[0] == -pixfrac + 1.0 / oversample * pixfrac
    assert steps[-1] == pixfrac - 1.0 / oversample * pixfrac

    for oversample in [4, 8, 9, 16]:
        for pixfrac in [0.1, 0.5, 1.0]:
            steps = utils.pixfrac_steps(oversample, pixfrac)

            assert len(steps) == oversample
            assert steps[0] == -pixfrac + 1.0 / oversample * pixfrac
            assert steps[-1] == pixfrac - 1.0 / oversample * pixfrac


def data_path():
    return os.path.join(os.path.dirname(__file__), "data")


def test_update_slit_metadata():
    
    import jwst.datamodels
    
    os.chdir(data_path())
    
    file = 'jw01345062001_03101_00001_nrs2_phot.138.1345_933.fits'

    def _assert_clean_data(slit):

        """
        Helper function to assert that the slit data is clean i.e.
        does not hit any of the 'if' cases in the method, before we 
        tamper with it.
        """
        
        meta = slit.meta.instance
        is_fixed = meta["instrument"]["lamp_mode"] == "FIXEDSLIT"

        # needed for the current file used,
        # in order to make 'source_type' not None. 
        # allowed options are: "POINT", "EXTENDED" and "UNKNOWN"
        slit.source_type = "UNKNOWN"

        assert not (is_fixed & (slit.source_name is None))
        assert hasattr(slit, "source_type")
        assert not (slit.source_type is None)
        assert slit.slitlet_id != 0



    # there are 4 'if' cases in the method. First assert that the data
    # does not hit any, and then we tamper with it to hit every 'if' case

    #case 1: FIXEDSLIT & source__name is None:

    with jwst.datamodels.open(file) as slit_1:
        
        _assert_clean_data(slit_1)
        slit_1.meta.instance["instrument"]["lamp_mode"] = "FIXEDSLIT"
        slit_1.source_name = None
        utils.update_slit_metadata(slit_1)

        targ = slit_1.meta.instance["target"]
        
        assert slit_1.source_name == \
            f"{targ['proposer_name'].lower()}_{slit_1.name}".lower()
        
        assert slit_1.source_ra == targ["ra"]
        assert slit_1.source_dec == targ["dec"]



    #case 2: source type is missing:
    with jwst.datamodels.open(file) as slit_2:

        _assert_clean_data(slit_2)
        del slit_2.source_type

        utils.update_slit_metadata(slit_2)

        assert slit_2.source_type == "EXTENDED"


    #case 3: source type is None:
    with jwst.datamodels.open(file) as slit_3:
        
        _assert_clean_data(slit_3)
        slit_3.source_type = None
        assert(slit_3.source_type == None)

        utils.update_slit_metadata(slit_3)

        assert slit_3.source_type == "EXTENDED"

    #case: slitled_id is missing:

    with jwst.datamodels.open(file) as slit_4:

        _assert_clean_data(slit_4)
        del slit_4.slitlet_id

        utils.update_slit_metadata(slit_4)

        assert slit_4.slitlet_id == 9999


def test_slit_things():

    import jwst.datamodels
    
    os.chdir(data_path())
    
    file = 'jw01345062001_03101_00001_nrs2_phot.138.1345_933.fits'
    
    with jwst.datamodels.open(file) as slit:
        corr = utils.slit_normalization_correction(slit, verbose=True)
        
        sign = utils.get_slit_sign(slit)
        assert sign == -1
        
        
    
    
