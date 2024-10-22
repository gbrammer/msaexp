import os
import copy
import glob

import numpy as np
import matplotlib.pyplot as plt
import jwst

from .. import utils
from .. import slit_combine
from .. import spectrum

if jwst.__version__ < "100.16":
    TARGETS = ["1345_933"]
else:
    TARGETS = ["4233_19489"]


def data_path():
    return os.path.join(os.path.dirname(__file__), "data")


def test_globals():

    assert utils.BAD_PIXEL_FLAG & 1025 == 1025


def test_split():

    os.chdir(data_path())

    files = glob.glob(f"*phot.*{TARGETS[0]}.fits")
    files.sort()

    groups = slit_combine.split_visit_groups(
        files,
        join=[0, 3],
        gratings=["PRISM"],
        split_uncover=True,
    )

    assert len(groups) == 1
    for g in groups:
        assert len(groups[g]) == 3

    groups = slit_combine.split_visit_groups(
        files,
        join=[0, 2, 3],
        gratings=["PRISM"],
        split_uncover=True,
    )

    assert len(groups) == 3


def test_prf():

    waves = np.logspace(np.log10(0.7), np.log10(5.3), 256)

    path = slit_combine.slit_prf_fraction(
        waves,
        sigma=0.0,
        x_pos=0.0,
        slit_width=0.2,
        pixel_scale=0.10544,
        verbose=True,
    )

    assert path.shape == waves.shape
    assert np.all(path < 1)
    assert np.all(path > 0)

    path_broad = slit_combine.slit_prf_fraction(
        waves,
        sigma=0.2,
        x_pos=0.0,
        slit_width=0.2,
        pixel_scale=0.10544,
        verbose=True,
    )

    path_off = slit_combine.slit_prf_fraction(
        waves,
        sigma=0.2,
        x_pos=0.2,
        slit_width=0.2,
        pixel_scale=0.10544,
        verbose=True,
    )

    assert path_broad.shape == waves.shape
    assert path_off.shape == waves.shape

    assert np.all(path_broad < path)
    assert np.all(path_off < path_broad)


def test_slit_group():
    """
    Test drizzle combination from slitlet
    """
    os.chdir(data_path())

    files = glob.glob(f"*phot.*{TARGETS[0]}.fits")
    files.sort()

    default_kwargs = dict(
        position_key="y_index",
        diffs=True,
        grating_diffs=True,
        stuck_threshold=0.5,
        hot_cold_kwargs=None,
        bad_shutter_names=None,
        dilate_failed_open=True,
        undo_barshadow=2,
        min_bar=0.4,
        bar_corr_mode="wave",
        fix_prism_norm=True,
        sky_arrays=None,
        sky_file=None,
        estimate_sky_kwargs=None,
        flag_profile_kwargs={},
        flag_percentile_kwargs={},
        undo_pathloss=True,
        trace_with_xpos=False,
        trace_with_ypos=False,
        trace_from_yoffset=True,
        fit_shutter_offset_kwargs=None,
        shutter_offset=0.0,
        nod_offset=None,
        pad_border=2,
        weight_type="ivm",
        reference_exposure="auto",
    )

    obj = slit_combine.SlitGroup(files, "stest", **default_kwargs)

    for attr in [
        "files",
        "fit",
        "N",
        "mask",
        "sci",
        "dq",
        "slits",
        "var_rnoise",
        "var_poisson",
        "var_total",
        "bar",
        "wave",
        "dwave_dx",
        "meta",
        "meta_comment",
        "name",
        "sh",
        "slits",
        "wtr",
        "xslit",
        "xtr",
        "ypix",
        "yslit",
        "ytr",
    ]:
        assert hasattr(obj, attr)

    assert obj.files == files
    assert obj.name == "stest"
    assert obj.fit is None
    assert obj.N == len(files)
    assert len(obj.slits) == obj.N

    assert obj.sh == (25, 341)

    assert obj.filter == "CLEAR"
    assert obj.grating == "PRISM"
    assert len(obj.exptime) == obj.N
    assert obj.exptime.sum() == 3063.666

    assert np.allclose(obj.slit_pixel_scale, 0.1044, rtol=1.0e-3)
    assert np.allclose(obj.slit_shutter_scale, 0.2283, rtol=1.0e-3)
    assert np.allclose(obj.source_ypixel_position, 0.5079880, rtol=1.0e-3)

    assert obj.calc_reference_exposure == 2

    info = obj.info
    assert hasattr(info, "colnames")
    assert len(info) == obj.N

    for attr in [
        "mask",
        "sci",
        "dq",
        "bar",
        "var_rnoise",
        "var_poisson",
        "var_total",
        "wave",
        "yslit",
    ]:
        assert getattr(obj, attr).shape == (obj.N, obj.sh[0] * obj.sh[1])

    assert np.abs(obj.mask.sum() - (13967 - 82)) < 32

    meta = {
        "diffs": True,
        "grating_diffs": True,
        "trace_with_xpos": False,
        "trace_with_ypos": False,
        "trace_from_yoffset": True,
        "shutter_offset": 0.0,
        "stuck_threshold": 0.5,
        "bad_shutter_names": [],
        "dilate_failed_open": True,
        "undo_barshadow": False,
        "min_bar": 0.4,
        "bar_corr_mode": "wave",
        "fix_prism_norm": True,
        "wrapped_barshadow": False,
        "own_barshadow": True,
        "nod_offset": 5.06403847240714,
        "undo_pathloss": True,
        "reference_exposure": "auto",
        "pad_border": 2,
        "position_key": "y_index",
        "nhot": 0,
        "ncold": 0,
        "has_sky_arrays": False,
        "weight_type": "ivm",
        "percentile_outliers": 0,
        "removed_pathloss": "PATHLOSS_UN",
    }

    for k in meta:
        assert k in obj.meta
        assert obj.meta[k] == meta[k]

    assert np.allclose(np.nanmin(obj.wave), 1.34484, rtol=1.0e-3)
    assert np.allclose(np.nanmax(obj.wave), 5.3821, rtol=1.0e-3)

    assert np.nanmin(obj.bar[obj.mask]) > default_kwargs["min_bar"]

    # These are the same if there is no sky model
    assert np.allclose(obj.data[obj.mask], obj.sci[obj.mask])

    assert np.allclose(np.nanmin(obj.fixed_yshutter), -1.13236, rtol=1.0e4)
    assert np.allclose(np.nanmax(obj.fixed_yshutter), 1.132596, rtol=1.0e4)

    assert np.nanmax(obj.sky_background) == 0

    ##########
    # Sky
    obj.estimate_sky(
        mask_yslit=[[-4.5, 4.5]],
        min_bar=0.95,
        var_percentiles=[-5, -5],
        df=51,
        high_clip=0.8,
        use=False,
        outlier_threshold=7,
        absolute_threshold=0.2,
        make_plot=False,
    )
    assert np.nanmax(obj.sky_background) == 0

    obj.estimate_sky(
        mask_yslit=[[-4.5, 4.5]],
        min_bar=0.95,
        var_percentiles=[-5, -5],
        df=51,
        high_clip=0.8,
        use=True,
        outlier_threshold=7,
        absolute_threshold=0.2,
        make_plot=True,
    )
    plt.close("all")

    assert hasattr(obj, "sky_data")
    assert np.nanmax(obj.sky_background) > 0

    obj.estimate_sky(
        mask_yslit=[[-4.5, 4.5]],
        min_bar=0.95,
        var_percentiles=[1, 99],
        df=51,
        high_clip=0.8,
        use=True,
        outlier_threshold=7,
        absolute_threshold=0.2,
        make_plot=False,
    )

    obj.estimate_sky(
        mask_yslit=[[-4.5, 4.5]],
        min_bar=None,
        var_percentiles=[1, 99],
        df=51,
        high_clip=0.8,
        use=True,
        outlier_threshold=7,
        absolute_threshold=0.2,
        make_plot=False,
    )

    obj.estimate_sky(
        mask_yslit=[[-4.5, 4.5]],
        min_bar=0.95,
        var_percentiles=None,
        df=51,
        high_clip=0.8,
        use=True,
        outlier_threshold=7,
        absolute_threshold=0.2,
        make_plot=False,
    )

    # Read global sky
    obj.get_global_sky(sky_file="read")
    assert obj.meta["sky_file"] == "jw01345062001_sky.csv"

    obj.apply_normalization_correction()

    obj.meta["bar_corr_mode"] = "slit"
    obj.apply_spline_bar_correction()
    obj.meta["bar_corr_mode"] = "wave"
    obj.apply_spline_bar_correction()

    obj.flag_hot_cold_pixels()

    obj.flag_percentile_outliers(
        plev=[0.95, 0.999, 0.99999],
        scale=2,
        dilate=False,
        update=False,
    )

    obj.flag_percentile_outliers(
        plev=[0.95, 0.999, 0.99999],
        scale=2,
        dilate=True,
        update=True,
    )

    obj.flag_from_profile(
        grow=2,
        nfilt=-32,
        require_multiple=True,
        make_plot=False,
    )

    obj.flag_from_profile(
        grow=2,
        nfilt=-32,
        require_multiple=True,
        make_plot=True,
    )
    plt.close("all")

    obj.flag_from_profile(
        grow=2,
        nfilt=-32,
        require_multiple=False,
        make_plot=True,
    )

    # With wavecorr
    if False:
        default_kwargs["trace_with_xpos"] = True
        wobj = slit_combine.SlitGroup(files, "stest", **default_kwargs)

        mask = obj.mask & wobj.mask
        assert ~np.allclose(wobj.wave[mask], obj.wave[mask])

    default_kwargs["trace_with_xpos"] = False
    default_kwargs["undo_pathloss"] = 2
    pobj = slit_combine.SlitGroup(files, "stest", **default_kwargs)


def test_full_script():

    os.chdir(data_path())

    files = glob.glob(f"*phot.*{TARGETS[0]}.fits")
    files.sort()

    default_kwargs = dict(
        target=TARGETS[0],
        root="nirspec",
        path_to_files="./",
        files=None,
        do_gratings=["PRISM", "G395H", "G395M", "G235M", "G140M"],
        join=[0, 3, 5],
        split_uncover=True,
        stuck_threshold=0.0,
        pad_border=2,
        sort_by_sn=False,
        position_key="y_index",
        mask_cross_dispersion=None,
        cross_dispersion_mask_type="trace",
        trace_from_yoffset=False,
        reference_exposure="auto",
        trace_niter=4,
        offset_degree=0,
        degree_kwargs={},
        recenter_all=False,
        nod_offset=None,
        initial_sigma=7,
        fit_type=1,
        initial_theta=None,
        fix_params=False,
        input_fix_sigma=None,
        fit_params_kwargs=None,
        diffs=True,
        undo_pathloss=True,
        undo_barshadow=2,
        sky_arrays=None,
        use_first_sky=False,
        drizzle_kws={
            "step": 1,
            "with_pathloss": True,
            "wave_sample": 1.05,
            "ny": 13,
            "dkws": {"oversample": 16, "pixfrac": 0.8},
        },
        get_xobj=True,
        trace_with_xpos=False,
        trace_with_ypos=True,
        get_background=False,
        make_2d_plots=True,
        plot_kws={},
    )

    default_kwargs["diffs"] = True
    _ = slit_combine.extract_spectra(**default_kwargs)

    _ = slit_combine.extract_spectra(
        **default_kwargs, estimate_sky_kwargs={"make_plot": True}
    )

    default_kwargs["diffs"] = False
    _ = slit_combine.extract_spectra(
        **default_kwargs, estimate_sky_kwargs={"make_plot": True}
    )
    plt.close("all")

    # Fit redshift
    spec = spectrum.read_spectrum(
        f"nirspec_prism-clear_{TARGETS[0]}.spec.fits"
    )

    _ = spectrum.fit_redshift(
        f"nirspec_prism-clear_{TARGETS[0]}.spec.fits",
        z0=(4.1, 4.4),
    )

    assert np.allclose(_[2]["z"], 4.2337, rtol=1.0e2)
