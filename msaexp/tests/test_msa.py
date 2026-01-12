import os
import numpy as np

from .. import msa


def test_meta_parser():
    import matplotlib.pyplot as plt

    uri = (
        "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/"
    )
    meta = msa.MSAMetafile(uri + "jw02756001001_01_msa.fits")

    assert len(meta.metadata_id_list) == 2
    assert meta.metadata_id_unique.N == 2
    assert len(meta.key_pairs) == 6

    regs = meta.regions_from_metafile(as_string=True, with_bars=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _ = meta.plot_slitlet(
        source_id=110003, cutout_size=12.5, rgb_filters=None, ax=ax
    )
    plt.close("all")


def test_wrapper():

    uri = (
        "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/"
    )
    regs = msa.regions_from_metafile(
        uri + "jw02756001001_01_msa.fits", as_string=True, with_bars=True
    )


def test_summary():

    uri = (
        "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/"
    )
    meta = msa.MSAMetafile(uri + "jw02756001001_01_msa.fits")

    res = meta.make_summary_table(
        msa_metadata_id=None, image_path=None, write_tables=False
    )


def test_mast_queries():

    uri = (
        "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/"
    )
    meta = msa.MSAMetafile(uri + "jw02756001001_01_msa.fits")

    # Query exposure metadata
    mast = meta.query_mast_exposures()

    # regions from siaf transforms
    regs = meta.regions_from_metafile_siaf()

    regs = meta.all_regions_from_metafile_siaf()

    transforms = msa.fit_siaf_shutter_transforms()

    inv_transforms = msa.load_siaf_inverse_shutter_transforms()

    # Fit for a pointing offset
    meta.fit_mast_pointing_offset()

    # regions from siaf transforms
    regs = meta.regions_from_metafile_siaf()


def test_pad():

    uri = (
        "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/"
    )
    file = uri + "jw02756001001_01_msa.fits"

    for p in [0, 1]:
        output_file = msa.pad_msa_metafile(
            file,
            pad=p,
            source_ids=None,
            slitlet_ids=None,
            positive_ids=False,
            prefix="src_",
            verbose=True,
            primary_sources=True,
        )

        assert os.path.exists(output_file)
        os.remove(output_file)

    meta = msa.MSAMetafile(file)
    slitlet_ids = np.unique(meta.shutter_table["slitlet_id"])[:10]

    for p in [0, 1]:
        output_file = msa.pad_msa_metafile(
            file,
            pad=p,
            source_ids=None,
            slitlet_ids=slitlet_ids,
            positive_ids=False,
            prefix="src_",
            verbose=True,
            primary_sources=True,
        )

        assert os.path.exists(output_file)
        os.remove(output_file)

    source_ids = np.unique(meta.shutter_table["source_id"])[:10]

    for p in [0, 1]:

        output_file = msa.pad_msa_metafile(
            file,
            pad=p,
            source_ids=source_ids,
            slitlet_ids=None,
            positive_ids=False,
            prefix="src_",
            verbose=True,
            primary_sources=True,
        )

        assert os.path.exists(output_file)
        os.remove(output_file)

        output_file = msa.pad_msa_metafile(
            file,
            pad=p,
            source_ids=source_ids,
            slitlet_ids=slitlet_ids,
            positive_ids=False,
            prefix="src_",
            verbose=True,
            primary_sources=True,
        )

        assert os.path.exists(output_file)
        os.remove(output_file)

    output_file = msa.pad_msa_metafile(
        file,
        pad=0,
        source_ids=None,
        slitlet_ids=None,
        positive_ids=True,
        prefix="src_",
        verbose=True,
        primary_sources=True,
    )

    assert os.path.exists(output_file)
    os.remove(output_file)


def test_wavelength_limits():
    import matplotlib.pyplot as plt
    import grizli.utils
    
    cols, rows, quadrants = [], [], []
    for q in [1,2,3,4]:
        for c in np.arange(50, 360, 50, dtype=int):
            for r in np.arange(50, 170, 50, dtype=int):
                quadrants.append(q)
                rows.append(r)
                cols.append(c)
    
    tab = msa.get_shutter_wavelength_limits(
        cols, rows, quadrants, grating='prism', filter='clear'
    )

    # Check
    ref = grizli.utils.read_catalog(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "wavelength_range_coeffs_prism_clear_check.csv"
        )
    )

    for c in tab.colnames:
        if "wave" in c:
            v1 = np.isfinite(tab[c])
            v2 = np.isfinite(ref[c])
            assert(v1.sum() == v2.sum())
            assert(np.allclose(tab[c][v1 & v2], ref[c][v1 & v2], rtol=1.e-2))

    # Compare to "MSA Target Info" table export from APT
    MAKE_PLOT = False

    for APT_FILE in [
        "4233-obs2-exp1-c1_uds_obs2az_prism_bkge1n1-PRISM-CLEAR.csv",
        "4233-obs2-exp2-c1_uds_obs2az_g395me2n1-G395M-F290LP.csv",
    ]:
        apt = grizli.utils.read_catalog(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                APT_FILE
            )
        )

        grating = os.path.basename(APT_FILE).split('-')[-2].lower()
        filter = os.path.basename(APT_FILE).split('-')[-1].lower().split('.')[0]

        tab = msa.get_shutter_wavelength_limits(
            apt['Column (Disp)'],
            apt['Row (Spat)'],
            apt['Quadrant'],
            grating=grating, filter=filter,
        )

        if MAKE_PLOT:
            fig, ax = plt.subplots(1,1)

        for det in ['NRS1','NRS2']:
            for limit in ['Min', 'Max']:
                this_col = f'{det}_wave_{limit}'.lower()
                apt_col = f'{det} {limit} Wave'
                # plt.scatter(
                #     apt[apt_col], tab[this_col],
                #     c=tab['quadrant'],
                #     alpha=0.5
                # )
                # apt_ll = apt[apt_col] == -2.0
                # tab[apt_col] = apt[apt_col]

                test = np.isfinite(tab[this_col] + apt[apt_col])
                test &= apt[apt_col] > 0

                if test.sum() > 0:
                    if MAKE_PLOT:
                        plt.scatter(
                            apt[apt_col][test],
                            (tab[this_col] - apt[apt_col])[test],
                            c=tab['quadrant'][test],
                        )
                    else:
                        assert np.allclose(
                            tab[this_col][test],
                            apt[apt_col][test],
                            atol=0.15
                        )

