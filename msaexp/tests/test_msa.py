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
