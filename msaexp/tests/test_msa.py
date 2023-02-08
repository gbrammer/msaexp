from .. import msa

def test_meta_parser():
    import matplotlib.pyplot as plt
    
    uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
    meta = msa.MSAMetafile(uri+'jw02756001001_01_msa.fits')
    
    regs = meta.regions_from_metafile(as_string=True, with_bars=True)
    
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    _ = meta.plot_slitlet(source_id=110003, cutout_size=12.5,
                          rgb_filters=None, ax=ax)
    plt.close('all')


def test_wrapper():
    
    uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
    regs = msa.regions_from_metafile(uri+'jw02756001001_01_msa.fits',
                                       as_string=True,
                                       with_bars=True)
    
def test_summary():
    
    uri = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'
    meta = msa.MSAMetafile(uri+'jw02756001001_01_msa.fits')
    
    res = meta.make_summary_table(msa_metadata_id=None,
                                  image_path=None,
                                  write_tables=False)