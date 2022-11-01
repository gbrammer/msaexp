def summary_from_metafiles():
    """
    """
    import glob
    from grizli import utils, prep
    import astropy.io.fits as pyfits
    
    files = glob.glob('*msa.fits')
    files.sort()
    
    for file in files:
        im = pyfits.open(file)
        tab = utils.GTable(im['SOURCE_INFO'].data)
        tab.write(file.replace('.fits', '.csv'), overwrite=True)
        prep.table_to_regions(tab, file.replace('.fits','.reg'), 
                              comment=tab['source_name'], size=0.1)
                              