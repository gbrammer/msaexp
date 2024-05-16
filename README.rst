.. image:: https://github.com/gbrammer/msaexp/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/gbrammer/msaexp/actions

.. image:: https://badge.fury.io/py/msaexp.svg
    :target: https://badge.fury.io/py/msaexp
    
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7579050.svg
   :target: https://doi.org/10.5281/zenodo.7579050
   
.. image:: https://readthedocs.org/projects/msaexp/badge/?version=latest
    :target: https://msaexp.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    
`msaexp`
===================================================================================
Tools for extracting JWST NIRSpec MSA spectra directly from the telescope exposures

Under heavy construction....

.. note::
    Please submit any questions/comments/problems you have through the `Issues <https://github.com/gbrammer/msaexp/issues>`_ interface.

~~~~~~~~~~~~~
Documentation
~~~~~~~~~~~~~

Documentation will be here: https://msaexp.readthedocs.io/, though it's essentially just the module API for now.

The overall procedure is demonstrated in these notebooks:

- `spectral-pipeline-2024.ipynb <https://github.com/gbrammer/msaexp/blob/main/docs/examples/spectral-pipeline-2024.ipynb>`_: Run the pipeline from scratch on an example dataset from GO-4233 (RUBIES)
- `spectral-extractions-2024.ipynb <https://github.com/gbrammer/msaexp/blob/main/docs/examples/spectral-extractions-2024.ipynb>`_: Brief demo of some fitting applications to the reduced spectra

Try running the 2024 demo notebooks directly on GitHub in a Codespace: 
  1. Fork the repository from https://github.com/gbrammer/msaexp
  2. "<> Code" pulldown > "Codespaces" > "+" to start the codespace
  3. Wait for the initialization to complete
  4. Navigate to the "docs/examples/" directory in the codespace
  5. Open "spectral-pipeline-2024.ipynb" and run it
  6. Commit to your forked repository any changes to the notebook itself or files created that you want to save outside the codespace
  7. Profit!

The notebooks below use some of the older deprecated methodolgy for spectral combination and extraction:

- `drizzled-nirspec.ipynb <https://github.com/gbrammer/msaexp/blob/main/docs/examples/drizzled-nirspec.ipynb>`_: Demo of new drizzling and combination code (`0.6.0`) with prism data from JWST program `GO-1433 <https://www.stsci.edu/cgi-bin/get-proposal-info?id=1433&observatory=JWST>`_ (PI: Dan Coe)
- `process-rxj2129.ipynb <https://github.com/gbrammer/msaexp/blob/main/docs/examples/process-rxj2129.ipynb>`_: Demo with prism data from JWST program `DD-2767 <https://www.stsci.edu/cgi-bin/get-proposal-info?id=2756&observatory=JWST>`_ (PI: Pat Kelly)
- `process-smacs0723.ipynb <https://github.com/gbrammer/msaexp/blob/main/docs/examples/process-smacs0723.ipynb>`_: Demo with medium resolution data from the JWST `ERO-2736 <https://www.stsci.edu/cgi-bin/get-proposal-info?id=2736&observatory=JWST>`_ program on the cluster SMACS-0723 (PI: Klaus Pontoppidon)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extracted spectra from public datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Browsable table: https://s3.amazonaws.com/msaexp-nirspec/extractions/nirspec_graded.html 

Full catalog: https://s3.amazonaws.com/msaexp-nirspec/extractions/nirspec_graded_v0.ecsv

Updated and additional extractions (2024-Feb): https://s3.amazonaws.com/msaexp-nirspec/extractions/nirspec_graded_v2.html
(Though see https://github.com/gbrammer/msaexp/pull/54)

