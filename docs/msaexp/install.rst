Installation instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

Install with `pip`
==================
.. code:: bash
    
    # Install with pip
    pip install msaexp
    python -c "import msaexp; print(msaexp.__version__)"
    

Install from the repository
===========================
.. code:: bash
    
    ### [OPTIONAL!] create a fresh conda environment
    conda create -n py39 python=3.9
    conda activate py39
    
    cd /usr/local/share/python # or some other location

    ### Fetch the eazy-py repo
    git clone https://github.com/gbrammer/msaexp.git
    
    ### Build the python code and include test tools
    cd msaexp
    pip install .[test]
    
    ### Run the test
    pytest
    
    
Binder Demo
~~~~~~~~~~~
.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gbrammer/msaexp/HEAD?filepath=docs%2Fexamples%2Fprocess-rxj2129.ipynb
