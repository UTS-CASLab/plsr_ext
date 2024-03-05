**plsr_ext** is an open-source library for the extended versions of the 
Partial Least Squares Regression (PLSR).

======
Models
======
This library includes the following models:

- Just-in-time PLSR
- Locally Weighted PLSR
- K-nearest-based Locally Weighted PLSR
- PLSR using NIPALS algorithm
- Recursive PLSR

============
Installation
============

Dependencies
~~~~~~~~~~~~

**lwpr** requires:

- Python
- NumPy
- Scipy
- sklearn

From the downloaded source code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please navigate to the downloaded package and type ::

    python install .

After install the library, you can check if the library has been 
installed in your machine or not by executing the testing scripts in 
the "plsr_ext" subdirectory such as::
    python test_plsr.py

Install from Github code
~~~~~~~~~~~~~~~~~~~~~~~~~

To install lwpr from source, proceed as follows::
    
    git clone https://github.com/UTS-CASLab/plsr_ext.git  
    cd plsr_ext
    python setup.py install
