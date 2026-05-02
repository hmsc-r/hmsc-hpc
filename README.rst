========
Hmsc-HPC
========

This repository contains the TensorFlow implementation of Hierarchical Modelling of Species Communities (HMSC), serving as an extension to the existing R package, Hmsc-R. It provides a flexible framework for Joint Species Distribution Modelling (JSDM), enabling robust analysis and modeling of species communities.

The Hmsc-HPC paper is published in `PLOS Computational Biology <https://doi.org/10.1371/journal.pcbi.1011914>`_

Contents
--------

* **Code (Directory: hmsc/\*):** contains GPU-compatible implementation of HMSC model fitting algorithm.
* **Demo (Directory: examples/basic_example/\*):** demonstrates the implementation. It serves as a comprehensive guide and example showcasing the functionalities and usage of Hmsc-HPC.
* **Data (Directory: examples/big_spatial/\*):** contains the data and scripts used for performance comparisons presented in the manuscript.

Instructions for new users
--------------------------

#. Open the `basic_example/example.Rmd <examples/basic_example/example.Rmd>`_ notebook.
#. Follow the step-by-step instructions and code snippets to explore the functionalities and usage of the Hmsc-HPC implementation.
#. `Detailed install instruction for CSC clusters <docs/csc_install.md>`_
