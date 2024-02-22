#!/bin/bash

module use /appl/local/csc/modulefiles/
module load tensorflow/2.12

pip install --user --upgrade ../..
