#!/usr/bin/env bash

# setup conda environment
eval "$(conda shell.bash hook)"
# check if conda env exist
if conda env list | grep ".*RUN_ENV.*" >/dev/null 2>&1; then
  conda activate dis

else
  echo "===SETUP CONDA ENVIRONMENT=="
  conda env create -f dis.yml
  conda activate dis
  pip install csv-logger
  pip install wget
  pip install tensorboard
  pip install mlflow
fi


