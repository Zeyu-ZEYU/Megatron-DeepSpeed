#! /usr/bin/env bash

pip install deepspeed==0.10.3
pip install six
pip install regex
# install NVIDIA/apex (needs cuda, should git checkout 741bdf50825a97664db085749)
# export PATH=/bigtemp/qxc4fh/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/bigtemp/qxc4fh/cuda/lib64:$LD_LIBRARY_PATH
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
pip install pybind11
