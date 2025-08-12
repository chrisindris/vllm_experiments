#!/bin/bash
# This script creates a temporary virtual environment and saves the requirements to a file (which can then be used to install the environment on a compute node)
# for vLLM: https://github.com/vllm-project/vllm
# Run it on the login node:
# ./vllm_setup.sh

module load cuda/12.6 opencv/4.11 python/3.12
virtualenv --no-download ~/vllm_env
source ~/vllm_env/bin/activate
pip install --no-index --upgrade pip
MAX_JOBS=4 pip install --no-index --no-build-isolation vllm transformers==4.53.2 flash-attn==2.7.4.post1 blobfile huggingface_hub torch==2.6.0 triton==3.2.0 # this works
# use transformers<4.54.0 because of: (ValueError: 'aimv2' is already used by a Transformers config, pick another name.)
# use torch==2.6.0 and triton==3.2.0 because of: ImportError: cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'
# TODO: does flashinfer_python work? Seems like it forces earlier versions of vllm, not ideal.
pip freeze > vllm_requirements.txt
deactivate
rm -rf ~/vllm_env