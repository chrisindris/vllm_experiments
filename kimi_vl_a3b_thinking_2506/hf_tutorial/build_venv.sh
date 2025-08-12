#!/bin/bash

# RUN ON: Login Node
# RUN AS: ./build_venv.sh

module load cuda/12.6 opencv/4.11 python/3.12
virtualenv --no-download ~/vllm_env
source ~/vllm_env/bin/activate
pip install --no-index --upgrade pip
# MAX_JOBS=4 pip install --no-index --no-build-isolation vllm transformers==4.53.2 flash-attn==2.7.4.post1 blobfile huggingface_hub torch==2.6.0 triton==3.2.0 # this works, and makes kimivl work
if [ ! -f "flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" ]; then
    echo "Downloading flash-attention wheel..."
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
else
    echo "Flash-attention wheel already exists, skipping download."
fi
MAX_JOBS=4 pip install --no-index --no-build-isolation vllm==0.9.2 "transformers<4.54.0" flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl blobfile huggingface_hub torch triton==3.2.0 # This also works for kimivl
# use transformers<4.54.0 because of: (ValueError: 'aimv2' is already used by a Transformers config, pick another name.)
# use torch==2.6.0 and triton==3.2.0 because of: ImportError: cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'
# TODO: does flashinfer_python work? Seems like it forces earlier versions of vllm, not ideal.
pip freeze > vllm_requirements_2.txt
deactivate
rm -rf ~/vllm_env