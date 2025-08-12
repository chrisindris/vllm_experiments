#!/bin/bash

# NOTE: RUN THIS ON LOGIN NODE
# vLLM models typically come from the huggingface hub. 
module load python/3.12 git-lfs/3.4.0
virtualenv --no-download temp_env && source temp_env/bin/activate
pip install --no-index huggingface_hub
huggingface-cli download facebook/opt-125m # default location: $HOME/.cache/huggingface/hub
rm -r temp_env

# once that's done, you can run sbatch vllm_example.sh