#!/bin/bash

# RUN ON: Login Node
# RUN AS: ./ load_model.sh

# vLLM models typically come from the huggingface hub. 
module load python/3.12 git-lfs/3.4.0
virtualenv --no-download temp_env && source temp_env/bin/activate
pip install --no-index huggingface_hub
#huggingface-cli download moonshotai/Kimi-VL-A3B-Thinking-2506 # default location: $HOME/.cache/huggingface/hub
HF_HUB_DISABLE_XET=1 hf download --max-workers=4 moonshotai/Kimi-VL-A3B-Thinking-2506 # using --local-dir and --cache-dir; default location: $HOME/.cache/huggingface/hub
rm -r temp_env

# once that's done, you can run sbatch vllm_example.sh