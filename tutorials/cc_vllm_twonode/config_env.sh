#!/bin/bash

module load python/3.12 gcc cuda/12.6 opencv/4.11 arrow/19

virtualenv --no-download $SLURM_TMPDIR/ENV

source $SLURM_TMPDIR/ENV/bin/activate

pip install --upgrade pip --no-index

pip install ray -r /project/def-wangcs/indrisch/vllm/tutorials/cc_vllm_singlenode/vllm_requirements.txt --no-index

deactivate