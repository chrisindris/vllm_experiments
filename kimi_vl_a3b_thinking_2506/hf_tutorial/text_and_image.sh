#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1  
#SBATCH --mem=16000M       
#SBATCH --time=0-00:05
#SBATCH --output=%N-%j.out

# before running this script, run the vllm_setup.sh script to create the vllm_requirements.txt file
# run with sbatch text_and_image.sh
# it seems like at least --time=0-00:05 is needed for the job to run

module load python/3.12 gcc opencv/4.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install -r vllm_requirements_2.txt --no-index

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=DEBUG && python text_and_image.py