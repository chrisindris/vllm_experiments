#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1  
#SBATCH --mem=16000M       
#SBATCH --time=0-00:05
#SBATCH --output=%N-%j.out

# before running this script, run the vllm_setup.sh script to create the vllm_requirements.txt file
# run with sbatch kimivl_3d_test.sh
# it seems like at least --time=0-00:05 is needed for the job to run

module load python/3.12 gcc opencv/4.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install -r /project/def-wangcs/indrisch/vllm/kimi_vl_a3b_thinking_2506/hf_tutorial/vllm_requirements_2.txt --no-index

./experiment_runner.sh 0