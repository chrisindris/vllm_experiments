#!/bin/bash
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=2 
#SBATCH --mem=32000M       
#SBATCH --time=0-00:03
#SBATCH --output=%N-%j.out

## Create a virtualenv and install Ray on all nodes ##

module load gcc cuda/12.6 python/3.12 arrow/19 opencv/4.11

srun -N $SLURM_NNODES -n $SLURM_NNODES config_env.sh

export HEAD_NODE=$(hostname --ip-address) # store head node's address
export RAY_PORT=34567 # choose a port to start Ray on the head node 

## Set Huggingface libraries to OFFLINE mode ##

export HF_HUB_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

source $SLURM_TMPDIR/ENV/bin/activate

## Start Ray cluster Head Node ##
ray start --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=2 --block &
sleep 10