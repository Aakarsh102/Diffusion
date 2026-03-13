#!/bin/bash
#PBS -N train_job
#PBS -A lighthouse-purdue
#PBS -q debug
#PBS -l select=1:system=polaris:ngpus=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o negsm.out


cd $PBS_O_WORKDIR

echo "Running on: $(hostname)"

# -----------------------
# Load Conda Environment
# -----------------------

# Load conda properly in non-interactive shell
source ~/.bashrc

module use /soft/modulefiles; module load conda;
conda activate trlx-stable


echo "Python being used:"
which python

# -----------------------
# Optional HF cache
# -----------------------
export HF_HOME=/lus/eagle/projects/lighthouse-purdue/rai53/.hf
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false


export NCCL_SOCKET_IFNAME=ib0

#python data/tinygsm.py
#python see.py --ckpt /lus/eagle/projects/lighthouse-purdue/rai53/PUMA/ckpts/date=2026-02-24-12-30/step=50000.pt --dataset tinygsm
python debug_mdm_checkpoint.py



#torchrun \
#    --nproc_per_node=1 \
#    --standalone \
#    train.py --cfg yaml_files/tinygsm_puma.yaml

