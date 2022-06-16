#!/bin/bash
#SBATCH -A INF22_teongrav_0
#SBATCH -p m100_usr_prod
#SBATCH --time 01:00:00                    # format: HH:MM:SS
#SBATCH -N 1                               # 1 node
#SBATCH --ntasks-per-node=32              # 8 tasks out of 128
#SBATCH --gres=gpu:1                       # 1 gpus per node out of 4
#SBATCH --mem=32Gb                         # memory per node out of 246000MB
#SBATCH --output=/m100/home/userexternal/adiana00/Tesi-ML/output/output.out
#SBATCH --job-name=fsrcnn_test_job
#SBATCH --error=/m100/home/userexternal/adiana00/Tesi-ML/error_file/run0.err
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.diana3@campus.unimib.it

cd /m100/home/userexternal/adiana00/Tesi-ML/FSRCNN/

module load profile/deeplrn
module load autoload cineca-ai/2.1.0
module load autoload python
module load autoload pytorch
export PYTHONPATH=$PYTHONPATH:/cineca/prod/opt/libraries/cineca-ai/2.1.0/none/cineca-ai-conda-env-py3.8-cuda-openmpi-11.0/lib/python3.8/site-packages

python3 diagnostic.py
