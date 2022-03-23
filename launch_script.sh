#!/bin/bash
#SBATCH -A INF22_teongrav_0
#SBATCH -p m100_usr_prod
#SBATCH --time 24:00:00                    # format: HH:MM:SS
#SBATCH -N 1                               # 1 node
#SBATCH --ntasks-per-node=64               # 8 tasks out of 128
#SBATCH --gres=gpu:4                       # 1 gpus per node out of 4
#SBATCH --mem=40Gb                         # memory per node out of 246000MB
#SBATCH --job-name=sr_job_run0
#SBATCH --error=error_file/run0.err
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.diana3@campus.unimib.it

cd /m100/home/userexternal/adiana00/Tesi-ML/FSRCNN/

python3 test_cineca.py
