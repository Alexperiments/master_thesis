#!/bin/bash
#SBATCH -A INF21_teongrav_0
#SBATCH -p m100_usr_prod
#SBATCH --time 24:00:00                    # format: HH:MM:SS
#SBATCH -N 1                               # 1 node
#SBATCH --ntasks-per-node=64                # 8 tasks out of 128
#SBATCH --gres=gpu:4                       # 1 gpus per node out of 4
#SBATCH --mem=40Gb                         # memory per node out of 246000MB
#SBATCH --job-name=sr_job_run0
#SBATCH --error=error_file/run0.err
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.diana3@campus.unimib.it

module load profile/deeplrn
module load cuda/10.2
module load gnu/8.4.0
module load python/3.8.2
module load openblas/0.3.9--gnu--8.4.0
module load magma/2.5.3--cuda--10.2
module load cudnn/8.0.4--cuda--10.2
module load nccl/2.7.8--cuda--10.2
module load ffmpeg/4.3--gnu--8.4.0
module load numpy/1.19.4--python--3.8.2
module load pytorch/1.7--cuda--10.2
module load szip/2.1.1--gnu--8.4.0
module load zlib/1.2.11--gnu--8.4.0
module load hdf5/1.10.6--gnu--8.4.0
module load tensorflow/2.3.0--cuda--10.2

cd /m100_scratch/userexternal/frigamon/test3_numba_054/FSRCNN/network/

python3 training_routine_tensorboard.py
