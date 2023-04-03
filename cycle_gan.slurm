#!/bin/bash


##NECESSARY JOB SPECIFICATIONS

#SBATCH --job-name=cycle_gan_faces
#SBATCH --time=2:00:00                 #Set the wall clock limit to 9hr and 30min
#SBATCH --nodes=1
#SBATCH --ntasks=1                     #Request 1 task
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G                     
#SBATCH --output=cycle_gan_faces_experimental_.%j     #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:a100:1               #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu                 #Request 1 GPU per node can be 1 or 2
#SBATCH --reservation=benchmarking
#SBATCH --nodelist=fc024
##SBATCH --exclusive

module load WebProxy 

module load    GCCcore/9.3.0    Python/3.8.2

module load   GCC/10.3.0 OpenMPI/4.1.1 PyTorch/1.12.1-CUDA-11.3.1

#python -m venv cvenv # Create virtual env

source cvenv/bin/activate

##Install Dependencies

#pip install --upgrade pip

#pip install torchvision

#pip install yacs

#pip install opencv-python

#pip install einops


#mpirun -n 1 python3 /scratch/group/pvamu-tamu/code/pseudo-sr-main/train.py /scratch/group/pvamu-tamu/code/pseudo-sr-main/configs/faces.yaml

CUDA_VISIBLE_DEVICES=0 python3 /scratch/group/pvamu-tamu/code/pseudo-sr-main/train.py /scratch/group/pvamu-tamu/code/pseudo-sr-main/configs/faces.yaml

