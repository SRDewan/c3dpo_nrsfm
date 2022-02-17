#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=opFile.txt

module load cuda/11.0
module load cudnn/8-cuda-11.0
module load python/3.9.1

conda activate c3dpo 
python ./experiment.py --cfg_file ./cfgs/cars.yaml
