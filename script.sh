#!/bin/bash

#SBATCH --job-name=scenegraph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx-small
#SBATCH --output=train_outs/gen_data/out/%x.%j.out
#SBATCH --error=train_outs/gen_data/errors/%x.%j.err
#SBATCH --mail-type=ALL

python create_description.py