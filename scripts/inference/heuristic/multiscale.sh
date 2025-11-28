#!/bin/bash
#SBATCH --job-name=job
#SBATCH -p cryoem
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mj7341@princeton.edu

module purge
module load anaconda3/2023.3
conda activate mar_orders

IMAGENET_PATH=/scratch/gpfs/ZHONGE/mj7341/data/imagenet
RESUME_DIR=/scratch/gpfs/ZHONGE/mj7341/github_repos/mar/pretrained_models/mar/mar_large
OUTPUT_DIR=/scratch/gpfs/ZHONGE/mj7341/research/03_mar/MAR/heuristic_orders/mar_large/multiscale

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    main_mar_spatial.py \
    --model mar_large --diffloss_d 8 --diffloss_w 1280 \
    --eval_bsz 256 --num_images 50000 \
    --num_iter 256 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
    --output_dir $OUTPUT_DIR \
    --resume $RESUME_DIR \
    --order_mode multiscale \
    --data_path ${IMAGENET_PATH} --evaluate