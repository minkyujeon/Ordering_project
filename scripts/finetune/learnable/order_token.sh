#!/bin/bash
#SBATCH --job-name=mar_ig_train
#SBATCH -p cryoem
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mj7341@princeton.edu

# ----------------------------
# Environment setup
# ----------------------------
module purge
module load anaconda3/2023.3
source ~/.bashrc
conda activate mar_orders

# ----------------------------
# Paths
# ----------------------------
OUTPUT_DIR=/scratch/gpfs/ZHONGE/mj7341/research/03_mar/MAR/learnable/order_token
IMAGENET_PATH=/scratch/gpfs/ZHONGE/mj7341/data/imagenet
VAE_PATH=/scratch/gpfs/ZHONGE/mj7341/github_repos/mar/pretrained_models/vae/kl16.ckpt
RESUME_DIR=/scratch/gpfs/ZHONGE/mj7341/github_repos/mar/pretrained_models/mar/mar_base/checkpoint-last.pth

mkdir -p "$OUTPUT_DIR"

# # Derive master and ranks from SLURM
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=12345

# (optional) NCCL networking hints — pick the right interface for your cluster
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
# If you know the interface, uncomment and set it (e.g., ib0 or eno1):
# export NCCL_SOCKET_IFNAME=ib0

# Keep W&B offline
export WANDB_MODE=offline
# ----------------------------
# Launch
# ----------------------------
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_PER_TASK bash -lc "
  source ~/.bashrc
  conda activate mar_orders
  echo Running on host: \$(hostname)

  which python; python -V
  LAUNCHER=\$(command -v torchrun || echo 'python -m torch.distributed.run')

  \$LAUNCHER \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main_mar_order_token.py \
      --img_size 256 \
      --vae_path $VAE_PATH \
      --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
      --model mar_base \
      --diffloss_d 6 --diffloss_w 1024 \
      --start_epoch 0 --epochs 100 \
      --warmup_epochs 10 \
      --batch_size 64 --diffusion_batch_mul 4 \
      --pretrained_mar_path $RESUME_DIR \
      --output_dir $OUTPUT_DIR \
      --data_path $IMAGENET_PATH \
      --order_mode gradient \
      --save_last_freq 10 \
      --online_eval \
      --eval_freq 10 \
      --num_images_eval 5000 \
"