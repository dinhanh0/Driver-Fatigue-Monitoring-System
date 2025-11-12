#!/bin/bash
#SBATCH --job-name=drowsiness_eval
#SBATCH --output=outputs/eval_%j.log
#SBATCH --error=outputs/eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Job submission script for evaluation on Gilbreth cluster
# Submit with: sbatch gilbreth_eval.sh

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Load modules
module load anaconda
module load cuda

# Activate environment
source activate drowsiness

# Evaluation
echo "Running evaluation..."
python -m src.eval \
    --splits_csv data/processed/dataset_index.csv \
    --subset test \
    --ckpt outputs/model1/best.pt \
    --batch_size 16 \
    --num_classes 8 \
    --seq_len 40 \
    --img_size 224 \
    --num_workers 4

echo "Evaluation complete!"
echo "Job finished at: $(date)"
