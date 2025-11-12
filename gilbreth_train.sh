#!/bin/bash
#SBATCH --job-name=drowsiness_train
#SBATCH --output=outputs/train_%j.log
#SBATCH --error=outputs/train_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Job submission script for training on Gilbreth cluster
# Submit with: sbatch gilbreth_train.sh

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load anaconda
module load cuda

# Activate environment
source activate drowsiness

# Check GPU
echo "GPU Info:"
nvidia-smi

# Create output directory
mkdir -p outputs

# Download datasets (only if not already present)
echo "Checking for datasets..."
python -c "from src.data.kaggle_fetch import download_datasets; download_datasets('data/raw_datasets')"

# Build dataset index
echo "Building dataset index..."
python -m src.main --build-index

# Run preprocessing (if not already done)
if [ ! -d "data/processed/windows/train" ]; then
    echo "Running preprocessing..."
    python -m src.preprocess \
        --index-csv data/processed/dataset_index.csv \
        --out-dir data/processed/windows \
        --win-sec 3.0 \
        --stride-sec 1.0 \
        --img-size 224
fi

# Training
echo "Starting training..."
python -m src.train \
    --splits_csv data/processed/dataset_index.csv \
    --windows_root data/processed/windows \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --num_classes 8 \
    --seq_len 40 \
    --img_size 224 \
    --patience 15 \
    --num_workers 4 \
    --use_amp \
    --out_dir outputs/model1

echo "Training complete!"
echo "Job finished at: $(date)"
