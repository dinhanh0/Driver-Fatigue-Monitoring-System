# Running on Gilbreth Cluster

## Initial Setup (One Time)

1. **Upload project to Gilbreth:**
```bash
# From your local machine
scp -r Facial-Recognition-Driver username@gilbreth.rcac.purdue.edu:~/
```

2. **SSH into Gilbreth:**
```bash
ssh username@gilbreth.rcac.purdue.edu
cd ~/Facial-Recognition-Driver
```

3. **Run setup script:**
```bash
bash gilbreth_setup.sh
```

## Running Training

**Submit training job:**
```bash
sbatch gilbreth_train.sh
```

**Check job status:**
```bash
squeue -u $USER
```

**View output logs:**
```bash
tail -f outputs/train_JOBID.log
```

**Cancel job if needed:**
```bash
scancel JOBID
```

## Running Evaluation

**After training completes:**
```bash
sbatch gilbreth_eval.sh
```

## Expected Timeline

- **Setup:** 10-15 minutes
- **Dataset download:** 5-10 minutes (automatic)
- **Preprocessing:** 15-20 minutes
- **Training:** 5-7 hours (GPU)
- **Evaluation:** <5 minutes

## Checking Results

**Training metrics:**
```bash
cat outputs/train_JOBID.log | grep "Epoch"
```

**Best model saved at:**
```
outputs/model1/best.pt
```

**Download results to local machine:**
```bash
scp username@gilbreth.rcac.purdue.edu:~/Facial-Recognition-Driver/outputs/model1/best.pt .
```

## Troubleshooting

**Job won't start:**
- Check partition availability: `sinfo -p gpu`
- Try longer time limit or different partition

**Out of memory:**
- Reduce batch size in `gilbreth_train.sh` (line with `--batch_size`)
- Try `--batch_size 4` instead of 8

**Slow dataset download:**
- May need to configure Kaggle credentials:
```bash
mkdir -p ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Check GPU usage while job runs:**
```bash
# Get the node your job is on
squeue -u $USER
# SSH to that node
ssh node_name
nvidia-smi
```

## Monitoring Training

View training progress in real-time:
```bash
watch -n 10 "tail -50 outputs/train_*.log | grep -E 'Epoch|loss|acc|f1|LR'"
```

## Phase IV Deliverables

After training completes, you'll have:
- `outputs/model1/best.pt` - Trained model checkpoint
- `outputs/train_*.log` - Training logs with metrics
- `outputs/eval_*.log` - Evaluation results

These contain all the metrics you need for your presentation!
