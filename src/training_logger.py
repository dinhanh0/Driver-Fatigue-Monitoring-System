"""
Training output manager to standardize logging and metrics.
"""
import json
from pathlib import Path
from datetime import datetime
import torch


class TrainingLogger:
    def __init__(self, out_dir: Path, config: dict):
        """
        Initialize training logger with output directory and config.
        
        Args:
            out_dir: Path to output directory for logs and checkpoints
            config: Dictionary containing training configuration
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.out_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        
        # Save initial config
        self._save_config()
        
    def _save_config(self):
        """Save training configuration to JSON file."""
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def log_epoch(self, epoch: int, metrics: dict, phase: str = 'train'):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
            phase: Either 'train' or 'val'
        """
        self.current_epoch = epoch
        
        # Format metrics string
        metrics_str = " ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        log_str = f"[{phase.upper()}] Epoch {epoch:3d}: {metrics_str}"
        
        # Write to console and log file
        print(log_str)
        with open(self.run_dir / "training.log", 'a') as f:
            f.write(log_str + '\n')
            
        # Update best validation accuracy
        if phase == 'val' and metrics.get('acc', 0) > self.best_val_acc:
            self.best_val_acc = metrics.get('acc', 0)
            self.epochs_no_improve = 0
            return True  # Signal to save checkpoint
        elif phase == 'val':
            self.epochs_no_improve += 1
            
        return False
            
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler = None):
        """
        Save model checkpoint with metrics.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save  
            scheduler: Optional learning rate scheduler to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        # Save latest checkpoint
        latest_path = self.run_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # If this is best validation accuracy, save best checkpoint
        best_path = self.run_dir / 'best.pt'
        if self.epochs_no_improve == 0:
            torch.save(checkpoint, best_path)