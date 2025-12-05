# src/visual_demo.py
# Animated demo for a single window (.npz):
# - Plays frames as a video
# - Plots EAR/MAR/blur/illum over time with moving cursor
# - Shows model's probability and predicted label

from pathlib import Path
import sys
import argparse

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ------------------------------------------------------------------
# Paths / imports
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.resnet18_lstm import ResNet18LSTM  # backbone from your project

WINDOWS_ROOT = PROJECT_ROOT / "data" / "processed" / "windows_from_index"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_LOSS_PATH = MODEL_DIR / "resnet18_lstm_binary_best_loss.pt"
BEST_ACC_PATH = MODEL_DIR / "resnet18_lstm_binary_best_acc.pt"

# ------------------------------------------------------------------
# ImageNet normalization (must match train.py)
# ------------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
    1, 3, 1, 1
)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
    1, 3, 1, 1
)


# ------------------------------------------------------------------
# Binary label mapping (same as train.py)
# ------------------------------------------------------------------
def raw_to_binary(y_raw: int) -> int:
    if y_raw in (0, 1):
        return int(y_raw)
    return 0 if y_raw <= 0 else 1


# ------------------------------------------------------------------
# Model used for demo: same as VideoFeatResNet18LSTM in train.py
# ------------------------------------------------------------------
class VideoFeatResNet18LSTM(nn.Module):
    """
    Same architecture as in train.py:
      - ResNet18LSTM backbone outputs embedding
      - Concatenate with aggregated handcrafted features (EAR/MAR/blur/illum)
      - Final classifier -> single logit
    """

    def __init__(
        self,
        feat_dim_extra: int,  # F
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.5,
        use_nat: bool = False,
        nat_kernel: int = 7,
        nat_dilation: int = 1,
        nat_blocks: int = 1,
        backbone_name: str = "resnet18",
    ):
        super().__init__()

        backbone_feat_dim = lstm_hidden * (2 if bidirectional else 1)
        self.backbone_feat_dim = backbone_feat_dim
        self.feat_dim_extra = feat_dim_extra

        self.backbone = ResNet18LSTM(
            num_classes=backbone_feat_dim,
            feat_dim=512,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            use_nat=use_nat,
            nat_kernel=nat_kernel,
            nat_dilation=nat_dilation,
            nat_blocks=nat_blocks,
            backbone_name=backbone_name,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_feat_dim + feat_dim_extra, 1),
        )

    def forward(self, clip, feats_seq):
        # clip:      [B, T, C, H, W]
        # feats_seq: [B, T, F]
        backbone_vec = self.backbone(clip)  # [B, D]
        feats_agg = feats_seq.mean(dim=1)  # [B, F]
        x = torch.cat([backbone_vec, feats_agg], dim=1)  # [B, D+F]
        logits = self.classifier(x).squeeze(-1)  # [B]
        return logits


# ------------------------------------------------------------------
# Load checkpoint + build model
# ------------------------------------------------------------------
def load_model(device, use_best_loss: bool = False):
    ckpt_path = BEST_LOSS_PATH if use_best_loss else BEST_ACC_PATH
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint from: {ckpt_path}")
    seq_len = ckpt.get("seq_len", 50)
    C = ckpt.get("C", 3)
    H = ckpt.get("H", 224)
    W = ckpt.get("W", 224)
    feat_dim = ckpt.get("feat_dim", 4)
    print(f"seq_len={seq_len}, C={C}, H={H}, W={W}, feat_dim={feat_dim}")

    model = VideoFeatResNet18LSTM(
        feat_dim_extra=feat_dim,
        lstm_hidden=256,
        lstm_layers=1,
        bidirectional=False,
        dropout=0.5,
        use_nat=False,
        nat_kernel=7,
        nat_dilation=1,
        nat_blocks=1,
        backbone_name="resnet18",
    ).to(device)

    try:
        model.load_state_dict(ckpt["model_state"])
    except Exception as e:
        print("ERROR loading state_dict. Make sure this matches your train.py architecture.")
        raise e

    model.eval()
    return model, seq_len, feat_dim


# ------------------------------------------------------------------
# Load a single window file (.npz)
# ------------------------------------------------------------------
def load_window(npz_path: Path):
    d = np.load(npz_path)
    clip = d["X_video"]  # [T, C, H, W], values in [0,1]
    feats = d["X_feats"]  # [T, F]
    y_raw = int(d["y"])
    y_bin = raw_to_binary(y_raw)

    return clip, feats, y_raw, y_bin


# ------------------------------------------------------------------
# Prepare tensors for the model
# ------------------------------------------------------------------
def prepare_tensors(clip_np, feats_np, seq_len, device):
    # clip_np: [T, C, H, W], feats_np: [T, F]
    T, C, H, W = clip_np.shape
    F = feats_np.shape[1]

    # Pad/trim to seq_len (same logic as train.py)
    if T > seq_len:
        clip_np = clip_np[:seq_len]
        feats_np = feats_np[:seq_len]
    elif T < seq_len:
        pad_len = seq_len - T
        clip_pad = np.zeros((pad_len, C, H, W), dtype=np.float32)
        feats_pad = np.zeros((pad_len, F), dtype=np.float32)
        clip_np = np.concatenate([clip_np, clip_pad], axis=0)
        feats_np = np.concatenate([feats_np, feats_pad], axis=0)

    x = clip_np.astype(np.float32)  # [T, C, H, W]
    f = feats_np.astype(np.float32)  # [T, F]

    x_t = torch.from_numpy(x)  # [T, C, H, W]
    f_t = torch.from_numpy(f)  # [T, F]

    # ImageNet normalization (broadcasted over T)
    x_t = (x_t - IMAGENET_MEAN) / IMAGENET_STD

    # Add batch dimension
    x_t = x_t.unsqueeze(0).to(device)  # [1, T, C, H, W]
    f_t = f_t.unsqueeze(0).to(device)  # [1, T, F]

    return x_t, f_t


def center_figure_on_screen(fig):
    """
    Center a Matplotlib figure window on the primary screen without changing size.
    Uses the figure's inch size * dpi to estimate window size.
    """
    try:
        manager = plt.get_current_fig_manager()

        # TkAgg backend
        if hasattr(manager, "window"):
            win = manager.window
            try:
                win.update_idletasks()
            except Exception:
                pass

            # Estimate window size from figure size
            dpi = fig.get_dpi()
            w = int(fig.get_figwidth() * dpi)
            h = int(fig.get_figheight() * dpi)

            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()

            x = max(0, (sw - w) // 2)
            y = max(0, (sh - h) // 2)

            # Only position, don't force a new size
            win.geometry(f"+{x}+{y}")

        # Qt backend
        elif hasattr(manager, "window") and hasattr(manager.window, "move"):
            win = manager.window
            screen = win.screen()
            rect = screen.availableGeometry()

            dpi = fig.get_dpi()
            w = int(fig.get_figwidth() * dpi)
            h = int(fig.get_figheight() * dpi)

            x = rect.x() + (rect.width() - w) // 2
            y = rect.y() + (rect.height() - h) // 2
            win.move(x, y)

    except Exception as e:
        print(f"[center_figure_on_screen] Warning: could not center figure: {e}")


# ------------------------------------------------------------------
# Build animation
# ------------------------------------------------------------------
def make_animation(
    clip_np,
    feats_np,
    prob_impaired,
    pred_label,
    gt_label,
    fps=5,
    save_path=None,
    show=True,
):
    """
    Clean UI: video left, EAR/MAR top-right, Blur/Light mid-right, Risk bottom.
    Fully patched for:
      - Matplotlib scalar xdata errors
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # -----------------------------
    # Unpack
    # -----------------------------
    T, C, H, W = clip_np.shape
    frames = np.transpose(clip_np, (0, 2, 3, 1))
    frames = np.clip(frames, 0.0, 1.0)

    EAR = feats_np[:, 0]
    MAR = feats_np[:, 1]
    BlurVar = feats_np[:, 2]
    Illum = feats_np[:, 3]
    t_idx = np.arange(T)

    # -----------------------------
    # Figure Layout
    # -----------------------------
    fig = plt.figure(figsize=(8, 5), dpi=120)
    center_figure_on_screen(fig)

    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.8, 1],
        height_ratios=[2.5, 2.2, 1],  # more space for video, but risk always visible
    )

    # -----------------------------
    # VIDEO PANEL
    # -----------------------------
    # Video only spans the top 2 rows now
    ax_frame = fig.add_subplot(gs[0:2, 0])
    im = ax_frame.imshow(frames[0], animated=True)
    ax_frame.set_aspect("equal", adjustable="box")
    ax_frame.axis("off")

    correct = pred_label == gt_label
    tint_color = (0, 1, 0, 0.30) if correct else (1, 0, 0, 0.30)
    overlay = ax_frame.imshow(
        np.ones_like(frames[0]) * np.array(tint_color[:3]).reshape(1, 1, 3),
        alpha=tint_color[3],
        animated=True,
    )

    text_info = ax_frame.text(
        0.02,
        0.98,
        f"Frame 0/{T-1}\nGT: {gt_label}  Pred: {pred_label}",
        transform=ax_frame.transAxes,
        fontsize=12,
        color="white",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        animated=True,
    )

    # -----------------------------
    # EAR/MAR
    # -----------------------------
    ax_earmar = fig.add_subplot(gs[0, 1])
    ax_earmar.set_title("EAR/MAR", pad=12)

    ax_earmar.plot(t_idx, EAR, label="EAR", color="tab:blue")
    ax_earmar.plot(t_idx, MAR, label="MAR", color="orange")
    ax_earmar.legend(fontsize=8)
    ax_earmar.grid(True, alpha=0.3)

    ax_earmar.axhline(0.5, linestyle="--", color="purple", alpha=0.5)
    ax_earmar.axhline(0.18, linestyle="--", color="pink", alpha=0.4)

    vline_earmar = ax_earmar.axvline(0, color="k", linestyle="--", animated=True)

    # -----------------------------
    # BLUR / LIGHT
    # -----------------------------
    ax_blurillum = fig.add_subplot(gs[1, 1])
    ax_blurillum.set_title("Blur/Light", pad=12)

    ax_blurillum.plot(t_idx, BlurVar, label="Blur", color="limegreen")
    ax_blurillum.plot(t_idx, Illum, label="Light", color="red")
    ax_blurillum.legend(fontsize=8)
    ax_blurillum.grid(True, alpha=0.3)

    vline_blurillum = ax_blurillum.axvline(0, color="k", linestyle="--", animated=True)

    # -----------------------------
    # RISK LINE
    # -----------------------------
    ax_risk = fig.add_subplot(gs[2, :])
    ax_risk.set_title("Fatigue Risk Timeline", pad=12)

    ax_risk.plot(t_idx, np.full(T, prob_impaired), color="red", linewidth=2)
    ax_risk.set_ylim(0, 1)
    ax_risk.grid(True, alpha=0.3)

    vline_risk = ax_risk.axvline(0, color="k", linestyle="--", animated=True)

    # -----------------------------
    # Force initial layout so nothing is cut off
    # -----------------------------
    fig.canvas.draw()
    plt.pause(0.01)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.97])
    fig.canvas.draw()

    # -----------------------------
    # Animation Update
    # -----------------------------
    def update(i):
        im.set_data(frames[i])

        # MUST use sequences to avoid xdata error
        vline_earmar.set_xdata([i])
        vline_blurillum.set_xdata([i])
        vline_risk.set_xdata([i])

        text_info.set_text(f"Frame {i}/{T-1}\nGT: {gt_label}  Pred: {pred_label}")

        return [im, overlay, text_info, vline_earmar, vline_blurillum, vline_risk]

    # -----------------------------
    # Animation
    # -----------------------------
    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=1000 / fps,
        blit=True,
    )

    if save_path:
        anim.save(save_path, fps=fps)

    if show:
        plt.show()


# ------------------------------------------------------------------
# Main / CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Animated visual demo for a single NPZ window."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a specific .npz window. If omitted, use first test window.",
    )
    parser.add_argument(
        "--use-best-loss",
        action="store_true",
        help="Use best-loss checkpoint instead of best-acc.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for the animation.",
    )
    parser.add_argument(
        "--save-mp4",
        type=str,
        default=None,
        help="Optional path to save animation as MP4 (requires ffmpeg).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display window (useful if only saving).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Which split to sample windows from (default: test).",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Index of window within the chosen split (0-based). "
            "If omitted, a random window is selected."
        ),
    )

    parser.add_argument(
        "--find-misclassified",
        type=int,
        default=0,
        help=(
            "If > 0, scan up to this many windows from the chosen split, "
            "report misclassified ones, and use the first misclassified "
            "window (if any) for the demo."
        ),
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, seq_len, feat_dim = load_model(device, use_best_loss=args.use_best_loss)

    # Pick file
    rng = np.random.default_rng()

    # If user explicitly gave a file, that overrides everything
    if args.file is not None:
        npz_path = Path(args.file)
        if not npz_path.exists():
            raise FileNotFoundError(f"Given --file does not exist: {npz_path}")
        print(f"Using file from --file:\n  {npz_path}")
    else:
        # Use split directory
        split_dir = WINDOWS_ROOT / args.split
        if not split_dir.exists():
            raise RuntimeError(f"Split directory not found: {split_dir}")

        all_files = sorted(split_dir.rglob("*.npz"))
        if not all_files:
            raise RuntimeError(f"No .npz files found in {split_dir}")

        print(f"Found {len(all_files)} windows in split '{args.split}'.")

        # Optional: search for misclassified examples
        misclassified = []
        if args.find_misclassified > 0:
            max_to_scan = min(args.find_misclassified, len(all_files))
            print(
                f"\nScanning up to {max_to_scan} windows for misclassified examples..."
            )
            for i, path in enumerate(all_files[:max_to_scan]):
                clip_np_tmp, feats_np_tmp, y_raw_tmp, y_bin_tmp = load_window(path)
                x_tmp, f_tmp = prepare_tensors(
                    clip_np_tmp, feats_np_tmp, seq_len, device
                )
                with torch.no_grad():
                    logits_tmp = model(x_tmp, f_tmp)
                    prob_tmp = torch.sigmoid(logits_tmp)[0].item()
                pred_tmp = 1 if prob_tmp >= 0.5 else 0

                if pred_tmp != y_bin_tmp:
                    misclassified.append((path, y_bin_tmp, pred_tmp, prob_tmp))

            if misclassified:
                print("\nMisclassified windows found (within scanned subset):")
                for (path, y_gt_m, y_pred_m, p_m) in misclassified:
                    print(
                        f"  {path} | GT={y_gt_m}, Pred={y_pred_m}, P(impaired)={p_m:.3f}"
                    )
            else:
                print("No misclassified windows found in the scanned subset.")

        # Decide which window to visualize
        if args.index is not None:
            if not (0 <= args.index < len(all_files)):
                raise IndexError(
                    f"--index {args.index} is out of range for {len(all_files)} files."
                )
            npz_path = all_files[args.index]
            print(f"\nUsing window by index from split '{args.split}':")
            print(f"  index={args.index}, path={npz_path}")
        elif misclassified:
            # If we found misclassified examples, use the first one
            npz_path = misclassified[0][0]
            print(f"\nUsing FIRST misclassified window for the demo:")
            print(f"  {npz_path}")
        else:
            # Fallback: random window from the split
            rand_idx = rng.integers(len(all_files))
            npz_path = all_files[rand_idx]
            print(
                f"\nNo --file and no usable misclassified window; using random window:"
            )
            print(f"  index={rand_idx}, path={npz_path}")

    # Load window
    clip_np, feats_np, y_raw, y_bin = load_window(npz_path)

    print("\n===== DEMO WINDOW INFO =====")
    print(f"File:             {npz_path}")
    print(f"Raw label in file: {y_raw}")
    print(f"Binary GT label:   {y_bin} (0=normal, 1=impaired)")

    # Inspect feature ranges
    print("EAR min/max:", feats_np[:, 0].min(), feats_np[:, 0].max())
    print("MAR min/max:", feats_np[:, 1].min(), feats_np[:, 1].max())
    print("BlurVar min/max:", feats_np[:, 2].min(), feats_np[:, 2].max())
    print("Illum min/max:", feats_np[:, 3].min(), feats_np[:, 3].max())

    # Prepare tensors & run model
    x_t, f_t = prepare_tensors(clip_np, feats_np, seq_len, device)
    with torch.no_grad():
        logits = model(x_t, f_t)  # [1]
        prob_impaired = torch.sigmoid(logits)[0].item()

    pred_label = 1 if prob_impaired >= 0.5 else 0

    print(f"P(impaired):       {prob_impaired:.4f}")
    print("Threshold:         0.50")
    print(
        f"Predicted class:   {pred_label} ({'IMPAIRED' if pred_label == 1 else 'NORMAL'})"
    )
    print("============================\n")

    # For the animation, we want the *original* [0,1] frames, not normalized.
    make_animation(
        clip_np=clip_np,
        feats_np=feats_np,
        prob_impaired=prob_impaired,
        pred_label=pred_label,
        gt_label=y_bin,
        fps=args.fps,
        save_path=args.save_mp4,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
