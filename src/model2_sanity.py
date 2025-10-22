# src/model2_sanity.py
import torch
from src.models.model2_hmgru_tna import HMGRU_TNA 

def main():
    B, T = 2, 48         # batch size, frames per window
    Df = 4               # EAR/MAR/BLUR/ILLUM as in your preprocessing
    num_classes = 2

    model = HMGRU_TNA(num_classes=num_classes, feat_dim=Df)
    model.eval()

    # Random tensors shaped like your preprocessed windows (.npz)
    x_video = torch.rand(B, T, 3, 224, 224)
    x_feats = torch.rand(B, T, Df)

    with torch.no_grad():
        logits = model(x_video, x_feats)
    print("logits:", logits.shape, "->", logits)

if __name__ == "__main__":
    main()