import torch
import random
import numpy as np

def augment_clip(clip, label):
    """
    Simple augmentations for video clips
    clip: (T, C, H, W) tensor
    Returns: augmented (T, C, H, W) tensor
    """
    # Only augment during training, not all clips
    if random.random() > 0.5:
        return clip
    
    # Random horizontal flip
    if random.random() > 0.5:
        clip = torch.flip(clip, dims=[3])
    
    # Random brightness adjustment
    if random.random() > 0.5:
        brightness = random.uniform(0.8, 1.2)
        clip = clip * brightness
        clip = torch.clamp(clip, 0, 1)
    
    # Random contrast
    if random.random() > 0.5:
        contrast = random.uniform(0.8, 1.2)
        mean = clip.mean(dim=[2, 3], keepdim=True)
        clip = (clip - mean) * contrast + mean
        clip = torch.clamp(clip, 0, 1)
    
    # Small random noise
    if random.random() > 0.7:
        noise = torch.randn_like(clip) * 0.02
        clip = clip + noise
        clip = torch.clamp(clip, 0, 1)
    
    return clip

def temporal_shift(clip, max_shift=2):
    """
    Randomly shift frames in time
    clip: (T, C, H, W)
    """
    if random.random() > 0.5:
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            clip = torch.roll(clip, shift, dims=0)
    return clip

def random_temporal_crop(clip, min_len_ratio=0.8):
    """
    Randomly crop temporal dimension and pad back
    clip: (T, C, H, W)
    """
    T = clip.shape[0]
    if random.random() > 0.5 and T > 10:
        new_len = int(T * random.uniform(min_len_ratio, 1.0))
        start = random.randint(0, T - new_len)
        cropped = clip[start:start+new_len]
        # Pad back to original length by repeating last frame
        if new_len < T:
            pad_len = T - new_len
            padding = cropped[-1:].repeat(pad_len, 1, 1, 1)
            clip = torch.cat([cropped, padding], dim=0)
        else:
            clip = cropped
    return clip
