import os
from PIL import Image
import numpy as np
import torch


DATA_DIR = "./datanew"   # ← hardcoded as requested


def compute_mean_std(data_dir):
    """
    Compute per-channel mean and std from raw images.
    No resizing / cropping / transforms applied.
    """

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    image_paths = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in valid_exts:
                image_paths.append(os.path.join(root, fname))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")

    print(f"Found {len(image_paths)} images. Computing mean/std...")

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_pixels = 0

    for idx, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")

        # Convert to [0,1]
        image = np.array(image, dtype=np.float32) / 255.0  # H x W x 3

        # Convert to tensor: 3 x H x W
        image = torch.from_numpy(image).permute(2, 0, 1)

        pixels = image.shape[1] * image.shape[2]

        channel_sum += image.sum(dim=(1, 2))
        channel_sum_sq += (image ** 2).sum(dim=(1, 2))
        total_pixels += pixels

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images...")

    mean = channel_sum / total_pixels
    std = torch.sqrt((channel_sum_sq / total_pixels) - (mean ** 2))

    # Round to 4 decimal places (REQUIRED)
    mean = [round(x.item(), 4) for x in mean]
    std = [round(x.item(), 4) for x in std]

    return mean, std


if __name__ == "__main__":
    mean, std = compute_mean_std(DATA_DIR)

    print("\n=== FINAL RESULTS (copy into train.py) ===")
    print(f"mean = {mean}")
    print(f"std  = {std}")