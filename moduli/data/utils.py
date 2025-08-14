import numpy as np
from PIL import Image
from typing import List, Tuple

def normalize_dataset(image_paths: List[str]) -> Tuple[List[float], List[float]]:
    """
    Compute mean and std for a set of images.
    
    Args:
        image_paths (List[str]): list of paths to images.

    Returns:
        Tuple[List[float], List[float]]: mean and std of all images in the paths' list.
    """
    sum_r, sum_g, sum_b = 0, 0, 0
    sq_sum_r, sq_sum_g, sq_sum_b = 0, 0, 0
    num_pixels = 0
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image_np = np.array(image, dtype=np.float32) / 255
        h, w, _ = image_np.shape
        num_pixels += h * w
    
        sum_r += np.sum(image_np[:, :, 0])
        sum_g += np.sum(image_np[:, :, 1])
        sum_b += np.sum(image_np[:, :, 2])
    
        sq_sum_r += np.sum(image_np[:, :, 0]**2)
        sq_sum_g += np.sum(image_np[:, :, 1]**2)
        sq_sum_b += np.sum(image_np[:, :, 2]**2)

    mean = [sum_r / num_pixels, sum_g / num_pixels, sum_b / num_pixels]
    std = [
        np.sqrt(sq_sum_r / num_pixels - mean[0]**2),
        np.sqrt(sq_sum_g / num_pixels - mean[1]**2),
        np.sqrt(sq_sum_b / num_pixels - mean[2]**2)
    ]
    print(f"Computed mean: {mean}\nComputed std: {std}")
    return mean, std