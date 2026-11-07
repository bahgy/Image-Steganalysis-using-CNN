import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.common.image_loader import load_image_tensor
from src.steganalysis.residual import extract_residual


index = 0

cover_path = f"data/raw/Cover/cover_{index}.png"
stego_path = f"data/raw/Stego/stego_{index}.png"

cover = load_image_tensor(cover_path, color_mode="L")
stego = load_image_tensor(stego_path, color_mode="L")

cover = cover.unsqueeze(0)
stego = stego.unsqueeze(0)

cover_residual = extract_residual(cover)
stego_residual = extract_residual(stego)

residual_difference = (
    cover_residual - stego_residual
).abs()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(cover.squeeze(), cmap="gray")
axes[0, 0].set_title("Cover")

axes[0, 1].imshow(stego.squeeze(), cmap="gray")
axes[0, 1].set_title("Stego")

axes[0, 2].imshow(
    np.abs(cover.squeeze().numpy() - stego.squeeze().numpy()),
    cmap="hot"
)
axes[0, 2].set_title("Original Difference")

axes[1, 0].imshow(cover_residual.squeeze(), cmap="gray")
axes[1, 0].set_title("Cover Residual")

axes[1, 1].imshow(stego_residual.squeeze(), cmap="gray")
axes[1, 1].set_title("Stego Residual")

axes[1, 2].imshow(
    residual_difference.squeeze(),
    cmap="hot"
)
axes[1, 2].set_title("Residual Difference")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()

print(
    "Residual difference mean:",
    residual_difference.mean().item()
)

print(
    "Residual difference max:",
    residual_difference.max().item()
)
