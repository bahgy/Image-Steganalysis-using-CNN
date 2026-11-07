import torch

from src.common.image_loader import load_image_tensor
from src.steganalysis.residual import extract_residual


cover = load_image_tensor(
    "data/raw/Cover/cover_0.png",
    color_mode="L"
).unsqueeze(0)

stego = load_image_tensor(
    "data/raw/Stego/stego_0.png",
    color_mode="L"
).unsqueeze(0)


cover_residual = extract_residual(cover)
stego_residual = extract_residual(stego)


print("Original difference:")
print((cover - stego).abs().mean().item())

print("\nResidual difference:")
print((cover_residual - stego_residual).abs().mean().item())

print("\nCover residual mean:")
print(cover_residual.abs().mean().item())

print("\nStego residual mean:")
print(stego_residual.abs().mean().item())
