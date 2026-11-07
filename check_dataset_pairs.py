from pathlib import Path
import numpy as np
from PIL import Image

cover_dir = Path("data/raw/Cover")
stego_dir = Path("data/raw/Stego")

for i in range(50):
    cover = np.array(
        Image.open(cover_dir / f"cover_{i}.png").convert("L")
    ).astype(np.int16)

    stego = np.array(
        Image.open(stego_dir / f"stego_{i}.png").convert("L")
    ).astype(np.int16)

    diff = np.abs(cover - stego)

    print(
        f"Pair {i}: "
        f"Changed pixels = {np.count_nonzero(diff)}, "
        f"Max difference = {diff.max()}, "
        f"Mean difference = {diff.mean():.6f}"
    )
