from pathlib import Path
import random


def split_dataset(root_dir="data/raw", test_ratio=0.2, seed=42):
    root_dir = Path(root_dir)

    cover_dir = root_dir / "Cover"
    stego_dir = root_dir / "Stego"

    pairs = []

    for cover_path in sorted(cover_dir.glob("cover_*.png")):
        index = cover_path.stem.replace("cover_", "")
        stego_path = stego_dir / f"stego_{index}.png"

        if stego_path.exists():
            pairs.append((cover_path, stego_path))

    if not pairs:
        raise ValueError(f"No image pairs found in {root_dir}")

    random.Random(seed).shuffle(pairs)

    test_count = max(1, int(len(pairs) * test_ratio))

    test_pairs = pairs[:test_count]
    train_pairs = pairs[test_count:]

    train_samples = []
    test_samples = []

    for cover_path, stego_path in train_pairs:
        train_samples.append((cover_path, 0))
        train_samples.append((stego_path, 1))

    for cover_path, stego_path in test_pairs:
        test_samples.append((cover_path, 0))
        test_samples.append((stego_path, 1))

    return train_samples, test_samples
