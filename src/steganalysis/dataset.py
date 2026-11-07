
from pathlib import Path

from torch.utils.data import Dataset
from src.common.image_loader import load_image_tensor


class SteganalysisDataset(Dataset):

    def __init__(
        self,
        root_dir="data/raw",
        image_size=(64, 64),
        color_mode="RGB"
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.color_mode = color_mode

        self.classes = {
            "Cover": 0,
            "Stego": 1
        }

        self.samples = []

        supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        for class_name, label in self.classes.items():
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Dataset folder not found: {class_dir}"
                )

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in supported_extensions:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError("No supported images found in dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = load_image_tensor(
            image_path,
            image_size=self.image_size,
            color_mode=self.color_mode
        )

        return image, label
