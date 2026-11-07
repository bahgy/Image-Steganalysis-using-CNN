import matplotlib.pyplot as plt

from src.common.image_loader import load_image_tensor
from src.steganalysis.residual import extract_residual


image = load_image_tensor(
    "data/raw/Cover/cover_0.png",
    color_mode="L"
)

image = image.unsqueeze(0)

residual = extract_residual(image)

original = image.squeeze().numpy()
residual_image = residual.squeeze().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(original, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(residual_image, cmap="gray")
axes[1].set_title("Residual Image")
axes[1].axis("off")

plt.tight_layout()
plt.show()
