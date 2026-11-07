import sys
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.steganalysis.residual_model import ResidualSteganalysisCNN

DEVICE = torch.device("cpu")
MODEL_PATH = "checkpoints/residual_lsb_hard_100pct.pth"

def predict(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("L")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    image = transform(image)
    image = torch.round(image * 255).to(torch.int16)
    image = torch.remainder(image, 2).float()
    image = image.unsqueeze(0)

    model = ResidualSteganalysisCNN(in_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted = torch.argmax(probabilities).item()

    label = "Stego" if predicted == 1 else "Cover"
    confidence = probabilities[predicted].item() * 100

    print(f"Result: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print("Warning: This model was trained on synthetic data and is not reliable for arbitrary real-world images.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py image.png")
        sys.exit(1)

    predict(sys.argv[1])
