import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.common.image_loader import load_image_tensor
from src.steganalysis.split import split_dataset
from src.steganalysis.model import SteganalysisCNN


# 1. Dataset wrapper
class ImageSamplesDataset(Dataset):

    def __init__(self, samples, color_mode="RGB"):
        self.samples = samples
        self.color_mode = color_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = load_image_tensor(
            image_path,
            image_size=None,
            color_mode=self.color_mode
        )

        return image, label


# 2. Training function
def train_model(epochs=20, batch_size=8, learning_rate=0.001):

    torch.manual_seed(42)
    torch.set_num_threads(2)

    device = torch.device("cpu")

    # Split data
    train_samples, test_samples = split_dataset()

    # Create datasets
    train_dataset = ImageSamplesDataset(train_samples, color_mode="L")
    test_dataset = ImageSamplesDataset(test_samples, color_mode="L")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Create model
    model = SteganalysisCNN(in_channels=1).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    print("Starting training...")
    print("Device:", device)
    print("Training images:", len(train_dataset))
    print("Testing images:", len(test_dataset))

    # 3. Training loop
    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        average_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    # 4. Evaluate model
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * correct / total

    print("\nTraining completed!")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # 5. Save trained model
    model_path = Path("models/steganalysis_cnn.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)

    print("Model saved to:", model_path)


if __name__ == "__main__":
    train_model()
