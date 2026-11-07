import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.steganalysis.train import ImageSamplesDataset
from src.steganalysis.split import split_dataset
from src.steganalysis.residual_model import ResidualSteganalysisCNN


class LSBDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Recover original pixel values, then extract LSB
        image = torch.round(image * 255).to(torch.int16)
        image = torch.remainder(image, 2).float()

        return image, label


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total


torch.manual_seed(42)

device = torch.device("cpu")

# Split paired images into training and testing
train_samples, test_samples = split_dataset(root_dir="data/hard")

train_base = ImageSamplesDataset(train_samples, color_mode="L")
test_base = ImageSamplesDataset(test_samples, color_mode="L")

train_dataset = LSBDataset(train_base)
test_dataset = LSBDataset(test_base)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

model = ResidualSteganalysisCNN(in_channels=1).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    if (epoch + 1) % 10 == 0:
        train_loss, train_accuracy = evaluate(
            model, train_loader, criterion, device
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {train_loss:.6f} "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

# Evaluate on unseen test images
test_loss, test_accuracy = evaluate(
    model, test_loader, criterion, device
)

print("\n===== FINAL RESULTS =====")
print(f"Test Loss: {test_loss:.6f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Confusion matrix
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

for actual in [0, 1]:
    for predicted in [0, 1]:
        count = sum(
            1 for a, p in zip(all_labels, all_predictions)
            if a == actual and p == predicted
        )
        print(f"Actual {actual}, Predicted {predicted}: {count}")

torch.save(
    model.state_dict(),
    "models/residual_lsb_full.pth"
)

print("\nModel saved: models/residual_lsb_full.pth")
