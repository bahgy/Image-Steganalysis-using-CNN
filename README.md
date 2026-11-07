# Image Steganalysis using CNN & GANs

A simple CNN-based image steganalysis project that classifies images into:

- Cover: An image without the tested hidden-data pattern.
- Stego: An image containing the tested LSB modification pattern.

## Model

The project uses a Convolutional Neural Network (CNN) implemented with PyTorch.

## Prediction

Run:

```bash
python predict.py path/to/image.png
The pretrained checkpoint is stored in `checkpoints/`.

## Dataset

Training and testing datasets are not included in this repository.

## Important Limitations

The model was trained and evaluated on synthetic images. It is not guaranteed to work reliably on real-world images.

This project detects a specific LSB pattern. It is not an AI-generated image detector.
