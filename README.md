cat > README.md <<'EOF'
# Image Steganalysis using CNN

A beginner-friendly image steganalysis project using PyTorch and Convolutional Neural Networks (CNN).

## Project Overview

This project aims to classify images into two categories:

- Cover: An image without the targeted hidden-data modification.
- Stego: An image containing hidden data using LSB-based techniques.

## Technologies

- Python
- PyTorch
- CNN
- NumPy
- Pillow

## Current Features

- Image preprocessing.
- LSB extraction.
- CNN model training.
- Cover and Stego classification.
- Prediction for a single uploaded image.

## Project Structure

```text
GAN-Steganalysis/
├── src/
│   └── steganalysis/
├── checkpoints/
├── predict.py
├── requirements.txt
└── README.md
