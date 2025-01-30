# Image Denoising using Autoencoders

## Overview

This project implements an image denoising model using convolutional autoencoders. The model is trained to remove noise from images by learning an efficient image reconstruction technique. The dataset used is **CIFAR-10**, and the model is built using **TensorFlow/Keras**.

## Technologies Used

- Python
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib

## Dataset

The **CIFAR-10 dataset** is used, which consists of 60,000 color images (32x32 pixels) in 10 classes.

## Model Architecture

The autoencoder consists of:

- **Encoder:** Convolutional layers with ReLU activation and max-pooling layers to extract meaningful features.
- **Decoder:** Convolutional layers with upsampling layers to reconstruct the denoised image.

## Training Process

- The dataset is normalized to **[0, 1]** range.
- Gaussian noise is added to the training images.
- The autoencoder is trained using **Mean Squared Error (MSE) loss** and **Adam optimizer**.
- The model is evaluated on test images, comparing noisy vs. denoised outputs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AmIT0yAdAv/Image-Denoising-Autoencoder.git
   cd Image-Denoising-Autoencoder
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```
3. Run the training script:
   ```bash
   python autoencoder.py
   ```

## Future Improvements

- Train on higher-resolution images for better quality.
- Experiment with different loss functions (e.g., SSIM loss).
- Use **GANs** or **Transformers** for better denoising performance.


---

Feel free to contribute! 🚀

