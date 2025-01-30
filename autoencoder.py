import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# Load CIFAR-10 dataset (or replace with your custom dataset)
(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

# Normalize data to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add noise to images
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip values to be in the range [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# Define the autoencoder model
input_img = Input(shape=(32, 32, 3))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, x)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(x_train_noisy, x_train, epochs=10, batch_size=64, shuffle=True, validation_data=(x_test_noisy, x_test))

# Denoise sample images
predicted = autoencoder.predict(x_test_noisy[:10])

# Display results
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for i in range(10):
    axes[0, i].imshow(x_test_noisy[i])
    axes[0, i].axis('off')
    axes[1, i].imshow(predicted[i])
    axes[1, i].axis('off')
    axes[2, i].imshow(x_test[i])
    axes[2, i].axis('off')

axes[0, 0].set_title("Noisy")
axes[1, 0].set_title("Denoised")
axes[2, 0].set_title("Original")
plt.show()
