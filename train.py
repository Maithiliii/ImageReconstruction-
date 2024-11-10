import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# SSIM + L1 Loss
def combined_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.5 * ssim_loss + 0.5 * l1_loss

# Function to preprocess and add noise to the image
def preprocess_and_add_noise(image_path, img_size=(128, 128), noise_factor=0.1):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, img_size)
    normalized_image = resized_image / 255.0
    noisy_image = normalized_image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=normalized_image.shape)
    noisy_image = np.clip(noisy_image, 0., 1.)
    return normalized_image, noisy_image

# Function to load and preprocess the dataset
def load_dataset(dataset_path, img_size=(128, 128), noise_factor=0.1):
    clean_images, noisy_images = [], []
    for filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, filename)
        if image_path.endswith('.jpg') or image_path.endswith('.png'):
            clean_image, noisy_image = preprocess_and_add_noise(image_path, img_size, noise_factor)
            clean_images.append(clean_image)
            noisy_images.append(noisy_image)
    clean_images = np.array(clean_images).reshape(-1, img_size[0], img_size[1], 1)
    noisy_images = np.array(noisy_images).reshape(-1, img_size[0], img_size[1], 1)
    return noisy_images, clean_images

# Load the dataset
dataset_path = r"C:\Users\HP\Downloads\archive (2)\brain_tumor_dataset"  # Set your dataset path
noisy_images, clean_images = load_dataset(dataset_path)

# Build the Denoising Autoencoder Model with Skip Connections
input_img = Input(shape=(128, 128, 1))

# Encoder
x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x2 = MaxPooling2D((2, 2), padding='same')(x1)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
x3 = MaxPooling2D((2, 2), padding='same')(x2)
x3 = Conv2D(256, (3, 3), activation='relu', padding='same')(x3)
encoded = MaxPooling2D((2, 2), padding='same')(x3)

# Decoder with Skip Connections
x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = concatenate([x, x3])  # Skip connection from encoder layer
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = concatenate([x, x2])  # Skip connection
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = concatenate([x, x1])  # Skip connection
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Compile the model with SSIM + L1 loss
dae = Model(input_img, decoded)
dae.compile(optimizer=Adam(learning_rate=0.0001), loss=combined_loss)

# Train the DAE model
history = dae.fit(noisy_images, clean_images, epochs=100, batch_size=16, shuffle=True, validation_split=0.2)

# Save the trained DAE model
dae.save("denoising_autoencoder_model.h5")
