import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained DAE model
dae = load_model("denoising_autoencoder_model.h5", compile=False)

# Function to preprocess and add noise to the test image
def preprocess_and_add_noise(image_path, img_size=(128, 128), noise_factor=0.1):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, img_size)
    normalized_image = resized_image / 255.0
    noisy_image = normalized_image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=normalized_image.shape)
    noisy_image = np.clip(noisy_image, 0., 1.)
    return normalized_image.reshape(1, img_size[0], img_size[1], 1), noisy_image.reshape(1, img_size[0], img_size[1], 1)

# Test the model on a sample image
sample_image_path = r"C:\Users\HP\Downloads\archive (2)\brain_tumor_dataset\Y22.jpg" # Set path to your test image
clean_image, noisy_image = preprocess_and_add_noise(sample_image_path)

# Reconstruct the image using the trained model
reconstructed_image = dae.predict(noisy_image)

# Display the noisy, original, and reconstructed images
plt.figure(figsize=(15, 4))

# Display noisy image
plt.subplot(1, 3, 1)
plt.title("Noisy Image")
plt.imshow(noisy_image[0].reshape(128, 128), cmap='gray')

# Display original (clean) image
plt.subplot(1, 3, 2)
plt.title("Original Image")
plt.imshow(clean_image[0].reshape(128, 128), cmap='gray')

# Display reconstructed image
plt.subplot(1, 3, 3)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image[0].reshape(128, 128), cmap='gray')

plt.show()
