import tensorflow as tf
import tensorflow_addons as tfa
import os
from PIL import Image
import numpy as np

# Function to read and preprocess an image
def load_and_preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((1920, 1080))  # Resize the image to 1920x1080 if needed
    img_array = np.array(img) / 255.0  # Normalize pixel values to the range [0, 1]
    return img_array

# Function to apply mean filter
def mean_filter(image):
    kernel = tf.constant([[1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0]]) / 9.0
    image = tf.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis=-1)
    filtered_image = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return tf.squeeze(filtered_image)

# Function to apply Gaussian filter
def gaussian_filter(image):
    filtered_image = tfa.image.gaussian_filter2d(image, filter_shape=3, sigma=1.0)
    return filtered_image

# Function to apply edge detection
def edge_detection(image):
    edges = tfa.image.canny_edges(image, sigma=1.0)
    return tf.cast(edges, tf.float32)

# Function to save the filtered image
def save_image(filtered_image, output_path):
    filtered_image = Image.fromarray((filtered_image.numpy() * 255).astype(np.uint8))
    filtered_image.save(output_path)

# Input directory containing images
input_directory = "/path/to/your/input"

# Output directory for filtered images
output_directory = "/path/to/your/output"

# List all files in the input directory
image_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg') or f.endswith('.png')]

# Process each image
for filename in image_files:
    input_path = os.path.join(input_directory, filename)
    output_path_mean = os.path.join(output_directory, f"mean_{filename}")
    output_path_gaussian = os.path.join(output_directory, f"gaussian_{filename}")
    output_path_edge_detection = os.path.join(output_directory, f"edge_{filename}")

    # Load and preprocess the image
    img_array = load_and_preprocess_image(input_path)

    # Apply mean filter
    mean_filtered_image = mean_filter(img_array)
    save_image(mean_filtered_image, output_path_mean)

    # Apply Gaussian filter
    gaussian_filtered_image = gaussian_filter(img_array)
    save_image(gaussian_filtered_image, output_path_gaussian)

    # Apply edge detection
    edge_detected_image = edge_detection(img_array)
    save_image(edge_detected_image, output_path_edge_detection)
