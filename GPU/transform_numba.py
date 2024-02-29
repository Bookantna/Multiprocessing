import tensorflow as tf
from PIL import Image

def combine_images(image1, image2):
    # Assuming image1 and image2 have the same shape
    with tf.device('/gpu:0'): # Use GPU device
        # Convert NumPy arrays to TensorFlow tensors
        image1 = tf.convert_to_tensor(image1.array, dtype=tf.float32)
        image2 = tf.convert_to_tensor(image2.array, dtype=tf.float32)
        # Perform element-wise square and square root operations
        squared_sum = tf.math.sqrt(tf.math.square(image1) + tf.math.square(image2))
        # Convert TensorFlow tensor back to NumPy array
        new_image = Image(array=squared_sum.numpy())
    return new_image

def canny_edge_detector(image, low_threshold, high_threshold):
    # Assuming image is a TensorFlow tensor
    with tf.device('/gpu:0'): # Use GPU device
        # Convert the image to grayscale
        image = tf.image.rgb_to_grayscale(image)
        # Apply a Gaussian blur to reduce noise
        image = tf.image.blur(image, (5, 5))
        # Compute the gradient magnitude and direction using Sobel operator
        dx, dy = tf.image.sobel_edges(image)
        magnitude = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy))
        direction = tf.math.atan2(dy, dx)
        # Perform non-maximum suppression to thin the edges
        nms = tf.image.non_max_suppression(magnitude, direction, 1.0)
        # Apply double thresholding to classify the edges
        weak_edges = tf.math.greater_equal(nms, low_threshold)
        strong_edges = tf.math.greater_equal(nms, high_threshold)
        # Perform hysteresis to connect the weak edges to the strong edges
        output = tf.image.connected_components(tf.cast(strong_edges, tf.int32))[0]
        output = tf.math.greater(output, 0)
        output = tf.math.logical_and(output, weak_edges)
        # Convert the output to a binary image
        output = tf.cast(output, tf.uint8) * 255
        # Convert TensorFlow tensor back to NumPy array
        output = output.numpy()
    return output

def gaussian_blur(image, kernel_size, sigma=3.0):
    # Assuming image is a TensorFlow tensor
    with tf.device('/gpu:0'): # Use GPU device
        # Create a kernel that follows a Gaussian distribution
        kernel = tf.exp(-tf.range(-(kernel_size // 2), kernel_size // 2 + 1, dtype=tf.float32) ** 2 / (2 * sigma ** 2))
        kernel = tf.einsum('i,j->ij', kernel, kernel)
        kernel = kernel / tf.reduce_sum(kernel)
        # Reshape the kernel to match the image dimensions
        kernel = tf.reshape(kernel, [kernel_size, kernel_size, 1, 1])
        # Perform convolution with the kernel and the image
        new_image = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
        # Convert TensorFlow tensor back to NumPy array
        new_image = new_image.numpy()
    return new_image
