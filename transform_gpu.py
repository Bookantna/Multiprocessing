import tensorflow as tf
import tensorflow_addons as tfa

def edge_detection(img):
  image = tf.image.rgb_to_grayscale(img)
  sobely_edge_kernel = tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=tf.float32)
  sobelx_edge_kernel = tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=tf.float32)
  sobely_edge_image = tf.nn.conv2d(image[tf.newaxis, ...], sobely_edge_kernel[:, :, tf.newaxis, tf.newaxis], strides=[1, 1, 1, 1], padding='SAME')
  sobelx_edge_image = tf.nn.conv2d(image[tf.newaxis, ...], sobelx_edge_kernel[:, :, tf.newaxis, tf.newaxis], strides=[1, 1, 1, 1], padding='SAME')
  edge_image = tf.sqrt(sobelx_edge_image**2 + sobely_edge_image**2)
  return tf.squeeze(edge_image)

def gaussian_blur(img, kernel_size, sigma=3.0):
  size = kernel_size**2
  gaussian = tfa.image.gaussian_filter2d(img, filter_shape=size, sigma=sigma)
  return gaussian

def mean_filter(img, kernel_size):
  size = kernel_size**2
  mean = tfa.image.mean_filter2d(img, filter_shape=size)
  return mean