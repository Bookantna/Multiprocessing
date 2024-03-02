from transform_gpu import *
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import sys
import time

n = int(sys.argv[2])
kernel_size = [int(sys.argv[1])]*n

def main():
    img_path = "C:/Users/User/Desktop/codes/Multiprocessing/input/city_1920x1080_by_Pedro_lastra.png"
    img_raw = tf.io.read_file(img_path)
    img = tf.io.decode_image(img_raw)

    city = [tf.image.convert_image_dtype(img, tf.float32)]*n
    # Start timer
    st = time.perf_counter()
    [gaussian_blur(img, size) for img, size in zip(city, kernel_size)]
    # End timer
    et = time.perf_counter()
    print(f"Performance time(Gaussian blur): {et - st}")

    # city_MeanFilter(Box blur)
    # Start timer
    st = time.perf_counter()
    [mean_filter(img, size) for img, size in zip(city, kernel_size)]
    # End timer
    et = time.perf_counter()
    print(f"Performance time(Mean filter): {et - st}")

    # city_Edge detection 
    # Start timer
    st = time.perf_counter()

    [edge_detection(img) for img in city]
    
    # End timer
    et = time.perf_counter()
    print(f"Performance time(Edge detection): {et - st}")




if __name__ == "__main__":
    main()
