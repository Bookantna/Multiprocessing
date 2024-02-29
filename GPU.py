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
    
    st = time.perf_counter()
    city_gaussian_blur = [gaussian_blur(img, size) for img, size in zip(city, kernel_size)]
    et = time.perf_counter()
    print(f"Performance time(Gaussian blur): {et - st}")

    # city_MeanFilter(Box blur)
    st = time.perf_counter()
    city_mean_filter = [mean_filter(img, size) for img, size in zip(city, kernel_size)]
    et = time.perf_counter()
    print(f"Performance time(Mean filter): {et - st}")

    # city_Edge detection 
    st = time.perf_counter()

    city_edge = [edge_detection(img) for img in city]
    
    et = time.perf_counter()
    print(f"Performance time(Edge detection): {et - st}")




if __name__ == "__main__":
    main()
