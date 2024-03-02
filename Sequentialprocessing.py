from transform_M import *
import sys
import time

n = int(sys.argv[2])
kernel_size = [int(sys.argv[1])] * n

city = [Image(path='city_1920x1080_by_Pedro_lastra.png')] * n

def main():
    # city_guassianBlur standard sigma = 3
    # Start timer
    st = time.perf_counter()
    city_gaussian_blur = [gaussian_blur(img, size) for img, size in zip(city, kernel_size)]
    # End timer
    et = time.perf_counter()
    print(f"Performance time(Gaussian blur): {et - st}")

    # city_MeanFilter(Box blur)
    # Start timer
    st = time.perf_counter()
    city_mean_filter = [mean_filter(img, size) for img, size in zip(city, kernel_size)]
    # End timer
    et = time.perf_counter()
    print(f"Performance time(Mean filter): {et - st}")

    # city_Edge detection 
    # Start timer
    st = time.perf_counter()

    city_sobelX = [edge_detection(img, 'x') for img in city]
    city_sobelY = [edge_detection(img, 'y') for img in city]

    city_edge_detection = [combine_images(sobelX, sobelY) for sobelX, sobelY in zip(city_sobelX, city_sobelY)]
    
    # End timer
    et = time.perf_counter()
    print(f"Performance time(Edge detection): {et - st}")


if __name__ == "__main__":
    main()
