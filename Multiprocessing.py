from transform_M import *
import concurrent.futures as cf
import sys
import time

n = int(sys.argv[2])
kernel_size = [int(sys.argv[1])]*n

city = [Image(path='city_1920x1080_by_Pedro_lastra.png')]*n

def main():
    with cf.ProcessPoolExecutor() as executor:
        
        # city_guassianBlur standard sigma = 3
        st = time.perf_counter()
        city_gaussian_blur = list(executor.map(gaussian_blur, city, kernel_size))
        et = time.perf_counter()
        print(f"Performance time(Guassian blur):{et-st}")

        # city_MeanFilter(Box blur)
        st = time.perf_counter()
        city_mean_filter = list(executor.map(mean_filter, city, kernel_size))
        et = time.perf_counter()
        print(f"Performance time(Mean filter):{et-st}")

        # city_Edge dectection 
        st = time.perf_counter()

        city_sobelX = list(executor.map(edge_detection, city, ['x']*n))
        city_sobelY = list(executor.map(edge_detection, city, ['y']*n))

        city_edge_detection = list(executor.map(combine_images, city_sobelX, city_sobelY))
        
        et = time.perf_counter()
        print(f"Performance time(Edge detection):{et-st}")


if __name__ == "__main__":
    main()