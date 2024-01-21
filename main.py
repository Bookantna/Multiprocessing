import concurrent.futures as cf
import time 
from transform_M import *

def main():
    
    '''
    st = time.perf_counter()
    
    blur_lake = blur(lake,3)
    blur_lake.write_image("blur_lake.png")
    et = time.perf_counter()
    print(f"Used time Thread:{et-st}.")
    
    
    st = time.perf_counter()
    with cf.ProcessPoolExecutor() as executor:
        blur_lake2 = executor.submit(blur, lake, 3)
        result = blur_lake2.result()
        result.write_image("blur_lake2.png")
    et = time.perf_counter()
    print(f"Used time Multi:{et-st}.")
    '''
    n = 50
    lake = [Image(filename="lake.png")]*n
    kernel = [3]*n
    
    st = time.perf_counter()
    
    with cf.ProcessPoolExecutor() as executor:
        blur_lake = list(executor.map(guassian_blur,lake, kernel))
        for i, blurred_image in enumerate(blur_lake):
            executor.submit(blur_lake[i].write_image(f"blur_lakeMulti{i+1}.png"))
    
    et = time.perf_counter()

    print(f"Used time Thread:{et-st}.")

    #blur normal
    st = time.perf_counter()
    blur_lake = list(map(blur, lake, kernel))
    #blur_lake = blur(lake,3)
    for i in range(len(blur_lake)):
        blur_lake[i].write_image(f"blur_lakeN{i+1}.png")

    et = time.perf_counter()
    print(f"Used time Thread:{et-st}.")


if __name__ == "__main__":
    main()