"""
Python Image Manipulation Empty Template by Kylie Ying (modified from MIT 6.865)

YouTube Kylie Ying: https://www.youtube.com/ycubed 
Twitch KylieYing: https://www.twitch.tv/kylieying 
Twitter @kylieyying: https://twitter.com/kylieyying 
Instagram @kylieyying: https://www.instagram.com/kylieyying/ 
Website: https://www.kylieying.com
Github: https://www.github.com/kying18 
Programmer Beast Mode Spotify playlist: https://open.spotify.com/playlist/4Akns5EUb3gzmlXIdsJkPs?si=qGc4ubKRRYmPHAJAIrCxVQ 
"""

from image import Image
import numpy as np
import concurrent.futures as cf
from multiprocessing import cpu_count



def brighten(image, factor):
    # when we brighten, we just want to make each channel higher by some amount 
    # factor is a value > 0, how much you want to brighten the image by (< 1 = darken, > 1 = brighten)
    # Assign value to x_pixels, y_pixels and num_channels
    x_pixels, y_pixels, num_channels = image.array.shape
    new_image = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  

    new_image.array = image.array * factor
    return new_image

def adjust_contrast(image, factor, mid=0.5):
    # adjust the contrast by increasing the difference from the user-defined midpoint by factor amount
    # Assign values to x_pixels, y_pixels and num_channels
    x_pixels, y_pixels, num_channels = image.array.shape
    # Create new image
    new_image = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  

    new_image.array = (image.array - mid) * factor + mid

    return new_image 

def combine_images(image1, image2):
    # combine two images using the squared sum of squares: value = sqrt(value_1**2, value_2**2)

    # Assign values to x_pixels, y_pixels and num_channels
    x_pixels, y_pixels, num_channels = image1.array.shape
    # Create new image
    new_image = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                # Use pythagoras theorem to find the combined pixel between Xsobel and Ysobel
                new_image.array[x, y, c] = ((image1.array[x, y, c]**2) + (image2.array[x, y, c]**2))**0.5
    return new_image

def parse_image(image, func, kernel):
    x_all, y_all, num_all = image.array.shape
    # Create new image
    new_image = Image(x_pixels=x_all, y_pixels=y_all, num_channels=num_all)
    cores_num = 9
    #psutil.cpu_count(logical=False)
    #  y_Percore
    y_pc = y_all // cores_num

    # x_remain, y_remain, num_remain
    y_rem = y_all % cores_num

    parse_image = [Image(x_pixels=x_all, y_pixels=y_pc, num_channels=num_all) for _ in range(cores_num)]

    # store image to parsed blank image
    for core in range(cores_num):
        for i in range(x_all):
            for j in range(y_pc):
                for k in range(num_all):
                    parse_image[core].array[i, j, k] = image.array[i , j + core * y_pc, k]

    with cf.ProcessPoolExecutor() as executor:
        filtered_images = list(executor.map(func, parse_image, [kernel] * cores_num))

    for core in range(cores_num):
        for i in range(x_all):
            for j in range(y_pc):
                for k in range(num_all):
                    new_image.array[i , j + core * y_pc, k] = filtered_images[core].array[i, j, k]

    return new_image


def mean_filter(image, kernel_size):
    # Assign values to x_pixels, y_pixels and num_channels
    x_pixels, y_pixels, num_channels = image.array.shape
    # Create neighbor range to determine the range of kernel
    neighbor_range = kernel_size // 2
    # Create new image
    new_image = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)
    
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0, x-neighbor_range), min(x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0, y-neighbor_range), min(y_pixels-1, y+neighbor_range)+1):
                        # sum value in kernel
                        total += image.array[x_i, y_i , c]
                new_image.array[x, y, c] = total / (kernel_size **2)
    return new_image

def edge_detection(image, kernel):
    # the kernel should be a 2D array that represents the kernel we'll use!
    # for the sake of simiplicity of this implementation, let's assume that the kernel is SQUARE
    # for example the sobel x kernel (detecting horizontal edges) is as follows:
    # [1 0 -1]
    # [2 0 -2]
    # [1 0 -1]

    # Select kernel
    if kernel.lower() == 'y':
        kernel = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]])
    elif kernel.lower() == 'x':
        kernel = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]])
    # Assign values to x_pixels, y_pixels and num_channels
    x_pixels, y_pixels, num_channels = image.array.shape
    # Create new image
    new_image = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  

    # Create neighbor range to determine the range of kernel
    kernel_size = kernel.shape[0]
    neighbor_range = kernel_size // 2

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                # Loop within kernel
                for x_i in range(max(0, x-neighbor_range), min(x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0, y-neighbor_range), min(y_pixels-1, y+neighbor_range)+1):
                        # Position of each pixel in kernel
                        x_k = x_i + neighbor_range - x
                        y_k = y_i + neighbor_range - y
                        kernel_val = kernel[x_k, y_k]
                        # Convolution
                        total += image.array[x_i, y_i, c] * kernel_val
                # black and white only to add colors include c below in brackets
                new_image.array[x, y]  = total
    return new_image

def gaussian_blur(image, kernel_size, sigma=3.0):
    # Assign values to x_pixels, y_pixels and num_channels
    x_pixels, y_pixels, num_channels = image.array.shape
    # Create new image
    new_image = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  

    # Create Kernel by gaussian blur formula
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (kernel_size-1)/2)**2 + (y - (kernel_size-1)/2)**2) / (2*sigma**2)),
        (kernel_size, kernel_size)
    )
    # Create neighbor range to determine the range of kernel
    neighbor_range = kernel_size//2
    # Divide the kernel with sum of kernel
    kernel = kernel / np.sum(kernel)
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                # set region equal to 0
                region = 0
                for x_i in range(max(0, x-neighbor_range), min(x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0, y-neighbor_range), min(y_pixels-1, y+neighbor_range)+1):
                        # Position of each pixel in kernel
                        x_k = x_i + neighbor_range - x
                        y_k = y_i + neighbor_range - y
                        kernel_val = kernel[x_k, y_k]
                        # Convolution
                        region += image.array[x_i, y_i, c] * kernel_val
                new_image.array[x, y, c] = region
    return new_image
        ''' def median_filter_custom_color(image, kernel_size):
    pad_width = kernel_size // 2
    padded_image = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
    
    new_image = np.zeros_like(image)
    
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            for k in range(new_image.shape[2]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size, k]
                median_value = np.median(region)
                new_image[i, j, k] = median_value
                
    return new_image'''





