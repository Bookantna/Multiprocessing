import numpy as np
import os
import png
from numba import float32
import numba
from numba.experimental import jitclass

spec = [
    ('input_path', numba.types.string),
    ('output_path', numba.types.string),
    ('x_pixels', float32),
    ('y_pixels', float32),
    ('num_channels', float32),
    ('array', float32[:, :, :]),
    ('path',numba.types.string)
]

@jitclass(spec)
class Image:
    def __init__(self, x_pixels=0, y_pixels=0, num_channels=0, path=''):
        self.input_path = 'input/'
        self.output_path = 'output/'

        if x_pixels and y_pixels and num_channels:
            self.x_pixels = x_pixels
            self.y_pixels = y_pixels
            self.num_channels = num_channels
            self.array = np.zeros((x_pixels, y_pixels, num_channels))

        elif path:
            # Use Unicode string for filename in jitclass.Image
            self.array = self.read_image(filename=str(path))
            self.x_pixels, self.y_pixels, self.num_channels = self.array.shape
        else:
            raise ValueError("Your path is not exist or missing dimension value")
        
    def read_image(self, filename, gamma=2.2):
        im = png.Reader(self.input_path + filename).asFloat()
        resized_image = np.vstack(list(im[2]))
        resized_image.resize(im[1], im[0], 3)
        resized_image = resized_image ** gamma
        return resized_image
    def write_image(self, output_file_name, gamma=2.2):
        im = np.clip(self.array, 0, 1)
        y, x = self.array.shape[0], self.array.shape[1]
        im = im.reshape(y, x*3)
        writer = png.Writer(x, y)
        with open(self.output_path + output_file_name, 'wb') as f:
            writer.write(f, 255*(im**(1/gamma)))

        self.array.resize(y, x, 3)
