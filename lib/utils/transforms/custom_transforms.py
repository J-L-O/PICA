import PIL
import cv2
import numpy as np


class AdaptiveGaussianThreshold(object):
    """Convert the image to black-and-white using adaptive Gaussian thresholding.

    Args:
        block_size (int): Size of a pixel neighborhood that is used to
        calculate a threshold value for the pixel: 3, 5, 7, and so on.
    """

    def __init__(self, block_size=7, c=2):
        self.block_size = block_size
        self.c = c

    def __call__(self, sample: PIL.Image) -> np.ndarray:
        img = np.array(sample)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, self.block_size, self.c)
        img = np.expand_dims(img, 0)
        img = np.repeat(img, 3, axis=0)
        img = np.transpose(img, (1, 2, 0))

        return img
