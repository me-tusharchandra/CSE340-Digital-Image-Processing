# Written by Tushar Chandra - 2021211

import cv2
import numpy as np

# Loading the image
image = cv2.imread('ruler.512.tiff', cv2.IMREAD_GRAYSCALE)

# Defining kernels
k3x3 = np.array([[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]]) / 16.0

k5x5 = np.array([[1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1]]) / 256.0

def gaussian(img, k):
    length, breadth = img.shape
    k_size = k.shape[0]
    ans = np.zeros_like(img)
    off = k_size // 2 # offset

    # Pad the image
    img = np.pad(img, ((off, off), (off, off)), mode='constant')

    for y in range(off, length + off):
        for x in range(off, breadth + off):
            region = img[y - off:y + off + 1, x - off:x + off + 1]
            ans[y - off, x - off] = np.sum(region * k)

    return ans

def median(img, k):
    length, breadth = img.shape
    ans = np.zeros_like(img)
    off = k // 2 # offset

    # Pad the image
    img = np.pad(img, ((off, off), (off, off)), mode='constant')

    for y in range(off, length + off):
        for x in range(off, breadth + off):
            region = img[y - off:y + off + 1, x - off:x + off + 1]
            ans[y - off, x - off] = np.median(region)

    return ans

# Gaussian Filters
gaussian3x3 = gaussian(image, k3x3)
gaussian5x5 = gaussian(image, k5x5)

# Median Filters
median3x3 = median(image, 3)
median5x5 = median(image, 5)

# Saving Filtered Images
cv2.imwrite('gaussian_3x3_result.tiff', gaussian3x3)
cv2.imwrite('gaussian_5x5_result.tiff', gaussian5x5)
cv2.imwrite('median_3x3_result.tiff', median3x3)
cv2.imwrite('median_5x5_result.tiff', median5x5)
