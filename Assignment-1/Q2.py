# Written by Tushar Chandra - 2021211

import cv2
import numpy as np
from PIL import Image

# Loading the image
image = cv2.imread('tank.tiff', cv2.IMREAD_GRAYSCALE)

# Converting image to NumPy array
arr = np.array(image, dtype=float)

#Convolution Function
def convolve(image, kernel):
    length, breadth = image.shape
    kLength, kBreadth = kernel.shape
    pad_height, pad_width = kLength // 2, kBreadth // 2
    ans = np.zeros_like(image)

    for i in range(pad_height, length - pad_height):
        for j in range(pad_width, breadth - pad_width):
            roi = image[i - pad_height : i + pad_height + 1, j - pad_width : j + pad_width + 1]
            ans[i, j] = np.sum(roi * kernel)

    return ans

# Defing Kernels
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


roberts_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
roberts_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])


scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])


frei_chen_x = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
frei_chen_y = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])


kirsch_kernels = [
    np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # Direction 0
    np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # Direction 1
    np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # Direction 2
    np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # Direction 3
    np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # Direction 4
    np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # Direction 5
    np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # Direction 6
    np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])  # Direction 7
]

#Applying Convolution
prewitt_x_result = convolve(arr, prewitt_x)
prewitt_y_result = convolve(arr, prewitt_y)

sobel_x_result = convolve(arr, sobel_x)
sobel_y_result = convolve(arr, sobel_y)

roberts_x_result = convolve(arr, roberts_x)
roberts_y_result = convolve(arr, roberts_y)

scharr_x_result = convolve(arr, scharr_x)
scharr_y_result = convolve(arr, scharr_y)

frei_chen_x_result = convolve(arr, frei_chen_x)
frei_chen_y_result = convolve(arr, frei_chen_y)

kirsch_results = [convolve(arr, kernel) for kernel in kirsch_kernels]

# Normailizing results
prewitt_x_result = np.clip((prewitt_x_result / prewitt_x_result.max()) * 255, 0, 255)
prewitt_y_result = np.clip((prewitt_y_result / prewitt_y_result.max()) * 255, 0, 255)

sobel_x_result = np.clip((sobel_x_result / sobel_x_result.max()) * 255, 0, 255)
sobel_y_result = np.clip((sobel_y_result / sobel_y_result.max()) * 255, 0, 255)

roberts_x_result = np.clip((roberts_x_result / roberts_x_result.max()) * 255, 0, 255)
roberts_y_result = np.clip((roberts_y_result / roberts_y_result.max()) * 255, 0, 255)

scharr_x_result = np.clip((scharr_x_result / scharr_x_result.max()) * 255, 0, 255)
scharr_y_result = np.clip((scharr_y_result / scharr_y_result.max()) * 255, 0, 255)

frei_chen_x_result = np.clip((frei_chen_x_result / frei_chen_x_result.max()) * 255, 0, 255)
frei_chen_y_result = np.clip((frei_chen_y_result / frei_chen_y_result.max()) * 255, 0, 255)

kirsch_results = [np.clip((result / result.max()) * 255, 0, 255) for result in kirsch_results]

# Converting results to images
prewitt_x_image = Image.fromarray(prewitt_x_result.astype(np.uint8))
prewitt_y_image = Image.fromarray(prewitt_y_result.astype(np.uint8))

sobel_x_image = Image.fromarray(sobel_x_result.astype(np.uint8))
sobel_y_image = Image.fromarray(sobel_y_result.astype(np.uint8))

roberts_x_image = Image.fromarray(roberts_x_result.astype(np.uint8))
roberts_y_image = Image.fromarray(roberts_y_result.astype(np.uint8))

scharr_x_image = Image.fromarray(scharr_x_result.astype(np.uint8))
scharr_y_image = Image.fromarray(scharr_y_result.astype(np.uint8))

frei_chen_x_image = Image.fromarray(frei_chen_x_result.astype(np.uint8))
frei_chen_y_image = Image.fromarray(frei_chen_y_result.astype(np.uint8))

kirsch_images = [Image.fromarray(result.astype(np.uint8)) for result in kirsch_results]

# Combining the x and y results by taking the absolute value and summing them
combined_prewitt = cv2.addWeighted(prewitt_x_result, 0.5, prewitt_y_result, 0.5, 0)
combined_sobel = cv2.addWeighted(sobel_x_result, 0.5, sobel_y_result, 0.5, 0)
combined_roberts = cv2.addWeighted(roberts_x_result, 0.5, roberts_y_result, 0.5, 0)
combined_scharr = cv2.addWeighted(scharr_x_result, 0.5, scharr_y_result, 0.5, 0)
combined_frei_chen = cv2.addWeighted(frei_chen_x_result, 0.5, frei_chen_y_result, 0.5, 0)
combined_kirsch = np.max(kirsch_results, axis=0)


cv2.imshow('Combined Prewitt', combined_prewitt)
cv2.imshow('Combined Sobel', combined_sobel)
cv2.imshow('Combined Roberts', combined_roberts)
cv2.imshow('Combined Scharr', combined_scharr)
cv2.imshow('Combined Frei-Chen', combined_frei_chen)
cv2.imshow('Combined Kirsch', combined_kirsch)

cv2.waitKey(0)
cv2.destroyAllWindows()