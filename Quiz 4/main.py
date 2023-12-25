# Written by Tushar Chandra - 2021211

# Question: Implement a bilateral filter with separable 1D kernels as discussed in class. i.e. implement a bilateral filter row-wise first and get the results. Then apply bilateral filter column-wise on the obtained results to get the final output. 
# The test image in both bmp and pgm is attached. 
# During demo you need to execute all the steps to your TA. 

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, sigma):
    return np.exp(-(x ** 2) / (2 * sigma ** 2))

def bilateralFilter(image, sigmaSpatial, intensity):
    radius = int(3 * sigmaSpatial)
    spatialKernel = gaussian(np.arange(-radius, radius + 1), sigmaSpatial)

    def bilateralFilter(signal, spatialKernel, intensity):
        
        paddedSignal = np.pad(signal, radius, mode='reflect')
        filteredSignal = np.zeros_like(signal)

        for i in range(len(signal)):
            localRegion = paddedSignal[i:i + 2 * radius + 1]
            intensityWeights = gaussian(localRegion - localRegion[radius], intensity)
            weights = spatialKernel * intensityWeights
            weights /= weights.sum()
            filteredSignal[i] = (weights * localRegion).sum()

        return filteredSignal

    rowFiltered = np.zeros_like(image)
    for i in range(image.shape[0]):
        rowFiltered[i] = bilateralFilter(image[i], spatialKernel, intensity)

    columnFiltered = np.zeros_like(image)
    for i in range(image.shape[1]):
        columnFiltered[:, i] = bilateralFilter(rowFiltered[:, i], spatialKernel, intensity)

    return columnFiltered

# Load the image
imgPath = 'cameraman.bmp'
# change the format from .bmp to .pgm, rest of the code will remain same.

img = np.array(Image.open(imgPath))

# Values for sigma_spatial and Intensity
sigmaSpatialValues = [2, 3, 4, 5]
intensityValues = [30, 40, 50, 60]

# Creating subplots
fig, axes = plt.subplots(len(sigmaSpatialValues), len(intensityValues) + 1, figsize=(15, 10))

for i, sigma_spatial in enumerate(sigmaSpatialValues):
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')
    for j, intensity in enumerate(intensityValues):
        finalImage = bilateralFilter(img, sigma_spatial, intensity)
        axes[i, j + 1].imshow(finalImage, cmap='gray')
        axes[i, j + 1].set_title(f'Sigma_spatial={sigma_spatial}, Intensity={intensity}')
        axes[i, j + 1].axis('off')

plt.tight_layout()
plt.show()