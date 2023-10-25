# Written by: Tushar Chandra - 2021211

# Importing necessary libraries
import os
import cv2
import numpy as np

# Output directory
output_dir = 'Output1'
os.makedirs(output_dir, exist_ok=True)

# Loading the image
img = cv2.imread('barbara.bmp', cv2.IMREAD_GRAYSCALE)

# Salt and Pepper Noise function
def snp(img, noise):
    noisyImage = img.copy()
    totalPixels = img.size
    numSalt = int(totalPixels * noise)
    numPepper = int(totalPixels * noise)

    # Adding salt noise
    saltCoords = [np.random.randint(0, i - 1, numSalt) for i in img.shape]
    noisyImage[saltCoords[0], saltCoords[1]] = 255

    # Adding pepper noise
    pepperCoords = [np.random.randint(0, i - 1, numPepper) for i in img.shape]
    noisyImage[pepperCoords[0], pepperCoords[1]] = 0

    return noisyImage

# Median filter function
def median(img, winSize):
    length, breadth = img.shape
    res = np.zeros((length, breadth), dtype=np.uint8)

    # Calculate the half-window size
    halfWindow = winSize // 2

    # Iterate through each pixel in the image
    for i in range(length):
        for j in range(breadth):
            window = []

            # Iterate over a neighborhood around the current pixel
            for m in range(-halfWindow, halfWindow + 1):
                for n in range(-halfWindow, halfWindow + 1):

                    # Check if the pixel coordinates are within the image boundaries
                    if 0 <= i + m < length and 0 <= j + n < breadth:
                        window.append(img[i + m, j + n])
            # Calculate the median of the pixel values in the window
            res[i, j] = np.median(window)

    return res

# Function to compute PSNR
def PSNR(cleanImg, denoisedImg):
    mse = np.mean((cleanImg - denoisedImg) ** 2)
    if mse == 0:
        return float('inf')

    maxIntensity = 255

    psnr = 10 * np.log10((maxIntensity ** 2) / mse)
    return psnr

# Noise levels
noiseLevels = [0.05, 0.15, 0.20, 0.25]

# Window sizes
windowSizes = [3, 5, 7]

parameters = {} 
psnrsz = {} 

print()

# Looping over all noise levels
for noiseLevel in noiseLevels:
    print("--------------------")
    print(f"Noise Level: {noiseLevel * 100}%")
    noisyImage = snp(img, noiseLevel)
    cleanImage = img
    psnrsz[noiseLevel] = -1 

    # Looping over all window sizes
    for winSize in windowSizes:
        denoisedImage = median(noisyImage, winSize)
        psnr_value = PSNR(cleanImage, denoisedImage)

        if psnr_value > psnrsz[noiseLevel]:
            psnrsz[noiseLevel] = psnr_value
            parameters[noiseLevel] = winSize

        denoised_image_path = os.path.join(output_dir, f'denoised_image_{int(noiseLevel * 100)}_{winSize}x{winSize}.png')

        cv2.imwrite(denoised_image_path, denoisedImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # Display information about the denoising process
        print(f"Window Size: {winSize}x{winSize}, PSNR: {psnr_value:.2f}")

    # Save the noisy image for this noise level outside the window size loop
    noisy_image_path = os.path.join(output_dir, f'noisy_image_{int(noiseLevel * 100)}.png')
    cv2.imwrite(noisy_image_path, noisyImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Printing the best parameters
print()
print("--------------------")
print("Best Denoising Parameters:")
for noiseLevel, winSize in parameters.items():
    print(f"{int(noiseLevel * 100)}% noise: {winSize}x{winSize} (PSNR: {psnrsz[noiseLevel]:.2f})")

print()
print("--------------------")