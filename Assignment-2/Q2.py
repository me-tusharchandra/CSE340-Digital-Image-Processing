import os
import cv2
import numpy as np

def compute_psnr(img, de_noisedImg):
    mse = np.mean((img - de_noisedImg) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

def reduce_size(img, factor):
    # r is reduced
    rLength = img.shape[0] // factor
    rBreadth = img.shape[1] // factor
    rimg = np.empty((rLength, rBreadth), dtype=np.uint8)

    for i in range(rLength):
        for j in range(rBreadth):
            mean = np.mean(img[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor])
            rimg[i, j] = mean

    return rimg

def nnInterpolation(img, factor):
    # s is scaled
    length, breadth = img.shape
    sLength = length * factor
    sBreadth = breadth * factor
    sImg = np.empty((int(sLength), int(sBreadth)), dtype=np.uint8)

    for i in range(int(sLength)):
        for j in range(int(sBreadth)):
            srcI, srcJ = i / factor, j / factor
            srcI, srcJ = int(max(0, min(srcI, length - 1))), int(max(0, min(srcJ, breadth - 1)))
            sImg[i, j] = img[srcI, srcJ]

    return sImg

def bilinear_interpolation(img, factor):
    length, breadth = img.shape
    sLength = length * factor
    sBreadth = breadth * factor
    sImg = np.empty((int(sLength), int(sBreadth)), dtype=np.uint8)

    for i in range(int(sLength)):
        for j in range(int(sBreadth)):
            srcI, srcJ = i / factor, j / factor
            srcI1, srcJ1 = int(srcI), int(srcJ)
            srcI2, srcJ2 = min(srcI1 + 1, length - 1), min(srcJ1 + 1, breadth - 1)
            delI, delJ = srcI - srcI1, srcJ - srcJ1

            pixel1 = img[srcI1, srcJ1]
            pixel2 = img[srcI1, srcJ2]
            pixel3 = img[srcI2, srcJ1]
            pixel4 = img[srcI2, srcJ2]

            sImg[i, j] = int((1 - delI) * (1 - delJ) * pixel1
                                    + delI * (1 - delJ) * pixel2
                                    + (1 - delI) * delJ * pixel3
                                    + delI * delJ * pixel4)

    return sImg

def bicubicInterpolation(img, factor):
    length, breadth = img.shape
    sLength = length * factor
    sbreadth = breadth * factor
    sImg = np.empty((int(sLength), int(sbreadth)), dtype=np.uint8)

    for i in range(int(sLength)):
        for j in range(int(sbreadth)):
            srcI, srcJ = i / factor, j / factor
            srcI1, srcJ1 = int(srcI), int(srcJ)
            srcI2, srcJ2 = min(srcI1 + 1, length - 1), min(srcJ1 + 1, breadth - 1)
            delI, delJ = srcI - srcI1, srcJ - srcJ1

            pixels = []
            for x in range(-1, 3):
                for y in range(-1, 3):
                    xIndex = max(0, min(breadth - 1, srcJ1 + x))
                    yIndex = max(0, min(length - 1, srcI1 + y))
                    pixels.append(img[yIndex, xIndex])

            pixels = np.array(pixels).reshape((4, 4))

            interpolatedValues = 0
            for x in range(4):
                for y in range(4):
                    interpolatedValues += pixels[y, x] * cubic(delI - y) * cubic(delJ - x)

            sImg[i, j] = int(interpolatedValues)

    return sImg

def lanczosInterpolation(img, factor, a=3):
    length, breadth = img.shape
    sLength = length * factor
    sbreadth = breadth * factor
    sImg = np.empty((int(sLength), int(sbreadth)), dtype=np.uint8)

    for i in range(int(sLength)):
        for j in range(int(sbreadth)):
            srcI, srcJ = i / factor, j / factor
            x1, y1 = int(srcJ - a), int(srcI - a)

            interpolatedValues = 0
            norm = 0
            for y in range(2 * a + 1):
                for x in range(2 * a + 1):
                    weight = sinc((srcJ - (x1 + x)) / a) * sinc((srcI - (y1 + y)) / a)
                    interpolatedValues += weight * img[max(0, min(length - 1, y1 + y)), max(0, min(breadth - 1, x1 + x))]
                    norm += weight

            interpolatedValues /= norm
            sImg[i, j] = int(interpolatedValues)

    return sImg

def b_splineInterpolation(img, factor, a=2):
    length, breadth = img.shape
    sLength = length * factor
    sBreadth = breadth * factor
    sImg = np.empty((int(sLength), int(sBreadth)), dtype=np.uint8)

    for i in range(int(sLength)):
        for j in range(int(sBreadth)):
            srcI, srcJ = i / factor, j / factor
            x1, y1 = int(srcJ - a), int(srcI - a)

            interpolatedValues = 0
            norm = 0
            for y in range(2 * a + 1):
                for x in range(2 * a + 1):
                    weight = b_spline(srcJ - (x1 + x), a) * b_spline(srcI - (y1 + y), a)
                    interpolatedValues += weight * img[max(0, min(length - 1, y1 + y)), max(0, min(breadth - 1, x1 + x))]
                    norm += weight

            interpolatedValues /= norm
            sImg[i, j] = int(interpolatedValues)

    return sImg

def cubic(t):
    if abs(t) <= 1:
        return (1.5 * abs(t) - 2.5) * abs(t) ** 2 + 1
    else:
        return 0

def sinc(x):
    if x == 0:
        return 1
    return np.sin(np.pi * x) / (np.pi * x)

def b_spline(t, a):
    if -a <= t <= -1:
        return ((a + 2) * t ** 3 / 6) - (a * t ** 2 / 2) + (a * t / 3) + 1
    elif -1 < t <= 0:
        return ((-t ** 3 / 2) - (t ** 2) + (a + 2) / 2)
    elif 0 < t <= 1:
        return ((t ** 3 / 2) - (t ** 2) - (a * t / 2) + (a / 2))
    elif 1 < t <= a:
        return ((-a + 2) * t ** 3 / 6) - (a * t ** 2 / 2) + (a * t / 2) - (a / 6)
    else:
        return 0

if __name__ == "__main__":
    image = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
    image = image[:-1, :-1] #512x512

    output_folder = "Output2"
    os.makedirs(output_folder, exist_ok=True)

    reduced_image = reduce_size(image, factor=4)

    print()
    print("----------------------------------------")

    # Interpolation methods
    interpolation_methods = ["nearest_neighbor", "bilinear", "bicubic", "lanczos", "b_spline"]
    for method in interpolation_methods:
        if method == "nearest_neighbor":
            super_resolved_image = nnInterpolation(reduced_image, factor=4)
        elif method == "bilinear":
            super_resolved_image = bilinear_interpolation(reduced_image, factor=4)

        output_path = os.path.join(output_folder, f"super_resolved_{method}.png")
        cv2.imwrite(output_path, super_resolved_image)

        psnr = compute_psnr(image, super_resolved_image)
        print(f"PSNR for {method.capitalize()} Interpolation: {psnr:.2f} dB")

        print("----------------------------------------")

        
