# CSE340-Digital-Image-Processing

## Assignment 1
**Question 1**: Implement a Gaussian filter of size 3x3 and 5x5. Also implement a median filter of the same size. Apply it on the ‘ruler’ image sent for the assignment.

**Solution**: I implemented Gaussian and Median filters with 3x3 and 5x5 kernel sizes on the 'ruler' image using OpenCV and NumPy. For Gaussian filtering, I defined kernels with specified weights, padded the image, and applied convolution operations to produce two filtered images, 'gaussian3x3' and 'gaussian5x5'. Additionally, I created Median filter kernels for 3x3 and 5x5 neighborhood regions, performed the filtering, and generated 'median3x3' and 'median5x5' filtered images. Finally, I saved all four filtered images to separate files for further analysis and visualization. These filters are fundamental tools in image processing for smoothing and reducing noise, and they help in enhancing image quality.
  

**Question 2**: Sharpen the ‘tank’ image using first and second order techniques.

**Solution**: I sharpened the 'tank' image using various first and second-order image enhancement techniques. First, I loaded the image and converted it to a NumPy array for processing. Then, I defined several convolution kernels, including Prewitt, Sobel, Roberts, Scharr, Frei-Chen, and Kirsch, for both horizontal and vertical edge detection. I applied convolution operations using these kernels to obtain the corresponding edge-detected images. I normalized the results and converted them to images using the Pillow library. Additionally, I created combined images by taking the absolute value and summing the horizontal and vertical results for Prewitt, Sobel, Roberts, Scharr, Frei-Chen, and by selecting the maximum value among the Kirsch results. Finally, I displayed the sharpened images for visual inspection. These techniques are commonly used for image enhancement and edge detection in image processing applications.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Assignment 2


