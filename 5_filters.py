import numpy as np
import cv2

# Load the input image
input_image = cv2.imread(".\input_images\image-1.jpg", cv2.IMREAD_GRAYSCALE)


####################
# 1. Mean Filter
####################


# Define a function mean_filter that takes an image and a kernel_size as input.
def mean_filter(image, kernel_size):
    # Get the height and width of the input image.
    height, width = image.shape

    # Create an output image (meanResult) with the same dimensions and data type as the input image.
    meanResult = np.zeros_like(image, dtype=np.uint8)

    # Calculate the radius of the kernel (k) by dividing the kernel_size by 2.
    k = kernel_size // 2

    # Loop through the rows of the image (excluding the edges defined by k).
    for i in range(k, height - k):
        # Loop through the columns of the image (excluding the edges defined by k).
        for j in range(k, width - k):
            # Initialize a variable sum_val to store the sum of pixel values within the kernel.
            sum_val = 0

            # Loop through the rows and columns of the kernel centered at (i, j).
            for m in range(-k, k + 1):
                for n in range(-k, k + 1):
                    # Accumulate the pixel values within the kernel.
                    sum_val += image[i + m, j + n]

            # Compute the mean (average) value by dividing the sum by the total number of pixels in the kernel.
            meanResult[i, j] = sum_val // (kernel_size**2)

    # Return the resulting image after applying the mean filter.
    return meanResult


####################
# 2. Median Filter
####################


def median_filter(image, kernel_size):
    # Get the dimensions of the image
    height, width = image.shape

    # Create an output image with the same dimensions as the input
    medianResult = np.zeros((height, width), dtype=np.uint8)

    # Calculate the padding required based on the kernel size
    pad = kernel_size // 2

    # Iterate over each pixel in the image
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Extract the neighborhood around the current pixel
            neighborhood = image[i - pad : i + pad + 1, j - pad : j + pad + 1]

            # Calculate the median value of the neighborhood
            median_value = np.median(neighborhood)

            # Assign the median value to the output image
            medianResult[i, j] = median_value

    return medianResult


####################
# 3. Gaussian Filter
####################


# Define the Gaussian function
def gaussianFunction(x, y, sigma):
    return (1.0 / (2 * np.pi * sigma**2)) * np.exp(
        -(x**2 + y**2) / (2 * sigma**2)
    )


# Create a Gaussian kernel
def create_gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: gaussianFunction(x - (size - 1) / 2, y - (size - 1) / 2, sigma),
        (size, size),
    )
    return kernel / np.sum(kernel)  # Normalize the kernel


# Convolve the image with the Gaussian kernel
def gaussian_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)


####################
# 4. Sobel Filter
####################


def sobel_filter(image):
    # Define Sobel filter kernels for horizontal and vertical edges
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Apply convolution to the image using the Sobel kernels
    gradient_x = cv2.filter2D(image, -1, sobel_x)
    gradient_y = cv2.filter2D(image, -1, sobel_y)

    # Combine the gradient magnitudes
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Perform normalization to map values to the 0-255 range
    gradient_magnitude = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )

    return gradient_magnitude


####################
# 5. Laplacian Filter
####################

def laplacian_filter(image):
    # Define the Laplacian filter kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    # Apply convolution to the image using the Laplacian kernel
    laplacianOutput = cv2.filter2D(image, -1, kernel)

    # Perform normalization to map values to the 0-255 range
    laplacianOutput = cv2.normalize(
        laplacianOutput, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )

    return laplacianOutput


# Define kernel size
kernel_size = 5
# Define sigma (standard deviation)
sigma = 1.0


# Apply filters
mean_filtered = mean_filter(input_image, kernel_size)
median_filtered = median_filter(input_image, kernel_size)

# Create the Gaussian kernel
gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)

gaussian_filtered = gaussian_filter(input_image, gaussian_kernel)

# sobel_filtered = sobel_filter(input_image)
laplacian_filtered = laplacian_filter(input_image)

# Save the results in a single output file
output_image = np.hstack(
    (
        input_image,
        mean_filtered,
        median_filtered,
        gaussian_filtered,
        # sobel_filtered,
        laplacian_filtered,
    )
)
cv2.imwrite("output_image.jpg", output_image)

# Display and save the individual filtered images if needed
cv2.imwrite(".\output_images\mean_filtered.jpg", mean_filtered)
cv2.imwrite(".\output_images\median_filtered.jpg", median_filtered)
cv2.imwrite(".\output_images\gaussian_filtered.jpg", gaussian_filtered)
# cv2.imwrite(".\output_images\sobel_filtered.jpg", sobel_filtered)
cv2.imwrite(".\output_images\laplacian_filtered.jpg", laplacian_filtered)

# Optionally, display the output image
cv2.imshow("Output Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
