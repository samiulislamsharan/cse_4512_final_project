import cv2
import numpy as np

# Load the input image
input_image = cv2.imread("./input_images/image-1.jpg", cv2.IMREAD_GRAYSCALE)


####################
# Method to put text on the center of an image
####################


def put_text_on_center(
    image,
    text,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    color=(255, 255, 255),
    thickness=1,
):
    # Get the dimensions of the image
    height, width = 320, 320

    # Calculate the size of the text to be placed
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the position to place the text at the center
    x = int((width - text_size[0]) / 2)
    y = int((height + text_size[1]) / 2)

    # Create a copy of the input image to avoid modifying the original image
    image_with_text = image.copy()

    # Put the text on the image
    cv2.putText(
        image_with_text, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
    )

    return image_with_text


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
# This function generates a 2D matrix that represents a Gaussian kernel based on the provided sigma and size parameters, and it ensures that the kernel is properly normalized for subsequent image filtering operations.
def create_gaussian_kernel(sigma, kernel_size):
    # This lambda function defines the function to be applied to each coordinate (x, y).
    # It calculates the Gaussian value at the specified (x, y) position in the kernel matrix using the gaussianFunction.
    # The kernel matrix is of size (size, size).

    kernel = np.fromfunction(
        lambda x, y: gaussianFunction(
            x - (kernel_size - 1) / 2, y - (kernel_size - 1) / 2, sigma
        ),
        (kernel_size, kernel_size),
    )
    return kernel / np.sum(kernel)  # Normalize the kernel


# Convolve the image with the Gaussian kernel
def gaussian_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)


####################
# 4. Box Filter
####################


def box_filter(image, kernel_size):
    # Get the dimensions of the image
    height, width = image.shape

    # Create an output image with the same dimensions as the input
    output = np.zeros((height, width), dtype=np.uint8)

    # Calculate the padding required based on the kernel size
    pad = kernel_size // 2

    # Iterate over each pixel in the image
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Extract the neighborhood around the current pixel
            neighborhood = image[i - pad : i + pad + 1, j - pad : j + pad + 1]

            # Calculate the mean value of the neighborhood
            mean_value = np.mean(neighborhood)

            # Assign the mean value to the output image
            output[i, j] = mean_value

    return output


####################
# 5. Laplacian Filter
####################


def laplacian_filter(image, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))

    # Calculate the center position of the kernel
    center = (kernel_size - 1) // 2

    # Define the Laplacian kernel
    kernel[center, center] = 4
    kernel[center - 1, center] = -1
    kernel[center + 1, center] = -1
    kernel[center, center - 1] = -1
    kernel[center, center + 1] = -1

    kernel = kernel - np.mean(kernel)

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
mean_filtered = put_text_on_center(mean_filter(input_image, kernel_size), "Mean Filter")
median_filtered = put_text_on_center(
    median_filter(input_image, kernel_size), "Median Filter"
)

# Create the Gaussian kernel
gaussian_kernel = create_gaussian_kernel(sigma, kernel_size)

gaussian_filtered = put_text_on_center(
    gaussian_filter(input_image, gaussian_kernel), "Gaussian Filter"
)

box_filtered = put_text_on_center(box_filter(input_image, kernel_size), "Box Filter")
laplacian_filtered = put_text_on_center(
    laplacian_filter(input_image, kernel_size), "Laplacian Filter"
)

# Save the results in a single output file
output_image = np.hstack(
    (
        put_text_on_center(input_image, "Input Image"),
        mean_filtered,
        median_filtered,
        gaussian_filtered,
        box_filtered,
        laplacian_filtered,
    )
)
cv2.imwrite("output_image.jpg", output_image)

# Display and save the individual filtered images if needed
cv2.imwrite("./output_images/mean_filtered.jpg", mean_filtered)
cv2.imwrite("./output_images/median_filtered.jpg", median_filtered)
cv2.imwrite("./output_images/gaussian_filtered.jpg", gaussian_filtered)
cv2.imwrite("./output_images/box_filtered.jpg", box_filtered)
cv2.imwrite("./output_images/laplacian_filtered.jpg", laplacian_filtered)

# Optionally, display the output image
cv2.imshow("Output Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
