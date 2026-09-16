import cv2
import numpy as np

# Path to the image on your computer
image_path = r"C:\Users\timbe\Downloads\duck.jpg"

# Read the image from disk (loaded in color, BGR order, by default)
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read image at: {image_path}")

# Convert the color image to greyscale
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Build a random 3x3 kernel with a wide range of integer values (-10 to 10) and
# print it so you can see exactly what's being applied. Unlike a blur kernel
# (small positive weights that sum to ~1, averaging neighbors together), these
# large, mixed-sign, unnormalized weights push pixel values well outside their
# original range -- producing sharpening, embossing, or noisy high-contrast
# effects instead of smoothing.
random_kernel = np.random.randint(-10, 11, size=(3, 3)).astype(np.float32)
print("Random 3x3 kernel:")
print(random_kernel)

# Convolve the greyscale image with that kernel: cv2.filter2D slides the 3x3
# kernel over every pixel, multiplying it element-wise with the pixel's 3x3
# neighborhood and summing the result to produce the new pixel value.
# ddepth=-1 keeps the output in the same 8-bit depth as the input, so cv2
# automatically clips results below 0 or above 255.
random_kernel_image = cv2.filter2D(grey_image, -1, random_kernel)

# Make resizable windows so large images can be shown at full extent,
# scaled down to fit your screen instead of being cropped
max_display_width, max_display_height = 900, 700

def resize_to_fit(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# Show the original, greyscale, and random-kernel-filtered images (resized for
# display only; saved files stay full-res)
cv2.imshow("Original", resize_to_fit(image, max_display_width, max_display_height))
cv2.imshow("Greyscale", resize_to_fit(grey_image, max_display_width, max_display_height))
cv2.imshow("Random Kernel Filter", resize_to_fit(random_kernel_image, max_display_width, max_display_height))

# Pre-scale the greyscale image once so the trackbar callbacks don't have to resize every frame
display_grey = resize_to_fit(grey_image, max_display_width, max_display_height)

def dilate_and_subtract(binary_img, kernel_size):
    """Morphological gradient: grow the white shape outward with dilation, then
    subtract the original from the dilated version. What's left is just the band
    of pixels the dilation added -- i.e. the outline/edge of the shape."""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary_img, kernel)
    return cv2.subtract(dilated, binary_img)

# OpenCV's native trackbar widget can't draw custom text at its ends, so the min/max
# for each slider is baked directly into its label text instead.
blur_label = "Blur Kernel [0-20]"
threshold_label = "Threshold [0-255]"
kernel_label = "Kernel Size [0-10]"

blur_window = "Blurred"
binary_window = "Black and White"
edge_window = "Dilated Edge"
cv2.namedWindow(blur_window)
cv2.namedWindow(binary_window)
cv2.namedWindow(edge_window)

current_blurred = display_grey  # starting point before any blur slider movement
current_binary = None  # shared between the threshold and kernel-size callbacks

def update_blur(blur_value):
    global current_blurred
    # Map the slider's 0-20 range to odd kernel sizes 1, 3, 5, ..., 41
    kernel_size = 2 * blur_value + 1
    # A Gaussian blur averages each pixel with its neighbors, weighted so closer
    # neighbors count more -- this smooths out noise before we threshold/edge-detect
    current_blurred = cv2.GaussianBlur(display_grey, (kernel_size, kernel_size), 0)
    cv2.imshow(blur_window, current_blurred)
    update_binary(cv2.getTrackbarPos(threshold_label, binary_window))

def update_binary(threshold_value):
    global current_binary
    # Any pixel above the threshold becomes white (255), anything at or below becomes black (0)
    _, current_binary = cv2.threshold(current_blurred, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow(binary_window, current_binary)
    update_edge(cv2.getTrackbarPos(kernel_label, edge_window))

def update_edge(kernel_value):
    if current_binary is None:
        return
    # Map the slider's 0-10 range to odd kernel sizes 1, 3, 5, ..., 21
    kernel_size = 2 * kernel_value + 1
    edge = dilate_and_subtract(current_binary, kernel_size)
    cv2.imshow(edge_window, edge)

# Sliders are created in reverse pipeline order (edge, then threshold, then blur)
# so that each one already exists if an earlier creation triggers an immediate callback
cv2.createTrackbar(kernel_label, edge_window, 1, 10, update_edge)
cv2.createTrackbar(threshold_label, binary_window, 127, 255, update_binary)
cv2.createTrackbar(blur_label, blur_window, 0, 20, update_blur)

# Render the initial frames before the wait loop starts
update_blur(0)

# Wait for a key press, then close the windows
cv2.waitKey(0)

# Read back each slider's final position so we can save matching full-resolution results
final_blur_kernel_size = 2 * cv2.getTrackbarPos(blur_label, blur_window) + 1
final_threshold = cv2.getTrackbarPos(threshold_label, binary_window)
final_kernel_size = 2 * cv2.getTrackbarPos(kernel_label, edge_window) + 1
cv2.destroyAllWindows()

# Save the greyscale image and the random-kernel-filtered image to disk
cv2.imwrite("greyscale_output.jpg", grey_image)
cv2.imwrite("random_kernel_output.jpg", random_kernel_image)

# Apply the chosen blur to the full-resolution greyscale image and save it
blurred_full_res = cv2.GaussianBlur(grey_image, (final_blur_kernel_size, final_blur_kernel_size), 0)
cv2.imwrite("blurred_output.jpg", blurred_full_res)

# Apply the chosen threshold to the full-resolution blurred image and save it too
_, binary_full_res = cv2.threshold(blurred_full_res, final_threshold, 255, cv2.THRESH_BINARY)
cv2.imwrite("binary_output.jpg", binary_full_res)

# Dilate and subtract the full-resolution binary image using the chosen kernel size and save it
edge_full_res = dilate_and_subtract(binary_full_res, final_kernel_size)
cv2.imwrite("edge_output.jpg", edge_full_res)
