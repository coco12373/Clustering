import os
import cv2
import numpy as np
import random

def get_image_files(directory):
    """Return a list of image file paths in the specified directory."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

def random_color():
    """Generate a random color in BGR format."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# def detect_and_save_corners(image_path, output_directory):
#     """Detect corners in the image and save the result."""
#     # Read the image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Detect corners using the Shi-Tomasi corner detector
#     corners = cv2.goodFeaturesToTrack(image, maxCorners=1000, qualityLevel=0.01, minDistance=10)
#     corners = np.int0(corners)
    
#     # Convert the grayscale image to BGR so we can draw colored shapes
#     image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
#     # Draw a square with random color around each detected corner
#     for corner in corners:
#         x, y = corner.ravel()
#         top_left = (x - 100, y - 100)
#         bottom_right = (x + 100, y + 100)
#         cv2.rectangle(image_colored, top_left, bottom_right, random_color(), 4)
    
#     # Save the image with corners to the output directory
#     output_path = os.path.join(output_directory, os.path.basename(image_path))
#     cv2.imwrite(output_path, image_colored)
def detect_and_save_corners(image_path, output_directory, patch_directory):
    """Detect corners in the image, save the result, and save patches."""
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detect corners using the Shi-Tomasi corner detector
    corners = cv2.goodFeaturesToTrack(image, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    corners = np.int0(corners)
    
    # Convert the grayscale image to BGR so we can draw colored shapes
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Create the patch directory if it doesn't exist
    create_directory(patch_directory)
    
    # Get the image filename without extension
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create a subdirectory for the image patches
    image_patch_directory = os.path.join(patch_directory, f"{image_filename}_patches")
    create_directory(image_patch_directory)
    
    # Process each corner and save the patch
    for i, corner in enumerate(corners):
        x, y = corner.ravel()
        top_left = (x - 100, y - 100)
        bottom_right = (x + 100, y + 100)
        
        # Check if the patch coordinates are within the image boundaries
        if top_left[0] < 0 or top_left[1] < 0 or bottom_right[0] >= image.shape[1] or bottom_right[1] >= image.shape[0]:
            print(f"Patch {i} in image {image_filename} is out of image boundaries.")
            continue
        
        # Extract the patch from the image
        patch = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        # Check if the extracted patch is empty (all black)
        if np.count_nonzero(patch) == 0:
            print(f"Patch {i} in image {image_filename} is empty.")
            continue
        
        # Save the patch to the image-specific patch directory
        patch_path = os.path.join(image_patch_directory, f"patch_{i}.png")
        cv2.imwrite(patch_path, patch)
        
        # Draw a square with random color around each detected corner
        cv2.rectangle(image_colored, top_left, bottom_right, random_color(), 4)
    
    # Save the image with corners to the output directory
    output_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(output_path, image_colored)

def create_directory(directory):
    """Create the directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    images_directory = 'images'  # Change this to your images directory
    output_directory = 'output_images_rec_c'  # Directory to save the results
    patch_directory = 'patch'  # Directory to save the patches

    # Create the output directory
    create_directory(output_directory)
    
    # Get all image files in the specified directory
    image_files = get_image_files(images_directory)
    
    if not image_files:
        print("No image files found in the directory.")
        return
    
    # Process each image file and save the results
    for image_file in image_files:
        image_filename = os.path.splitext(os.path.basename(image_file))[0]
        image_patch_directory = os.path.join(patch_directory, f"{image_filename}_patches")
        detect_and_save_corners(image_file, output_directory, image_patch_directory)
    
    print(f"Processed images are saved in the directory '{output_directory}'.")
    print(f"Patches are saved in the subdirectories of '{patch_directory}'.")
if __name__ == "__main__":
    main()
