#!/usr/bin/env python3
import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.abspath(os.path.join(script_dir, "../.."))

# --- Step 1: Load Images with Quality Checks ---
def is_blurry(image, threshold=100):
    """Check if the image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Laplacian variance: {laplacian_var}")
    return laplacian_var < threshold

def is_underexposed(image, threshold=10):
    """Check if the image is underexposed (too dark)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_intensity = np.mean(gray)
    return mean_intensity < threshold

def is_overexposed(image, saturation_percent_threshold=1.0):
    """Check if the image is overexposed based on the percentage of saturated pixels (value = 255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    total_pixels = gray.size
    saturated_pixels = np.sum(gray == 255)
    saturation_percent = (saturated_pixels / total_pixels) * 100
    return saturation_percent > saturation_percent_threshold

def load_images(image_pattern, blur_threshold=5, underexposed_threshold=10, overexposed_threshold=1.0):
    print(f"Loading images from {image_pattern}")

    # Define output directories relative to top_dir
    output_dirs = {
        "blurred": os.path.join(top_dir, "outputs/blurred"),
        "underexposed": os.path.join(top_dir, "outputs/underexposed"),
        "overexposed": os.path.join(top_dir, "outputs/overexposed")
    }

    # Create output directories if they don't exist
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Remove existing symlinks in output directories
    for dir_path in output_dirs.values():
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.islink(item_path):
                os.remove(item_path)

    image_paths = glob.glob(image_pattern)
    valid_images = []
    valid_paths = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not read {path}. Skipping.")
            continue

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check for blur
        if is_blurry(gray, blur_threshold):
            print(f"Rejected {path}: Blurry (Laplacian variance < {blur_threshold})")
            os.symlink(os.path.abspath(path), os.path.join(output_dirs["blurred"], os.path.basename(path)))
            continue

        # Check for underexposure
        if is_underexposed(gray, underexposed_threshold):
            print(f"Rejected {path}: Underexposed (mean intensity < {underexposed_threshold})")
            os.symlink(os.path.abspath(path), os.path.join(output_dirs["underexposed"], os.path.basename(path)))
            continue

        # Check for overexposure
        if is_overexposed(gray, overexposed_threshold):
            print(f"Rejected {path}: Overexposed (saturation > {overexposed_threshold}%)")
            os.symlink(os.path.abspath(path), os.path.join(output_dirs["overexposed"], os.path.basename(path)))
            continue

        valid_images.append(img)
        valid_paths.append(path)

    return valid_images, valid_paths
 
def align_single_image(reference, img, i, sift, aligned_dir):
    """Align a single image to the reference and save the result."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(reference, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print(f"Warning: Not enough keypoints for image {i}. Skipping.")
        return None

    # Match descriptors using FLANN
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print(f"Warning: Not enough good matches for image {i}. Skipping.")
        return None

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix
    try:
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    except cv2.error as e:
        print(f"Warning: Failed to compute homography for image {i}: {e}. Skipping.")
        return None

    if M is None:
        print(f"Warning: Homography matrix is None for image {i}. Skipping.")
        return None

    # Warp the image to align with the reference
    h, w = reference.shape
    try:
        aligned_img = cv2.warpPerspective(img_gray, M, (w, h))
        cv2.imwrite(os.path.join(aligned_dir, f"aligned_{i:03d}.png"), aligned_img)
        return aligned_img
    except cv2.error as e:
        print(f"Warning: Failed to warp image {i}: {e}. Skipping.")
        return None

def align_images(images):
    if len(images) < 1:
        print("Error: No images to align.")
        return []

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    top_dir = os.path.abspath(os.path.join(script_dir, "../.."))

    # Define output directory for aligned images
    aligned_dir = os.path.join(top_dir, "outputs/aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    # Use the first image as the reference
    reference = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    aligned_images = [reference]
    cv2.imwrite(os.path.join(aligned_dir, "aligned_000.png"), reference)  # Save reference

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    for i, img in enumerate(images[1:], start=1):
        aligned_img = align_single_image(reference, img, i, sift, aligned_dir)
        if aligned_img is not None:
            aligned_images.append(aligned_img)

    if len(aligned_images) < 2:
        print("Error: At least 2 images are required for stacking.")
        return []

    return aligned_images


# --- Step 3: Stack Aligned Images ---
def stack_images(aligned_images):
    # Convert to float32 for averaging
    stacked = np.zeros_like(aligned_images[0], dtype=np.float32)
    for img in aligned_images:
        stacked += img.astype(np.float32)
    stacked /= len(aligned_images)
    return np.uint8(stacked)

# --- Step 4: Save the Result ---
def save_result(stacked_img, output_path="stacked_moon.png"):
    cv2.imwrite(output_path, stacked_img)
    print(f"Stacked image saved as {output_path}")

def main():
    # Load images (adjust the pattern to match your files)
    images, _ = load_images(os.path.join(top_dir, "../org/IMG_*.JPG"))
    print(f"Loaded {len(images)} images for stacking.")
    
    if len(images) < 2:
        print("Error: At least 2 images are required for stacking.")
    else:
        # Align images
        aligned_images = align_images(images)

        # Stack images
        stacked_img = stack_images(aligned_images)

        # Save the result
        save_result(stacked_img, os.path.join(top_dir, "outputs","stacked_moon.png"))

# --- Main Workflow ---
if __name__ == "__main__":
    main()
