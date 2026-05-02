#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform

import cv2
import numpy as np
import glob

# --- Step 1: Load Images with Quality Checks ---
def is_blurry(image, threshold=100):
    """Check if the image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
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

def load_images(image_pattern, blur_threshold=100, underexposed_threshold=10, overexposed_threshold=240):
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
            continue

        # Check for underexposure
        if is_underexposed(gray, underexposed_threshold):
            print(f"Rejected {path}: Underexposed (mean intensity < {underexposed_threshold})")
            continue

        # Check for overexposure
        if is_overexposed(gray, overexposed_threshold):
            print(f"Rejected {path}: Overexposed (mean intensity > {overexposed_threshold})")
            continue

        valid_images.append(img)
        valid_paths.append(path)

    return valid_images, valid_paths

# --- Step 2: Align Images Using Feature Matching ---
def align_images(images):
    # Use the first image as the reference
    reference = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    aligned_images = [reference]

    # Initialize ORB detector
    orb = cv2.ORB_create()

    for img in images[1:]:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(reference, None)
        kp2, des2 = orb.detectAndCompute(img_gray, None)

        if des1 is None or des2 is None:
            print("Warning: Could not find enough features for alignment. Skipping image.")
            continue

        # Match descriptors using Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography matrix
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Warp the image to align with the reference
        h, w = reference.shape
        aligned_img = cv2.warpPerspective(img_gray, M, (w, h))
        aligned_images.append(aligned_img)

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

# --- Main Workflow ---
if __name__ == "__main__":
    # Load images with quality checks
    images, image_paths = load_images(
        "moon_*.jpg",
        blur_threshold=100,
        underexposed_threshold=10,
        overexposed_threshold=240
    )

    if len(images) < 2:
        print("Error: At least 2 valid images are required for stacking.")
    else:
        print(f"Loaded {len(images)} valid images for stacking.")

        # Align images
        aligned_images = align_images(images)

        # Stack images
        stacked_img = stack_images(aligned_images)

        # Save the result
        save_result(stacked_img)

# --- Step 1: Load Images ---
def load_images(image_pattern):
    image_paths = glob.glob(image_pattern)
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    return images, image_paths

# --- Step 2: Align Images Using Feature Matching ---
def align_images(images):
    # Use the first image as the reference
    reference = images[0]
    aligned_images = [reference]

    # Initialize ORB detector
    orb = cv2.ORB_create()

    for img in images[1:]:
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(reference, None)
        kp2, des2 = orb.detectAndCompute(img, None)

        # Match descriptors using Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography matrix
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Warp the image to align with the reference
        h, w = reference.shape
        aligned_img = cv2.warpPerspective(img, M, (w, h))
        aligned_images.append(aligned_img)

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

# --- Main Workflow ---
if __name__ == "__main__":
    # Load images (adjust the pattern to match your files)
    images, _ = load_images("../../org/IMG_*.JPG")

    if len(images) < 2:
        print("Error: At least 2 images are required for stacking.")
    else:
        # Align images
        aligned_images = align_images(images)

        # Stack images
        stacked_img = stack_images(aligned_images)

        # Save the result
        save_result(stacked_img)
