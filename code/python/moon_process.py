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

def setup_output_directories(top_dir):
    """Create output directories and clean up existing symlinks."""
    output_dirs = {
        "preprocessed": os.path.join(top_dir, "outputs/preprocessed"),
        "blurred": os.path.join(top_dir, "outputs/blurred"),
        "underexposed": os.path.join(top_dir, "outputs/underexposed"),
        "overexposed": os.path.join(top_dir, "outputs/overexposed")
    }

    # Create output directories if they don't exist
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Remove existing files in output directories
    for dir_path in output_dirs.values():
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)

    return output_dirs

def preprocess_moon_image(img):
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def detect_moon_bbox(img, light_threshold=200):
    """
    Detect the Moon's bounding box using projection histograms.
    - Thresholds the image to isolate bright pixels (Moon).
    - Uses X and Y axis projections to find the Moon's center and extent.
    - Returns a square crop centered on the Moon, padded with black if needed.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate bright regions (Moon)
    _, thresh = cv2.threshold(gray, light_threshold, 255, cv2.THRESH_BINARY)

    # Project onto X and Y axes
    x_proj = np.sum(thresh, axis=0)  # Sum along rows (X-axis projection)
    y_proj = np.sum(thresh, axis=1)  # Sum along columns (Y-axis projection)

    # Find non-zero regions in projections
    x_nonzero = np.where(x_proj > 0)[0]
    y_nonzero = np.where(y_proj > 0)[0]

    if len(x_nonzero) == 0 or len(y_nonzero) == 0:
        print("No Moon detected")
        return img  # Fallback: return original if no Moon detected

    # Determine bounding box from projections/
    x1, x2 = x_nonzero[0], x_nonzero[-1]
    y1, y2 = y_nonzero[0], y_nonzero[-1]

    # Expand bounding box to a square, centered on the Moon
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    size = max(x2 - x1, y2 - y1) * 1.3  # Add 10% padding
    x1 = int(center_x - size // 2)
    y1 = int(center_y - size // 2)
    x2 = int(x1 + size)
    y2 = int(y1 + size)

    # Ensure the bounding box is within the image
    h_img, w_img = gray.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    # Crop the Moon region
    moon_crop = gray[y1:y2, x1:x2]

    # Pad with black to make it square
    if moon_crop.shape[0] != moon_crop.shape[1]:
        size = max(moon_crop.shape[0], moon_crop.shape[1])
        padded = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - moon_crop.shape[0]) // 2
        x_offset = (size - moon_crop.shape[1]) // 2
        padded[y_offset:y_offset + moon_crop.shape[0], x_offset:x_offset + moon_crop.shape[1]] = moon_crop
        moon_crop = padded

    return moon_crop

def validate_image(img, path, output_dirs, blur_threshold, underexposed_threshold, overexposed_threshold):
    """Validate an image and save the preprocessed version to the appropriate directory if rejected."""
    grayfull = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check for blur
    if is_blurry(grayfull, blur_threshold):
        print(f"Rejected {path}: Blurry (Laplacian variance < {blur_threshold})")
        output_path = os.path.join(output_dirs["blurred"], os.path.basename(path))
        cv2.imwrite(output_path, img)  # Save original image for rejected cases
        return False

    # Check for underexposure
    if is_underexposed(grayfull, underexposed_threshold):
        print(f"Rejected {path}: Underexposed (mean intensity < {underexposed_threshold})")
        output_path = os.path.join(output_dirs["underexposed"], os.path.basename(path))
        cv2.imwrite(output_path, img)  # Save original image for rejected cases
        return False

    # Check for overexposure
    if is_overexposed(grayfull, overexposed_threshold):
        print(f"Rejected {path}: Overexposed (saturation > {overexposed_threshold}%)")
        output_path = os.path.join(output_dirs["overexposed"], os.path.basename(path))
        cv2.imwrite(output_path, img)  # Save original image for rejected cases
        return False

    # Preprocess and crop to Moon disk for valid images
    moon_img = detect_moon_bbox(img)
    if moon_img is None:
        print(f"Warning: Moon detection failed for {path}. Skipping.")
        return False

    # Save preprocessed image for valid cases
    output_path = os.path.join(output_dirs["preprocessed"], os.path.basename(path))
    cv2.imwrite(output_path, moon_img)

    return True, moon_img  # Return the cropped Moon image if valid


def load_images(image_pattern, blur_threshold=5, underexposed_threshold=10, overexposed_threshold=1.0):
    print(f"Loading images from {image_pattern}")

    # Setup output directories
    output_dirs = setup_output_directories(top_dir)

    image_paths = glob.glob(image_pattern)
    valid_images = []
    valid_paths = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not read {path}. Skipping.")
            continue

        result = validate_image(img, path, output_dirs, blur_threshold, underexposed_threshold, overexposed_threshold)
        
        if isinstance(result, tuple) and result[0]:
            valid_images.append(result[1])  # Append the cropped Moon image
            valid_paths.append(path)
        return valid_images, valid_paths

    return valid_images, valid_paths

def crop_to_moon_disk(img):
    """Crop image to the Moon's bounding box (assumes Moon is the brightest object)."""
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return img[y:y+h, x:x+w]
    return img  # Fallback: return original if no contour found

def align_single_image_phase_correlation(reference, img):
    """Align using phase correlation (for translation-only)."""
    shift, error, phasediff = phase_cross_correlation(reference, img)
    rows, cols = reference.shape
    M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
    aligned = cv2.warpAffine(img, M, (cols, rows))
    return aligned

def align_single_image(reference, img, i, sift, aligned_dir):
    """Align a single image to the reference and save the result."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    reference_gray = reference if len(reference.shape) == 2 else cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Try feature-based alignment first
    kp1, des1 = sift.detectAndCompute(reference_gray, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            try:
                M, _ = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if M is not None:
                    h, w = reference_gray.shape
                    aligned_img = cv2.warpAffine(img_gray, M, (w, h))
                    cv2.imwrite(os.path.join(aligned_dir, f"aligned_{i:03d}.png"), aligned_img)
                    return aligned_img
            except cv2.error:
                pass

    # Fallback: Phase correlation (translation-only)
    print(f"Warning: Feature alignment failed for image {i}. Trying phase correlation.")
    aligned_img = align_single_image_phase_correlation(reference_gray, img_gray)
    cv2.imwrite(os.path.join(aligned_dir, f"aligned_{i:03d}.png"), aligned_img)
    return aligned_img

def align_images(images):
    if len(images) < 1:
        print("Error: No images to align.")
        return []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    top_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    aligned_dir = os.path.join(top_dir, "outputs/aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    # Use the first image as the reference (handle both grayscale and BGR)
    reference = images[0]
    if len(reference.shape) == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    aligned_images = [reference]
    cv2.imwrite(os.path.join(aligned_dir, "aligned_000.png"), reference)

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
    return

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
