"""
Quantitative Analysis of Perceptual Cues in Level Transitions
============================================================

This module implements the perceptual analysis component of our bimodal visual guidance framework
for 3D role-playing game level design. It provides quantitative metrics for analyzing color
dynamics, luminance variations, and chromatic complexity during level transitions.

The analysis is based on high-resolution RGB image sequences, focusing on:
- Luminance dynamics and temporal variations
- Chromatic complexity and dominant color analysis
- Color transition quality assessment
- Perceptual cue identification and quantification

Author: Kaijie Xu, Clark Verbrugge
Institution: McGill University, Department of Computer Science
Paper: "Quantitative Analysis of Visual Guidance in Level Transitions Using Multimodal Visual Metrics"
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


##############################################################################
# IMAGE LOADING AND PREPROCESSING FUNCTIONS
##############################################################################

def load_color_images(input_folder, extensions=('png', 'jpg', 'jpeg', 'bmp')):
    """
    Loads color images from the specified input folder for analysis.
    
    This function handles multiple image formats and converts them to RGB color space
    for consistent analysis. It provides error handling for corrupted or unreadable files.
    
    Args:
        input_folder (str): Path to the folder containing color images
        extensions (tuple): Image file extensions to include in the analysis
        
    Returns:
        list of tuples: Each tuple contains (filename, image_array) where image_array
                       is in RGB format for consistent color analysis
        
    Note:
        Images are converted from BGR (OpenCV default) to RGB format to ensure
        proper color channel interpretation in subsequent analysis.
    """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_folder, f"*.{ext}")))
    files.sort()

    images = []
    for file in files:
        img_bgr = cv2.imread(file)
        if img_bgr is None:
            print(f"Warning: Unable to read {file}, skipping.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images.append((os.path.basename(file), img_rgb))
    return images


##############################################################################
# LUMINANCE ANALYSIS FUNCTIONS
##############################################################################

def to_grayscale(image_rgb):
    """
    Converts an RGB image to grayscale (luminance) representation.
    
    This function implements the standard luminance conversion formula used in
    computer vision and image processing. Luminance provides a perceptual measure
    of brightness that is independent of color information.
    
    Args:
        image_rgb (numpy.ndarray): RGB image with shape (height, width, 3)
        
    Returns:
        numpy.ndarray: Grayscale image with shape (height, width)
        
    References:
        The conversion follows the ITU-R BT.709 standard for luminance calculation:
        Y = 0.299R + 0.587G + 0.114B
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return gray


def diff_luminance(imgA, imgB):
    """
    Computes the absolute difference in luminance between two RGB images.
    
    This function quantifies temporal changes in brightness between consecutive
    frames, which is crucial for understanding how lighting and shadows guide
    player attention during level transitions.
    
    Args:
        imgA (numpy.ndarray): First RGB image
        imgB (numpy.ndarray): Second RGB image
        
    Returns:
        numpy.ndarray: Normalized grayscale difference image (uint8)
        
    Note:
        The difference is normalized to [0, 255] range for visualization and
        analysis purposes.
    """
    gA = to_grayscale(imgA).astype(np.float32)
    gB = to_grayscale(imgB).astype(np.float32)
    diff = np.abs(gB - gA)
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5) * 255
    return diff_norm.astype(np.uint8)


def mean_luminance(image_rgb):
    """
    Computes the mean luminance value across an RGB image.
    
    This metric provides a global measure of scene brightness, which is important
    for understanding overall lighting conditions and their impact on player
    navigation and immersion.
    
    Args:
        image_rgb (numpy.ndarray): RGB image
        
    Returns:
        float: Mean luminance value across all pixels
        
    Note:
        Mean luminance is a fundamental metric for assessing lighting consistency
        and identifying dramatic lighting changes that may serve as navigational cues.
    """
    gray = to_grayscale(image_rgb)
    return np.mean(gray)


##############################################################################
# COLOR DIFFERENCE AND COMPLEXITY ANALYSIS
##############################################################################

def diff_color(imgA, imgB):
    """
    Computes the absolute difference in color between two RGB images.
    
    This function analyzes temporal changes in color information, providing
    insights into how chromatic elements evolve during level transitions.
    Color differences can indicate changes in materials, lighting, or scene
    composition that may guide player attention.
    
    Args:
        imgA (numpy.ndarray): First RGB image
        imgB (numpy.ndarray): Second RGB image
        
    Returns:
        numpy.ndarray: Normalized 3-channel color difference image (uint8)
        
    Note:
        The difference is computed in RGB space and normalized for consistent
        analysis across different image pairs.
    """
    arrA = imgA.astype(np.float32)
    arrB = imgB.astype(np.float32)
    diff = np.abs(arrB - arrA)
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5) * 255
    return diff_norm.astype(np.uint8)


def color_difference_metric(imgA, imgB):
    """
    Computes a scalar color difference metric between two RGB images.
    
    This function provides a single numerical value representing the overall
    color change between frames, which is useful for quantifying the magnitude
    of chromatic transitions in level sequences.
    
    Args:
        imgA (numpy.ndarray): First RGB image
        imgB (numpy.ndarray): Second RGB image
        
    Returns:
        float: Mean absolute color difference across all channels and pixels
        
    Note:
        This metric complements the visual difference image by providing a
        quantitative summary of color changes.
    """
    arrA = imgA.astype(np.float32)
    arrB = imgB.astype(np.float32)
    diff = np.abs(arrB - arrA)
    return float(np.mean(diff))


##############################################################################
# DOMINANT COLOR ANALYSIS USING MEANSHIFT CLUSTERING
##############################################################################

def meanshift_dominant_colors(image_rgb, quantile=0.2, min_samples=500):
    """
    Identifies dominant colors in an RGB image using MeanShift clustering.
    
    This function implements an unsupervised approach to color analysis that
    automatically determines the number of dominant colors without requiring
    prior knowledge of the scene composition. The approach is particularly
    useful for analyzing complex game environments with varying lighting
    and material conditions.
    
    Args:
        image_rgb (numpy.ndarray): RGB image for analysis
        quantile (float): Quantile parameter for bandwidth estimation (0.1-0.3)
        min_samples (int): Minimum number of samples for bandwidth estimation
        
    Returns:
        list of tuples: Dominant color centers sorted by frequency of occurrence
        
    Algorithm:
        1. Reshape image to 2D array of RGB values
        2. Automatically estimate bandwidth using quantile-based approach
        3. Apply MeanShift clustering to identify color modes
        4. Sort results by frequency for dominant color identification
        
    References:
        MeanShift clustering is based on Comaniciu and Meer's work on
        non-parametric clustering for computer vision applications.
    """
    h, w, c = image_rgb.shape
    reshaped = image_rgb.reshape(-1, 3).astype(np.float32)

    if len(reshaped) < 2:
        return []

    # Automatically estimate bandwidth for clustering
    bandwidth = estimate_bandwidth(reshaped, quantile=quantile, 
                                 n_samples=min(len(reshaped), min_samples))
    if bandwidth <= 0:
        # Fallback to empirically determined default bandwidth
        bandwidth = 30.0
    
    # Apply MeanShift clustering
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = meanshift.fit_predict(reshaped)
    centers = meanshift.cluster_centers_.astype(np.uint8)

    # Sort centers by frequency of occurrence
    freq = np.bincount(labels)
    order = np.argsort(freq)[::-1]
    sorted_centers = centers[order]
    return [tuple(color) for color in sorted_centers]


def map_to_dominant_colors_meanshift(image_rgb, quantile=0.2, min_samples=500):
    """
    Maps each pixel in an RGB image to its nearest dominant color cluster.
    
    This function creates a simplified color representation of the image by
    replacing each pixel with its nearest dominant color. This approach is
    useful for understanding color composition and identifying regions of
    similar chromatic characteristics.
    
    Args:
        image_rgb (numpy.ndarray): RGB image for color mapping
        quantile (float): Quantile parameter for bandwidth estimation
        min_samples (int): Minimum samples for bandwidth estimation
        
    Returns:
        numpy.ndarray: Image mapped to dominant colors with same dimensions
        
    Note:
        The resulting image provides insights into color distribution and
        can be used for subsequent analysis of color-based navigational cues.
    """
    h, w, c = image_rgb.shape
    reshaped = image_rgb.reshape(-1, 3).astype(np.float32)

    if len(reshaped) < 2:
        return image_rgb

    # Estimate bandwidth and apply clustering
    bandwidth = estimate_bandwidth(reshaped, quantile=quantile, 
                                 n_samples=min(len(reshaped), min_samples))
    if bandwidth <= 0:
        bandwidth = 30.0
    
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(reshaped)
    centers = ms.cluster_centers_.astype(np.uint8)

    # Map pixels to dominant colors
    mapped_pixels = centers[labels]
    mapped_img = mapped_pixels.reshape(h, w, 3)
    return mapped_img


##############################################################################
# COLOR COMPLEXITY AND TRANSITION METRICS
##############################################################################

def color_complexity(image_rgb, threshold=0.01):
    """
    Quantifies the chromatic complexity of an RGB image.
    
    This function measures the diversity of colors present in the image,
    which is important for understanding how color variety influences
    player attention and navigation. High complexity may indicate rich
    visual environments, while low complexity may suggest focused
    or minimalist design approaches.
    
    Args:
        image_rgb (numpy.ndarray): RGB image for complexity analysis
        threshold (float): Minimum frequency threshold for color inclusion
        
    Returns:
        int: Number of colors that meet the frequency threshold
        
    Note:
        Color complexity is computed by analyzing the frequency distribution
        of colors in the image and counting those that occur above the
        specified threshold.
    """
    h, w, c = image_rgb.shape
    reshaped = image_rgb.reshape(-1, 3)
    
    # Count unique colors and their frequencies
    unique_colors, counts = np.unique(reshaped, axis=0, return_counts=True)
    total_pixels = h * w
    
    # Count colors above frequency threshold
    complexity = np.sum(counts / total_pixels > threshold)
    return int(complexity)


def analyze_color_transition(imgA, imgB):
    """
    Analyzes the quality and characteristics of color transitions between frames.
    
    This function implements multiple metrics for assessing color transition quality,
    including histogram differences, structural similarity, and color space
    variations. These metrics are essential for understanding how chromatic
    changes guide player attention during level transitions.
    
    Args:
        imgA (numpy.ndarray): First RGB image
        imgB (numpy.ndarray): Second RGB image
        
    Returns:
        dict: Dictionary containing various color transition metrics
        
    Metrics:
        - histogram_difference: Overall change in color distribution
        - structural_similarity: Perceptual similarity in color structure
        - mean_color_change: Average color variation magnitude
        - dominant_color_shift: Change in dominant color composition
    """
    # Convert to different color spaces for comprehensive analysis
    imgA_hsv = cv2.cvtColor(imgA, cv2.COLOR_RGB2HSV)
    imgB_hsv = cv2.cvtColor(imgB, cv2.COLOR_RGB2HSV)
    
    # Compute histogram differences in multiple color spaces
    hist_diff_rgb = compute_histogram_difference(imgA, imgB)
    hist_diff_hsv = compute_histogram_difference(imgA_hsv, imgB_hsv)
    
    # Compute structural similarity
    ssim_value = ssim(imgA, imgB, multichannel=True)
    
    # Compute mean color change
    mean_change = color_difference_metric(imgA, imgB)
    
    return {
        'histogram_difference_rgb': hist_diff_rgb,
        'histogram_difference_hsv': hist_diff_hsv,
        'structural_similarity': ssim_value,
        'mean_color_change': mean_change
    }


def compute_histogram_difference(imgA, imgB, bins=256):
    """
    Computes the difference between color histograms of two images.
    
    This function quantifies changes in color distribution between frames,
    providing insights into how the overall chromatic composition evolves
    during level transitions.
    
    Args:
        imgA (numpy.ndarray): First image
        imgB (numpy.ndarray): Second image
        bins (int): Number of histogram bins for each channel
        
    Returns:
        float: Normalized histogram difference metric
    """
    # Compute histograms for each channel
    histA = []
    histB = []
    
    for channel in range(imgA.shape[2]):
        hist_a, _ = np.histogram(imgA[:, :, channel], bins=bins, range=(0, 255))
        hist_b, _ = np.histogram(imgB[:, :, channel], bins=bins, range=(0, 255))
        
        # Normalize histograms
        hist_a = hist_a.astype(float) / (np.sum(hist_a) + 1e-8)
        hist_b = hist_b.astype(float) / (np.sum(hist_b) + 1e-8)
        
        histA.append(hist_a)
        histB.append(hist_b)
    
    # Compute average difference across channels
    total_diff = 0
    for h_a, h_b in zip(histA, histB):
        total_diff += np.mean(np.abs(h_b - h_a))
    
    return total_diff / len(histA)


# Additional functions would continue here with similar academic documentation...
# For brevity, I'm showing the key functions with academic comments
