"""
Quantitative Analysis of Geometric Constraints in Level Transitions
================================================================

This module implements the geometric analysis component of our bimodal visual guidance framework
for 3D role-playing game level design. It provides quantitative metrics for analyzing spatial
depth distribution, geometric constraints, and navigational cues during level transitions.

The analysis is based on depth maps derived from raycast grids, focusing on:
- Spatial correlation analysis between depth values and coordinates
- Field of View (FOV) detection and analysis
- Contour orientation analysis for leading line detection
- Object detection and spatial distribution metrics
- Transition quality assessment using structural similarity measures

Author: Kaijie Xu, Clark Verbrugge
Institution: McGill University, Department of Computer Science
Paper: "Quantitative Analysis of Visual Guidance in Level Transitions Using Multimodal Visual Metrics"
"""

import os
import glob
import re
import math
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
from skimage.measure import find_contours, regionprops, label
from skimage.morphology import remove_small_objects, remove_small_holes, closing, square
from skimage.filters import threshold_otsu
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon
from matplotlib.patches import Rectangle


##############################################################################
# DATA LOADING AND PREPROCESSING FUNCTIONS
##############################################################################

def load_depth_csv_regex(file_path):
    """
    Loads depth data from CSV files containing floating-point depth values.
    
    This function parses depth data extracted from raycast grids, handling various
    CSV formats and providing fallback mechanisms for malformed data.
    
    Args:
        file_path (str): Path to the CSV file containing depth data
        
    Returns:
        numpy.ndarray: 2D array of depth values, with NaN for invalid entries
        
    Note:
        The function assumes a resolution of 100x100 pixels by default, but this
        can be adjusted based on the specific depth map resolution used.
    """
    float_pattern = re.compile(r'-?\d+\.\d+')
    depth_list = []

    if not os.path.isfile(file_path):
        print(f"Error: File {file_path} does not exist.")
        return None

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            matches = float_pattern.findall(line)
            if not matches:
                # Fallback for malformed data: create row with NaN values
                depth_row = [np.nan] * 100  # Adjust resolution as needed
            else:
                depth_row = [float(x) for x in matches]
            depth_list.append(depth_row)

    depth_array = np.array(depth_list)
    return depth_array


##############################################################################
# SPATIAL CORRELATION ANALYSIS
##############################################################################

def compute_spatial_correlation(depth):
    """
    Computes spatial correlation coefficients between depth values and their coordinates.
    
    This metric quantifies the relationship between spatial position and depth values,
    providing insights into how geometric constraints are distributed across the scene.
    High correlation values indicate systematic depth variations that may serve as
    navigational cues.
    
    Args:
        depth (numpy.ndarray): 2D depth map array
        
    Returns:
        tuple: (correlation_x, correlation_y) correlation coefficients
        
    References:
        This analysis is based on spatial statistics principles used in computer
        vision and geographic information systems for quantifying spatial patterns.
    """
    valid = ~np.isnan(depth)
    if not np.any(valid):
        return np.nan, np.nan

    y_indices, x_indices = np.indices(depth.shape)
    x_valid = x_indices[valid].flatten()
    y_valid = y_indices[valid].flatten()
    depth_valid = depth[valid].flatten()

    if np.std(x_valid) == 0 or np.std(depth_valid) == 0:
        corr_x = np.nan
    else:
        corr_x = np.corrcoef(x_valid, depth_valid)[0, 1]

    if np.std(y_valid) == 0 or np.std(depth_valid) == 0:
        corr_y = np.nan
    else:
        corr_y = np.corrcoef(y_valid, depth_valid)[0, 1]

    return corr_x, corr_y


##############################################################################
# FIELD OF VIEW (FOV) DETECTION AND ANALYSIS
##############################################################################

def detect_fov_from_labels(labels):
    """
    Detects and analyzes the Field of View (FOV) from labeled depth data.
    
    The FOV represents the navigable space visible to the player, excluding
    blocking objects. This function implements morphological processing to
    identify the largest connected navigable region, which is crucial for
    understanding spatial constraints and navigational possibilities.
    
    Args:
        labels (numpy.ndarray): Labeled depth data where 0 and >1 represent
                               navigable space, and 1 represents blocking objects
        
    Returns:
        numpy.ndarray: Binary mask of the largest connected FOV region
        
    Algorithm:
        1. Create binary FOV mask from navigable labels
        2. Apply morphological operations to remove noise
        3. Identify largest connected component
        4. Smooth boundaries for consistent analysis
        
    Note:
        The morphological parameters (min_size=500, area_threshold=500) are
        empirically determined and may need adjustment for different resolutions.
    """
    # Step 1: Create binary FOV mask from navigable labels
    fov_mask = (labels == 0) | (labels > 1)

    # Step 2: Apply morphological processing to remove noise and artifacts
    fov_mask = remove_small_objects(fov_mask, min_size=500)
    fov_mask = remove_small_holes(fov_mask, area_threshold=500)

    # Step 3: Identify the largest connected component (main navigable area)
    labeled_fov, num_features = label(fov_mask, connectivity=2, return_num=True)
    if num_features == 0:
        return np.ones_like(labels, dtype=bool)
    else:
        largest_label = np.argmax([np.sum(labeled_fov == i) for i in range(1, num_features + 1)]) + 1
        fov_mask = (labeled_fov == largest_label)

    # Step 4: Smooth boundaries for consistent analysis
    fov_mask = closing(fov_mask, square(5))

    return fov_mask


##############################################################################
# CONTOUR ORIENTATION ANALYSIS FOR LEADING LINE DETECTION
##############################################################################

def compute_contour_orientation(contour):
    """
    Computes the orientation of a contour using Principal Component Analysis (PCA).
    
    This function analyzes the dominant direction of contours, which is essential
    for identifying leading lines and directional cues in level design. The
    orientation is computed relative to the x-axis and normalized to [0, 180) degrees.
    
    Args:
        contour (numpy.ndarray): Contour coordinates in (row, col) format
        
    Returns:
        float: Orientation angle in degrees [0, 180), or NaN if insufficient data
        
    Algorithm:
        1. Center the contour coordinates
        2. Compute covariance matrix
        3. Perform eigenvalue decomposition
        4. Extract principal axis direction
        5. Convert to angle representation
        
    References:
        This approach is based on PCA-based orientation analysis commonly used
        in computer vision for shape analysis and feature extraction.
    """
    coords = contour  # Shape (N,2) => (row, col)
    if coords.shape[0] < 2:
        return np.nan

    # Center the coordinates for PCA
    mean_r = np.mean(coords[:, 0])
    mean_c = np.mean(coords[:, 1])
    centered = coords - np.array([mean_r, mean_c])

    # Compute covariance matrix and perform eigenvalue decomposition
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx_max = np.argmax(eigvals)
    principal_axis = eigvecs[:, idx_max]  # (2,)

    # Extract directional components
    dy = principal_axis[0]  # Row direction
    dx = principal_axis[1]  # Column direction

    # Convert to angle representation
    angle = math.degrees(math.atan2(dy, dx))
    angle = angle % 180.0  # Normalize to [0, 180)
    return angle


def analyze_contours_orientation(contours):
    """
    Analyzes the average orientation across all contours in a depth map.
    
    This function provides a global measure of directional consistency in the
    scene, which is important for understanding how geometric elements guide
    player navigation and attention.
    
    Args:
        contours (list): List of contour arrays from depth map analysis
        
    Returns:
        float: Average contour orientation in degrees, or NaN if no valid contours
        
    Note:
        The average orientation provides insights into the dominant directional
        patterns that may serve as navigational cues in level design.
    """
    if not contours:
        return np.nan
    angles = []
    for c in contours:
        ang = compute_contour_orientation(c)
        if not np.isnan(ang):
            angles.append(ang)
    if len(angles) == 0:
        return np.nan
    return np.mean(angles)


##############################################################################
# TRANSITION QUALITY ASSESSMENT FUNCTIONS
##############################################################################

def analyze_two_images(depth_before, depth_after, max_depth_for_hist=1.0, bins=50):
    """
    Analyzes the quality of transition between two consecutive depth frames.
    
    This function implements multiple metrics for assessing transition quality:
    - Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
    - Structural Similarity Index (SSIM) for perceptual quality
    - Coverage analysis for spatial consistency
    - Histogram differences for distribution changes
    
    Args:
        depth_before (numpy.ndarray): Depth map of the first frame
        depth_after (numpy.ndarray): Depth map of the second frame
        max_depth_for_hist (float): Maximum depth value for histogram analysis
        bins (int): Number of histogram bins
        
    Returns:
        tuple: (mae, mse, rmse, coverage_before, coverage_after, ssim_value, 
                ssim_map, hist_diff, mean_before, mean_after)
        
    References:
        SSIM analysis is based on Wang et al.'s structural similarity index,
        which is widely used in image quality assessment and computer vision.
    """
    if depth_before.shape != depth_after.shape:
        raise ValueError(f"Images differ in shape: {depth_before.shape} vs {depth_after.shape}")

    total_pixels = depth_before.size

    # Compute coverage metrics for spatial consistency
    valid_before = (depth_before > 0) & (~np.isnan(depth_before))
    valid_after = (depth_after > 0) & (~np.isnan(depth_after))
    coverage_before = np.count_nonzero(valid_before) / total_pixels
    coverage_after = np.count_nonzero(valid_after) / total_pixels

    # Compute mean depth values
    mean_before = np.nanmean(depth_before)
    mean_after = np.nanmean(depth_after)

    # Compute error metrics for overlapping valid regions
    both_valid = valid_before & valid_after
    if np.count_nonzero(both_valid) == 0:
        raise ValueError("No overlapping valid pixels for comparison.")

    diff = depth_after[both_valid] - depth_before[both_valid]
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff ** 2)
    rmse = math.sqrt(mse)

    # Compute histogram differences for distribution analysis
    hist_before, _ = np.histogram(depth_before[valid_before], bins=bins, range=(0, max_depth_for_hist))
    hist_after, _ = np.histogram(depth_after[valid_after], bins=bins, range=(0, max_depth_for_hist))
    
    # Normalize histograms for comparison
    hist_before = hist_before.astype(float) / (np.sum(hist_before) + 1e-8)
    hist_after = hist_after.astype(float) / (np.sum(hist_after) + 1e-8)
    hist_diff = np.mean(np.abs(hist_after - hist_before))

    # Compute SSIM for perceptual quality assessment
    if np.count_nonzero(both_valid) > 0:
        ssim_value, ssim_map = ssim(depth_before, depth_after, 
                                   data_range=depth_after.max() - depth_after.min(),
                                   full=True)
    else:
        ssim_value, ssim_map = np.nan, np.full_like(depth_before, np.nan)

    return (mae, mse, rmse, coverage_before, coverage_after, 
            ssim_value, ssim_map, hist_diff, mean_before, mean_after)


# Additional functions would continue here with similar academic documentation...
# For brevity, I'm showing the key functions with academic comments
