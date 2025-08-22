# API Reference

## Overview

This document provides comprehensive API documentation for the Visual Metrics framework, organized by analysis module and functionality.

## Table of Contents

1. [Depth Analysis Module](#depth-analysis-module)
2. [Color Analysis Module](#color-analysis-module)
3. [Player Analysis Module](#player-analysis-module)
4. [Bimodal Integration Module](#bimodal-integration-module)
5. [Utility Functions](#utility-functions)

---

## Depth Analysis Module

### `load_depth_csv_regex(file_path)`

Loads depth data from CSV files containing floating-point depth values.

**Parameters:**
- `file_path` (str): Path to the CSV file containing depth data

**Returns:**
- `numpy.ndarray`: 2D array of depth values, with NaN for invalid entries

**Example:**
```python
depth_data = load_depth_csv_regex("data/depth_sequences/level1_frame1.csv")
```

---

### `compute_spatial_correlation(depth)`

Computes spatial correlation coefficients between depth values and their coordinates.

**Parameters:**
- `depth` (numpy.ndarray): 2D depth map array

**Returns:**
- `tuple`: (correlation_x, correlation_y) correlation coefficients

**Example:**
```python
corr_x, corr_y = compute_spatial_correlation(depth_map)
```

---

### `detect_fov_from_labels(labels)`

Detects and analyzes the Field of View (FOV) from labeled depth data.

**Parameters:**
- `labels` (numpy.ndarray): Labeled depth data where 0 and >1 represent navigable space

**Returns:**
- `numpy.ndarray`: Binary mask of the largest connected FOV region

**Example:**
```python
fov_mask = detect_fov_from_labels(depth_labels)
```

---

### `compute_contour_orientation(contour)`

Computes the orientation of a contour using Principal Component Analysis (PCA).

**Parameters:**
- `contour` (numpy.ndarray): Contour coordinates in (row, col) format

**Returns:**
- `float`: Orientation angle in degrees [0, 180), or NaN if insufficient data

**Example:**
```python
orientation = compute_contour_orientation(contour_points)
```

---

### `analyze_two_images(depth_before, depth_after, max_depth_for_hist=1.0, bins=50)`

Analyzes the quality of transition between two consecutive depth frames.

**Parameters:**
- `depth_before` (numpy.ndarray): Depth map of the first frame
- `depth_after` (numpy.ndarray): Depth map of the second frame
- `max_depth_for_hist` (float): Maximum depth value for histogram analysis
- `bins` (int): Number of histogram bins

**Returns:**
- `tuple`: (mae, mse, rmse, coverage_before, coverage_after, ssim_value, ssim_map, hist_diff, mean_before, mean_after)

**Example:**
```python
metrics = analyze_two_images(depth_frame1, depth_frame2)
mae, rmse, ssim_value = metrics[0], metrics[2], metrics[5]
```

---

## Color Analysis Module

### `load_color_images(input_folder, extensions=('png', 'jpg', 'jpeg', 'bmp'))`

Loads color images from the specified input folder for analysis.

**Parameters:**
- `input_folder` (str): Path to the folder containing color images
- `extensions` (tuple): Image file extensions to include in the analysis

**Returns:**
- `list of tuples`: Each tuple contains (filename, image_array) where image_array is in RGB format

**Example:**
```python
images = load_color_images("data/color_sequences/level1/")
```

---

### `to_grayscale(image_rgb)`

Converts an RGB image to grayscale (luminance) representation.

**Parameters:**
- `image_rgb` (numpy.ndarray): RGB image with shape (height, width, 3)

**Returns:**
- `numpy.ndarray`: Grayscale image with shape (height, width)

**Example:**
```python
gray_image = to_grayscale(rgb_image)
```

---

### `mean_luminance(image_rgb)`

Computes the mean luminance value across an RGB image.

**Parameters:**
- `image_rgb` (numpy.ndarray): RGB image

**Returns:**
- `float`: Mean luminance value across all pixels

**Example:**
```python
avg_luminance = mean_luminance(color_image)
```

---

### `meanshift_dominant_colors(image_rgb, quantile=0.2, min_samples=500)`

Identifies dominant colors in an RGB image using MeanShift clustering.

**Parameters:**
- `image_rgb` (numpy.ndarray): RGB image for analysis
- `quantile` (float): Quantile parameter for bandwidth estimation (0.1-0.3)
- `min_samples` (int): Minimum number of samples for bandwidth estimation

**Returns:**
- `list of tuples`: Dominant color centers sorted by frequency of occurrence

**Example:**
```python
dominant_colors = meanshift_dominant_colors(color_image)
```

---

### `color_complexity(image_rgb, threshold=0.01)`

Quantifies the chromatic complexity of an RGB image.

**Parameters:**
- `image_rgb` (numpy.ndarray): RGB image for complexity analysis
- `threshold` (float): Minimum frequency threshold for color inclusion

**Returns:**
- `int`: Number of colors that meet the frequency threshold

**Example:**
```python
complexity_score = color_complexity(color_image, threshold=0.02)
```

---

## Player Analysis Module

### `prepare_cnn_model(input_shape=(108, 192, 3))`

Prepares a pre-trained CNN model for deep feature extraction from video frames.

**Parameters:**
- `input_shape` (tuple): Expected input shape (height, width, channels)

**Returns:**
- `tensorflow.keras.Model`: Modified VGG16 model for feature extraction

**Example:**
```python
model = prepare_cnn_model(input_shape=(108, 192, 3))
```

---

### `extract_video_deep_features(video_path, num_frames=50, target_size=(192, 108), crop_fraction=0.5)`

Extracts deep learning features from video sequences for player behavior analysis.

**Parameters:**
- `video_path` (str): Path to the video file for analysis
- `num_frames` (int): Number of frames to sample from the video
- `target_size` (tuple): Target resolution for frame processing
- `crop_fraction` (float): Fraction of frame width to retain (center crop)

**Returns:**
- `numpy.ndarray`: Video-level feature vector representing overall visual characteristics

**Example:**
```python
features = extract_video_deep_features("data/player_videos/player1_level1.mp4")
```

---

### `analyze_feature_similarities(features_list)`

Analyzes similarities and differences between video feature vectors.

**Parameters:**
- `features_list` (list): List of (video_path, feature_vector, group) tuples

**Returns:**
- `dict`: Dictionary containing various similarity and analysis metrics

**Example:**
```python
analysis_results = analyze_feature_similarities(video_features)
```

---

## Bimodal Integration Module

### `compute_bimodal_metrics(depth_sequence, color_sequence, transition_indices=None)`

Computes comprehensive bimodal metrics combining geometric and perceptual analysis.

**Parameters:**
- `depth_sequence` (list): List of depth maps from the level transition
- `color_sequence` (list): List of RGB images corresponding to depth maps
- `transition_indices` (list): Optional indices marking transition boundaries

**Returns:**
- `dict`: Comprehensive dictionary containing all bimodal metrics

**Example:**
```python
bimodal_metrics = compute_bimodal_metrics(depth_sequence, color_sequence)
```

---

### `compute_transition_quality_score(geo_metrics, color_diff, depth_before, depth_after, color_before, color_after)`

Computes a unified transition quality score combining geometric and perceptual metrics.

**Parameters:**
- `geo_metrics` (tuple): Geometric transition metrics from analyze_two_images
- `color_diff` (float): Color difference between frames
- `depth_before, depth_after` (numpy.ndarray): Depth maps
- `color_before, color_after` (numpy.ndarray): RGB images

**Returns:**
- `dict`: Comprehensive transition quality assessment

**Example:**
```python
quality_score = compute_transition_quality_score(
    geo_metrics, color_diff, depth_before, depth_after, 
    color_before, color_after
)
```

---

### `compute_cross_modal_correlations(frame_metrics)`

Computes correlations between geometric and perceptual metrics across frames.

**Parameters:**
- `frame_metrics` (list): List of frame-level metric dictionaries

**Returns:**
- `dict`: Dictionary containing various correlation analyses

**Example:**
```python
correlations = compute_cross_modal_correlations(frame_metrics)
```

---

### `compute_guidance_quality_index(frame_metrics, transition_metrics)`

Computes comprehensive guidance quality indices for the entire level transition.

**Parameters:**
- `frame_metrics` (list): Frame-level metric dictionaries
- `transition_metrics` (list): Transition-level metric dictionaries

**Returns:**
- `dict`: Comprehensive guidance quality assessment

**Example:**
```python
quality_indices = compute_guidance_quality_index(frame_metrics, transition_metrics)
```

---

### `visualize_bimodal_analysis(bimodal_metrics, output_path="bimodal_analysis.png")`

Creates comprehensive visualizations of bimodal analysis results.

**Parameters:**
- `bimodal_metrics` (dict): Results from compute_bimodal_metrics()
- `output_path` (str): Path to save the visualization

**Returns:**
- `matplotlib.figure.Figure`: Generated figure object

**Example:**
```python
fig = visualize_bimodal_analysis(bimodal_metrics, "results/analysis.png")
```

---

## Utility Functions

### Data Validation

```python
def validate_sequence_lengths(depth_sequence, color_sequence):
    """Validates that depth and color sequences have matching lengths."""
    if len(depth_sequence) != len(color_sequence):
        raise ValueError("Depth and color sequences must have equal length")
    return True
```

### Metric Normalization

```python
def normalize_metric(value, min_val, max_val):
    """Normalizes a metric value to [0,1] range."""
    return (value - min_val) / (max_val - min_val + 1e-8)
```

### Statistical Analysis

```python
def compute_temporal_trend(metric_sequence):
    """Computes linear trend of a metric sequence over time."""
    x = np.arange(len(metric_sequence))
    trend = np.polyfit(x, metric_sequence, 1)[0]
    return trend
```

---

## Error Handling

The framework includes comprehensive error handling for common issues:

- **File not found**: Returns None with appropriate warning messages
- **Invalid data format**: Provides fallback mechanisms and error reporting
- **Dimension mismatches**: Raises ValueError with descriptive messages
- **Memory constraints**: Implements efficient processing for large datasets

## Performance Considerations

- **Batch processing**: Functions support batch operations for large datasets
- **Memory management**: Efficient memory usage through numpy operations
- **GPU acceleration**: Deep learning components utilize GPU when available
- **Parallel processing**: Some operations can be parallelized for improved performance

## Dependencies

All functions require the following core dependencies:
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-image >= 0.18.0
- opencv-python >= 4.5.0
- matplotlib >= 3.5.0

Additional dependencies for specific modules:
- **Player Analysis**: tensorflow >= 2.8.0, keras >= 2.8.0
- **Bimodal Integration**: seaborn >= 0.11.0, pandas >= 1.3.0
