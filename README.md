# Visual Metrics for Game Level Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FCOG64752.2025.11114135-blue.svg)](https://doi.org/10.1109/COG64752.2025.11114135)

## Overview

This repository contains the implementation of our research on [**Quantitative Analysis of Visual Guidance in Level Transitions Using Multimodal Visual Metrics**](https://doi.org/10.1109/COG64752.2025.11114135), presented at the 2025 IEEE Conference on Games (CoG).

Our work introduces a novel **bimodal quantitative framework** for evaluating visual guidance during level transitions in 3D role-playing games, specifically analyzing Dark Souls III. By integrating depth map analysis with RGB image sequence analysis, we provide measurable patterns for design feedback and procedural content generation.

## 🎯 Key Contributions

- **Bimodal Analysis Framework**: Separates geometric constraints (depth) from perceptual cues (color)
- **Quantitative Metrics**: Luminance dynamics, chromatic complexity, spatial depth distribution
- **Cross-Modal Integration**: Quantifies synergy between spatial and chromatic guidance mechanisms
- **Real-time Design Feedback**: Provides actionable insights for level designers
- **Procedural Generation Objectives**: Establishes evaluation criteria for automated level creation


## 🏗️ Architecture

The repository implements a modular architecture with three core analysis components:

```
src/
├── depth_analysis/          # Geometric constraints analysis
│   └── depth_metrics_analysis.py
├── color_analysis/          # Perceptual cues analysis
│   └── color_metrics_analysis.py
├── player_analysis/         # Player behavior analysis
│   └── player_behavior_analysis.py
└── bimodal_integration/     # Cross-modal analysis (Core Innovation)
    └── bimodal_analysis.py
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for deep learning components)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nortrom1213/Visual-Metrics.git
   cd Visual-Metrics
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

#### 1. Depth Analysis 
```python
from src.depth_analysis.depth_metrics_analysis import *

# Load depth sequence
depth_sequence = load_depth_csv_regex("path/to/depth_data.csv")

# Analyze spatial correlation
spatial_corr_x, spatial_corr_y = compute_spatial_correlation(depth_sequence)

# Detect field of view
fov_mask = detect_fov_from_labels(depth_labels)
```

#### 2. Color Analysis 
```python
from src.color_analysis.color_metrics_analysis import *

# Load color sequence
color_sequence = load_color_images("path/to/color_images/")

# Analyze luminance dynamics
mean_lum = mean_luminance(color_image)

# Extract dominant colors
dominant_colors = meanshift_dominant_colors(color_image)
```

#### 3. Bimodal Integration (Testing)
```python
from src.bimodal_integration.bimodal_analysis import *

# Compute comprehensive bimodal metrics
bimodal_metrics = compute_bimodal_metrics(
    depth_sequence, 
    color_sequence
)

# Generate visualizations
fig = visualize_bimodal_analysis(bimodal_metrics)

# Access guidance quality indices
overall_effectiveness = bimodal_metrics['guidance_quality']['overall_effectiveness']
synergy_index = bimodal_metrics['guidance_quality']['synergy_index']
```

## 📊 Data Structure

The framework expects data in the following format:

```
data/
├── depth_sequences/         # CSV files with depth values
│   ├── level1_frame1.csv
│   ├── level1_frame2.csv
│   └── ...
├── color_sequences/         # RGB images (PNG/JPG)
│   ├── level1_frame1.png
│   ├── level1_frame2.png
│   └── ...
└── player_videos/          # MP4 files for behavior analysis
    ├── player1_level1.mp4
    └── ...
```

## 🔬 Methodology

### Bimodal Analysis Approach

1. **Geometric Analysis (Depth Maps)**
   - Spatial correlation analysis
   - Field of View (FOV) detection
   - Contour orientation analysis
   - Object detection and spatial distribution

2. **Perceptual Analysis (RGB Sequences)**
   - Luminance dynamics and temporal variations
   - Chromatic complexity using MeanShift clustering
   - Color transition quality assessment
   - Dominant color identification

3. **Cross-Modal Integration**
   - Geometric-perceptual correlation analysis
   - Unified transition quality scoring
   - Guidance consistency assessment
   - Synergy index computation

### Key Metrics

- **Spatial Consistency**: How well geometric structure is maintained
- **Chromatic Coherence**: How smoothly color information transitions
- **Bimodal Alignment**: How well spatial and chromatic cues work together
- **Overall Quality Score**: Unified metric for transition assessment

## 🤝 Contributing

We welcome contributions to improve the framework:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{xu2025quantitative,
  title={Quantitative Analysis of Visual Guidance in Level Transitions Using Multimodal Visual Metrics},
  author={Xu, Kaijie and Verbrugge, Clark},
  booktitle={2025 IEEE Conference on Games (CoG)},
  pages={1--8},
  year={2025},
  organization={IEEE},
  doi={10.1109/COG64752.2025.11114135}
}
```