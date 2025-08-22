"""
Bimodal Integration Analysis for Visual Guidance in Level Transitions
===================================================================

This module implements the core innovation of our research: the integration of geometric
(depth) and perceptual (color) analysis to provide comprehensive understanding of visual
guidance effectiveness during level transitions.

The bimodal approach addresses the key limitation of previous work by:
- Separating geometric constraints (from depth maps) and perceptual cues (from RGB sequences)
- Quantifying the synergy between spatial and chromatic guidance mechanisms
- Providing unified metrics for level transition quality assessment
- Enabling real-time design feedback and procedural generation objectives

Author: Kaijie Xu, Clark Verbrugge
Institution: McGill University, Department of Computer Science
Paper: "Quantitative Analysis of Visual Guidance in Level Transitions Using Multimodal Visual Metrics"
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

# Import our analysis modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from depth_analysis.depth_metrics_analysis import (
    compute_spatial_correlation, detect_fov_from_labels, 
    analyze_contours_orientation, analyze_two_images
)
from color_analysis.color_metrics_analysis import (
    mean_luminance, color_difference_metric, 
    meanshift_dominant_colors, color_complexity
)


##############################################################################
# BIMODAL METRIC COMPUTATION
##############################################################################

def compute_bimodal_metrics(depth_sequence, color_sequence, transition_indices=None):
    """
    Computes comprehensive bimodal metrics combining geometric and perceptual analysis.
    
    This function implements the core innovation of our research by integrating
    depth-based geometric constraints with color-based perceptual cues. The
    resulting metrics provide a unified understanding of visual guidance
    effectiveness that cannot be achieved through unimodal analysis alone.
    
    Args:
        depth_sequence (list): List of depth maps from the level transition
        color_sequence (list): List of RGB images corresponding to depth maps
        transition_indices (list): Optional indices marking transition boundaries
        
    Returns:
        dict: Comprehensive dictionary containing all bimodal metrics
        
    Bimodal Metrics:
        - Geometric-Perceptual Correlation: Measures how well spatial and chromatic
          cues align during transitions
        - Transition Quality Score: Unified assessment combining both modalities
        - Guidance Consistency Index: Quantifies the coherence of visual guidance
        - Cross-Modal Feature Analysis: Identifies synergistic guidance mechanisms
    """
    if len(depth_sequence) != len(color_sequence):
        raise ValueError("Depth and color sequences must have equal length")
    
    n_frames = len(depth_sequence)
    
    # Initialize metric storage
    bimodal_metrics = {
        'frame_metrics': [],
        'transition_metrics': [],
        'correlation_analysis': {},
        'guidance_quality': {}
    }
    
    # Compute frame-level bimodal metrics
    for i in range(n_frames):
        depth_frame = depth_sequence[i]
        color_frame = color_sequence[i]
        
        # Geometric metrics (depth-based)
        spatial_corr_x, spatial_corr_y = compute_spatial_correlation(depth_frame)
        mean_depth = np.nanmean(depth_frame)
        depth_coverage = np.count_nonzero(~np.isnan(depth_frame)) / depth_frame.size
        
        # Perceptual metrics (color-based)
        mean_lum = mean_luminance(color_frame)
        color_comp = color_complexity(color_frame)
        dominant_colors = meanshift_dominant_colors(color_frame)
        dom_color_count = len(dominant_colors)
        
        # Store frame-level metrics
        frame_metric = {
            'frame_index': i,
            'geometric': {
                'spatial_corr_x': spatial_corr_x,
                'spatial_corr_y': spatial_corr_y,
                'mean_depth': mean_depth,
                'depth_coverage': depth_coverage
            },
            'perceptual': {
                'mean_luminance': mean_lum,
                'color_complexity': color_comp,
                'dominant_color_count': dom_color_count
            }
        }
        bimodal_metrics['frame_metrics'].append(frame_metric)
    
    # Compute transition-level metrics
    if transition_indices is None:
        # Default: analyze consecutive frame pairs
        transition_indices = list(range(n_frames - 1))
    
    for i in transition_indices:
        if i + 1 >= n_frames:
            continue
            
        # Geometric transition analysis
        depth_before = depth_sequence[i]
        depth_after = depth_sequence[i + 1]
        geo_metrics = analyze_two_images(depth_before, depth_after)
        
        # Perceptual transition analysis
        color_before = color_sequence[i]
        color_after = color_sequence[i + 1]
        color_diff = color_difference_metric(color_before, color_after)
        
        # Bimodal transition quality
        transition_quality = compute_transition_quality_score(
            geo_metrics, color_diff, depth_before, depth_after,
            color_before, color_after
        )
        
        transition_metric = {
            'transition_index': i,
            'geometric_transition': geo_metrics,
            'perceptual_transition': {'color_difference': color_diff},
            'bimodal_quality': transition_quality
        }
        bimodal_metrics['transition_metrics'].append(transition_metric)
    
    # Compute cross-modal correlations
    bimodal_metrics['correlation_analysis'] = compute_cross_modal_correlations(
        bimodal_metrics['frame_metrics']
    )
    
    # Compute overall guidance quality
    bimodal_metrics['guidance_quality'] = compute_guidance_quality_index(
        bimodal_metrics['frame_metrics'], bimodal_metrics['transition_metrics']
    )
    
    return bimodal_metrics


def compute_transition_quality_score(geo_metrics, color_diff, depth_before, depth_after,
                                   color_before, color_after):
    """
    Computes a unified transition quality score combining geometric and perceptual metrics.
    
    This function implements the core innovation of our bimodal approach by
    creating a single quality metric that reflects both spatial consistency
    and chromatic coherence during level transitions.
    
    Args:
        geo_metrics (tuple): Geometric transition metrics from analyze_two_images
        color_diff (float): Color difference between frames
        depth_before, depth_after (numpy.ndarray): Depth maps
        color_before, color_after (numpy.ndarray): RGB images
        
    Returns:
        dict: Comprehensive transition quality assessment
        
    Quality Components:
        - Spatial Consistency: How well geometric structure is maintained
        - Chromatic Coherence: How smoothly color information transitions
        - Bimodal Alignment: How well spatial and chromatic cues work together
        - Overall Quality Score: Unified metric for transition assessment
    """
    # Extract geometric metrics
    mae, mse, rmse, coverage_before, coverage_after, ssim_value, _, hist_diff, mean_before, mean_after = geo_metrics
    
    # Compute spatial consistency score (0-1, higher is better)
    spatial_consistency = 1.0 / (1.0 + rmse)  # Normalize RMSE
    
    # Compute chromatic coherence score (0-1, higher is better)
    # Normalize color difference to [0,1] range (assuming max difference is 255)
    chromatic_coherence = 1.0 - (color_diff / 255.0)
    chromatic_coherence = np.clip(chromatic_coherence, 0.0, 1.0)
    
    # Compute coverage consistency
    coverage_consistency = 1.0 - abs(coverage_after - coverage_before)
    
    # Compute SSIM-based quality
    ssim_quality = ssim_value if not np.isnan(ssim_value) else 0.5
    
    # Compute bimodal alignment score
    # This measures how well geometric and chromatic changes correlate
    bimodal_alignment = compute_bimodal_alignment(
        depth_before, depth_after, color_before, color_after
    )
    
    # Compute overall quality score (weighted combination)
    weights = {
        'spatial': 0.3,
        'chromatic': 0.25,
        'coverage': 0.15,
        'ssim': 0.2,
        'bimodal': 0.1
    }
    
    overall_quality = (
        weights['spatial'] * spatial_consistency +
        weights['chromatic'] * chromatic_coherence +
        weights['coverage'] * coverage_consistency +
        weights['ssim'] * ssim_quality +
        weights['bimodal'] * bimodal_alignment
    )
    
    return {
        'spatial_consistency': spatial_consistency,
        'chromatic_coherence': chromatic_coherence,
        'coverage_consistency': coverage_consistency,
        'ssim_quality': ssim_quality,
        'bimodal_alignment': bimodal_alignment,
        'overall_quality': overall_quality,
        'geometric_metrics': {
            'mae': mae, 'rmse': rmse, 'ssim': ssim_value, 'hist_diff': hist_diff
        },
        'perceptual_metrics': {
            'color_difference': color_diff,
            'luminance_change': abs(mean_luminance(color_after) - mean_luminance(color_before))
        }
    }


def compute_bimodal_alignment(depth_before, depth_after, color_before, color_after):
    """
    Computes the alignment between geometric and chromatic changes during transitions.
    
    This function quantifies how well spatial and perceptual cues work together
    to guide player navigation. High alignment indicates synergistic guidance,
    while low alignment may suggest conflicting or ineffective visual cues.
    
    Args:
        depth_before, depth_after (numpy.ndarray): Depth maps
        color_before, color_after (numpy.ndarray): RGB images
        
    Returns:
        float: Alignment score between 0 and 1 (higher is better)
        
    Alignment Computation:
        1. Compute spatial change magnitude (depth differences)
        2. Compute chromatic change magnitude (color differences)
        3. Correlate spatial and chromatic changes across the image
        4. Normalize to [0,1] range for quality assessment
    """
    # Compute spatial changes (depth differences)
    valid_depth = ~np.isnan(depth_before) & ~np.isnan(depth_after)
    if not np.any(valid_depth):
        return 0.5  # Neutral score for invalid data
    
    depth_diff = np.abs(depth_after - depth_before)
    depth_diff_valid = depth_diff[valid_depth]
    
    # Compute chromatic changes (luminance differences)
    color_before_gray = cv2.cvtColor(color_before, cv2.COLOR_RGB2GRAY)
    color_after_gray = cv2.cvtColor(color_after, cv2.COLOR_RGB2GRAY)
    color_diff = np.abs(color_after_gray.astype(float) - color_before_gray.astype(float))
    color_diff_valid = color_diff[valid_depth]
    
    # Normalize both differences to [0,1] range
    depth_diff_norm = (depth_diff_valid - depth_diff_valid.min()) / (depth_diff_valid.max() - depth_diff_valid.min() + 1e-8)
    color_diff_norm = (color_diff_valid - color_diff_valid.min()) / (color_diff_valid.max() - color_diff_valid.min() + 1e-8)
    
    # Compute correlation between spatial and chromatic changes
    if len(depth_diff_norm) > 1:
        correlation, _ = pearsonr(depth_diff_norm, color_diff_norm)
        # Convert correlation [-1,1] to alignment score [0,1]
        alignment = (correlation + 1) / 2
    else:
        alignment = 0.5
    
    return alignment


##############################################################################
# CROSS-MODAL CORRELATION ANALYSIS
##############################################################################

def compute_cross_modal_correlations(frame_metrics):
    """
    Computes correlations between geometric and perceptual metrics across frames.
    
    This analysis reveals how well spatial and chromatic guidance mechanisms
    work together throughout the level transition sequence. Strong correlations
    indicate coordinated guidance, while weak correlations may suggest
    independent or conflicting visual cues.
    
    Args:
        frame_metrics (list): List of frame-level metric dictionaries
        
    Returns:
        dict: Dictionary containing various correlation analyses
        
    Correlation Types:
        - Temporal correlations: How metrics evolve together over time
        - Cross-metric correlations: How different metric types relate
        - Transition correlations: How well metrics predict transition quality
    """
    if not frame_metrics:
        return {}
    
    # Extract metric arrays
    n_frames = len(frame_metrics)
    
    # Geometric metrics
    spatial_corr_x = [m['geometric']['spatial_corr_x'] for m in frame_metrics]
    spatial_corr_y = [m['geometric']['spatial_corr_y'] for m in frame_metrics]
    mean_depth = [m['geometric']['mean_depth'] for m in frame_metrics]
    depth_coverage = [m['geometric']['depth_coverage'] for m in frame_metrics]
    
    # Perceptual metrics
    mean_luminance = [m['perceptual']['mean_luminance'] for m in frame_metrics]
    color_complexity = [m['perceptual']['color_complexity'] for m in frame_metrics]
    dominant_color_count = [m['perceptual']['dominant_color_count'] for m in frame_metrics]
    
    # Filter out NaN values for correlation computation
    valid_indices = []
    for i in range(n_frames):
        if (not np.isnan(spatial_corr_x[i]) and not np.isnan(spatial_corr_y[i]) and
            not np.isnan(mean_depth[i]) and not np.isnan(mean_luminance[i])):
            valid_indices.append(i)
    
    if len(valid_indices) < 2:
        return {'error': 'Insufficient valid data for correlation analysis'}
    
    # Compute cross-modal correlations
    correlations = {}
    
    # Spatial correlation vs. luminance
    valid_x = [spatial_corr_x[i] for i in valid_indices]
    valid_y = [spatial_corr_y[i] for i in valid_indices]
    valid_lum = [mean_luminance[i] for i in valid_indices]
    
    corr_x_lum, p_x_lum = pearsonr(valid_x, valid_lum)
    corr_y_lum, p_y_lum = pearsonr(valid_y, valid_lum)
    
    correlations['spatial_luminance'] = {
        'corr_x': corr_x_lum, 'p_x': p_x_lum,
        'corr_y': corr_y_lum, 'p_y': p_y_lum
    }
    
    # Depth vs. color complexity
    valid_depth = [mean_depth[i] for i in valid_indices]
    valid_comp = [color_complexity[i] for i in valid_indices]
    
    corr_depth_comp, p_depth_comp = pearsonr(valid_depth, valid_comp)
    correlations['depth_color_complexity'] = {
        'correlation': corr_depth_comp, 'p_value': p_depth_comp
    }
    
    # Coverage vs. dominant colors
    valid_coverage = [depth_coverage[i] for i in valid_indices]
    valid_colors = [dominant_color_count[i] for i in valid_indices]
    
    corr_coverage_colors, p_coverage_colors = pearsonr(valid_coverage, valid_colors)
    correlations['coverage_dominant_colors'] = {
        'correlation': corr_coverage_colors, 'p_value': p_coverage_colors
    }
    
    # Temporal correlations (how metrics evolve over time)
    frame_indices = list(range(n_frames))
    correlations['temporal_evolution'] = {
        'spatial_corr_x_trend': np.polyfit(frame_indices, spatial_corr_x, 1)[0],
        'luminance_trend': np.polyfit(frame_indices, mean_luminance, 1)[0],
        'depth_trend': np.polyfit(frame_indices, mean_depth, 1)[0],
        'color_complexity_trend': np.polyfit(frame_indices, color_complexity, 1)[0]
    }
    
    return correlations


##############################################################################
# GUIDANCE QUALITY ASSESSMENT
##############################################################################

def compute_guidance_quality_index(frame_metrics, transition_metrics):
    """
    Computes comprehensive guidance quality indices for the entire level transition.
    
    This function provides the unified assessment framework that combines
    geometric and perceptual analysis into actionable design feedback.
    The resulting indices can be used for real-time design evaluation
    and as objective functions for procedural content generation.
    
    Args:
        frame_metrics (list): Frame-level metric dictionaries
        transition_metrics (list): Transition-level metric dictionaries
        
    Returns:
        dict: Comprehensive guidance quality assessment
        
    Quality Indices:
        - Overall Guidance Effectiveness: Combined geometric and perceptual quality
        - Consistency Score: How well guidance is maintained throughout transition
        - Synergy Index: How well spatial and chromatic cues work together
        - Design Recommendations: Actionable feedback for level designers
    """
    if not frame_metrics or not transition_metrics:
        return {'error': 'Insufficient data for quality assessment'}
    
    # Compute frame-level quality scores
    frame_qualities = []
    for metrics in frame_metrics:
        # Combine geometric and perceptual metrics into frame quality
        geo_score = compute_geometric_quality_score(metrics['geometric'])
        per_score = compute_perceptual_quality_score(metrics['perceptual'])
        
        # Bimodal frame quality (geometric and perceptual should complement each other)
        frame_quality = (geo_score + per_score) / 2
        frame_qualities.append(frame_quality)
    
    # Compute transition-level quality scores
    transition_qualities = [t['bimodal_quality']['overall_quality'] for t in transition_metrics]
    
    # Overall guidance effectiveness
    overall_effectiveness = np.mean(frame_qualities)
    
    # Consistency score (how stable guidance is throughout transition)
    consistency_score = 1.0 - np.std(frame_qualities)
    consistency_score = np.clip(consistency_score, 0.0, 1.0)
    
    # Synergy index (how well geometric and perceptual cues work together)
    synergy_index = compute_synergy_index(frame_metrics, transition_metrics)
    
    # Design recommendations
    recommendations = generate_design_recommendations(
        frame_qualities, transition_qualities, frame_metrics, transition_metrics
    )
    
    return {
        'overall_effectiveness': overall_effectiveness,
        'consistency_score': consistency_score,
        'synergy_index': synergy_index,
        'frame_quality_stats': {
            'mean': np.mean(frame_qualities),
            'std': np.std(frame_qualities),
            'min': np.min(frame_qualities),
            'max': np.max(frame_qualities)
        },
        'transition_quality_stats': {
            'mean': np.mean(transition_qualities),
            'std': np.std(transition_qualities),
            'min': np.min(transition_qualities),
            'max': np.max(transition_qualities)
        },
        'design_recommendations': recommendations
    }


def compute_geometric_quality_score(geometric_metrics):
    """
    Computes a quality score for geometric guidance based on depth analysis.
    
    Args:
        geometric_metrics (dict): Geometric metric dictionary
        
    Returns:
        float: Quality score between 0 and 1
    """
    # Normalize and combine geometric metrics
    spatial_corr_x = geometric_metrics['spatial_corr_x']
    spatial_corr_y = geometric_metrics['spatial_corr_y']
    mean_depth = geometric_metrics['mean_depth']
    depth_coverage = geometric_metrics['depth_coverage']
    
    # Convert correlations to quality scores (absolute values, normalized)
    corr_quality_x = abs(spatial_corr_x) if not np.isnan(spatial_corr_x) else 0.0
    corr_quality_y = abs(spatial_corr_y) if not np.isnan(spatial_corr_y) else 0.0
    
    # Coverage quality (higher is better)
    coverage_quality = depth_coverage
    
    # Depth quality (assume moderate depth is optimal for guidance)
    # This is a simplified assumption - could be refined based on design principles
    depth_quality = 1.0 - abs(mean_depth - 0.5) if not np.isnan(mean_depth) else 0.5
    
    # Combine metrics with weights
    weights = {'corr_x': 0.3, 'corr_y': 0.3, 'coverage': 0.25, 'depth': 0.15}
    quality = (
        weights['corr_x'] * corr_quality_x +
        weights['corr_y'] * corr_quality_y +
        weights['coverage'] * coverage_quality +
        weights['depth'] * depth_quality
    )
    
    return quality


def compute_perceptual_quality_score(perceptual_metrics):
    """
    Computes a quality score for perceptual guidance based on color analysis.
    
    Args:
        perceptual_metrics (dict): Perceptual metric dictionary
        
    Returns:
        float: Quality score between 0 and 1
    """
    mean_lum = perceptual_metrics['mean_luminance']
    color_comp = perceptual_metrics['color_complexity']
    dom_color_count = perceptual_metrics['dominant_color_count']
    
    # Luminance quality (assume moderate brightness is optimal)
    # This could be refined based on specific design requirements
    lum_quality = 1.0 - abs(mean_lum - 128) / 128
    
    # Color complexity quality (assume moderate complexity is optimal)
    # Too simple: boring, too complex: overwhelming
    comp_quality = 1.0 - abs(color_comp - 10) / 20  # Assume 10 is optimal
    comp_quality = np.clip(comp_quality, 0.0, 1.0)
    
    # Dominant color quality (assume moderate number is optimal)
    color_quality = 1.0 - abs(dom_color_count - 5) / 10  # Assume 5 is optimal
    color_quality = np.clip(color_quality, 0.0, 1.0)
    
    # Combine metrics with weights
    weights = {'luminance': 0.4, 'complexity': 0.35, 'dominant_colors': 0.25}
    quality = (
        weights['luminance'] * lum_quality +
        weights['complexity'] * comp_quality +
        weights['dominant_colors'] * color_quality
    )
    
    return quality


def compute_synergy_index(frame_metrics, transition_metrics):
    """
    Computes how well geometric and perceptual cues work together.
    
    Args:
        frame_metrics (list): Frame-level metrics
        transition_metrics (list): Transition-level metrics
        
    Returns:
        float: Synergy index between 0 and 1
    """
    if not frame_metrics or not transition_metrics:
        return 0.5
    
    # Extract geometric and perceptual quality scores
    geo_qualities = []
    per_qualities = []
    
    for metrics in frame_metrics:
        geo_score = compute_geometric_quality_score(metrics['geometric'])
        per_score = compute_perceptual_quality_score(metrics['perceptual'])
        geo_qualities.append(geo_score)
        per_qualities.append(per_score)
    
    # Compute correlation between geometric and perceptual quality
    if len(geo_qualities) > 1:
        correlation, _ = pearsonr(geo_qualities, per_qualities)
        # Convert to synergy index [0,1]
        synergy = (correlation + 1) / 2
        synergy = np.clip(synergy, 0.0, 1.0)
    else:
        synergy = 0.5
    
    return synergy


def generate_design_recommendations(frame_qualities, transition_qualities, 
                                  frame_metrics, transition_metrics):
    """
    Generates actionable design recommendations based on bimodal analysis.
    
    Args:
        frame_qualities (list): Frame-level quality scores
        transition_qualities (list): Transition-level quality scores
        frame_metrics (list): Frame-level metrics
        transition_metrics (list): Transition-level metrics
        
    Returns:
        list: List of design recommendations
    """
    recommendations = []
    
    # Analyze overall quality
    mean_frame_quality = np.mean(frame_qualities)
    mean_transition_quality = np.mean(transition_qualities)
    
    if mean_frame_quality < 0.6:
        recommendations.append({
            'type': 'warning',
            'message': 'Overall visual guidance quality is below optimal levels',
            'suggestion': 'Review spatial and chromatic design elements for consistency'
        })
    
    if mean_transition_quality < 0.6:
        recommendations.append({
            'type': 'warning',
            'message': 'Level transitions show poor quality characteristics',
            'suggestion': 'Improve transition smoothness and visual continuity'
        })
    
    # Analyze consistency
    frame_std = np.std(frame_qualities)
    if frame_std > 0.2:
        recommendations.append({
            'type': 'info',
            'message': 'Visual guidance quality varies significantly across frames',
            'suggestion': 'Ensure consistent application of guidance principles throughout'
        })
    
    # Analyze specific metrics for detailed recommendations
    if frame_metrics:
        # Check for common issues
        low_coverage_frames = [i for i, m in enumerate(frame_metrics) 
                              if m['geometric']['depth_coverage'] < 0.7]
        if low_coverage_frames:
            recommendations.append({
                'type': 'info',
                'message': f'Frames {low_coverage_frames} show low spatial coverage',
                'suggestion': 'Review geometric constraints and ensure adequate navigable space'
            })
    
    return recommendations


##############################################################################
# VISUALIZATION AND REPORTING
##############################################################################

def visualize_bimodal_analysis(bimodal_metrics, output_path="bimodal_analysis.png"):
    """
    Creates comprehensive visualizations of bimodal analysis results.
    
    Args:
        bimodal_metrics (dict): Results from compute_bimodal_metrics()
        output_path (str): Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: Generated figure object
    """
    if not bimodal_metrics:
        print("No bimodal metrics to visualize.")
        return None
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Frame-level metric evolution
    ax1 = plt.subplot(2, 3, 1)
    frame_metrics = bimodal_metrics['frame_metrics']
    
    if frame_metrics:
        frame_indices = [m['frame_index'] for m in frame_metrics]
        
        # Geometric metrics
        spatial_corr_x = [m['geometric']['spatial_corr_x'] for m in frame_metrics]
        spatial_corr_y = [m['geometric']['spatial_corr_y'] for m in frame_metrics]
        
        ax1.plot(frame_indices, spatial_corr_x, 'b-', label='Spatial Corr X', alpha=0.7)
        ax1.plot(frame_indices, spatial_corr_y, 'g-', label='Spatial Corr Y', alpha=0.7)
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Spatial Correlation')
        ax1.set_title('Geometric Metric Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Perceptual metric evolution
    ax2 = plt.subplot(2, 3, 2)
    if frame_metrics:
        mean_lum = [m['perceptual']['mean_luminance'] for m in frame_metrics]
        color_comp = [m['perceptual']['color_complexity'] for m in frame_metrics]
        
        ax2.plot(frame_indices, mean_lum, 'r-', label='Mean Luminance', alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(frame_indices, color_comp, 'm-', label='Color Complexity', alpha=0.7)
        
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Luminance', color='r')
        ax2_twin.set_ylabel('Color Complexity', color='m')
        ax2.set_title('Perceptual Metric Evolution')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    
    # 3. Transition quality scores
    ax3 = plt.subplot(2, 3, 3)
    transition_metrics = bimodal_metrics['transition_metrics']
    
    if transition_metrics:
        trans_indices = [t['transition_index'] for t in transition_metrics]
        overall_quality = [t['bimodal_quality']['overall_quality'] for t in transition_metrics]
        spatial_quality = [t['bimodal_quality']['spatial_consistency'] for t in transition_metrics]
        chromatic_quality = [t['bimodal_quality']['chromatic_coherence'] for t in transition_metrics]
        
        ax3.plot(trans_indices, overall_quality, 'k-', label='Overall Quality', linewidth=2)
        ax3.plot(trans_indices, spatial_quality, 'b--', label='Spatial Quality', alpha=0.7)
        ax3.plot(trans_indices, chromatic_quality, 'r--', label='Chromatic Quality', alpha=0.7)
        
        ax3.set_xlabel('Transition Index')
        ax3.set_ylabel('Quality Score')
        ax3.set_title('Transition Quality Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Correlation analysis heatmap
    ax4 = plt.subplot(2, 3, 4)
    correlation_analysis = bimodal_metrics['correlation_analysis']
    
    if 'spatial_luminance' in correlation_analysis:
        # Create correlation matrix for visualization
        corr_data = {
            'Spatial Corr X': [correlation_analysis['spatial_luminance']['corr_x']],
            'Spatial Corr Y': [correlation_analysis['spatial_luminance']['corr_y']],
            'Depth': [correlation_analysis.get('depth_color_complexity', {}).get('correlation', 0)],
            'Coverage': [correlation_analysis.get('coverage_dominant_colors', {}).get('correlation', 0)]
        }
        
        corr_df = pd.DataFrame(corr_data, index=['Luminance'])
        sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, ax=ax4)
        ax4.set_title('Cross-Modal Correlations')
    
    # 5. Guidance quality summary
    ax5 = plt.subplot(2, 3, 5)
    guidance_quality = bimodal_metrics['guidance_quality']
    
    if 'overall_effectiveness' in guidance_quality:
        quality_metrics = ['Overall Effectiveness', 'Consistency', 'Synergy']
        quality_values = [
            guidance_quality['overall_effectiveness'],
            guidance_quality['consistency_score'],
            guidance_quality['synergy_index']
        ]
        
        bars = ax5.bar(quality_metrics, quality_values, alpha=0.7)
        ax5.set_ylabel('Quality Score')
        ax5.set_title('Guidance Quality Summary')
        ax5.set_ylim(0, 1)
        
        # Color code bars
        for i, (bar, value) in enumerate(zip(bars, quality_values)):
            if value < 0.6:
                bar.set_color('red')
            elif value < 0.8:
                bar.set_color('orange')
            else:
                bar.set_color('green')
    
    # 6. Design recommendations
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    if 'design_recommendations' in guidance_quality:
        recommendations = guidance_quality['design_recommendations']
        if recommendations:
            recommendation_text = "Design Recommendations:\n\n"
            for i, rec in enumerate(recommendations[:5]):  # Show first 5
                recommendation_text += f"{i+1}. {rec['message']}\n"
                recommendation_text += f"   → {rec['suggestion']}\n\n"
            
            ax6.text(0.1, 0.9, recommendation_text, transform=ax6.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        else:
            ax6.text(0.5, 0.5, "No specific recommendations\n(Good overall quality)",
                     transform=ax6.transAxes, ha='center', va='center',
                     fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig


##############################################################################
# MAIN ANALYSIS PIPELINE
##############################################################################

def main():
    """
    Main execution function for bimodal analysis.
    
    This function demonstrates the complete bimodal analysis pipeline,
    showing how geometric and perceptual analysis can be integrated
    to provide comprehensive understanding of visual guidance effectiveness.
    """
    print("Bimodal Visual Guidance Analysis")
    print("=================================")
    print("This module implements the core innovation of our research:")
    print("- Integration of geometric (depth) and perceptual (color) analysis")
    print("- Unified metrics for level transition quality assessment")
    print("- Cross-modal correlation analysis for guidance effectiveness")
    print("- Actionable design recommendations for level designers")
    print("\nTo use this module:")
    print("1. Prepare depth and color sequences for your level transitions")
    print("2. Call compute_bimodal_metrics() with your data")
    print("3. Use the resulting metrics for design evaluation and improvement")
    print("4. Generate visualizations with visualize_bimodal_analysis()")
    
    # Example usage would go here
    print("\nExample usage:")
    print("```python")
    print("bimodal_metrics = compute_bimodal_metrics(depth_sequence, color_sequence)")
    print("fig = visualize_bimodal_analysis(bimodal_metrics)")
    print("```")


if __name__ == "__main__":
    main()
