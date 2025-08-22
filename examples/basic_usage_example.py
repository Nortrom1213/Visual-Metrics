"""
Basic Usage Example for Visual Metrics Framework
===============================================

This example demonstrates the basic usage of the Visual Metrics framework
for analyzing level transitions in game design.

Author: Kaijie Xu, Clark Verbrugge
Institution: McGill University, Department of Computer Science
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our analysis modules
from depth_analysis.depth_metrics_analysis import (
    compute_spatial_correlation, detect_fov_from_labels
)
from color_analysis.color_metrics_analysis import (
    mean_luminance, color_complexity, meanshift_dominant_colors
)
from bimodal_integration.bimodal_analysis import (
    compute_bimodal_metrics, visualize_bimodal_analysis
)


def create_sample_data():
    """
    Creates sample depth and color data for demonstration purposes.
    
    In a real application, this would be replaced with actual data
    from your game level transitions.
    """
    print("Creating sample data for demonstration...")
    
    # Create sample depth sequence (simulating a level transition)
    n_frames = 10
    height, width = 100, 100
    
    depth_sequence = []
    for i in range(n_frames):
        # Simulate depth changes during transition
        base_depth = 0.5 + 0.3 * np.sin(i * np.pi / n_frames)
        depth_frame = np.random.normal(base_depth, 0.1, (height, width))
        
        # Add some spatial structure
        y, x = np.mgrid[:height, :width]
        depth_frame += 0.1 * np.sin(x * 0.1) + 0.1 * np.cos(y * 0.1)
        
        # Ensure depth values are in [0, 1] range
        depth_frame = np.clip(depth_frame, 0, 1)
        depth_sequence.append(depth_frame)
    
    # Create sample color sequence (simulating RGB images)
    color_sequence = []
    for i in range(n_frames):
        # Simulate color changes during transition
        base_luminance = 128 + 64 * np.sin(i * np.pi / n_frames)
        color_frame = np.random.normal(base_luminance, 30, (height, width, 3))
        
        # Add some color variation
        color_frame[:, :, 0] += 20 * np.sin(i * 0.5)  # Red channel
        color_frame[:, :, 1] += 20 * np.cos(i * 0.5)  # Green channel
        color_frame[:, :, 2] += 20 * np.sin(i * 0.3)  # Blue channel
        
        # Ensure color values are in [0, 255] range
        color_frame = np.clip(color_frame, 0, 255).astype(np.uint8)
        color_sequence.append(color_frame)
    
    print(f"Created {n_frames} frames of {height}x{width} data")
    return depth_sequence, color_sequence


def demonstrate_depth_analysis(depth_sequence):
    """Demonstrates depth analysis capabilities."""
    print("\n=== Depth Analysis Demonstration ===")
    
    # Analyze first frame
    depth_frame = depth_sequence[0]
    
    # Compute spatial correlation
    corr_x, corr_y = compute_spatial_correlation(depth_frame)
    print(f"Spatial correlation - X: {corr_x:.3f}, Y: {corr_y:.3f}")
    
    # Compute basic statistics
    mean_depth = np.nanmean(depth_frame)
    depth_std = np.nanstd(depth_frame)
    print(f"Depth statistics - Mean: {mean_depth:.3f}, Std: {depth_std:.3f}")
    
    # Analyze depth coverage
    valid_pixels = ~np.isnan(depth_frame)
    coverage = np.sum(valid_pixels) / depth_frame.size
    print(f"Depth coverage: {coverage:.3f}")


def demonstrate_color_analysis(color_sequence):
    """Demonstrates color analysis capabilities."""
    print("\n=== Color Analysis Demonstration ===")
    
    # Analyze first frame
    color_frame = color_sequence[0]
    
    # Compute luminance
    avg_luminance = mean_luminance(color_frame)
    print(f"Average luminance: {avg_luminance:.1f}")
    
    # Compute color complexity
    complexity = color_complexity(color_frame, threshold=0.01)
    print(f"Color complexity: {complexity}")
    
    # Extract dominant colors
    dominant_colors = meanshift_dominant_colors(color_frame, quantile=0.2)
    print(f"Number of dominant colors: {len(dominant_colors)}")
    
    if dominant_colors:
        print("Top 3 dominant colors (RGB):")
        for i, color in enumerate(dominant_colors[:3]):
            print(f"  {i+1}. RGB{color}")


def demonstrate_bimodal_integration(depth_sequence, color_sequence):
    """Demonstrates bimodal integration capabilities."""
    print("\n=== Bimodal Integration Demonstration ===")
    
    # Compute comprehensive bimodal metrics
    print("Computing bimodal metrics...")
    bimodal_metrics = compute_bimodal_metrics(depth_sequence, color_sequence)
    
    # Display key results
    guidance_quality = bimodal_metrics['guidance_quality']
    
    if 'overall_effectiveness' in guidance_quality:
        print(f"Overall guidance effectiveness: {guidance_quality['overall_effectiveness']:.3f}")
        print(f"Guidance consistency: {guidance_quality['consistency_score']:.3f}")
        print(f"Bimodal synergy index: {guidance_quality['synergy_index']:.3f}")
        
        # Display frame quality statistics
        frame_stats = guidance_quality['frame_quality_stats']
        print(f"Frame quality - Mean: {frame_stats['mean']:.3f}, Std: {frame_stats['std']:.3f}")
        
        # Display transition quality statistics
        trans_stats = guidance_quality['transition_quality_stats']
        print(f"Transition quality - Mean: {trans_stats['mean']:.3f}, Std: {trans_stats['std']:.3f}")
    
    # Generate visualization
    print("Generating visualization...")
    fig = visualize_bimodal_analysis(bimodal_metrics, "example_bimodal_analysis.png")
    print("Visualization saved as 'example_bimodal_analysis.png'")
    
    return bimodal_metrics


def demonstrate_correlation_analysis(bimodal_metrics):
    """Demonstrates correlation analysis capabilities."""
    print("\n=== Correlation Analysis Demonstration ===")
    
    correlation_analysis = bimodal_metrics['correlation_analysis']
    
    if 'spatial_luminance' in correlation_analysis:
        spatial_lum = correlation_analysis['spatial_luminance']
        print("Spatial-Luminance Correlations:")
        print(f"  X-direction: {spatial_lum['corr_x']:.3f} (p={spatial_lum['p_x']:.3f})")
        print(f"  Y-direction: {spatial_lum['corr_y']:.3f} (p={spatial_lum['p_y']:.3f})")
    
    if 'depth_color_complexity' in correlation_analysis:
        depth_comp = correlation_analysis['depth_color_complexity']
        print(f"Depth-Color Complexity: {depth_comp['correlation']:.3f} (p={depth_comp['p_value']:.3f})")
    
    if 'temporal_evolution' in correlation_analysis:
        temporal = correlation_analysis['temporal_evolution']
        print("Temporal Evolution Trends:")
        print(f"  Spatial correlation X: {temporal['spatial_corr_x_trend']:.6f}")
        print(f"  Luminance: {temporal['luminance_trend']:.6f}")
        print(f"  Depth: {temporal['depth_trend']:.6f}")
        print(f"  Color complexity: {temporal['color_complexity_trend']:.6f}")


def main():
    """Main demonstration function."""
    print("Visual Metrics Framework - Basic Usage Example")
    print("=" * 50)
    print("This example demonstrates the core capabilities of our")
    print("bimodal visual guidance analysis framework.")
    print()
    
    try:
        # Create sample data
        depth_sequence, color_sequence = create_sample_data()
        
        # Demonstrate individual analysis capabilities
        demonstrate_depth_analysis(depth_sequence)
        demonstrate_color_analysis(color_sequence)
        
        # Demonstrate bimodal integration
        bimodal_metrics = demonstrate_bimodal_integration(depth_sequence, color_sequence)
        
        # Demonstrate correlation analysis
        demonstrate_correlation_analysis(bimodal_metrics)
        
        # Display design recommendations if available
        guidance_quality = bimodal_metrics['guidance_quality']
        if 'design_recommendations' in guidance_quality:
            recommendations = guidance_quality['design_recommendations']
            if recommendations:
                print("\n=== Design Recommendations ===")
                for i, rec in enumerate(recommendations):
                    print(f"{i+1}. {rec['message']}")
                    print(f"   → {rec['suggestion']}")
            else:
                print("\n=== Design Recommendations ===")
                print("No specific recommendations (Good overall quality)")
        
        print("\n" + "=" * 50)
        print("Demonstration completed successfully!")
        print("Check 'example_bimodal_analysis.png' for visualization results.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please ensure all dependencies are installed and the framework is properly set up.")


if __name__ == "__main__":
    main()
