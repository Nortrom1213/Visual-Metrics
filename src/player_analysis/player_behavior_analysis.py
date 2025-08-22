"""
Quantitative Analysis of Player Behavior and Visual Attention in Level Transitions
===============================================================================

This module implements the player behavior analysis component of our bimodal visual guidance
framework for 3D role-playing game level design. It provides quantitative metrics for analyzing
player navigation patterns, visual attention, and behavioral responses during level transitions.

The analysis is based on video recordings of player gameplay, focusing on:
- Deep learning-based feature extraction from video sequences
- Player navigation pattern analysis across different level types
- Behavioral response quantification to visual guidance cues
- Cross-dataset validation of visual guidance effectiveness

Author: Kaijie Xu, Clark Verbrugge
Institution: McGill University, Department of Computer Science
Paper: "Quantitative Analysis of Visual Guidance in Level Transitions Using Multimodal Visual Metrics"
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


##############################################################################
# GLOBAL PARAMETERS AND CONFIGURATION
##############################################################################

# Analysis Parameters
NUM_FRAMES = 50  # Number of frames to uniformly sample from each video sequence
TARGET_SIZE = (192, 108)  # Target frame size (width, height) for normalization
CROP_FRACTION = 0.5  # Fraction of the original width to retain (center crop)

# Model Configuration
# Note: Input shape is (height, width, 3); TARGET_SIZE is (width, height)
CNN_MODEL = None  # Will be initialized in prepare_cnn_model()


##############################################################################
# DEEP LEARNING MODEL PREPARATION
##############################################################################

def prepare_cnn_model(input_shape=(108, 192, 3)):
    """
    Prepares a pre-trained CNN model for deep feature extraction from video frames.
    
    This function initializes a VGG16 model pre-trained on ImageNet and modifies it
    to extract global average pooling features. The resulting model provides a
    512-dimensional feature vector for each input frame, capturing high-level
    visual characteristics that are relevant for understanding player attention
    and visual guidance effectiveness.
    
    Args:
        input_shape (tuple): Expected input shape (height, width, channels)
        
    Returns:
        tensorflow.keras.Model: Modified VGG16 model for feature extraction
        
    References:
        VGG16 architecture is based on Simonyan and Zisserman's work on deep
        convolutional networks for large-scale image recognition. The model
        has been widely adopted in computer vision for transfer learning
        applications due to its robust feature representations.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    from tensorflow.keras.layers import GlobalAveragePooling2D
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model


# Initialize the CNN model for feature extraction
CNN_MODEL = prepare_cnn_model(input_shape=(TARGET_SIZE[1], TARGET_SIZE[0], 3))


##############################################################################
# VIDEO FEATURE EXTRACTION FUNCTIONS
##############################################################################

def extract_video_deep_features(video_path, num_frames=NUM_FRAMES, target_size=TARGET_SIZE,
                                crop_fraction=CROP_FRACTION):
    """
    Extracts deep learning features from video sequences for player behavior analysis.
    
    This function implements a comprehensive approach to video analysis by:
    1. Uniformly sampling frames across the video timeline
    2. Applying spatial cropping to focus on central regions of interest
    3. Resizing frames to a standardized resolution for consistent analysis
    4. Extracting deep features using a pre-trained CNN model
    5. Aggregating features across frames to create video-level representations
    
    Args:
        video_path (str): Path to the video file for analysis
        num_frames (int): Number of frames to sample from the video
        target_size (tuple): Target resolution for frame processing
        crop_fraction (float): Fraction of frame width to retain (center crop)
        
    Returns:
        numpy.ndarray: Video-level feature vector representing overall visual characteristics
        
    Algorithm:
        1. Open video file and determine total frame count
        2. Generate uniform frame indices for sampling
        3. For each sampled frame:
           - Extract frame from video
           - Apply center cropping if specified
           - Resize to target resolution
           - Convert BGR to RGB color space
           - Preprocess for CNN input
           - Extract deep features
        4. Average features across all frames to create video-level representation
        
    Note:
        The center cropping approach focuses analysis on the central visual field,
        which typically contains the most relevant navigational information and
        visual guidance cues during gameplay.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
    
    # Determine total frames and generate sampling indices
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_samples = min(num_frames, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    features = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Apply center cropping to focus on central region of interest
        if crop_fraction < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * crop_fraction)
            left = (w - new_w) // 2
            frame = frame[:, left:left + new_w]
            
        # Resize frame to target resolution for consistent processing
        frame_resized = cv2.resize(frame, target_size)
        
        # Convert BGR (OpenCV) to RGB (Keras) for proper color interpretation
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Prepare frame for CNN input
        x = keras_image.img_to_array(frame_rgb)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Extract deep features using pre-trained model
        feat = CNN_MODEL.predict(x)
        features.append(feat.flatten())
    
    cap.release()
    
    if len(features) == 0:
        return None
        
    # Aggregate features across frames to create video-level representation
    video_feature = np.mean(np.array(features), axis=0)
    return video_feature


##############################################################################
# DATASET ORGANIZATION AND FEATURE EXTRACTION
##############################################################################

def load_all_features():
    """
    Processes videos from multiple sources and extracts comprehensive feature representations.
    
    This function implements a systematic approach to dataset organization by:
    1. Processing videos from the current directory (representing different level transitions)
    2. Analyzing supplementary datasets from YouTube and Bilibili platforms
    3. Identifying videos that deviate from standard navigation patterns
    4. Assigning appropriate group labels for comparative analysis
    
    The function organizes videos into the following groups:
    - Valley: Reference navigation patterns from the valley level
    - OtherTransition: Standard level transitions from the main dataset
    - BadTransition: Navigation patterns that deviate from expected behavior
    - Supplementary: Additional data from external sources for validation
    
    Returns:
        list of tuples: Each tuple contains (video_path, feature_vector, group)
                       for subsequent analysis and comparison
        
    Dataset Structure:
        The function processes videos from multiple directories to ensure
        comprehensive coverage of different navigation scenarios and player
        behaviors. This multi-source approach enhances the robustness of
        the analysis and provides validation across different contexts.
    """
    all_features = []
    
    # Process videos from the current directory (main dataset)
    current_videos = glob.glob("*.mp4")
    for video_path in current_videos:
        # Identify reference video (valley level) for baseline comparison
        if video_path.lower().startswith("valley"):
            group = "Valley"
        else:
            group = "OtherTransition"
            
        feature_vector = extract_video_deep_features(video_path)
        if feature_vector is not None:
            all_features.append((video_path, feature_vector, group))
    
    # Process YouTube dataset (supplementary data)
    youtube_dir = "YoutubeDataset"
    if os.path.exists(youtube_dir):
        youtube_videos = glob.glob(os.path.join(youtube_dir, "*.mp4"))
        for video_path in youtube_videos:
            feature_vector = extract_video_deep_features(video_path)
            if feature_vector is not None:
                all_features.append((video_path, feature_vector, "Valley"))
    
    # Process Bilibili dataset (supplementary data)
    bilibili_dir = "BilibiliDataset"
    if os.path.exists(bilibili_dir):
        bilibili_videos = glob.glob(os.path.join(bilibili_dir, "*.mp4"))
        for video_path in bilibili_videos:
            feature_vector = extract_video_deep_features(video_path)
            if feature_vector is not None:
                all_features.append((video_path, feature_vector, "Valley"))
    
    # Process bad transition videos (deviant behavior analysis)
    bad_transition_dir = "BadTransition"
    if os.path.exists(bad_transition_dir):
        bad_videos = glob.glob(os.path.join(bad_transition_dir, "*.mp4"))
        for video_path in bad_videos:
            feature_vector = extract_video_deep_features(video_path)
            if feature_vector is not None:
                all_features.append((video_path, feature_vector, "BadTransition"))
    
    return all_features


##############################################################################
# FEATURE ANALYSIS AND COMPARISON FUNCTIONS
##############################################################################

def analyze_feature_similarities(features_list):
    """
    Analyzes similarities and differences between video feature vectors.
    
    This function implements comprehensive feature analysis to understand:
    1. How different navigation patterns relate to each other
    2. Which visual characteristics are most important for player guidance
    3. How bad transitions differ from successful navigation patterns
    4. The consistency of visual guidance across different datasets
    
    Args:
        features_list (list): List of (video_path, feature_vector, group) tuples
        
    Returns:
        dict: Dictionary containing various similarity and analysis metrics
        
    Analysis Methods:
        - Euclidean distance computation between feature vectors
        - Principal Component Analysis (PCA) for dimensionality reduction
        - Group-wise similarity analysis for pattern identification
        - Cross-dataset validation of feature consistency
    """
    if not features_list:
        return {}
    
    # Extract feature vectors and group labels
    paths, features, groups = zip(*features_list)
    features_array = np.array(features)
    groups_array = np.array(groups)
    
    # Compute pairwise distances between all feature vectors
    distances = euclidean_distances(features_array)
    
    # Perform PCA for dimensionality reduction and visualization
    pca = PCA(n_components=min(3, len(features_array)))
    features_pca = pca.fit_transform(features_array)
    
    # Analyze group-wise similarities
    group_analysis = {}
    unique_groups = set(groups)
    
    for group in unique_groups:
        group_indices = np.where(groups_array == group)[0]
        group_features = features_array[group_indices]
        
        # Compute intra-group similarity (cohesion)
        if len(group_indices) > 1:
            group_distances = distances[np.ix_(group_indices, group_indices)]
            intra_group_similarity = np.mean(group_distances)
        else:
            intra_group_similarity = 0.0
            
        # Compute inter-group differences (separation)
        other_indices = np.where(groups_array != group)[0]
        if len(other_indices) > 0:
            inter_group_distances = distances[np.ix_(group_indices, other_indices)]
            inter_group_difference = np.mean(inter_group_distances)
        else:
            inter_group_difference = 0.0
            
        group_analysis[group] = {
            'intra_group_similarity': intra_group_similarity,
            'inter_group_difference': inter_group_difference,
            'sample_count': len(group_indices)
        }
    
    return {
        'distances': distances,
        'pca_features': features_pca,
        'explained_variance': pca.explained_variance_ratio_,
        'group_analysis': group_analysis,
        'paths': paths,
        'groups': groups
    }


def visualize_feature_analysis(analysis_results, output_path="player_behavior_analysis.png"):
    """
    Creates comprehensive visualizations of player behavior analysis results.
    
    This function generates multiple visualization types to facilitate
    understanding of player navigation patterns and visual guidance effectiveness:
    1. PCA-based feature space visualization
    2. Group-wise similarity heatmaps
    3. Distance distribution analysis
    4. Cross-group comparison plots
    
    Args:
        analysis_results (dict): Results from analyze_feature_similarities()
        output_path (str): Path to save the generated visualization
        
    Returns:
        matplotlib.figure.Figure: Generated figure object
        
    Visualization Components:
        - 2D/3D PCA plots showing feature space organization
        - Heatmaps of pairwise distances between videos
        - Bar charts of group-wise similarity metrics
        - Scatter plots highlighting group separations
    """
    if not analysis_results:
        print("No analysis results to visualize.")
        return None
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # PCA visualization
    ax1 = plt.subplot(2, 3, 1)
    pca_features = analysis_results['pca_features']
    groups = analysis_results['groups']
    
    # Color code by group
    unique_groups = list(set(groups))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
    
    for i, group in enumerate(unique_groups):
        group_indices = [j for j, g in enumerate(groups) if g == group]
        ax1.scatter(pca_features[group_indices, 0], pca_features[group_indices, 1],
                   c=[colors[i]], label=group, alpha=0.7)
    
    ax1.set_xlabel(f"PC1 ({analysis_results['explained_variance'][0]:.2%} variance)")
    ax1.set_ylabel(f"PC2 ({analysis_results['explained_variance'][1]:.2%} variance)")
    ax1.set_title("PCA Feature Space Visualization")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distance heatmap
    ax2 = plt.subplot(2, 3, 2)
    im = ax2.imshow(analysis_results['distances'], cmap='viridis')
    ax2.set_title("Pairwise Feature Distances")
    ax2.set_xlabel("Video Index")
    ax2.set_ylabel("Video Index")
    plt.colorbar(im, ax=ax2)
    
    # Group analysis visualization
    ax3 = plt.subplot(2, 3, 3)
    group_analysis = analysis_results['group_analysis']
    groups_list = list(group_analysis.keys())
    intra_similarities = [group_analysis[g]['intra_group_similarity'] for g in groups_list]
    inter_differences = [group_analysis[g]['inter_group_difference'] for g in groups_list]
    
    x = np.arange(len(groups_list))
    width = 0.35
    
    ax3.bar(x - width/2, intra_similarities, width, label='Intra-group Similarity', alpha=0.8)
    ax3.bar(x + width/2, inter_differences, width, label='Inter-group Difference', alpha=0.8)
    ax3.set_xlabel('Groups')
    ax3.set_ylabel('Distance Metric')
    ax3.set_title('Group-wise Similarity Analysis')
    ax3.set_xticks(x)
    ax3.set_xticklabels(groups_list)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Sample count visualization
    ax4 = plt.subplot(2, 3, 4)
    sample_counts = [group_analysis[g]['sample_count'] for g in groups_list]
    ax4.bar(groups_list, sample_counts, alpha=0.8)
    ax4.set_xlabel('Groups')
    ax4.set_ylabel('Number of Videos')
    ax4.set_title('Dataset Distribution by Group')
    ax4.grid(True, alpha=0.3)
    
    # Distance distribution analysis
    ax5 = plt.subplot(2, 3, 5)
    distances_flat = analysis_results['distances'].flatten()
    ax5.hist(distances_flat, bins=50, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Feature Distance')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Feature Distances')
    ax5.grid(True, alpha=0.3)
    
    # 3D PCA visualization if available
    if pca_features.shape[1] >= 3:
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        for i, group in enumerate(unique_groups):
            group_indices = [j for j, g in enumerate(groups) if g == group]
            ax6.scatter(pca_features[group_indices, 0], 
                       pca_features[group_indices, 1],
                       pca_features[group_indices, 2],
                       c=[colors[i]], label=group, alpha=0.7)
        ax6.set_xlabel(f"PC1 ({analysis_results['explained_variance'][0]:.2%} variance)")
        ax6.set_ylabel(f"PC2 ({analysis_results['explained_variance'][1]:.2%} variance)")
        ax6.set_zlabel(f"PC3 ({analysis_results['explained_variance'][2]:.2%} variance)")
        ax6.set_title("3D PCA Feature Space")
        ax6.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig


##############################################################################
# MAIN ANALYSIS PIPELINE
##############################################################################

def main():
    """
    Main execution function for player behavior analysis.
    
    This function orchestrates the complete analysis pipeline:
    1. Loads and processes video data from multiple sources
    2. Extracts deep learning features from video sequences
    3. Performs comprehensive similarity and pattern analysis
    4. Generates visualizations and analysis reports
    5. Saves results for further investigation
    
    The analysis provides insights into:
        - How different visual guidance approaches affect player navigation
        - Which visual characteristics are most important for successful transitions
        - How player behavior varies across different level types
        - The effectiveness of visual guidance across different datasets
    """
    print("Starting Player Behavior Analysis...")
    
    # Load and extract features from all video sources
    print("Loading video features...")
    features_list = load_all_features()
    
    if not features_list:
        print("No valid video features found. Exiting.")
        return
    
    print(f"Successfully processed {len(features_list)} videos.")
    
    # Perform comprehensive feature analysis
    print("Analyzing feature similarities...")
    analysis_results = analyze_feature_similarities(features_list)
    
    # Generate visualizations
    print("Creating visualizations...")
    fig = visualize_feature_analysis(analysis_results)
    
    if fig:
        print("Analysis complete. Results saved to 'player_behavior_analysis.png'")
        plt.show()
    else:
        print("Error generating visualizations.")
    
    # Print summary statistics
    print("\n=== Analysis Summary ===")
    group_analysis = analysis_results['group_analysis']
    for group, metrics in group_analysis.items():
        print(f"{group}: {metrics['sample_count']} videos, "
              f"Intra-similarity: {metrics['intra_group_similarity']:.3f}, "
              f"Inter-difference: {metrics['inter_group_difference']:.3f}")


if __name__ == "__main__":
    main()
