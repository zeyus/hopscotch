#!/usr/bin/env python3
"""
Script to generate animated 3D visualizations of motion capture data.
This can visualize one or multiple body points over time from the hopscotch dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Animate mocap data in 3D")
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/mocap_data.feather",
        help="Path to mocap data file (feather format)"
    )
    parser.add_argument(
        "--subjects", 
        type=int, 
        nargs="+",
        help="Subject IDs to visualize (can specify multiple)"
    )
    parser.add_argument(
        "--conditions", 
        type=str, 
        nargs="+",
        help="Conditions to visualize (e.g., 'k' or 'h', can specify multiple)"
    )
    parser.add_argument(
        "--obstacles", 
        type=int, 
        nargs="+",
        help="Obstacle counts to visualize (can specify multiple)"
    )
    parser.add_argument(
        "--points", 
        type=str, 
        nargs="+",
        default=["head", "foot_front_r", "foot_front_l"],
        help="Body points to visualize (without .X/.Y/.Z suffix)"
    )
    parser.add_argument(
        "--frames", 
        type=int, 
        default=300,
        help="Number of frames to animate"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file path for saving animation (mp4 format)"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30,
        help="Frames per second for the animation"
    )
    parser.add_argument(
        "--trail-length", 
        type=int, 
        default=20,
        help="Length of trailing points to show"
    )
    parser.add_argument(
        "--normalize-time", 
        action="store_true",
        help="Normalize time across all sequences to make them the same length"
    )
    parser.add_argument(
        "--target-frames", 
        type=int, 
        default=100,
        help="Number of frames to normalize to when using --normalize-time"
    )
    parser.add_argument(
        "--color-by", 
        choices=["point", "subject", "condition"],
        default="condition",
        help="Color coding scheme: by point type, subject, or condition"
    )
    parser.add_argument(
        "--compare", 
        choices=["subjects", "conditions", "points"],
        default="conditions",
        help="Primary comparison focus: subjects, conditions, or points"
    )
    parser.add_argument(
        "--synchronize", 
        action="store_true",
        help="Synchronize starting positions across conditions"
    )
    return parser.parse_args()


def load_data(filepath, subjects=None, conditions=None, obstacles=None):
    """Load mocap data with optional filtering."""
    try:
        data = pd.read_feather(filepath)
        logging.info(f"Loaded data with shape: {data.shape}")
        
        # Apply filters if specified
        if subjects is not None:
            data = data[data['subject'].isin(subjects)]
        if conditions is not None:
            data = data[data['condition'].isin(conditions)]
        if obstacles is not None:
            data = data[data['obstacles'].isin(obstacles)]
            
        if len(data) == 0:
            raise ValueError("No data matches the specified filters")
        
        # Log how many trials we're working with
        trial_count = data.groupby(['subject', 'condition', 'obstacles']).ngroups
        logging.info(f"Found {trial_count} unique trials matching filters")
            
        logging.info(f"Filtered data shape: {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"Data file not found: {filepath}")
        raise


def normalize_sequence(x, y, z, target_frames):
    """Normalize a sequence to have a specific number of frames."""
    original_frames = len(x)
    
    if original_frames == target_frames:
        return x, y, z
        
    # Create evenly spaced indices for the target frame count
    original_indices = np.arange(original_frames)
    target_indices = np.linspace(0, original_frames - 1, target_frames)
    
    # Interpolate x, y, z values to the new indices
    x_norm = np.interp(target_indices, original_indices, x)
    y_norm = np.interp(target_indices, original_indices, y)
    z_norm = np.interp(target_indices, original_indices, z)
    
    return x_norm, y_norm, z_norm


def prepare_points_data(data, points, max_frames=None, normalize_time=False, target_frames=100):
    """Extract coordinates for specified body points."""
    
    # Group data by subject AND condition since it's a within-subject design
    subject_condition_groups = data.groupby(['subject', 'condition', 'obstacles'])
    
    coords_data = {}
    
    for (subject_id, condition, obstacles), group_data in subject_condition_groups:
        for point in points:
            x_col = f"{point}.X"
            y_col = f"{point}.Y"
            z_col = f"{point}.Z"
            
            if not all(col in group_data.columns for col in [x_col, y_col, z_col]):
                logging.warning(f"Point {point} not found in data for subject {subject_id}, condition {condition}")
                continue
                
            # Extract coordinates
            x = group_data[x_col].values
            y = group_data[y_col].values
            z = group_data[z_col].values
            
            # Check for empty data or NaN/Inf values
            if len(x) == 0 or np.all(np.isnan(x)) or np.all(np.isnan(y)) or np.all(np.isnan(z)) or \
               np.all(np.isinf(x)) or np.all(np.isinf(y)) or np.all(np.isinf(z)):
                logging.warning(f"Skipping {point} for subject {subject_id}, condition {condition}, obstacles {obstacles} due to invalid data")
                continue
                
            # Limit to max_frames if specified and not normalizing
            if max_frames is not None and not normalize_time and max_frames < len(x):
                x = x[:max_frames]
                y = y[:max_frames]
                z = z[:max_frames]
                
            # Normalize time if requested
            if normalize_time:
                x, y, z = normalize_sequence(x, y, z, target_frames)
                
            # Create a unique identifier for this point-subject-condition combination
            point_key = f"{point}_S{subject_id}_C{condition}_O{obstacles}"
            
            coords_data[point_key] = {
                'x': x,
                'y': y,
                'z': z,
                'point': point,
                'subject': subject_id,
                'condition': condition,
                'obstacles': obstacles
            }
    
    return coords_data


def animate_3d(coords_data, output=None, fps=30, trail_length=20, color_by="condition", compare="conditions"):
    """Create a 3D animation of the points."""
    # Safety check - make sure we have data to plot
    if not coords_data:
        logging.error("No valid data to animate")
        return None
        
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color schemes based on user's choice
    if color_by == "point":
        # Get unique point types
        unique_values = set(data['point'] for data in coords_data.values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
        color_dict = {value: colors[i] for i, value in enumerate(unique_values)}
    elif color_by == "subject":
        # Get unique subjects
        unique_values = set(data['subject'] for data in coords_data.values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
        color_dict = {value: colors[i] for i, value in enumerate(unique_values)}
    else:  # color_by == "condition"
        # Get unique conditions
        unique_values = set(data['condition'] for data in coords_data.values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
        color_dict = {value: colors[i] for i, value in enumerate(unique_values)}
    
    try:
        # Find global min/max for better scaling with safeguards for NaN/Inf
        all_x = np.concatenate([data['x'] for data in coords_data.values()])
        all_y = np.concatenate([data['y'] for data in coords_data.values()])
        all_z = np.concatenate([data['z'] for data in coords_data.values()])
        
        # Filter out any NaN or Inf values safely
        all_x = all_x[np.isfinite(all_x)]
        all_y = all_y[np.isfinite(all_y)]
        all_z = all_z[np.isfinite(all_z)]
        
        # Additional check for empty arrays after filtering
        if len(all_x) == 0 or len(all_y) == 0 or len(all_z) == 0:
            raise ValueError("No valid coordinate data remains after filtering NaN/Inf values")
            
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        z_min, z_max = np.min(all_z), np.max(all_z)
        
        # Ensure equal aspect ratio
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        if max_range <= 0:
            max_range = 1.0  # Fallback to prevent division by zero
            
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        
        # Set axis limits
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    except Exception as e:
        logging.error(f"Error setting axis limits: {e}")
        logging.error("Setting default axis limits")
        # Set some reasonable default limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    
    # Initialize plot objects
    points_objects = {}
    trails_objects = {}
    
    # Create marker styles to differentiate series
    line_styles = ['-', '--', '-.', ':']
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    marker_sizes = [8, 7, 9, 7, 9, 8, 8, 7, 9, 7]
    
    # Determine secondary grouping scheme based on comparison focus
    if compare == "conditions":
        secondary_key = 'condition'
    elif compare == "subjects":
        secondary_key = 'subject'
    else:  # compare == "points"
        secondary_key = 'point'
    
    # Keep track of already used style combinations
    used_styles = {}
    
    for idx, (point_key, data) in enumerate(coords_data.items()):
        # Determine color based on coloring scheme
        if color_by == "point":
            point_color = color_dict[data['point']]
            color_label = f"Point: {data['point']}"
        elif color_by == "subject":
            point_color = color_dict[data['subject']]
            color_label = f"Subject: {data['subject']}"
        else:  # color_by == "condition"
            point_color = color_dict[data['condition']]
            color_label = f"Condition: {data['condition']}"
        
        # Get the secondary grouping key for consistent styling
        secondary_value = data[secondary_key]
        
        # Use consistent marker style for the same secondary grouping value
        if secondary_value not in used_styles:
            style_idx = len(used_styles) % len(marker_styles)
            used_styles[secondary_value] = {
                'marker': marker_styles[style_idx],
                'size': marker_sizes[style_idx],
                'line': line_styles[len(used_styles) % len(line_styles)]
            }
        
        marker_style = used_styles[secondary_value]['marker']
        marker_size = used_styles[secondary_value]['size']
        line_style = used_styles[secondary_value]['line']
        
        # Create label based on what we're comparing
        if compare == "conditions":
            label = f"{color_label} - O{data['obstacles']} - {data['point']} - S{data['subject']}"
        elif compare == "subjects":
            label = f"{color_label} - S{data['subject']} - {data['point']} - C{data['condition']}O{data['obstacles']}"
        else:  # compare == "points"
            label = f"{color_label} - {data['point']} - S{data['subject']} - C{data['condition']}O{data['obstacles']}"
        
        # Current position marker
        points_objects[point_key], = ax.plot(
            [], [], [], 
            marker=marker_style, 
            markersize=marker_size,
            color=point_color, 
            label=label
        )
        
        # Trail (showing past positions)
        trails_objects[point_key], = ax.plot(
            [], [], [], 
            line_style, 
            linewidth=2, 
            color=point_color, 
            alpha=0.6
        )
    
    # # Add legend with a better layout for potentially many items
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])
    
    # # Determine legend columns based on number of items
    # num_items = len(points_objects)
    # if num_items <= 3:
    #     ncols = num_items
    # elif num_items <= 6:
    #     ncols = 3
    # else:
    #     ncols = 4
        
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
    #           ncol=ncols, fontsize='small', frameon=True)
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Motion Capture Animation')
    
    # Determine number of frames from the first point
    num_frames = len(next(iter(coords_data.values()))['x'])
    
    def init():
        """Initialize the animation."""
        for point_obj in points_objects.values():
            point_obj.set_data([], [])
            point_obj.set_3d_properties([])
        
        for trail_obj in trails_objects.values():
            trail_obj.set_data([], [])
            trail_obj.set_3d_properties([])
        
        return list(points_objects.values()) + list(trails_objects.values())
    
    def update(frame):
        """Update function for animation."""
        for point_key, point_obj in points_objects.items():
            data = coords_data[point_key]
            
            if frame < len(data['x']):  # Ensure we don't exceed data length
                # Get current position and check for NaN/Inf
                x_current = data['x'][frame]
                y_current = data['y'][frame]
                z_current = data['z'][frame]
                
                if np.isfinite(x_current) and np.isfinite(y_current) and np.isfinite(z_current):
                    # Update current position
                    point_obj.set_data([x_current], [y_current])
                    point_obj.set_3d_properties([z_current])
                    
                    # Update trail with finite values only
                    start_idx = max(0, frame - trail_length)
                    trail_x = data['x'][start_idx:frame+1]
                    trail_y = data['y'][start_idx:frame+1]
                    trail_z = data['z'][start_idx:frame+1]
                    
                    # Filter out NaN/Inf values from trails
                    valid_mask = np.logical_and.reduce([
                        np.isfinite(trail_x), 
                        np.isfinite(trail_y), 
                        np.isfinite(trail_z)
                    ])
                    
                    if np.any(valid_mask):
                        trails_objects[point_key].set_data(
                            trail_x[valid_mask], 
                            trail_y[valid_mask]
                        )
                        trails_objects[point_key].set_3d_properties(trail_z[valid_mask])
                else:
                    # Hide point if coordinates are invalid
                    point_obj.set_data([], [])
                    point_obj.set_3d_properties([])
                    trails_objects[point_key].set_data([], [])
                    trails_objects[point_key].set_3d_properties([])
        
        # Add frame counter and coloring info
        title = f'Motion Capture Animation - Frame {frame}/{num_frames-1} (Colored by {color_by})'
        ax.set_title(title)
        
        return list(points_objects.values()) + list(trails_objects.values())
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames,
        init_func=init, blit=False, interval=1000/fps
    )
    
    # Save animation if output is specified
    if output:
        logging.info(f"Saving animation to {output}")
        ani.save(output, writer='ffmpeg', fps=fps)
    
    plt.tight_layout()
    plt.show()
    
    return ani


def apply_synchronization(coords_data, synchronize=False):
    """Optionally synchronize starting positions across different conditions."""
    if not synchronize:
        return coords_data
    
    # Group by subject and point to find shifts needed
    subjects_points = {}
    for key, data in coords_data.items():
        sp_key = (data['subject'], data['point'])
        if sp_key not in subjects_points:
            subjects_points[sp_key] = []
        subjects_points[sp_key].append(key)
    
    # For each subject-point group, shift data to align starting positions
    for (subject, point), keys in subjects_points.items():
        if len(keys) <= 1:  # Skip if only one condition
            continue
            
        # Get reference data (first key)
        ref_key = keys[0]
        ref_x_start = coords_data[ref_key]['x'][0]
        ref_y_start = coords_data[ref_key]['y'][0]
        ref_z_start = coords_data[ref_key]['z'][0]
        
        # Adjust other conditions to match the reference start point
        for key in keys[1:]:
            x_shift = ref_x_start - coords_data[key]['x'][0]
            y_shift = ref_y_start - coords_data[key]['y'][0]
            z_shift = ref_z_start - coords_data[key]['z'][0]
            
            # Apply shifts
            coords_data[key]['x'] = coords_data[key]['x'] + x_shift
            coords_data[key]['y'] = coords_data[key]['y'] + y_shift
            coords_data[key]['z'] = coords_data[key]['z'] + z_shift
            
    return coords_data


def main():
    args = parse_args()
    
    # Load data
    data = load_data(
        args.data, 
        subjects=args.subjects, 
        conditions=args.conditions, 
        obstacles=args.obstacles
    )
    
    # Prepare coordinate data
    coords_data = prepare_points_data(
        data, 
        args.points, 
        max_frames=args.frames,
        normalize_time=args.normalize_time,
        target_frames=args.target_frames
    )
    
    if not coords_data:
        logging.error("No valid points found in the data")
        return
    
    # Optionally synchronize starting positions
    if args.synchronize:
        coords_data = apply_synchronization(coords_data, synchronize=True)
        logging.info("Synchronized starting positions across conditions")
    
    # Create animation
    animate_3d(
        coords_data,
        output=args.output,
        fps=args.fps,
        trail_length=args.trail_length,
        color_by=args.color_by,
        compare=args.compare
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
