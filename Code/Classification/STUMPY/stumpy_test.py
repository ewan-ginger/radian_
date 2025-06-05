import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stumpy
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks

# === Load CSVs ===
session_df = pd.read_csv('Data/Session 1/passing_session_1.1.csv')
truncated_template = pd.read_csv('Data/Session 1/truncated_pass_template_1.csv')

# === Define Sensor Axes (Orientation + Gyroscope) ===
orientation_axes = ['orientation_x', 'orientation_y', 'orientation_z']
gyroscope_axes = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']
axes = orientation_axes + gyroscope_axes  # Combine both lists

# === Detect Peaks in Gyroscope Data ===
gyro_peaks = {}
event_regions = {}
window_size = 10

for axis in gyroscope_axes:
    # Find peaks with prominence of 6
    peaks, _ = find_peaks(session_df[axis].values, prominence=6)
    gyro_peaks[axis] = peaks
    
    # Create event regions around peaks
    regions = []
    for peak in peaks:
        start = max(0, peak - window_size)
        end = min(len(session_df), peak + window_size)
        regions.append((start, end))
    event_regions[axis] = regions

# === Plot Sensor Data with Peaks and Event Regions ===
def plot_sensor_data(df1, df2, title1='Session', title2='Template'):
    fig, axs = plt.subplots(len(axes), 2, figsize=(12, 10), sharex=True)
    
    for i, axis in enumerate(axes):
        # Plot session data
        axs[i, 0].plot(df1[axis], label=title1)
        axs[i, 0].set_title(f"{title1} - {axis}")
        
        # Add peaks and event regions for gyroscope axes
        if axis in gyroscope_axes:
            # Add red crosses for peaks
            if axis in gyro_peaks and len(gyro_peaks[axis]) > 0:
                # Convert to numpy arrays to avoid pandas indexing issues
                x_peaks = gyro_peaks[axis]
                y_peaks = df1[axis].values[gyro_peaks[axis]]
                axs[i, 0].plot(x_peaks, y_peaks, 'rx', markersize=8)
            
            # Add yellow regions for event windows
            if axis in event_regions:
                y_min, y_max = axs[i, 0].get_ylim()
                for start, end in event_regions[axis]:
                    axs[i, 0].axvspan(start, end, color='yellow', alpha=0.3)
        
        # Plot template data
        axs[i, 1].plot(df2[axis], label=title2, color='orange')
        axs[i, 1].set_title(f"{title2} - {axis}")

    plt.tight_layout()
    plt.show()

plot_sensor_data(session_df, truncated_template)

# === Define Index Window ===
index_window = 10  # Define the matching index window

# === Find Matches for Each Axis ===
matches_per_axis = {}

for axis in axes:
    session_data = session_df[axis].to_numpy()
    template_data = truncated_template[axis].to_numpy()
    
    # Compute the distance profile
    distance_profile = stumpy.mass(template_data, session_data)

    # Compute dynamic max distance threshold
    base_threshold = max(np.mean(distance_profile) - 1 * np.std(distance_profile), np.min(distance_profile))

    # Apply stricter threshold for gyroscope data
    if axis in gyroscope_axes:
        max_distance = 0.75 * base_threshold  # Reduce by 25%
    else:
        max_distance = base_threshold  # Keep original for orientation

    # Get match results with adjusted thresholds
    match_results = stumpy.match(template_data, session_data, max_distance=max_distance, normalize=True)
    
    if match_results.ndim == 1:
        match_indices = match_results.astype(int)
    else:
        match_indices = match_results[:, 1].astype(int)
    
    matches_per_axis[axis] = set(match_indices)

# === Find Matches Within the Index Window Across All Sensors ===
def find_windowed_matches(matches_per_axis, window):
    sorted_matches = {axis: sorted(matches) for axis, matches in matches_per_axis.items()}
    strict_matches = set()
    potential_matches = set()

    # Use the first sensor's matches as reference
    reference_axis = axes[0]
    reference_matches = sorted_matches[reference_axis]

    for ref_idx in reference_matches:
        # Count how many axes have a match within the window
        match_count = sum(
            any(abs(ref_idx - other_idx) <= window for other_idx in sorted_matches[axis]) 
            for axis in axes
        )

        if match_count == len(axes):  # All axes match
            strict_matches.add(ref_idx)
        elif match_count == len(axes) - 1:  # Missing one match
            potential_matches.add(ref_idx)

    return sorted(strict_matches), sorted(potential_matches)

common_matches, potential_matches = find_windowed_matches(matches_per_axis, index_window)

print(f"\n{len(common_matches)} strict matches (within window of {index_window}): {common_matches}")
print(f"{len(potential_matches)} potential matches (missing one axis, within window): {potential_matches}")

# === Plot Matches ===
def plot_matches(session, common_matches, potential_matches, template_len):
    fig, axs = plt.subplots(len(axes), 1, figsize=(12, 14), sharex=True)
    
    for i, axis in enumerate(axes):
        y_data = session[axis].to_numpy()
        axs[i].plot(y_data, label=f'Session Data ({axis})')
        
        y_min = np.min(y_data)
        y_max = np.max(y_data)

        # Plot strict matches (Red)
        for idx in common_matches:
            rect = Rectangle((idx, y_min), template_len, y_max - y_min,
                             color='red', alpha=0.3)
            axs[i].add_patch(rect)

        # Plot potential matches (Blue)
        for idx in potential_matches:
            rect = Rectangle((idx, y_min), template_len, y_max - y_min,
                             color='blue', alpha=0.3)
            axs[i].add_patch(rect)

        axs[i].set_title(f"Matches in Session Data ({axis})")
        axs[i].legend()
    
    plt.tight_layout()
    plt.show()

plot_matches(session_df, common_matches, potential_matches, len(truncated_template))

def plot_template_with_overlaid_matches(template, session, common_matches, potential_matches):
    template_len = len(template)
    fig, axs = plt.subplots(len(axes), 1, figsize=(14, 16), sharex=True)

    for i, axis in enumerate(axes):
        template_data = template[axis].to_numpy()
        axs[i].plot(template_data, label='Template', color='black', linewidth=2)

        # Overlay strict matches (red)
        for idx in common_matches:
            if idx + template_len <= len(session):
                match_segment = session[axis].to_numpy()[idx:idx + template_len]
                axs[i].plot(match_segment, color='#F47A09', alpha=0.5, label='Strict Match' if idx == common_matches[0] else "")

        # Overlay potential matches (blue)
        for idx in potential_matches:
            if idx + template_len <= len(session):
                match_segment = session[axis].to_numpy()[idx:idx + template_len]
                axs[i].plot(match_segment, color='#2830E3', alpha=0.4, linestyle='--', label='Potential Match' if idx == potential_matches[0] else "")

        axs[i].set_title(f"Overlayed Matches on Template - {axis}")
        axs[i].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('DTW_matches.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_template_with_overlaid_matches(truncated_template, session_df, common_matches, potential_matches)

# === Compute and Plot MASS Distance Profile ===
def plot_mass_distance_profile(session, template, axes):
    fig, axs = plt.subplots(len(axes), 1, figsize=(12, 14), sharex=True)
    
    for i, axis in enumerate(axes):
        session_data = session[axis].to_numpy()
        template_data = template[axis].to_numpy()
        distance_profile = stumpy.mass(template_data, session_data)
        
        axs[i].plot(distance_profile, label=f'Distance Profile ({axis})', color='purple')
        axs[i].set_title(f"MASS Distance Profile ({axis})")
        axs[i].legend()
    
    plt.tight_layout()
    plt.show()

plot_mass_distance_profile(session_df, truncated_template, axes)