import matplotlib.pyplot as plt
import numpy as np
from helper_code import *
try:
    from .normalize_wave import band_pass_filter, apply_notch_filter
except ImportError:
    from modules.normalize_wave import band_pass_filter, apply_notch_filter

def visualize_ecg(record):
    """
    Visualize ECG data for a single record.
    
    Args:
        record (str): Path to the ECG record (without extension)
    """
    # Load the signal data and metadata
    signal, fields = load_signals(record)
    header = load_header(record)
    
    # Get metadata
    sampling_frequency = get_sampling_frequency(header)
    signal_names = get_signal_names(header)
    num_samples = signal.shape[0]
    
    # Create time axis in seconds
    time = np.arange(num_samples) / sampling_frequency
    
    # Create figure with subplots for each lead
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle(f'ECG Record: {record}', fontsize=16, fontweight='bold')
    
    # Standard 12-lead ECG layout
    lead_positions = [
        (0, 0), (0, 1), (0, 2),  # I, II, III
        (1, 0), (1, 1), (1, 2),  # AVR, AVL, AVF
        (2, 0), (2, 1), (2, 2),  # V1, V2, V3
        (3, 0), (3, 1), (3, 2)   # V4, V5, V6
    ]
    
    # Plot each lead
    for i, (signal_name, (row, col)) in enumerate(zip(signal_names, lead_positions)):
        ax = axes[row, col]
        
        # Plot the signal
        ax.plot(time, signal[:, i], 'b-', linewidth=0.8)
        ax.set_title(f'Lead {signal_name}', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits for better comparison
        y_range = np.max(signal[:, i]) - np.min(signal[:, i])
        y_center = np.mean(signal[:, i])
        ax.set_ylim(y_center - y_range*0.6, y_center + y_range*0.6)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Extract just the record name from the path
    record_name = record.split('/')[-1]
    
    # Show the plot
    plt.savefig(f'./visualizations/{record_name}.png')
    
    return fig


def visualize_ecg_filtered(record, apply_bandpass=True, apply_notch=True, 
                          low_cutoff=0.5, high_cutoff=40.0, notch_freq=50.0):
    """
    Visualize ECG data with optional filtering for a single record.
    
    Args:
        record (str): Path to the ECG record (without extension)
        apply_bandpass (bool): Whether to apply band-pass filter
        apply_notch (bool): Whether to apply notch filter for power line interference
        low_cutoff (float): Low cutoff frequency for band-pass filter (Hz)
        high_cutoff (float): High cutoff frequency for band-pass filter (Hz)
        notch_freq (float): Notch frequency for power line interference (50 Hz or 60 Hz)
    """
    # Load the signal data and metadata
    signal, fields = load_signals(record)
    header = load_header(record)
    
    # Get metadata
    sampling_frequency = get_sampling_frequency(header)
    signal_names = get_signal_names(header)
    num_samples = signal.shape[0]
    
    # Apply filters if requested
    filtered_signal = signal.copy()
    filter_info = "Raw"
    
    if apply_bandpass:
        filtered_signal = band_pass_filter(filtered_signal, sampling_frequency, 
                                         low_cutoff=low_cutoff, high_cutoff=high_cutoff)
        filter_info = f"Band-pass ({low_cutoff}-{high_cutoff} Hz)"
    
    if apply_notch:
        filtered_signal = apply_notch_filter(filtered_signal, sampling_frequency, 
                                           notch_freq=notch_freq)
        if "Band-pass" in filter_info:
            filter_info += f" + Notch ({notch_freq} Hz)"
        else:
            filter_info = f"Notch ({notch_freq} Hz)"
    
    # Create time axis in seconds
    time = np.arange(num_samples) / sampling_frequency
    
    # Create figure with subplots for each lead
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle(f'ECG Record: {record} - {filter_info}', fontsize=16, fontweight='bold')
    
    # Standard 12-lead ECG layout
    lead_positions = [
        (0, 0), (0, 1), (0, 2),  # I, II, III
        (1, 0), (1, 1), (1, 2),  # AVR, AVL, AVF
        (2, 0), (2, 1), (2, 2),  # V1, V2, V3
        (3, 0), (3, 1), (3, 2)   # V4, V5, V6
    ]
    
    # Plot each lead
    for i, (signal_name, (row, col)) in enumerate(zip(signal_names, lead_positions)):
        ax = axes[row, col]
        
        # Plot the filtered signal
        ax.plot(time, filtered_signal[:, i], 'b-', linewidth=0.8)
        ax.set_title(f'Lead {signal_name}', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits for better comparison
        y_range = np.max(filtered_signal[:, i]) - np.min(filtered_signal[:, i])
        y_center = np.mean(filtered_signal[:, i])
        ax.set_ylim(y_center - y_range*0.6, y_center + y_range*0.6)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Extract just the record name from the path
    record_name = record.split('/')[-1]
    
    # Save the plot with filter info in filename
    filter_suffix = ""
    if apply_bandpass and apply_notch:
        filter_suffix = "_filtered_bp_notch"
    elif apply_bandpass:
        filter_suffix = "_filtered_bp"
    elif apply_notch:
        filter_suffix = "_filtered_notch"
    
    plt.savefig(f'./visualizations/{record_name}{filter_suffix}.png')
    
    return fig

# Test the function with the first record
if __name__ == "__main__":
    records = find_records('./data')
    if records:
        record = records[0]
        print(f"Visualizing record: {record}")
        # Original visualization
        visualize_ecg(f'./data/{record}')
        # Filtered visualization
        visualize_ecg_filtered(f'./data/{record}', apply_bandpass=True, apply_notch=True)
    else:
        print("No records found in ./data directory")

