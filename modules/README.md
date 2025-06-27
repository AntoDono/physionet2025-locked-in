# ECG Signal Processing Modules

This directory contains modules for ECG signal processing and visualization.

## normalize_wave.py

Contains signal processing functions for ECG data normalization and filtering.

### Functions

#### `band_pass_filter(ecg_signal, sampling_frequency, low_cutoff=0.5, high_cutoff=40.0, order=4)`

Applies a Butterworth band-pass filter to ECG signals to remove baseline drift and high-frequency noise.

**Parameters:**
- `ecg_signal` (numpy.ndarray): ECG signal data (1D or 2D array)
- `sampling_frequency` (float): Sampling frequency in Hz
- `low_cutoff` (float, optional): Low cutoff frequency in Hz (default: 0.5)
- `high_cutoff` (float, optional): High cutoff frequency in Hz (default: 40.0)
- `order` (int, optional): Filter order (default: 4)

**Returns:**
- numpy.ndarray: Filtered ECG signal

**Example:**
```python
from modules.normalize_wave import band_pass_filter
from helper_code import load_signals, get_sampling_frequency, load_header

# Load ECG data
signal, fields = load_signals('./data/record_001')
header = load_header('./data/record_001')
fs = get_sampling_frequency(header)

# Apply band-pass filter
filtered_signal = band_pass_filter(signal, fs)

# Apply custom filter
custom_filtered = band_pass_filter(signal, fs, low_cutoff=1.0, high_cutoff=30.0)
```

#### `apply_notch_filter(ecg_signal, sampling_frequency, notch_freq=50.0, quality_factor=30)`

Applies a notch filter to remove power line interference (50/60 Hz).

**Parameters:**
- `ecg_signal` (numpy.ndarray): ECG signal data
- `sampling_frequency` (float): Sampling frequency in Hz
- `notch_freq` (float, optional): Frequency to notch out (default: 50.0 Hz)
- `quality_factor` (float, optional): Quality factor (default: 30)

#### `normalize_amplitude(ecg_signal, method='z_score')`

Normalizes ECG signal amplitude using different methods.

**Parameters:**
- `ecg_signal` (numpy.ndarray): ECG signal data
- `method` (str): Normalization method ('z_score', 'min_max', or 'robust')

## visualize.py

Contains visualization functions for ECG data.

### Functions

#### `visualize_ecg(record)`

Original visualization function for ECG records.

#### `visualize_ecg_filtered(record, apply_bandpass=True, apply_notch=True, ...)`

Enhanced visualization function with optional filtering.

**Parameters:**
- `record` (str): Path to ECG record
- `apply_bandpass` (bool): Whether to apply band-pass filter
- `apply_notch` (bool): Whether to apply notch filter
- `low_cutoff` (float): Low cutoff frequency for band-pass filter
- `high_cutoff` (float): High cutoff frequency for band-pass filter
- `notch_freq` (float): Notch frequency (50 Hz or 60 Hz)

## Usage Examples

### Basic Filtering
```python
from modules.normalize_wave import band_pass_filter
from helper_code import *

# Load data
signal, fields = load_signals('./data/record_001')
header = load_header('./data/record_001')
fs = get_sampling_frequency(header)

# Apply standard ECG filter (0.5-40 Hz)
filtered_signal = band_pass_filter(signal, fs)
```

### Visualization with Filtering
```python
from modules.visualize import visualize_ecg_filtered

# Create filtered visualization
visualize_ecg_filtered('./data/record_001', 
                      apply_bandpass=True, 
                      apply_notch=True,
                      low_cutoff=0.5,
                      high_cutoff=40.0,
                      notch_freq=50.0)
```

### Complete Processing Pipeline
```python
from modules.normalize_wave import band_pass_filter, apply_notch_filter, normalize_amplitude

# Load and process ECG data
signal, fields = load_signals('./data/record_001')
header = load_header('./data/record_001')
fs = get_sampling_frequency(header)

# Step 1: Band-pass filter
filtered_signal = band_pass_filter(signal, fs, low_cutoff=0.5, high_cutoff=40.0)

# Step 2: Remove power line interference
clean_signal = apply_notch_filter(filtered_signal, fs, notch_freq=50.0)

# Step 3: Normalize amplitude
normalized_signal = normalize_amplitude(clean_signal, method='z_score')
```

## Filter Design Notes

### Recommended Frequency Ranges
- **Standard ECG**: 0.5-40 Hz (removes baseline drift and high-frequency noise)
- **Diagnostic ECG**: 0.5-100 Hz (preserves more diagnostic information)
- **Monitoring ECG**: 0.5-30 Hz (adequate for rhythm monitoring)
- **Pediatric ECG**: 0.5-100 Hz (higher frequencies may be clinically relevant)

### Power Line Interference
- **Europe/Asia**: 50 Hz
- **North America**: 60 Hz

### Filter Order
- Higher order = steeper roll-off but potential instability
- Order 4 is a good compromise for most applications
- For critical applications, consider order 2-6

## Dependencies

- numpy
- scipy (for signal processing)
- matplotlib (for visualization)
- wfdb (for ECG data loading) 