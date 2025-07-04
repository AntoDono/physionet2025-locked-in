import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import pywt
from scipy.ndimage import median_filter

def band_pass_filter(ecg_signal, sampling_frequency, low_cutoff=0.5, high_cutoff=40.0, order=4):
    """
    Apply a band-pass filter to ECG signal data.
    
    This function applies a Butterworth band-pass filter to remove baseline drift 
    and high-frequency noise from ECG signals. The default frequency range (0.5-40 Hz) 
    is suitable for standard ECG analysis.
    
    Args:
        ecg_signal (numpy.ndarray): ECG signal data. Can be 1D (single lead) or 2D (multiple leads).
                                   If 2D, shape should be (samples, leads).
        sampling_frequency (float): Sampling frequency of the ECG signal in Hz.
        low_cutoff (float, optional): Low cutoff frequency in Hz. Default is 0.5 Hz.
        high_cutoff (float, optional): High cutoff frequency in Hz. Default is 40.0 Hz.
        order (int, optional): Filter order. Default is 4.
    
    Returns:
        numpy.ndarray: Filtered ECG signal with the same shape as input.
        
    Raises:
        ValueError: If cutoff frequencies are invalid or sampling frequency is too low.
        
    Example:
        # For a single lead
        filtered_signal = band_pass_filter(signal_1d, 400)
        
        # For multiple leads with custom frequency range
        filtered_signal = band_pass_filter(signal_2d, 400, low_cutoff=1.0, high_cutoff=30.0)
    """
    
    # Input validation
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("ECG signal must be a numpy array")
    
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be positive")
    
    if low_cutoff <= 0:
        raise ValueError("Low cutoff frequency must be positive")
    
    if high_cutoff <= low_cutoff:
        raise ValueError("High cutoff frequency must be greater than low cutoff frequency")
    
    # Check Nyquist frequency
    nyquist_freq = sampling_frequency / 2.0
    if high_cutoff >= nyquist_freq:
        raise ValueError(f"High cutoff frequency ({high_cutoff} Hz) must be less than Nyquist frequency ({nyquist_freq} Hz)")
    
    # Normalize frequencies to Nyquist frequency
    low_normalized = low_cutoff / nyquist_freq
    high_normalized = high_cutoff / nyquist_freq
    
    # Design Butterworth band-pass filter
    b, a = butter(order, [low_normalized, high_normalized], btype='band', analog=False)
    
    # Handle different input shapes
    original_shape = ecg_signal.shape
    
    if ecg_signal.ndim == 1:
        # Single lead signal
        filtered_signal = filtfilt(b, a, ecg_signal)
    elif ecg_signal.ndim == 2:
        # Multiple leads signal - filter each lead separately
        num_samples, num_leads = ecg_signal.shape
        filtered_signal = np.zeros_like(ecg_signal)
        
        for lead_idx in range(num_leads):
            filtered_signal[:, lead_idx] = filtfilt(b, a, ecg_signal[:, lead_idx])
    else:
        raise ValueError("ECG signal must be 1D or 2D array")
    
    return filtered_signal


def apply_notch_filter(ecg_signal, sampling_frequency, notch_freq=50.0, quality_factor=30):
    """
    Apply a notch filter to remove power line interference from ECG signals.
    
    Args:
        ecg_signal (numpy.ndarray): ECG signal data.
        sampling_frequency (float): Sampling frequency in Hz.
        notch_freq (float, optional): Frequency to notch out in Hz. Default is 50 Hz (Europe).
                                     Use 60 Hz for North America.
        quality_factor (float, optional): Quality factor of the notch filter. Default is 30.
    
    Returns:
        numpy.ndarray: ECG signal with power line interference removed.
    """
    
    # Input validation
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("ECG signal must be a numpy array")
    
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be positive")
    
    nyquist_freq = sampling_frequency / 2.0
    if notch_freq >= nyquist_freq:
        raise ValueError(f"Notch frequency ({notch_freq} Hz) must be less than Nyquist frequency ({nyquist_freq} Hz)")
    
    # Design notch filter
    b, a = signal.iirnotch(notch_freq, quality_factor, sampling_frequency)
    
    # Apply filter
    if ecg_signal.ndim == 1:
        filtered_signal = filtfilt(b, a, ecg_signal)
    elif ecg_signal.ndim == 2:
        num_samples, num_leads = ecg_signal.shape
        filtered_signal = np.zeros_like(ecg_signal)
        
        for lead_idx in range(num_leads):
            filtered_signal[:, lead_idx] = filtfilt(b, a, ecg_signal[:, lead_idx])
    else:
        raise ValueError("ECG signal must be 1D or 2D array")
    
    return filtered_signal


def median_filter_noise_reduction(ecg_signal, kernel_size=3):
    """
    Apply median filtering to remove impulse noise and artifacts from ECG signals.
    
    Median filtering is effective at removing salt-and-pepper noise and impulse artifacts
    while preserving the overall signal morphology and sharp features like QRS complexes.
    
    Args:
        ecg_signal (numpy.ndarray): ECG signal data. Can be 1D (single lead) or 2D (multiple leads).
        kernel_size (int, optional): Size of the median filter kernel. Default is 3.
                                   Larger values provide more smoothing but may blur signal features.
    
    Returns:
        numpy.ndarray: ECG signal with impulse noise removed.
        
    Example:
        # Remove impulse noise from single lead
        denoised_signal = median_filter_noise_reduction(noisy_signal, kernel_size=5)
    """
    
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("ECG signal must be a numpy array")
    
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer")
    
    if ecg_signal.ndim == 1:
        # Single lead signal
        filtered_signal = median_filter(ecg_signal, size=kernel_size)
    elif ecg_signal.ndim == 2:
        # Multiple leads signal - filter each lead separately
        filtered_signal = np.zeros_like(ecg_signal)
        for lead_idx in range(ecg_signal.shape[1]):
            filtered_signal[:, lead_idx] = median_filter(ecg_signal[:, lead_idx], size=kernel_size)
    else:
        raise ValueError("ECG signal must be 1D or 2D array")
    
    return filtered_signal


def wavelet_denoising(ecg_signal, wavelet='db4', mode='soft', sigma=None):
    """
    Apply wavelet denoising for sophisticated noise removal while preserving signal morphology.
    
    Wavelet denoising is particularly effective for ECG signals as it can remove noise
    while preserving important morphological features like P waves, QRS complexes, and T waves.
    
    Args:
        ecg_signal (numpy.ndarray): ECG signal data. Can be 1D (single lead) or 2D (multiple leads).
        wavelet (str, optional): Wavelet to use for decomposition. Default is 'db4' (Daubechies 4).
                                Other good options: 'db6', 'db8', 'coif2', 'coif4', 'bior4.4'.
        mode (str, optional): Thresholding mode. 'soft' or 'hard'. Default is 'soft'.
                             Soft thresholding generally preserves signal morphology better.
        sigma (float, optional): Noise standard deviation. If None, estimated from signal.
    
    Returns:
        numpy.ndarray: Denoised ECG signal with preserved morphology.
        
    Example:
        # Denoise with default parameters
        denoised_signal = wavelet_denoising(noisy_signal)
        
        # Denoise with custom wavelet and parameters
        denoised_signal = wavelet_denoising(noisy_signal, wavelet='db6', mode='soft')
    """
    
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("ECG signal must be a numpy array")
    
    if wavelet not in pywt.wavelist():
        raise ValueError(f"Wavelet '{wavelet}' not supported. Use pywt.wavelist() to see available wavelets")
    
    if mode not in ['soft', 'hard']:
        raise ValueError("Mode must be 'soft' or 'hard'")
    
    def denoise_1d(signal_1d):
        # Wavelet decomposition
        coeffs = pywt.wavedec(signal_1d, wavelet, mode='symmetric')
        
        # Estimate noise standard deviation if not provided
        if sigma is None:
            # Estimate sigma from the finest detail coefficients
            sigma_est = np.median(np.abs(coeffs[-1])) / 0.6745
        else:
            sigma_est = sigma
        
        # Calculate threshold using Donoho-Johnstone threshold
        threshold = sigma_est * np.sqrt(2 * np.log(len(signal_1d)))
        
        # Apply thresholding to detail coefficients
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode=mode) for detail in coeffs[1:]]
        
        # Reconstruct signal
        denoised = pywt.waverec(coeffs_thresh, wavelet, mode='symmetric')
        
        # Ensure the denoised signal has the same length as the input
        if len(denoised) != len(signal_1d):
            if len(denoised) > len(signal_1d):
                # Truncate if too long
                denoised = denoised[:len(signal_1d)]
            else:
                # Pad with zeros if too short
                denoised = np.pad(denoised, (0, len(signal_1d) - len(denoised)), mode='constant')
        
        return denoised
    
    if ecg_signal.ndim == 1:
        # Single lead signal
        denoised_signal = denoise_1d(ecg_signal)
    elif ecg_signal.ndim == 2:
        # Multiple leads signal - denoise each lead separately
        denoised_signal = np.zeros_like(ecg_signal)
        for lead_idx in range(ecg_signal.shape[1]):
            denoised_signal[:, lead_idx] = denoise_1d(ecg_signal[:, lead_idx])
    else:
        raise ValueError("ECG signal must be 1D or 2D array")
    
    return denoised_signal


def normalize_amplitude(ecg_signal, method='z_score', target_range=(-1, 1)):
    """
    Normalize ECG signal amplitude to standardize signal ranges across different recordings.
    
    This function provides multiple normalization methods to standardize ECG signal amplitudes,
    which is crucial for comparative analysis and machine learning applications.
    
    Args:
        ecg_signal (numpy.ndarray): ECG signal data. Can be 1D (single lead) or 2D (multiple leads).
        method (str): Normalization method. Options: 'z_score', 'min_max', 'robust', 'amplitude'.
        target_range (tuple, optional): Target range for min_max and amplitude normalization. 
                                       Default is (-1, 1).
    
    Returns:
        numpy.ndarray: Normalized ECG signal.
        
    Methods:
        - 'z_score': Zero mean, unit variance (mean=0, std=1) - ideal for comparative analysis
        - 'min_max': Scale to target range (default: -1 to 1)
        - 'robust': Use median and IQR for outlier-resistant normalization
        - 'amplitude': Normalize by maximum absolute amplitude to target range
        
    Example:
        # Z-score normalization for comparative analysis
        normalized_signal = normalize_amplitude(signal, method='z_score')
        
        # Amplitude normalization to [-1, 1] range
        normalized_signal = normalize_amplitude(signal, method='amplitude', target_range=(-1, 1))
    """
    
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("ECG signal must be a numpy array")
    
    if len(target_range) != 2 or target_range[0] >= target_range[1]:
        raise ValueError("Target range must be a tuple (min, max) with min < max")
    
    def normalize_1d(signal_1d, method):
        if method == 'z_score':
            # Z-score normalization (mean=0, std=1) - ideal for comparative analysis
            mean_val = np.mean(signal_1d)
            std_val = np.std(signal_1d)
            if std_val == 0:
                return np.zeros_like(signal_1d)
            return (signal_1d - mean_val) / std_val
            
        elif method == 'min_max':
            # Min-max normalization to target range
            min_val, max_val = np.min(signal_1d), np.max(signal_1d)
            if max_val == min_val:
                return np.full_like(signal_1d, (target_range[0] + target_range[1]) / 2)
            normalized = (signal_1d - min_val) / (max_val - min_val)
            return normalized * (target_range[1] - target_range[0]) + target_range[0]
            
        elif method == 'robust':
            # Robust normalization using median and IQR
            median_val = np.median(signal_1d)
            q75, q25 = np.percentile(signal_1d, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                return np.zeros_like(signal_1d)
            return (signal_1d - median_val) / iqr
            
        elif method == 'amplitude':
            # Amplitude normalization - normalize by maximum absolute value
            max_abs = np.max(np.abs(signal_1d))
            if max_abs == 0:
                return np.zeros_like(signal_1d)
            normalized = signal_1d / max_abs
            # Scale to target range
            return normalized * max(abs(target_range[0]), abs(target_range[1]))
        else:
            raise ValueError("Method must be 'z_score', 'min_max', 'robust', or 'amplitude'")
    
    if ecg_signal.ndim == 1:
        # Single lead signal
        normalized_signal = normalize_1d(ecg_signal, method)
    elif ecg_signal.ndim == 2:
        # Multiple leads signal - normalize each lead separately
        normalized_signal = np.zeros_like(ecg_signal)
        for lead_idx in range(ecg_signal.shape[1]):
            normalized_signal[:, lead_idx] = normalize_1d(ecg_signal[:, lead_idx], method)
    else:
        raise ValueError("ECG signal must be 1D or 2D array")
    
    return normalized_signal

def pad_signal(ecg_signal, padding_length=2048):
    """
    Pad or truncate a single ECG signal to a fixed length.

    This function takes a single ECG signal and pads it with zeros or truncates
    it to ensure it has a specified length.

    Args:
        ecg_signal (numpy.ndarray): A 1D or 2D ECG signal array in the format
                                    (# samples) or (# samples, # leads).
        padding_length (int, optional): The target length for the signal. 
                                       Default is 2048.
    
    Returns:
        numpy.ndarray: The signal, padded or truncated to `padding_length`.
    """
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("ECG signal must be a numpy array")

    if ecg_signal.ndim not in [1, 2]:
        raise ValueError("ECG signal must be a 1D or 2D array")

    current_length = ecg_signal.shape[0]

    if current_length == padding_length:
        return ecg_signal
    
    if current_length > padding_length:
        # Truncate the signal
        return ecg_signal[:padding_length]
    else:
        # Pad the signal
        pad_width = padding_length - current_length
        if ecg_signal.ndim == 1:
            # Pad a 1D signal
            return np.pad(ecg_signal, (0, pad_width), mode='constant')
        else: # ndim == 2
            # Pad a 2D signal only on the sample axis (axis 0)
            return np.pad(ecg_signal, ((0, pad_width), (0, 0)), mode='constant')

def trim_signal(ecg_signal, length=2048):
    """
    Trim a single ECG signal to a fixed length if it is longer.

    This function takes a single ECG signal and trims it to a specified length.
    If the signal is shorter than or equal to the target length, it is returned unmodified.
    
    Args:
        ecg_signal (numpy.ndarray): A 1D or 2D ECG signal array in the format
                                    (# samples) or (# samples, # leads).
        length (int, optional): The maximum target length for the signal. Default is 2048.

    Returns:
        numpy.ndarray: The signal, trimmed if it was longer than `length`.
    """
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("ECG signal must be a numpy array")

    if ecg_signal.ndim not in [1, 2]:
        raise ValueError("ECG signal must be a 1D or 2D array")

    if ecg_signal.shape[0] > length:
        return ecg_signal[:length]
    else:
        return ecg_signal
        
