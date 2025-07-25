import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

# Add the parent directory to the Python path to access helper_code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_code import *

try:
    from normalize_wave import normalize_amplitude, wavelet_denoising, trim_signal, pad_signal
except ImportError:
    from .normalize_wave import normalize_amplitude, wavelet_denoising, trim_signal, pad_signal

from scipy import signal  # Added for signal resampling

def data_stats(df):
    """
    Print statistics about the data
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data
        
    Returns:
        None
    """
    print("=" * 60)
    print(f"Size: {len(df)} | Positives: {df[df['label'] == 1].shape[0]} | Negatives: {df[df['label'] == 0].shape[0]}")
    print(f"Ratio: {df[df['label'] == 1].shape[0] / df[df['label'] == 0].shape[0]:.2f}")
    print("=" * 60)

def load_record(record_path, sequence_length=512):
    """
    Load a single record from the given path
    
    Args:
        record_path (str): The path to the record
        sequence_length (int): The desired sequence length for normalization
        
    Returns:
        dict: A dictionary containing the record data, or None if loading fails
    """
    try:
        label = load_label(record_path)
        header = load_header(record_path)
        age = load_age(record_path)
        sex = load_sex(record_path)
        source = load_source(record_path)
        signal, fields = load_signals(record_path)
        num_leads = get_num_signals(header)
        frequency = get_sampling_frequency(header)
        signal_names = get_signal_names(header)
        
        # Extract record name from path
        record_name = os.path.basename(record_path)
        
        reordered_signal = reorder_signal(signal, signal_names, ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"])
        
        record_data = {
            "id": record_name,
            "label": label, 
            "age": age, 
            "sex": sex, 
            "source": source, 
            "num_leads": num_leads, 
            "frequency": frequency, 
            "signal_names": signal_names,
            "signal": reordered_signal
        }
        
        # Apply normalization to the single record
        record_data = normalize_single_record(record_data, sequence_length)
        # Apply amplitude normalization
            
        return record_data
        
    except FileNotFoundError:
        print(f"File not found: {record_path}")
        return None
    except Exception as e:
        print(f"Error loading record: {e}")
        return None

def normalize_single_record(record_data, length=512, target_frequency=400):
    """
    Normalize a single record's signal data
    
    Args:
        record_data (dict): Dictionary containing record data with 'signal' and 'frequency' keys
        length (int): Desired sequence length
        target_frequency (int): Target sampling frequency in Hz
        
    Returns:
        dict: Updated record data with normalized signal
    """
    ecg_signal = record_data['signal']
    orig_freq = record_data.get('frequency', None)
    
    # If frequency information is available and different, resample
    if orig_freq and not np.isnan(orig_freq) and orig_freq != target_frequency:
        num_samples = ecg_signal.shape[0]
        target_samples = int(round(num_samples * target_frequency / orig_freq))
        # Resample along the time axis (axis 0)
        ecg_signal = signal.resample(ecg_signal, target_samples, axis=0)
    
    # Apply amplitude normalization
    ecg_signal = normalize_amplitude(ecg_signal)
    
    # Trim and pad to fixed length
    ecg_signal = trim_signal(ecg_signal, length)
    ecg_signal = pad_signal(ecg_signal, length)
    
    # Update the record data
    record_data['signal'] = ecg_signal
    
    return record_data

def load_data(path="./data", name="data", num_records=None, use_cache=True, sequence_length=512):
    """
    Load the data from the path
    
    Args:
        path (str): The path to the data
        use_cache (bool): Whether to use the cached data
        
    Returns:
        pandas.DataFrame: The dataframe containing the data
    """
    
    if os.path.exists("saved_data") and use_cache:
        df = pd.read_pickle(f"./saved_data/{name}.pkl")
        if num_records is not None:
            df = df.sample(num_records, random_state=42)
        data_stats(df)
        return df
    else:
        if not os.path.exists("saved_data"):
            os.makedirs("saved_data")
    
    records = find_records(path)
    random.seed(42)
    random.shuffle(records)
    
    if num_records is not None:
        records = records[:num_records]
    
    data = []
    pbar = tqdm(records)
    pbar.set_description("Fetching data")
    
    for record in pbar:
        
        record_path = f"{path}/{record}"
        
        try:
            label = load_label(record_path)
            header = load_header(record_path)
            # probability = get_probability(record_path)
            age = load_age(record_path)
            sex = load_sex(record_path)
            source = load_source(record_path)
            signal, fields = load_signals(record_path)
            num_leads = get_num_signals(header)
            frequency = get_sampling_frequency(header)
            signal_names = get_signal_names(header)
            
            reordered_signal = reorder_signal(signal, signal_names, ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"])
            data.append({
                "id": record,
                "label": label, 
                "age": age, 
                "sex": sex, 
                "source": source, 
                "num_leads": num_leads, 
                "frequency": frequency, 
                "signal_names": signal_names,
                "signal": reordered_signal})
    
        except FileNotFoundError:
            print(f"File not found: {record_path}")
            continue
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
        
    df = pd.DataFrame(data)
    df = batch_normalize_data(df, length=sequence_length)
    df.to_pickle(f"./saved_data/{name}.pkl")
    data_stats(df)
    return df

def batch_normalize_data(df, length=512, target_frequency=400):
    """
    Resample each ECG signal to a common sampling frequency, then normalize, trim, and pad it.

    Args:
        df (pd.DataFrame): DataFrame containing a column ``signal`` with ECG data and a column ``frequency``
                           with the original sampling frequency of each record.
        length (int, optional): Desired number of samples after trimming (before optional padding). Default 512.
        target_frequency (int, optional): Target sampling frequency in Hz. Default 400 Hz.

    Returns:
        pd.DataFrame: DataFrame with the ``signal`` column updated to the resampled, normalized, trimmed, and padded signals.
    """

    pbar = tqdm(range(len(df)))
    pbar.set_description("Resampling Normalizing Denoising Padding Data")

    for i in pbar:
        # Retrieve signal and its original sampling frequency
        ecg_signal = df.iloc[i]['signal']  # shape: (samples, leads)
        orig_freq = df.iloc[i].get('frequency', None)

        # If frequency information is available and different, resample
        if orig_freq and not np.isnan(orig_freq) and orig_freq != target_frequency:
            num_samples = ecg_signal.shape[0]
            target_samples = int(round(num_samples * target_frequency / orig_freq))
            # Resample along the time axis (axis 0)
            ecg_signal = signal.resample(ecg_signal, target_samples, axis=0)
            pbar.set_description(f"Resampling from {orig_freq} Hz to {target_frequency} Hz")

        # Apply amplitude normalization
        ecg_signal = normalize_amplitude(ecg_signal)

        # Trim and pad to fixed length
        ecg_signal = trim_signal(ecg_signal, length)
        ecg_signal = pad_signal(ecg_signal, length)

        # Save back to DataFrame
        df.at[i, 'signal'] = ecg_signal

    return df

def generate_masked_ecg(row, *, num_lead_masks: int = 1, num_span_masks: int = 1,
                        lead_mask_ratio: float = 0.25, span_length: int | None = None,
                        span_mask_ratio: float = 0.1, mask_value: float = 0.0,
                        copy_row: bool = True):
    """Generate masked ECG samples (lead‐masking and span‐masking) from one DataFrame row.

    Given a single row coming _after_ ``batch_normalize_data`` (thus containing a
    fixed-length, fully-processed ``signal`` ndarray), this utility creates
    several augmented versions of that signal by masking either complete leads
    (columns) or contiguous spans along the time axis.

    Args:
        row (pd.Series): A row produced by ``batch_normalize_data`` that at
            minimum includes a ``signal`` field containing a 2-D ndarray of
            shape (samples, leads).
        num_lead_masks (int, optional): How many lead-masked samples to
            generate from the input row. Default 1.
        num_span_masks (int, optional): How many span-masked samples to
            generate from the input row. Default 1.
        lead_mask_ratio (float, optional): Fraction of the total leads to mask
            in each lead-masked sample. At least one lead will always be masked.
            Default 0.25 (25 %).
        span_length (int | None, optional): Exact length (in samples) of the
            contiguous time span to mask for span-masking. If *None*, the span
            length is derived from ``span_mask_ratio`` instead. Default None.
        span_mask_ratio (float, optional): When ``span_length`` is *None*, this
            parameter specifies the fraction of the total signal length to mask
            along the time axis. Default 0.10 (10 %).
        mask_value (float, optional): The value used to fill the masked
            elements. Default 0.0.
        copy_row (bool, optional): Whether to copy non-signal fields when
            building the masked samples. Set to *True* if you intend to append
            the returned samples back into a DataFrame; *False* returns only
            the modified signals. Default True.

    Returns:
        list[dict | np.ndarray]: A list containing the newly-generated masked
            samples. By default each element is a ``dict`` mirroring the input
            row with an updated ``signal`` entry plus metadata about the mask
            that was applied. If ``copy_row`` is *False* the list will instead
            contain masked ``np.ndarray`` objects.
    """

    # Extract original signal and basic dimensions
    original_signal = row['signal']
    if original_signal is None:
        raise ValueError("Row must contain a 'signal' field with ECG data.")

    # Ensure we are working with a NumPy array
    signal_array = np.asarray(original_signal)
    if signal_array.ndim != 2:
        raise ValueError("'signal' must be a 2-D array of shape (samples, leads).")

    num_samples, num_leads = signal_array.shape
    masked_samples = []

    # ---------- Lead masking ----------
    if num_lead_masks > 0 and lead_mask_ratio > 0:
        n_mask_leads = max(1, int(round(num_leads * lead_mask_ratio)))
        for _ in range(num_lead_masks):
            masked = signal_array.copy()
            leads_to_mask = np.random.choice(num_leads, n_mask_leads, replace=False)
            masked[:, leads_to_mask] = mask_value

            if copy_row:
                sample = row.to_dict()
                sample['signal'] = masked
                sample['mask_type'] = 'lead'
                sample['masked_leads'] = leads_to_mask.tolist()
                masked_samples.append(sample)
            else:
                masked_samples.append(masked)

    # ---------- Span masking ----------
    if num_span_masks > 0 and (span_length or span_mask_ratio):
        if span_length is None:
            span_length = max(1, int(round(num_samples * span_mask_ratio)))
        span_length = min(span_length, num_samples)

        for _ in range(num_span_masks):
            masked = signal_array.copy()
            start_idx = np.random.randint(0, num_samples - span_length + 1)
            end_idx = start_idx + span_length
            masked[start_idx:end_idx, :] = mask_value

            if copy_row:
                sample = row.to_dict()
                sample['signal'] = masked
                sample['mask_type'] = 'span'
                sample['masked_span'] = (int(start_idx), int(end_idx))
                masked_samples.append(sample)
            else:
                masked_samples.append(masked)

    return masked_samples

def mask_ecg(df: pd.DataFrame, *, num_lead_masks: int = 1, num_span_masks: int = 1,
             lead_mask_ratio: float = 0.25, span_length: int | None = None,
             span_mask_ratio: float = 0.1, mask_value: float = 0.0, include_original: bool = True) -> pd.DataFrame:
    """Augment an ECG DataFrame by applying lead and/or span masking to every row.

    This function iterates over *df*, generates masked variants for each row via
    ``generate_masked_ecg``, and returns a new DataFrame containing both the
    original rows and all newly created masked rows.

    Parameters mirror those of ``generate_masked_ecg`` and are forwarded as-is.

    Args:
        df (pd.DataFrame): DataFrame that has already been processed by
            ``batch_normalize_data``.
        num_lead_masks, num_span_masks, lead_mask_ratio, span_length,
        span_mask_ratio, mask_value: See ``generate_masked_ecg``.

    Returns:
        pd.DataFrame: Concatenation of the original *df* and the generated
        masked samples (rows are shuffled for randomness).
    """

    if 'signal' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'signal' column. Make sure to run batch_normalize_data first.")

    augmented_rows = []
    pbar = tqdm(df.iterrows(), total=len(df), desc="Masking ECG data")

    for _, row in pbar:
        # Save original row (add explicit mask_type for clarity)
        orig_row = row.to_dict()
        orig_row.setdefault('mask_type', 'original')
        
        if include_original:
            augmented_rows.append(orig_row)

        # Generate masked variants
        masked_variants = generate_masked_ecg(
            row,
            num_lead_masks=num_lead_masks,
            num_span_masks=num_span_masks,
            lead_mask_ratio=lead_mask_ratio,
            span_length=span_length,
            span_mask_ratio=span_mask_ratio,
            mask_value=mask_value,
            copy_row=True,
        )
        augmented_rows.extend(masked_variants)

    augmented_df = pd.DataFrame(augmented_rows)
    # Shuffle rows to mix originals and masks
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return augmented_df

def balance_data(df, strategy='oversample', num_records=None, ratio=1):
    """
    Balance the data by either over-sampling the minority class or
    down-sampling the majority class.

    A warning is issued if the class imbalance ratio is over 10 when
    oversampling, as this may lead to overfitting.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        strategy (str): 'oversample' or 'downsample'.
        num_records (int, optional): Total number of records to sample after balancing.
        ratio (float): The negative to positive ratio. Default is 1 (balanced).
                      For 'downsample': controls how many times larger the majority class 
                      should be compared to the minority class.
                      For 'oversample': always uses ratio=1 (balanced) regardless of input.
    
    Returns:
        pd.DataFrame: A balanced dataframe.
    """
    df_1 = df[df['label'] == 1]
    df_0 = df[df['label'] == 0]

    if len(df_1) == len(df_0) and ratio == 1:
        return df.copy()

    if len(df_1) > len(df_0):
        majority_df, minority_df = df_1, df_0
    else:
        majority_df, minority_df = df_0, df_1

    if len(minority_df) == 0:
        print("WARNING: Minority class has no samples. Cannot balance data.")
        return df

    if strategy == 'oversample':
        if ratio != 1:
            print(f"WARNING: ratio parameter ({ratio}) is ignored for oversampling. Using balanced ratio=1.")
        
        imbalance_ratio = len(majority_df) / len(minority_df)
        if imbalance_ratio > 10:
            print("\n" + "!"*60)
            print("! WARNING: High class imbalance detected for oversampling.")
            print(f"! Majority: {len(majority_df)} samples. Minority: {len(minority_df)} samples.")
            print(f"! Ratio: {imbalance_ratio:.1f} to 1.")
            print("! Oversampling may lead to overfitting.")
            print("!"*60 + "\n")
        
        resampled_minority = minority_df.sample(len(majority_df), replace=True, random_state=42)
        balanced_df = pd.concat([majority_df, resampled_minority])
    
    elif strategy == 'downsample':
        # Calculate target sizes based on ratio
        minority_size = len(minority_df)
        majority_target_size = int(minority_size * ratio)
        
        # Ensure we don't try to sample more than what's available
        if majority_target_size > len(majority_df):
            print(f"WARNING: Requested ratio ({ratio}) would require {majority_target_size} majority samples, ")
            print(f"but only {len(majority_df)} available. Using all available majority samples.")
            majority_target_size = len(majority_df)
            actual_ratio = majority_target_size / minority_size
            print(f"Actual ratio will be: {actual_ratio:.2f}")
        
        resampled_majority = majority_df.sample(majority_target_size, replace=False, random_state=42)
        balanced_df = pd.concat([resampled_majority, minority_df])
        
    else:
        raise ValueError("Strategy must be 'oversample' or 'downsample'")
    
    if num_records is not None:
        balanced_df = balanced_df.sample(num_records, random_state=42)
    else:
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    data_stats(balanced_df)
    return balanced_df
            
            
if __name__ == "__main__":
    from visualize import visualize_ecg
    df = load_data(use_cache=False, num_records=5000)
    df = batch_normalize_data(df, length=512)
    df = balance_data(df, strategy='oversample')
    print(df['label'].value_counts())
    print("Example signal shape: ", df.iloc[0]['signal'].shape)
    visualize_ecg(df.iloc[0]['signal'])