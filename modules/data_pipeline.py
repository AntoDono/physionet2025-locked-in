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
    
def load_data(path="./data", num_records=None, use_cache=True, sequence_length=512):
    """
    Load the data from the path
    
    Args:
        path (str): The path to the data
        use_cache (bool): Whether to use the cached data
        
    Returns:
        pandas.DataFrame: The dataframe containing the data
    """
    
    if os.path.exists("saved_data") and use_cache:
        df = pd.read_pickle("./saved_data/data.pkl")
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
    df.to_pickle("./saved_data/data.pkl")
    data_stats(df)
    return df

def batch_normalize_data(df, length=512):
    pbar = tqdm(range(len(df)))
    pbar.set_description("Normalizing Denoising Padding Data")
    for i in pbar:
        # Convert to numpy array if it isn't already
        signal = df.iloc[i]['signal'] # Signal is in (# samples, # leads) format
        
        # Apply normalization and denoising
        # signal = wavelet_denoising(signal) # this shit is so ass :sob:
        signal = normalize_amplitude(signal)
        signal = trim_signal(signal, length)
        signal = pad_signal(signal, length)
        
        # Assign back to dataframe using .at for proper assignment
        df.at[i, 'signal'] = signal
    return df

def balance_data(df, strategy='oversample', num_records=None):
    """
    Balance the data by either over-sampling the minority class or
    down-sampling the majority class.

    A warning is issued if the class imbalance ratio is over 10 when
    oversampling, as this may lead to overfitting.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        strategy (str): 'oversample' or 'downsample'.
    
    Returns:
        pd.DataFrame: A balanced dataframe.
    """
    df_1 = df[df['label'] == 1]
    df_0 = df[df['label'] == 0]

    if len(df_1) == len(df_0):
        return df.copy()

    if len(df_1) > len(df_0):
        majority_df, minority_df = df_1, df_0
    else:
        majority_df, minority_df = df_0, df_1

    if len(minority_df) == 0:
        print("WARNING: Minority class has no samples. Cannot balance data.")
        return df

    if strategy == 'oversample':
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
        resampled_majority = majority_df.sample(len(minority_df), replace=False, random_state=42)
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