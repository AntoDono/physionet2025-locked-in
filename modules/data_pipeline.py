import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add the parent directory to the Python path to access helper_code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_code import *
from normalize_wave import normalize_amplitude, wavelet_denoising

def data_stats(df):
    print(f"Number of records: {len(df)}")
    print(f"Number of records with label 1: {df[df['label'] == 1].shape[0]}")
    print(f"Number of records with label 0: {df[df['label'] == 0].shape[0]}")

def load_data(path="./data"):
    
    if os.path.exists("saved_data"):
        df = pd.read_pickle("./saved_data/data.pkl")
        data_stats(df)
        return df
    else:
        os.makedirs("saved_data")
    
    records = find_records(path)
    
    data = []
    
    for record in tqdm(records):
        
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
    df.to_pickle("./saved_data/data.pkl")
    data_stats(df)
    return df

def batch_normalize_data(df):
    for i in tqdm(range(len(df))):
        # Convert to numpy array if it isn't already
        signal = df.iloc[i]['signal']
        
        # Apply normalization and denoising
        signal = normalize_amplitude(signal)
        signal = wavelet_denoising(signal)
        
        # Assign back to dataframe using .at for proper assignment
        df.at[i, 'signal'] = signal
    return df
            
df = load_data()
df = batch_normalize_data(df)
print(df.head())