#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

# from helper_code import *
# from implementations.gnn import train_model as train_gnn_model
# from implementations.resnet import train_model as train_resnet_model
from implementations.resnet_trans import train_model as train_resnet_trans_model
from implementations.resnet_trans import load_model as load_resnet_trans_model
from modules.data_pipeline import load_record
# from implementations.moe import train_model as train_moe_model
# from implementations.moe_lstm import train_model as train_moe_lstm_model

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    
    ### FULL TRAIN CONFIG (PRETRAIN + FINETUNE AT ONCE)
    # train_resnet_trans_model(
    #     training_data_folder=data_folder, 
    #     model_folder=model_folder, 
    #     sequence_length=512, 
    #     dropout=0.1,
    #     epochs=200,
    #     lr=5e-4,
    #     batch_size=128,
    #     generate_holdout=False, # if we disable this, we use the actual holdout data, which is the chagas dataset
    #     device='cuda:0', 
    #     false_negative_penalty=1.0,
    #     finetune=False,
    #     train_verbose=verbose,
    #     aggressive_masking=True,  # Re-mask data each epoch for different patterns
    # )
    
    # ## PRETRAIN CONFIG
    # train_moe_model(
    #     training_data_folder=data_folder, 
    #     model_folder=model_folder, 
    #     sequence_length=512, 
    #     dropout=0.1,
    #     epochs=30,
    #     lr=5e-4,
    #     batch_size=256,
    #     generate_holdout=False, # if we disable this, we use the actual holdout data, which is the chagas dataset
    #     device='cuda:0', 
    #     model_path=None,
    #     false_negative_penalty=1.0,
    #     finetune=False,
    #     train_verbose=verbose,
    #     aggressive_masking=True,  # Re-mask data each epoch for different patterns
    # )
    #     # PRETRAIN CONFIG resnet only transformer
    # train_moe_lstm_model(
    #     training_data_folder=data_folder, 
    #     model_folder=model_folder, 
    #     sequence_length=512, 
    #     dropout=0.1,
    #     epochs=40,
    #     lr=5e-4,
    #     batch_size=256,
    #     generate_holdout=True, # if we disable this, we use the actual holdout data, which is the chagas dataset
    #     device='cuda:0', 
    #     model_path=None,
    #     false_negative_penalty=1.0,
    #     finetune=False,
    #     train_verbose=verbose,
    #     aggressive_masking=True,  # Re-mask data each epoch for different patterns
    #     pretrain_transformer_only= True # Freeze ResNet, only train transformer and classifier
    # )
    
    ### SUBMISSION TRAIN CONFIG
    train_resnet_trans_model(
        training_data_folder=data_folder, 
        model_folder=model_folder, 
        sequence_length=512, 
        dropout=0.1,
        epochs=100,
        lr=3e-5,
        batch_size=128,
        generate_holdout=False, # if we disable this, we use the actual holdout data, which is the chagas dataset
        device='cuda:0', 
        model_path="./model/model.pkl",
        finetune=False,
        alternate_finetune=False,
        false_negative_penalty=1.0,
        train_verbose=verbose,
        aggressive_masking=True,  # Re-mask data each epoch for different patterns
        pretrain_transformer_only= False # Freeze ResNet, only train transformer and classifier
    )
    

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    # model_filename = os.path.join(model_folder, 'model.sav')
    # model = joblib.load(model_filename)
    # return model
    model_path = os.path.join(model_folder, 'model.pkl')
    return load_resnet_trans_model(model_path, 
                                   is_finetune=False, device='cuda:0')

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    import torch
    record_data = load_record(record, sequence_length=512)
    signal_tensor = torch.FloatTensor(record_data['signal']).to(model.device)
    signal_tensor = signal_tensor.unsqueeze(0)
    binary_output, probability_output = model.predict(signal_tensor)

    return binary_output.item(), probability_output.item()

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)

    # Extract the age from the record.
    age = get_age(header)
    age = np.array([age])

    # Extract the sex from the record and represent it as a one-hot encoded vector.
    sex = get_sex(header)
    sex_one_hot_encoding = np.zeros(3, dtype=bool)
    if sex.casefold().startswith('f'):
        sex_one_hot_encoding[0] = 1
    elif sex.casefold().startswith('m'):
        sex_one_hot_encoding[1] = 1
    else:
        sex_one_hot_encoding[2] = 1

    # Extract the source from the record (but do not use it as a feature).
    source = get_source(header)

    # Load the signal data and fields. Try fields.keys() to see the fields, e.g., fields['fs'] is the sampling frequency.
    signal, fields = load_signals(record)
    channels = fields['sig_name']

    # Reorder the channels in case they are in a different order in the signal data.
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels = len(reference_channels)
    signal = reorder_signal(signal, channels, reference_channels)

    # Compute two per-channel features as examples.
    signal_mean = np.zeros(num_channels)
    signal_std = np.zeros(num_channels)

    for i in range(num_channels):
        num_finite_samples = np.sum(np.isfinite(signal[:, i]))
        if num_finite_samples > 0:
            signal_mean[i] = np.nanmean(signal)
        else:
            signal_mean = 0.0
        if num_finite_samples > 1:
            signal_std[i] = np.nanstd(signal)
        else:
            signal_std = 0.0

    # Return the features.

    return age, sex_one_hot_encoding, source, signal_mean, signal_std

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)