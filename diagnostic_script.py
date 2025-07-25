#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import sys
import os

# Add the evaluation directory to the path
sys.path.append('evaluation-2025')
from helper_code import compute_challenge_score

def diagnose_model_probabilities(model, val_loader, device='cpu'):
    """
    Diagnose model probability distributions and calibration
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for ecg_data, labels in val_loader:
            ecg_data = ecg_data.to(device)
            ecg_data = ecg_data.unsqueeze(0)
            labels = labels.to(device)
            
            logits = model(ecg_data)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels_np)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print("=== PROBABILITY DISTRIBUTION ANALYSIS ===")
    print(f"Total samples: {len(all_probs)}")
    print(f"Positive cases: {np.sum(all_labels)} ({100*np.mean(all_labels):.1f}%)")
    print()
    
    print("Overall probability statistics:")
    print(f"  Min probability: {np.min(all_probs):.4f}")
    print(f"  Max probability: {np.max(all_probs):.4f}")
    print(f"  Mean probability: {np.mean(all_probs):.4f}")
    print(f"  Std probability: {np.std(all_probs):.4f}")
    print()
    
    print("Probability statistics by class:")
    pos_probs = all_probs[all_labels == 1]
    neg_probs = all_probs[all_labels == 0]
    
    print(f"  Positive cases (n={len(pos_probs)}):")
    print(f"    Mean: {np.mean(pos_probs):.4f}")
    print(f"    Std:  {np.std(pos_probs):.4f}")
    print(f"    Min:  {np.min(pos_probs):.4f}")
    print(f"    Max:  {np.max(pos_probs):.4f}")
    
    print(f"  Negative cases (n={len(neg_probs)}):")
    print(f"    Mean: {np.mean(neg_probs):.4f}")
    print(f"    Std:  {np.std(neg_probs):.4f}")
    print(f"    Min:  {np.min(neg_probs):.4f}")
    print(f"    Max:  {np.max(neg_probs):.4f}")
    print()
    
    # Calculate challenge score
    challenge_score = compute_challenge_score(all_labels, all_probs)
    print(f"Challenge Score: {challenge_score:.6f}")
    print()
    
    # Check how many positives are in top percentiles
    sorted_indices = np.argsort(all_probs)[::-1]  # Highest to lowest
    sorted_labels = all_labels[sorted_indices]
    
    for percentile in [1, 2, 5, 10, 20]:
        n_top = int(len(all_probs) * percentile / 100)
        top_positives = np.sum(sorted_labels[:n_top])
        total_positives = np.sum(all_labels)
        
        print(f"Top {percentile}% contains {top_positives}/{total_positives} positives ({100*top_positives/total_positives:.1f}%)")
    
    print()
    print("=== POTENTIAL ISSUES ===")
    
    # Check for conservative probabilities
    if np.max(all_probs) < 0.8:
        print("❌ ISSUE: Maximum probability is very low (<0.8)")
        print("   Your model is being too conservative!")
    
    if np.min(all_probs) > 0.2:
        print("❌ ISSUE: Minimum probability is high (>0.2)")
        print("   Your model is not confident about negatives!")
        
    if abs(np.mean(pos_probs) - np.mean(neg_probs)) < 0.2:
        print("❌ ISSUE: Small difference between positive and negative probabilities")
        print("   Your model can't distinguish classes well!")
        
    if challenge_score < 0.2:
        print("❌ ISSUE: Very low challenge score")
        print("   Your probability ranking is poor!")
    
    return {
        'all_probs': all_probs,
        'all_labels': all_labels,
        'challenge_score': challenge_score,
        'pos_mean': np.mean(pos_probs),
        'neg_mean': np.mean(neg_probs)
    }

if __name__ == "__main__":
    # You would call this with your trained model
    from implementations.resnet_trans import load_model, ECGDatasetFromPipeline
    from modules.data_pipeline import load_data
    model = load_model("./model/model.pkl", is_finetune=False, device='cuda:0')
    df = load_data(path="./holdout_data", use_cache=False, sequence_length=512)
    val_loader = ECGDatasetFromPipeline(df)
    diagnose_model_probabilities(model, val_loader, device='cuda:0')
    print("Run this script with your trained model to diagnose probability issues!") 