#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import sys
import os

# Add the evaluation directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.evaluation_helper_code import compute_challenge_metrics

def diagnose_model_probabilities(model, val_loader, device='cpu', verbose=True):
    """
    Diagnose model probability distributions and calibration
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for ecg_data, labels in val_loader:
            ecg_data = ecg_data.to(device)
            
            if ecg_data.dim() < 3:
                ecg_data = ecg_data.unsqueeze(0)
                
            labels = labels.detach().cpu().numpy().flatten()
            
            binary, probability = model.predict(ecg_data)
            
            all_probs.extend(probability)
            all_labels.extend(labels)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    pos_probs = all_probs[all_labels == 1]
    neg_probs = all_probs[all_labels == 0]
    
    
    
    # Calculate challenge score
    challenge_metrics = compute_challenge_metrics(all_labels, all_probs)
    
    if verbose:
        
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
        
        print(f"Challenge Score: {challenge_metrics['challenge_score']:.6f}")
        print(f"TP: {challenge_metrics['tp']}, FN: {challenge_metrics['fn']}, TN: {challenge_metrics['tn']}, FP: {challenge_metrics['fp']}")
        
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
        
        print("=== Challenge Score Diagnostics ===")
        
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
            
        if challenge_metrics['challenge_score'] < 0.2:
            print("❌ ISSUE: Very low challenge score")
            print("   Your probability ranking is poor!")
    
    return {
        'all_probs': all_probs,
        'all_labels': all_labels,
        'challenge_score': challenge_metrics['challenge_score'],
        'tp': challenge_metrics['tp'],
        'fn': challenge_metrics['fn'],
        'tn': challenge_metrics['tn'],
        'fp': challenge_metrics['fp'],
    }

def print_ascii_confusion_matrix(tp, tn, fp, fn):
    """Print an ASCII confusion matrix"""
    # Convert to integers in case they're floats
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)
    
    print("\n=== CONFUSION MATRIX ===")
    print("                 Predicted")
    print("               Pos    Neg")
    print("         Pos │ {:4d} │ {:4d} │".format(tp, fn))
    print("  Actual     ├──────┼──────┤")
    print("         Neg │ {:4d} │ {:4d} │".format(fp, tn))
    print()
    
    # Calculate metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")

if __name__ == "__main__":
    # You would call this with your trained model
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from implementations.resnet_trans import load_model, ECGDatasetFromPipeline
    from data_pipeline import load_data
    model = load_model("./model/model.pkl", is_finetune=False, device='cuda:0')
    df = load_data(path="./holdout_data", use_cache=False, sequence_length=512)
    val_loader = ECGDatasetFromPipeline(df)
    results = diagnose_model_probabilities(model, val_loader, device='cuda:0')
    
    # Print ASCII confusion matrix
    print_ascii_confusion_matrix(results['tp'], results['tn'], results['fp'], results['fn'])
    
    print("Run this script with your trained model to diagnose probability issues!") 