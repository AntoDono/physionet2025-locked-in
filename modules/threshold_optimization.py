import torch
import numpy as np
import os
import sys
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def binary_threshold_optimization(model, val_dataset, increments, device='cpu'):
    """
    Find the optimal threshold for binary classification.
    
    Args:
        model: The trained model to evaluate
        val_dataset: DataLoader that yields (inputs, labels)
        increments: Step size for threshold search (e.g., 0.01 for 1% increments)
        
    Returns:
        best_threshold: The threshold that maximizes accuracy
    """
    thresholds = np.arange(0, 1 + increments, increments)
    best_threshold = thresholds[0]
    best_accuracy = 0.0
    
    # Get the device from the model
    
    # Collect all predictions and labels
    all_preds = []
    all_labels = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in val_dataset:
            inputs, labels = batch
            # Move inputs to the same device as the model
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            preds = model(inputs)
            probs = torch.sigmoid(preds)
            
            all_preds.append(probs)
            all_labels.append(labels)
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Test each threshold
    for threshold in tqdm.tqdm(thresholds):
        # Convert probabilities to binary predictions
        binary_preds = (all_preds >= threshold).float()
        
        # Calculate accuracy
        correct = (binary_preds == all_labels).float().sum()
        accuracy = correct / len(all_labels)
        
        # Update best threshold if this one is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            
    return best_threshold, best_accuracy

if __name__ == "__main__":
    print("Threshold Optimization on ECG GNN Model")
    
    # from implementations.gnn import load_model, ECGDatasetFromPipeline
    from implementations.resnet_trans import load_model, ECGDatasetFromPipeline
    from modules.data_pipeline import load_data
    from torch.utils.data import DataLoader
    
    model = load_model("./model/model.pkl", is_finetune=False, device="cuda:1")
    