import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd

# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.data_pipeline import load_data, batch_normalize_data, balance_data, data_stats

# Dataset class to handle dataframe
class ECGDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_data = row['signal']  # Shape: (sequence_length, 12)
        label = row['label']
        
        # Transpose to (12, sequence_length) for Conv1d
        ecg_data = ecg_data.T
        
        return torch.FloatTensor(ecg_data), torch.FloatTensor([label])

# Simple CNN model
class ECG_CNN(nn.Module):
    def __init__(self, sequence_length=512, inner_channels=32, num_leads=12, dropout=0.1):
        super(ECG_CNN, self).__init__()
        
        # Conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_leads, inner_channels, kernel_size=5, padding='same'),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(inner_channels, inner_channels * 2, kernel_size=5, padding='same'),
            nn.BatchNorm1d(inner_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(inner_channels * 2, inner_channels * 4, kernel_size=3, padding='same'),
            nn.BatchNorm1d(inner_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(inner_channels * 2, inner_channels * 4, kernel_size=3, padding='same'),
            nn.BatchNorm1d(inner_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(inner_channels * 4, inner_channels * 4, kernel_size=3, padding='same'),
            nn.BatchNorm1d(inner_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # Calculate final sequence length after pooling
        final_seq_len = sequence_length // 16  # 3 MaxPool2d layers
        # Total features after flattening = channels * sequence_length
        flattened_features = inner_channels * 4 * final_seq_len
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flattened_features, inner_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_channels * 4, inner_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_channels * 2, 1)  # Binary classification
        )
        
    def forward(self, x):
        # x shape: (batch, 12, sequence_length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten
        x = x.flatten(start_dim=1)
        
        # FC layers
        x = self.fc(x)
        
        return x

# Simple training function
def train_cnn(data_folder="./data", epochs=10, batch_size=32, lr=0.001, sequence_length=512):
    """Minimal training function for the CNN"""
    
    # Load data
    print("Loading data...")
    df = load_data(path=data_folder, use_cache=True, sequence_length=sequence_length)
    
    # Create hold-out validation set with 100 positive and 100 negative samples
    positive_df = df[df['label'] == 1]
    negative_df = df[df['label'] == 0]
    
    # Sample 100 from each class for validation, this is holdout validation
    holdout_positive = positive_df.sample(n=min(100, len(positive_df)), random_state=42)
    holdout_negative = negative_df.sample(n=min(100, len(negative_df)), random_state=42)
    
    # Combine to create validation set
    holdout_df = pd.concat([holdout_positive, holdout_negative])
    print("\nHoldout validation set:")
    data_stats(holdout_df)
    
    # Remove validation samples from main dataset
    remaining_indices = df.index.difference(holdout_df.index)
    train_df = df.loc[remaining_indices]
    
    # Create holdout guidance dataset - another 100 positive and 100 negative samples
    remaining_positive = train_df[train_df['label'] == 1]
    remaining_negative = train_df[train_df['label'] == 0]
    
    guidance_positive = remaining_positive.sample(n=min(100, len(remaining_positive)), random_state=43)
    guidance_negative = remaining_negative.sample(n=min(100, len(remaining_negative)), random_state=43)
    
    guidance_df = pd.concat([guidance_positive, guidance_negative])
    print("\nHoldout guidance set:")
    data_stats(guidance_df)
    
    # Remove guidance samples from training set
    train_indices = train_df.index.difference(guidance_df.index)
    train_df = train_df.loc[train_indices]
    
    # Balance only the training data
    print("\nTraining set (before balancing):")
    data_stats(train_df)
    
    train_df = balance_data(train_df, strategy='downsample', ratio=3)
    print("\nTraining set (after balancing):")
    data_stats(train_df)
    
    # Create datasets and dataloaders
    train_dataset = ECGDataset(train_df)
    holdout_val_dataset = ECGDataset(holdout_df)
    holdout_guidance_dataset = ECGDataset(guidance_df)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(holdout_val_dataset, batch_size=batch_size, shuffle=False)
    guidance_dataloader = DataLoader(holdout_guidance_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nTraining on {len(train_dataset)} records, guidance on {len(holdout_guidance_dataset)} records, validating on {len(holdout_val_dataset)} records")
    
    # Initialize model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = ECG_CNN(sequence_length=sequence_length).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    
    # Initialize guidance tracking variables
    prev_guidance_accuracy = 0.0
    loss_multiplier = 1.0  # Multiplier for reward/penalty system
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for ecg_data, labels in pbar:
            ecg_data = ecg_data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(ecg_data)
            
            # ===================================
            ##  PENALTY FOR FALSE NEGATIVES
            # ===================================
            
            # Calculate predictions for false negative detection
            with torch.no_grad():
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                # False negatives: predicted 0, actual 1
                false_negatives = (predictions == 0) & (labels == 1)
            
            # Create weights: 1.7x weight for false negatives and regular loss for everything else
            weights = torch.ones_like(labels)
            weights[false_negatives] = 1.7  # Cost sensitive, using this to balance the classes
            
            # Calculate weighted loss
            loss = F.binary_cross_entropy_with_logits(outputs, labels, weight=weights)
            
            # Apply loss multiplier from guidance reward/penalty
            loss = loss * loss_multiplier
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        # Epoch statistics
        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Train Accuracy = {accuracy:.4f}")
        
        # Guidance evaluation for reward/penalty system
        model.eval()
        guidance_correct = 0
        guidance_total = 0
        
        with torch.no_grad():
            for ecg_data, labels in guidance_dataloader:
                ecg_data = ecg_data.to(device)
                labels = labels.to(device)
                
                outputs = model(ecg_data)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                guidance_correct += (predictions == labels).sum().item()
                guidance_total += labels.size(0)
        
        guidance_accuracy = guidance_correct / guidance_total
        print(f"Epoch {epoch+1}: Guidance Accuracy = {guidance_accuracy:.4f}")
        
        # Update loss multiplier based on guidance performance
        if epoch > 0:  # Skip first epoch as we don't have previous accuracy
            if guidance_accuracy > prev_guidance_accuracy:
                # Reward: reduce loss multiplier (minimum 0.8)
                loss_multiplier = 0.8
                print(f"  -> Performance improved! Loss multiplier: {loss_multiplier:.3f} (reward)")
            else:
                # Penalty: increase loss multiplier (maximum 1.5)
                loss_multiplier = 1.5
                print(f"  -> Performance degraded! Loss multiplier: {loss_multiplier:.3f} (penalty)")
        
        prev_guidance_accuracy = guidance_accuracy
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for ecg_data, labels in val_dataloader:
                ecg_data = ecg_data.to(device)
                labels = labels.to(device)
                
                outputs = model(ecg_data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_avg_loss = val_loss / len(val_dataloader)
        val_accuracy = val_correct / val_total
        print(f"Epoch {epoch+1}: Val Loss = {val_avg_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
        print("-" * 50)
        
        model.train()  # Set back to training mode
    
    print("\nTraining complete!")
    return model

# Test the implementation
if __name__ == "__main__":
    # Quick test of the model architecture
    print("Testing model architecture...")
    model = ECG_CNN(sequence_length=512)
    
    # Test with random data
    test_input = torch.randn(4, 12, 512)  # batch_size=4, 12 leads, 512 samples
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Train the model
    print("\n" + "="*50)
    print("Training CNN on ECG data...")
    print("="*50)
    
    trained_model = train_cnn(
        data_folder="../data",
        epochs=64,
        batch_size=32,
        lr=0.002,
        sequence_length=256
    )
