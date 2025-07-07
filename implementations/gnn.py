import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from pathlib import Path
import sys
from tqdm import tqdm
import pandas as pd
# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_pipeline import load_data, batch_normalize_data, balance_data, data_stats
from modules.normalize_wave import normalize_amplitude, wavelet_denoising, trim_signal, pad_signal
from helper_code import reorder_signal, get_signal_names, load_header, load_signals

# ============================================================================
# CNN LEAD ENCODER 
# ============================================================================

class LeadEncoder(nn.Module):
    def __init__(self, in_ch=1, node_dim=64):  # Reduced from 128 to 64
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1: [1 × L] → [8 × L/2] - Reduced channels
            nn.Conv1d(in_ch, 8, kernel_size=7, padding=3),
            nn.ReLU(),  # Removed BatchNorm for CPU efficiency
            nn.MaxPool1d(2),
            # Block 2: [8 × L/2] → [16 × L/4] - Reduced channels
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3: → [32 × L/8] - Reduced channels
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 4: → [64 × L/16] - Reduced channels
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # collapse time → node embedding
        self.pool = nn.AdaptiveAvgPool1d(1)   # → [64 × 1]
        self.proj = nn.Linear(64, node_dim)   # → [node_dim]
    
    def forward(self, x):
        # x: [B × 1 × L]
        out = self.cnn(x)           # [B × 128 × (L/16)]
        out = self.pool(out)        # [B × 128 × 1]
        out = out.squeeze(-1)       # [B × 128]
        return self.proj(out)       # [B × node_dim]

# ============================================================================
# GAT MODEL (EXACTLY AS PROVIDED)
# ============================================================================

class ECGGATModel(nn.Module):
    def __init__(self, 
                 input_dim=128,      # CNN output dimension
                 hidden_dim=64,      # Hidden dimension for GAT layers
                 num_heads=4,        # Number of attention heads
                 num_layers=2,       # Number of GAT layers
                 dropout=0.3,        # Dropout rate
                 num_classes=1):     # Binary classification
        super(ECGGATModel, self).__init__()
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers - simplified and fixed
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def create_fully_connected_edges(self, num_nodes=12):
        """Create fully connected graph edges for 12 ECG leads"""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Connect all nodes except self-loops
                    edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t()  # Shape: [2, num_edges]
    
    def forward(self, node_features, edge_index=None, batch=None, return_attention=False):
        """
        Forward pass - SIMPLIFIED AND FIXED
        
        Args:
            node_features: [batch_size, 12, input_dim] 
            
        Returns:
            logits: [batch_size, num_classes]
        """
        
        # Input is [batch_size, 12, input_dim]
        batch_size = node_features.size(0)
        num_nodes_per_graph = node_features.size(1)  # Should be 12
        
        # Flatten to [batch_size * 12, input_dim]
        x = node_features.view(-1, node_features.size(-1))
        
        # Create batch tensor: [0,0,0...0, 1,1,1...1, 2,2,2...2, ...]
        batch = torch.arange(batch_size, device=node_features.device).repeat_interleave(num_nodes_per_graph)
        
        # Create edge index for fully connected graph
        base_edges = self.create_fully_connected_edges(num_nodes_per_graph).to(node_features.device)
        
        # Expand edges for each graph in batch
        edge_list = []
        for i in range(batch_size):
            batch_edges = base_edges + i * num_nodes_per_graph
            edge_list.append(batch_edges)
        edge_index = torch.cat(edge_list, dim=1)
        
        # GAT layers
        x = self.gat1(x, edge_index)  # [batch_size * 12, hidden_dim * num_heads]
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.gat2(x, edge_index)  # [batch_size * 12, hidden_dim]
        x = F.elu(x)
        
        # Global mean pooling to get graph-level representation
        graph_repr = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(graph_repr)  # [batch_size, num_classes]
        
        return logits

# ============================================================================
# COMPLETE MODEL (EXACTLY AS PROVIDED)
# ============================================================================

class ECGChagas(nn.Module):
    """Complete ECG to Chagas prediction pipeline"""
    
    def __init__(self, 
                 cnn_node_dim=64,
                 gat_hidden_dim=32,
                 gat_heads=8,
                 gat_layers=4,
                 dropout=0.1):
        super(ECGChagas, self).__init__()
        
        # Lead encoder (your existing CNN)
        self.lead_encoder = LeadEncoder(in_ch=1, node_dim=cnn_node_dim)
        
        # GAT model
        self.gat = ECGGATModel(
            input_dim=cnn_node_dim,
            hidden_dim=gat_hidden_dim,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=dropout
        )
        
    def forward(self, ecg_data, return_attention=False):
        """
        Forward pass through complete pipeline
        
        Args:
            ecg_data: [batch_size, sequence_length, 12] raw ECG signals
            return_attention: Whether to return GAT attention weights
            
        Returns:
            logits: [batch_size, 1] Chagas predictions
            attention_weights: (optional) GAT attention weights
        """
        batch_size = ecg_data.size(0)
        
        # Process each lead through CNN
        lead_features = []
        for lead_idx in range(12):
            # Get lead signal: [batch_size, 1, sequence_length]
            lead_signal = ecg_data[:, :, lead_idx].unsqueeze(1)
            
            # Pass through CNN: [batch_size, node_dim]
            features = self.lead_encoder(lead_signal)
            lead_features.append(features)
        
        # Stack lead features: [batch_size, 12, node_dim]
        node_features = torch.stack(lead_features, dim=1)
        
        # Pass through GAT
        return self.gat(node_features, return_attention=return_attention)

# ============================================================================
# DATASET FROM DATAFRAME
# ============================================================================

class ECGDatasetFromPipeline(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_data = row['signal']  # This is (sequence_length, 12)
        label = row['label']
        
        return torch.FloatTensor(ecg_data), torch.FloatTensor([label])

# ============================================================================
# SIMPLE TRAINING FUNCTION WITH VALIDATION
# ============================================================================

def train_model(training_data_folder, sequence_length=1024, device='cpu'):
    """
    Simple training function that takes a folder and trains the model.
    Includes train/validation split and performance metrics.
    
    Args:
        training_data_folder: Path to folder with .dat/.hea files
        sequence_length: The length to pad/trim signals to.
    """
    
    print("Loading data using data pipeline...")
    df = load_data(path=training_data_folder, use_cache=True, sequence_length=sequence_length)
    
    # Create hold-out validation set with 100 positive and 100 negative samples
    positive_df = df[df['label'] == 1]
    negative_df = df[df['label'] == 0]
    
    # Sample 100 from each class for validation, this is holdout validation
    val_positive = positive_df.sample(n=min(100, len(positive_df)), random_state=42)
    val_negative = negative_df.sample(n=min(100, len(negative_df)), random_state=42)
    
    # Combine to create validation set
    val_df = pd.concat([val_positive, val_negative])
    print("Validation set:")
    data_stats(val_df)
    
    # Remove validation samples from main dataset
    remaining_indices = df.index.difference(val_df.index)
    train_df = df.loc[remaining_indices]
    
    # Balance only the training data
    print("Training set:")
    train_df = balance_data(train_df, strategy='oversample')
    
    # Create datasets
    train_dataset = ECGDatasetFromPipeline(train_df)
    val_dataset = ECGDatasetFromPipeline(val_df)
    
    print(f"Training on {len(train_dataset)} records (after balancing), validating on {len(val_dataset)} records")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Check label distribution
    print("Checking label distribution...")
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy().flatten())
    
    train_pos = sum(train_labels)
    train_total = len(train_labels)
    print(f"Training set: {train_pos}/{train_total} positive ({100*train_pos/train_total:.1f}%)")
    
    # Initialize model
    device = torch.device(device)
    print(f"Using device: {device}")
    
    model = ECGChagas(
        cnn_node_dim=128,
        gat_hidden_dim=64,
        gat_heads=8,
        gat_layers=2,
        dropout=0.1
    ).to(device)
    
    # Simple training setup
    criterion = nn.BCEWithLogitsLoss()
    epochs = 15
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    print("Starting training...")
    
    # Train for 10 epochs
    for epoch in range(epochs):
        # TRAINING PHASE
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (ecg_data, labels) in enumerate(pbar):
            # Comprehensive data validation
            if not torch.isfinite(ecg_data).all():
                print(f"\nEpoch {epoch}, Batch {batch_idx}: FOUND PROBLEMATIC DATA!")
                print(f"  ECG data shape: {ecg_data.shape}")
                print(f"  Contains NaN: {torch.isnan(ecg_data).any().item()}")
                print(f"  Contains Inf: {torch.isinf(ecg_data).any().item()}")
                print(f"  Min value: {torch.min(ecg_data).item()}")
                print(f"  Max value: {torch.max(ecg_data).item()}")
                print(f"  Mean: {torch.mean(ecg_data).item()}")
                print(f"  Std: {torch.std(ecg_data).item()}")
                
                # Check which samples and leads have issues
                nan_mask = torch.isnan(ecg_data)
                inf_mask = torch.isinf(ecg_data)
                
                if nan_mask.any():
                    nan_samples = torch.any(nan_mask.view(ecg_data.size(0), -1), dim=1).nonzero().squeeze()
                    print(f"  Samples with NaN: {nan_samples.tolist() if nan_samples.numel() > 1 else [nan_samples.item()]}")
                
                if inf_mask.any():
                    inf_samples = torch.any(inf_mask.view(ecg_data.size(0), -1), dim=1).nonzero().squeeze()
                    print(f"  Samples with Inf: {inf_samples.tolist() if inf_samples.numel() > 1 else [inf_samples.item()]}")
                
                print("  Skipping this batch...\n")
                continue

            ecg_data = ecg_data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(ecg_data)
            
            # ===================================
            ## DOUBLE PENALITY FOR FALSE NEGATIVES
            # ===================================
            
            # Calculate predictions for false negative detection
            with torch.no_grad():
                predictions = (torch.sigmoid(logits) > 0.5).float()
                # False negatives: predicted 0, actual 1
                false_negatives = (predictions == 0) & (labels == 1)
            
            # Create weights: 1.5x negatives for false negatives and regular loss for everything else
            weights = torch.ones_like(labels)
            weights[false_negatives] = 1.5
            
            # Calculate weighted loss
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
                
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.sigmoid(logits) > 0.5
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_description(f"Epoch {epoch} - Loss: {loss.item():.4f}")
        
        # VALIDATION PHASE
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels_list = []
        
        with torch.no_grad():
            for ecg_data, labels in val_loader:
                ecg_data = ecg_data.to(device)
                labels = labels.to(device)
                
                logits = model(ecg_data)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.sigmoid(logits)
                predictions = probs > 0.5
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                # Store for detailed metrics
                val_predictions.extend(probs.cpu().numpy().flatten())
                val_labels_list.extend(labels.cpu().numpy().flatten())
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        # Step the learning rate scheduler
        scheduler.step(val_loss_avg)
        
        # Calculate AUC if possible
        try:
            from sklearn.metrics import roc_auc_score, precision_score, recall_score
            val_auc = roc_auc_score(val_labels_list, val_predictions)
            val_precision = precision_score(val_labels_list, [p > 0.5 for p in val_predictions])
            val_recall = recall_score(val_labels_list, [p > 0.5 for p in val_predictions])
            
            print(f"Epoch {epoch}:")
            print(f"  Train - Loss: {train_loss_avg:.4f}, Acc: {train_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.7f}")
            print(f"  Val   - Loss: {val_loss_avg:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
            print(f"  Val   - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            
        except ImportError:
            print(f"Epoch {epoch}:")
            print(f"  Train - Loss: {train_loss_avg:.4f}, Acc: {train_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.7f}")
            print(f"  Val   - Loss: {val_loss_avg:.4f}, Acc: {val_acc:.4f}")
            print("  (Install sklearn for AUC/Precision/Recall metrics)")
        
        print("-" * 50)
    
    # Save model
    model_path = os.path.join(training_data_folder, 'trained_model.pkl')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

# ============================================================================
# SIMPLE INFERENCE FUNCTION
# ============================================================================

def load_and_predict(model_path, record_path, sequence_length=512, device='cpu'):
    """
    Load model and predict on a single record.
    
    Args:
        model_path: Path to saved model
        record_path: Path to ECG record (without extension)
        sequence_length: The length to pad/trim signals to.
    """
    
    device = torch.device(device)
    
    # Load model with updated architecture
    model = ECGChagas(
        cnn_node_dim=64,
        gat_hidden_dim=32,
        gat_heads=2,
        gat_layers=2,
        dropout=0.2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and preprocess ECG signal using pipeline logic
    header = load_header(record_path)
    signal, _ = load_signals(record_path)
    signal_names = get_signal_names(header)

    signal = reorder_signal(signal, signal_names, ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"])
    
    signal = normalize_amplitude(signal)
    signal = wavelet_denoising(signal)
    signal = trim_signal(signal, sequence_length)
    ecg_data = pad_signal(signal, sequence_length)
    
    # Predict
    with torch.no_grad():
        ecg_tensor = torch.FloatTensor(ecg_data).unsqueeze(0).to(device)  # [1, sequence_length, 12]
        logits = model(ecg_tensor)
        probability = torch.sigmoid(logits).cpu().numpy()[0, 0]
        binary_pred = int(probability > 0.3)
    
    return binary_pred, probability

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Train
    data_folder = "/Users/vivian/Desktop/updated_physionet/python-example-2025/training_data"
    model = train_model(data_folder)