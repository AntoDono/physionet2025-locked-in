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
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_pipeline import load_data, balance_data, mask_ecg, data_stats
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
        
    def train_mode(self):
        # Unfreeze and train all components
        for p in self.cnn.parameters():
            p.requires_grad = True
        for p in self.pool.parameters():
            p.requires_grad = True
        for p in self.proj.parameters():
            p.requires_grad = True
        self.cnn.train()
        self.pool.train()
        self.proj.train()
        
    def finetune_mode(self):
        # Freeze and eval all components
        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.pool.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = False
        self.cnn.eval()
        self.pool.eval()
        self.proj.eval()
    
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
        
        self.is_finetune = False
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers - simplified and fixed
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        
        # Finetune block - complex residual adaptation
        self.finetune_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
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
    
    def load_state_dict_flexible(self, state_dict):
        """Load state dict while ignoring finetune_block mismatches"""
        # Filter out finetune_block keys from loaded state_dict
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('finetune_block')}
        
        # Load everything except finetune_block
        self.load_state_dict(filtered_state_dict, strict=False)
        print("Loaded model state (ignoring finetune_block for compatibility)")
    
    def create_fully_connected_edges(self, num_nodes=12):
        """Create fully connected graph edges for 12 ECG leads"""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Connect all nodes except self-loops
                    edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t()  # Shape: [2, num_edges]
    
    def finetune_mode(self):
        # Freeze GAT layers
        for p in self.gat1.parameters():
            p.requires_grad = False
        for p in self.gat2.parameters():
            p.requires_grad = False
        self.gat1.eval()
        self.gat2.eval()

        # Enable finetune block
        for p in self.finetune_block.parameters():
            p.requires_grad = True
        self.finetune_block.train()

        # Leave classifier trainable
        self.classifier.train()
        self.is_finetune = True
        
    def train_mode(self):
        # Unfreeze GAT layers
        for p in self.gat1.parameters():
            p.requires_grad = True
        for p in self.gat2.parameters():
            p.requires_grad = True
        self.gat1.train()
        self.gat2.train()

        # Freeze finetune block in normal training
        for p in self.finetune_block.parameters():
            p.requires_grad = False
        self.finetune_block.eval()

        # Leave classifier trainable
        self.classifier.train()
        self.is_finetune = False
    
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
        
        # Apply finetune block if in finetune mode
        if self.is_finetune:
            graph_repr = graph_repr + self.finetune_block(graph_repr)
        
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
    
    def finetune_mode(self):
        """Set model to finetune mode - only classifier trains"""
        self.lead_encoder.finetune_mode()
        self.gat.train_mode()
        print("Setting model to finetune mode - only classifier trains")
        
    def train_mode(self):
        """Set model to full training mode - all components train"""
        self.lead_encoder.train_mode()
        self.gat.train_mode()
        print("Setting model to train mode - all components train")
    
    def load_state_dict_flexible(self, state_dict):
        """Load state dict while ignoring finetune_block mismatches"""
        # Filter out finetune_block keys from loaded state_dict  
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('gat.finetune_block')}
        
        # Load everything except finetune_block
        self.load_state_dict(filtered_state_dict, strict=False)
        print("Loaded complete model state (ignoring gat.finetune_block for compatibility)")
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
# Focal Loss Definition
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks.
    
    Args:
        alpha (float): balancing factor for positive class in [0,1] (default = 0.25).
            - alpha weights the positive class
            - (1-alpha) weights the negative class
        gamma (float): focusing parameter ≥ 0 (default = 2).
            - Higher gamma puts more focus on hard examples
        reduction (str): 'none' | 'mean' | 'sum' (default = 'mean').
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.tensor([alpha], dtype=torch.float32)

    def forward(self, inputs, targets):
        """
        inputs: predictions (logits) with shape (N,) or (N, 1) for binary classification
        targets: ground-truth labels with shape (N,) (0 or 1 values)
        """
        # Move alpha to same device as inputs
        self.alpha = self.alpha.to(inputs.device)
        
        # Flatten predictions and targets
        logits = inputs.view(-1)
        y = targets.view(-1).float()
        
        # Get probabilities using sigmoid
        p = torch.sigmoid(logits)
        
        # Compute p_t (probability of true class)
        pt = p * y + (1 - p) * (1 - y)
        
        # Alpha weighting: alpha for positive class, (1-alpha) for negative class
        alpha_factor = self.alpha[0] * y + (1 - self.alpha[0]) * (1 - y)
        
        # Focal loss formula: -α * (1 - p_t)^γ * log(p_t)
        loss = -alpha_factor * (1 - pt).pow(self.gamma) * torch.log(pt + 1e-8)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ============================================================================
# SIMPLE TRAINING FUNCTION WITH VALIDATION
# ============================================================================

def train_model(training_data_folder, model_folder, sequence_length=1024, dropout=0.1, 
                epochs=15, lr=0.002, batch_size=32, device='cpu', train_verbose=True, 
                generate_holdout=False, model_path=None, false_negative_penalty=1.0,
                finetune=False, aggressive_masking=False, alternate_finetune=False):
    """
    Simple training function that takes a folder and trains the model.
    Includes train/validation split and performance metrics.
    
    Args:
        training_data_folder: Path to folder with .dat/.hea files
        sequence_length: The length to pad/trim signals to.
        aggressive_masking: If True, re-mask the data each epoch for different masking patterns
    """
    final_stats = {'precision': 0, 'recall': 0, 'auc': 0, 'accuracy': 0, 'loss': 0}
    
    print("Loading data using data pipeline...")
    df = load_data(path=training_data_folder, name=f"{training_data_folder.replace('/', '-').replace('.', '-')}", 
                   use_cache=True, sequence_length=sequence_length)
    
    if generate_holdout:
        negative_rows = df[df['label'] == 0].sample(n=100)
        positive_rows = df[df['label'] == 1].sample(n=100)
        val_df = pd.concat([negative_rows, positive_rows])
        val_ids = val_df['id'].tolist()
        df = df[~df['id'].isin(val_ids)]
    else:
        val_df = load_data(path="./holdout_data", name="holdout-data", use_cache=False, sequence_length=sequence_length)
        val_ids = val_df['id'].tolist()
        df = df[~df['id'].isin(val_ids)] # removes the validation set from the training set
    
    # Create datasets
    df = balance_data(df, strategy="downsample", ratio=1)
    
    # Store the balanced but unmasked df for aggressive masking
    if aggressive_masking:
        df_unmasked = df.copy()
        print("Aggressive masking enabled - data will be re-masked each epoch")
    
    df = mask_ecg(df, num_lead_masks=2, num_span_masks=2, 
                  lead_mask_ratio=0.25, span_length=None, 
                  span_mask_ratio=0.2, mask_value=0.0)
    
    data_stats(df)
    
    train_dataset = ECGDatasetFromPipeline(df)
    val_dataset = ECGDatasetFromPipeline(val_df)
    
    del df
    
    if train_verbose:
        print(f"Training on {len(train_dataset)} records (after balancing), validating on {len(val_dataset)} records")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
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
    
    if train_verbose:
        print(f"Using device: {device}")
    
    model = ECGChagas(
        cnn_node_dim=512,
        gat_hidden_dim=256,
        gat_heads=32,
        gat_layers=16,
        dropout=dropout
    ).to(device)
    
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict_flexible(state_dict)
    
    if finetune:
        model.finetune_mode()
    else:
        model.train_mode()
    
    # Simple training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_loss = float('inf')
    
    print("Starting training...")
    
    # Train for epochs
    for epoch in range(epochs):
        # Re-mask data each epoch if aggressive masking is enabled
        # lead_mask_ratio = np.random.uniform(0.2, 0.5)
        # span_mask_ratio = np.random.uniform(0.2, 0.5)
        lead_mask_ratio = 0.5
        span_mask_ratio = 0.5
        if aggressive_masking:
            df_epoch = mask_ecg(df_unmasked.copy(), num_lead_masks=1, num_span_masks=1, 
                              lead_mask_ratio=lead_mask_ratio, span_length=None, 
                              span_mask_ratio=span_mask_ratio, mask_value=0.0, include_original=False)
            train_dataset = ECGDatasetFromPipeline(df_epoch)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            if train_verbose:
                print(f"Epoch {epoch}: Aggressive masking - Re-masked data with new patterns")
                print(f"  Lead mask ratio: {lead_mask_ratio}")
                print(f"  Span mask ratio: {span_mask_ratio}")
        
        # TRAINING PHASE
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        if finetune and alternate_finetune:
            if (epoch + 1) % 5 == 0:
                model.train_mode()
                print("Setting model to train mode - All components train")
            else:
                model.finetune_mode()
                print("Setting model to finetune mode - Only finetune blocks and classifier train")
        
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
            weights[false_negatives] = false_negative_penalty # Cost sensitive, using this to balance the classes
            
            # Calculate weighted loss
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
                
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.sigmoid(logits) > 0.5
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_description(f"Epoch {epoch} - Loss: {loss.item():.7f}")
        
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

        print(f"Validation loss: {val_loss_avg:.7f}, Best loss so far: {best_loss:.7f}")
        
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            # Temporarily set to train mode to save all parameters
            current_mode = model.is_finetune
            model.train_mode()
            torch.save(model.state_dict(), os.path.join(model_folder, f'gnn_trained_model.pkl'))
            # Restore original mode
            if current_mode:
                model.finetune_mode()
            print("*" * 50)
            print(f"Best model saved with loss {val_loss_avg:.7f} and acc {val_acc:.7f} to {os.path.join(model_folder, f'gnn_trained_model.pkl')}")
        else:
            print(f"No improvement. Current: {val_loss_avg:.7f}, Best: {best_loss:.7f}")
        
        # Step the learning rate scheduler
        scheduler.step()
        
        val_auc = roc_auc_score(val_labels_list, val_predictions)
        val_precision = precision_score(val_labels_list, [p > 0.5 for p in val_predictions])
        val_recall = recall_score(val_labels_list, [p > 0.5 for p in val_predictions])
        
        # Update final stats
        final_stats['loss'] = val_loss_avg
        final_stats['accuracy'] = val_acc
        final_stats['precision'] = val_precision
        final_stats['recall'] = val_recall
        final_stats['auc'] = val_auc
        
        # Calculate AUC if possible
        if train_verbose:
            try:
                print(f"Epoch {epoch}:")
                print(f"  Train - Loss: {train_loss_avg:.7f}, Acc: {train_acc:.7f}, LR: {scheduler.get_last_lr()[0]:.7f}")
                print(f"  Val   - Loss: {val_loss_avg:.7f}, Acc: {val_acc:.7f}, AUC: {val_auc:.7f}")
                print(f"  Val   - Precision: {val_precision:.7f}, Recall: {val_recall:.7f}")
                
            except ImportError:
                print(f"Epoch {epoch}:")
                print(f"  Train - Loss: {train_loss_avg:.7f}, Acc: {train_acc:.7f}, LR: {scheduler.get_last_lr()[0]:.7f}")
                print(f"  Val   - Loss: {val_loss_avg:.7f}, Acc: {val_acc:.7f}")
                print("  (Install sklearn for AUC/Precision/Recall metrics)")
            
            print("-" * 50)
    
    # Save model
    print(final_stats)
    
    return model, final_stats

# ============================================================================
# SIMPLE INFERENCE FUNCTION
# ============================================================================

def load_model(model_path, device='cpu'):
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
        cnn_node_dim=512,
        gat_hidden_dim=256,
        gat_heads=32,
        gat_layers=16,
        dropout=0.1
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict_flexible(state_dict)
    model.eval()
    
    return model

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Train
    data_folder = "../data"
    model, final_stats = train_model(data_folder)
    print(final_stats)