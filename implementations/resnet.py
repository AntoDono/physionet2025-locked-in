import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from pathlib import Path
import sys
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import math

# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_pipeline import load_data, balance_data, mask_ecg, data_stats
from modules.normalize_wave import normalize_amplitude, wavelet_denoising, trim_signal, pad_signal
from helper_code import reorder_signal, get_signal_names, load_header, load_signals

# ============================================================================
# SQUEEZE AND EXCITATION BLOCK
# ============================================================================

class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation block for 1D signals"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, C, L]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# ============================================================================
# RESIDUAL BLOCK WITH SE AND DILATED CONVOLUTIONS
# ============================================================================

class ResidualBlock1D(nn.Module):
    """Advanced residual block with SE attention and dilated convolutions"""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, use_se=True, dropout=0.1):
        super().__init__()
        self.use_se = use_se
        
        # Main path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE block
        if use_se:
            self.se = SqueezeExcitation1D(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out

# ============================================================================
# MULTI-SCALE FEATURE EXTRACTION
# ============================================================================

class MultiScaleBlock(nn.Module):
    """Extract features at multiple scales using different kernel sizes"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        assert out_channels % 4 == 0
        branch_channels = out_channels // 4
        
        # Different kernel sizes for multi-scale processing
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch_pool = self.branch_pool(x)
        
        return torch.cat([branch1, branch3, branch5, branch_pool], dim=1)

# ============================================================================
# LEAD ATTENTION MODULE
# ============================================================================

class LeadAttention(nn.Module):
    """Cross-lead attention mechanism"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, 12, embed_dim]
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x

# ============================================================================
# ECGRESNET MODEL
# ============================================================================

class ECGResNet(nn.Module):
    """State-of-the-art ResNet for ECG classification"""
    
    def __init__(self, lead_base_size=256, num_classes=1, dropout=0.1):
        super().__init__()
        
        self.is_finetune = False
        
        # Calculate scaled dimensions based on lead_base_size
        initial_channels = lead_base_size // 4  # 64 for base_size=256
        mid_channels = lead_base_size  # 256 for base_size=256
        final_channels = lead_base_size * 2  # 512 for base_size=256
        
        # Simplified lead encoder with 2 stages and fewer blocks
        self.lead_encoder = nn.ModuleList([
            nn.Sequential(
                # Initial convolution
                nn.Conv1d(1, initial_channels, kernel_size=15, stride=2, padding=7, bias=False),
                nn.BatchNorm1d(initial_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                
                # Multi-scale block
                MultiScaleBlock(initial_channels, mid_channels),
                
                # Stage 1: mid_channels -> mid_channels (2 blocks instead of 3)
                ResidualBlock1D(mid_channels, mid_channels, dilation=1, use_se=True, dropout=dropout),
                ResidualBlock1D(mid_channels, mid_channels, dilation=2, use_se=True, dropout=dropout),
                
                # Stage 2: mid_channels -> final_channels (2 blocks instead of 3)
                ResidualBlock1D(mid_channels, final_channels, stride=2, dilation=1, use_se=True, dropout=dropout),
                ResidualBlock1D(final_channels, final_channels, dilation=2, use_se=True, dropout=dropout),
                
                # Global pooling
                nn.AdaptiveAvgPool1d(1)
            ) for _ in range(12)  # One encoder per lead
        ])
        
        # Lead attention layers - scale number of heads based on lead_base_size
        num_heads = max(4, final_channels // 32)  # 16 heads for base_size=256
        self.lead_attention = nn.Sequential(
            LeadAttention(final_channels, num_heads=num_heads, dropout=dropout),
            LeadAttention(final_channels, num_heads=num_heads, dropout=dropout),
        )
        
        # Finetune adaptation block
        self.finetune_block = nn.Sequential(
            nn.Linear(final_channels, final_channels),
            nn.LayerNorm(final_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_channels, final_channels * 2),
            nn.LayerNorm(final_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_channels * 2, final_channels),
            nn.LayerNorm(final_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Final classifier with deep architecture - scale based on final_channels
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, final_channels // 2),
            nn.LayerNorm(final_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_channels // 2, final_channels // 4),
            nn.LayerNorm(final_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_channels // 4, final_channels // 8),
            nn.LayerNorm(final_channels // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_channels // 8, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def finetune_mode(self):
        """Set model to finetune mode - freeze encoders, train adaptation layers"""
        # Freeze lead encoders
        for encoder in self.lead_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()
        
        # Freeze attention
        for p in self.lead_attention.parameters():
            p.requires_grad = False
        self.lead_attention.eval()
        
        # Enable finetune block
        for p in self.finetune_block.parameters():
            p.requires_grad = True
        self.finetune_block.train()
        
        # Enable classifier
        for p in self.classifier.parameters():
            p.requires_grad = True
        self.classifier.train()
        
        self.is_finetune = True
        
    def train_mode(self):
        """Set model to full training mode"""
        # Unfreeze all
        for encoder in self.lead_encoder:
            for p in encoder.parameters():
                p.requires_grad = True
            encoder.train()
        
        for p in self.lead_attention.parameters():
            p.requires_grad = True
        self.lead_attention.train()
        
        # Disable finetune block in normal training
        for p in self.finetune_block.parameters():
            p.requires_grad = False
        self.finetune_block.eval()
        
        for p in self.classifier.parameters():
            p.requires_grad = True
        self.classifier.train()
        
        self.is_finetune = False
    
    def load_state_dict_flexible(self, state_dict):
        """Load state dict while ignoring finetune_block mismatches"""
        # Filter out finetune_block keys from loaded state_dict
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('finetune_block')}
        
        # Load everything except finetune_block
        self.load_state_dict(filtered_state_dict, strict=False)
        print("Loaded model state (ignoring finetune_block for compatibility)")
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: [batch_size, sequence_length, 12] ECG signals
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Process each lead independently
        lead_features = []
        for i in range(12):
            # Extract lead: [batch_size, 1, sequence_length]
            lead = x[:, :, i].unsqueeze(1)
            # Encode: [batch_size, 1024, 1]
            features = self.lead_encoder[i](lead)
            # Squeeze: [batch_size, 1024]
            lead_features.append(features.squeeze(-1))
        
        # Stack: [batch_size, 12, 1024]
        lead_features = torch.stack(lead_features, dim=1)
        
        # Apply cross-lead attention
        lead_features = self.lead_attention(lead_features)
        
        # Global pooling across leads: [batch_size, 1024]
        global_features = lead_features.mean(dim=1)
        
        # Apply finetune block if in finetune mode
        if self.is_finetune:
            global_features = global_features + self.finetune_block(global_features)
        
        # Classification
        logits = self.classifier(global_features)
        
        return logits

# ============================================================================
# DATASET FROM DATAFRAME (Same as GNN)
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
# Focal Loss Definition (Same as GNN)
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
# TRAINING FUNCTION WITH VALIDATION (Same interface as GNN)
# ============================================================================

def train_model(training_data_folder, model_folder, sequence_length=1024, dropout=0.1, 
                epochs=15, lr=0.002, batch_size=32, device='cpu', train_verbose=True, 
                generate_holdout=False, model_path=None, false_negative_penalty=1.0,
                finetune=False, lead_base_size=32):
    """
    Simple training function that takes a folder and trains the model.
    Includes train/validation split and performance metrics.
    
    Args:
        training_data_folder: Path to folder with .dat/.hea files
        sequence_length: The length to pad/trim signals to.
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
    
    model = ECGResNet(lead_base_size=lead_base_size, num_classes=1, dropout=dropout).to(device)
    
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict_flexible(state_dict)
    
    if finetune:
        model.finetune_mode()
    else:
        model.train_mode()
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    best_loss = float('inf')
    
    print("Starting training...")
    
    # Train for epochs
    for epoch in range(epochs):
        # TRAINING PHASE
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (ecg_data, labels) in enumerate(pbar):
            # Data validation
            if not torch.isfinite(ecg_data).all():
                print(f"\nEpoch {epoch}, Batch {batch_idx}: FOUND PROBLEMATIC DATA!")
                print(f"  ECG data shape: {ecg_data.shape}")
                print(f"  Contains NaN: {torch.isnan(ecg_data).any().item()}")
                print(f"  Contains Inf: {torch.isinf(ecg_data).any().item()}")
                print(f"  Skipping this batch...\n")
                continue

            ecg_data = ecg_data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(ecg_data)
            
            # Calculate predictions for false negative detection
            with torch.no_grad():
                predictions = (torch.sigmoid(logits) > 0.5).float()
                # False negatives: predicted 0, actual 1
                false_negatives = (predictions == 0) & (labels == 1)
            
            # Create weights: penalty for false negatives
            weights = torch.ones_like(labels)
            weights[false_negatives] = false_negative_penalty
            
            # Calculate weighted loss
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.sigmoid(logits) > 0.5
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_description(f"Epoch {epoch} - Loss: {loss.item():.7f}")
        
        # Step scheduler
        scheduler.step()
        
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

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            torch.save(model.state_dict(), os.path.join(model_folder, f'resnet_trained_model.pkl'))
            print("*" * 50)
            print(f"Best model saved with loss {val_loss_avg:.7f} and acc {val_acc:.7f} saved at {os.path.join(model_folder, f'resnet_trained_model.pkl')}")
        
        val_auc = roc_auc_score(val_labels_list, val_predictions)
        val_precision = precision_score(val_labels_list, [p > 0.5 for p in val_predictions])
        val_recall = recall_score(val_labels_list, [p > 0.5 for p in val_predictions])
        
        # Update final stats
        final_stats['loss'] = val_loss_avg
        final_stats['accuracy'] = val_acc
        final_stats['precision'] = val_precision
        final_stats['recall'] = val_recall
        final_stats['auc'] = val_auc
        
        # Print metrics
        if train_verbose:
            print(f"Epoch {epoch}:")
            print(f"  Train - Loss: {train_loss_avg:.7f}, Acc: {train_acc:.7f}, LR: {scheduler.get_last_lr()[0]:.7f}")
            print(f"  Val   - Loss: {val_loss_avg:.7f}, Acc: {val_acc:.7f}, AUC: {val_auc:.7f}")
            print(f"  Val   - Precision: {val_precision:.7f}, Recall: {val_recall:.7f}")
            print("-" * 50)
    
    print(final_stats)
    
    return model, final_stats

# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================

def load_model(model_path, device='cpu', lead_base_size=256):
    """
    Load model and prepare for inference.
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
        lead_base_size: Base size for lead encoder dimensions
    """
    
    device = torch.device(device)
    
    # Load model with ResNet architecture
    model = ECGResNet(lead_base_size=lead_base_size, num_classes=1, dropout=0.1).to(device)
    
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
    model_folder = "./models"
    os.makedirs(model_folder, exist_ok=True)
    
    model, final_stats = train_model(
        training_data_folder=data_folder,
        model_folder=model_folder,
        sequence_length=1024,
        dropout=0.1,
        epochs=20,
        lr=0.001,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        train_verbose=True,
        false_negative_penalty=1.5,
        lead_base_size=256  # Can be adjusted: 128 (smaller), 256 (default), 512 (larger)
    )
    print(f"Final stats: {final_stats}") 