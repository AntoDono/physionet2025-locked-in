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
from sklearn.isotonic import IsotonicRegression
import math
import joblib

# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_pipeline import load_data, balance_data, mask_ecg, data_stats
from modules.diagnostic_script import diagnose_model_probabilities

# ============================================================================
# TRANSFORMER SUB-MODULE
# ============================================================================

class TransformerModule(nn.Module):
    def __init__(self, sequence_length=1024, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input projection from 12 leads to d_model
        self.input_projection = nn.Linear(12, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Finetune block
        self.finetune_block = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model * 4, d_model * 8),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model * 8, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model * 2, d_model)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def train_mode(self):
        # Unfreeze and train main components
        for p in self.input_projection.parameters():
            p.requires_grad = True
        for p in self.transformer.parameters():
            p.requires_grad = True
        self.input_projection.train()
        self.transformer.train()
        
        # Freeze finetune block
        for p in self.finetune_block.parameters():
            p.requires_grad = False
        self.finetune_block.eval()
    
    def finetune_mode(self):
        # Freeze main components
        for p in self.input_projection.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False
        self.input_projection.eval()
        self.transformer.eval()
        
        # Unfreeze finetune block
        for p in self.finetune_block.parameters():
            p.requires_grad = True
        self.finetune_block.train()
    
    def forward(self, x, is_finetune=False):
        # x: [batch_size, sequence_length, 12]
        x = self.input_projection(x)  # [batch_size, sequence_length, d_model]
        x = self.pos_encoding(x)
        x = self.transformer(x)  # [batch_size, sequence_length, d_model]
        
        # Global pooling
        x = x.transpose(1, 2)  # [batch_size, d_model, sequence_length]
        x = self.pool(x).squeeze(-1)  # [batch_size, d_model]
        
        # Apply finetune block if in finetune mode
        if is_finetune:
            x = self.finetune_block(x)
        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================================
# RESNET SUB-MODULE
# ============================================================================

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetModule(nn.Module):
    def __init__(self, base_channels=64, num_blocks=[2, 2, 2, 2], dropout=0.1):
        super().__init__()
        self.in_channels = base_channels
        
        # Initial convolution
        self.conv1 = nn.Conv1d(12, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, num_blocks[3], stride=2)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Output dimension
        self.output_dim = base_channels * 8
        
        # Finetune block
        self.finetune_block = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(self.output_dim * 4, self.output_dim * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(self.output_dim * 8, self.output_dim * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(self.output_dim * 8, self.output_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(self.output_dim * 4, self.output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(self.output_dim * 2, self.output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def train_mode(self):
        # Unfreeze and train main components
        for name, p in self.named_parameters():
            if 'finetune_block' not in name:
                p.requires_grad = True
        
        # Freeze finetune block
        for p in self.finetune_block.parameters():
            p.requires_grad = False
        self.finetune_block.eval()
    
    def finetune_mode(self):
        # Freeze main components
        for name, p in self.named_parameters():
            if 'finetune_block' not in name:
                p.requires_grad = False
        
        # Unfreeze finetune block
        for p in self.finetune_block.parameters():
            p.requires_grad = True
        self.finetune_block.train()
    
    def forward(self, x, is_finetune=False):
        # x: [batch_size, sequence_length, 12]
        x = x.transpose(1, 2)  # [batch_size, 12, sequence_length]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.squeeze(-1)  # [batch_size, output_dim]
        
        # Apply finetune block if in finetune mode
        if is_finetune:
            x = self.finetune_block(x)
        
        return x


# ============================================================================
# CLASSIFIER MODULE
# ============================================================================

class Classifier(nn.Module):
    def __init__(self, transformer_dim, resnet_dim, hidden_dim=256, num_classes=1, dropout=0.1):
        super().__init__()
        
        # Combine features from both modules
        combined_dim = transformer_dim + resnet_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, transformer_features, resnet_features):
        # Concatenate features from both modules
        combined = torch.cat([transformer_features, resnet_features], dim=1)
        return self.classifier(combined)


# ============================================================================
# COMPLETE MODEL
# ============================================================================

class ECGChagas(nn.Module):
    """Complete ECG to Chagas prediction pipeline with parallel transformer and resnet"""
    
    def __init__(self, 
                 sequence_length=1024,
                 # Transformer parameters
                 transformer_d_model=128,
                 transformer_nhead=8,
                 transformer_layers=4,
                 transformer_dim_feedforward=512,
                 # ResNet parameters
                 resnet_base_channels=64,
                 resnet_num_blocks=[2, 2, 2, 2],
                 # Classifier parameters
                 classifier_hidden_dim=256,
                 # General parameters
                 dropout=0.1,
                 base_parameter=1.0,
                 device='cpu'):  # Scale factor for both modules
        super().__init__()
        
        self.device = device
        self.is_finetune = False
        self.threshold = None
        # self.isotonic_regression = IsotonicRegression(out_of_bounds='clip') # Turns logits into actual probabilities
        
        # Scale dimensions based on base_parameter
        transformer_d_model = int(transformer_d_model * base_parameter)
        transformer_dim_feedforward = int(transformer_dim_feedforward * base_parameter)
        resnet_base_channels = int(resnet_base_channels * base_parameter)
        classifier_hidden_dim = int(classifier_hidden_dim * base_parameter)
        
        # Initialize modules
        self.transformer = TransformerModule(
            sequence_length=sequence_length,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout
        ).to(device)
        
        self.resnet = ResNetModule(
            base_channels=resnet_base_channels,
            num_blocks=resnet_num_blocks,
            dropout=dropout
        ).to(device)
        
        self.classifier = Classifier(
            transformer_dim=transformer_d_model,
            resnet_dim=self.resnet.output_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=1,
            dropout=dropout
        ).to(device)
    
    def forward(self, ecg_data):
        """
        Forward pass through complete pipeline
        
        Args:
            ecg_data: [batch_size, sequence_length, 12] raw ECG signals
            
        Returns:
            logits: [batch_size, 1] Chagas predictions
        """
        # Pass data through both modules in parallel
        transformer_features = self.transformer(ecg_data, is_finetune=self.is_finetune)
        resnet_features = self.resnet(ecg_data, is_finetune=self.is_finetune)
        
        # Pass through classifier
        logits = self.classifier(transformer_features, resnet_features)

        return logits
    
    def predict(self, ecg_data):
        
        # Pass data through both modules in parallel
        transformer_features = self.transformer(ecg_data, is_finetune=self.is_finetune)
        resnet_features = self.resnet(ecg_data, is_finetune=self.is_finetune)
        
        # Pass through classifier
        logits = self.classifier(transformer_features, resnet_features)
        
        # Convert logits to CPU numpy array for isotonic regression
        logits_cpu = logits.detach().cpu().numpy().flatten()
        # probability = self.isotonic_regression.transform(logits_cpu)
        probability = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        
        if self.threshold is not None:
            binary = (probability > self.threshold) # returns 1 if probability > threshold, 0 otherwise
        else:
            binary = (probability > 0.5) # returns 1 if probability > 0.5, 0 otherwise
        
        return binary, probability
    
    def finetune_mode(self):
        """Set model to finetune mode - only finetune blocks and classifier train"""
        self.transformer.finetune_mode()
        self.resnet.finetune_mode()
        self.classifier.train()
        self.is_finetune = True
        print("Setting model to finetune mode - only finetune blocks and classifier train")
    
    def train_mode(self):
        """Set model to full training mode - all components train"""
        self.transformer.train_mode()
        self.resnet.train_mode()
        self.classifier.train()
        self.is_finetune = False
        print("Setting model to train mode - all components train")
    
    def load_state_dict_flexible(self, state_dict, load_finetuned_blocks=False):
        """Load state dict while ignoring finetune_block mismatches"""
        # Filter out finetune_block keys from loaded state_dict
        
        if load_finetuned_blocks:
            filtered_state_dict = state_dict
        else:
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                if not ('finetune_block' in k)}
        
        # Load everything except finetune_block
        self.load_state_dict(filtered_state_dict, strict=False)
        print("Loaded model state (ignoring finetune_blocks for compatibility)")


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

def train_model(training_data_folder, model_folder, sequence_length=1024, dropout=0.1, 
                epochs=15, lr=0.002, batch_size=32, device='cpu', train_verbose=True, 
                generate_holdout=False, model_path=None, finetune=False, aggressive_masking=False, 
                alternate_finetune=False, pretrain_transformer_only=False, false_negative_penalty=1.0):
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
    
    data_stats(df)
    
    df_copy = df.copy()
    train_dataset = ECGDatasetFromPipeline(df_copy)
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
    print(f"Training resnet_trans set: {train_pos}/{train_total} positive ({100*train_pos/train_total:.1f}%)")
    
    # Initialize model
    device = torch.device(device)
    
    if train_verbose:
        print(f"Using device: {device}")
    
    model = ECGChagas(
        sequence_length=sequence_length,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_layers=4,
        transformer_dim_feedforward=256,
        resnet_base_channels=64,
        resnet_num_blocks=[2, 2, 2, 2],
        classifier_hidden_dim=128,
        dropout=dropout,
        base_parameter=1.0
    ).to(device)
    
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict_flexible(state_dict)
    
    if finetune:
        model.finetune_mode()
    elif pretrain_transformer_only:
        # Freeze ResNet, only train transformer and classifier
        for p in model.resnet.parameters():
            p.requires_grad = False
        model.resnet.eval()
        model.transformer.train()
        model.classifier.train()
        print("Pretraining transformer only - ResNet frozen")
    else:
        model.train_mode()
    
    # Simple training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_challenge_score = 0
    
    print("Starting training...")
    
    # Train for epochs
    for epoch in range(epochs):
        # Re-mask data each epoch if aggressive masking is enabled
        # TRAINING PHASE
        print()
        print("="*10 + f" NEW EPOCH " + "="*10)
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        if aggressive_masking:
            print("Aggressive masking enabled - data will be re-masked to generate more data")
            lead_mask_ratio = 0.5
            span_mask_ratio = 0.5
            print(f"    - Lead mask ratio: {lead_mask_ratio}")
            print(f"    - Span mask ratio: {span_mask_ratio}")
            df = mask_ecg(df_copy, num_lead_masks=1, num_span_masks=1, 
                        lead_mask_ratio=lead_mask_ratio, span_length=None, 
                        span_mask_ratio=span_mask_ratio, mask_value=0.0, include_original=True)
            train_dataset = ECGDatasetFromPipeline(df)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if finetune and alternate_finetune: # Alternate finetune mode
            if (epoch + 1) % 5 == 0:
                model.train_mode()
                print("Setting model to train mode - All components train")
            else:
                model.finetune_mode()
                print("Setting model to finetune mode - Only finetune blocks and classifier train")
                
        logits_for_iso = []
        labels_for_iso = []
        
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
            logits_for_iso.extend(logits.detach().cpu().numpy().flatten())
            labels_for_iso.extend(labels.detach().cpu().numpy().flatten())
            
            # ===================================
            ## WEIGHTED PENALITY FOR FALSE NEGATIVES
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
        
        # model.isotonic_regression.fit(logits_for_iso, labels_for_iso)
        diagnostics = diagnose_model_probabilities(model, val_loader, device='cuda:0', verbose=False)
        challenge_score = diagnostics['challenge_score']
        tp = diagnostics['tp']
        fn = diagnostics['fn']
        tn = diagnostics['tn']
        fp = diagnostics['fp']

        if challenge_score > best_challenge_score:
            best_challenge_score = challenge_score
            torch.save(model.state_dict(), os.path.join(model_folder, f'resnet_trans_trained_model.pkl'))
            # joblib.dump(model.isotonic_regression, os.path.join(model_folder, f'resnet_trans_trained_model_iso.joblib'))
            print("*" * 50)
            print(f"Best model saved.")
            print(f"    - Challenge score: {challenge_score:.7f}")
            print(f"    - Acc: {val_acc:.7f}")
            print(f"    - Saved model at {os.path.join(model_folder, f'resnet_trans_trained_model.pkl')}")
            print(f"    - Saved isotonic regression at {os.path.join(model_folder, f'resnet_trans_trained_model_iso.joblib')}")
        
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
                print(f"  Val   - Challenge score: {challenge_score:.7f}")
                print(f"  Val   - Total Predictions: {tp + fn + tn + fp} | TP: {tp}, FN: {fn}, TN: {tn}, FP: {fp}")
            except ImportError:
                print(f"Epoch {epoch}:")
                print(f"  Train - Loss: {train_loss_avg:.7f}, Acc: {train_acc:.7f}, LR: {scheduler.get_last_lr()[0]:.7f}")
                print(f"  Val   - Loss: {val_loss_avg:.7f}, Acc: {val_acc:.7f}")
                print(f"  Val   - Challenge score: {challenge_score:.7f}")
                print(f"  Val   - Total Predictions: {tp + fn + tn + fp} | TP: {tp}, FN: {fn}, TN: {tn}, FP: {fp}")
                print("  (Install sklearn for AUC/Precision/Recall metrics)")
            
            print("-" * 50)
    
    # Save model
    print(final_stats)
    
    return model, final_stats

# ============================================================================
# SIMPLE INFERENCE FUNCTION
# ============================================================================

def load_model(model_path, is_finetune=False, device='cpu'):
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
        sequence_length=512,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_layers=4,
        transformer_dim_feedforward=256,
        resnet_base_channels=64,
        resnet_num_blocks=[2, 2, 2, 2],
        classifier_hidden_dim=128,
        dropout=0.1,
        base_parameter=1.0,
        device=device
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict_flexible(state_dict, load_finetuned_blocks=is_finetune)
    # model.isotonic_regression = joblib.load(os.path.join(model_path.split('.')[0] + '_iso.joblib'))
    if is_finetune:
        model.finetune_mode()
    else:
        model.train_mode()
        
    from modules.threshold_optimization import binary_threshold_optimization
    df = load_data("./holdout_data", "holdout-data", use_cache=False, sequence_length=512)
    val_dataset = ECGDatasetFromPipeline(df)
    val_dataset = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    best_threshold, best_accuracy = binary_threshold_optimization(model, val_dataset, 0.001, device='cuda:0')
    model.threshold = best_threshold
    
    print(f"Best threshold: {best_threshold} | Accuracy: {best_accuracy}")
    
    return model

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Train
    data_folder = "../data"
    model, final_stats = train_model(data_folder)
    print(final_stats) 