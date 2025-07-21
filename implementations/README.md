Each implementation must have a `train_model` function, it must have the params:

- `training_data_folder`
- `model_folder`
- `sequence_length`
- `dropout`
- `epochs`
- `batch_size`
- `lr`
- `device`
- `train_verbose`

---

## State-of-the-Art ResNet Architecture Features:

### Core Architecture Components:
- **Squeeze-and-Excitation (SE) Blocks**: Implemented SE attention mechanism for channel-wise feature recalibration, which significantly improves model performance
- **Multi-Scale Feature Extraction**: Uses an Inception-like module with different kernel sizes (1, 3, 5) to capture ECG features at multiple temporal scales
- **Dilated Convolutions**: Employs dilated convolutions with increasing dilation rates (1, 2, 4) to increase receptive field without losing resolution
- **Lead-Specific Processing**: Each of the 12 ECG leads has its own dedicated encoder (similar to the GNN approach), allowing the model to learn lead-specific features
- **Cross-Lead Attention**: Implements multi-head attention mechanism to model inter-lead relationships, similar to the GAT's graph attention but using transformer-style attention

### Deep Architecture:
- 12 parallel ResNet encoders (one per lead)
- 3 stages of residual blocks with progressive channel expansion (256→512→1024)
- Multiple attention layers for lead interaction
- Deep classifier with layer normalization

### Modern Training Techniques:
- AdamW optimizer with weight decay
- Cosine annealing with warm restarts for learning rate scheduling
- Gradient clipping for stability
- Comprehensive data validation

### Same Interface as GNN:
- `train_model()` function with identical parameters
- `load_model()` function for inference
- Support for finetune mode
- False negative penalty weighting
- Same data pipeline and augmentation

### Model Specifications:
- **Input**: (batch_size, sequence_length, 12) ECG signals
- **Architecture**: 12 parallel ResNet encoders → Cross-lead attention → Global pooling → Deep classifier
- **Parameters**: Significantly more parameters than the GNN to match "amount of resources"
- **Advanced Features**: SE blocks, dilated convolutions, multi-scale processing, attention mechanisms