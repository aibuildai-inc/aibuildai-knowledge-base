# 9th place solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #9
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/9th-place-solution

# 9th Place Solution - Child Mind Institute Competition

Thank you Kaggle and Child Mind Institute for hosting this fascinating competition! This was an incredible learning experience working with multimodal sensor data for behavioral analysis.

## Solution Overview

**Three model types** leveraging different sensor modalities:
- **IMU-only**: Inertial measurement data (raw features + engineered)
- **THM + IMU**: Thermal (raw + aggregations + calculated) + inertial fusion
- **TOF + IMU**: Time-of-flight (aggregations + calculated only, no raw) + inertial fusion



## Base Model Architecture

**Hybrid CNN-RNN Multi-Modal Neural Network:** (similar to public one)
- **Feature Extraction**: Residual CNN blocks + SE-attention → spatial feature learning
- **Temporal Modeling**: Bidirectional GRU + attention mechanism → complex temporal dependencies
- **Multi-Sensor Fusion**: Parallel processing of IMU + thermal + ToF sensor data → classification

```python
class Net(nn.Module):
    def __init__(self, cfg):
        # IMU deep branch
        self.imu_block1 = ResidualSECNNBlock(self.imu_dim, 64, 3)
        self.imu_block2 = ResidualSECNNBlock(64, self.sequence_length, 5)
        
        # Bidirectional GRU
        self.bigru = nn.GRU(self.sequence_length, self.sequence_length, 
                           bidirectional=True, batch_first=True)
        
        # Attention mechanism
        self.attention = AttentionLayer(self.sequence_length * 2)
```

## Key Improvements

### 1. **Handedness Normalization**
- **Issue**: Left vs right-handed subjects create inconsistent patterns
- **Fix**: Coordinate system flip (`-acc_x, -rot_y, -rot_z`)

### 2. **Normalization Strategy**
- **No preprocessing normalization** → let model learn optimal scaling
- **Batch normalization** throughout network → stable training

### 3. **Target modification**
- I used gesture + orientation target (a total of 51 targets)
### 4. **Feature Engineering Strategy**
- **TOF data**: Only fe features (not rawt)
- **THM data**: raw + fe features
- **IMU data**: raw + fe features (same as public notebooks)

### 5. **Data Augmentation**
- Mixup augmentation for temporal sequences
-  Same augmenations than public notebooks

### 6. **Model Ensembling**
- EMA (Exponential Moving Average) for stable predictions (I used ema 0.8, ema 0.9 y normal data for ensembling)
- Each model train with a subset of the total variables, for gaining robustness and diversity

### 7. **Inverse Target Validation Strategy**
This was a crucial innovation that significantly improved the postprocessing effectiveness:

- **Normal Model**: 51 targets where 50 are 0 and 1 is 1 (standard classification)
- **Inverse Model**: 51 targets where 50 are 1 and 1 is 0 (inverted classification)

For each model type (IMU-only, THM+IMU, TOF+IMU), I trained both normal and inverse versions.

**Why this works:** (though I'm not entirely certain)
- **Complementary pattern learning**: Captures both positive and negative behavioral indicators
- **Enhanced uncertainty representation**: When both models agree → high confidence, when they disagree → uncertainty requiring subject-specific context
- **Implicit regularization**: Forces the model to learn consistent representations in both directions
- **Better exploitation of multi-class exclusivity**: If something is NOT class A, B, C... it provides information about what it could be

## The Game Changer: Postprocessing Pipeline

**Discovery**: 13 days before competition end, I realized that past predictions from the same subject can significantly improve current predictions.

### Processing Flow

```
Input Sequence → Sensor Detection → Model Selection → Normal + Inverse Prediction → Subject History → Final Output
```

**Step-by-step Process:**

1. **Sensor Availability Check**
   ```
   Available Sensors → Model Choice
   IMU only        → IMU Model (Normal + Inverse)
   IMU + THM       → THM+IMU Model (Normal + Inverse)
   IMU + TOF       → TOF+IMU Model (Normal + Inverse)
   ```

2. **Dual Model Prediction**
   - Generate predictions using both normal and inverse models
   - Apply inverse target validation combining both outputs
   - The combination of normal and inverse predictions creates a more robust uncertainty estimate

3. **Subject-Aware Postprocessing**
   ```
   Current Prediction (Normal + Inverse) + Historical Sequences (same subject) → Enhanced Prediction
   ```
   - Leverage temporal consistency within subjects
   - Use past predictions to inform current ones
   - The inverse model is particularly effective here because it helps identify "what this specific subject would NOT do" based on their history

### Results Impact

| Metric | No Postprocessing | With Postprocessing | Improvement |
|--------|------------------|-------------------|-------------|
| **IMU CV** | 0.81 | 0.84 | +0.03 |
| **ALL CV** | 0.84 | 0.89 | +0.05 |
| **Public LB** | 0.850 | 0.882 | +0.032 |
| **Private LB** | 0.841 | 0.868 | +0.027 |

*CV results based on 1 fold out of 5 for time efficiency*


### Model Training Details
- **Sequence length**: Primarily 140, with 160 for final ensemble diversity
- **Loss function**: BCEWithLogitsLoss for multi-label classification
- **Optimizer**: AdamW with linear learning rate scheduling
- **Regularization**: Dropout layers + L2 weight decay
- **Final ensemble**: Multiple models trained on complete dataset for maximum performance (both normal and inverse versions)
- **Preprocessing variants**: Tested different preprocessing pipelines for robustness (no gains observed)

## Lessons Learned

- **Late-stage discoveries** can dramatically change competition outcomes
- **Inverse modeling approach** can provide complementary information that standard approaches miss
- **Luck factor**: I believe I had some luck in discovering the postprocessing method that worked so well, though I'm still not entirely sure why the inverse target approach works as effectively as it does

Thank you to all participants for the great competition and shared insights!
