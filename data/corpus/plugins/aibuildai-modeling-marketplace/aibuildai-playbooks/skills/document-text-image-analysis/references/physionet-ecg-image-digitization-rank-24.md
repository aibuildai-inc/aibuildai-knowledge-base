# 24th Place Solution

Competition: physionet-ecg-image-digitization
Rank: #24
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/24th-place-solution

# 24th Place Solution: V19 + V16V18 Ensemble (19.59 SNR)

## Approach: Regression, Not Segmentation

We initially tried segmentation (predicting binary trace masks), but it failed—thick traces, overlapping leads, and the need for sub-pixel accuracy made it impractical. Instead, we frame this as **per-pixel y-coordinate regression**: for each x-position, predict the continuous y-coordinate of the ECG trace.

---

## Data Strategy

**Problem**: Kaggle GT is in millivolts, not pixels. Converting back introduces alignment errors.

**Solution**: Train primarily on **synthetic ECGs** with pixel-perfect ground truth, then mix in Kaggle data for domain adaptation.

We extract **individual rows** as separate samples (4× more data), cropping each row centered on its baseline (±250px) and using only the signal region (3926px wide, no margins).

---

## Architecture

### V16: Per-Lead Regression Baseline

```
Input: [B, 3, 500, 3926] (single row crop)
  │
  ▼
ConvNeXt-Base Encoder (ImageNet-22k pretrained)
  │
  ▼
U-Net Decoder with Skip Connections
  │
  ▼
Height Attention (softmax over y-axis)
  │
  ▼
1D Conv Regression Head → Sigmoid
  │
  ▼
Output: [B, 3926] normalized y-coordinates
```

### V18: Cross-Row Refiner (Stacked on V16)

Takes V16's predictions and refines them by looking at **all 4 rows simultaneously**:

- Creates Gaussian "guide channel" from V16 prediction (σ=15px)
- EfficientNet-B0 encoder (4 channels: RGB + guide)
- **3 Cross-Row Transformer blocks** with self-attention across rows
- Outputs small residual corrections (Tanh × learnable scale ~0.1)

**Key insight**: Cross-row attention implicitly learns Einthoven's Law (II = I + III) and catches inter-row inconsistencies.

### V19: BiLSTM + Deformable Conv

```
Input: [B, 3, 500, 3926]
  │
  ▼
ConvNeXt-Base Encoder
  │
  ▼
U-Net Decoder with Deformable Conv (last 2 stages)
  │
  ▼
Height Attention
  │
  ▼
Bidirectional LSTM (2 layers, hidden=128)
  │
  ▼
Linear Head → Sigmoid
  │
  ▼
Output: [B, 3926]
```

**Deformable Conv**: Learns adaptive receptive fields for curved/warped traces.
**BiLSTM**: Captures long-range temporal dependencies across the full 10-second ECG.

---

## Why Ensemble Works: Complementary Failure Modes

| | V16+V18 | V19 |
|---|---------|-----|
| **Strength** | Cross-row consistency | Long-range temporal |
| **Weakness** | Local receptive field | No cross-row awareness |
| **Fails on** | Baseline drift, rhythm consistency | Cross-row artifacts, lead relationships |

The architectures have **uncorrelated errors** (~0.3-0.4 correlation). When averaged, mistakes cancel out.

```python
ensemble = 0.4 * V19 + 0.6 * V16V18  # Empirically best ratio
```

---

## Training

**Loss Function**:
```python
Loss = 1.0 × MaskedL1 + 0.2 × SNRLoss + 0.1 × GradientLoss
```

- **MaskedL1**: Pixel accuracy on valid regions
- **SNRLoss**: Direct optimization of competition metric (`-10*log10(signal²/noise²)`)
- **GradientLoss**: Smoothness constraint on first derivative

**Setup**:
- AdamW (lr=1e-4, weight_decay=0.01)
- Cosine annealing schedule
- Mixed precision (FP16) with gradient scaling
- Heavy augmentation (noise, blur, color jitter, coarse dropout)
- Multi-GPU via DistributedDataParallel

**V18 Training**: Freeze V16, train refiner to predict small residual corrections.

---

## Post-Processing

- Savitzky-Golay smoothing (window=7, polyorder=2)
- Einthoven's Law correction (α=0.25 error distribution)
- Amplitude clamping (±10 mV)

---

## Results

| Model | Val SNR | Public LB |
|-------|---------|-----------|
| V16 only | ~19.5 dB | ~19.1 |
| V16 + V18 | ~19.7 dB | ~19.5 |
| V19 only | ~20 dB | ~19.5 |
| **Ensemble** | — | **~19.75** |

---

## Key Takeaways

1. **Regression > Segmentation** for sub-pixel trace extraction
2. **Synthetic data** with pixel-perfect GT is essential
3. **Architectural diversity** → uncorrelated errors → effective ensemble
4. **Cross-row attention** (V18) and **BiLSTM** (V19) complement each other perfectly
5. **Direct SNR loss** optimization helps


Thanks for reading!

### Special thanks to @hengck23 for the stage 0/1 pipeline
