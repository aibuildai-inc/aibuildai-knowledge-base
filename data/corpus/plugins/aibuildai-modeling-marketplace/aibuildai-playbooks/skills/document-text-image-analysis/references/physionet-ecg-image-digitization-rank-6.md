# 6th place solution

Competition: physionet-ecg-image-digitization
Rank: #6
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/6th-place-solution

## Acknowledgments
I would like to express my sincere gratitude to Kaggle and the competition organizers for providing this invaluable opportunity. Special thanks to @hengck23 for sharing his strong baseline, which served as a crucial foundation for my work.

## Summary
My solution focuses on optimizing Stage 2 of the baseline provided by @hengck23. The key insight is to directly regress the lead signal, bypassing the traditional pipeline of "segmentation followed by post-processing." By treating this as a direct regression task, the model learns the signal features more effectively and reduces the cumulative error often introduced during the post-processing stage.

## Overall Pipeline



### Resample

The training dataset contains ECG signals with diverse sampling rates, ranging from `2.5 kHz` to `10 kHz`. To ensure high-quality ground truth and maintain optimal Signal-to-Noise Ratio (SNR), a consistent resampling strategy is required. I implemented a benchmarking framework to evaluate the fidelity of various resampling algorithms—including `polyphase, linear, cubic spline, and FFT-based methods`. The performance was measured using a transformation:
1. Up/Down-sample: Resample the original signal to a target length (e.g., 2560, 5120, or 10250). 
2. Restore: Resample the signal back to its original length.
3. Evaluate: Calculate the SNR by comparing the "Restored" signal against the "Original" ground truth.

The key observations is
- scipy.signal.resample (FFT-based) yields significantly superior results compared to torch.nn.functional.interpolate (Linear/Bilinear) for this signal processing task.
- fidelity is positively correlated with the intermediate sampling density; higher intermediate lengths (10250 > 5120 > 2560) result in substantially lower information loss.



Moreover, to accelerate image resampling and integrate it into the training process, I use `resample_torch`

```python
def resample_torch(self, x, num, dim=-1):
        dim = (x.dim() + dim) if dim < 0 else dim
        X = torch.fft.fft(x, dim=dim)
        Nx = X.shape[dim]

        sl = [slice(None)] * X.ndim
        newshape = list(X.shape)
        newshape[dim] = num
        Y = torch.zeros(newshape, dtype=X.dtype, device=X.device)

        N = min(num, Nx)
        sl[dim] = slice(0, (N + 1) // 2)
        Y[sl] = X[sl]
        sl[dim] = slice(-(N - 1) // 2, None)
        Y[sl] = X[sl]

        if N % 2 == 0:
            if N < Nx:
                sl[dim] = slice(N//2, N//2+1)
                Y[sl] += X[sl]
            elif N < num:
                sl[dim] = slice(num-N//2, num-N//2+1)
                Y[sl] /= 2
                temp = Y[sl]
                sl[dim] = slice(N//2, N//2+1)
                Y[sl] = temp

        y = torch.fft.ifft(Y, dim=dim).real * (float(num) / float(Nx))
        return y
```

### Signal Regression Head
This module converts 2D feature embeddings into precise physical voltage values for ECG leads by using a Soft-Argmax mechanism to estimate vertical coordinates.

```python
# Signal Regression Head
class MaskEmbeddingToLeadSignalSoftArgmax(nn.Module):
    def __init__(self, n_leads=4, embedding_dim=32, temperature=0.5):
        super().__init__()
        
        self.n_leads = n_leads
        self.temperature = temperature
        self.lead_y_logits = nn.Conv2d(embedding_dim, n_leads, kernel_size=1)
        
        self.register_buffer('zero_mv', torch.tensor([703.5, 987.5, 1271.5, 1531.5]).view(1, 4, 1))
        self.register_buffer('mv_to_pixel', torch.tensor(79.0))

    def forward(self, masked_feat):
        B, C, H, W = masked_feat.shape
        device = masked_feat.device
        dtype = masked_feat.dtype

        y_logits = self.lead_y_logits(masked_feat)
        prob = torch.softmax(y_logits / self.temperature, dim=2)

        y_coord = torch.arange(H, device=device, dtype=dtype).view(1,1,H,1)
        y_pixel = (prob * y_coord).sum(dim=2)  # [B, L, W]
        
        pred_mv = (self.zero_mv - y_pixel) / self.mv_to_pixel
        return pred_mv, prob
```


## Experimental Results
| Signal Length | TTA (hflip) | Epochs | LB Score | 
|---------------|-------------|--------|----------|
| 2560          | False       | 60     | 21.36    |
| 5120          | False       | 100    | 22.20    |
| 5120          | True        | 100    | 22.36    |
| 5120          | True        | 150    | 22.43    |


## Code:
[Training Code ](https://github.com/GWwangshuo/Kaggle-2025-PhysioNet/tree/main)
