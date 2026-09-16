# 27th place solution - Only Pre&Postprocessing

Competition: physionet-ecg-image-digitization
Rank: #27
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/27th-place-solution-only-pre-and-postprocessing

## Acknowledgements

First of all, **huge thanks to PhysioNet and Kaggle** for organizing this challenging and meaningful competition.

We would also like to express our sincere gratitude to the community members who shared high-quality public baselines and insights. In particular, **huge thanks to [@hengck23](https://www.kaggle.com/hengck23), [@wasupandceacar](https://www.kaggle.com/wasupandceacar)**, and others for their generous public releases, which provided extremely solid foundations to build upon.

Finally, **congratulations to the winning teams** — their solutions were inspiring and very enlightening :).  

P/s: we will public the notebook soon after some cleaning.

## Overview

Our final solution achieved **19.59 dB SNR on the Private Leaderboard**, ranking **27th**.
  
Overall, our pipeline follows a standard and well-proven structure:
  
**Normalization / Rectification → Segmentation → Post-processing**
  
- **Stage 0 & Stage 1 (Normalization / Rectification)**  
  Taken directly from **hengck23’s pipeline**, which remains one of the strongest and most robust approaches for ECG image alignment and grid rectification.
  
- **Stage 2 (Segmentation)**  
  Based on the U-Net style segmentation model released by **wasupandceacar**, serving as a very strong baseline for pixel-level ECG trace extraction.
  
On top of these baselines, our main contributions are:
- Improved preprocessing before Stage 0   
- A new pixel-to-series postprocessing function   
- Additional safeguards against outliers and noise   
- A learned signal refinement model applied after postprocessing   

## Improved preprocessing before stage 0
   
While Stage 0 and Stage 1 from hengck23 are already strong, we observed that **input image quality** (contrast, illumination, stains, shadows) significantly affects downstream rectification quality.
   
Inspired by the notebook  *Visual QA for all stages of ECG digitization* by [@sanpier](https://www.kaggle.com/sanpier),  we adopted a **dual-path strategy**:
   
- Run Stage 0 → Stage 1 **without preprocessing**   
- Run Stage 0 → Stage 1 **with preprocessing**   
- Compare Stage-1 quality metrics   
- Select the better output for Stage 2   

This alone improved our public leaderboard score by **~0.3 dB**.   

## Image-type-aware preprocessing

We further observed that:   
- The public image-type classifier is not always reliable   
- Different ECG image sources degrade in very different ways   
  
Therefore, we trained **our own image-type classifier** and designed a **safe, gated preprocessing function** that:   
- Never applies destructive operations unconditionally   
- Uses illumination and contrast as safety gates   
- Applies only mild, source-specific adjustments   
    
This refinement provided another **~0.3 dB improvement** (so in total 0.6dB).   

Example code:

```python
def preprocess_by_source(img_bgr, pred_src):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    std0 = float(gray.std())
    illum = illumination_strength(img_bgr, sigma=35)

    # universal safe fixes
    if illum > 0.14:
        img_bgr = bg_correct_lab_l(img_bgr)

    if std0 < 30:
        img_bgr = clahe_luminance_bgr(img_bgr, clip=1.15, tile=8)

    if s in ["0003", "0011"]:  # color scans / mold color scans
        x = grayworld_white_balance(x)
        if float(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std()) < 35:
            x = clahe_luminance_bgr(x, clip=1.2, tile=8)

    elif s in ["0006"]:  # screen photos
        x = denoise_bilateral(x, d=5, sigmaColor=25, sigmaSpace=25)
        if float(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std()) < 35:
            x = clahe_luminance_bgr(x, clip=1.2, tile=8)

    elif s in ["0005", "0009", "0010"]:  # mobile printed / stained / damaged
        if illum > 0.14:
            x = denoise_median(x, k=3)
        elif std0 < 35:
            x = denoise_bilateral(x, d=5, sigmaColor=25, sigmaSpace=25)

        # DO NOT apply strong CLAHE for 0009 (stained) unless really low contrast
        if s == "0009":
            if float(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std()) < 25:
                x = clahe_luminance_bgr(x, clip=1.1, tile=8)

    elif s in ["0004", "0012", "0001"]:
        pass

    return img_bgr
```

More details and full code are available in our public notebook.   

## Postprocessing: Pixel → Series

Our **largest single gain (+ ~1.3 dB compared to baseline)** came from redesigning the pixel-to-series conversion.
   
Baseline approaches typically:   
- Take argmax per column   
- Replace low-confidence columns with a fixed baseline   

We found this often introduces quantization noise, sudden jumps and artificial flat regions. Instead, we built a more stable conversion function with three key ideas:   
1) **Quadratic sub-pixel refinement**     
   For each column, we first take the hard argmax y₀, then refine it using a 3-point quadratic (parabolic) fit around (y₀−1, y₀, y₀+1). This provides sub-pixel accuracy and reduces staircase artifacts in the recovered waveform.   

2) **Controlled gap handling via NaN + interpolation**     
   Columns with no foreground (or very low confidence) are marked as missing (NaN), rather than being replaced with a constant baseline. We then interpolate both the extracted y-path and its confidence across these gaps using neighboring columns. This preserves continuity and avoids introducing artificial plateaus.   

3) **Confidence-aware despiking in pixel space**     
   After interpolation, we apply a small median-filter based “despike” step that only removes sharp outliers when the model confidence is low. This helps remove random jumps caused by noise/grid artifacts without suppressing real ECG peaks (e.g. QRS complexes).   

Finally, the cleaned sub-pixel trace is converted into mV using the known `zero_mv` offsets and `mv_to_pixel` scale, and lightly smoothed with Savitzky–Golay filtering.   

```python
def pixel_to_series_v2_47_quadratic_interpnan(pixel, zero_mv, length, mv_to_pixel, miss_thr=0.1):
    """
    Convert stage-2 pixel probabilities (4,H,W) into 4-channel voltage series (4,W),
    then optionally resample to `length`.

    Key ideas:
      - Gaussian blur for argmax stability
      - Quadratic sub-pixel refinement around the argmax
      - Mark low-confidence / empty columns as NaN, then interpolate (instead of baseline fill)
      - Confidence-aware despiking in pixel space
      - Convert pixels -> mV and apply light SavGol smoothing
    """
    _, H, W = pixel.shape
    series = []

    for j in range(4):
        p = pixel[j]  # (H, W)

        # 1) Smooth only to stabilize argmax (thin traces are noisy)
        p_smooth = cv2.GaussianBlur(p, (3, 1), 0)

        # 2) Hard argmax per column (integer y)
        idx = p_smooth.argmax(axis=0)
        s = idx.astype(np.float32)
        conf = p.max(axis=0).astype(np.float32)

        # 3) Quadratic sub-pixel refinement using (y-1, y, y+1)
        for x in range(W):
            y0 = idx[x]
            if 1 < y0 < H - 2 and p[y0, x] > miss_thr:
                vL = float(p[y0 - 1, x])
                vC = float(p[y0,     x])
                vR = float(p[y0 + 1, x])
                denom = 2.0 * (vL - 2.0 * vC + vR)
                if abs(denom) > 1e-6:
                    delta = (vL - vR) / denom
                    if -0.7 <= delta <= 0.7:
                        s[x] = float(y0) + float(delta)

        # 4) Missing/low-confidence columns -> NaN, then interpolate
        miss = ((p > miss_thr).sum(axis=0) == 0) | (conf <= miss_thr)
        s = s.astype(float)
        s[miss] = np.nan
        conf = conf.astype(float)
        conf[miss] = np.nan

        if np.isnan(s).all():
            s[:] = float(zero_mv[j])
            conf[:] = 0.0
        else:
            s = interpolate_nans(s)
            conf = interpolate_nans(conf)

        series.append((s.astype(np.float32), conf.astype(np.float32)))

    # 5) Postprocessing per channel: despike (pixel space) -> convert to mV -> SavGol
    final_series = []
    for k, (s, conf) in enumerate(series):
        s_clean = conf_aware_despike(s, conf, kernel_size=3, threshold=2.0, conf_guard=0.3)
        s_mv = (zero_mv[k] - s_clean) / mv_to_pixel
        s_mv = savgol_filter(s_mv, window_length=7, polyorder=5)
        final_series.append(s_mv)

    series_final = np.stack(final_series).astype(np.float32)

    # 6) Optional resample along time axis
    if length is not None and length != W:
        series_final = torch.from_numpy(series_final).unsqueeze(1)  # (4,1,W)
        series_final = F.interpolate(series_final, size=length, mode="linear", align_corners=False)
        series_final = series_final.squeeze(1).cpu().numpy()

    return series_final

```

## Other minor improvements   

We added several lightweight postprocessing safeguards to improve robustness on difficult images. Each provided small but consistent gains (**≈ +0.01 to +0.05 dB individually**).   

- **Boundary spike cleaning:** fixes short artificial jumps at lead boundaries after lead splitting.   
- **Physiological consistency:** applies a soft Einthoven correction (II ≈ I + III) on short leads.   
- **Image-type–dependent calibration:** adjusts pixel-to-mV scaling based on image source:   
```python
if src in ['0005', '0006', '0009', '0010']:
    mv_to_pixel = 78.3
else:
    mv_to_pixel = 79.5
```

## Learned signal refinement (Black-Box)

Finally, we introduced a **learned refinement stage**:   

This black-box refinement consistently improved results by +0.4 to +0.6 dB, depending on configuration, and turned out to be one of the most effective final steps.   

Specifically, after extracting the 4-row voltage series, we applied a **black-box 1D refinement network** to denoise and correct systematic extraction errors. The refiner is an enhanced **1D U-Net++** (and other variants for ensembling).   

Practical setup:   
- Train 1D neural networks directly on:   
  - Extracted ECG signals (after postprocessing)   
  - Ground-truth ECG waveforms   

- We trained **separate refiners per target sampling length / frequency** (e.g., 2500/2560/5000/5120/10000/10250),   
- Trained multiple folds / variants, then **ensembled** them at inference (simple averaging).   

This final refinement step was one of our biggest gains, improving roughly **+0.4 to +0.6 dB** depending on configuration.   

## Some visuals explaining some of our ideas
In these image the extracted series are in red, the blue series are ground truth ones.   

.png?generation=1769595407668225&alt=media)

Here the segmentation is less confident and all values turn to baseline.

.png?generation=1769595434457066&alt=media)

Here the segmentation create weird confident pixel (sometimes due to grid, sometimes due to noise in B/W scans).

.png?generation=1769595455111012&alt=media)

Here outlier peak

.png?generation=1769595473117005&alt=media)

Here outlier due to miss-predict grid as signal

.png?generation=1769595489612472&alt=media)

Here outlier peak

.png?generation=1769595505436361&alt=media)

Here the case where we haven't yet added some specific tailoring :)


## Discussion on other things we have tried but not too successful :)

We also invested significant effort into rebuilding **Stage 2 (segmentation)** from scratch, but did not achieve improvements over the public baselines.
   
- For ground truth generation, we followed a slightly different path than most write-ups:  
  we converted the provided CSV signals into `.dat` / `.hea` files and reused **ecg_image_kit** to regenerate ECG images, allowing us to extract point-level annotations from the resulting JSON files.
   
- From these points, we experimented with many ways of rendering segmentation masks:   
  - Different drawing backends (OpenCV, Pillow, Matplotlib),   
  - Different line modes (anti-aliasing, `cv2.LINE_8`, etc.),   
  - Different line thicknesses.     
  We observed that **thinner lines consistently gave better extraction metrics**.   
   
- We also tried:   
  - Upscaling point coordinates (×2, ×4) before drawing,   
  - Resampling points so that, e.g., 250 Hz signals mapped exactly to 2500 columns.   

- However, even on clean image types (e.g. `0001`), these generated masks only achieved **~24–28 dB upper-bound SNR** when re-extracted, and performance degraded quickly across configurations.   

- We trained segmentation models:   
  - On **non-rectified images** with heavy augmentation (Public LB ≈ 13.5 dB),   
  - On **Stage-1-rectified images**, where masks also had to be rectified.   
    We tried two rectification strategies:   
    1) Draw mask first, then warp using the homography,   
    2) Warp point coordinates first, then draw the mask.     
    Despite trying multiple losses and regularizations, these models plateaued at **~14–15 dB Public LB**.   
  
Overall, despite extensive experiments, we were unable to outperform the public Stage-2 segmentation models.     
We suspect the issue may stem from subtle differences in GT construction (our ecg_image_kit-based pipeline vs. others), but we are still curious why rebuilding Stage 2 did not yield gains.   

Finally, despite what did not work, we learned a huge amount from this competition.     
The required precision—both in image geometry and signal reconstruction—was truly impressive, and it gave us a much deeper appreciation of how sensitive ECG digitization is to even sub-pixel errors.   

Thanks a lot for reading through to the end 🙂     
We’re very happy to discuss ideas, failures, or details further in the comments.
