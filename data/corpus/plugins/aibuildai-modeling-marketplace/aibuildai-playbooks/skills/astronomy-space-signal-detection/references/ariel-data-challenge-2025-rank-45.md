# 45th place solution

Competition: ariel-data-challenge-2025
Rank: #45
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/45th-place-solution

# NeurIPS Ariel Data Challenge 2025: A Hybrid Approach to Exoplanet Transit Spectroscopy

We express our gratitude to the organizers and participants of the NeurIPS Ariel Data Challenge 2025 for creating this valuable scientific competition.

## Executive Summary

Notebook: https://www.kaggle.com/code/devadevam/fork-of-fork-of-fork-of-fork-of-notebookfc5-e6859d

main ideas:
- estimating sigma as difference between the base non-ML model's prediction and the target value of mu (check the `SafeMLCalibrator` class)
- Astrophysical features through dimensional analysis (the `add_interaction_features` function)
- atmospheric physics-based features: equilibrium temperature modeling, stellar insolation calculations, Rayleigh scattering slope analysis, and molecular absorption signatures (the `add_atmospheric_physics_features` function)

## Core Methodology

Our solution architecture employs a multi-layered approach combining:

- **Foundation Layer**: Analytical transit modeling providing physically-motivated baseline predictions
- **Enhancement Layer**: Machine learning calibration for systematic bias correction and prediction refinement  
- **Feature Layer**: Physics-informed engineering incorporating stellar and atmospheric physics
- **Validation Layer**: Robust uncertainty quantification with proper statistical calibration

New ideas first:

## Feature Engineering

### Temporal and Spectral Characterization

We extract comprehensive statistical descriptors from preprocessed time series:

**Statistical Moments**: Complete characterization including central tendencies, dispersion measures, asymmetry (skewness), and tail behavior (kurtosis)

**Distributional Features**: Multi-percentile analysis (5th, 25th, 50th, 75th, 95th percentiles) and root-mean-square calculations

**Temporal Correlations**: Autocorrelation function analysis at multiple lag intervals for time-domain pattern recognition

**Frequency Domain Analysis**: Fast Fourier Transform decomposition with energy partitioning across low, mid, and high-frequency bands, spectral centroid computation, and zero-crossing rate analysis

### Astrophysical Feature Engineering

We incorporate domain-specific features derived from fundamental stellar and planetary physics:

**Orbital Mechanics**: Impact parameter calculations, orbital velocity estimates, mean motion derivations, and Keplerian mass-radius relationships

**Atmospheric Physics**: Equilibrium temperature modeling, stellar insolation calculations, Rayleigh scattering slope analysis, and molecular absorption signatures (H₂O, CO₂, CO, CH₄)

**Cross-Parameter Interactions**: Systematically constructed interaction terms, logarithmic transformations for scale invariance, and trigonometric mappings of orbital geometry

### Spectroscopic Analysis Features

Drawing from established atmospheric retrieval methodologies:

- **Spectral Modeling**: Blackbody correlation analysis and continuum characterization
- **Morphological Analysis**: Spectral slope, curvature measurements, and absorption feature detection
- **Coherence Metrics**: Transmission spectrum consistency analysis across wavelength channels

### Multi-Scale and Quality Metrics

- **Hierarchical Analysis**: Feature computation across multiple smoothing scales (3, 7, 15, 31 data points)
- **Signal Quality**: Transit signal-to-noise ratio quantification and V-shaped morphology assessment
- **Data Integrity**: Completeness scoring and quality assurance metrics

## Machine Learning Enhancement Framework

### Safe ML Calibrator Design

Our primary methodological contribution centers on conservative machine learning integration:

**Scale Calibration Module**:
- Linear regression framework: `depth_calibrated = α + β × depth_analytical`
- Rigorous out-of-fold cross-validation preventing information leakage

**Spectral Shape Calibration**:
- Principal Component Analysis on normalized transmission spectra
- Regularized regression on principal components using Ridge methodology
- Full spectral reconstruction with uncertainty propagation

**Uncertainty Calibration System**:
- Instrument-specific Ridge regression models for FGS and AIRS uncertainty estimation
- Geometric blending in logarithmic space for numerical stability
- Integration time weighting for observation quality assessment

### Statistical Regularization

- **Feature Curation**: Systematic removal of constant and near-constant predictors
- **Multicollinearity Management**: Correlation-based pruning with threshold τ = 0.95
- **Selection Methodology**: Permutation importance-based feature ranking
- **Conservative Blending**: Strictly constrained combination weights (α ≤ 0.3, β ≤ 0.3, γ = 0.55)

### Quality-Aware Training Protocol

We implement sophisticated quality assessment for optimal training sample weighting:

- **Malformation Detection**: Automated identification of corrupted transit signals
- **Signal-to-Noise Quantification**: Comprehensive SNR-based scoring methodology
- **Adaptive Cross-Validation**: Quality-weighted fold construction for robust model evaluation

## Uncertainty Quantification

Sigma is estimated as the difference between base model's extimation of mu and the target value.

### Baseline Uncertainty Modeling

**FGS Photometric Uncertainties**:
- In-transit versus out-of-transit variance estimation with robust statistical measures
- Dataset-wide median scaling with conservative clipping bounds

**AIRS Spectroscopic Uncertainties**:
- Wavelength-dependent variance modeling accounting for detector characteristics
- Integration time normalization with conservative scaling factors

### Uncertainty Estimation

- **Predictive Modeling**: Ridge regression on engineered uncertainty features
- **Robust Combination**: Geometric blending with baseline uncertainty estimates
- **Planet-Specific Calibration**: Optimal uncertainty multiplier (c*) determination
- **Physical Constraints**: Spectral smoothness requirements ensuring realistic uncertainty profiles

## Model Training and Validation

### Cross-Validation Architecture

- **Stratification Strategy**: 8-fold cross-validation with quality-aware stratification
- **Sample Weighting**: Quality-based importance weighting during training
- **Unbiased Evaluation**: Strict out-of-fold prediction protocols
- **Independent Validation**: Hold-out testing on reserved training data

### Hyperparameter Optimization

- **Regularization Strategy**: Conservative Ridge penalties (α = 0.5-1.0) preventing overfitting
- **Dimensionality Control**: Limited principal component retention (2-4 components)
- **Safety Constraints**: Maximum blending weights capped at 0.3 for methodological conservatism

## Data Processing Pipeline

This part has been mostly taken from this public code: https://www.kaggle.com/code/antonsibilev/very-fast-1h-optimized-nb-with-0-333

### Instrumental Calibration

**AIRS-CH0 Spectroscopic Data:**
- Applied instrument-specific linear correction coefficients to mitigate systematic offsets
- Implemented correlated double sampling (CDS) for readout noise suppression
- Performed wavelength-dependent flat field correction with hot pixel identification and masking
- Executed robust statistical binning with iterative sigma-clipping for outlier rejection
- Applied adaptive smoothing algorithms with phase-aware processing to preserve transit signals while reducing noise

**FGS1 Photometric Data:**
- Optimized CDS processing for single-channel detector characteristics
- Implemented spatial binning across detector arrays to improve signal-to-noise ratio
- Established comprehensive quality assurance protocols for data completeness validation

### Transit Signal Processing

- **Noise Reduction**: Savitzky-Golay filtering with optimized window sizes and polynomial orders
- **Feature Detection**: Piecewise polynomial fitting algorithms for precise ingress/egress identification
- **Phase-Aware Processing**: Adaptive transit masking that preserves astrophysical signals
- **Quality Assessment**: Multi-metric evaluation system for identifying low signal-to-noise or malformed transits

## Analytical Transit Modeling Framework

### Mathematical Formulation

An extended transit model that captures the fundamental physics of planetary occultation:

```
F(t) = F₀(t) × [1 - δ × T(t)]
```

Where:
- F₀(t) represents the stellar flux continuum modeled through polynomial detrending
- δ quantifies the wavelength-dependent transit depth
- T(t) describes the normalized transit light curve profile

### Optimization Strategy

- **Parameter Estimation**: Nelder-Mead simplex optimization with robust convergence criteria
- **Constraint Management**: Delta-margin constraints preventing boundary effects and ensuring physical validity
- **Phase Analysis**: Gradient-based algorithms for precise transit timing determination
- **Systematic Correction**: Third-order polynomial detrending for instrumental and stellar variability removal
