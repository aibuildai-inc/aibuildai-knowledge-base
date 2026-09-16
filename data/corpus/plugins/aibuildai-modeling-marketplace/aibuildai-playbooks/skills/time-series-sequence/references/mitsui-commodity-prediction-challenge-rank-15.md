# MITSUI&CO. Commodity Prediction Challenge - 15th Place Solution Writeup

Competition: mitsui-commodity-prediction-challenge
Rank: #15
Source: https://www.kaggle.com/c/mitsui-commodity-prediction-challenge/writeups/mitsui-and-co-commodity-prediction-challenge-15th

# Overview
This solution achieves 15th place using a **online training approach** combined with advanced feature engineering and ensemble learning. The key innovation is retraining models every 7 days during inference using newly available labeled data, allowing the model to adapt to changing market conditions.

## Key Innovation: Online Training During Inference
### The Core Idea
Unlike traditional approaches that use a static model trained once, or train with temporal cross-validation, this solution implements a dynamic training strategy:
* **Initial Training:** Train ensemble models on historical data using **CombinatorialPurgedGroupKFold** cross-validation, as applied in my previous Kaggle competitions.
* **Inference Stage:** As new test data arrives and becomes labeled:
    * Accumulate labeled samples in a buffer
    * Every 7 days (when buffer reaches 7 samples):  train a new set of models (a new fold) for the ensemble.
    * Use the last 7 samples as validation set
    * Make predictions with the updated models
### Rationale
Financial markets exhibit non-stationary behavior, with patterns evolving over time. Continuous retraining allows the model to adapt to recent market regime changes and learn evolving correlations between commodities.

## Model Architecture

### Strategy Pattern Design
The solution implements a **flexible model strategy architecture** that allows easy addition/removal of models:
```python
class ModelStrategy(ABC):
    @abstractmethod
    def build_model(self, input_dim, output_dim)
    def train(...)
    def predict(...)
```

This design enables rapid experimentation with different architectures while maintaining consistent training/evaluation pipelines.

### Ensemble Models
Three complementary deep learning architectures are combined:

#### 1. Attention DNN 
* Multi-layer perceptron with attention mechanism
* Learns to focus on important features dynamically
* Architecture: 1024 → 768 (Attention) → 512 → 384 → 256 → 424 outputs
* Custom ranking loss balancing MSE and Spearman correlation

#### 2. Residual DNN
* Deep network with skip connections preventing gradient vanishing
* Two residual blocks with Add layers for better gradient flow
* More stable training in deep architectures

#### 3. AutoEncoder
* Learns compressed representations (128-dim latent space)
* Multi-task learning: reconstruction + prediction
* Regularizes features through bottleneck for denoising

### Dynamic Ensemble Weighting
The ensemble employs **performance-based weighting** rather than uniform averaging:

1. **Validation Scoring:** Each model evaluated on holdout folds during CV
2. **Test Scoring:** Models also evaluated on test sets when labels become available
3. **Adaptive Weights:** Strategy weights computed from rank-based scoring:
```python
   ranks = len(scores) - np.argsort(np.argsort(scores)) + 1
   weights = ranks / ranks.sum()
```
**4. Fold-level Weights:** Balance performance (α=0.1) and timeliness (1-α=0.9)

This dual evaluation ensures models that perform well on both validation and test data receive higher weights in the final ensemble.

## Feature Engineering
### Advanced Financial Features (2,658 features generated)
#### 1. Momentum Indicators
* **Rate of Change (ROC):** Multi-period momentum (5, 10, 20 days)
* **RSI (Relative Strength Index):** Overbought/oversold signals
* **Williams %R:** Price position in recent range
* **CCI (Commodity Channel Index):** Deviation from typical price

#### 2. Volatility Features
* **Historical Volatility:** Multiple horizons (5-60 days)
* **Garman-Klass Estimator:** Uses OHLC for better vol estimation
* **Volatility of Volatility:** Second-order volatility
* **Semi-Volatility:** Separate upside/downside volatility
* **Jump Variation:** Detects price discontinuities

#### 3. Moving Averages & Trends
* **SMA/EMA:** Multiple periods (5-200 days)
* **MACD:** Convergence/divergence signals
* **Bollinger Bands:** Price position and band width
* **Golden/Death Cross:** 50/200-day MA crossovers

#### 4. Market Microstructure
* **Volume Momentum:** Rate of volume change
* **Amihud Illiquidity:** Price impact per unit volume
* **Volume Trends:** Short vs long-term volume ratios

#### 5. Cross-Market Features
* **Rolling Correlations:** Between LME, JPX, US, FX markets (10-60 day windows)
* **Correlation Stability:** Variance of correlation over time
* **Spread Features:** Price differences within market groups
* **Spread Z-scores:** Mean reversion signals
* **Half-life:** Mean reversion speed estimation

#### 6. Regime Detection
* **ADX (Average Directional Index):** Trend strength
* **Volatility Regime:** Short vs long vol comparison
* **Trend Slope:** Linear regression coefficient
* **Hurst Exponent:** Trending vs mean-reverting behavior

#### 7. Statistical Moments
* **Skewness:** Distribution asymmetry (tail risk)
* **Kurtosis:** Fat-tail detection
* **Z-scores:** Standardized price deviations
* **Coefficient of Variation:** Risk-adjusted returns

### Feature Selection
From 2,658 engineered features, select top 800 using:
* **Mutual Information Regression:** Captures non-linear relationships
* **RobustScaler:** Handles outliers in financial data
* Reduces dimensionality while preserving predictive power
## Training Strategy
### Cross-Validation
* **CombinatorialPurgedGroupKFold:** 5 splits, 1 test splits
* **Purging:** Prevents data leakage between adjacent time periods
* **Embargo:** Adds buffer between train/test to account for autocorrelation
### Custom Loss Function
Hybrid ranking loss combining:
```python
loss = 0.2 * MSE + 0.8 * (1 - Spearman_Correlation)
```
* MSE ensures magnitude accuracy
* Spearman correlation optimizes ranking (the competition metric)
* 80% weight on ranking reflects competition objective

### Training Optimizations
* **EarlyStopping:** Patience=20, prevents overfitting
* **ReduceLROnPlateau:** Adaptive learning rate
* **Gradient Clipping:** Stabilizes training (clipnorm=1.0)
* **Batch Normalization:** Accelerates convergence

## Limitations and Future Work

### Issue: Feature Selection Limitation
**Problem:** Feature selection utilizes only the first target (target_0) for scoring, but applies the selected features to all 424 targets.
 
**Impact:** Despite this limitation, this solution achieved 15th place. Proper multi-target feature selection could improve performance.

### Better Approach

* Select features independently per target group
* Use average mutual information across all targets
* Implement hierarchical feature selection

### Potential Improvements

#### 1. Better Feature Selection:
* Target-specific feature selection
* Group targets by correlation structure
* Ensemble of feature selectors
#### 2. Model Architecture
* Add transformer layers for sequence modeling
* Implement temporal convolutional networks
* Multi-scale feature extraction
#### 3. Continuous Training
* Experiment with different retrain frequencies (3, 5, 10 days)
* Implement incremental learning (update weights vs full retrain)
* Use ensemble of models from different time windows

#### 4. Meta-Learning
* Learn optimal ensemble weights over time
* Detect regime changes and adjust accordingly
* Implement adaptive feature importance

## Results Analysis
### Score Progression
* Round 1 (Public): 0.134
* Round 2 (Private): 0.110
* Round 3 (Private): 0.445
### Round 3 Analysis
The significant score improvement in Round 3 (0.110 → 0.445) demonstrates:
* The continuous training strategy successfully adapted to the final test period
* Models effectively learned from accumulating labeled data during inference
* The ensemble weighting scheme properly balanced recent vs historical performance
* Feature engineering captured evolving commodity market dynamics in the later period

## Alternative Approach: Target-Specific Feature Engineering

An alternative approach was developed that showed promising initial results but did not complete successfully in Round 3 due to infrastructure issues.

### Key Differences from Final Solution

#### 1. Target-Pair-Based Feature Engineering
Instead of generic rolling features, this approach used the competition-provided `target_pairs.csv` to create targeted features:
* **Relationship Mapping:** Each target was mapped to specific commodity pairs from the metadata
* **Custom Rolling Windows:** Window sizes matched the lag values specified for each target-pair relationship
* **Precision Features:** Rolling mean/std calculated only for columns directly related to each target

Example: For a target related to "JPX_Copper - US_Oil", features were computed only from those specific instruments rather than all 558 columns.

#### 2. Simpler Architecture, More Frequent Updates
* **Lightweight DNN:** 128 → 64 → 32 → 424 (much smaller than final ensemble)
* **Faster Training:** Enabled more frequent retraining (every 7 days vs every 7 days in final)
* **Time-Weighted Ensemble:** Recent models weighted exponentially higher (1.0 → 1.2 → 1.44...)

#### 3. Performance Comparison

| Metric | Target-Specific Approach | Final Ensemble Approach |
|--------|-------------------------|------------------------|
| Round 1 (Public) | 0.196 | 0.134 |
| Round 2 (Private) | 0.357 | 0.110 |
| Round 3 (Private) | **Failed** | 0.445 |
| Memory Usage | Lower | Higher |

### Why It Failed

The target-specific approach initially outperformed the baseline (0.196 vs 0.134 in Round 1), but it ultimately did not complete. The likely reasons for failure appear to be:

* **Infrastructure Fragility:** The lighter-weight pipeline may have been more vulnerable to instability in Kaggle's inference API.
* **Memory Constraints:** Rolling-window calculations required retaining growing historical data, which likely exceeded available memory limits over time.
* **Lack of Robust Checkpointing:** Unlike the final solution—which loaded pre-trained models—this method depended entirely on training during inference, offering no clear path for recovery if interrupted.

### Lessons Learned

This experience highlights the critical importance of **robustness over raw performance** in production ML systems:

1. **Defensive Programming:** Always implement fallback mechanisms for inference failures
2. **Memory Management:** Monitor and limit historical data accumulation
3. **Model Checkpointing:** Pre-train stable base models to avoid cold-start scenarios
4. **Testing at Scale:** Local validation doesn't capture all production edge cases

It's worth noting that **many participants experienced similar failures** in Round 3. While disappointing, this represents a valuable learning opportunity about the challenges of deploying adaptive models in constrained environments.

**Future iterations of this competition would benefit from enhanced pipeline stability to support online learning approaches** — the problem domain is fascinating, and online learning approaches show great promise if given a more robust infrastructure to operate within.

## Conclusion

This work demonstrates the effectiveness of adaptive learning strategies for non-stationary financial time series prediction, achieving 15th place in a competitive field. The core contribution lies in the implementation of an online training framework that continuously integrates newly labeled data during inference, enabling the model to track evolving market dynamics over a 9-month prediction horizon.

The experimental results validate several key hypotheses. First, periodic model retraining (every 7 days) meaningfully improved predictive performance, as evidenced by the substantial score improvement from Round 2 (0.110) to Round 3 (0.445). Second, ensemble diversity—combining attention-based, residual, and autoencoder architectures—proved more robust than reliance on a single model class. Third, comprehensive feature engineering spanning momentum, volatility, microstructure, and cross-market dynamics provided richer signal than price-based features alone.

The identified limitations, particularly the single-target feature selection strategy, suggest concrete directions for future work. Target-specific or group-based feature selection could further enhance performance, as could investigation of optimal retraining frequencies and incremental learning approaches. The comparative analysis of the lightweight target-specific approach, while ultimately unsuccessful due to infrastructure constraints, underscores the critical importance of robustness and fault tolerance in production machine learning systems—lessons that extend well beyond the competition setting.

More broadly, this solution reinforces the principle that in financial forecasting, model adaptability may matter as much as initial model sophistication. As market regimes shift, the ability to continuously learn from recent data provides a sustainable advantage over static approaches. The methods presented here—dynamic ensemble weighting, purged cross-validation, and hybrid ranking loss functions—offer a template for similar problems involving non-stationary time series with explicit ranking objectives.

The complete implementation is available in the referenced Kaggle notebook, providing a reproducible foundation for practitioners seeking to apply online learning techniques to commodity price forecasting and related financial prediction tasks.

## Key Takeaways
1. **Online Learning Works:** Retraining every 7 days allowed adaptation to market changes
2. **Ensemble Diversity:** Combining attention, residual, and autoencoder architectures captures different patterns
3. **Feature Engineering Matters:** 2,658 advanced financial features provide rich signal
4. **Robust Scaling:** RobustScaler handles outliers common in commodity data
