# 16th Place Solution - Probablistic Regression

Competition: ariel-data-challenge-2025
Rank: #16
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/16th-place-solution-probablistic-regression

Thanks to the organizers and my fellow Kagglers for a really interesting competition. My goal from the start was to improve on my Ariel 2024 neural network solution (which earned bronze). I didn’t go down the physics-based modeling route (where many others excelled) — instead I focused on refining a probabilistic neural network approach.

**Model**

The core model is a two-input Conv1D neural network built in TensorFlow/Keras. One branch processes the FGS1 signal, the other processes the AIRS signal:

Conv1D (ReLU)
MaxPooling
Flatten
The two branches are concatenated into a shared representation, combining complementary information. The key difference from a standard regression model is the probabilistic output layer: instead of predicting a single value per wavelength, the model outputs both a mean and an uncertainty across all 283 wavelengths.

**Mixture Distribution**

The output is defined as a Laplace-based mixture distribution with 15 components × 283 wavelengths. Parameters from the dense layer are split into:

mixture logits (weights)
per-wavelength means (locs)
per-wavelength scales (uncertainties)

Stability tricks:

Scales clipped using variance-based bounds
Softplus applied to enforce positivity

TensorFlow Probability’s DistributionLambda then builds the full mixture model. This setup lets the network learn both expected values and uncertainties, which was central to the challenge.

**Data Preprocessing**

The preprocessing pipeline was fairly straightforward:

Load raw data
Winsorize
Segment signals into transient shapes

For each segment, the mean is compared with the “unobscured” parts (outside the segment). A surprise boost came when I added the inverse of the segmented transients (domes) alongside the original dips — the model benefited from explicitly seeing both contrasting patterns.

Further gains came from combining segments of different lengths, so the final model used multi-segment data (coarse + fine).

**Why This Helped**

Segments give compact but informative context for transits.
Adding domes (inverse of dips) enriched the feature space and helped the network learn contrasting shapes.
Multi-segments allowed the model to generalize better across fine and coarse structures.

**Training Loss**

The loss was Negative Log Likelihood (NLL) with wavelength-based weighting:

Weights were split between FGS1 and AIRS
f_share = 0.1702 gave FGS1 its relative importance
Combined into a single weight array so both channels influenced training consistently

**Other Tweaks

Adding a small VSN gate (Conv1D + sigmoid) gave a small but consistent lift.
Ensembling different mixtures (Laplace, Normal, Logistic) across folds and seeds helped stabilize scores.

**Results & Takeaways**

The mixture distribution was the most impactful part, though it’s still not perfect — per-wavelength variation can sometimes collapse toward the mean.
With 10-fold CV, validation–public gaps were usually around 0.018–0.02 (val higher) and validation–private gaps were <0.01, so generalization was stable.
Full training time was about 1 hour per model, which made experimentation manageable.

Overall, it was a satisfying to develop an approach which gave competitive scores, and most importantly, can be adapted to other datasets where uncertainty estimation is important. Big thanks again to the organizers and the community — for this great learning experience.
