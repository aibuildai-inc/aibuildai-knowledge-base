# 43rd Place Solution - Two-stage nnU-Net for Hole Filling

Competition: vesuvius-challenge-surface-detection
Rank: #43
Source: https://www.kaggle.com/c/vesuvius-challenge-surface-detection/writeups/43rd-place-solution-two-stage-nnu-net-for-hole-f

I would like to thank Kaggle and the sponsors for hosting such a fantastic competition. The contest was incredibly fierce, and while our final ranking wasn't quite as ideal as we had hoped, the entire journey has been an invaluable learning experience for me. Now, please allow me to walk you through our methodology:

---

# **Stage 1: Model Ensemble**
### **Model Architecture**
**nnU-Net** is an exceptionally powerful and robust framework that I frequently utilize in both my professional work and various competitions, including this one. For our Stage 1 pipeline, we relied on an ensemble of five models to ensure a stable and high-quality initial segmentation. 
The ensemble consisted of the following components:

* **2x SegMambaV2 (3d_fullres)**: We trained two versions of this architecture—one using the original official dataset and another incorporating external data to improve the model's generalization capabilities.
* **1x nnU-Net Residual Encoder (ResEncL_fullres)**
* **1x nnU-Net Base Model (3d_lowres)**: This was included to provide a larger receptive field.
* **1x SMP-3D MIT-B5**: A custom 3D Unet model contributed by my teammate **lhwcv**, which provided critical architectural diversity to our final Stage 1 predictions.


### **Loss Function and Training Strategy**
We implemented a **tri-phase loss training schedule** to progressively refine the model's performance:
* **Phase 1: Foundation Training**
    * We began with the standard **nnU-Net CE + Dice loss**.
    * To better align with the competition's evaluation objectives, we integrated a **custom Surface Dice loss** (inspired by the 1st place solution from the SenNet competition) and **BoundaryDOULoss**. 
    * Both of these losses were heavily modified to adapt to nnU-Net's $0-1$ dual-channel output while ensuring that `ignore` regions were correctly handled.
* **Phase 2: Precision Enhancement**
    * Once training reached the midway point, we introduced **Tversky Loss** into the training objective.
    * The primary goal here was to specifically suppress **False Positives (FP)**, improving the model's reliability in distinguishing subtle signals from noise.
* **Phase 3: Topological Optimization**
    * In the final stage, we combined our custom loss functions with **Topoloss** (from the *SHAPR_torch* repository).
    * This enabled the joint optimization of metrics highly correlated with the competition's scoring system. By focusing on topological features such as the **number of connected components** and the presence of **holes**, we were able to significantly enhance the structural integrity of our segmentations.

### **Augmentation**
* Random **flips across the X, Y, and Z axes** and **3D rotations**.
* **Gaussian Noise** and **Gaussian Blur (Filtering)** 
* **Low-resolution Simulation** and **3D Cutout**

---

# **Stage 2**
We used the best Stage 1 ensemble to generate out-of-fold (OOF) predictions for the full training set. These results were combined with the original images as a multi-channel input to train our Stage 2 models. The second stage integrated an ensemble of two models: **nnU-Net_ResEncL** and **SegMambaV2**.

In Stage 2 training, we introduced custom augmentations specifically targeting the Stage 1 prediction mask (Channel 1) to simulate adhesions, holes, and fractures that typically impact the final score. These included:
1. **Opening and Closing operations (0.2 probability)**: Used to simulate imprecise surfaces and optimize the Surface Dice metric.
2. **Simulated Holes (0.5 probability)**: Applied only to non-ignore regions of the prediction mask. We generated 0 to $n$ simulated holes for each of the 1 to $n$ scroll instances in a case.
3. **Sawtooth Fractures and Random Adhesions (0.5 probability)**: Targeted non-ignore regions, where random adhesions were simulated between the two nearest instances.

# **Post-processing**
We utilized only open-source post-processing methods. After seeing the top solutions, the gap between our post-processing strategies and theirs became clear. Although I attempted some of those advanced techniques, none were successful. This remains perhaps the only regret for me in this competition.

# **Comparison of Inference Results**
The following shows the comparison between Stage 1 and Stage 2 inference results for cases not included in the training set:

