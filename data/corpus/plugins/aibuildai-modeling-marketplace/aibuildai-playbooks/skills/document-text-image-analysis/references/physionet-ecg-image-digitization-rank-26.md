# 26th Place Solution

Competition: physionet-ecg-image-digitization
Rank: #26
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/26th-place-solution

Thanks to PhysioNet and Kaggle for hosting such a fantastic competition. Huge thanks to [hengck23](https://www.kaggle.com/hengck23), [wasupandceacar](https://www.kaggle.com/wasupandceacar), and others for their selfless open-source contributions.  We only started putting serious effort into this during the final week, so we focused on optimizing the top-scoring open-source solutions.  We are thrilled to have achieved this rank and are eagerly looking forward to seeing the winning solutions!
- --
### **Overview**：
Our solution is primarily based on hengck23's open-source code.     We focused our optimization on Stage 1 and Stage 2 (we left Stage 0 as is, because we ran out of time).     The biggest improvement came from the two-stage training for Stage 2, which combines segmentation and signal extraction.  This allowed the raw signal data to be directly involved in the training process—an idea that had already been suggested in hengck23's discussion thread (though I didn't quite grasp it from the diagrams at first...).
- --
# **Stage 1**

### **Data Preparation:**
We utilized a modified version of ecg-image-kit based on several external datasets (e.g., PTB, PTB-XL, Georgia, CPSC-2018, etc.) to randomly generate approximately 10k raw ECG images.  Combined with the official dataset, we processed these through our Stage 0 and Stage 1 inference pipelines.  The resulting outputs served as pre-annotations, which were then manually refined using our custom-built annotation software.

### **Model:**   
Standard UNet. Backbone: EfficientNet-V2-S, ResNet34. (AdamW (lr: 1e-4, wd: 2e-5).)

### **Augmentation:**   
Various Albumentations techniques including Noise, Blur, ToGray, and CLAHE.

### **Simple Weight Averaging (SWA):**  
During training, we maintained the top three models based on validation loss and scores, then performed offline weight averaging.

### **Inference Post-processing:**  
1.  Optimized output_to_predict: When assigning values within the 44/57 range for each point, the logic was enhanced to consider not just the current line position but also a voting value from a surrounding 3x3 window.  
2.  Grid Completion: Missing grid points from the segmentation were interpolated and filled based on directional vectors.

- --
# **Stage 2**

### **Data Preparation:**  
Generated 20k ECG images (with lead masks) using a modified ecg-image-kit, combined with the official dataset.       All training data were pre-processed through our optimized Stage 0 + Stage 1 inference pipeline to obtain rectified, undistorted images.

### **Cross-Validation:** 
Performed 5-fold CV on the official dataset using patient IDs as groups.       Synthetic data were included only in the training set of each fold and excluded from the validation set.

### **Two-Stage Training:** 
* Phase 1: Fine-tuned wasupandceacar’s best open-source segmentation model using synthetic data and masks.     
* Phase 2: Used Phase 1 weights as the pre-trained backbone.       We implemented a Differentiable Soft-Argmax layer after the segmentation head to extract signals from the probability maps (logits).       Using fixed cropping parameters and zero-baseline coordinates, the predicted pixel-level signals were converted into physical signals to calculate L1 + L2 loss against the ground truth (GT) signals.

### **Model:**  
wasupandceacar’s ResNet34-UNet. imagesize:1696 x 4352

### **Segmentation Phase Augmentation:**   
Various Albumentations techniques including Noise, Blur, ToGray, and CLAHE.

### **Signal Phase Augmentation:**  
Combined segmentation-phase augmentations with custom-designed effects: simulated black/yellow stains, lens distortion, paper creases, partial shadows, and scanner noise.

### **Training Configuration:**  
4x RTX 4090 (DDP), EMA (0.992), AdamW, Warmup + Cosine Scheduler, 2e-5 learning rate.       Data sampling for each epoch: 80% official data and 20% synthetic data.

### **Inference Configuration:**
1. Ensemble Strategy: Integrated three best models (varying in augmentations, learning rates, and folds). We performed simple averaging at the probability level (segmentation logits) before passing it through the same differentiable post-processing used in training.       
2. Legacy Logic: Retained the open-source logic for ECG type classification and its corresponding pre-processing.       Despite imperfect classification accuracy, it provided a 0.2 LB boost.
