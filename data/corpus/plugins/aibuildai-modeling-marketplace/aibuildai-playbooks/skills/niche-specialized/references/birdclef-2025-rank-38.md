# Place 38 | 0.902 AUC score

Competition: birdclef-2025
Rank: #38
Source: https://www.kaggle.com/c/birdclef-2025/discussion/583447

This is my submisstion code: https://www.kaggle.com/code/maxmelichov/bird25-0-902-auc



---

### **Data and Challenge**

The goal of the BirdCLEF+ 2025 competition was to identify the species of various taxonomic groups (birds, amphibians, mammals, insects) present in soundscape recordings from the Middle Magdalena Valley of Colombia and the El Silencio Natural Reserve.

What’s in the Test Set?
The test data comprises long, real-world soundscape recordings captured in Colombia. Each soundscape typically contains calls from multiple species across different taxonomic groups—the exact challenge we want our model to solve.

What Are We Given to Train On?
For training, we’re provided with:

Short audio clips—each featuring the call of a single species.

These recordings come from xeno-canto.org, iNaturalist, and the Colombian Sound Archive (CSA) of the Humboldt Institute for Biological Resources Research in Colombia.

Each clip is carefully labeled with its species ID and taxonomy, but rarely contains more than one species per recording.

The Real Challenge
This sets up a fundamental mismatch:

Training: Single-species, high-quality, labeled clips.

Testing: Multi-species, real-world, often noisy soundscapes.

The competition, therefore, isn’t just about building a good classifier. It’s about finding robust methods to bridge the gap between clean, isolated calls and complex, overlapping natural soundscapes—a leap that pushed every part of my pipeline, from preprocessing to model selection and pseudo-labeling.

---

### **Step 1: Preprocessing**

* Used `snakers4/silero-vad` for Voice Activity Detection (VAD) to remove human voices from the training data.

---

### **Model 1: EfficientNet-B0 CNN with Strong Augmentation**

* **Spectrogram settings:**
  `N_FFT=1024, HOP_LENGTH=64, N_MELS=148, FMIN=20, FMAX=16000`
* **Augmentations:** Time masking, frequency masking, random brightness/contrast, high-frequency boost, dynamic range compression/expansion, frequency shift, subtle broadband noise, mid-frequency boost, slight blur (time direction), and mix original with blurred version.
* **Training:**

  * EfficientNet-B0 backbone
  * Mixup (α=0.15


Absolutely! Here’s your revised full LinkedIn post, with all your corrections reflected, and a clean structure. This version highlights your journey, technical pipeline, and the ensemble details exactly as you described:

---

**Long story short: I finished 41st out of 2,162 teams in the BirdCLEF+ 2025 Kaggle competition.**

Big thanks to **Ariel** for the support and advice along the way!

Here’s an overview of what actually worked (there were plenty of failed experiments behind the scenes):

---

### **Data and Challenge**

* **train\_audio/**: Short recordings, each with a single bird, amphibian, mammal, or insect species.
* **train\_soundscapes/**: Unlabeled audio from the same recording locations as the test data, but usually containing multiple overlapping species.
* **Key difference:** Test data contains multiple species per recording, while most training data has only one.

---

### **Step 1: Preprocessing**

* Used `snakers4/silero-vad` for Voice Activity Detection (VAD) to remove human voices from the training data.

---

### **Model 1: EfficientNet-B0 CNN with Heavy Augmentation**

* **Spectrogram settings:**
  `N_FFT=1024, HOP_LENGTH=64, N_MELS=148, FMIN=20, FMAX=16000`
* **Augmentations:**

  * Time masking, frequency masking
  * Random brightness/contrast
  * High-frequency boost
  * Dynamic range compression/expansion
  * Frequency shift
  * Subtle background noise (broadband and mid frequencies)
  * Slight blur in the time direction, mixing original and blurred
* **Training setup:**

  * EfficientNet-B0 backbone
  * Mixup (α = 0.15)
  * BCE loss (primary label = 1, secondary = 0.75), multilabel/multiclass
  * Trained on the middle 5 seconds of each clip (tried energy-based and random segments but performed worse)
* **Results:**

  * 0.817 AUC (Area Under the Curve; how well the model ranks true labels higher than false ones)
* **Pseudo-labeling:**

  * Inferred on all `train_soundscapes` (split into 5s chunks, average predictions, use as new labels)
  * Boosted performance to 0.835 AUC
  * Switching backbone to EfficientNetV2-S reached 0.843, but best pseudo-labeling results were still with EfficientNet-B0.

---

### **Model 2: Enhanced Pooling and Mixup**

* **Architecture:**

  * EfficientNet-B0 backbone
  * Added GeM pooling layers on features from layers 3 and 4
  * Mixup α = 0.5
* **Alternative Experiments:**

  * Tried adaptive pooling, denoiser augmentation, and more advanced approaches (NatureLM-audio segment extraction, 3-channel Mels, cutmix, teacher-student), but no additional improvements.
* **Best Model 2 Result:** 0.855 AUC

---

### **Ensembling**

Combined my three best CNN models:

1. **GeM pooling (no denoiser)**
2. **GeM pooling with denoiser**
3. **Adaptive pooling (no denoiser)**

* **Ensemble AUC:** 0.868

After this, most further tweaks (bigger backbones, more mixup strategies, new pseudo-labeling techniques, heavier augmentation) failed to improve validation. I hit a plateau and considered giving up.

---

### **Exploring SED (Sound Event Detection): My First Steps**

In the last three weeks of this two-month competition, SED-based approaches started popping up on the Kaggle forums. This was my first time trying SED models. With the steep learning curve and limited time, I couldn’t get my own SED models above 0.841 AUC.

To keep moving forward, I leveraged top SED model predictions shared by others in the forums:

* Used single SED model results (NFNet backbone, 0.86 AUC) and a triple SED ensemble (0.85 AUC, with Power Adjustment for Low-Rank) from public Kaggle kernels and posts.
* Combined these with my three CNN models using a Quantile-Mix ensemble (α = 0.5), which pushed the score up to 0.893 AUC.

---

### **Final Push: Pretraining on BirdCLEF 2021–2024**

* Pretrained all three CNN models on BirdCLEF data from 2021–2024, then fine-tuned on 2025 data.
* **Result:** Single-model AUC improved from 0.855 to 0.868.
* Final ensemble (3 CNNs + SEDs) scored **0.894 AUC** on the public leaderboard (45th place).
* On the final private leaderboard: **0.902 AUC**, landing at **41st place**.
* For reference, 1st place finished at 0.930 AUC—just a 3% gap!

---

**Key takeaways:**

* Strong preprocessing (VAD, MelSpectrogram, heavy augmentation) is crucial.
* Pseudo-labeling unlabeled soundscapes provides a significant boost if segment selection is careful.
* Pooling strategies and label weight tuning can give CNNs a few more points.
* SED is a powerful approach for multi-species data, but there’s a learning curve if you’re new to it (like me!).
* Pretraining on historical BirdCLEF data really helps.
* The Kaggle community is an incredible resource—leveraging public notebooks and shared solutions is invaluable, especially when time is short.

---

I learned a ton during this competition—about audio modeling, new architectures, and pushing through plateaus. Huge thanks again to Ariel, everyone in the forums, and all those who shared code and kernels. I’m already looking forward to next year and hoping to break into the top 10!

---

If you want to chat about the technical details, see code, or ask anything about the pipeline, feel free to reach out!
