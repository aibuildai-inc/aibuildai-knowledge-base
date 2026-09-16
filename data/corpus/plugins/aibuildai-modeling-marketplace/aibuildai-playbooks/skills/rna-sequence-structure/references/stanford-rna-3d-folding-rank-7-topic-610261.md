# 7th Place Solution: Ensemble of Two Protenix Models

Competition: stanford-rna-3d-folding
Rank: #7
Source: https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/7th-place-solution-ensemble-of-two-protenix-model

# 7th Place Solution: Ensemble of Fine-tuned Protenix

First of all, I would like to express my sincere gratitude to the competition host team and the Kaggle team for organizing this interesting competition. I am also deeply thankful to all the participants who continuously shared valuable insights and information. The three months spent tackling machine learning for biomolecular 3D structure prediction have been truly enjoyable and rewarding.

## Summary

My solution consists of an ensemble of two types of fine-tuned [Protenix](https://github.com/bytedance/Protenix) (AlphaFold3 clone), **with and without MSA**. In the early phase of the competition I tried [RibonanzaNet2_DDPM](https://www.kaggle.com/code/shujun717/ribonanzanet2-ddpm-training), [DRfold2](https://github.com/leeyang/DRfold2), and [NuFold](https://github.com/kiharalab/NuFold), but Protenix gave the best results in my experiments, so I focused exclusively on Protenix in the latter half. Protenix finetuning was conducted with the code provided by @lihaoweicvch and [d4t4 team](https://www.kaggle.com/competitions/stanford-rna-3d-folding/writeups/10th-place-solution-d4t4-team), with minor modifications.
- [GitHub](https://github.com/lhwcv/Protenix-RNA-Kaggle) (credit: @lihaoweicvch)
- [Discussion post]( https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/573495)

I tried two hill-climbing strategies and selected both two models as final submissions:
1. **Public LB score focused**: Mainly trained on CASP16 VFold predicted structures
2. **Local score focused**: Mainly trained on host-provided PDB data (Kaggle v1/v2 data)

TM-Score results are as follows:

#### 1. Public LB Focused Models

|                      Model                      | Public LB | Private LB |
| :---------------------------------------------: | --------: | ---------: |
|                     wo-MSA                      |   0.43529 |    0.39196 |
|                    with-MSA                     |   0.43205 |    0.41014 |
| Two models ensemble<br>(**Final Submission 1**) |   0.46901 |    0.43514 |

#### 2. Local Score Focused Models

|                      Model                      | Local Validation | Public LB |             Private LB             |
| :---------------------------------------------: | ---------------: | --------: | :--------------------------------: |
|                     wo-MSA                      |           0.5982 |   0.43135 |              0.47654               |
|                    with-MSA                     |           0.5943 |   0.42501 |              0.46793               |
| Two models ensemble<br>(**Final Submission 2**) |           0.6092 |   0.43991 | **0.48511**<br>(private 7th place) |

---

## Dataset
Kaggle v1/v2 data and CASP16 VFold prediction were used for model training.

#### Data Filtering

- **Quality Criteria**<br>
Only data that met the following criteria were used:
    - Consecutive nucleotide C1'-C1' distance ≤ 15 Å
    - Resolution ≤ 15 Å
    - Contains all four nucleotides (A, U, G, C)

- **Deduplication**<br>
For sequences with multiple structure data, only one structure was selected for training according to the following rules:
    - Keep the structure with the lowest resolution (Å)
    - If resolution is NaN, select one structure according to the following priority of experimental methods:
        1. X-ray
        2. Cryo-EM
        3. NMR

#### Train/Validation Split

##### 1-a: PublicLB focused / wo-MSA model
Train and validation data is same, casp16 vfold predicted data.

##### 1-b: PublicLB focused / with-MSA model
- Dataset: Kaggle v1 (with MSA)
- Validation: the most recent 37 sequences
- Training: 733 sequences / Validation: 37 sequences

##### 2-a: local LB focused / wo-MSA model
- Dataset: Kaggle v1 (with MSA)
- Validation: the most recent 40 sequences
- Training: 2407 sequences / Validation: 40 sequences<br>
Splitting the dataset solely based on a temporal cutoff date resulted in limited diversity within the validation set (e.g., the sequence length are very similar). To improve diversity, some sequences were manually reallocated between the training and validation sets (i.e., prioritizing the assignment of diverse types of sequences to the validation set over strict adherence to the temporal cutoff rule).

##### 2-b: local LB focused / with-MSA model
- Only the MSA data provided in the above training/validation set was used.
- Training: 1398 sequences / Validation: 36 sequences

---

## Training

#### Training Configuration
- **Learning rate**: 1e-4
- **No LR scheduler**
- **warmup steps**: 50
- **EMA (Exponential Moving Average) decay**: 0.995
- **Diffusion sampling**: 20
- **Trunk recycling**: 4
- **Sequence length cutoff**: 416 nucleotides
- **GPU**: RTX5090 or RTX4090

#### 1. Public LB Focused Models
- **wo-MSA model**
    - Fine-tuned with CASP16 VFold structures (44 sequences) without MSA. This **CASP16 overfitting checkpoint** was used not only for submission but also for further fine-tuning of other models.
    - num_steps: 2,000
- **with-MSA model**
    - The above **CASP16 overfitting checkpoint** was loaded and the model was trained with Kaggle v1 data having MSA.
    - num_steps: 7,300

#### 2. Local Score Focused Models
- **wo-MSA model**
    - Fine-tuned with Kaggle v1/v2 data without MSA.
    - num_steps: 48,200
- **with-MSA model**
    - The **CASP16 overfitting checkpoint** was loaded and the model was fine-tuned with Kaggle v1/v2 data having MSA.
    - num_steps: 29,320

---

## Inference

#### Inference Configuration
For final submissions:
- **Max sequence length**: 850
    - Long sequences were truncated to 850 residues to fit within VRAM constraints in the Kaggle environment.
- **Diffusion sampling**: 200
- **Trunk recycling**: 10
- **Number of predictions**: Five structures were predicted by each of the two Protenix models (with/without MSA), for a total of ten structures per target.

### Post-Processing (Actually not worked on private test set)

Some techniques were employed for truncated position padding and structure selection. They slightly improved both the local score and public LB score, suggesting these tricks were working properly, at least during the competition period. But in reality, they did not produce meaningful gains on the private LB, or worsened the score in some cases. Having believed they were reasonable approaches, the result was disappointing, but I'm sharing them here anyway.

- **Padding Method**<br>
The coordinates of the truncated position must be filled with some value. Although some public notebooks seemed to pad with (0, 0, 0), padding with the same coordinates as the truncated end resulted in a slight boost in public LB within the scope of my experiments (see figure below ). However, the private LB did not show a meaningful improvement over zero-padding. If the private set did not include any sequences exceeding 850 nucleotides, truncation would not have been triggered, and thus it may not have been an essential factor.<br>[Related discussion post](https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/578396#3200403)




- **K-Medoids Clustering for Structure Selection**<br>
Five structures needed to be selected out of ten (two Protenix output five structures each). Although selection based on confidence (pLDDT) is a common approach, I was concerned that it tends to favor similar structures, limiting the diversity. Given that the competition metric adopts the highest TM-score among five submissions, selecting structurally diverse candidates seemed to be more effective. To achieve this, k-medoids clustering (k = 5) was performed on the ten predicted structures using `1 - TM-score` as a distance metric, and the five medoids were selected for submission. This approach led to improvements in public LB scores for both the LB-focused model and the local score-focused model, compared to pLDDT-based selection. However, on the private LB there was little difference overall, and in some cases pLDDT-based selection actually performed better.

#### Comparison of Post-Processing Methods

**Submission 1 (Public LB Focused Model) Variants**

| Model                  | Padding  | Structure Selection                      | Public LB | Private LB |
| ---------------------- | -------- | ---------------------------------------- | --------: | ---------: |
| **Final Submission 1** | end-same | k-medoids (k=5)                          |   0.46901 |    0.43514 |
| Variant 1-1            | zero     | k-medoids (k=5)                          |   0.46576 |    0.43217 |
| Variant 1-2            | end-same | pLDDT top 2+3<br>(no-MSA 2 + with-MSA 3) |   0.44640 |    0.43643 |
| Variant 1-3            | zero     | pLDDT top 2+3                            |   0.44960 |    0.43443 |

<br>

**Submission 2 (Local Score Focused Model) Variants**

| Model                  | Padding  | Structure Selection | Public LB | Private LB |
| ---------------------- | -------- | ------------------- | --------: | ---------: |
| **Final Submission 2** | end-same | k-medoids (k=5)     |   0.43991 |    0.48511 |
| Variant 2-1            | zero     | k-medoids (k=5)     |   0.43906 |    0.48459 |
| Variant 2-2            | end-same | pLDDT top 2+3       |   0.43826 |    0.48831 |
| Variant 2-3            | zero     | pLDDT top 2+3       |   0.43568 |    0.48861 |

*The 'variants' load the same weights as **Final Submission 1 or 2**. Only the post-processing methods in inference were changed.*
