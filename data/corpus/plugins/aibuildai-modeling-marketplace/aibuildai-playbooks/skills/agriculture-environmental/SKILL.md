---
description: >-
  ML playbook for agriculture environmental competitions. Use when tackling a Kaggle-style competition involving agriculture environmental. Teaches how to reason about invest heavily in domain-aware data cleaning before modeling, design spatial-temporal cv that matches test distribution, not train convenience, exploit spatial-environmental covariates even when images are primary, use multi-head architectures when targets share physical relationships. Analysis of 30+ top-solution writeups across 7 competitions: soil property prediction from spectroscopy, building energy forecasting, plant trait estimation from images, pasture biomass from overhead photos, forest cover classification, and species geolocation.
---

# Agriculture & Environmental Prediction Playbook

Agriculture and environmental prediction tasks involve forecasting physical, biological, or chemical properties from multi-modal observations (spectral data, images, weather, spatial covariates). These tasks are distinctive for their real-world measurement noise, domain physics constraints, strong spatial-temporal structure, and often small training sets relative to feature complexity. The core challenge is extracting robust signal from noisy, heterogeneous data while respecting domain knowledge and spatial/temporal dependencies.

**Source material:** Analysis of 30+ top-solution writeups across 7 competitions: soil property prediction from spectroscopy, building energy forecasting, plant trait estimation from images, pasture biomass from overhead photos, forest cover classification, and species geolocation.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Invest Heavily in Domain-Aware Data Cleaning Before Modeling | 20/30+ solutions explicitly emphasized cleaning as critical, often the single largest score improvement | principles/01.md |
| 2 | Design Spatial-Temporal CV That Matches Test Distribution, Not Train Convenience | 25/30 solutions used stratified group k-fold by spatial or temporal variables; multiple solutions trained 2+ different CV schemes for ensemble diversity | principles/02.md |
| 3 | Exploit Spatial-Environmental Covariates Even When Images Are Primary | 15/20 image-based solutions used ancillary geodata (climate, soil, NDVI, altitude); winner of PlantTraits used structured self-attention over metadata | principles/03.md |
| 4 | Use Multi-Head Architectures When Targets Share Physical Relationships | 12/20 solutions used separate regression heads for related outputs; top biomass solutions added classification heads alongside regression | principles/04.md |
| 5 | Train Site/Group-Specific Models When Data Exhibits Strong Clustering | 15/30 solutions trained separate models per site, building, or meter; ASHRAE top-3 all used site+meter-level models | principles/05.md |
| 6 | Engineer Log/Normalized Targets and Reverse Transform, Handling Zeros Carefully | 20/25 regression solutions used log1p transform; multiple solutions tried per-area or per-volume normalization | principles/06.md |
| 7 | Leverage Domain-Specific Pretrained Backbones Over Generic ImageNet | 18/20 image-based solutions used DINOv2/DINOv3; PlantTraits winner used flora-specific DINOv2 pretrained on Pl@ntNet | principles/07.md |
| 8 | Apply Test-Time Augmentation and Pseudo-Labeling for Small Datasets | 10/20 solutions used TTA; 5/20 used pseudo-labeling or test-time training; biomass winner gained 0.02 R2 from pseudo-labels | principles/08.md |
| 9 | Ensemble Across Diverse Splits, Architectures, and Granularities | 25/30 solutions ensembled multiple models; top solutions used 5-20 models with different CV splits, backbones, or site-level vs. global models | principles/09.md |
| 10 | Apply Group-Specific Post-Processing When Test Has Known Structure | 8/20 solutions used site/state-specific scaling; biomass 2nd-place gained 0.015 R2 from state-based scaling | principles/10.md |
| 11 | Validate Augmentation Choices Against Physical Realism, Not Just CV Score | 15/20 image solutions used heavy augmentation; multiple noted that certain augs (vertical flip for plant photos, extreme color jitter) hurt despite improving CV | principles/11.md |
| 12 | Encode Cyclic Time Features as Sine/Cosine Pairs, Not Raw Integers | 5/10 time-series solutions explicitly used cyclic encoding; ASHRAE winner listed it as a key feature | principles/12.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles are interdependent: strong cleaning enables reliable CV, which guides augmentation and ensemble choices; multi-modal fusion only helps if you've cleaned the data first. Start with cleaning and CV design (principles 1-2), then iterate on modeling (3-8), and finish with ensemble and post-processing (9-10). In small-data environmental tasks, robustness through diversity (ensembles, TTA, CV) often outweighs squeezing the last 0.5% from a single model.
