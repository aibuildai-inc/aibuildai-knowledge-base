# 4th Place Solution — LightGBM with SMILES-Derived Features

Competition: neurips-open-polymer-prediction-2025
Rank: #4
Source: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/4th-place-solution-lightgbm-with-smiles-derived-fe

# Feature Engineering

The NeurIPS 2025 Open Polymer Prediction challenge aimed to predict key polymer properties from their molecular structures. These include density, thermal conductivity (Tc), and glass transition temperature (Tg), as well as two measures of molecular size and packing efficiency: the radius of gyration (Rg) and the fractional free volume (FFV).
My solution was based on a LightGBM baseline model, to which I added a set of new features derived from the chemical structure of the polymers.  
The main goal was to capture positional relationships between the radical markers `*` in the SMILES, since topology and local neighborhood strongly influence properties such as **Tc** and **Rg**.  
In addition to standard 2D descriptors, I incorporated 3D geometric descriptors calculated from RDKit-generated conformations.  
These attributes capture the global shape of the molecule and the spatial distribution of atoms, providing a complementary perspective beyond topology.  
In order to provide a more comprehensive molecular representation,I incorporated features related to the presence and characteristics of aromatic and aliphatic rings in the structures. 
Working with glass transition temperature (Tg) was one of the main challenges of the competition. Tg values showed high variability and were strongly influenced by subtle structural changes. To improve the learning process, I used a quantile regression objective with α = 0.85 for Tg (and also for Rg), focusing the model on learning from the higher values of the distribution. For the other targets (Tc, Density, and FFV), I kept a standard quantile of 0.5 (median).  

---

## Data Augmentation via SMILES Substitutions

During the error analysis, I identified molecules with the worst predictions for Rg and Tc.  
From these difficult cases, I designed a process of controlled data augmentation,  
generating synthetic SMILES with small structural variations.  
This expanded the training set with structurally plausible examples,  
helping the model learn local substitution trends and improving its  
generalization on molecules with uncommon topologies.  

---

- **Functional group substitutions**  
  I introduced simple substituents such as **F, Cl, CN, and CF₃** (which tend to increase rigidity and polarity),  
  as well as more flexible groups such as **CH₃, OCH₃, and O** (which generally reduce Rg).  
  → Each substituent was assigned a **ΔRg** (positive or negative), adjusted with  
  small random noise to reduce the risk of overfitting.  


## Datasets

For training, I combined multiple SMILES datasets to expand structural diversity and improve coverage for each target property.

### Number of SMILES used for training
- **Tg**: 8,244  
- **FFV**: 7,030  
- **Tc**: 866  
- **Density**: 1,247  
- **Rg**: 614  

### Dataset sources
- [SMILES Extra Data (by dmitryuarov)](https://www.kaggle.com/datasets/dmitryuarov/smiles-extra-data)  
  Provided additional polymer structures with multiple target annotations.  

- [Tc SMILES (by minatoyukinaxlisa)](https://www.kaggle.com/datasets/minatoyukinaxlisa/tc-smiles)  
  Focused on **Tc**, useful for strengthening a target with fewer samples.  

- [Tg of Polymer Dataset (by akihiroorita)](https://www.kaggle.com/datasets/akihiroorita/tg-of-polymer-dataset)  
  Specialized in **Tg**, expanding the range of glass transition temperatures included in training.  

### Code Reference

This solution builds upon the public notebook [NeurIPS Baseline + External Data](https://www.kaggle.com/code/dmitryuarov/neurips-baseline-external-data) by **Dmitry Uarov**.  
I used it as a foundation and extended it in several ways to improve predictive performance:

- Added new chemically derived features to capture positional relationships of radicals (`*`) in SMILES.  
- Integrated 3D geometric descriptors calculated from RDKit-generated conformations.  
- Included descriptors related to aromatic and aliphatic rings to account for rigidity, conjugation, and flexibility.  
- Applied controlled data augmentation via SMILES transformations targeting molecules with poor predictions in **Tc** and **Rg**.
