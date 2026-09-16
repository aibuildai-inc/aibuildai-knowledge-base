# 3rd place solution

Competition: novozymes-enzyme-stability-prediction
Rank: #3
Source: https://www.kaggle.com/c/novozymes-enzyme-stability-prediction/discussion/375964

Thanks a lot to the hosts and Kaggle for hosting this interesting competition. Congratulations to the other competitors for the great solutions and results.Special thanks to the competitors who selflessly shared their knowledge during this competition, @cdeotte , @tilii7 , @ropeonmars , @roberthatch , @dschettler8845 , @jinyuansun , @kvigly55 , @vslaykovsky , @oxzplvifi , @shlomoron ..., If I've missed someone, please let me know.


# Training Set
I used the dataset [D1](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/359284) ,[D2](https://www.kaggle.com/datasets/shlomoron/train-wildtypes-af) and [D3](https://www.kaggle.com/code/vslaykovsky/14656-unique-mutations-voxel-features-pdbs).
# Feature Engineering
I selected the 21 amino acids closest to the mutant amino acids to construct the features.
- The relative distances of the 21 amino acid mutation sites were calculated.
$$D_{ci}=\frac{r_{c}-r_{i}}{max_{i∈K}{(r_{c}-r_{i})}}$$,$$r_{ci}=\sqrt{{(x_{c}-x_{i})}^{2}+{(y_{c}-y_{i})}^{2}+{(z_{c}-z_{i})}^{2}}$$
- The average relative distance between the 21 amino acids and the mutation site was calculated.
- The angles of the 21 amino acids to the mutation site were calculated.
$$\alpha_{1}=arccos\frac{(z_{c}-z_{i})}{r}$$
$$\alpha_{2}=arccos\frac{(x_{c}-x_{i})}{r}$$
$$\alpha_{3}=arccos\frac{(y_{c}-y_{i})}{r}$$
$$\beta_{4}=arctan\frac{(y_{c}-y_{i})}{(x_{c}-x_{i})}$$
$$\beta_{5}=arctan\frac{(z_{c}-z_{i})}{(y_{c}-y_{i})}$$
$$\beta_{6}=arctan\frac{(x_{c}-x_{i})}{(z_{c}-z_{i})}$$
- B_factor, sasa, blosum, demask...
- Use the adjusted blosum62 to multiply sasa, b_factor...
```python
def sigmoid_w_adjustment(x, adjustment_factor=3.0):
    return 1-(1/(1+np.exp(-x/adjustment_factor)))
```
- Features of the whole protein sequence, mutant amino acids and 21 amino acids extracted by esm.
- There are a few other interesting features [here](https://www.kaggle.com/code/dschettler8845/novo-esp-residue-depth-and-more-w-biopython).

# Model
## XGB_v1
This model gets public score **0.52132** and private score: **0.46642**.
We used [D1](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/359284) to train my first Xgboost model. This dataset is wildtypes and mutants selected from kaggle's training set with 4000+ rows of train data. First of all, I did some simple processing on the dataset, removing duplicate rows and deletion mutation, deleting rows that did not match pdb, and retaining groups with mutation samples > 20. Finally, we get a training set with 1800+ rows.
We used the features mentioned above and took Tm rank as target to train the model,so we have 298 features for our model.

## XGB_v2
This model gets public score **0.40512** and private score: **0.30495**. 
We used [D2](https://www.kaggle.com/datasets/shlomoron/train-wildtypes-af) to train my second Xgboost model. This dataset is wildtypes and mutants selected from kaggle's training set. But this data set is cleaned up in more detail.The cleaned dataset contains only 986 mutations from 30 groups, all groups are same-source, same-pH.

## XGB_v3

This model gets public score **0.40435** and private score: **0.43294**. 
We used [D3](https://www.kaggle.com/code/vslaykovsky/14656-unique-mutations-voxel-features-pdbs) to train my third Xgboost model. Since the strong correlation between dTm and ddg(ddG is a measure of the change in energy between the folded and unfolded states (𝚫Gfolding) and the change in 𝚫Gfolding when a point mutation is present.), we can also use ddg as target in order to increase the training data. This external dataset contains 14656 unique mutations. Since the wild type is more stable than most mutants, we kept only unstable mutations and ensure that the pH and source of each group are the same. After a simple cleanup, the training set contains 6000+ rows.

# Ensemble
I used the three xgb models ensemble with  some public notebooks, I don't have a good idea here, just try to change the weight until the score stops improving.
# reference
- [XGBoost - 5000 Mutations 200 PDB Files [LB 0.410]](https://www.kaggle.com/code/cdeotte/xgboost-5000-mutations-200-pdb-files-lb-0-410)
- [🧬 NESP: ThermoNet v2 🧬](https://www.kaggle.com/code/vslaykovsky/nesp-thermonet-v2)
- [NESP: AlphaFold+GetArea exploration](https://www.kaggle.com/code/roberthatch/nesp-alphafold-getarea-exploration)
- [Difference Features - [LB 0.600]](https://www.kaggle.com/code/cdeotte/difference-features-lb-0-600)
- [RMSD from Molecular Dynamics](https://www.kaggle.com/code/oxzplvifi/rmsd-from-molecular-dynamics)
- [NOVO-ESP – Residue Depth and More w/ BioPython](https://www.kaggle.com/code/dschettler8845/novo-esp-residue-depth-and-more-w-biopython)
- [NOVO ESP – ELI5 - Performant Approaches [LB=0.451]](https://www.kaggle.com/code/dschettler8845/novo-esp-eli5-performant-approaches-lb-0-451)

---

Update:
The ensemble weight of this notebook wihch got 0.542 in private Leaderboard.
> [3Dgeometry](https://www.kaggle.com/code/gehallak/nesp-3d-geometry-0-32-lb)(0.32) + 1.5×foldx(0.413) + 1.5×[ThermoNet v2](https://www.kaggle.com/code/vslaykovsky/nesp-thermonet-v2)(0.495) + 2×[rosetta](https://www.kaggle.com/code/shlomoron/nesp-relaxed-rosetta-scores)(0.471) + 1.5×[rmsd](https://www.kaggle.com/code/oxzplvifi/rmsd-from-molecular-dynamics)(0.410) + xgb2(0.404) + 1.5×xgb1(0.521) + 1.5×[difplddt](https://www.kaggle.com/code/cdeotte/difference-features-lb-0-600)(0.299) + 1.5×xgb3(0.405)

| Model | Public LB | Private LB | Link |
| --- | --- | --- | --- |
| 3Dgeometry | 0.32 | 0.338 | [link](https://www.kaggle.com/code/gehallak/nesp-3d-geometry-0-32-lb) |
| Foldx | 0.413 | 0.375 | [link](https://www.kaggle.com/code/zrongchu/foldx) |
| Thermonet v2 | 0.495 | 0.460 | [link](https://www.kaggle.com/code/vslaykovsky/nesp-thermonet-v2) |
| rosetta | 0.471 | 0.438 | [link](https://www.kaggle.com/code/shlomoron/nesp-relaxed-rosetta-scores) |
| rmsd | 0.410 | 0.350 | [link](https://www.kaggle.com/code/oxzplvifi/rmsd-from-molecular-dynamics) |
| xgb2 | 0.404 | 0.432 | [link](https://www.kaggle.com/code/zrongchu/4000-unique-mutation-with-dtm/notebook) |
| xgb1 | 0.524 | 0.464 | [link](https://www.kaggle.com/code/zrongchu/nesp-xgboost) |
| difplddt | 0.297 | 0.287 | [link](https://www.kaggle.com/code/cdeotte/difference-features-lb-0-600) |
| xgb3 | 0.405 | 0.304 | [link](https://www.kaggle.com/code/zrongchu/nesp-xgboost?scriptVersionId=113857946) |
