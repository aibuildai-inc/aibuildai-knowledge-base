# 8th Place Solution | No Tg Post Processing

Competition: neurips-open-polymer-prediction-2025
Rank: #8
Source: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/8th-place-solution-no-tg-post-processing

# External data
After I shared with all participants external datasets, I (and many others) discovered another important data - [POINT2 data](https://github.com/Jiaxin-Xu/POINT2/blob/main/results/Tg_uq/BNN_MACCS_2_2048/smiles_train.csv). I did a simple check of each external dataset, tracking the difference in SMILES that are in both the organizers' and the external data. I also used other external data, but that data is the key for winning.


There are 91 SMILES intersect here. We can assume that all the data from POINT2 is lower on >20°C than the organizers' data (with the exception of some serious deviations). This is clearly related to the duration and parameters of the simulation, the targets of which are easily adjusted in all external data (many participants didn't understand why I subtracted 0.118 for `Density` in my [baseline](https://www.kaggle.com/code/dmitryuarov/neurips-baseline-external-data)). So, simple and obvious adjustment **BEFORE** training:
```
extra_data_tg['Tg'] += 20
```
## Important note
Before someone start complaining about the ban on using external data, please keep in mind that I was the first who took [a big step for saving this competition](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/587318). We all have tried many times to get a clear answer about what can and can not be used, but in response we have received silence. Therefore, I followed two simple rules: 1 - the data is open to everyone, 2 - the data is not prohibited by the organizers. I pursued only academic goals. My conscience is clear, and I have not broken any rules. When I studied in detail the issue of the permissibility of using external data, I came to the conclusion that this is a legal hole in which: first - "No license" ≠ "free to use", second -  many external data for academic purpose only (PI1M, for example). Almost all of us have violated... or not violated the rules (it depends on the interpretation of the rules by the organizers solely on their desire). In the future, it's quite easy to patch this hole with more competent work by lawyers or a clearer indication by organizers of what can and cannot be used. *I'm not going to discuss this anymore, because I'm definitely tired of it - I'm a programmer, not a lawyer.*

# Feature engineering
I use Mordred molecular descriptor, because it turned out to be more accurate in calculations and gave more information. Also I used Morgan fingerprints (512 bits) for `Tg`, `FFV`, `Rg` and MACCS fingerprints for `Tc` and `Density`. After removing unnecessary features, each target had the following number of features: `Tg`: 789, `FFV`: 836, `Tc`: 520, `Density`: 423, `Rg`: 654.

# Cross-validation
I used an extremely conservative data separation by creating Butina clusters and after dropping in train data all SMILES which have >0.9 Tanimo similarity with validation data. 5 folds.
```
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs

fps = [rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(Chem.MolFromSmiles(s)) for s in train['SMILES'].tolist()]

dists = []
nfps = len(fps)
for i in range(nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1-x for x in sims])
clusters = Butina.ClusterData(dists, nfps, distThresh=0.7, isDistData=True)
groups = [0] * nfps
for idx, cluster in enumerate(clusters):
    for member in cluster:
        groups[member] = idx

train['Tanimoto_group'] = groups

def filter_train_by_val_similarity(train_idx, val_idx, fps_all, sim_thresh=0.9):
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    train_fps = [fps_all[i] for i in train_idx]

    banned = set()
    for v in val_idx:
        if fps_all[v] is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps_all[v], train_fps)
        for loc, s in enumerate(sims):
            if s >= sim_thresh:
                banned.add(train_idx[loc])

    keep = np.array([i for i in train_idx if i not in banned], dtype=int)
    return keep

fps = [rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(Chem.MolFromSmiles(s)) for s in train_part['SMILES'].tolist()]
gkf = RandomGroupKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=seed)
for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_part, train_part[target], train_part['Tanimoto_group'])):
    trn_idx_filtered = filter_train_by_val_similarity(trn_idx, val_idx, fps, sim_thresh=0.9)
```

# Models
For all targets I used the same strategy but with different parameters and linear learning schedulers for TabM :


| Target | k | n_blocks | d_block | dropout | 
| --- | --- | --- | --- | --- |
| Tg | 64 | 2 | 512  | 0.1 |
| FFV | 64 | 2 | 512  | 0.1 |
| Tc | 32 | 2 | 128 | 0.0 |
| Density| 32 | 3| 256 | 0.0 |
| Rg | 32 | 2 | 256  | 0.0 |

Where are CatBoost and LGBM? For many targets they had high std between folds, so I decided not to use them. In addition, their presence almost did not improve the overall result in any way, because TabM had a huge gap from the trees for all targets except `Rg`. I don't see the point in sharing CV results for each target and for each model, because all of these results strongly depend on the amount of data and chosen cross-validation.

# Final thoughts
I refrained from writing anything on the first day after the end of the competition, because obviously everything suits me, and everyone else wants to run this roulette again in the hope of improving the Private LB results. No need to swear, please. I've lost for the last 3 competitions for all sorts of reasons and I know perfectly well what it means to lose months of work to nowhere. I adhere to the version that the data in the test is indicated in Celsius, but due to the fact that their modeling was clearly different from the original, the values turned out to be much higher. This is also confirmed by the fact that in almost all external data for any target, there is a distribution bias. I agree that there is too high bias, but it is quite possible. We all knew what we were doing, when saw many problems in this competition. Many wiser participants than the rest of us refrained from taking part, and they were right to save time and nerve cells. 

Good luck and patience to all ❤️
