# My incredibly lucky 6th place solution w code

Competition: novozymes-enzyme-stability-prediction
Rank: #6
Source: https://www.kaggle.com/c/novozymes-enzyme-stability-prediction/discussion/375900

Somehow I ended up in 6th place, and I can't believe how lucky I am. I started working on this competition with only 4 weeks left and my main idea was to use a rotation and translation equivariant network (e3nn), which I failed to develop into a strong solution on its own. Most of my ensemble comes from public notebooks, I will reference them shortly but in this post I will quickly discuss my e3nn network.

# Data

I used the same dataset compiled by the thermonet notebook, so nothing fancy really.

# My e3nn network

The good performance of thermonet surprised me a lot, mostly because 3D conv is not rotation equivariant, meaning that if you rotate a protein structure, it would give you different results. Obviously, you would think this would be a problem because we know rotating a protein structure shouldn't change its properties but thermonet actually outperformed my e3nn network on both private and public. However, it could be that my e3nn is simply unoptimized because I've never used it before and don't know how to use it well. 

For convenience, i simply ported most the data pipeline from the themonet notebook but instead of 3D voxel grids I used atom types and xyz coordinates as inputs to e3nn, a euclidean neural network specifically designed to tackle 3D structures with rotation, translation, and permutation equivariance. The rest of my solution is identical to the thermonet notebook.

My code: https://github.com/Shujun-He/Novaenzymes

Some notable specifics of my e3nn:

1. I calculate the geometric center of the residue of the mutated position and only take atoms within a certain range (8)
2. Input node features other than atom types include residue and whether the structure is a mutant structure or wt structure

My e3nn used in the final sub scored 0.362/0.346 on private/public.

# Ensemble
I ensemble an 8 fold e3nn with the public 0.603 notebook (https://www.kaggle.com/code/oxzplvifi/deletion-specific-ensemble) with 0.8 (public 0.603 notebook)/0.2 (8 fold e3nn) weight. For inference, I use both alphafold structures and rosetta relaxed structures (generated with thermonet codebase) and averaged the predictions.


# References
pLDDT, BLOSUM, DeepDDG and DeMaSk procedures are from:
https://www.kaggle.com/code/dschettler8845/novo-esp-eli5-performant-approaches-lb-0-425
with additional references:
https://www.kaggle.com/code/kvigly55/plldt-and-ddg
https://www.kaggle.com/code/lucasmorin/nesp-changes-eda-and-baseline
https://www.kaggle.com/code/hengck23/lb0-335-deepdgg-server-benchmark
Differential pLDDT procedure is from:
https://www.kaggle.com/code/cdeotte/difference-features-lb-0-600
with mutation PDB files from:
https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/361816
https://www.kaggle.com/datasets/roberthatch/nesp-kvigly-test-mutation-pdbs
RMSD and SASA from MD simulation procedure:
https://www.kaggle.com/code/oxzplvifi/rmsd-from-molecular-dynamics
with the following two MD simulation outputs used:
https://www.kaggle.com/datasets/oxzplvifi/novozymes-md for SASA measurement
https://www.kaggle.com/datasets/oxzplvifi/novozymes-md2 for RMSD measurement
The output files of Rosetta and Thermonet were loaded directly since I am not sure yet if they can be implemented in R:
https://www.kaggle.com/code/shlomoron/nesp-relaxed-rosetta-scores for Rosetta
https://www.kaggle.com/code/vslaykovsky/nesp-thermonet-v2 for Thermonet

e3nn github:https://github.com/e3nn/e3nn
e3nn paper:https://arxiv.org/abs/2207.09453
