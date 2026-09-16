# 8th place solution (KF Part)

Competition: stanford-ribonanza-rna-folding
Rank: #8
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460222

First of all, I would like to thank the competition hosts for their exceptional organization and the challenging yet enriching environment they created. Their dedication to fostering a space for learning and innovation is deeply appreciated. I also express my profound gratitude to my teammates, @onodera and @christofhenkel. Their collaboration, expertise, and unwavering commitment throughout this project were invaluable.

Here I will try to summarize some of the main points of our solution.

# Solutions from our temamates

- [onodera part](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460956)

# CV Strategy: GroupKFold

In the field of RNA secondary structure estimation, various studies including [Sato et al., 2023,](https://academic.oup.com/bib/article/24/4/bbad186/7179751?login=false) have highlighted the inappropriateness of using random splits for evaluation.This stems from the concept of 'Families' in RNA experimental data - groups of RNA molecules sharing specific functions or structures, resulting in high structural similarity within the same family. For instance, [Fu et al., 2022](https://academic.oup.com/nar/article/50/3/e14/6430845) noted that the [E2EFold model was overestimated due to this issue](https://academic.oup.com/view-large/figure/333767038/gkab1074fig3.jpg).

To address this, we adopted clustering based on RNA sequence edit distances, using cluster IDs for GroupKFold. Additionally, we aggregated all the limited samples with seqlen=206 into fold=0. This strategy allowed for continuous evaluation of the model's performance on longer sequences.



# Datasets

As part of my models, I experimented with alternating training using both the RMDB dataset and the competition dataset. This approach was adopted with the anticipation that it would enhance the model's ability to handle longer RNA sequences, a critical aspect in RNA secondary structure prediction.

# Features

We processed input sequences using the 'arnie' package, mainly utilizing eternafold, rnasoft, and rnastructure. The model inputs included embedding layers for structure and loop_type, and a Graph Neural Network (GNN) adjacency matrix for bpp (base pair probability).

However, bpp alone, representing the probability of forming a pair, requires many layers when combined with CNNs to view the entire graph. As an alternative to using Transformers for a global view, we utilized “structure” (renamed to 'chunk' due to naming conflicts) and “segment”, as defined in [bpRNA](https://academic.oup.com/nar/article/46/11/5381/4994207) [Danaee et al., 2018].



# Models

My architecture is inspired by [LegNet](https://academic.oup.com/bioinformatics/article/39/8/btad457/7230784?login=false) from @dmitrypenzar1996 and [OpenVaccine's 6th place solution](https://www.kaggle.com/competitions/stanford-covid-vaccine/discussion/189241) [nyanp]. It consists solely of CNN and GNN layers. The adjacency matrices for bpp, structure, chunk, and segment significantly differ, so I prepared independent CNN+GNN blocks for each type. Group Convolution and einsum enabled this without for loops.

Following [LegNet](https://academic.oup.com/bioinformatics/article/39/8/btad457/7230784?login=false), prediction head outputs a 100d vector instead of a 1d scalar, with weighted summation over bin values for the final output. This approach stabilized learning by controlling the output range in regression tasks.



# Loss

We opted for an MAE + MSE weighted by signal-to-noise ratio as my optimization function. MSE provides gradients similar to MAE in the early stages of training but approaches zero later, aiding in convergence.

# Pseudo labels

We conducted pseudo label training using predictions on the test set created collaboratively with teammates @onodera and @christofhenkel.

# Result

|Name|CV|CV (seqlen=206)|Public LB|Private LB|
|:--:|:--:|:--:|
|single best (scratch training)|0.1336|0.1133|0.14012|0.14299|
|single best (+pseudo label 1st)|0.1313|0.1152|0.13828|0.14222|
|single best (+pseudo label 2nd)|0.1306|0.1221|0.13739|0.14186|
|blending w/ all models|0.127645|0.109341|0.13626|0.14263|

# References

- Fu, Laiyi, et al. "[UFold: fast and accurate RNA secondary structure prediction with deep learning.](https://academic.oup.com/nar/article/50/3/e14/6430845)" Nucleic acids research 50.3 (2022): e14-e14.
- Sato, Kengo, and Michiaki Hamada. "[Recent trends in RNA informatics: a review of machine learning and deep learning for RNA secondary structure prediction and RNA drug discovery.](https://academic.oup.com/bib/article/24/4/bbad186/7179751?login=false)" Briefings in Bioinformatics (2023): bbad186.
- Danaee, Padideh, et al. "[bpRNA: large-scale automated annotation and analysis of RNA secondary structure.](https://academic.oup.com/nar/article/46/11/5381/4994207)" *Nucleic acids research* 46.11 (2018): 5381-5394.
- Sato, Kengo, and Michiaki Hamada. "[Recent trends in RNA informatics: a review of machine learning and deep learning for RNA secondary structure prediction and RNA drug discovery.](https://academic.oup.com/bib/article/24/4/bbad186/7179751?login=false)" Briefings in Bioinformatics (2023): bbad186.
- Penzar, Dmitry, et al. "[LegNet: a best-in-class deep learning model for short DNA regulatory regions.](https://academic.oup.com/bioinformatics/article/39/8/btad457/7230784?login=false)" *Bioinformatics* 39.8 (2023): btad45
