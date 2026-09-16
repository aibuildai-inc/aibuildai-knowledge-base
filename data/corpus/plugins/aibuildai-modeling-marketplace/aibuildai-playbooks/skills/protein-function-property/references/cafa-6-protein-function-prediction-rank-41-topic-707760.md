# 41st Place Solution for the CAFA 6 Protein Function Prediction Competition

Competition: cafa-6-protein-function-prediction
Rank: #41
Source: https://www.kaggle.com/c/cafa-6-protein-function-prediction/writeups/41st-place-solution-for-the-cafa-6-protein-functio



I would like to begin by saying that this was my first Silver Medal in a Kaggle competition, and I am very happy with the final solution I was able to develop. I am also grateful to the competition organizers for designing such a challenging and meaningful problem. I would also like to thank the Kaggle community particularly the discussions and public notebooks which were extremely helpful throughout the competition and contributed significantly to my understanding and final approach.

# Context
- **Business Context**: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview 
- **Data Context**: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data
# Overview of the Approach
My final solution combined three complementary sources of information:
1. A deep learning model trained on protein language model embeddings and taxonomic information.
2. A post-processing step that enforced the hierarchical structure of the Gene Ontology (GO).
3. An ensemble that blended neural network predictions with sequence homology and existing annotation evidence.

The overall philosophy was simple: use the neural network to learn sequence-function relationships, apply biological constraints to ensure logically consistent predictions, and then combine multiple evidence sources to improve robustness.
## Data Representation

Each protein was represented using two types of input:

- ESM2 embeddings: Precomputed embeddings generated from the ESM2 protein language model, which provide rich representations of protein sequences.
- Taxonomic information: Species identifiers converted into learned embeddings to provide biological context.

These two inputs were concatenated to form the final representation used by the model.

## Model Architecture
The predictive model followed a shared-backbone, multi-head design.

A lightweight convolutional backbone processed the combined input representation and learned general protein features shared across all prediction tasks.

The network then branched into three independent prediction heads corresponding to the three Gene Ontology sub-ontologies:

- Biological Process (BP)
- Molecular Function (MF)
- Cellular Component (CC)

This design allowed the model to learn common representations while still specializing for the different aspects of protein function.

## Training Strategy

Protein function prediction is an extremely imbalanced multi-label classification problem. Some GO terms are very common and broad, while others are rare and highly informative.

To address this, I used a positive-weighted binary cross-entropy loss weighted by normalized Information Accretion (IA) values provided by the competition.
$$\mathcal{L}_{\text{term}}(y_j, \hat{y}_j) = - w_j^{\text{IA}} \cdot \left[ w_A^{\text{pos}} \cdot y_j \cdot \log(\hat{y}_j) + (1 - y_j) \cdot \log(1 - \hat{y}_j) \right]$$

Where:
- **y_j** (which is either 0 or 1) is the ground-truth label for term j.
- **y_hat_j** (a value between 0 and 1) is the model's predicted probability for term j.
- **w_A^pos** is the static positive class weight multiplier specific to aspect A (like BP, MF, or CC) to counteract label sparsity.
- **w_j^IA** is the Information Accretion (IA) value provided by the competition for term j , min-max normalized to the stable gradient range of 0.5 to 1.0.

This encouraged the model to place greater emphasis on learning informative and specific annotations rather than focusing primarily on frequent, generic terms. Since the evaluation metric itself rewards more informative predictions, aligning the training objective with the competition metric proved beneficial. 

## Validation Strategy
Validation was performed using a train–test split of the training data, with Micro-F1 score used as the evaluation metric. Hyperparameters and ensemble weights were selected based on validation performance before generating the final submissions.

## Hierarchical Post-Processing

Gene Ontology is organized as a directed acyclic graph (DAG), where child terms imply the presence of their ancestor terms.

Because the neural network predicts each GO term independently, its raw outputs may violate these hierarchical relationships. For example, a model could assign a higher confidence to a specific child term than to its parent.

To address this, I applied GO propagation using the True Path Rule. Prediction scores were propagated upward through the ontology so that every ancestor term received a confidence score at least as high as any of its descendants.

This produced biologically consistent predictions and provided a noticeable improvement in performance.

## Multi-Source Ensembling

The final submission combined three complementary prediction sources.

### 1. Neural Network Predictions

The tri-head deep learning model captured general sequence-function relationships from protein embeddings and taxonomy information.

### 2. BLAST Predictions

BLAST-based predictions transferred annotations from proteins with similar sequences.

These predictions are particularly reliable when close homologs exist and complement the neural network's ability to generalize beyond exact sequence matches.

### 3. GOA Propagation Predictions

The final component incorporated predictions derived from UniProt Gene Ontology Annotation (GOA) resources after applying the same hierarchical propagation procedure.

This provided an additional source of biological evidence independent of both the neural network and BLAST.

### Final Ensemble

The three prediction sources were combined using weighted averaging. The ensemble leveraged:

learned sequence patterns from the neural network,
evolutionary evidence from sequence homology,
and existing biological knowledge from curated annotations.

The combination consistently outperformed any individual component used in isolation.

# What Was Important

The most significant performance gains came from hierarchical post-processing and multi-source biological evidence integration. GO hierarchy propagation consistently improved predictions by enforcing biologically valid relationships, while ensembling further strengthened robustness. Among external sources, GOA annotations provided the largest boost, followed by BLAST homology signals, which were particularly effective for proteins with close sequence matches.

On the modeling side, I experimented with multiple backbone choices, including different ESM2 variants (650M, 3B, and 15B) as well as ProtT5, with ESM2-3B performing best overall. The final tri-head Conv1D architecture consistently outperformed simpler MLPs and alternative designs.

Training stability was strongly influenced by the choice of IA scaling and positive class weights, and careful tuning was necessary for stable convergence.

Performance analysis showed that the model handled Molecular Function (MF) best, followed by Cellular Component (CC), while Biological Process (BP) remained the most challenging.

# What I Tried That Didn't Help

More complex architectural variations generally did not improve validation performance. Increases in model complexity often led to marginal or inconsistent gains compared to the simpler tri-head Conv1D backbone combined with well-tuned loss weighting and biological post-processing. I also experimented with alternative predictors such as tree-based models, but these did not perform well and underperformed the neural architecture on validation.

# Future Work

Future improvements could include multi-task loss designs that better capture dependencies between GO sub-ontologies, as well as more careful architectural modeling of hierarchical relationships directly within the network. Additionally, more robust cross-validation strategies beyond a simple train–validation split could help reduce variance in performance estimation and improve generalization, especially given the imbalance and hierarchical structure of the GO labels.

# Sources
https://www.kaggle.com/datasets/kaitonmh/cafa-6-embeddings

https://www.kaggle.com/code/redberry/blast-basic-protein-function-predictions-cafa6

https://www.kaggle.com/code/yongsukprasertsuk/cafa-6-goa-propagation
