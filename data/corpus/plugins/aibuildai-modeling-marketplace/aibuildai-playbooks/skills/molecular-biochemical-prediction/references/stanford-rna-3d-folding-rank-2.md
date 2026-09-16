# 2nd Place Solution

Competition: stanford-rna-3d-folding
Rank: #2
Source: https://www.kaggle.com/c/stanford-rna-3d-folding/writeups/2nd-place-solution

First of all, thank you to the organizers and Kaggle for hosting the competition, and thank you to all the participants contributing to one of the major problems in biology.

# Background
I am a bioinformatician and have participated in CASP[1] since CASP11. Hence, I am familiar with processing PDB[2] data and identifying useful data to some extent (I have mainly studied proteins, so my knowledge of RNA and DNA is more limited).

#  Overview of the approach 
My approach is primarily Template-Based Modeling (TBM), which involves identifying similar data from a dataset of known 3D structures and transferring the atomic coordinates to the prediction target. TBM was the most effective approach for protein structure prediction prior to the emergence of AlphaFold2[3] (or residue–residue contact prediction[4]; note that AF2 also incorporates TBM-like processes).

# Details of the submission 
## Representation-Based Sequence Search and Alignment (RBSSA)
In modern deep learning models, not only the outputs of the final layer but also those of intermediate layers can be used for various tasks. In this document, I use the term representation (although embedding may be more common) to refer to such intermediate outputs. Recently, in protein research, using representations has become a popular technique for remote homology search[5,6,7], which is an important step in TBM.

I evaluated several foundation models for RNA and DNA, but some were unsuitable due to licensing restrictions. RNAErnie[8,9] and RibonanzaNet[10] were potential candidates. In preliminary experiments, RNAErnie performed slightly better, but because it required substantially more disk space due to its larger output, I chose RibonanzaNet. Since RibonanzaNet2[11] was published very recently, I did not have sufficient time to implement and evaluate it.

Single representations before the decoder layer of RibonanzaNet were generated for both target and template sequences, which were then aligned using a simple dynamic programming approach (Figure 1). A high score suggests a possible evolutionary relationship, making the pair a TBM candidate. Aligning a target sequence against all cluster representatives (~~4,779~~ 3,991 sequences) took about 160 seconds (including RibonanzaNet inference for a target sequence) for a 300-nt target sequence.


Figure 1. Schematic of Representation-Based Sequence Search and Alignment. In Smith–Waterman alignment, bases in the target and template database sequences are compared: matches receive positive scores, mismatches receive negative scores, and the highest scoring path is the optimal alignment. For alignments using matrix-like representations, match/mismatch scores are replaced with similarity functions such as cosine similarity or Pearson correlation coefficient.

For RBSSA, I used my own tool[12], which was originally designed for multiple sequence alignment construction[13]. (After the competition, I updated it to improve user experience for remote homology search[14].)


## Template Dataset Preparation
I downloaded RNA-containing entries from PDB on 2025-05-21. The structures were clustered by sequence similarity using MMseqs2[15] at 95% sequence identity, with some additional miselleneous filtering. This resulted in 4,779 clusters. Cluster representatives were selected based on the number of ~~C1' atoms~~ modeled nucleotides. RibonanzaNet representation were generated for these cluster representatives to construct a RBSSA database.

## Standard TBM
Because my representation-based search tool does not provide statistical confidence, it can produce many false positives. Therefore, in addition to RBSSA, I applied a standard TBM approach: searching template sequences with blastn[16] and aligning the hits using the Smith–Waterman algorithm[17].

## Deep Learning-Based Structure Prediction
Since suitable templates may not always exist, or templates may have missing regions, I also utilized Chai-1[18] and Boltz-1[19]—AlphaFold3-inspired tools and current state-of-the-art deep learning-based structure predictors.

If extra time was available,
・Mutated some nucleotides and fed them into predictors to generate more diverse structures.

・Predicted structures using dna or rna sequences in the all_sequences column.

・Predicted structures of multimeric RNA as a concatenated monomer.

## Assembly Structures
Because TBM often leaves missing regions, these gaps were patched using structural alignments with models predicted by DL-based methods or other templates.. Structural alignment was performed with BioPython's SVDSuperimposer[20]. No further refinements, such as molecular dynamics, were applied due to prioritization and time constraints of the notebook.

# Results & Discussions
The results are shown in Table 1.

|  | private | public (Sep.) | public (May) |
| --- | --- | --- | --- |
| full | 0.56125 | 0.59655 | 0.605 |
| full (dot) | 0.56966 | 0.66171 | 0.672 |
| -RBSSA | 0.50551 | 0.44539 | 0.46 |
| -RBSSA -TBM | 0.374 | 0.32237 | 0.37 |


Table 1. Summary of results.

'Public (Sep.)' refers to the scores in the Public Score column on my current submissions page, while Public (May) refers to the scores visible on the public leaderboard at the first deadline. Note that 'Public (May)' notebooks used a development version, so the overall pipeline is not consistent. 'full (dot)' indicates RBSSA with dot product as the scoring function, while 'full' uses cosine similarity. '-RBSSA' means without RBSSA, and '-TBM' means without standard TBM.

The table highlights the effectiveness of both TBM and RBSSA.
However, the 1st place solution https://www.kaggle.com/competitions/stanford-rna-3d-folding/writeups/1st-place-solution discussions revealed that standard sequence alignment can achieve very high scores. I will further evaluate TBM variations using different sequence alignment algorithms.

# Acknowledgements
I would like to thank the competition organizers and the Kaggle team for hosting this exciting challenge. I am also grateful to the Kaggle community for sharing helpful notebooks and discussions, and to the developers and maintainers of the open-source tools and public databases that I used. Finally, I appreciate the assistance of ChatGPT and DeepL in improving my English text.

# References
[1] https://predictioncenter.org/
[2] https://www.rcsb.org/
[3] https://github.com/google-deepmind/alphafold
[4] https://en.wikipedia.org/wiki/Protein_contact_map
[5] Kaminski, Kamil, et al. "pLM-BLAST: distant homology detection based on direct comparison of sequence representations from protein language models." Bioinformatics 39.10 (2023): btad579.
[6] Pantolini, Lorenzo, et al. "Embedding-based alignment: combining protein language models with dynamic programming alignment to detect structural similarities in the twilight-zone." Bioinformatics 40.1 (2024): btad786.
[7] Liu, Wei, et al. "PLMSearch: Protein language model powers accurate and fast sequence search for remote homology." Nature communications 15.1 (2024): 2775.
[8] https://github.com/CatIIIIIIII/RNAErnie
[9] https://huggingface.co/multimolecule/rnaernie
[10] https://github.com/Shujun-He/RibonanzaNet
[11] https://www.kaggle.com/models/shujun717/ribonanzanet2
[12] https://github.com/yamule/matrix_align/tree/523262de8fc5da274478d3d31e46a089991056a2
[13] https://en.wikipedia.org/wiki/Multiple_sequence_alignment
[14] https://github.com/yamule/matrix_align/commit/ad04b715ec293b8dffa16bb74506d49bd3294e05
[15] https://github.com/soedinglab/MMseqs2/
[16] https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html
[17] https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
[18] https://github.com/chaidiscovery/chai-lab
[19] https://github.com/jwohlwend/boltz
[20] https://biopython.org/docs/1.85/api/Bio.SVDSuperimposer.html

The URLs were accessed on 2025-10-02.
