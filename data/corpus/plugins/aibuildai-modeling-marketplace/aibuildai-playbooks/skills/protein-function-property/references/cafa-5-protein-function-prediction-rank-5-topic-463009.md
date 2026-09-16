# 5th Place Solution for the CAFA 5 Protein Function Prediction Challenge

Competition: cafa-5-protein-function-prediction
Rank: #5
Source: https://www.kaggle.com/c/cafa-5-protein-function-prediction/discussion/463009

**A. MODEL SUMMARY**
**A1. Background on you/your team**
Competition Name: CAFA 5 Protein Function Prediction
Team Name: hfm7zc
Private Leaderboard Score: 0.56171 (maximum weighted F-measure, wFmax)
Private Leaderboard Place: 5

Name: Chengxin Zhang
Location: Department of Computational Medicine and Bioinformatics, University of Michigan – Ann Arbor, MI, USA
Email: zcx@umich.edu 

Name: P Lydia Freddolino
Location: Department of Biological Chemistry, University of Michigan – Ann Arbor, MI, USA
Email: lydsf@umich.edu

**A2. Background on you/your team**
Our team consists of two faculty members working on algorithm development of protein function annotation tools as well as their proteome-wide application to microbes and human. We have previously participated in the CAFA3 challenge as team Zhang-Freddolino lab, which was ranked first in the CAFA3 limited knowledge Biological Process category in CAFA3 [1]. We participated in CAFA5 to benchmark our latest protein function prediction methods, with a particular focus on the assessment of utility of structure templates and deep learning in function prediction. C.Z. and P.L.F. conceived the project and designed the pipeline. C.Z. developed the method, performed the CAFA5 prediction and submitted the prediction result.

**A3. Summary**
Our workflow, StarFunc, first independently generates five sets of Gene Ontology (GO) predictions for a query protein using five component methods:
- Sequence homolog search through the UniProt-GOA database by BLASTp [2].
- Structure alignment between the AlphaFold2 model [3] of the query protein and template structure from the BioLiP2 [4] and AlphaFold databases [5] by Foldseek [6] plus TM-align [7].
- Protein-protein interaction (PPI) partners recorded by the STRING database [8].
- Logistic regression models (one model per GO term) which use Pfam family matches as the input features.
- Prediction from a deep learning model (SPROF-GO) [9].
The prediction scores from these five components, as well as the background frequency of the GO term in the UniProt-GOA database (i.e., the “Naïve” method in CAFA3 evaluation [1]), are used as six input features to train three random forest models (one model per GO aspect) to derive the consensus score used for final submission. 

**A4. Features Selection / Engineering**
Based on wFmax of the component methods in an independent test set, the features (i.e., predictions from component methods) ranked in descending order of importance are sequence, deep learning, structure, Pfam, PPI and Naïve (Figure 1). 

**Figure 1.** Importance of different features, as measured by the wFmax of GO prediction on independent test set.

A major reason for the high performance of the sequence-based component method is the introduction of a new scoring function that weights each BLASTp hit by both the bit-score and the sequence identity [10]. 
For the component methods and the consensus prediction, only experimental GO terms (evidence codes EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP, TAS, and IC) are included in the training dataset. On the other hand, during training, if a protein has already has a non-experimental GO annotation in UniProt-GOA, this GO annotation is also included in the final result, where the prediction score is calculated as one minus the error rate associated with the evidence code, as determined by our previous study [11]. Note that GO annotations with IBA evidence are always excluded as they are found to have higher error rate than other types of GO annotations [11].

**A5. Training Method(s)**
We tested traditional Gradient Boosting Decision Tree (GBDT), Random Forest (RF), and Dropouts meet Multiple Additive Regression Trees (DART) implemented by LightGBM to ensemble the predictions from the five component methods. On an independent test set based on UniProt-GOA release 2023-05-18, we found that comparable results can be achieved by GBDT with 1000 trees and RF with 4000 trees, both of which are more accurate than DART. We eventually selected RF to derives the consensus prediction.

**A6. Interesting findings**
The treatment of term GO:0005515 “protein binding” is different from previous CAFA challenges. In previous CAFA challenges (CAFA1 to CAFA3), proteins whose only Molecular Function (MF) leaf term is "protein binding" will be excluded from MF evaluation. According to the CAFA3 paper, "protein binding is a highly generalized function description, does not provide more specific information about the actual function of a protein, and in many cases may indicate a non-functional, non-specific binding. If it is the only annotation that a protein has gained, … we deleted these annotations from our benchmark set." 
However, the CAFA5 Kaggle challenge does not treat the “protein binding” term differently, while it is unclear whether the “protein binding” term will be treated differently in the academic paper reporting CAFA5 official assessment. Due to these discrepancies, we submitted two different models using the same architecture to CAFA5, where model “zcx” exclude protein binding-only proteins when training the MF prediction model while model “hfm7zc” include these training proteins. Model “hfm7zc” seems to achieve better Leaderboard score (0.56171, ranked 5) than “zcx” (0.55539, ranked 8), showing that different treatment of “protein binding” can give a significant impact on the final Leaderboard ranking. We would like to register our opinion, however, that for the academic CAFA5 assessment, the exclusion of protein binding as a leaf term would be more appropriate and biologically useful . 
We also note, in contrast to the findings reported below in section A7, that based on our prior experience we expect that for more challenging annotation targets, it is likely that the non-sequence pipelines will be more important  [12, 13].

**A7. Simple Features and Methods**
Based on independent test, 98.7% of the performance (in terms of wFmax) can be achieved by the sequence-based component alone.

**A8. Model Execution Time**
The random forest models as well as the logistic regression models mentioned in section A3 can be trained within < 1 day using the CAFA5 training set. To perform inference, the overall pipeline takes a few minutes per protein, where most of the time is spent by the structure template alignment component. If only the sequence-based component is run, it takes at most a few seconds per protein in large-scale inference.

**A9. Acknowledgement**
We thank Quancheng Liu and Dr Xiaoqiong Wei for technical supports and insightful discussions. This work used the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

**A10. References**
1.	Zhou NH, et al: The CAFA challenge reports improved protein function prediction and new functional annotations for hundreds of genes through experimental screens. Genome Biology 2019, 20.
2.	Altschul SF, Madden TL, Schaffer AA, Zhang J, Zhang Z, Miller W, Lipman DJ: Gapped BLAST and PSI-BLAST: a new generation of protein database search programs. Nucleic Acids Res 1997, 25:3389-3402.
3.	Jumper J, Evans R, Pritzel A, Green T, Figurnov M, Ronneberger O, Tunyasuvunakool K, Bates R, Zidek A, Potapenko A, et al: Highly accurate protein structure prediction with AlphaFold. Nature 2021, 596:583-589.
4.	Zhang C, Zhang X, Freddolino PL, Zhang Y: BioLiP2: an updated structure database for biologically relevant ligand-protein interactions. Nucleic Acids Res 2023.
5.	Varadi M, Anyango S, Deshpande M, Nair S, Natassia C, Yordanova G, Yuan D, Stroe O, Wood G, Laydon A, et al: AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models. Nucleic Acids Res 2022, 50:D439-D444.
6.	van Kempen M, Kim SS, Tumescheit C, Mirdita M, Lee J, Gilchrist CLM, Soding J, Steinegger M: Fast and accurate protein structure search with Foldseek. Nat Biotechnol 2023.
7.	Zhang Y, Skolnick J: TM-align: a protein structure alignment algorithm based on the TM-score. Nucleic Acids Res 2005, 33:2302-2309.
8.	Szklarczyk D, Kirsch R, Koutrouli M, Nastou K, Mehryary F, Hachilif R, Gable AL, Fang T, Doncheva NT, Pyysalo S, et al: The STRING database in 2023: protein-protein association networks and functional enrichment analyses for any sequenced genome of interest. Nucleic Acids Research 2023, 51:D638-D646.
9.	Yuan Q, Xie J, Xie J, Zhao H, Yang Y: Fast and accurate protein function prediction from sequence through pretrained language model and homology-based label diffusion. Briefings in Bioinformatics 2023, 24.
10.	Zhang C, Freddolino PL: A large-scale assessment of sequence database search tools for homology-based protein function prediction. bioRxiv 2023:2023.2011. 2014.567021.
11.	Wei X, Zhang C, Freddolino PL, Zhang Y: Detecting Gene Ontology misannotations using taxon-specific rate ratio comparisons. Bioinformatics 2020, 36:4383-4388.
12.	Zhang C, Freddolino PL, Zhang Y: COFACTOR: improved protein function prediction by combining structure, sequence and protein-protein interaction information. Nucleic Acids Res 2017, 45:W291-W299.
13.	Zhang CX, Zheng W, Freddolino PL, Zhang Y: MetaGO: Predicting Gene Ontology of Non-homologous Proteins Through Low-Resolution Protein Structure Prediction and Protein Protein Network Mapping. Journal of Molecular Biology 2018, 430:2256-2265.
