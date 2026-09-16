# GraphAttention solution approach

Competition: stanford-ribonanza-rna-folding
Rank: #46
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460252

Hi there,thanks Stanford University for this competition!
Might be someone will found my work is interesting.
Note enough of time was my biggest problem,but after all this is my first medal,even though its silver, i am really happy.


1. Preprocessing RNA sequence to graph:
1.1. Used Eternafold pkg for extracting secondary structure.
1.2. OHE nucleotids -> node features
1.3. Used as edge features -> [phosphodiester_bond,base_pairing(canonical or wobble),BPPS]
2. My final solution was:
2.1. Random walk as positional encoding.
2.2. Architecture: 
Combination of:
local attention -> GraphTransformer.
global attention -> Attention encoder.
20 layers depth nn(192 hidden dim), for me expands hidden dimension doesn't help.



This works helps me a lot:
https://arxiv.org/abs/2009.03509
https://arxiv.org/abs/2205.12454

3.
Default splitting data on train and validation(10 % of SN filter =1) 
During train process using rna sequence with signal to noise >= 0.8

For final submission used only one model prediction no stucks, that might increase quality of prediction as i said before no time was problem.


github:https://github.com/cerenov94/ribonanzaRNA
Instruments:
Pytorch Geometric,Graphein
