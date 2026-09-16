# 11th place solution: Transformer with MFE distance embeddings (with code)

Competition: stanford-ribonanza-rna-folding
Rank: #11
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460192

I greatly appreciate the efforts of the hosts in organizing this competition. This was an unfamiliar domain filled with both challenges and fun. I wonder if the competition was worth it for organizers after all. Hopefully, the solutions shared by kagglers will prove beneficial to you and other researchers.

Here is a quick summary of my solution:
## Features
* **Eternafold MFE and BPPs**: I experimented with other packages (RNAstructure, Vienna, CONTRAfold, rnasoft, ipknot).  I found that only the features from Eternafold contributed significantly to my model.
* **CapR structure**. This led to faster convergence of the model.
* **bpRNA structure**. I am unsure of its contribution to the score.

## Model architecture
I forked Huggingface's BERT and implemented several modifications:
* **Attention injection support**: I added BPPs with 2d cnn on top to 8/16 of model's attention heads. The most effective variant involved a separate CNN layer for each attention head. However, this significantly increased training time with only a marginal improvement in score. I sacrificed a few score points in favor of faster experimentation. Most of my models in the final ensemble only use one 2d cnn layer per transformer block.
* **1d CNN layers between transformer layers**: This addition was based on the premise that closely situated nucleotides influence each other, and a CNN layer could effectively extract these features.
* **Relative embeddings**: I only used relative embeddings in my model. Initially, they seemed a safer choice due to the length differences between train/test sequences. More importantly, they appeared more intuitive in the context of RNA structure, as opposed to absolute embeddings. I used two distance types:
  * The shortest distance along ribose-phosphate chain.
  * The shortest distance along the MFE structure.  
  * To illustrate this point, consider the image below, which depicts an RNA molecule from [this notebook](https://www.kaggle.com/code/brainbowrna/rna-science-computational-environment). C and G are encircled and located at opposite ends of the RNA chain. Despite their positioning, they are actually paired with each other, indicating that the distance between them is merely 1 in terms of the RNA folding structure.  


## Performance
The single model achieved a public score of approximately 0.1432. However, by averaging models with minor variations, I was able to achieve a score below 0.14.  

The code is available at https://github.com/chubasik/stanford-ribonanza-rna-folding

P.S: I had to post it a second time for it to be marked as a write-up
