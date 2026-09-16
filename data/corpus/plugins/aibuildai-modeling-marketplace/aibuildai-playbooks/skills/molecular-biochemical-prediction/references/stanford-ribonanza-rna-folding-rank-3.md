# [3rd Place Solution] AlphaFold Style Twin Tower Architecture + Squeezeformer

Competition: stanford-ribonanza-rna-folding
Rank: #3
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460403

A big thanks to the organizers for making this competition happen and putting their efforts in solving hard problems, such as RNA structure prediction. This sentence stuck with me throughout the whole competition: “... without being able to understand how RNA molecules fold, we are missing a deeper **understanding of how nature works, how life began**, and how we can design…“

Many will glance over this statement but it is a huge mystery that sometimes kept me awake at night. I mean it is SO strange… the evolution/creation of 4 nucleotides with specific physical/chemical properties that can code and take on different structures to fulfill specific actions in the body, and we still don’t understand the origin/purpose of these molecules… Hopefully we are in the right direction in finding out the truth and reason behind it or even “what” created them. 

Solution is not based on dozens of ensemble models, but rather 2 strong independent models. There are 2 models since I wasn’t sure if I will be able to produce an Alpha fold style viable solution in the allotted time, so I focused on a smaller “safe” version model first and a bigger “riskier” model later in the competition. But the ensemble of the 2 provided crucial in gaining the 3rd place.

## TLDR
Blend of 2 Independent models. The smaller “safer” version model is based on augmented Squeezeformer architecture, which consists of RelativeMultiheadSelfAttention, Convolution and FeedFoward modules. Learnable BPP’s through 2d convolution are added to the attention scores, and augmented low Signal-to-Noise data is additionally used in training apart from clean training data. Bigger twin-tower model is based on augmented Alpha Fold style architecture, that consists of MSA stack representation and Pair stack representation, that communicate in a criss-cross fashion through Outer Product Mean and Pair Representation Bias.

##Data Preprocessing/Cross Validation

Training data was split in 4 folds similar to this [notebook](https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb/notebook). There was a stable correlation between local cv vs lb. BPP’s were processed and cached as .npz files. Most of the time, models were evaluated only on fold0. 
Twin-tower model was trained only on clean training data and it has the power to predict its confidence/error_estimate similar to pLDDT in AlphaFold. After getting a good generalization from the twin-tower model on the clean data, new training data was created by augmenting the noisy low SNR data with the model’s confidence/error_estimate. For a particular nucleotide position in the low SNR dataset the idea is to combine how confident the model's prediction is with the position's reactivity error from the experiment and thus “fix” the noisy data. This gave significant improvements in the smaller “safer” model which was trained on clean dataset + improved low snr dataset. Even with this useful new data being available for the bigger twin-tower model, it was never trained on this extra data because of time constraints and only the clean dataset was used for final submission. Twin-tower experiments are currently underway on the full dataset.

##Squeezeformer Model
The smaller “safer” model is based on Squeezeformer architecture which was used in previous Google American Sign Language competition. The inputs to the model were tokenized RNA sequences and BPP matrices. The model consists of 14xSqueezeformer blocks and an output projection layer that predicts chemical reactivities at each position. One Squeezeformer block consists of three modules: Relative MultiHeadSelfAttention module, Convolution module, and FeedForward module.



Instead of absolute positional encodings, relative encodings are used for generalization on longer sequences. Attention scores apart from relative pos scores are further affected by BasePairProbability matrices, even though I was reluctant to use them at first since they are created with software that is not capable of detecting long range pseudo-knots and this bias is unfortunately induced in the model. The attention scores in the transformer are calculated in this manner: (content score + relative position score)/sqrt(head_dim) + bpp_bias_score. Bpp_bias_score is obtained by passing the BPP matrices through a 2D convolution block.

## AlphaFold style Twin-tower Model
This model was inspired from Google's AlphaFold and its derivatives OpenComplex/RhoFold. The original AlphaFold architecture relies on two different input representations for its predictions. It jointly uses MSA(Multiple Sequence Alignment) and Pair Representation features. The MSA representation uses row-wise attention to find intra-sequence features, while the column-wise attention is used to obtain inter-sequence evolutionary signals from the MSA stack. Since MSA for this competition wasn’t helpful in extracting evolutionary information, because of its synthetic nature of the RNA sequences, just the tokenized input sequence was used. Because MSA was not used, the axial-self attention was replaced with relative multi head attention and convolution.


(At the time of submission, only the Single representation branch was used for predictions, the pair features were completely ignored due to time restrictions)

###The workflow of the data from the input all the way to the prediction is as follows:

1) Input sequence gets tokenized, sequence and pair masks are generated

2) Tokenized sequence are embedded through embedding networks: MSANet and PairNet
- PairRepresentation features are embedded using Relative2D positional encodings to provide information about position of residues. Maximum position is clipped at 32, and each position afterwards is considered as “far” away. This inductive bias trains the model not to rely heavily on nucleotide positions and generalize better to any length, as stated in the alpha fold supplemental materials.
- MSA representation is passed through a simple Embedding layer (since positional encodings are added later in the transformer layer).

3) Embedded MSA and Pair features are passed through the main trunk of the network, which consists of 8 Chemformer blocks illustrated above. Some key points are:
- MSA representation is updated within the Squeezeformer attention 
instead of axial self-attention proposed in the original paper.
- Pair representation is updated only with triangular multiplicative updates. Triangular self-attention wasn’t used in this model because of memory constraints but using both should give an increase in score.
- The MSA representation updates the pair representation through an element-wise outer product that is summed over the MSA sequence dimension.
- The Pair representation updates the MSA representation through a projection of additional logits from the pair stack to bias the MSA attention scores.
- Both representations are passed through a 2-layer MLP that acts as transition before the communication.
- This communication is repeated within each block, so 8 times in total.
- Residual connections and Row-wise/Column-wise dropouts are used.

4) The processed representations MSA Features and Pair Features are then passed through various heads to extract meaningful information such as confidence/error_estimate, chemical reactivity, base_pair probabilities etc.

## Training

Training procedure is as follows:

First the Twin-tower model is trained on the clean dataset for 60 epochs with lr of 1e-3. Batch size of 8 per gpu was used with gradient accumulation of 4, leading to an effective batch size of 64 on 2 GPUs. The optimizer used was AdamW with cosine scheduling, weight decay parameter is 0.05 and warmup is 0.5. Model was trained on 2x4090 for 30 hours total. No extensive parameter tuning was done for this model.
This leads to a 0.13746 public/0.14398 private single model score. After this a synthetic dataset was created by blending the model’s predictions with a subset of noisy data that was between 0.35<data<1 SNR. Selecting a lower SNR threshold increased the training examples but lowered the quality. Current synthetic dataset is created with simple weighing, because of time constraints even though the model was trained and outputs valid plddt/error_estimates, they were not fully utilized. I didn’t have time to experiment with the formula for combining plddt score with experimental reactivity_error on a per nucleotide basis which is guaranteed to produce better synthetic data. 
The smaller “safer” model was trained for 200 epochs with lr of 7e-4 a batch size of 64. Optimizer and lr scheduling was the same as before, however no grad accumulation was used. This model was trained both on clean dataset and the synthetically created dataset which doubled the training examples. This single model achieves 0.13865 public/0.14256 private score.

A day before the competition ended I decided to just run longer epochs on the Twin-tower model, so I continued the same training procedure with starting weights from epoch 30 of the previous iteration. This gave improvements just by running the model longer which led to the final 0.13706 public/0.14366 private score of this single model. The final score is a blend of the twin-tower model which was trained only on the input sequence and tried to learn all of the interactions on its own and the squeeze model which contributed with the BPP’s and the synthetic data to the final prediction.


##Not utilized/Can be improved for Twin-tower model

- Model was trained only with sequences as input. 
- BPP’s and other supplemental data can be used. 
- Synthetic dataset which improved the smaller model should be used.
- With my tests, by increasing the depth (8->12) blocks, there is an increase in score, submission is only 8 blocks long. 
- Deeper model 16,24 blocks will be tested soon by utilizing checkpoiting/rematerialization to counter the memory problem and interleaving the convolution blocks after several MHSA blocks to aid training for this deeper model. 
- Recycling from the original paper was not utilized.
- Only triangular multiplicative updates are used in the Pair Representation, triangular self-attention was not utilized.
- Since RNA can take on multiple conformations, dropout can be utilized at inference time and results averaged to get a better estimate of the RNA’s structure

All of these adjustments are very likely to provide benefits to the model. Some of the tests are currently underway.


##Comments 

The twin-tower model was a large task, partially because I was competing solo. The model at submission time was trained only on clean dataset, and only the MSA Feature pathway was used in the predictions, the other Pair Feature pathway was completely ignored, but it can be used to try and recreate the BPP’s which should increase the score (tests are underway as im writing). Best submission of the model without BPP’s, loop types or any pre/post processing is 0.13709 public and 0.14366 private. However, the twin-tower model has a bigger gap in generalization compared to the smaller “safer” squeezeformer model which scored 0.13865 public but 0.14265 private, so currently investigating the reason behind it.

I am new to machine learning. I started to learn the field in May of this year, so I am sure there will be a lot of mistakes in the code and in my approach and sorry if my explanation is all over the place, all this is new to me and I am still learning.

Open Sourced Code:
https://github.com/GosUxD/OpenChemFold

## References:
[1] Squeezeformer: An Efficient Transformer for Automatic Speech Recognition
Sehoon Kim, Amir Gholami, Albert Shaw, Nicholas Lee, Karttikeya Mangalam, Jitendra Malik, Michael W. Mahoney, Kurt Keutzer arXiv:2206.00888 [eess.AS] https://doi.org/10.48550/arXiv.2206.00888

[2] Winner of Google American Sign Language Fingerspelling Competition
https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution

[3] Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583–589 (2021). https://doi.org/10.1038/s41586-021-03819-2

[4] OpenComplex github code repository:
https://github.com/baaihealth/OpenComplex

[5] E2Efold-3D: End-to-End Deep Learning Method for accurate de novo RNA 3D Structure Prediction
Tao Shen, Zhihang Hu, Zhangzhi Peng, Jiayang Chen, Peng Xiong, Liang Hong, Liangzhen Zheng, Yixuan Wang, Irwin King, Sheng Wang, Siqi Sun, Yu Li. arXiv:2207.01586 [q-bio.QM] https://doi.org/10.48550/arXiv.2207.01586
