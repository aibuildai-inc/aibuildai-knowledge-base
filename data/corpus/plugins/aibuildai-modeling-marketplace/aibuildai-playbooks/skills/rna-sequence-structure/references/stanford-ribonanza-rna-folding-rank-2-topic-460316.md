# 2nd place solution - Squeezeformer + BPP Conv2D Attention

Competition: stanford-ribonanza-rna-folding
Rank: #2
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460316

Thanks to Kaggle and the hosts for organizing this competition.  It was truly inspiring and challenging, and I learned a lot from this one.👍

###Code: https://github.com/hoyso48/Stanford---Ribonanza-RNA-Folding-2nd-place-solution

# TLDR

**Keypoints:**

- Squeezeformer[1] + GRU head.
- Simple Conv2DNet for bpp, adding it as a bias to the attention matrix.
- ALiBi positional encoding[2] for robust generalization on longer sequences.
- Weighted loss with signal_to_noise, with longer epochs.
- Additional features for minor score improvements.

I adopted Squeezeformer, which I became familiar with after the ASL fingerspelling competition.Thanks to @christofhenkel and @goldenlock for their solutions in the last ASL competition. The most crucial part of my solution is how to utilize the bpp matrix. I applied a simple shallow Conv2DNet to bpp and directly added it to the attention matrix.

**Features:**

I used some features found useful in the OpenVaccine Challenge, to help fast initial convergence. These included:

- CapR looptype.
- eternafold mfe.
- predicted Looptype with eternafold mfe.
- bpp features (sum, nzero, max).

However, unlike in the OpenVaccine challenge, these features only marginally helped (about -0.0005). Therefore, I believe these features should be removed in the future for the simplicity.

#Model



**Squeezeformer Encoder:**

I chose Squeezeformer with minor modifications (BN after conv1d, SwiGLU in FFN, etc.), which mixes Conv1D blocks with Transformer. While I tried other recent Conv-Transformer Hybrid architectures, Squeezeformer was the most efficient. Compared to a Vanilla Transformer, Squeezeformer showed strong performance early in training and consistently showed faster convergence.

The models used the following parameters: dim=192, num_heads=4, kernel_size=17, num_layers=12.

**GRU head:**

Adding a single GRU layer after the encoder yielded minor improvements.



**ALiBi positional encoding:**

I adopted AliBi positional encoding as it claimed to generalize better over long sequences than other methods.

**BPP as Attention Bias:**

The bpp matrix (using only the provided one) was added as a bias in the attention matrix after multiplied by per-head predefined scales, significantly improved performance (around -0.0025).

**BPP 2DConvNet:**

Using Bpp directly as an attention bias was a good start, but I felt it needed more flexibility(I felt it was too sparse). Among various options, adding a 2D CNN on top of the BPP matrix proved very helpful (-0.002). However, multiple 2D CNNs applied to the BPP matrix (usually 206 x 206) were inefficient in terms of training/inference time. Thus, I just used a simple shallow 2-layer 2DCNN, with the output matrix shared across all Transformer block layers.

#Training

- Epochs: 200.
- Batch size: 256.
- Learning rate: 2e-3, with Cosine Decay and warmup.
- Optimizer: AdamW, weight decay = 0.01.
- Loss: Weighted MAE (weight = log1p(signal_to_noise).clip(0,10)).


The single model CV (K-fold, k=5) scored 0.119, with a public LB of 0.140 and a private LB of 0.142. After ensemble on different seed I got public LB of 0.135 and private LB of 0.140.
Although there were some questionable correlations between CV/LB in certain submissions, generally, they aligned well with the CV.
With the above setup, training a single model took around 30 hours on a single RTX 4090. 

##Discarded ideas & Thoughts:

- **Self-Supervised Learning (SSL)**: At first I motivated to participate in this competition as it would be really nice if any SSL method could be successfully applied without using any features other than the sequence.  Initial trials with Data2Vec and BERT-like SSL methods showed inconsistent improvements. Due to the additional training time required, I did not consider SSL further. However, I believe there is still huge potential in this idea.
- **Large Models**: Attempts to train larger models (dim > 512) with proper regularizations were unsuccessful. I think this and SSL failure suggests that the primary challenge lies in the inherent noise within the training dataset.
- **Augmentations**: Most augmentation methods I tried had no effect.
- **Pseudo Labels**: While pseudo labeling might help in LB, it didn't improve CV in my case, so I didn't use it for safety&training time. However, after seeing the correlation between public and private LB, I think it might have been slightly beneficial in both public and private LB.


For me, this competition was a series of choices regarding whether to experiment with or adopt some promising ideas, especially when there were only 2-3 weeks left. Some of the ideas I thought might be helpful were abandoned without further consideration because they required more time for implementation and training. I think that this strategy may have made my solution somewhat suboptimal or redundant, but overall I see it worked quite well as my solution appeared to capture most of the crucial aspects of other teams' solutions.

##References
[1]Sehoon Kim, Amir Gholami, Albert Shaw, Nicholas Lee, Karttikeya Mangalam, Jitendra Malik, Michael W. Mahoney, and Kurt Keutzer. 2022. Squeezeformer: An Efficient Transformer for Automatic Speech Recognition. arXiv:2206.00888 [eess.AS]. [https://doi.org/10.48550/arXiv.2206.00888](url)

[2]Ofir Press, Noah A. Smith, Mike Lewis. 2022. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. arXiv:2108.12409 [cs.CL]. [https://doi.org/10.48550/arXiv.2108.12409](url)
