# 5th Place Solution

Competition: stanford-ribonanza-rna-folding
Rank: #5
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460250

I would like to congratulate my teammates @ayaanjang and @junseonglee11 on this hard earned (master triplet) win! We each had our own takes on the competition and they were combined for the final ensemble. Below we present the most important performance contributors; each of our transformer based models in the ensemble used a different combination of the following methods.

## Code
- Roger: https://github.com/s-rog/StanfordRibonanza2023
- Ayaan: https://github.com/ehdgnsdl/2023-Stanford-Ribonanza-RNA-Folding
- Junseong: https://github.com/JunSeongLee1102/2023-Kaggle-Stanford-Ribonanza-RNA-Folding
- Docker [Image](https://hub.docker.com/layers/junseonglee1102/ngc-custom/xformer-tmux/images/sha256-c98a8b37c2134b268056f5a77797b8f7e847938530d1fa603a4e1422153e1ada?context=repo)

 


## TLDR
We started with @iafoss's baseline model and made the following changes:
- Utilize lower SN samples
- Utilize Eterna BPP
- Replace layer norm with RMS norm
- Replace absolute positional embedding with relative positional bias
- Replace encoder layer pre-norm layout with ResiDual norm layout
- Remove bias from both QKV and FFN layers
- Increase head size from 32 to 48
- Add a single layer bidirectional GRU at the end of each encoder layer
- Multi-stage pseudo label training
- Misc: inverse square root LR schedule, 0.1 weight decay

## Data Sampling
I set a fixed number of sequences per epoch and used `WeightedRandomSampler` to draw samples without replacement weighted by SN. A bias term is added to the SN as a hyperparameter and is set on a schedule, increasing the amount of data used and regularization throughout training. The schedule hyperparameter can look something like:
`{(i * 10): v / 10 for i, v in enumerate(range(-12, -7))}`

## Data Augmentation
We were able to utilize flip to an extent. While training with flip usually resulted in longer training and marginally worse CV, there is a boost to be had through flip TTA.
Augmenting BPP with gaussian noise also improved CV and long range knot visibility.

## Norm
I experimented with pre-norm, post-norm with [Admin](https://arxiv.org/abs/2004.08249) and [ResiDual](https://arxiv.org/abs/2304.14802)-norm. ResiDual gave the best CV as well as the best training stability. Changing from layer to RMS norm gave another slight boost.

## Attention Bias: Positional
Relative positional bias is used in all 12 layers of the encoder during attention (all 12 layers use the same positional bias). Alibi and dynamic positional biases were also tested, but relative performed the best with our architecture. Rotary embedding was also worse than relative. Credits to @lucidrains for the [implementation](https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py#L252).

## GRU
@junseonglee11 added a GRU at the end of each encoder layer which improved perf when we could not scale the attention layers further (the residual output from ResiDual does not interact with the GRU). I included this in my model and experimented with many layouts but it seems like simple is best. I also tried replacing the GRU with a 1D FusedMBConv block, which improved CV but worsened LB.

## Multi-head RNN and LSTM
Similar to the multi-head attention structure, multiple RNN layers can replace single RNN layers by halving their hidden dimension (increasing # of layers and decreasing the hidden dimension offset each other). @junseonglee11 tested 2, 3, 4, 8 heads instead of single RNN layer. Among them, 2 head RNN structures combining a GRU layer and a LSTM layer gave the best performance.

## BPP BMM Convolutional Block
@ayaanjang used ["1D conv + ResidualBPPAttention (using bmm)"](https://www.kaggle.com/code/nyanpn/6th-place-cnn-gcn) to integrate BPP, see the following code:
```
class ResidualBPPAttention(nn.Module):
    def __init__(self, d_model:int, kernel_size:int, dropout:float):
        super().__init__()
        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, src, attn):
        h = self.conv2(self.conv1(torch.bmm(src, attn)))
        return self.relu(src + h)

class Conv(nn.Module):
    def __init__(self, d_in:int, d_out:int, kernel_size:int, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(d_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        return self.dropout(self.relu(self.bn(self.conv(src))))
```
This BPP attention block was added between the FFN and the GRU, also not affecting the residual output from ResiDual. The inclusion of BPP greatly reduced CV.

## Attention Bias: BPP
I used a different method to add BPP. In the first 6 layers of the encoder, BPP is added to the existing positional/mask bias after a convolution (every layer uses the same BPP derived bias). With 6 heads, the single channel BPP is passed to a 1x1 convolution layer that outputs 6 channels. Reducing the number of layers BPP is used in improved CV as well as long range knot visibility (12 -> 6 layers). Reducing the number of BPP heads, adding Contra BPP, using 3x3 conv all yielded no improvements.

## Psuedo Labels: 2 Stage Pipeline
To generate psuedo labels, a 5 fold ensemble was trained with flip (and inferenced with TTA) then filtered by stdev quantile 0.75. Filtered valued have their reactivities set to nan, credits to @ayaanjang for this pipeline. This parquet is then used as train to train a single pretrained model, which is then used to train 5 folds again on the original competition train set. This method from @junseonglee11 is quite effective and efficient time wise. The final stage took under 2 hours for each fold on a single 4090.

## Psuedo Labels: 3 Stage Pipeline
@ayaanjang used a different pipeline as follows: train dataset only -> train + pseudo dataset -> train dataset only. The LRs for the stages are: 2e-3 -> 2e-4 -> 2e-5. The Pseudo labels were also filtered with the same method above, without filtering @ayaanjang found there was no increase in performance. The 3rd stage finetuning serves to reduce overfitting.


## GRU Mystery
Now a mystery... Here is my GRU code:
```
class GRU(nn.Module):
    def __init__(self, d_model: int, p_dropout: float):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = *x.shape[:2], x.size(-1) // 2
        x = x.view(B, L, D, 2).transpose(2, 3).flatten(2)
        x = self.gru(x)[0]
        x = x.view(B, L, 2, D).transpose(2, 3).flatten(2)
        return self.dropout(x)
```
At first I added the transpose and flattern operations thinking that they would delegate each direction in the GRU to half the (3) heads or the other way around, mix them between heads. Upon further inspection I think it does neither... but it noticeably improves performance. If you have a satisfactory explanation please comment!

## Thanks
- @ayaanjang and @junseonglee11 again for their hard work
- @iafoss for his excellent baseline, taught me a lot as this is my first transformer model
- @lucidrains for his various transformer implementations that I referred to excessively
- The devs behind [xformers](https://github.com/facebookresearch/xformers) and [apex](https://github.com/NVIDIA/apex) for greatly speeding up training
- Authors of the numerous papers that helped guide our methods
- All kagglers who participated in discussions with us
- Competition hosts for everything!
