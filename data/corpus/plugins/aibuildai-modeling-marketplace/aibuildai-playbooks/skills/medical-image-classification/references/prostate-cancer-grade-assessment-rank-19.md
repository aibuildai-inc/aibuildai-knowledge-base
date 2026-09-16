# 3rd public/20th private solution-segmentation + simple tiles and multiheaded attention

Competition: prostate-cancer-grade-assessment
Rank: #19
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169437

First, thanks to the organizers @wouterbulten as i understand it is not easy to collect a dataset like this. Second, thanks to my teammates @rvslight @aksell7  @ruozha001  who worked hard with me. Third, congrats to @iafoss for his solo gold and thanks to him for sharing the incredible tile idea and also all the participants who worked hard on this competition.

We suffered in the shakeup, dropping from 3rd to 20th place, but i think our approach is quite interesting, and our selected sub was pretty good and balanced at both public (0.921) and private (0.927) with just 4 models.

First I will briefly describe important details of my method using simple tiles which can generate a 0.927 single model single fold private score. My main idea is to keep things simple, apply attention, and use enough augmentation to avoid overfitting to label noise. My pure pytorch code is released on github at https://github.com/Shujun-He/PANDA (see folder layer1test4maxmeanwuncertainty for the pure pytorch pipeline and I will clearn up and update later). Later, I will detail the segmentation part of our solution. 

Our best private score (not selected) was achieved by ensembling 5 models (2 simple tiles and 3 segmented tiles) and using median avg (middle 3). Best simple tile (given by iafoss' tile function) setting was 36x256x256, and any number above 36 also works.

# Model architecture

Since iafoss released his tile idea, I immediately thought of using attention so the network can learn importance of different tiles and make predictions based on the set of tiles for each WSI.  Here sometimes a full blown transformer encoder layer is used and sometimes just nn.MultiheadAttention + Mish activation. Also, resnext50 proved to be much better in this competition.
. 

Mathematically, each tile becomes a feature vector after passed through the backbone, and the transformer encoder layer just operates on the set of feature vectors. 2D positional encoding can be added here but I did not think it was important based on reading about prostate cancer diagnosis. Here we usually used model=512 and nhead=8 same as the default setting of original transformer paper.

I actually used multitasking learning by adding multiple attention classifiers on top the backbone:

```python
class MultiheadAttentionClassifier(nn.Module):
    def __init__(self,num_classes,out_features,ninp,nhead,dropout,attention_dropout=0.1):
        super(MultiheadAttentionClassifier, self).__init__()
        self.attention=nn.MultiheadAttention(ninp, nhead, dropout=attention_dropout)
        self.classifier=nn.Linear(ninp*2,num_classes)
        self.dropout=nn.Dropout(dropout)
        self.mish=Mish()

    def forward(self,x):
        x=x.permute(1,0,2)
        x,_=self.attention(x,x,x)
        x=self.mish(x)
        x=x.permute(1,0,2)
        max_x,_=torch.max(x,dim=1)
        x=torch.cat([torch.mean(x,dim=1),max_x],dim=-1)
        x=self.dropout(x)
        x=self.classifier(x)
        return x
```


This always resulted in much better CV convergence than just using isup grade, and lb was alway higher than CV so I stuck with the multitasking learning.


# Augmentation

Augmentation wise I use cutout (replacing cutout region with just white pixels) 50% of
the time and the other 50% I change the gamma. The tiles always have 50% chance of being
rotated/ flipped/transposed. In our N=64 runs, I used a new augmentation which I call
whiteout, where I simply turn some tiles white so the model can learn to be invariant to white
tiles. 

```python
def whiteout(tensor,n=6):
    sample_shape=tensor.shape
    to_drop=np.random.choice(tensor.shape[1],size=n,replace=False)
    tensor[:,to_drop]=1
    return tensor
```

Later I found that after whiteout, even when using masked pooling (blocking white tiles), the model gives almost identical results, indicating that our model is invariant to white tiles.

```python
MultiheadAttentionClassifier with masked pooling and masked attention:
class MultiheadAttentionClassifier(nn.Module):
    def __init__(self,num_classes,out_features,ninp,nhead,dropout,nlayers=1,attention_dropout=0.1):
        super(MultiheadAttentionClassifier, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, ninp*2, attention_dropout)
        self.attention = nn.TransformerEncoder(encoder_layers, nlayers)
        self.classifier=nn.Linear(ninp*2,num_classes)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask):
        x=self.dropout(x)
        x=x.permute(1,0,2)
        src_key_padding_mask=mask==0
        x=self.attention(x,src_key_padding_mask=src_key_padding_mask)
        x=x.permute(1,0,2)
        max_x,_=torch.max(x+src_key_padding_mask.unsqueeze(-1)*(-1e-9),dim=1)
        mean_x=torch.sum(x*mask.unsqueeze(-1),dim=1)
        tile_count=torch.sum(mask,dim=1).unsqueeze(-1)
        mean_x=mean_x/tile_count
        x=torch.cat([torch.mean(x,dim=1),max_x],dim=-1)
        x=self.dropout(x)
        x=self.classifier(x)
        return x
```

# Progressive upsampling

One thing that really sped up my training was the usage of progressive upsampling. Training is
usually 45 epochs with first ten epochs on half resolution tiles (downsized with cv2.resize). At
25 and 36 epochs, learning rate is reduced 10 times. This is a cool idea for people with limited computing power and for people who have a lot, it speed up training even more.

# Segmentation model

To be updated. But to put it simply, we basically used masks on lowest resolution images to train a segmentation model distinguishing if the particular tile has cancer in it or not. Subsequently tiles were selected based on which ones are more likely to contain cancer based on trained segmentation model. This method should be better at predicting class 2,3,4,5, which was the case in CV at least. Somehow this method worked not so well in the private test set; however, ensembling this method with my models which use simple tiles still gave a boost.

Combining the segmentation tiles with simple tiles worked well in public and also in private (just not as much as the boost in private given by denoising). Based on lb and cv, we thought that segmentation tiles would have better performance on 2,3,4,5 while simple tiles would be better at 0, 1, so the combination logically made sense.

I had some worries that this method may be too biased towards predicting cancer, which is probably the reason it did not work well in private (judging from single model scores of segmentation tiles). Surprisingly, in private test set, when we made a mistake, where we used simple tiles instead of segmentation tiles on a model trained on segmentation tiles, we received a higher private lb in that submission than using segmentation tiles, which was of course not submitted since we identified that error.

# Conclusion

In the end, we had multiple moments where we selected a 0.932 run, which would have resulted in a gold medal rather than a high silver. However, we changed it based on some reasoning that i still don't think is wrong. So just unlucky. 

About top solutions, I see most of them using some type of denoising method or just getting lucky based on some public kernels. Of course, using a large ensemble (~10 models) helps as well. What is really surprising to me is how denoising did not bring any recognizable improvement on public lb. I cannot help but think that there is some unintended difference between public and test set, because there is no reason denoising shouldn't work for public lb. In fact, I tried to do some denoising, but the results were not convincing and I stopped, which I do not consider a mistake, because there was no way to validate that anyone's denoising method was indeed working properly.
