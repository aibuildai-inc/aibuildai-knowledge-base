# 10th place solution

Competition: asl-signs
Rank: #10
Source: https://www.kaggle.com/c/asl-signs/discussion/406434

>First, I would like to thank the Armed Forces of Ukraine, Security Service of Ukraine, Defence Intelligence of Ukraine, State Emergency Service of Ukraine for providing safety and security to participate in this great competition, complete this work, and help science, technology, and business not stop and move forward.

TLDR: Transformer models with embedding based on Linear layers for pose,lips,eye, and STGCN + Linear layer for target hand

**Preprocessing**
Find target hand using nan in time dimension, flip if need, select poses where target hand is not nan, resize to 16 in time dimension, and finally normalize with
 ```python
xyz = (xyz - np.nanmean(xyz[:,FACE,:],axis(0,1))) / np.nanstd(xyz[:,FACE,:],axis=(0,1))
```
**Modeling**
(All validation scores are based on @hengck23 split on participant id)
After a stuck at 0.75 lb (~0.7 validation), I understand that I need to change strategy.I go back and start modelling only with target hand. After finding best normalization strategy using face I get ~0.69 val score with only hand.
Then I tried to add pose,lips,eyes I get ~0.71 val score. So for me was clear that most of the signal in dominant hand, so i tried to get different representation and ensemble it. Firstly I try to plot  target hand and then run 2.5d CNN or TransformerCNN based models, and get with this 0.6 val score only with hand. Then I tried to run  STGCN(Spatio-Temporal Graph Convolutional Networks) for target hand and get ~0.66 score. Because CNN gets too much time to process I decide to drop it. So finally after ensembling pose,lip,eye,hand,STGCN(hand) -> attention block i get ~0.72 score
```python
x = torch.cat([
            self.pose_emb(x[:,:,:8,:].view(B,L,-1), x_mask)[0],
            self.lip_emb(x[:,:,8:20+8,:].view(B,L,-1), x_mask)[0],
            self.reye_emb(x[:, :, 20+8       :20+8+16, :].view(B, L, -1), x_mask)[0],
            self.leye_emb(x[:, :, 20+8+16    :20+8+16+16, :].view(B, L, -1), x_mask)[0],
            self.hand_emb(x[:, :, 20+8+16+16 :20+8+16+16+21, :].view(B, L, -1), x_mask)[0],
            self.graph_emb(x[:, :,20+8+16+16:20+8+16+16+21, :].permute(0, 3, 1, 2).unsqueeze(-1)),
            self.pos_embed[:L, :].unsqueeze(0).repeat(B,1,1)],axis=-1)
```
Example how i ensemble it.

Then I started to add different augmentations (Here i have to spend definitely more time to increase my score even more), Most of the boost i get from :
* Mixup
```python
if np.random.random() < self.args.mixup:
                    indices = torch.randperm(batch["xyz"].size(0), device=batch["xyz"].device, dtype=torch.long)
                    beta = np.random.beta(0.2, 0.2)
                    batch["xyz"] = beta * batch["xyz"] + (1 - beta) * batch["xyz"][indices]
                    batch = self.run_logits(batch)
                    batch["loss"] = beta * self.args.loss(logits=batch["logits"], labels=batch["labels"]) + (
                                1 - beta) * self.args.loss(logits=batch["logits"],
                                                           labels=batch["labels"][indices])
```
* Copy paste
```python
if np.random.random() < self.args.cutout:
                indices_to_shuffle = torch.randperm(batch["xyz"].size(0), device=batch["xyz"].device, dtype=torch.long)
                cutout_len = self.args.cutout_len
                if type(cutout_len) == list:
                    cutout_len = random.uniform(*cutout_len)
                sz = int(batch["xyz"].size(1) * cutout_len)
                ind = np.random.randint(0, batch["xyz"].size(1) - sz)
                indices_to_cut = torch.range(ind, ind+sz, device=batch["xyz"].device, dtype=torch.long)
                beta = 1 - cutout_len
                batch["xyz"][:, indices_to_cut] = batch["xyz"][indices_to_shuffle][:,indices_to_cut]
                batch["mask"][:, indices_to_cut] = batch["mask"][indices_to_shuffle][:,indices_to_cut]

                batch = self.run_logits(batch)
                batch["loss"] = beta * self.args.loss(logits=batch["logits"], labels=batch["labels"]) + (
                        1 - beta) * self.args.loss(logits=batch["logits"],
                                                   labels=batch["labels"][indices_to_shuffle])
```

After that i get final score ~0.735-0.74 val score and 0.77 lb single model, 0.79 lb ensemble

**What didn't work :**
 * CNN on dominant hand, didn't get a lot of boost from it.
 * Different losses based on embedding for words.
 * Funny idea: Train model with arcface label : (pariticipant_id+sign nearly 5230 labels for full train) , get improvement in val score ~0.02 for val split based on label, because in lb we have samples from train too, i tried to add it to ensemble, but dont get any significant boost so decided to drop it.

One more time want to say thanks for every one who support Ukraine in such a hard times.
Slava Ukraini.
