# 14th place solution

Competition: waveform-inversion
Rank: #14
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587529

Thanks Kaggle and organizer for hosting this competition, this is the most interesting competition that I ever entered. And also great thanks to people who sharing ideas and codes, especially:
@brendanartley for modeling ideas
@jaewook704 @manatoyo for forward simulation code
@bguberfain for starter notebook

**Data Processing**
1. sign(x)*log(1+|x|)
2. reorder each receiver based on CMP(central middle point) wrt source along receiver dimension to make input better aligned with target. To keep receiver dimension 70, odd and even indices are split into two channels. (5,1000,70)->(10,1000,70) 
(-18MAE when ~160MAE  on 7k samples)
3. add (x,y) coordinate embedding as extra channels since FWI task is spatial variant

**Model**
1. I stack multiple U-net to mimic common iterative methods in physics / math. Here each U-net works as a single iteration step with a global receptive field. Scaling depth by stacking more U-nets works better than just scaling width or increase number of conv layers.
2. Based on my experiment on 40k samples, larger model always lead to better result. The largest model I can train : stack 5 U-nets with depth 4 channel dim 128. Not sure how much further gain is possible by continue scaling up.
3. I tried Convnext (using Bartley's model or replace 1st U-net in my model) without success and I don’t have enough computation resources to train CAFormer to fully converge so I end up without using any pretrained backbone.
4. Other details: few step stride conv to downsample input data; intermediate conv layer(from Bartely work); down by avg_pool up by bilinear; batch norm (consistent better than other type of normalization once converged ); skip connect

**Loss**
1. MAE with smaller weight to deeper positions (1~1/4 linearly) since I suspect it be more noisy (84->81 MAE on 10k)

**Augmentation**
1. Symmetric augmentation (-10 MAE when ~110 on 10k samples)
  - source is placed at [0, 17, 34, 52, 69], 34 after flip will be 35 this causes conflict in feature meaning. So I insert one more channel to represent source at 35 and fill by zero. Feature meaning is then self consistent before and after flip, though model still need to be trained to learn to handle this.
  - used as TTA: 15.28->14.71
2. Velocity map scaling(-10 MAE when ~100 on 10k samples)
  - Scale target velocity map by *(1+alpha) then compress seis data along time axis to /(1+alpha) 
  - After play with forward simulation code I find scale velocity also affect sesi data maginitue.I can’t track this in closed form so I use an empirical rule: *(1+alpha)^0.26. Based on validation this only has minor effects if any.
  - Here we are actually scaling time, this causes the augmented dataset with different source freq, so this doesn’t generate data with the same distribution as OpenFWI but luckily it still helps.

**Reconstruction Error Optimization**
1. notation: G - true inverse model, F - true forward model, M - our NN inverse model
2. here I minimize ||F(M(x_k)) - x ||^2 wrt x_k start from x rather than change model M
3. assume M is already a good estimation of G in terms of local change:  F(M(x+dx))-F(M(x))~=F(G(x+dx))-F(G(x))=dx then we can use simple iteration rule to reduce error without gradient of F:
x_k=x_k-lambda*(F(M(x_k)) - x ) 
that is we can manipulate x directly with change in y since f here close to identity map
4. When used together with TTA: M(x):=(M(x)+Flip(M(Flip(x))))/2
5. performance:(each iter costs around 1.5h for whole test)
17.17->14.45 (lambda0.85, 1 iters) 
17.17->13.49 (lambda0.7, 3 iters) 
14.71->12.34 (lambda 0.6, 5 iters) for model already finetuned with data generate by such iterative optimization

**Training stages(speed performance reported on single 4080s)**
1. 100 epoch on OpenFWI, 12 days 
  - AdamW, weight decay 1e-4, batch size 28, init lr 28/64*3e-4, 80% init lr+ 20% cos lr, EMA in last epoch(only minor diff, droped in later stages)
  - predict test set and run forward simulation to generate extra data, only symmetric TTA used in this stage
  - val/LB: 21.0/22.4(sym TTA)
2. 6+20 epoch on OpenFWI + *4 copy of stage 1 generated data, 6 days 
  - after first 6 epoch my computer restarted so I continue training from it
  - only cos lr is used without init constant phase, init lr 28/64*1.5e-4
  - val/LB: 17.17/???  (sym TTA)
     val/LB: 14.45/14.9 (sym TTA+lambda0.85, 1 iter)
     val/LB: 13.49/???  (sym TTA+lambda0.7, 3 iters)
  - predict test and run forward simulation to generate extra data (sym TTA+lambda0.85, 2 iters)
3. 10 epoch on OpenFWI + *10 copy of stage 2 generated data, 3 days 
  - only cos lr is used without init constant phase, init lr 28/64*1.5e-4
  - val/LB: 14.71/???  (sym TTA)
   val/LB: 12.34/12.7 (sym TTA+lambda0.6, 5 iters)
