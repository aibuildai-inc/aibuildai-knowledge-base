# #1 LB Ideas

Competition: tabular-playground-series-mar-2021
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-mar-2021/discussion/229833

Pretty much everything I have applied here is from learning from the Jan/Feb competitions so first thing to say is please check those comps' winning solutions as well as a lot of other great notebooks from the last 3 months on improving individual models! It has been great that Kaggle has hosted these and so many people have shared great ideas. 👍

We know that Jan/Feb was won by DAE approach and I made use of @ryanzhang code in pytorch this month. I think I already shared starter notebook which was my own starting point for this month.
https://www.kaggle.com/davidedwards1/tabularmarch21-dae-starter
Original @ryanzhang code
https://github.com/ryancheunggit/Denoise-Transformer-AutoEncoder

#### Folds
Stratified kfold, 10 folds

#### Tree Models
Ran 2 optuna cycles each for XGB and LGBM, the first to narrow down parameter ranges a bit. I did not spend all that much time on this as I just wanted to take the top 5 optuna models from each and move along to DAE. To provide a little diversity, for the XGB I used one-hot encoding and gave optuna the option to drop low-count columns (e.g. don't include any columns with sum < 40).

I also tried LGBM with one-hot columns and the option to drop some low count categorical columns, but results seemed worse than just using original columns and label encoder, so I didn't see this adding much.

#### DAE
In total had enough time for around 27 runs x 600 epochs each, and each run was assessed by running CV with the output features from the end of each trial. So I can only share results of my limited testing. I'm sure there's plenty I missed so I look forward to learning as well from anything others are willing to share post-competition.

**What Seemed To Work**
In the end I used 3 runs of the DAE for inference and blend, with one of them I took weights from a couple of different epochs, all of them had in common:
Learning rate 3e-04 to start and decreasing.
Around 50% flat noise to start. Edit: by flat, I mean the same across all columns, not over time.
Around 1200-1800 epochs total. I extended one DAE to 2400 on final 2 days but this provided very very small improvement.

**Reducing the Noise Level by Epoch**
All DAEs used in submission - updated the SwapNoiseMasker every epoch to decay the noise level. This was taking an idea from January winning solution 
https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216037
Which was to conduct a second set of training with lower noise. Used a slower decay than for the learning rate as it seemed like the model stopped improving with noise below a certain level. E.g. learning rate decay = 0.998, noise decay = 0.999, so 50% noise after 600 epochs drops to around 27%.

**All Categorical**
One run - used an idea from the notebook here 
https://www.kaggle.com/siavrez/kerasembeddings
'Binned' the continuous variables therefore providing a 100% categorical input feed (I used Kmeans, but I think the result is similar, I didn't spend time analysing). The results were slightly worse than the mixed categorical/continuous but I think this provided some diversity. If I'd had more time I'd have liked to experiment more with this as it reduces the complexity & losses needing to be combined and I suspect it could then be improved more.

**What Didn't Seem To Work**
Things which I couldn't get to show any improvement from very limited testing:
- Larger batch size
- Smaller embed dim (1 run at a larger dimension did not really show improvement either)
- Different loss and emphasis weights
- Changing emphasis over time (if not sure re emphasis - see DAE notebook and code)
- Random noise (at best a small amount seemed to make no difference)
- Lower noise
- Much higher noise
- Vary noise with cardinality (in either direction) - I think that higher noise on lower cardinality looked more promising but at best it looked slightly worse than flat noise in 1 run.
- Lower start learning rate, and/or flat learning rate

#### DAE/MLP Inference
Used the outline in my notebook here as a starting point.
https://www.kaggle.com/davidedwards1/tabularmarch21-dae-starter-cv-inference
Playing around with the MLP dropout, hidden size etc may have provided slightly different outcomes for blend but at best only fractionally improved single model CV for me. 10 fold was definitely better than 5 fold.

Additions to blend (to improve results & diversity - in private notebooks)

**Variation 1**
Rather than DAE > generate all fixed MLP intput > dataloader > MLP, also tried:
Train data > dataloader > low noise % > DAE feature > MLP (i.e. direct from source data through both neural networks, rather than saving features). And reduced the noise each epoch.

**Variation 2**
Tried including the X reconstruction and mask prediction in the DAE feature output and this seemed to improve CV as well as giving the best single model score I have (priv / public = 0.89964 / 0.89482). I barely had time to check this and suspect I probably could have improved a few things.

#### Blend
Used optuna and out of fold predictions to arrive at some weights for each model. 

For my second submission I capped the max weight per model, in case Optuna mix fitted CV better than test. However this performed slightly worse on both public and private LB, so it seemed like for this dataset the best blend calculated for CV was pretty well aligned with the best blend for test predictions.

#### Notebooks
My GPU quota is running a little low for this week and I need to tidy the files but will try to rerun and share the best Colab DAE / inference run on Kaggle with the settings which I used early next week in case they are of interest.

Blend notebook is here
https://www.kaggle.com/davidedwards1/tabmar21-tabular-blend-final-sub/output
The input datasets are private, but the individual model submission and OOF predictions are available in the blend notebook (version 2) output files if anyone is interested in the individual model predictions.
