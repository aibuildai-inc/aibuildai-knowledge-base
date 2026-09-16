# 3rd place solution

Competition: gendered-pronoun-resolution
Rank: #3
Source: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90424#latest-533098

Many thanks to google and to kaggle for the organization of the competition, and to the awesome community here - @kashnitsky, @mateiionita, @ceshine, @kenkrige and all others.

The kernel by Ceshine Lee was a very good starting point - I played with it, using bert-large-cased, added 1d-conv layer with kernel 1, spanextractor, and the best result was around 0.48 on the test_stage1 data. As others, later I discovered that intermediate layers of BERT encoder contain better representation for the model, for my architecture in particular [-5] and [-6]. This was bringing  ~0.41 depending on the seed. Afterwards I started adding the features, like syntactic distance between mentions, token distance and so on. It helped a bit, but not much. I was stuck, and at this point finally the guys from ontonotes replied to me, so I started using the external data.

### External data
Initially I though that the data will decide everything in this compeition, so while the models were running I put some efforts putting winobias, winogender, dpr and later ontonotes into gap-format. 1st place solution shows that more data is not always better performance =) Anyway I had the data and wasn't quite sure what to do with them.

At first I simply put all external datasets into one big dataset, split it to train and validation and tried to finetune both the bert and the head on it. The idea was to use the learned weights as initalization weights for the 'normal' gap training, where normal is using ~2400 gap datapoints with 5 fold CV. It didn't work, score became worse. Then I only trained a head for a 2 epochs without tuning the bert and it helped to come from ~0.4 to ~0.36. So I had the observation that better initialization of the head weights lead to better minima. I decided to work a bit on this idea. I started to use winobias as validation data, thus minimizing the gender bias. This brought somewhere around 0.35. Then (it was 2 days before the end of the competition) I tried training the head, then freeze the head and fine-tune the bert weights (only one layer either [-5] or [-6]) with very small learning rate. It had some positive effect and put me around 0.33-34. 

Afterwards I had no time for interesting stuff anymore, so I started with ensembling - I run 4 models overall, bert-large-cased with [-5] layer, bert-large-uncased[-5], bert-large-cased[-6] and bert-large-cased[-6]. The final prediction is just the average of all models with clipping at 1e-2.  

### Final Architecture
Here is the final overview of the architecture I used:
BERT -&gt; Conv1d with kernel=1 (1024x64) -&gt; SelfAttentiveSpanExtractor(64) * 3 -&gt; BatchNorm1d() -&gt; fully connected (64*3 x 64) -&gt; dropout (0.6) -&gt; fully connected (64 x 81) (here all the manual features come in) -&gt; fully connected (81 x 3).
