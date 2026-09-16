# 30th place solution

Competition: commonlit-evaluate-student-summaries
Rank: #30
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/447325

My thank you to Kaggle, the competition hosts, and the Kaggle community for a fun and educational three months. I'm really looking forward to reading everyone's solutions and learning from them. The 30th ranked submission was an ensemble of 6 deberta-v3-large models.

Throughout this competition, I worried about overfitting. My earliest taste of overfitting was when I reached a gold position briefly with 0.461 public LB score, by not splitting on prompt-id, and by doing a random 80/20 train/val split. After spending a a day being thrilled at the gold symbol in my public LB rank, I started thinking that it was all a mirage, and I was overfitting (and indeed the private LB score of that submission confirms this). Back then the model did not even train if I split by prompt-id. It was some more weeks before I learned to train transformer models in a stable way, with a somewhat steadily decreasing validation loss. I did not use many of the features from the top-scoring public notebooks, because I feared overfitting. I don't know if it was a good decision, but the shakedown was only by 5 places, and I don't regret it.


**Things that worked:**
- Training seperately for content & wording scores. I trained 8 models (4 per target, since there were 4 folds) for each "full model". For CV, I averaged out-of-fold predictions. For predicting on test data, I averaged the 4 models for each target. 
- It was 3 levels of modeling: 
1) deberta-v3-large models with many differences (pooling techniques, different inputs, frozen vs unfrozen, different max-len, etc)
2) Make LGBM predictions from each deberta's output, plus features like text length etc. The features list was the same for all models.
3) Both deberta and LGBM outputs of 6 models was fed into hill climbing algorithm (with negative weights) to predict the final output. Hill climbing, being a greedy algorithm, was sensitive to the order of inputs, and I tried to find alternatives but none of them worked as well.

- Layerwise learning rate decay - stabilizing transformer training using this method, with linear or cosine scheduler with warmup was the foundation of all the models. 
- Using the input text after replacing common-ngrams with prompt text with the "_" character. The inputs to most of the models was prompt_question + separator + prompt_title + [start] + modified input text + [end]. One model uses a modified prompt_text appended at the end -- I've explained that below. Another model had [P] prepended to each sentence in summary_text.
- Taking the encoder output of only some tokens, like [start] and [end], or [start] to [end], or gem pool of only [P] tokens.
- weighted layer pooling (all layers), gem pooling & mean max pooling
- The head was the same for all models: 2 linear layers with tanh inbetween.
- freezing embeddings & layers. I froze between 12 and 18 layers, not including embeddings layer, which was frozen for all models.
- pseudo labeling: I did it by training T5 on competition data, using it to generate texts, which I labeled using my best ensemble, then trained fresh models first on PL data, then on competition data. The first time I did it, it was leaky & LB score tanked. Second implemention gave a small LB boost (2-model ensemble score went from 0.431 to 0.430), but I don't know if it was worth all the trouble.
- mixed precision training seemed to give results that were as good as full precision training, while being much faster
- I included prompt text like this: Gather sentences from prompt_text that are quoted by at least one student, who had written a summary that was at least 20% original (as calculated by ngram matching). The ensemble of 6 models included one model that did this. That model's private LB score was 0.48 (public LB was 0.443). I don't know how useful it was. I was just happy to have found a way to include prompt_text.
- Hill climbing


**Things that did not work:**
- Re-initializing transformer layers
- AWP
- adversarial attack through SIFT
- MLM
- SWA (stochastic weight averaging - maybe my implementation was wrong)
- ridge regression or xgb for ensembling
- auxiliary loss calculated for classifying the summary into discretized score buckets
- RAPIDS SVR using embeddings of various models (without training on competition data)
- many features I'd calculated, such as mean sentence transformer cosine similarity scores (sentence-wise) between prompt_text and summary_text, improved CV score but made LB score much worse
- Trying to make vicuna-13b generate various features (including "gold" summaries) that deberta or LGBM could use did not work. Making chatgpt guess the individual rubric scores that must have gone into the calculation of wording & content also did not work.
- I felt that if a summary provided on-topic, original insight, it was scored higher. I spent a lot of energy in trying to predict this using various pretrained models. I also generated lots of chatgpt or llama-generated data for fine tuning deberta for this purpose. It did not work at all. I think I spent the first month mostly on this pursuit & in making vicuna-13b useful. By doing this I learned a lot about the datasets that are used for benchmarking LLMs, about finetuning llama and sentence transformers, etc. So maybe it was useful (I doubt it lol). 
- One alarming thing that did not work till the end was Clipping grad norm. The model stopped training if I clipped.
- Using a model trained on previous feedback competition data to guess syntax, cohesion, vocabulary, etc scores of competition data didn't work. But I tried it early on, and back then I didn't know that much about various transformer techniques that I learned later. So maybe it would've worked if I'd tried it near the end.
- I picked only 2 submissions because I was going to pick one submission that I'd made last minute, and it errored out, and I forgot to pick an alternative.
- I wish I'd trusted the CV score all along. It correlated the best with private LB.


**Credits & Gratitude:**
- I learned about LLRD from Vignesh in Sanyam's interview: https://www.youtube.com/watch?v=XLaq-boyAGk and then by reading @torch's wonderful notebooks - https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning
- Learned about pooling techniques from https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook
- Learned about GeM pool from Team hydrogen's notebooks ( https://www.kaggle.com/code/philippsinger/team-hydrogen-efficiency-prize-1st-place )
- Nvidia's videos with Chris Deotte, Christof Henkel, Jean-Francois Puget and Ahmet Erdem were super helpful for me. I learned how to do leak-proof PL (and other things) from them. Despite this, my first implemention of PL was leaky, but I was able to fix it because frankly, I learned from the best: https://www.youtube.com/watch?v=CwDKy0EKwHE and https://www.youtube.com/watch?v=PXc_SlnT2g0 
- Learned about hill climbing from https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609


**General Educational Material:**
I started learning ML in april, and until this competition, I was very intimidated by kaggle. If someone else like me is reading this, I highly recommend these courses. I found all them very helpful and educational: 
- Andrew Ng's introductory course https://www.coursera.org/specializations/machine-learning-introduction
- Fast.ai (book and videos) https://www.fast.ai/
- Karpathy's Zero to Hero course https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
- Kaggle Learn https://www.kaggle.com/learn
- Kaggle Playground Competitions
- Aladdin Pearson's youtube channel https://www.youtube.com/@AladdinPersson : He implements papers like UNet, Gans etc, and it makes the papers & code very approchable. 
- I found it very helpful to use weights & biases for tracking things and highly recommend them, especially for beginners like me.
- I used paperspace & jarvislabs.ai for training.
