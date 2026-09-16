# 55th place solution – Shortformer + Sliding Window + Topic-dependent Postprocessing

Competition: feedback-prize-2021
Rank: #55
Source: https://www.kaggle.com/c/feedback-prize-2021/discussion/313229

## My approach – Shortformer + Sliding Window Training / Inference + Topic-dependent Postprocessing
First of all, a huge thank you to the competition hosts and Kaggle for organizing this competition.
I can't wait to read all the other write-ups in detail.

My approach is a weighted ensemble of different BERT backbones and topic-dependent postprocessing. To enable training Shortformer-like models for this competition, I use a sliding window approach. Although I only achieved 55th place, I hope to still share and discuss some new ideas with you.

**Sliding Window Training / Inference**
By default, Shortformer-like models can’t process long documents. To overcome this issue, I split long essays into chunks with a left and right context. Then I only make predictions for the central part of the chunk, but not for the context window. [[flert-paper.png]](https://postimg.cc/V0XxRq3P) This approach was inspired by this [paper](https://arxiv.org/pdf/2011.06993.pdf) and improved my single models by 0.005+ compared to applying normal stride.

**Training Setup and Hyperparamters**
For training, I use learning rate of 1e-5, batch size of 4 (with gradient accumulation), linear/cosine lr schedule with warmup of 10%, BIO-label scheme and 8-bit Adam.

Furthermore, I tried to apply multi-sample dropout, data augmentation with [MASK], gradually (un)freezing layers, (grouped) discriminative learning rate, BIEO-label scheme, and within-task-pretraining. However, all these techniques did not lead to significant improvements.

**Error Analysis**
Improving deep learning systems is a highly iterative process. It is important to perform error analysis to identify areas to improve frequently. I wrote a function that takes the raw submission.csv, the post-processed submission.csv, and the ground truth train.csv and displays them side-by-side while ranking them from worst-to-best/best-to-worst overall or discourse-dependent performance. This visualization helped me quite a lot throughout this competition. [[error-analysis.png]](https://postimg.cc/BjMQG3tP)

**Models**
DebertaV3-Large
5 folds, 512 max length, 128+128 context window, public/private LB 0.698/0.709, CV 0.697

Funnel-Large
5 folds, 512 max length, 128+128 context window, public/private LB 0.692/0.704, CV 0.92

Roberta-Large
5 folds, 512 max length, 128+128 context window, public/private LB 0.684/0.695, CV 0.682

Longformer-Large
5 folds, 1536 max length, no context window, public/private LB 0.688/0.699, CV 0.686

**Ensamble**
I built a weighted ensemble of DebertaV3 (0.4788), Funnel Transformer (0.3192), Longformer (0.152), and Roberta (0.05) for submission by applying the ensemble pseudo code from this [discussion](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175344).

Using weighted ensemble improved private/public LB 0.702/0.714 compared to public/private LB 0.700/0.712 using average ensemble.

**Topic-dependent Postprocessing**
In the early days of the competition, I discovered 15 essay topics in the dataset. Later on this was also revealed in a [discussion post](https://www.kaggle.com/c/feedback-prize-2021/discussion/301481) and [notebook](https://www.kaggle.com/cdeotte/rapids-umap-tfidf-kmeans-discovers-15-topics) by Chris.

Using BERTopic, I quickly came up with a simple model that allowed me to predict the topic for a specific essay with over 99.5% accuracy. With this model, I can apply topic-dependent postprocessing, i.e. tuning individual `length_threshold` and `proba_threshold` dictionaries for each topic, rather than having two global dictionaries. However, I think my training/tuning setup was suboptimal for this postprocessing since I only consider the discourse, but not the topic in `StratifiedKFold`. With the right CV strategy, the score probably could have been further improved.

Even with my CV strategy, topic-dependent postprocessing still boosted my score slightly (public/private LB 0.706/0.716).

I also tried to add a prompt at the start of the essay, e.g. [[prompt.png]](https://postimg.cc/K3BB9jGM) Unfortunately, this did not lead to any significant improvements. That is because topic information is already present in the individual word(piece) embeddings based on topic-specific words that appear in an essay.

**Additional Trick: Replace \n by `<newline>` token**
The available public notebooks showed that \n is an important feature. However, \n is handled differently by different tokenizers (Some split ‘\n’ into ‘\’ and ’n’, some don’t split ‘\n’, some map it to `<UNK>`, etc.). That is why I mapped all \n to a `<newline>` token and added this token as a new special token to the tokenizer during initialization.

**Hardware Resources**
I have access to Kaggle + Colab.

**Final toughts**
In conclusion, FeedbackPrize was a fun competition.

However, I am still skeptical about using Token Classification models for this competition. That is because the naive decoding procedure (using softmax) feels similar to applying a Greedy Search to Neural Machine Translation, which may lead to suboptimal output sequences, even if the models were trained correctly. As an example, it was common for my prototype models to make predictions like this: 
[[suboptimal-output.png]](https://postimg.cc/G9GDnBRB) 
I was experimenting with a Conditional Random Field decoder because we can add constraints to such a model, that can prevent these results to occur. But I was unable to beat postprocessing using `length_threshold` and `proba_threshold` dictionaries. 

More research has to be done in this direction😊
