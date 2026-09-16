# 3rd solution

Competition: learning-equality-curriculum-recommendations
Rank: #3
Source: https://www.kaggle.com/c/learning-equality-curriculum-recommendations/discussion/394838

First of all, I would like to thank the organizers for hosting this high-quality competion, and my awesome teammates @xiamaozi11 @syzong @sayoulala @yzheng21, we all worked hard for this competion. And I learned a lot from the great notebooks and discussions, basically all the methods we used are from the kaggle community. Thanks to these generous and smart kagglers!

- tips from hosts: https://www.kaggle.com/code/jamiealexandre/tips-and-recommendations-from-hosts
- text pre-processing: https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/376873
- stage1 and stage2 train and submit pipeline: https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/373640
- stage1 and stage2 modeling: https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/381509, https://www.kaggle.com/code/ragnar123/lecr-xlm-roberta-base-baseline

### Summary

- CV strategy
- Stage1: Retriever
- Stage2: Ranker
- Finding threshold
- Post-Processing
- Ensemble

### training pipeline


### CV strategy

We only used 4,000 random topics which **category != 'source'** as hold out data. Those topics were as the validation data and never used in any training process. This simple CV strategy was unexpectedly stable. 

In the last month of the competition, we changed 4,000 topics to 1,000, which was still relatively consistent until we started to ensemble.

### Retriever

We used unsupervised SIMCSE (Simple Contrastive Learning of Sentence Embeddings: https://github.com/princeton-nlp/SimCSE) method for training retriever models.

#### training retriever

- only used positive samples from correlations.csv for unsupervised simcse training
- random choice 100 negative samples per validation topic from same language, for validation set 
- content text format: `title [SEP] kind [SEP] description [SED] text`, maxlen =  256 (string level)
- topic text format: `title [SEP] channel [SEP] category [SEP] level [SEP] language [SEP] description [SEP] context [SEP] parent_description [SEP] children_description`, maxlen = 256 (string level)
- `simcse_unsup_loss`

> ```
> def simcse_unsup_loss(feature_topic, feature_content) -> 'tensor':
    y_true = torch.arange(0, feature_topic.size(0), device=device)
    sim = F.cosine_similarity(feature_topic.unsqueeze(1), feature_content.unsqueeze(0), dim=2)
    sim = sim / 0.05
    loss = F.cross_entropy(sim, y_true)
    loss = torch.mean(loss)
    return loss
> ```
> from: https://github.com/yangjianxin1/SimCSE/blob/master/model.py

- train code like:

>```
>for step, (inputs_topic, inputs_content, labels) in enumerate(train_loader):
        inputs_topic = collate(inputs_topic)
        for k, v in inputs_topic.items():
            inputs_topic[k] = v.to(device)
        inputs_content = collate(inputs_content)
        for k, v in inputs_content.items():
            inputs_content[k] = v.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            feature_topic = model(inputs_topic)
            feature_content = model(inputs_content)
            loss = simcse_unsup_loss(feature_topic, feature_content)
>```

Performance on 1,000 topics validation data:

| model                                 | F2@5   | max positive score top50 | max positive score top100 |
|---------------------------------------|--------|--------------------------|---------------------------|
| paraphrase-multilingual-mpnet-base-v2 | 0.5250 | 0.9135                   | 0.9443                    |
| all-MiniLM-L6-v2                      | 0.4879 | 0.9045                   | 0.9353                    |
| mdeberta-v3-base                      | 0.4689 | 0.8938                   | 0.9187                    |

#### recall

We didn't use KNN to clusting, simply calculate cosine similarity for each topic and all content samples of the topic's language and then choose topN samples.

We also tested the retriever ensemble (weighted cosine similarity). Although the max positive score top50 score has been improved to 0.9235, but there is basically no change on the LB score. So we only used a single retriever model (paraphrase-multilingual-mpnet-base-v2) in the final submit.

### Ranker

The ranker in stage2 basically is a binary classification model.

We used our best simcse finetuned model (paraphrase-multilingual-mpnet-base-v2) to infer on train set topics, calculate cosine similarity for each topic and all content samples of the topic's language and then choose top100 samples. We also added all positive samples from correlations.csv.

texts were prepared as same as stage1. Pair format: `content [SEP] topic`, maxlen = 256 (token level)

Hard negative samples from retriever model can greatly improve the performance of ranker models.

| retrieve model (max positive score top100) | ranker f2 score (LB) |
|--------------------------------------------|----------------------|
| 0.80                                       | 0.585                |
| 0.94                                       | 0.688                |

We have used two model initialization methods. One is to directly load the model weight from huggingface, and the other is to load the model weight after simcse finetuning. The performance of the two methods is basically the same, while the latter is slightly higher and can converge faster.

We also used FGM, EMA on training. FGM+EMA can impove score by 0.01. 

| model                                                          | validation (1,000 topics) | LB score | PB score |
|----------------------------------------------------------------|---------------------------|----------|----------|
| mdeberta-v3-base (loading simcse weights)                      | 0.7149                    | 0.688    | 0.727    |
| mdeberta-v3-base                                               | 0.6378                    | 0.669    | 0.693    |
| xlm-roberta-large (loading simcse weights)                     | 0.6987                    | -    |-    |
| xlm-roberta-base (loading simcse weights)                      | 0.6780                    | -        | -        |
| paraphrase-multilingual-mpnet-base-v2 (loading simcse weights) | 0.6299                    | -        | -        |

### Finding threshold

We set the threshold in loop to calculate the f2 metric on the 1,000 topics validation data, codes like:

```
best_thres = 0.
best_score = 0.
best_n_rec = 10
for thres in tqdm(np.arange(0.01, 0.2, 0.005)):
    for n_rec in range(30, 50):
        test_sub = test_data[test_data['score'] >= thres].reset_index(drop=True)
        sub_df = test_sub.groupby('topic_id').apply(lambda g: g.head(n_rec)).reset_index(drop=True)
        score = calc_f2(sub_df, label_df)
        if score > best_score:
            best_score = score
            best_thres = thres
            best_n_rec = n_rec
```

When submitting a single model, this method basically CV-LB consistency (CV is about 0.02-0.03 higher than LB).

But When it came to the last two weeks of the competition, when started to ensemble, we lost the CV-LB consistency, and I think the reason may be that 1,000 topics validation data is not big enough.

### Post-Processing

When dividing the threshold, we will have a small number of topics that do not match any contents. We just simply using top4 contents ranked by the original scores.

We tried recalling more contents for this part of topics, but LB score didn't improve.

We also tried different languages using different threshold, both CV and LB score dropped a little.

### Ensemble

We had trained 20+ ranker models, trained on different number of recall samples per topic, like 50, 70, 100. 

- mdeberta (simcse weights, 4,000 validate topics)
- mdeberta (simcse weights, 4,000 validate topics, with FGM,EMA)
- mdeberta (simcse weights, 1,000 validate topics)
- mdeberta (simcse weights, 1,000 validate topics, with FGM,EMA)
- mdeberta (1,000 validate topics, with FGM,EMA)
- xlm-roberta-large (simcse weights, 1,000 validate topics, with FGM,EMA)
- xlm-roberta-base (simcse weights, 1,000 validate topics, with FGM,EMA)

We used LinearRegression to fit on 1,000 topics validation model output score to get coef_ array, and then used as blending weights:

```
pcols = [c for c in valid_data.columns if c.startswith('score')]
for cols in tqdm([i for i in combinations(pcols, 10)]):
    cols = list(cols)
    X = valid_data[cols].values
    y = valid_data['label'].values
    lr = LinearRegression().fit(X, y)
    coef = lr.coef_
    print(get_score(valid_data, df_target_metric, cols, coef))
```

We started with 100 recall samples per topic, but due to time limits, we can only use up to 6 models. So we tried 70 and 50 recall samples in the later stage of the competition.

| number of recall samples per topic | models | validation (1,000 topics) | LB score | PB score |
|------------------------------------|--------|---------------------------|----------|----------|
| 100                                | 6      | 0.725                     | 0.705    | 0.738    |
| 70                                 | 10     | 0.738                     | 0.715    | 0.751    |
| 50                                 | 12     | 0.743                     | 0.715    | 0.751    |

### Train Code:

https://github.com/syzong/2023-Kaggle-LECR-Top3-TrainCode
