# 2nd Place Solution

Competition: kaggle-llm-science-exam
Rank: #2
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/448256

Thanks to Kaggle for organizing this competition. I do similar things in my work, and it was interesting to see what an open community could achieve here. My final solution is mostly based on Deberta type models and only half of the test questions are handled by Mistral LLM.

## System components

1. Knowledge database
The Hugging Face dataset [graelo/wikipedia/20230601.en](https://huggingface.co/datasets/graelo/wikipedia/viewer/20230601.en) was used as a single source of knowledge. Each document was divided into stentenses and grouped into overlapping chunks with an average length of 1000 characters. The chunks were indexed using the `pyserini.index.lucene` package. The results index takes up 23 GB and retrieving 200*5 queries takes about 5 minutes.

2. Reranker model
Each retrieved chunk was rescored using a custom Reranker model. This Reranker `deberta-v3-base` model is built like the standard `DebertaV2ForMultipleChoice` model, but instead of predicting the correct answer given a context, this model predicts the best context for a given question/answer pair. To create the training dataset for this, a teacher model was used to label pseudo-ranks. Specifically, for each correct question/answer pair, few contexts was retrieved and evaluated using the teacher model. The best context was considered as correct option for the Reranker model. The teacher model was taken from early experiments and is based on `deberta-v3-large` model.

3. Multiple Choice model
Nothing special here, the standard `DebertaV2ForMultipleChoice` approach was used for Deberta and Mistral based models. Just one note: to train on datasets with different number of options, the loss function was scaled like this `loss * (self.num_labels / num_choices_in_sample)`.

4. Masked LM
Error analysis showed that the MC models performed very poorly on questions when the options are different in few words. For these specific cases, when the options are very similar and quite long, a special Masked LM was used. It is very easy to train this model, but it is quite difficult to apply this model for answering 5 options question. Ideally, we should find a multiple sequence alignment for 5 options and compare different words in a single pass. But I couldn't find a python package for this and decided to use `sed.standard_sed_backtrace` between each pairs of options. 

5. Cross reference options
To be honest, I thought the shakeup would be stronger. I assumed that the test data included more questions with option like `None of the above`. For example, when I replaced all correct answers in my validation sets with `None of the above`, the quality drastically dropped from 0.9 to 0.3 MAP@3 score. Keep that in mind, I built a second stage ranker with more training examples like this.

5. XGBRanker
XGBRanker was used as the fusion mechanism. This ranker is trained on validation sets and is based on few features: scores from retrival models, logits from Deberta/Mistral MC models and one feature flag for `None of the above` options.


## Failed experiments

* Focal loss for `DebertaV2ForMultipleChoice`
* Removing causal constraints for Llama attentions


## Datasets

| Author | Name | Size | Licence |
|--------|------|-----:|---------|
|@radek1 |all|39249||
|@leonidkulyk|stem_1k_v1|928||
|@nlztrk |eduqg_llm_formatted|3297||
|@mozattt|test|3084||
|@cdeotte|MMLU|17433|NC|
|AI2|[openbookqa](https://huggingface.co/datasets/openbookqa)|5957||
|AI2|[ai2_arc](https://huggingface.co/datasets/ai2_arc)|7787||
|AI2|[qasc](https://huggingface.co/datasets/qasc)|9060||
|AI2|[sciq](https://huggingface.co/datasets/sciq)|13679|NC|


## Ablation analysis

| System | Private Score | Public Score | Kernel |
|--------|---------------|--------------|--------|
| k=1, 2xDebertaMC | 0.909763 | 0.907823 ||
| k=8, 2xDebertaMC, Reranker | 0.916119 | 0.907823 ||
| k=8, 2xDebertaMC, Reranker, XGBRanker | 0.915963 | 0.909904 ||
| k=8, 2xDebertaMC, Reranker, XGBRanker, MLM | 0.915546 | 0.911568 ||
| k=8, 2xDebertaMC, Reranker, XGBRanker, MLM, MistralMC | 0.931489 | 0.931543 |[llm-xgboost-abc](https://www.kaggle.com/code/sorokin/llm-xgboost-abc?scriptVersionId=145958932)|

*k - number of documents retrieved from the index.


## Conclusion

I think the key components of this solution are the full Wikipedia index and XGBRanker on top of different models: BM-25/Reranker scores, Multiple Choice Deberta and Mistral logits.
