# 47th place solution - no training, no dataset label string matching

Competition: coleridgeinitiative-show-us-the-data
Rank: #47
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248254

Yes you read that right. All of the models I tried to train did worse than pre-trained models.

Inspired by [this publication about MRC for NER](https://arxiv.org/pdf/1910.11476.pdf), I tested many question-answering models on the [Hugging Face model hub](https://huggingface.co/models?pipeline_tag=question-answering) to see which ones were able to extract the dataset name from a sentence. I ended up using an [electra model trained on squad v2](https://huggingface.co/ahotrod/electra_large_discriminator_squad2_512?context=The+inverse+association+between+bilirubin+and+risk+of+coronary+disease+was+analyzed+in+the+European+study+%22Prospective+Epidemiological+Study+of+Myocardial+Infarction+PRIME%22+30.&question=What+is+the+name+of+study+used%3F).

I trained a distilbert, roberta, and fasttext model to be able to tell if a sentence contained a datatset, but sadly a regex search for words like "Data" or "Survey" did the best. I think the advantage of regex is that it has very high precision compared to the others.  I'm sure that if I knew how to fine-tune a q-a model better I could have gone much higher up the leaderboard. Not bad considering it had no training... 

In brief
1. String match to find sentences mentioning dataset.
2. Use question-answering model to extract name if probability above a certain threshold. e.g. "What is the data source used?"

Pretty crazy that this helped me go up 1400 positions from public to private! 

[Notebook here](https://www.kaggle.com/nbroad/coleridge-regex-electra)
