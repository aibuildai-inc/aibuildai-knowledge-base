# 18th place solution Generating Adversarial Data

Competition: llm-detect-ai-generated-text
Rank: #18
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470061

Thanks for my teammates and competition hosts for organizing this amazing competition

A brief summary of we did:
40% weight in TFID method. An average of 2 , 3, 4, and 5 n grams
60% weight in LLM ensemble. Mostly 7B Mistral/llama models. Our best submission used only our adversarially selected dataset described below.

The dataset we used was a dataset constructed through adversarial selection. We relied on all the awesome public datasets that were shared in this competition.
We train a small language model over 10% of the dataset. Predicted the rest. And then chose the hardest questions. Mixed it 50/50 with the original persuade dataset.

We actually found this method pretty late and our submissions were made on the final day using hastily trained models. I think this method could be potentially improved by using a better base model to select the adversaries, training larger lora rank, etc...
