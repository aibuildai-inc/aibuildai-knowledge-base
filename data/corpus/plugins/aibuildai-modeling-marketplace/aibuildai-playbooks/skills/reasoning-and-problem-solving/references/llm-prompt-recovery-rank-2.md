# 2nd place solution: Team Danube

Competition: llm-prompt-recovery
Rank: #2
Source: https://www.kaggle.com/c/llm-prompt-recovery/discussion/494497

Thanks for hosting this competition, we had fun trying to optimize for the given t5 metric and incorporating some useful modeling approaches to our solution. As always, great teamwork with @ilu000 and @ybabakhin.

Here we want to highlight the main chronological steps to come up with our solution and relevant thought processes.

### Mean prompts and bruteforce optimization
When we joined together in the competition, mean prompts dominated the public and our individual solutions and seemed the most low hanging fruit going forward, specifically also after we figured out that t5 metric is very sensitive and just raw llm predictions will not be too useful. So we decided to keep exploring optimal mean prompts.

We quickly saw that just manually trying things is not effective, so we started exploring how to brute force the actual tokens for optimal mean prompts. We took our training datasets, and tried to brute force each of the ~32k possible t5 tokens to form the most optimal mean prompt for the average of the target vectors. But after submitting that, we got bad results leaving us initially puzzled. Obviously, `</s>` is an important token in the optimization given that it is appended to each of the target vectors. After exploring this further, we found out that tensorflow is using the original `SentencePiece` tokenizer, which has protection against special token injection, and thus does not directly tokenize the string `</s>` as the correct eos token. 

Given this insight, we excluded special tokens in the brute force optimization, and, as many others, the optimizer now really started to like tokens like `lucrarea` being close to eos token in embedding space. This allowed us to get much closer LB scores compared to our CV scores and we managed to get to around 0.65 with just optimizing the mean prompt that way.

### Embedding models
However, we still wanted to directly find a model that can do parts of this job. Specifically, we tried to directly train a model predicting the expected embedding. A simple way of doing that, was to train a classification model in H2O LLM Studio, where we use the 768 output embedding dimensions as the target in our training data. We then also directly implemented a cosine similarity loss, so that the model would directly learn our target metric. With this approach, we managed to get to local scores of around 0.75+ with our embedding predictions. Our embedding models either used H2O-Danube / H2O-Danube2 models, or Mistral 7b with little difference only.

However, the main issue still was that we need to go back from predicted embeddings, to the string representations that are then used in the metric for calculating the t5 sim. We thought of directly modeling this as well in some way, but then resorted back to the bruteforce optimization routine that greedily tries token combinations to match the predicted embedding as closely as possible. Within a certain runtime, we managed to lose around 3-4pts with this optimization, so getting to local scores of around 0.71 and LB scores of around 0.68 or 0.69 just bruteforcing each individual embedding prediction.

### LLM predictions and improvements
This was bringing us into the gold zone. Now we tried to find ways of closing the gap between the bruteforce and predicted embeddings a bit further. The first thing we found, that it is actually helpful to initialize the optimization with raw llm predictions, but only predicting the actual change needed to make, such as “as a shanty”. So we prepared our data in that way, and trained llms and incorporated them into our solution as a starting point for the optimization, and also adding them to the embedding blend. Furthermore, for diversity, we also added few shot predictions to that mix. And to even further improve the quality and speed of the optimization, we also added a good working mean prompt there.

This means that our final starting point for the optimization is:
`Few shot predictions + LLM predictions + Mean prompt`

And the final predicted string is:
`Few shot predictions + LLM predictions + Mean prompt + Optimized embedding string (20 tokens)`

The following figure summarizes our full pipeline:



As another example, our final prediction for the single test sample would be:
`“Rephrase paragraph text by turning it into a shanty. shanty shanty.lucrarealucrarealucrarea sentence appealinglucrarea Improve respond storytelling tonelucrareaimplication. write someoneran. lucrarea]. Consider clarify paragraphlucrarea similarly serious themed way temporarily.! ElePT lyrics rhyme poem solve songlucrarea participating version Deliver tale Hum Cor slogan remake this pieceody”`

### Data and CV
For all the parts described above, we generated various kinds of data for training and optimization. We started with some public datasets, but quickly found out that supplementary texts provided by Kaggle were most useful. So we used different models (mostly gemma) for generating new original texts and rewrite prompts by using supplementary texts as few-shot examples. In the final data mix we included extra random original texts and random rewrite prompts for more diversity. We also crafted a validation set of ~350 samples where we saw good correlation between local mean prompt scores and submitted mean prompt scores and developed our solution on that. We had very good correlation between CV and LB in the end.

Thankfully, public and private was a fair split, which we think is also the only reasonable thing to do in a competition without training data. And thus, our best LB sub was also our best CV sub and also private sub!

Useful links:
- [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio)
- [h2o-danube2-1.8b-base](https://huggingface.co/h2oai/h2o-danube2-1.8b-base)

Code:
- [Inference kernel with all relevant code](https://www.kaggle.com/code/ilu000/2nd-place-team-danube-llm-prompt-recovery)
- [H2O LLM Studio training code](https://www.kaggle.com/datasets/philippsinger/kaggle-prompt-llmstudio-psi-working-v3)
- [Training data](https://www.kaggle.com/datasets/ybabakhin/llm-prompt-recovery-generated-data)
