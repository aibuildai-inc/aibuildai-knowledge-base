# Efficiency Track 1st Place Solution - Model Merging

Competition: eedi-mining-misconceptions-in-mathematics
Rank: #33
Source: https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/discussion/552108

### Starting note

Thank you to the competition hosts and Kaggle team for organizing the competition and also the efficiency track. It was a fantastic learning experience of both data science and engineering.

**Inference Code shared** [here](https://www.kaggle.com/code/dipamc77/qwen0-5-sentence-model?scriptVersionId=212466703)

### Efficiency Track Time/Score Tradeoff

I'll begin by mentioning the tradeoff efficiency track score had built-in. Specifically, if you assume the max Private LB will be close to 0.6, and we know the "baseline" LB score is close to 0. The trade off comes out to be `9 mins per 0.01 in LB score improvement`. After a point, this is extremely hard to tackle on the score front, and the problem essentially boils down to time optimization, or so I thought until the main differentiator of my solution gave a huge boost.

For this competition, its seems fairly straightforward to get a ~0.27-0.29 Private LB score in appoximately 30 mins. This can be achieved with some good quality synthetic data on missing concepts combined with the training data provided to train a single embedding model of ~400m size.

After this point, one can go in many directions to optimize the time or score.

### Score/Time improvement ideas

* **Decoder only model** - The question/subject/construct are same for 3 answer choices, which means their K-V cache be re-used with prefix caching. Hence I switched from encoder only models like BGE/GTE to Qwen2.5-0.5B. With prefix caching, the time reduced to 14 minutes.
* **Caching misconception embeddings** - Its straightforward to pre-compute and save the misconception embeddings.
* **Reranker** - Naively running a 400m sized reranker improves the score, but is quite slow. Smaller models don't make enough of a improvement in score to justify the time increase.
* **Shared model Reranker** - One way to overcome this can be to re-use the embedding model with a few additional layers to apply attention to the misconceptions sequences. However this model is harder to train after already training and overftting the embedding model due to the limited dataset, so joint training of both models may be needed. I didn't have time to implement this idea.
* **CPU Optimizations** - CPU optimizations like OpenVINO are likely to make a model faster with no drop in scores, I didn't try it. One minor detail to remember is CPU models run really slow on FP16/BF16, so stick to FP32.

### Main idea - Model merging

Model merging with techniques like SLERP or TIES available in [mergekit](https://github.com/arcee-ai/mergekit) can improve scores by merging different models trained on different datasets/folds, sometimes even merging with the base model can help. Since this improves the score without any change in runtime, its a free improvement in score. The weight merging curves were a bit of a hit and trial, more so because I didn't have much experience in it.

The scores improved even when merging models trained on the same data, with different hparams like batch sizes or weight decay. For my final submission I merged 6 models with SLERP with 4 merge steps. These are all different full fine tunes of Qwen2.5-0.5B-Instruct.

While I expected moderate improvement in score using merging, the improvement was dramatic, merging just 2 models increased the score from `0.305` -> `0.330` on public LB.

Two major caveats to using this method:

  1. Only works when the models getting merged have similar performance. When merging with base models, care needs to be taken to merge to assign higher weightage to the fine-tuned model, while not assigning too much that there is no improvement.
  2. Since CV/test set scores are the only way to assess the merged model, one can defiitely get carried away tuning merging curve parameters and overfit the CV scores. This pretty much happened with me where the CV and Public LB both improved after 4 merges, but the best Private LB came from 3 merges.

This score alone was such a big boost that I didn't spend any more time on the efficiency track, I expect combining it with a shared reranker and cpu optimizations can improve the model score/time further.

P.S - Had to do a little bit of hacking around in mergekit to make the merge work because its primarily made for text generation configs.

### Some more learnings on merging

I got so intrigued with the merging results that I stopped focussing on the competition to learn more about merging. First I wanted to know if it also works on larger models. I tried it on some of the LoRA models on 7B Qwen that I had trianed for the main track, and it didn't work at all at first, but I was doing something wrong. The correct way to use SLERP for such a setup is to first merge the LoRA weights into the base model (note that merging the LoRA is different from merging two models with SLERP). So first, load the PEFT model and use `merge_and_unload` to get the full finetuned weights, then do the same for the other LoRA, this gives two sets of model weights. Now we can combine these weights with SLERP. This process gets resource intensive as the model size increases, making it difficult to tune the SLERP curves.

The results of merging 7B LoRAs in this manner gave only a small improvement, in contrast with the dramatic improvement when doing it on 0.5B models. I even tried a full fine tune of a 7B model to see if it will do any better, but didn't spend enough time on this to get any major improvement.

### 33rd Place solution on main track

My solution was pretty straightforward for the main track

* Generate missing misconceptions with Claude-3.5-Sonnet/GPT-4o + filter poor data with GPT-4o. It helped to do CV by keeping datasets from different LLMs in different training/validation splits.
* 32B embedder - Trained with [CachedMultipleNegativesRankingLoss](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CachedMultipleNegativesRankingLoss.py) to limit to single GPU training
* 32B pointwise reranker with binary classifier - Rerank top 5
* 32B listwise reranker for the top 2 candidates - Limited to 2 because I was wary of positional bias, didn't experiment much.

All models are simple LoRA. I mostly did experiments on 7B models due to resource constraints, and trained all the 32B models in the last 2 days.

I also did some experimentation with CoT reasoning as input from 32B models but didn't get much score improvement/didn't spend enough time on it.

### Personal opinions on Efficiency track

Its a bit disappointing that the efficiency track doesn't award medals, I think most of the top teams didn't spend any time here. I got a bit too lucky with the merging solution in the end. If the efficiency track awarded medals, the solutions would be stronger and competition would be harder/more fun. The high amount of prize money certainly suggests that the organizers wanted more teams to focus here.

The metric seems to favour time optimization much more over scores, anything more than a single embedder is probably too slow. An improvement of 0.01 score for only 9 mins doesn't seem like a balanced trade-off. I may be wrong here, maybe that is exactly the trade-off the host wanted.
