# [8th Public / 18th Private] - Full Wikipedia Passage-level Retrieval

Competition: kaggle-llm-science-exam
Rank: #18
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446261

Congratulations to the top teams, especially to @philippsinger, @ilu000, and @ybabakhin! This was a fun competition and I learned a lot about training language models and effective inferencing for LLM on small GPUs. building the whole retrieval pipeline from scratch was a really fun exercise.

# A few thoughts about this competition

This competition was an rollercoaster for me: got to top20 just 1 week after joining, then the STEM 270K dataset changed the LB completely (7 days before competition end I was on ~190th place in the LB), got busy with work and abandoned the competition for weeks, and finally bouncing back to top20 in the last few days was both stressful and exciting. Not to mention that my Azure account got banned 11 hours before the end of competition (forgot to increase budget limits), just when I was uploading data to Kaggle, so my latest models did not make it to my final submissions 😅

At the end, the shakeup was super disappointing for me - after such an intense week without much sleep, I hoped for a better result. I'm glad I learned a lot by doing though 😁 My full inference code can be found here: https://www.kaggle.com/code/chankhavu/fork-of-0-92-submission

# My solution

My pipeline is quite standard - first, I retrieve relevant sentences from Wikipedia to add to the context for my final DeBERTa models. However, instead of retrieving articles and then sentences as in most public notebooks, I perform **passage-level retrieval from full-text Wikipedia!** This decision comes from the observation that, in the competition's 200 train set, there are many examples where the actual answer to the question can be found actually in a different, less related article. The re-ranking and prediction that comes after that are quite standard.

The code of my final submission is here: https://www.kaggle.com/code/chankhavu/fork-of-0-92-submission



## 1. Full Passage-level Retrieval - 54.5 million vectors/passages

I split whole Wikipedia to passages of 128 tokens each (which is approx. **100 words per passage**), with a stride of 96 tokens (so, for example, an article after removing citations/categories/bibliography and after tokenization will be split into passages with offsets `{ [0, 127], [96, 224], ... }`. This results in a HUGE embeddings dataset of **54.4 million vectors**, each has dimension 768, and takes **154 GBs** on the disk. Obviously, it's too large for Kaggle. We need to compress this index to be usable on Kaggle.

I've tried different combinations of FAISS indexes and quantization techniques - IVF, HNSW, Product Quantization (PQ) with OPQ preprocessing, Scalar Quantization (SQ), etc. I validate the indexes by generating questions with GPT-3.5, saving the Wikipedia article and passage that was used for generating the question, and measuring the recall of the index (if the retrieved passage is contained inside the passage used). The best and fastest combination is **IVF with 256 centroids**, and each vector is encoded into **96 bytes with Product Quantization**. Obviously, the more bytes in PQ the better, and 96 is the maximum possible PQ for GPU inference. The index is compressed to less than **6 GBs**, takes only ~20 seconds to perform a full search for 4000 questions (with `K=100` and `nprobe=96`)

With **only** this index, I was able to achieve **0.891 LB** using just **one public model** (from Chris notebook), and this is before the STEM 270K dataset was published!

## 2. Science Articles Sub-Index - 7.6 million vectors/passages

I split the STEM 270K articles to passages of 128 tokens each and with stride 64 tokens. Because we have less vectors than the Full Wikipedia Passage-level Index, we can afford to use more precise indexes and get rid of the PQ quantization (that, in my opinion, is too aggressive). I ended up using **IVF256** and use a simple **FP16** quantization for this, allowing a more precise retrieval. This boosted my **0.891** LB solution to **0.895** - the boost is much less than other teams, which I think is an indicator that my full passage-level retrieval is quite powerful by itself.

**12 hours before the end of the competition**, I also added 300K more articles from Wikipedia (I did not scrape using the API, just added from the Huggingface dataset) that are related to Science. I got those 300K articles so by asking GPT-4 to generate top-level topics in Science (Physics, Chemistry, Biology, Engineering, etc.) in general, and did some clustering with `be-base` embeddings. In total, the Science sub-index has **600K articles** and **7.6 million vectors** (passages). This added a slight boost to both my public and private scores as well.

## 3. Re-ranking sentences

This part is quite straightforward. I split the passages to chunks of 3 sentences with stride of 1 sentences, and re-rank those chunks using `BAAI/bge-large-en-v1.5`. Additionally, I also calculate the BM25 retrieval score for each of the sentences (using the all retrieved sentences for all questions as corpus).

## 4. Training DeBERTa models

I trained 6 DeBERTa models, each with slightly different set of training data and context length. All those models were trained in the last 3 days of the competition using 1xA100 and 1x3090, and are likely under-trained because I did not have much time. A few interesting observations:

* Training data was generated using GPT-3.5, using different prompts to add diversity. I also used the 60k dataset by Chris Deotte. I also prompted the model to try to include more numbers, formulas, etc. to make the question harder (from a tokenizing standpoint).
* **Training on worse context gives better models!** So I generated my training data with weaker version of my retrieval pipeline (using `bge-small`). The difference is ~0.005 on LB.
* This competition is about reading comprehension and unleashing the knowledge inside the LM. So, smaller `lr` and more Q/A examples works.

# What did not work

* I spent some time trying to train a re-ranking model with LightGBM/XGBoost using LambdaRank objectives and various embedding scores/bm25 scores/tfidf scores as features. This is worse than a simple embedding score unfortunately.
