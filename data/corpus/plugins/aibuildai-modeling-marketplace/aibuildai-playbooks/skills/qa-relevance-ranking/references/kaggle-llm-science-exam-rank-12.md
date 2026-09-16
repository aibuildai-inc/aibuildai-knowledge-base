# 13th place solution: Ensemble context ensemble model

Competition: kaggle-llm-science-exam
Rank: #12
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446301

First of all, thank you to Kaggle for organizing this competition and to my teammates.[@natnitarach](https://www.kaggle.com/natnitarach) [@pongtsu](https://www.kaggle.com/pongtsu) [@kunato](https://www.kaggle.com/kunato) [@yoyoismee](https://www.kaggle.com/yoyoismee) 🔥🔥🔥🔥🔥

We have been participating in this competition since the first week of the competition. We dropped from the top 20 down to the nowhere about 3 times in this competition but always managed to get back up. Each time we see the rankings fall from the top 20 to nowhere, it makes us feel frightened (its like playing roller coaster LOL😅) but at the end of the competition we were able to get the gold medal 🥇🎉. 
Our solution is ensemble of 2 wikipedia version + 270k dataset and ensemble of debertaV3 large model by max probability
I will split into 2 section retrival and modeling. Retrival has the big impact to our score.

**Retrival**
In the Retrieval phase, we use two sources of Wikipedia embeddings, utilizing the "bge-small-en" to create a FaissFlatL2 index.
1. Wikipedia cohere 35M -> 35M vector
2. [Wikipedia 2023](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701) -> split by character chunk = 1000 -> 21M vector

and [270k dataset](https://www.kaggle.com/datasets/mbanaei/all-paraphs-parsed-expanded)
In total, we had 35M + 21M = 56M vector to retrival search that is huge!. We use technique that split big index to small index then merge at the end we call it "Faiss Batch" (1 wikipedia has around 6 index each index is 10GB size). so it total size 100GB+ of only retrival part.


to inference you can see image above the step for inference was
1. The wikipedia dataset we use faiss query search to get top ~100 article (6 index * 15 neighbor)
2. Use TF-IDF for reranking from experiment we found that TF-IDF reranking has score better than bge-reranker

Example code of what we call Faiss Batch
```
from datasets import Dataset
ds = load_from_disk('/kaggle/input/rms1data')
test = Dataset.from_pandas(test)
import os
dir_path = '/kaggle/input/allyouneedret'
index_list = os.listdir(dir_path)
index_list = sorted(index_list)
print(index_list)
k = 10
total = 0
distance_list = []
indices_list = []
res = faiss.StandardGpuResources()
for indexBatch in index_list:
    index_Batch = "/kaggle/input/allyouneedret/" + indexBatch
    print(f"read index {index_Batch}")
    index1 = faiss.read_index(index_Batch)
    index1 = faiss.index_cpu_to_gpu(res,0,index1)
    distances1, indices1  = index1.search(query_vector,k)
    updated_indices2 = [[idx + total for idx in inner_list] for inner_list in indices1]
    total += index1.ntotal
    print(total)
    distance_list.append(distances1)
    indices_list.append(updated_indices2)
    del index1
    _ = gc.collect()
    libc.malloc_trim(0)
    torch.cuda.empty_cache()
concatenated_indices = np.concatenate(indices_list, axis=1)
concatenated_distances = np.concatenate(distance_list, axis=1)
```
what a messy code😂
in the 270k dataset we follow [discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/442595) that use TF-IDF 

at the end we will have 3 context and each model will predict 3 times.

**Model**
In the Modeling phase, We use DebertaV3 large and ensemble it.
we use same pipeline as [Chris](https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1) unfreeze embedding and use various token from 512-768 with [60k dataset](https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2) + dataset that we generate from chatGPT3.5 around 5k question.




**What Didn't Work**
- Attempts to compress the index (IVQ, PQ, etc.) resulted in low accuracy.
- We tried combining Platypus2-70B-instruct with our retrieval method, but it proved impractical due to the extended runtime, exceeding 9 hours during submission.
- The application of PEFT on DeBERTaV3 did not yield the desired results.

**To improve**
- Obtain better Wikipedia text sources.
- Enhance the performance of our models.

Github Code
https://github.com/nat-nischw/kaggel-llm-science-exam-2023
