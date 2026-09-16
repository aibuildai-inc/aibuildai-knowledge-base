# 26th place solution

Competition: map-charting-student-math-misunderstandings
Rank: #26
Source: https://www.kaggle.com/c/map-charting-student-math-misunderstandings/writeups/26th-place-solution

Thanks to organizers and host of competition! It was a fun challenge, and I'm happy to get my first medal.

[Github Repo](https://github.com/Fosca709/kaggle-map)

### Data Processing

Labels were formed like ‘True_Correct:NA’. But the ‘True’ and ‘False’ prefixes could be inferred from other data, so I removed them. So I only use 'Misconceptions' column, and replacing 'NA' with 'Correct' and 'Neither' categories. That reduces the total number of classes to 37. This makes a better CV score. It didn't improve the publice LB score, but it also didn't make it worse, so I decided to stick with it.

For the prompting side, I used an almost same prompt in [this notebook](https://www.kaggle.com/code/cdeotte/ettin-encoder-1b-cv-0-943), but added the true answer to the question.

### Data Augmentation

Upon inspecting my trained model's prediction errors on the validation dataset, I made several observations:

* A substantial proportion of errors were due to misclassifying the 'Neither' class.
* The model sometimes struggled to classify meaningless gibberish as the 'Neither' class.
* I also found one particularly interesting data point: A student had mistakenly provided a correct solution to a different question, and my model confidently classified it as 'Correct'.

This last observation, in particular, inspired the following data augmentation strategy: For a given data, I randomly sample another data from a different question. I then replaced the 'StudentExplanation' column of the original data with that from the randomly sampled data, and assigned the 'Neither' class to this augmented example.

In each training epoch, I randomly selected 25k data points (or 20k for 5-fold CV) and applied this augmentation. This consistently boosted both the CV and LB scores by 0.001~0.002 on a single model.

### Training

I trained several models using the data augmentation described above. After performing cross-validation, I re-trained on the full dataset and submitted it. For all models, I used LoRA with r=64.

|Model|CV|public LB|private LB|
|---|---|---|---|
|Qwen3-8B|0.9481|0.945|0.942|
|Qwen3-Embedding-8B|0.9472|0.946|0.944|
|Qwen2.5-Math-7B|0.9471|0.945|0.941|
|DeepSeekMath|0.9459|0.943|0.940|
|F2LLM-4B|0.9469|0.947|0.942|
|LGAI-Embedding-Preview|0.9456|0.941|0.940|
|Qwen3-Embedding-4B|0.9504|0.947|0.944|
|Qwen3-4B-2507|0.9482|0.946|0.942|
|Granite-4.0-micro|0.9465|0.943|0.942|
|DeepSeek-Prover-V2-7B|0.9462|0.946|0.942|
|DeepSeek-R1-0528-Qwen3-8B|0.9460|0.945|0.942|

Ensembling these made a final LB score of 0.947. I explored two ensemble techniques: simple averaging of predicted probabilities, and a method based on [this popular notebook](https://www.kaggle.com/code/kishanvavdara/ensemble-gemma-qwen-deepseek). I removed the 'agreement_bonus' term as it had no influence, and I excluded the 'Neither' class from the 'confidence_bonus'. This was because my models had already been extensively trained on 'Neither' classes via augmentation, and I want to give more diversity into the predictions.

### Things that didn't work

* When training smaller models (ettin-encoder variants) with full weights, the Muon optimizer outperformed AdamW. But it didn't when combined with LoRA. So I abandoned it for training larger models.
* The dataset contained label inconsistencies, so I experimented with noise-robust loss objectives. Bootstrapped Cross Entropy, as detailed in [this paper](https://arxiv.org/abs/1412.6596), provided a slight performance boost on a smaller model (ettin-encoder-17m) but actually led to worse performance on larger models.
* I observed that misclassifications between different specific misconceptions rarely occurred and had almost no influence on the overall score. This led me to experiment with a further reduced label set, aggregating all specific misconceptions into a single 'Misconception' class. This left only three target labels 'Correct', 'Neither', and 'Misconception'. My plan was to use a model trained this way for ensembling if it performed well. However, it performed significantly worse.
