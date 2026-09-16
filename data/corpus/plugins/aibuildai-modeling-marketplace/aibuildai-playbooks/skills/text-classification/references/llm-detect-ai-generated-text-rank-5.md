# 5th place solution: 1.7 million training examples + domain adaptation

Competition: llm-detect-ai-generated-text
Rank: #5
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470093

Thanks to the competition organizers for creating such an interesting competition! I learned a lot. And congrats to the other winners!

Here's an explanation of the 5th place solution...

# Overview

Finetuned [deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) and [mamba-790m](https://huggingface.co/state-spaces/mamba-790m) on varying mixtures of the following datasets:

|  Dataset | # Human documents | # Generated documents |
| --- | --- | --- |
| PERSUADE essays | 25,996 | 327,268 |
| Uncopyrighted Pile Completions | 512,371 | 512,371 |
| SlimPajama Completions | 233,146 | 233,146 |
| Tricky Crawl | 125,192 | 0 |

</p>Exact sampling proportions varied across models in the ensemble, but a reoccurring pattern is that everything except the Pile data was undersampled. The best models used 62 - 99% Pile data. It was our highest quality dataset. The Persuade essays were always 1% of the data mixture.

Then, at test time, our inference notebook does the following:
1. **Teacher model inference:** The test dataset is labeled with 1 DeBERTa model + 2 Mamba models with a context length of 1024 tokens. Final soft labels are a weighted average with 90% of the weight given to the DeBERTa model. **The use of Mamba hurt us here, we would have gotten 3rd place (0.977) without it.**
2. **Student model training:** Finetune 2 "short context student" models to imitate the teacher ensemble's predictions based on short randomly selected chunks of the test documents. One student model uses a context length of 128 *characters* (~32 tokens), the other uses 256 characters. These student models are pretrained on 1 million documents from the datasets above, mostly Pile data, and are both deberta-v3-large based.
3. **Student model inference:** The finetuned student models make predictions on overlapping chunks of the test documents with a stride equal to half their context length. Final predictions for each document are the average of all these predictions, with 60% of the weight given to predictions from the model with a context length of 128 characters, 40% to the one with a context length of 256 characters.

My intuition for why this domain adaptation strategy is beneficial is that the short-context student models learn to look for dataset-specific wording quirks that are correlated with whatever the long-context models look for in the broader documents. This strategy was primarily inspired by [SimCLR](https://arxiv.org/abs/2002.05709), but in hindsight I realize it is also similar to [Noisy Student Training](https://arxiv.org/abs/1911.04252) and [UDA](https://arxiv.org/abs/1904.12848)

On its own, the full-context DeBERTa model we used scores 0.970 on the private LB, the Mamba models score ~0.957. The final student models score 0.977 without Mamba, 0.972 with Mamba. We didn't pick the best submission as one of our final 3, so we got 5th place instead of 3rd. Also, the DeBERTa model we used wasn't the best... we had one score 0.976 on its own, without domain adaptation.

**Links:**
* **Inference/domain adaptation notebook:** [https://www.kaggle.com/jsday96/multi-context-students/](https://www.kaggle.com/jsday96/multi-context-students/)
* **Data:** [https://www.kaggle.com/datasets/jsday96/ai-content-detection](https://www.kaggle.com/datasets/jsday96/ai-content-detection)
* **Local training & data generation code:** [https://github.com/jday96314/AiContentDetection](https://github.com/jday96314/AiContentDetection)

# Data
## Things all datasets had in common
* Ran [vLLM](https://github.com/vllm-project/vllm) on 2x RTX 3090s + 1x RTX 4090 to generate examples of AI-authored text.
* Used 4-bit AWQ quantization to conserve GPU memory for any model with >= 13 billion parameters.
* Sampling temperatures randomly varied from 0 - 2 (used different randomly selected temperature for each vLLM API request).
* Top-K filter randomly selected from [disabled, 20, 40] for each vLLM API request.
* Top-P filter randomly varied from 0.5 - 1 for each vLLM API request.
* Frequency penalty randomly varied from 0 - 0.5 for each vLLM API request.

## Uncopyrighted Pile Completions

The general strategy for creating this dataset was to randomly select documents from [The Pile](https://pile.eleuther.ai/), randomly truncate them, and prompt locally hosted LLMs to generate plausible continuations for the last 25 - 75% of each document. I initially used an old version of the pile which contains copyrighted data and is no longer publicly available, but filtered it to only contain document completions based on subsets of The Pile which are free of copyright restrictions and are [still publicly available](https://huggingface.co/datasets/monology/pile-uncopyrighted). 

Document completions were generated with the following models. Document counts below are from after the filtering which ran as a post-processing step (and discarded ~16% of the data if I recall correctly).

|  Model | Document count |
| --- | --- |
| Airoboros-L2-13B-2.1-AWQ | 46,358 |
| CodeLlama-34B-AWQ | 29,885 |
| falcon-7b | 11,178 |
| Llama-2-13B-AWQ | 61,531 |
| Llama-2-13B-chat-AWQ | 43,145 |
| Llama-2-70B-AWQ | 31,297 |
| Mistral-7B-Instruct | 46,825 |
| Mistral-7B-v0.1 | 61,851 |
| mpt-7b | 12,232 |
| OpenHermes-2.5-Mistral-7B | 21,221 |
| StableBeluga2-70B-AWQ | 21,046 |
| WizardCoder-Python-34B-V1.0-AWQ | 41,716 |
| WizardLM-70B-V1.0-AWQ | 41,979 |
| zephyr-7b-beta | 42,093 |

</p> For foundation models, the prompts were just document prefixes. For models that were instruction or chat tuned, I randomly broke the document prefixes into two parts, used the first part to form "user" instructions requesting a plausible continuation to the document and used the second part as "leading words" in a part of the token sequence that would ordinarily be filled in by the model (after something like a "### Response:" tag which varies a bit depending on what format the model was trained to expect). The leading words help to ensure the models generate plausible looking document completions, without any weird prefixes at the start of their responses or refusals.

This data was generated in 2 stages. My observations from training models on the early data and the changes I made for the last ~15% are described below.
* Models trained on my initial data were worst at identifying text generated with a sampling temperature close to 1. In response, I switched from picking the sampling temperature from a uniform distribution ranging from 0 - 2, to instead using a gaussian distribution centered at 1 with a standard deviation of 0.2, clipped to be in the range 0 - 2. This helped to concentrate more of my hardware resources on the tricky data my classification models were struggling with.
* My models were best at identifying AI generated text related to the medical subsets of The Pile (PubMed Central & PubMed Abstracts). As a result, I figured adding more medical related text to the training dataset would be a waste of hardware resources and started filtering it out (before the LLM is prompted, not as a post-processing step like what I did to discard the data with copyright issues).
* My models were worst at identifying AI generated code. This data didn't seem too relevant to the competition, so I started filtering out everything from the GitHub slice of The Pile, similar to what I did for PubMed.

## SlimPajama Completions
The way I created this dataset is very similar to what I did for the the last of the Pile completions, just with source documents from [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) instead of The Pile and a slightly different combination of models. This data was generated somewhat preemptively to hopefully mitigate any accuracy lost from filtering out the problematic (copyrighted) parts of The Pile. It didn't really help on the public LB, but might have on the private LB - I'd need to look back at the scores more closely. 

The mixture of models I used to create this dataset is outlined below.

|  Model | Document count |
| --- | --- |
| airoboros-l2-70B-gpt4-1.4.1-AWQ | 23,399 |
| deepseek-coder-33B-base-AWQ | 20,718 |
| dolphin-2.6-mistral-7b | 24,168 |
| Mistral-7B-Instruct-v0.2 | 24,813 |
| Mistral-7B-v0.1 | 24,618 |
| mixtral-8x7b-v0.1-AWQ | 24,609 |
| Mixtral-8x7B-Instruct-v0.1-GPTQ | 24,581 |
| Nous-Hermes-2-SOLAR-10.7B-AWQ | 22,833 |
| Nous-Hermes-Llama2-AWQ | 18,741 |
| SOLAR-10.7B-v1.0-AWQ | 24,623 |

</p>One concern I had about SlimPajama (and why I didn't use it initially) is that it contains text from the internet that was collected *after* the release of ChatGPT, so there might be some AI-generated text mislabeled as "human" text when I train on it. It is plausible this might be why it seemed to be lower quality than The Pile data.

## PERSUADE essays
I generated a bit over 300k essays for the same assignments as the human essays in the PERSUADE corpus. These essays were divided into 3 subsets with varying generation strategies:
* **Basic:** ~137k essays using variety of prompting strategies. 0-shot or 1-shot with ~140 different sets of instructions used to alter the LLM's writing style. Used a combination of 11 different models.
* **Student Imitator:** In an attempt to make the AI-generated essays harder to identify, I finetuned Mistral 7B and Llama 2 13B to imitate the writing style of the students in the PERSUADE corpus. I used the resulting models to generate ~237k essays, ~177k of which were used for training and cross-validation purposes. The rest were discarded due to issues during finetuning that caused the LLMs to output incoherent text.
* **Adversarial:** I adversarially generated ~12k essays that confuse a mixture of 17 "victim classifiers". The main strategy for generating these essays to back-track and try again whenever the in-progress essay starts to sound "too AI generated". However, some LLMs are seemingly incapable of generating introductions that fool my classifiers for certain assignments, so many of these essays begin with authentic human text in order to "get past the intro". However, even with that in place these essays were still ~100x more computationally expensive to generate than "normal" essays, so I don't have many of them.

Unfortunately, these essays were not very useful in comparison to the Pile data. My classifiers seem to rapidly overfit to it and need more diverse data to generalize well.

I feel the exact combinations of models & instructions I used would be a bit too much to include here, so I suggest looking at the dataset I linked to in the introduction & my data generation code for details.

## Tricky Crawl
This was created by filtering the Common Crawl subset of The Pile (Pile-CC) to pull out human-authored text that was misclassified as AI generated by one of 2 moderately strong general-purpose AI content detection models. One of the classifiers was based on deberta-v3-base, the other on deberta-v3-xsmall.

The version of this dataset used to train the models in the final submission was created by filtering 1.5 million documents to pull out the ~125k most confusing to the victim classifiers. 

The general idea behind this dataset was to reduce the number of false positives by boosting the amount of tricky human text in the training dataset. Mixing this into the training dataset with a 5% sampling proportion (so 5% of the training examples are from this dataset) seemed beneficial for the short-context models, but didn't make much difference for the full-context ones. 

# Data augmentation
The competition organizer's data seems to be a bit corrupted, so the models in our solution were trained with the following data augmentation steps to make them more robust.
* **Buggy spell check** similar to the implementation described by @deltawi in [this discussion thread](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/456142). However, I used a regex which seems to handle ' characters in a way more consistent with the competition data and used [jamspell](https://github.com/bakwc/JamSpell) instead of pyspellchecker because jamspell is faster. This executes with 70% probability for persuade documents, 20% for all other datasets.
* **Blacklisted character removal** with the same blacklist suggested by @nbroad in [this discussion thread](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452172). This executes with 70% probability for persuade documents, 20% for all other datasets.
* **Typos are added.** More specifically...
 * All typos of typos supported by the [typo library](https://github.com/ranvijaykumar/typo) have 2 opportunities to be added to all training examples with 10% probability each time.
 * Capitalization of random characters is flipped. 2 opportunities to decapitalize an upper case letter, 1 opportunity to capitalize a lower case letter. All these "opportunities" for typos to be added have a 10% occurring for each document.

# Mamba vs. DeBERTa
## Efficiency
`mamba-790m` is roughly as fast as `deberta-v3-large` and consumes less GPU memory during training despite having over 2x as many parameters. It seems to have some major efficiency advantages due to being a structured state space model instead of a transformer.

## Accuracy (and stability problems)
`mamba-790m` did *almost* as well as DeBERTa during local CV on the Pile data (0.001 - 0.002 lower) and *almost* as well on the public LB (~0.003 lower than an "unlucky" DeBERTa run, ~0.006 lower than a "good" DeBERTa run). The main difference arose on the private LB. Mamba's score staid roughly the same from public to private while most DeBERTa models jumped by ~0.01. As a result, Mamba seemed to be slightly useful on the public LB when included in an ensemble with low weight, but wound up dragging us down on the private LB.

I think the weakness of Mamba here wasn't really due to architectural flaws or because structured state space models are inherently inferior to transformers, but rather because this was my first time using Mamba and I rushed to train it in the last 3-4 days of the competition. Additionally, the [mamba-ssm](https://github.com/state-spaces/mamba) library seems to be primarily intended for text generation, not classification, so it isn't super straightforward to use for workloads like this.

To elaborate on the **mistakes I made** when training it, I generally initialized the Mamba model like
```
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-790m")
model.lm_head = nn.Linear(model.config.d_model, 2)
```

which causes it to outputs predictions with shape (batch size, token count, 2). During early training runs, I then used the output from the last token in each sequence as class label predictions using an operation like
```
raw_predictions = output.logits[:, -1, :]
```

The problem with this approach is that the token sequences are padded and the padding was impacting the predictions. Getting predictions from the last token before the start of padding with an operation like
```
last_token_indices = torch.clamp(attention_masks.sum(dim = 1) - 1, min = 0)
raw_predictions = torch.gather(
    logits, 
    dim=1, 
    index=last_token_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, logits.shape[2])
).squeeze(1)
```

seemed to increase the optimal learning rate by a factor of roughly 8x and allowed for much faster convergence to a more accurate model during relatively small-scale training runs with a single epoch over "only" 100,000 example documents. However, it was unstable when I attempted to scale up to more data with a learning rate that high. I didn't really have time to train it properly after figuring this out. As a result, the 2 Mamba models used by the 5th place solution were trained using the following non-ideal setups:
* 1.25 million training documents with the original logit selection approach ([:, -1, :]) and a relatively low learning rate (2e-6 with a batch size of 4). This training run executed in the background while I used other GPUs to figure out how to train Mamba "properly".
* 350,000 training documents with the max lr set to a problematically high value (1.6e-5) for the first ~70% of training, followed by repeated issues in which the loss became nan and the model started outputting garbage, thereby requiring repeated restarts from old checkpoints with much lower learning rates. I wound up manually dropping the learning rate all the way down to 5e-7 in the last 30% of training and did not have time to train it on as much data as most other models. I have a suspicion I would have gotten better results if I started the run from the beginning with a well-configured learning rate schedule and used a larger amount of data.

# Additional training details
<ul>
  <li><strong>Dropout:</strong> Totally disabled for all models.
    <ul>
      <li>Encountered <strong>strange behavior</strong> in which if I trained the DeBERTa models with Huggingface transformers library's default dropout rates, they were consistently more accurate at inference time if I left dropout enabled at inference time (i.e. leaving the model in "train" mode instead of switching to "eval" mode). I suspect this is because DeBERTa uses non-standard <code>StableDropout</code> layers that appear to do some sort of internal normalization that the model was learning to rely on, thereby forcing me to keep dropout enabled for best results. Needless to say, randomly dropping-out connections in the model at test time isn't good for accuracy. I got better results with dropout disabled both at training time and at test time, even in comparison to averaging across multiple forward passes through a model with dropout enabled.</li>
      <li>The <code>mamba-ssm</code> library doesn't support dropout.</li>
    </ul>
  </li>
  <li><strong>Learning rate schedule:</strong> Linear warm-up followed by linear decay. Generally warmed up for the first 5% of training during the local training runs, 30% during domain adaptation at test time. <strong>Longer warmup during domain adaptation seemed to make it more stable</strong>, with good results more consistently, but I didn't experiment with this much.</li>
  <li><strong>Batch size:</strong> For the models with a context length of 1024 tokens, I used a batch size of 2 for DeBERTa, 4 for Mamba, primarily because <i>Mamba is more memory efficient</i>. I generally used a batch size of 16 for the shorter context models used in domain adaptation (32 - 96 tokens).</li>
  <li><strong>Data mixtures:</strong>
    <ul>
      <li><strong>DeBERTa model with 1024 token context:</strong> 99% Pile, 1% Persuade
        <ul>
          <li>The other datasets didn't seem to help on the public LB with long-context models, but our best single model on the private LB scored 0.976 with the data mixture used for the final submission's shorter-context DeBERTa models.</li>
        </ul>
      </li>
      <li><strong>Short-context "student" DeBERTa models:</strong> 62% Pile, 32% Pajama, 5% Tricky Crawl, 1% Persuade</li>
      <li><strong>Mamba:</strong> 77% Pile, 20% Pajama, 2% Tricky Crawl, 1% Persuade</li>
    </ul>
  </li>
  <li><strong>Data volumes:</strong> Typically just did a single epoch with some subsets of the available data undersampled.
    <ul>
      <li><strong>Full-context DeBERTa:</strong> 1.4 million training documents (too impatient to wait >2 days for it to train on more than this and doubt it would help much without an increase in <i>data diversity</i>)</li>
      <li><strong>Short-context DeBERTa:</strong> 1 million training documents (these models are finetuned at test time, so I didn't see any benefit pretraining longer)</li>
      <li><strong>Mamba:</strong> 350k - 1.25M training documents (would have trained on more if I had time - see section about DeBERTa vs. Mamba for rant about how this was rushed and non-ideal)</li>
    </ul>
  </li>
</ul>

# Alternatives we experimented with
* 1D conv ResNets (scored <= 0.87 on the LB, but EXTREMELY fast, ~10 minutes to evaluate on the leaderboard without GPU acceleration - might have scored better if I scaled it up, but DeBERTa gave better results right off the bat)
* Training deep learning based solutions using only Persuade related data (they didn't generalize well to the LB data - scaling up to bigger transformers doesn't help with this)
* Ensembling DeBERTa models with tf-idf (scored well on the public LB but got wrecked during the shakeup) 
* Smaller DeBERTa models (they don't score as well)
* Just using models with smaller context windows (they don't score as well)
* Tried an iterative pseudolabeling strategy similar to the one used in @asalhi's 21st place solution ([https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470148](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470148)). Discarded this due to big random score variations that seemed to depend on how the training/test data is shuffled. The domain adaptation strategy used in our 5th place solution scores well more consistently.
