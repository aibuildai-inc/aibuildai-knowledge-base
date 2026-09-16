# Comprehensive 1st Place Write-Up

Competition: llm-detect-ai-generated-text
Rank: #1
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/473295

Our team is still a bit stunned that we got 1st place (🤯). [Raja posted earlier with a short summary](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121), but here is a more detailed look at our solution and what we think made a difference.


# tldr

The hosts did not reveal much information about what the generated essays would be like, so we aimed to create a diverse set of generated essays in hopes that it would (1) make a model that generalizes and (2) have data similar to the private leaderboard. The modeling approach was less important, as we had multiple single models in the 0.970+ range due to the quality of the dataset. 

# Overview

- [Datamix](#datamix)
- [Models](#models)
- [Ensemble](#ensemble)


# Datamix
Our datamix was created in an incremental way focusing on size, diversity and complexity to facilitate good generalization capabilities and strong resistance against adversarial examples. For each datamix iteration, we attempted to plug blindspots of the previous generation models while maintaining robustness.

To maximally leverage in-domain human texts, we used the entire Persuade corpus comprising all 15 prompts. We also included diverse human texts from sources such as OpenAI GPT2 output dataset, [ELLIPSE corpus](https://github.com/scrosseye/ELLIPSE-Corpus), NarrativeQA, wikipedia, NLTK Brown corpus and IMDB movie reviews.

### Sources for our generated essays can be grouped under four categories:
1. Proprietary LLMs (gpt-3.5, gpt-4, claude, cohere, gemini, palm)
2. Open source LLMs (llama, falcon, mistral, mixtral)
3. Existing LLM generated text datasets
  - [Synthetic dataset made by T5](https://www.kaggle.com/datasets/conjuring92/fpe-processed-dataset?select=mlm_essays_processed.csv)
  - [DAIGT V2 subset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)
  - [OUTFOX](https://github.com/ryuryukke/OUTFOX)
  - [Ghostbuster data](https://github.com/vivek3141/ghostbuster-data)
  - [gpt-2-output-dataset](https://github.com/openai/gpt-2-output-dataset)
4. Fine-tuned open-source LLMs (mistral, llama, falcon, deci-lm, t5, pythia, BLOOM, GPT2).  For LLM fine-tuning, we leveraged the [PERSUADE corpus](https://github.com/scrosseye/persuade_corpus_2.0) in different ways:
  - Instruction tuning: Instructions were composed of different metadata e.g. prompt name, holistic essay score, ELL status and grade level. Responses were the corresponding student essays.
  - One topic held out: LLMs fine-tuned on PERSUADE essays with one prompt held out. When generating, only the held out prompt essays were generated. This was done to encourage new writing styles.
  - Span wise generation: Generate one span (discourse) at a time conditioned on the remaining essay.

For fine-tuning LLMs we used autotrain and custom code based on the transformers library. 

We used a wide variety of generation configs and prompting strategies to promote diversity & complexity to the data. Generated essays leveraged a combination of the following:
  - [Contrastive search](https://huggingface.co/blog/introducing-csearch)
  - Use of Guidance scale, typical_p, suppress_tokens
  - High temperature & large values of top-k 
  - Prompting to fill-in-the-blank: randomly mask words in an essay and asking LLM to reconstruct the original essay (similar to MLM)
  - Prompting without source texts
  - Prompting with source texts
  - Prompting to rewrite existing essays

Finally, we incorporated augmented essays to make our models aware of typical attacks on LLM content detection systems and obfuscations present in the provided training data. We mainly used a combination of the following augmentations on a random subset of essays:
  - Spelling correction
  - Deletion/insertion/swapping of characters
  - Replacement with synonym 
  - Introduce obfuscations
  - Back translation
  - Random capitalization
  - Swap sentence

As a minor detail, we created a heavily pre-processed version (removed special characters, normalized whitespace, and changed to all lowercase) of our datamix. Our hypothesis was a model trained on this version would learn deeper patterns and thus make a significant contribution in ensembling. Furthermore, the heavy pre-processing would reduce the risk of random character attacks in the hidden test set.

Our best performing models were trained on 160k samples (without pre-processing), out of which 40k were human written.

## Models

We adopted the following modelling strategies:
  - LLM (Q)LoRA fine-tuning: Mistral 7b
  - Deberta-v3
    - Classification
    - Custom Tokenizer + MLM + Pseudo label 
    - Ranking
  - Ghostbuster (llama 7b, tiny llama 1.1B)
  - Ahmet’s Unsupervised Approach

We hypothesize that our modelling strategies themselves had a lesser impact on the overall performance as compared to the datamix. Each of our individual models would have been in the gold medal range.


### LLM (Q)LoRA fine-tuning: Mistral 7b

We fine-tuned the mistralai/Mistral-7B-v0.1 backbone using (Q)LoRA with config provided below on our carefully curated datamix.
```
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=[“q_proj”, “k_proj”, “v_proj”, “o_proj”]
    )
```

We also used [autotrain-advanced](https://github.com/huggingface/autotrain-advanced) to train some LLMs for text generation.

### Deberta

#### Classification: deberta-v3-large
This uses the AutoModelForSequenceClassification in transformers
BCE loss with continuous labels (e.g. essays generated with instruction-tuned LLMs assigned less than 1 score)

#### Custom Tokenizer + MLM + Pseudo label:  deberta-v3-small
  - ([Inspiration](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/458522))
  - Derive a custom tokenizer based on train + test essays
  - Train a small model using Masked Language Modeling (MLM) on the train + test essays
  - Enrich training datamix with pseudo labeled test essays
  - Train for one epoch, followed by Inference on test essays
  - Intuition: these steps would provide a unique advantage to the resulting model due to a specialized understanding of the hidden test set.

#### Ranking: deberta-v3-large
Pairwise loss with margin


### Ghostbuster

([original repo here](https://github.com/vivek3141/ghostbuster))
This approach uses the token probs from two small-ish LLMs (originally davinci and ada from OpenAI) as well as the token probs for unigram and trigram models.
The sequence probs go through vector operations when mixing them across models (e.g. divide llm1 probs with llm2 probs, subtract unigram from trigram) and then into a scalar operation (take the max prob of the sequence)
Instead of using models via the OpenAI API, we used Llama 7b and Tiny Llama 1.1B (the models must have the same tokenizer)
The unigram and trigram models were trained using the Ghostbuster repo's code on the nltk Brown corpus using the Llama tokenizer.
The first 25 tokens in the sequence are ignored, as the models would not have much context to generate meaningful representations.

We saw a small boost when fine-tuning the tiny llama model on texts from the PERSUADE corpus
Instead of doing the structured search across all operations, we did 10 operations of each of the four model sequence probs (llama 7b, tiny llama, unigram, trigram) as well as the same 10 operations on the ratio of the following models: llama 7b/tiny llama, llama 7b/unigram, llama 7b/trigram, tiny llama/unigram, tiny llama/trigram, unigram/trigram. 
The ten operations were: min, max, mean, median, 10% quantile, 25% quantile, 75% quantile, 90% quantile, L2 norm, and variance
All together this is 100 features
Instead of doing logistic regression, we used an ensemble of an SVM classifier and a Random Forest Classifier. 
We did not do much tuning to hyperparameters.
Rapids was used to train significantly faster than sklearn


### [Ahmet’s Unsupervised Approach](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465156)
We implemented a variation of this approach by modifying the logic for weak labelling and final score computation.
We used confident predictions from our mistral model for weak labelling
For a given essay, we first picked neighboring human & generated essays based on a dynamic threshold. Thereafter, we used mean aggregation (instead of min aggregation in the original work) of similarity scores over selected essays.

## Ensemble

We used the rankings, rather than the raw prediction values when combining the predictions. The minimum scoring text gets a rank of 1, and the maximum scoring text gets a rank of n, where n is the number of essays in the test set. These ranks are averaged between models and the averaged rank becomes the final value in the “generated” column.  Minor weighting was done based on public LB and intuition. We applied the most weight to mistral-7b models.

Our highest scoring selected submission is [available here](https://www.kaggle.com/code/nbroad/r100-ensemble)

Our code is [available here](https://github.com/rbiswasfc/llm-detect-ai)

Huge shoutout to Raja (@conjuring92) and Udbhav (@ubamba98) for their great contributions!! 👏👏👏
