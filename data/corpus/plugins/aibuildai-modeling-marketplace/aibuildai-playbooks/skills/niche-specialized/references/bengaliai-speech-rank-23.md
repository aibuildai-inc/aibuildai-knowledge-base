# 23nd Solution - My first competition medal

Competition: bengaliai-speech
Rank: #23
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/448343

I would like to express my gratitude to the organizers of this competition. As a relative newcomer to Automatic Speech Recognition (ASR), I'm thrilled to have secured a medal in this event. I am immensely satisfied with the outcome, and I extend my thanks to both the competition participants and its hosts.


**The solution consists of 2 components:**

1.  ASR model
2. Ngram KenLM model


**1. ASR model**
Initially, I experimented with numerous Hugging Face (HF) models, training them on approximately 20,000 data samples. The models I tested included:
- LegolasTheElf/Wav2Vec2_XLSR_Bengali_1b
- arijitx/wav2vec2-large-xlsr-bengali
- arijitx/wav2vec2-xls-r-300m-bengali
- tanmoyio/wav2vec2-large-xlsr-bengali
- ai4bharat/indicwav2vec_v1_bengali
- Umong/wav2vec2-large-mms-1b-bengali
- kabir5297/Wav2Vec2-90k-Bengali
- tanmoyio/wav2vec2-large-xlsr-bengali
- bayartsogt/bengali-2023-0016
- shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice
- jonatasgrosman/wav2vec2-large-xlsr-53-english
- wav2vec2-xls-r-2b - training from scratch
- wav2vec2-xls-r-1b - training from scratch

Eventually, I settled on the "shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice" model. You can find all the fine-tuned models in my HF repository [https://huggingface.co/Aspik101](url)
The final fine-tuned model can be accessed: [https://huggingface.co/Aspik101/shahruk10_checkpoint-360_2444](url)

The final model underwent a two-step training process. In the first step, the model was trained on filtered data, where the Word Error Rate (WER) was less than 80, sourced from the yellowking model, MADASR dataset, and google/fleurs. This training was executed based on a streaming mode and took approximately 20 hours.

```python
trainer = Trainer(
        model=model,
        data_collator=daata_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=IterableWrapper(train),
        eval_dataset=IterableWrapper(val),
        tokenizer=processor.feature_extractor)
```
Loading data via streaming on Hugging Face can be useful when processing large data sets or when you want to feed data to a model while it is running rather than loading the entire data set into memory at once.

At this stage, the best result I managed to get was around 0.424 on the leaderboard

In the next step, the model was trained using data from the competition but from the **examples folder!** Thanks to this, I managed to get results of **0.404.** with KenLM model.


**2. Ngram KenLM model**
5-gram kenlm model trained on competition dataset external Bengali corpus IndicCorp V1+V2.


**What doesn't work**
- Punctuation model [https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase](url) On the test data set, my calculations show that I should get about 0.02 WER, unfortunately it did not work on the final data set
- Attempting to use a large language model to correct typos, such as Qwen 7b, from [https://huggingface.co/Qwen/Qwen-7B-Chat](url) showed potential but did not yield successful results in my case.
- I experimented with the "indicparser" as follows:
```python
from indicparser import graphemeParserg
gp=graphemeParser("bangla")
```
- Exploring the stacking of logits from various wav2vec models.
- Generating transcriptions using the Conformer model available [https://huggingface.co/bengaliAI/BanglaConformer](url), while considering only punctuation and combining it with the final wav2vec model.
- And many other ideas that I have already forgotten

In conclusion, I want to reiterate my appreciation to the competition organizers and participants. This journey has been an incredible learning experience for me, and I'm excited to continue exploring the world of ASR.
