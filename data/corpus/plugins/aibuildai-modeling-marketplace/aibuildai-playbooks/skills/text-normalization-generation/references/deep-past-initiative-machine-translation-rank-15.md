# 15th Place Gold Solution

Competition: deep-past-initiative-machine-translation
Rank: #15
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/15th-place-gold-solution

Thank you Kaggle for hosting the competition. Special thanks to our host @deeppast for proactively clarifying our queries and helping in the release of revised higher quality datasets, which made a real difference. Lastly, I would like to thank my teammate @aman1391 for forcefully inviting me to join this competition (please continue doing so 😆).

If we're being honest, the first few weeks were humbling. We experimented with focal loss, annotation tricks, and various architectural changes; yet the LB barely moved. The breakthrough came when we revisited the host's [discussion](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/678899) on data quality, and it crystallized something we had been overlooking: this competition was never about clever models. It was about DATA. The moment we shifted our entire focus to building robust OCR pipelines, curating clean parallel text from academic publications, and generating high-quality pseudo-labels — everything changed. Sometimes the most impactful insight is the one hiding in plain sight.

## Summary

We trained `ByT5` and `Qwen3` models on competition data, PDF data, and pseudo-labelled (PL) data created by a Gemma3-27B model trained on competition and PDF data. We created PDF data iteratively and trained our models. Our final submission consisted of an MBR ensemble of `ByT5 Small`, `ByT5-Base`, `Qwen3-14B`, and `Qwen3-4B-Instruct` models using `sacrebleu.metrics.CHRF(word_order=2).sentence_score` as MBR metric. We used beam search for ByT5s and temperature sampling for Qwen models.


## PDF Data Extraction

We leveraged `zai-org/GLM-OCR` to extract text from the PDF collection shared by the host. We ran OCR in two ways:

1. Run OCR on the full page
    - higher text quality
    - renders characters appropriately
    - lacks structure with a blob of Akkadian text followed by a blob of English/other language text
2. Run OCR on Tesseract-cropped portions
    - provides layout signals since text can be ordered 

We then leverage models hosted on [Groq](https://console.groq.com/docs/models) to generate aligned Akkadian-English data. We align the text data to belong to the full letter and use `Kimi K2` to get the parallel data. Example prompt:

```
# Context
You are given a Akkadian-Engligh texts in 2 formats.

## Format 1: Interleaved Akkadian and English
- Akkadian and English interleaved line-by-line
- Extracted from cropping each line of text using Tesseract
- Useful for understanding alignment
- Some of the Akkadian Text might be noisy, refer to the Format 2 to understand the rendering

## Format 2: 
- Akkadian text is clustered, followed by English
- Extracted by OCRing the entire page of text
- Akkadian text is better rendered, though alignment of Akkadian and English is not present

# Role

Your task is to extract Akkadian-English parallel data, understand alignment from `Format 1` and text rendering from `Format 2`.
The portion of Akkadian Text in `Format 1` must be followed in the parallel data as well i.e. **Do Not** merge consecutive Akkadian lines of text in `Format 1` into a single piece of text
The English translations need not be formal, complete or grammatically correct.

**Remember**

- Don't hallucinate or generate your own translations, strictly pick text from the given context
- Don't include XMLs in Akkadian text (like <sub>4</sub>), render them as characters
- The English translations need not be full sentences

```

We used different prompts for different PDF depending on the letter layout. For example, some Turkish translation PDFs had translations with line markers at the end.

```
# Context
You are given a Akkadian-Turkish text.

- Akkadian text is clustered, followed by it's translation in Turkish
- Extracted by OCRing the entire page of text


# Role

Your task is to extract Akkadian-Turkish-English parallel data, Akkadian and Turkish are present in the text, the Turkish text needs to be translated to English
The Akkadian and Turkish text alignment in present, you need to detect it via line numbers (eg. "1-5)" -> Turkish of Akkdian text from lines 1-5 )
The English translations need not be formal, complete or grammatically correct.

**Remember**

- Strictly pick Akkadian and Turkish text from the given context and correctly translated Turkish into English
- Don't include XMLs in Akkadian text (like <sub>4</sub>), render them as characters
- The English translations need not be full sentences
- Only extract cases where you are confident of the aligned translation

```

AKT-6 series PDFs were not easy to extract, and the above approaches did not work. A multi-modal approach with `meta-llama/llama-4-maverick-17b-128e-instruct` did seem to work, but the model was deprecated before we could process the data (PDF page images were super-resolutioned using a [light-weight model](https://huggingface.co/spaces/bookbot/Image-Upscaling-Playground)). Finally, we used our extracted OCR text with `Claude` to process them.


## Validation

We held-out 10% of the competition data to track our cv.

## Data Pre/Post Processing

We used Claude extensively to prepare our data preprocessing code as per the guidelines shared by the host, building cleaning pipelines that handled everything from LaTeX artifacts, superscript/subscript normalization, OCR character fixes (İ→I, ḥ→h, ß→š), Sumerogram expansion (KÙ.B.→KÙ.BABBAR), gap synchronization between transliteration and translation, and filtering misaligned Turkish/English rows. We also built a name correction postprocessor using lookup dictionaries that standardized proper name spellings across model outputs (e.g., ensuring "Šalim-Aššur" was consistently rendered regardless of which model produced it). Beyond preprocessing, we used Claude to get quality assessments of every OCR-extracted PDF; checking charset compliance, keyword alignment percentages, and identifying which sources were clean enough to train on and which needed filtering.

## Training

Our initial focus was to maximize the performance of `ByT5` and `Qwen3-4B` based on the competition data and our processing. Once our PDF extraction pipeline was in shape, we extracted all PDFs iteratively and trained our models periodically. We extracted ~31k samples from PDFs. We also found `Gemma3-27B` to learn well from (competition + PDF) data and used it to generate PL data for texts found in `published_texts.csv`. To increase PL data quality, we ensembled Gemma models using MBR. To include diversity in our submission, we also trained `Qwen3-14B`. All models were trained on the data mixture of competition, PDF, and PL. ByT5 models were trained with code adapted from public kernel, and Qwen models were trained using [unsloth](https://unsloth.ai/)

### Knowledge Distillation

We also trained a `Gemma3-27B` on all PDF data and used it to predict the competition data. The idea was simple: Gemma learned from Akkadian→English translation patterns and was used to generate "teacher" translations for the competition data. These Gemma translations served as soft labels for knowledge distillation into our ByT5 student models, where each training row appeared twice: once with the ground truth label and once with Gemma's translation at a reduced weight. Including this KD model provided diversity to our ensemble.

## Ensemble

Our private submission consists of 5 ByT5 models, 1 Qwen3-14B (inferred with temperature 0.0 and 0.7), and 2 Qwen3-4B models. The generation from all models was ensembled using MBR with `sacrebleu.metrics.CHRF(word_order=2).sentence_score`. This was recommended by `Claude`; the competition metric did not seem to work well for us. Our submission runs for almost 9 hours. 


## Failed Experiments

* ByT5 Large or XL did not work with our setup
* We had an idea similar to the CPT approach adopted by other teams. Our approach was to train a CPT and SFT model and then model-soup them together to get the benefits, which did not work (model soups in general did not work for us). We should have just trained in a staged manner
* We tried training `Qwen3.5` models for generating PL, but were not able to train them successfully
* Our experiment with normalizing PN/GN form with placeholders did not work
