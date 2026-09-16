# [20th Place] ByT5-base, span corruption, synth & soup

Competition: deep-past-initiative-machine-translation
Rank: #20
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/20th-place-byt5-base-span-corruption-synth-and-s

A huge thank you to the competition organizers and the Deep Past Initiative for hosting such an unique challenge - never heard about Akkadian language before :)

Our approach focused heavily on data curation (including synthetic translations via Gemini), a 3-phase training pipeline using byt5-base, and model souping to create a robust final ensemble without increasing inference time.

Here is a breakdown of our pipeline.

### **1. Data Preparation & Augmentation**
We started with the provided train.csv, but then thought the difference will be made by having more data (reading the other write-ups not so sure...). 

We managed to scale up to ~80K pairs (average quality):

*Sentence Segmentation:* Segmented the provided competition data into cleaner, sentence-level pairs.

*Gemini extraction & synthetic translation:* Used Gemini to extract and align Akkadian-English parallel data from OCR'd books and academic papers (AKTs plus most of the titles mentioned by the others - lost track at some point...). This created a large, mixed-quality dataset.

*Gold Data:* We curated a smaller, highly refined dataset ("gold data") consisting of the cleanest alignments for late-stage fine-tuning (~10K). Focused on quality of English and allowed noisy Akkadian (except fractions and other markers) thinking would make the model immune to mistakes.

*Extensive Normalization:* Because ByT5 works at the byte/character level, consistent text representation is crucial. We built a normalization pipeline that standardized gaps, converted decimal fractions to Unicode fractions (e.g., 0.5 -> ½, 0.33 -> ⅓), and standardized transliteration characters (e.g., sz -> š, Sa -> ṣa).

###**2. Training Pipeline (3 Phases)**
Went for google/byt5-base stock. Later tested with byt5-large and looking at the final scoring we should have not abandoned it. Looks like our validation data was not ideal.

Added special tokens: < gap >.

##*Phase 1: Unsupervised Denoising (Span Corruption)*
Before teaching the model to translate, we taught it the structure of Akkadian by concatenating all the available transliteration texts (parallel, extended, and extra datasets) and ran an unsupervised span corruption (Masked Language Modeling) task.

Parameters: noise_density = 0.15, mean_noise_span_length = 15.

This helped the model learn the byte-level morphology of Akkadian transliterations.

##*Phase 2: Supervised Fine-Tuning (Mixed Data)*
Using the ~80K mixed-quality dataset, fine-tuned the denoised ByT5 model to translate Akkadian to English.

To ensure stability and diversity, we trained two separate models on this dataset using two different random seeds (famous 42 and the current 2026).

Hyperparameters: Adafactor optimizer, learning_rate=5e-5, bf16=True, gradient accumulation steps=4, and label smoothing (0.1).

##*Phase 3: High-Quality Fine-Tuning (Gold Data)*
Took the model and ran a final, shorter fine-tuning phase (5 epochs) exclusively on the highly refined "Gold Data". This acts as a curriculum learning step, teaching the model to prioritize high-quality academic phrasing and formatting over the noisier OCR data.

###*3. Making the soup 🥣*
For the sake of simplicity and saving compute we used Weight Averaging.

Intermediate Soup (50/50): Averaged the weights of the two Phase 2 models (Seed 42 and Seed 2026) at a 1:1 ratio.

Final Soup (75/25): Took the Intermediate Soup and averaged it with the Phase 3 (Gold Data) model (obviously we first tried running a 2-3 epochs at lower learning rate as a polish, but failed).

Final_Weights = (0.75 * Intermediate_Soup) + (0.25 * Phase_3_Model)

This combination grounded the model in the vast vocabulary of the 80K dataset while gently shifting the distribution toward the high-quality gold data.

###*4. Inference & Post-Processing*
For inference, sequences longer than 512 were dynamically split at the nearest space boundary to avoid truncation, decoded separately, and re-stitched. This increased a bit the score: +0.02ish

Generation Parameters (we know... strange numbers):

num_beams = 12
no_repeat_ngram_size = 20
repetition_penalty = 1.6

####+ some post-processing:

*Capitalization:* regex fixing internal word capitalization errors.

*Terminology Standardization:* Hardcoded fixes for common domain terms (e.g., forcing "pašallu-gold" to "pašallum gold", "transport tariff" to "šadduātum tax"). 

*Onomasticon Fuzzy Matching:* Used difflib.get_close_matches against the onomasticon for composed names containing hypen. If the model generated a composed name but wasn't quite right, the script fuzzy-matched it against the onomasticon (with a 0.7 similarity cutoff) and snapped it to the historically accurate spelling.

Spent a few :) hours but worth every minute.
