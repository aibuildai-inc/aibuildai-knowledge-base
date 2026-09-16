# 25th Place Solution for DAIGT (Public LB: 0.966 Private LB: 0.927)

Competition: llm-detect-ai-generated-text
Rank: #25
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470134

We are very happy to share our solution and are so grateful to everyone on Kaggle for sharing your ideas, codes, and datasets.

## Overview
Our final solution follows this workflow:
1. Data cleaning and typo correction
2. BPE tokenizer training + TF-IDF feature extraction
3. Models: MultinomialNB x 1, SGDClassifier x 1, LGBMClassifier x 1, CatBoostClassifier x 1
4. Ensemble and identification of samples with prediction probability between 0.05 and 0.45 as hard samples
5. Use the Mistral-7B model to predict the hard samples and DistilBert to predict all samples
6. Use samples with blend probabilities less than 0.05 and greater than 0.5 as labeled data for unsupervised learning
7. Blend the results of steps 4, 5, and 6 for the final prediction

## Solution Details

1. **Data cleaning and typo correction**

    Inspired by [piotrkoz's Discussion](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/468767), we delved deeper into typo correction. We began by correcting large-scale character substitutions, followed by identifying and correcting words with a Levenshtein distance of 1 to misspelled words. This process was applied to both the training and test data to ensure consistency.
    ```Python
    text = "Thii code handlei the tokenization of your text data uiing the tokenizer from Hugging Face'i Traniformeri library."
    text = correct_substitutions(text, 0.06)
    text = correct_dist1(text)
    # "This code handles the tokenization of your text data using the tokenizer from Hugging Face's Transformers library."
    ```
    These corrections provided a **~0.01** improvement on the private LB.

2. **LLMs for ensemble**

    We fine-tuned Mistral-7B with LORA on the DAIGT-v2 dataset, which yielded a 0.884 on the public LB. Although the model did not perform as well on the private LB, we had no way of doing this before the competition ended. We also attempted to fine-tune DeBERTa-v3 but were unable to improve upon the public model. Thus, we used the public DistilRoberta model and our Mistral-7B model to test data predictions.

    We used the LLM predictions only for "hard samples" because we observed that TF-IDF features could robustly predict most samples, and LLMs might reverse predictions on some samples.

3. **Semi-supervised Learning**

    We were impressed by [aerdem4's notebook](https://www.kaggle.com/code/aerdem4/daigt-superfast-unsupervised-baseline), which achieved a 0.91 on the public LB using only the test set and two "magic words".  However, we questioned the method's robustness, as its performance heavily depended on the choice of magic words and the distribution of samples for each prompt ID. This could also explain the model's underperformance on the private LB. Ultimately, we discarded the magic words approach and used TF-IDF prediction probabilities to label samples as data, which led to a **~0.02** improvement on the private LB on the last day.

## Team Members
Gengyang Xiao @adrianxiao, Jiayang Zhao @superxiaotang, Junrui Wang @junruiwang, Zhiyang Zhang @zyzhang0109 (in alphabetical order)

Feel free to connect with any of us via Kaggle or LinkedIn.

## Acknowledgement
Many thanks to @thedrcat, @aerdem4, @mustafakeser4, @piotrkoz. 

We learned a lot from you. Thank you!
