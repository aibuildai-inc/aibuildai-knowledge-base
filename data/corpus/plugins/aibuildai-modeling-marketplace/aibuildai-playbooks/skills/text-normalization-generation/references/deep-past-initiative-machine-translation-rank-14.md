# LLM + Online Learning is all you need

Competition: deep-past-initiative-machine-translation
Rank: #14
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/llm-online-learning-is-all-you-need

Hi Kaggle

Thanks so much for the host and I am very delighted to join the competition. We are the lucky team and did not shake down.
Big Thanks my teammates zxc and Baiph, we work very hard and share the equal credits to the final results

## Overview of our solution
First of all, I did not think the quality of training data is not as good as expected.  We need lots of work to clean and format training data. In addition, we need to extract or pseudo-label from publication.csv or published_texts.csv. The test and train data varies lots and cv is not stable 

The test data is only 4000 sentences we can apply online leanring training strategy to improve the model performance further.

As for data, I used  Gemini 3.1 with web tool to generate external data.

As for model selection, we used LLM model as the backbone we tried qwne3,qwen2.5, phi-4, byt5

As for training strategy,  total pipeline can be divided into three stages, SFT (knowledge distial )-> RL (reward hacking ) -> Online Learning (fit test distribution), we found that online learning only worked if model archs are very different.

## Step1: Generate data from  publication.csv
I used Gemini 3.1 flash to extract all the sentence level  transliteration-translation pairs. The instructions are shown as following


    prompt = f"""
     Help me extract Old Assyrian transliteration and its translation
    
     Rules:
        1. Output a valid JSON strings.
        2. Only complete sentences need be extracted
        3. Break both the Akkadian transliteration and the matching English translation into sentences and align them pairwise.
        4. clean OCR mistakes in sentence_transliteration and sentence_translation
        5. remove unnecessary characters sentence_transliteration and sentence_translation

    Input:
    {page_text}

    The output should only be in json format like the below
    {output_example}"""




I did use the above data as sft data, but Qwen got lots of Hallucination issues. Maybe it need to be further clean or formating


## Step2:  Pseudo-labelling for  published_texts.csv
I used above data from Step1 to build the doc database, and use Qwen3 8B Embedding to retrieval top 3 samples and concat them togther to generate more relable translation .



After that I used it to SFT data, Qwen3 14B model got private 37.9 and public 36.1

## Step3:  Clean train.csv and RL
I used the same apporach RAG to clean train.csv, but I filter out the bad data using the following prompt, only bad data need to be regenerated.



After that I used it to RL data, Qwen3 14B model got private 38.1 and public 36.9

## Step4:  Online learning
My teammates think online learning can overcome the shift of data distribution between train and test data

We utilized Qwen3-14B, Qwen2.5-14B, and Phi-4 to generate **pseudo-labels** for the entire test dataset. Subsequently, we trained a ByT5-Large model and used its inference results to further fine-tune the LLMs. This **cross-model iterative process** was repeated to progressively enhance performance. Notably, we observed that **self-training** (e.g., training an LLM on LLM outputs or ByT5 on ByT5 outputs) yielded marginal improvements. Significant gains were only achieved through **heterogeneous iteration**—alternating between different model architectures. Finally, we ensembled the predictions from these 'online-trained' models to produce our final submission.

Hope you enjoy our solutions.
