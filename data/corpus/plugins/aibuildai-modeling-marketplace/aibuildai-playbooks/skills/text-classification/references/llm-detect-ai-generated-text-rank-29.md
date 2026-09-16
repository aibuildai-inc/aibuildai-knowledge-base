# 29th Place Solution

Competition: llm-detect-ai-generated-text
Rank: #29
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470256

Big thanks to the organizers for holding this competition. It has been a tremendous learning experience. 
I would like to extend my heartfelt thanks to my team members, @drpatrickchan and @jerifate, for their invaluable contributions and collaboration. Additionally, my sincere appreciation goes to the entire community for sharing their insights.

This is the brief summary of the solution.

# TF-IDF Model (from public notebook):
- Source: [LLM DAIGT excluded prompts](https://www.kaggle.com/code/batprem/llm-daigt-excluded-prompts)
- Scores: Private - 0.895, Public - 0.963

# Distil RoBERTa Model (from public notebook):
- Source: [[inference]DetectAI DistilRoberta[0.927]👍](https://www.kaggle.com/code/mustafakeser4/inference-detectai-distilroberta-0-927)
- Scores: Private - 0.884, Public - 0.927

# LSTM&Transformer CNN Approach:
- Inspired by: [Ensemble_TransformerCNN&Roberta](https://www.kaggle.com/code/ichigoe/ensemble-transformercnn-roberta)
- Enhancement: Additional Category Features such as Flesch reading ease, Fog index, Score to grade level, Comprehension level, and Polarity were added.
- Scores: Private - 0.766, Public - 0.926

# Deberta Model:
- Dataset: 900k rows, including various sources like llm-mistral-7b-instruct-texts, daigt-external-dataset, etc.
- Highlight: [daigt-v4-train-dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v4-train-dataset) with magic was particularly helpful for me. Thanks to @thedrcat for [the discussion](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/468246)
- Training Approach: Deberta-v3-large with 1024 max length. Additionally, custom truncation (see below) and label smoothing with a factor of 0.1 were used.
```
def custom_truncate(text, tokenizer_in, max_length=MAX_LEN):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    tokenized_sentences = [
        tokenizer_in.encode(sentence, add_special_tokens=False)
        for sentence in sentences
    ]

    # Calculate the total number of tokens
    total_tokens = sum(len(sentence) for sentence in tokenized_sentences)

    # If the total number of tokens is already within the max_length, return the original text
    if total_tokens <= max_length:
        return text

    # Determine an upper limit for the starting index
    tokens_so_far = 0
    for i, sentence_tokens in enumerate(tokenized_sentences):
        tokens_so_far += len(sentence_tokens)
        if tokens_so_far > max_length:
            max_start_index = i
            break

    # Choose a starting point that isn't too close to the end
    start_index = random.randint(0, max(max_start_index - 1, 0))
    truncated_tokens = []

    # Concatenate tokens of sentences until the max_length is reached
    for sentence_tokens in tokenized_sentences[start_index:]:
        if len(truncated_tokens) + len(sentence_tokens) <= max_length:
            truncated_tokens.extend(sentence_tokens)
        else:
            break

    # Convert tokens back to text
    truncated_text = tokenizer_in.decode(
        truncated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return truncated_text

def tokenize_with_custom_truncation(examples, max_length=MAX_LEN):
    text_li = examples["text"]
    # Apply the custom truncation
    truncated_text_li = []
    for text in text_li:
        truncated_text = custom_truncate(text, tokenizer, max_length)
        truncated_text_li.append(truncated_text)

    # Tokenize the truncated text
    return tokenizer(
        truncated_text_li,
        padding=True,
        truncation=True,
        max_length=max_length,
    )
```
- I believe this randomness helped to have a robust model
- Scores: Private - 0.901, Public - 0.931

# Ensemble Approach:
- Weight Distribution: TF-IDF (0.5), Distil RoBERTa (0.1), Three Tower (0.15), Deberta (0.25)
- Strategy: Balanced emphasis between TF-IDF and other models for effective performance.
- Final scores: Private - 0.921, Public - 0.970

*Update: [the solution code has been released.](https://www.kaggle.com/code/ympaik/29th-solution/notebook)*
