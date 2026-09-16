# 1st Place Solution

Competition: kaggle-llm-science-exam
Rank: #1
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446422

Thanks for this well executed competition that attracted a lot of interested from the community. Thanks a lot as well to my great teammates @ybabakhin and @philippsinger. 

Check out our tldr summary [here](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446240)!

## Context Retrieval

We started working on this competition very early and quickly faced a performance cap with the simple classification approach for both deberta and LLM models.

So, the next step was to explore the context retrieval from Wikipedia articles. Our first submission was with the `all-MiniLM-L6-v2` model. And to our surprise, this first attempt was much better than all our previous non-context models. From that point we were mostly focusing on improving the context retrieval part, while keeping the modeling part almost the same. We tried the majority all models from the top-20 list on MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard. In the end, we had a local choice of about 300 combinations of Retrieval + LLM models + different Wikipedia dumps. Our final submission has contexts from `e5-base-v2`, `e5-large-v2`, `gte-base`, `gte-large` and `bge-large` models.

Apart from exploring various models, we also tried different ways to encode both Wikipedia chunks and Prompts+Answers from the competition data. For Wikipedia we were mostly encoding `title` of the article + article chunk, and for the question the final model has 2 variations:
1. Just a simple concatenation: “{prompt} {A}, {B}, {C}, {D}, {E}”
2. Search contexts individually for each “{prompt} {A}”, “{prompt} {B}”, etc. option, and sort them by similarity in the end.

Our Wikipedia dumps had from 30M to 60M text chunks. However, we’ve implemented a pretty quick and scalable similarity search on GPUs. Firstly, we split the whole 60M database into smaller parts, and load them one after another. Then we calculate a similarity matrix between each part and whole test data (using simple matrix multiplication for torch tensors on GPU). All we have to do now is to store the global Top-k list among all parts. And to make it twice faster we’re doing it on 2 GPUs in parallel.

## Different Wikipedia dumps

As most teams, we started out using the [public wiki](https://www.kaggle.com/datasets/narek1110/wikipedia-22-12-en-embeddings-all-minilm-l6-v2) as our source of context and embeddings. We invested quite some time to create our own wiki corpus. First, we recreated the above wiki taking a snapshot from a newer date closer to the start of the competition. We added a few extra post processing steps, such as joining very small paragraphs (usually subtitles) with the subsequent paragraph. When investigating errors in the context matches, we noticed that all public wikipedia dump parsers are not properly expanding lua code which is often used on wikipedia to render values in scientific articles. To include these into our context, we decided to use the [cirrussearch dump](https://dumps.wikimedia.org/other/cirrussearch/) which is a dump of all (almost) fully rendered wikipedia pages. Unfortunately, this dump doesn’t include newline characters, so we merged the sentences together to different target lengths of either 256, 512 or 1024 characters while not breaking any sentences. In the end we considered all of these wikis (and multiple embeddings for each) for the blend and it helped to increase diversity. As a standalone wikipedia corpus, the cirrussearch wiki with a target length of 512 characters was best for us.

We also tried to filter the wikis for science articles, but never achieved a significant improvement on public leaderboard over using the whole wiki corpus. Apparently, the embedding models are strong enough to not get distracted by irrelevant articles and we decided to play safe and included the full wikis in our final submissions. 

## Longer contexts

We started by using a single most similar wiki chunk for both train and inference. However, we quickly noticed that increasing the context length during the inference might improve the results. The optimal strategy for us was to train models with 3-chunk contexts and run inference with 5-chunks. Increasing it further for training was too slow, and increasing it more for inference was probably adding too much noise to the context. One interesting test that we’ve done was to inverse the order of contexts during the inference which dropped the score quite a bit. So, it might mean that models learnt the position of the “best” context.

## Data and validation strategy

For training we mostly used only the publicly shared datasets. And adding more data wasn’t helpful for our models. For the validation, we used the original 200 samples provided as our validation data since the very beginning. However, after introducing the context retrieval part, we quickly got to 0.99 performance there. For the rest of the competition we were using 6k STEM questions from one of the publicly shared dataset.

## Models

Our final ensemble is a combination of fine-tuned 7B & 13B LLMs. All models are trained in binary fashion, meaning a single sample for each of the five options and final predicted probabilities are used for ranking. This makes the models independent of the ordering of answers. For training data, we found the following steps to be useful:

- Pick synthetically generated multi-choice samples. There are many of such samples shared on Kaggle. Interestingly, data quantity was not too important here to improve performance.
- For each sample, we generate the context using our RAG embedding approaches. This allows us to generate multiple contexts like in inference and adding noise here is helpful for training. Training on the true context the question was generated on, was a bit worse than training on the generated context.

We train all models with LORA on all linear layers. We use a binary classification head after the final next-token logits. We also use a lower learning rate for the head vs. the LORA layers. We did tuning on LORA settings, learning rate, and other minor things, but they were not that impactful as for some other deep learning use cases. All models are trained over a single epoch with typical cosine lr decay and BCE loss, picking the last checkpoint.

We had most success with the following LLM models:

- Llama-2-7b
- Mistral-7B-v0.1
- xgen-7b-8k-base
- Llama-2-13b

The following diagram explains our architecture and inference setup on a high level:



1. First, we feed the context and the question through the backbone and save the past_key_values. For each question, we only need to do this once as each answer has the same context and question. This is a big benefit of decoder only models for such use-cases and saves a lot of runtime.
2. We then have another backbone forward on each of the five answers where each one gets the past_key_values of the previous forward as cache input. We can run this in a batch of five now.
3. We take the final next token prediction of each output which are the logits over the full vocabulary and feed this through a final classification head trained from scratch. This head now learns to map the output of the LLM across the full vocabulary to a single binary prediction whether the answer is correct or not.
4. The final ordering is then simply determined by sorting the predicted probabilities.

One downside of this binary approach is that each sample lacks the cross-information from all the other options which might hold individual signal. We experimented a lot with multi-class model or simply adding the other answers as context to the binary model, but there can be a huge positional bias added by doing so. Hence, we tried to solve this with TTA approaches, using various ordered combinations of the answers, which definitely helped, but was more shaky and could not outcompete the binary model.

We still came up with, what we think is a pretty clever idea, another solution to help the models a bit with learning cross-information across samples not adding a positional bias. Here, the idea is to add the average next token logits of all other options as an additional input to the final classification head. So the final head gets logits for all possible tokens for the answer at hand, and the average of all logits for all possible tokens of all the other four answers. This does not add positional bias and gives extra information that boosted CV and LB and worked well when blended with the simpler approach.

As you may have guessed already from our team name, we used a forked [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) to experiment and to train our models. We have added a fully flexible way of training causal classification models. The [merged PR](https://github.com/h2oai/h2o-llmstudio/pull/449) allows to train binary and multiclass classification LLM models using the core techniques used in our solution. You can just add a csv with a binary target and input text and train models.

## Inference solution

Our final ensemble is a blend of five 7B models and one 13B model. Each model uses a different context approach using different wikis, embeddings and topk. The pipeline fits precisely into 9-hour runtime and utilizes 2.5TB of input data.



We have shared our inference code now:

A winning kernel with 0.933 Private: https://www.kaggle.com/code/ybabakhin/1st-place-team-h2o-llm-studio
A single model with 0.932 Private: https://www.kaggle.com/code/ybabakhin/1st-place-single-model-inference
