# 3rd Place Efficiency Solution

Competition: commonlit-evaluate-student-summaries
Rank: #32
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/448604

# 1. Summary and the Magic
I trained a deberta-v3-xsmall, running in ~20 minutes. With additional steps for another 4 minuts.
My solution is very similar to the 1st Place on the Efficiency Leaderboard, see [Shochoshi write-up](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/447944). The main idea was to give the model as much information as possible from the original text without increasing the token number by much. I did that by looking at sequences from the original text occurring in the prompt text and putting braces around them, the code and examples for this step can be seen [here](https://www.kaggle.com/code/raki21/marked-summary-gen-efficient-v3xsmall/notebook). If I had known of the optional token_type_ids I would probably have gone with these instead of using braces that use up a few additional tokens. I also needed to focus on longer sequences for that reason (at least 3-gram) while the winner focused especially on short sequences. I would have used up lot's of tokens to mark trivial single word overlaps like 'the' if I focused on short ones with my approach. I also utilized multiprocessing and tried out tons of tricks, but most did not stick. 

# 2. Getting Sidetracked
I also pursued the performance track, after seeing how effective the sequence length reduction was, hoping I could get a good medal by just being able to iterate more quickly and use huge ensembles. I also did not expect to win any prize money and because no medals are given for the Efficiency track my focus shifted towards the safer bet of the performance track.
In the end I split my submissions 2 for the performance and 1 for the efficiency track, getting 32nd place in performance and 3rd in Efficiency. The write-up for the performance track focusing more on topics like ensembling is [here](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446631).

# 3. Approach
1. Clean summaries and prompt from whitespaces.
2. Put braces around sequences that appear in the original text with n-gram >=3.
3. Predict content and wording with the same deberta-v3-xsmall (only 1 model here of course).
   3.1 This is done with Ray Multiprocessing to utilize the 4 cores well.

Training was pretty standard with embeddings and first layer frozen.

# 4. Journey
I tried a lot of options, at first I just looked at examples of summaries for different scores, a notebook I shared for that [here](https://www.kaggle.com/code/raki21/summary-examples-by-content-and-wording). I looked up the CommonLit site, tried to make sense of how the grade the student is in might impact mean scores for the metrics and just absorbed information from discussion threads and code. I looked at the efficiency score implementation and made sense of the time/performance tradeoff.

The tradeoff I worked with was based on the assumption that with a baseline of 0.84 and a best assumed score of around 0.42, the divisor of the first part of the efficiency metric would also be 0.420. With 9 hours of runtime / 540 minutes contributing to 1 score I worked with the assumption that 0.008 score is worth around 10 minutes.

I then implemented simple baselines like linear and non-linear models using only length of the summary. I then looked at the biggest error sources and saw that most of them by a large margin are children just copy pasting large parts of the original text. Those are long and rated well by my simple approach, but are rated very poorly by teachers, especially in wording. Just removing all common 3-grams (link ngram), counting the length of the remaining text instead and using this information again in an exponential formula decreases wording RMSE much furthers. I then switched to LGBM and added additional features like relative words copied, counts for N-grams... At this point I noticed content score being pretty good already but the wording prediction still being really bad. For better predictions I added a tiny bert transformer for wording only.

I wanted to use bigger transformers and tried out a lot of approaches to make them more efficient: conversion from torch to ONNX, static and dynamic quantization in torch and ONNX, multiprocessing with ray. In the end only ray sticked. The quantization gave 25% speedup on my CPU but only around 10% on the submission servers CPU, the tradeoff was not good enough as there was a performance drop of course. Maybe I could have made it work with quantization aware training, but then you have a conflict with the early frozen layers (at least I think so as I found nothing about the standard weights being quantized already). In the end I only had around 15 days to invest in total and a lot of that time was gone at this point.

I also tried misspelling correction, averaging weights of multiple models (does not work at all), varying token length, adjusting for token length > max_length by adding some score to the prediction for longer sequences. I also tried post processing with LGBMs (which are pretty lightweight and don't need much compute), but that did not work out well at first, I got LGBMs working in the end using a trick, but only at the last day and I only used that for my final performance submission, I described this in more detail in my performance write-up. I switched almost everything to deberta-v3-xsmall prediction after working on the performance track for a while. In an error analysis afterwards I found some texts with tons of tabs being misclassified and solved that byremoving white-spaces with a single space.

| Model                        | Local CV Score | Public Score | Private Score | Inference Time |
|-------------------------------|---------------|---------------|----------------|--------------------------|
| 0-submission                 | 1.030         | 0.840         | 0.985          | <1min                    |
| summary-length-only          | 0.677         | 0.596         | 0.610          | <1min                    |
| cutting 3-grams              | 0.621         | 0.535         | 0.565          | 1 min                    |
| LGBM+tiny-bert               | ?             | 0.471         | 0.515          | 31 min                   |
| v3-xsmall with all tricks    | ?             | 0.455         | 0.491          | 25 min                   |


I had a lot of things left to do: 
I only just got LGBMs behind a strong transformer working when the competition ended, I wanted to throw away some layers of deberta-v3-xsmall completely and see if it improves the time-performance tradeoff. I wanted to try pruning and distillation too. 

# 5. Acknowledgements
I especially want to thank @tsunotsuno for the great baseline.
There were also a lot of other great things I read up on intermediately like how and why to do layer freezing. I want to thank everyone I gained good ideas and insights from this competition and all the people freely sharing their models performance and settings. I also want to give a general thank you to the community, I really have been learning a lot in the last months on Kaggle!
