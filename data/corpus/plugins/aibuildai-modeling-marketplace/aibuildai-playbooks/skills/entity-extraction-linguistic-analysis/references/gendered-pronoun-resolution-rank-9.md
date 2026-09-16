# 9th place solution overview [0.20238]

Competition: gendered-pronoun-resolution
Rank: #9
Source: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90421#latest-525037

I want to start by thanking the Kaggle organizers for a great competition and all the community members for keeping the discussions alive and hosting amazing kernels. Secondly, I want to thank the Hugging Face team for open sourcing a fantastic library to play around with BERT models. My solution is a pure BERT fine-tuning based approach. It's an ensemble of three tasks - Question Answering (SQUAD), Multiple Choice (SWAG) and a span classification approach. Of these three, the most interesting one is the SQUAD model - I added the SWAG and span classification models much later - but they do add value. I've seen a couple of posts where people were curious about using a question answering system - so I will go into a bit of the detail about how I did it. I will also share the code once I do some cleanup and hopefully, that'll be of help too.

### Pro-reference resolution as a Question Answering Task
This was the first modeling idea that came to my mind when I saw this contest. And I immediately saw promising results with BERT fine-tuning. The interesting bit of this task IMO is actually asking the right question. It didn't seem like a good idea to have the question be something like "What does "he/she" refer to?" because i) There can be multiple "he/she"s and there's no easy way to disambiguate that for the model and ii) That sort of question is more human-like than machine-like in the sense that the words "refer" etc won't really help the model get any better understanding of the task.

So, instead, I feed it the "context tokens" surrounding the pronoun to be resolved as the "question". The answer is the entity itself. It's fascinating that the model learns to do co-reference resolution with this fuzzy input - meaning it learned to resolve a pronoun to an entity name even though that isn't directly posed as a classification task. This model is actually more powerful than what's necessary for this competition - in the sense that it produces the entity name corresponding to the pronoun without using the knowledge of A and B choices. And secondly, because of the nature of the task, we get scores (logits) over the token sequence (and not between A, B and neither). So I use another simple logistic regression with the logits produced by the BERT QA model as inputs.  The logits are only extracted for the relevant A, B, and pronoun spans. This logistic regression then gives us the necessary probabilities. There's one additional detail here - the presence of "Neither" category means we wouldn't have the exact answer for those examples. I dealt with this by removing those examples from the training data. This would hurt the performance a bit because of the reduced training data - but the good thing is that the model can always predict, by design, an entity that's neither A nor B - so there is no need to explicitly represent the "neither" examples during training. There are still other details that had to be taken care of - I'll try to elaborate them in a follow-up post when I share the code!

### SWAG &amp; Span Models
For the SWAG model, it's mostly just a clone of the existing "run_swag.py" example [here](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_swag.py) . Just had to do a few minor modifications to adapt to this challenge. And for the final SPAN extractor model - it's based off the amazing @ceshine 's [kernel](https://www.kaggle.com/ceshine/pytorch-bert-endpointspanextractor-kfold). I use a lower cased tokenizer instead of the uppercased one being used there. And then I modify the weight initializations and tweak a few other hyperparameters.

Then, I do a 5-fold split stratified on the "gender" dimension (as opposed to "label/target" in some public kernels). It seemed like a reasonable choice given this competition is meant to address the gender bias. The predictions on the test set are the average of the predictions of each fold.

Finally, I just take an average of all the above three model's predictions. That's it. It's interesting that it performed well in a limited data setting without a single hand-crafted feature.

### Fine tuning
I've implemented all of my kernels in pytorch. The number of layers to unfreeze required a lot of experimentation. The two choices that gave me good results are unfreezing either the last 6 or last 12 layers of the BERT large model. And I do this by setting the "requires_grad" attribute to False for all the parameters corresponding to those encoder layers.

### Failed experiments (a.k.a future work areas)
1) I've not seen any success by fine-tuning all 24 BERT large encoder layers. In fact, I couldn't get nearly as good results if I'd fine tuned more or less than 6 or 12 layers. It still puzzles me as to why that is the case and why they are the magic numbers!
2) For the SQUAD/SWAG models, I tried concatenating an additional embedding vector that encodes the word-piece token level info of whether it belongs to one of A, B or P. I thought this should've been valuable but I couldn't get it to do well in this setting.
3) This was an interesting one. I tried fine-tuning the BERT model in an unsupervised manner by training a language model on the texts extracted from the Wikipedia pages corresponding to the URLs provided. The idea behind this one was to see if I can get better BERT layer representations by leveraging the text from the contest's dataset. This way, the model might have a better initialization point and that can be handy given the limited training supervision. But that didn't seem to help much either and it's a computationally expensive process to run. My guess is that BERT representations are anyway originally obtained by training on Wikipedia. So the fine-tuning on this GAP Dataset which is also based on Wikipedia is probably pointless?
4) I hurriedly tried throwing in the title extracted from the wikipedia page's URL into the token sequence for the SQUAD Model. But the initial results didn't seem promising. 
5) I also tried throwing in a good number of hand engineered features from the public kernels to the models above. They didn't seem to help either.
6) I didn't do any data augmentation. So there might be some potential gains that could be obtained in that direction.
All of these experiments could've used a lot more love. So they definitely make great candidates for future exploration!

Overall, I had a great time contesting. Congrats to all the winners and good luck to all participants for future!

EDIT (10th June, 2019):
Link to the paper: https://arxiv.org/abs/1906.03695
Link to source code: https://github.com/rakeshchada/corefqa
