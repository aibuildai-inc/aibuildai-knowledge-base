# 14th Place Solution (with notebooks)

Competition: coleridgeinitiative-show-us-the-data
Rank: #14
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248511

One away from gold! I did this competition on Kaggle notebooks only:
- Additional hand-labeled dataset titles (from train only): https://www.kaggle.com/lichena/coleridgehandlabeled
- Preprocessing: https://www.kaggle.com/lichena/coleridge-pre-processing-hand
- Training: https://www.kaggle.com/lichena/coleridge-ner?scriptVersionId=66238738
- Inference: https://www.kaggle.com/lichena/coleridge-ner-inference

My highest chosen private LB submission scored 0.558/.437, while my best scored 0.545/0.444:
- I fine-tuned bert-base-cased + crf on BIO-tagged chunks of ~200-400 words. 
- The key for me was hand-annotating a larger training set. I used regex to find an initial set, then went through the list of false positives for my model that I thought were valid and recategorized them into the positive label set. Ultimately, this larger dataset scored 0.601 publicLB when naively submitted, a subset scored 0.621.
- For training, I used 80% chunks which were positive mentions, 20% chunks which had no mentions. I tried doing a naive cv of 20% chunks with positive mentions, but this didn't generalize well to the private lb
- Instead of the default clean_text() function provided, I also kept casing and parentheticals during training and inference, which boosted the LB score. 

If you're like me and are only doing competitions with kaggle notebooks, here's some tips I learned:
- Use CPU as much as you can - I would choose 2-3 files as the training set and run all the code with those test files first to test bugs
- When you have your pipeline, start with distilled models which are faster to train and save you gpu time. Run it on a small sample of data
- Start the competition early
- Look for competitions where there is a systematic advantage to be had - in this case, hand-annotation provided an "edge" that I knew many others weren't going to attempt to replicate.
