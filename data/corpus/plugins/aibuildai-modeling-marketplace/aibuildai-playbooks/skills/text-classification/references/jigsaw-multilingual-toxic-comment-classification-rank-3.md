# 3rd Place Solution

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #3
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160964

First of all we are grateful to Jigsaw for organizing such an interesting competition, and flawless execution in terms of data quality, train / validation/ test split etc. Also very thankful to Kaggle for providing the free TPUs which made quick experimentation possible. This is the first time I used just Kaggle Notebook (kernels) for training / inference and was impressed with how well they work most of the time, with a slight exception of lacking full blown IDE functionality, love everything about the Kernels. 

Last, but not the least, I want to thank my amazing teammates - @brightertiger (Ujjwal), @soloway (Igor) and@drpatrickchan (Dr Patrick) - who made our LB position possible and were super fun to work with. Our best submission was made in the last 15 min of the competition, so this was a relentless team, which never stopped improving.  We were constantly looking and re-evaluating new angles to improve our solution.  Although, I am writing the solution description, but I speak on everyone’s behalf here. 

Following were the major parts of our best submission:

**RoBERTa XLM pre-trained**: Like we saw with all the publicly shared kernels, RoBERTa XLM was the workhorse, which just delivered great performance with minimum effort / training. Used a lot of things discussed in the forum like translated data, open subtitles data along with averaging across multiple folds. Typically about ~200K or so observations were enough to train a single model, so there was a lot of scope to do multi-fold averaging given the huge data available at hand. A big thanks to @shonenkov and @xhlulu for their excellent kernels. 

**Language Specific Pre-trained Bert Models**: Used language specific pre-trained Bert models for all the 6 test languages. A given model with a given capacity is any day more powerful for a single language vs multiple languages. The only challenge was to merge the probabilities coming out of different models (which could have different distributions) into a single ranking which could be especially problematic for a ROC evaluation metric (Even if you got the rank ordering perfectly right within each language you can mess things up when combining across languages). Our teammate Igor could make it work magically. Big thanks to @shonenkov for his great [kernel](https://www.kaggle.com/shonenkov/tpu-training-super-fast-xlmroberta ) that is used for training these Bert Models.

**RoBERTa XLM MLM**: Borrowed the concept of training the model on domain specific data using this excellent [kernel](https://www.kaggle.com/riblidezso/finetune-xlm-roberta-on-jigsaw-test-data-with-mlm) by @riblidezso. While the performance from these models were similar to our existing XLM-Roberta models, they worked well in the blend. We added more data to the MLM step by using translations of training data, validation data and test dataset. This gave us a couple versions of the pre-trained model that we averaged in the blend. 

**Post Processing**: We observed that around 5-10% of the comments looked like automated / template based messages. E.g - Look at Test ID 39482. What’s really happening here is - the username / page title / attachment name is quoted by the bot here, which may contain profanities. But most likely due to the way data labels would have been generated, these are likely marked as non-toxic - our models unfortunately get confused by this. We applied regular expressions, clustering and other heuristics to adjust scores coming directly out of the models for these comments. This gave us close to additional 0.001  on LB

Other things which impacted results a tiny bit:
**TTA**: Some of our models did test time augmentation - where we did a weighted average of the prediction over foreign and english language.  
**Label smoothing**: Was used in most of our models including the mono lang models.

A weighted blend of the components # 1-3 put together and post processing described in # 4 gave us our 0.9523 on Public and 0.9509 on Private LB.
