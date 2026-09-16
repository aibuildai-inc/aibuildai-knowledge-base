# 🥈24th in two weeks and `No space left on device`!

Competition: bengaliai-speech
Rank: #24
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/450606

## First of all, THANK YOU ALL!
As a late joiner, It was so helpful to read insightful discussions of @imtiazprio @reasat @tugstugi @hengck23 @mbmmurad and list goes on!

## Approach
* Code: https://github.com/bayartsogt-ya/bengali-speech-2023
* Inference: https://www.kaggle.com/code/bayartsogtya/submit-to-restore-punctuation/notebook
* Backbone Model: [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
* LM: KenLM 5-gram (16G) trained on [IndicCorpV2 corpus](https://github.com/AI4Bharat/IndicBERT#indiccorp-v2) and [Bengali Hate Speech Dataset](https://github.com/rezacsedu/Bengali-Hate-Speech-Dataset/tree/main)
* More Data: Competition data + MadASR2023 + OpenSLR53
* Data Augmentation: `audiomentations.AddBackgroundNoise` using subset of "Bollywood Music", "Applause" and "Theme Music" from [AudioSet dataset](https://research.google.com/audioset/dataset/index.html)
* Restore Punctuation https://github.com/xashru/punctuation-restoration

## Important lesson for future me!
* **`[No space left on device]`** Just write your own custom dataset class!!!
    * Look at https://github.com/bayartsogt-ya/bengali-speech-2023/blob/main/train2.py.
    * Just increase `dataloader_num_workers` if you have enough cores. Preparing input and use `datasets.Dataset.set_transform` is complicated and **not** efficient.
    * Be simple! read it from a file system in `__getitem__` and apply whatever you want on the fly!
* **`[Quality vs Quantity]`** 0.475 on only validation VS 0.421 on train (filtered) validation madasr openslr53 😂
    * It is obvious that filtering on big datasets helps!
* **`[Manually Check Output]`** See where your model is making mistake on your validation data.
    * This helped me to see that punctuations (dari, comma, question mark, etc...) were counted as substitutions and deleted.
* **`[Stop procrastinating on small things]`** You could have checked different chunk_length_s way before deadline. But you did not here! -> This is not calling out you did not try to train whisper!

## Guilt of Overfitting to LB!

Because test data (Out of Distribution) data was so different from train datasets, it was really about overfitting to public leaderboard.
```python
!kaggle competitions submissions -v bengaliai-speech >> ./bengali-speech-submissions.csv
>>> df = pd.read_csv("./data/bengali-speech-submissions.csv")
>>> np.corrcoef(df.publicScore, df.privateScore)[0, 1]
0.990815988867224
...
>>> sns.lineplot(df, y="score", x="date", hue="split")
```


## In the End

It is all about learning!
Even though it is always so frustrating to feel you were so close or so much could have done or should have done, I appreciate this learning path and that's why I joined to Kaggle in the first place! 🫡
