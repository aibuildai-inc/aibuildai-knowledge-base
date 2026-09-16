# 9th place solution (🥧)

Competition: pii-detection-removal-from-educational-data
Rank: #9
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497177

I'm very surprised to have moved up so much on private lb. 


## Modeling

1. deberta v3 large
2. deberta v3 large with layer drop and multisample dropout
3. deberta v2 xlarge lora

Since deberta v2 and deberta v3 have different tokenizers, I first got predictions at token level, mapped them to characters, and then took the maximum over the characters that represented the final (spacy) token. All models had max length of 512, stride of 128

## Data

1. a slightly better version of my dataset made with mixtral 8x7b. I added in quotes by famous people (these person names were not pii), as well as phrases like, "With my teammates Joe and Sally" where Joe and Sally are pii, and also phrases like, "please check out my work at my site [URL]". You can easily prompt the model to add this to the essays.
2. a corrected train.json (from competition). There were about 30 or so instances of labels that needed to be changed, so I manually fixed them. 


I would train using 4 fold to find good settings and then train on all data. I also used a callback every epoch to swap out all the names with new ones.



## Postprocessing

- names must be title cased and should not contain anything other than A-Za-z and "."
- remove dr, mr, miss, etc. as being labeled as names. Make sure to change the following token to have B- instead of I-
- If a name is mentioned multiple times in one document and one of them is pii, mark them all as pii
- if an I- prediction does not have B- in front of it, change it to B-
- remove coursera, wikipedia, and .edu urls as predictions
- if a phone number prediction has more than 6 numbers, switch it to id_num
- regex expression for phone numbers


And that's it!



## Final thoughts

It can be really annoying making sure the mappings align properly. I can't tell you how much time I spent debugging alignment errors...


Code here: https://github.com/nbroad1881/pii-data-detection
