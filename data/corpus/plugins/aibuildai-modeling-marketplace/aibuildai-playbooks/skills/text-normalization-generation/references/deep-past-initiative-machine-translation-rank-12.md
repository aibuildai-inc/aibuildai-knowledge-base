# [12th] NotebookLM & Single ByT5-Base Model

Competition: deep-past-initiative-machine-translation
Rank: #12
Source: https://www.kaggle.com/c/deep-past-initiative-machine-translation/writeups/12th-place-notebooklm-and-single-byt5-base-model

First of all , I want to thank the organizers for such an interesting competition and I also want to congratulate the winners for the hard work they've done.  

Second, I also want to thank my family for the support, and thank my teammate and brother @hasinarazafindrazaka for his ML experience which has given us a substantial boost to the score.  
  

### Our Humble Approach 
  
We have,  extensively, made use of **Google's NotebookLM platform** for data extraction, data preprocessing, and pseudo-labelling. 
    
We used a single Byt5-base model fine tuned with three stages:    
- 1st stage with all available data (human + pseudo labelled and small custom data) of 81K unique pairs 
- 2nd stage with only human labelled data, ~28K pairs
- 3rd stage LoRA with only the competition training data ~1.5K



### How we built our training data

Creating a csv of sentence pairs with NotebookLM is quite straightforward with its reasoning model of 1 million token context length, and **an OCR of a 'refined quality'.** 

  
 #### How extracted data from PDFs

For most of the pdf that we had to deal with (like the AKT series), they had to be provided as image inputs due to their quality.
   
So, we basically screenshotted every single tablet, and processed them with NotebookLM with a small batch of 1-5 tablets (1-10 images) at a time, hopefully allowing the model to focus more and avoid hallucinations.   
Doing so resulted in more sentence pairs than giving larger batches of tablets.    

   
 
We had found out that we could speed up the process by opening multiple notebook tabs, we just needed to be organized.    

#### How we tried to recover original transliteration from publications.csv and CAD
  
After extracting some tablets and OA sentences, we made use of their tablet reference (eg: BIN 4 , CCT 2) to merge them with their original transliteration found in published_text.csv.    
  
To do so, we had to ask NotebookLM to find the 'aliases' of the tablet reference in published_text.csv, as hard coding a solution wasn't obvious... We gave it as input 2 files , one with 'aliases' and 'note' columns from published_text.csv and the other the list of tablet reference.
  
Then we batched tablets/sentences with ocr_transliteration, ocr_translation merged with original transliteration, and processed them with NotebookLM 

#### How we built a small custom data
  
As we processed data, we 've got a little bit familiar with some basic akkadian formulas, so we had decided to create a custom data that we filled with random names from onomasticon.csv and random numbers with a 0.2 probability of getting < gap >  for both  
The custom data was mostly about Seal, Witness, From ... To , month eponym, and 'owes' statements
    
We also asked NotebookLM to split some of the training data (train.csv), but as we already had sentences from PDFs, we asked him to replace names with PN and numbers with NUM that we randomly replaced later

#### Pseudo-labelling:

We made use of the untranslated texts from published_texts.csv that we asked NotebookLM to translate and split.   
The translation quality was not like a 'SIG5 copper' but we believed there could be some value that we could get out of it 

    
ŠU.NÍGIN, we got a total of ~81K unique pairs: ~28K human-labelled (from PDFs, train.csv included), ~11K of custom data, ~3K (from publications.csv, ... we didn't have much time), and pseudo-labelled data for the rest  
  
### Training and Model
  
We used  a  **single ByT5-base model**

    
We have split the fine tuning into 3 stages:

#####  - 1st stage with all available data , we called it 'pretraining':   
With 81K pairs, a batch size of 8, learning rate of 5e-5, and 5 epoches  
The purpose was that the model would learn akkadian in general and have some good 'momentum'
  
Submitting at this stage would give us ~35+, which was less than a model that was trained on human labelled only which could get up to  ~37 for 25K pairs (at that time). 
  
##### - 2nd stage with only human labelled data:  
With 28K pairs, batch size of 2, lr 1e-5 , and 2 epochs  

The hope was that the model output would align more with human labelled data and correct all biases from pseudo-labelling.  
  
Submitting at this stage would give us 38+
  
On our local validation with these 2 stages we could see a +2 to +3 BLEU score boost compared to a model trained only on human labelled data

#####  - 3rd stage LoRA with only the competition training data:  
With 1.5K , the main purpose was to align the output vocabulary of the model with the competition train.    
Submitting at this stage would give us ~39

#### Some notes

- We considered using APIs, but our ... 'working capital' was quite limited
- We only did very few unsuccessful experiments with ByT5-large
- We tried to extract data from 'Sentences_Oare_FirstWord_LinNum.csv' but result got surprisingly worse either on local validation or public leaderboard
- We plateaued at 38.1 for a long time, even if we added more data (eg ATHE CAD)... Break through only happened after we added some data from publications.csv (CMK, TPAK, APU,...) into the pretraining phase , and further refinement of our dataset 
    
To finish , we want to give tribute to all Assyriologists out there for the truly amazing work they've done , and the Google Team for providing us advanced AI tool and really powerful pretrained models to work with.
