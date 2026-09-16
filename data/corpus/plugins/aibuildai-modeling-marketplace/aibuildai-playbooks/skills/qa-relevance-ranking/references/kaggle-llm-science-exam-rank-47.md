# 47th place solution -- Simple Method Based on Open Codes

Competition: kaggle-llm-science-exam
Rank: #47
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446788

# Thanks 
First and foremost, we would like to express our sincere gratitude to the hosts and the Kaggle team for organizing this amazing competition.  Thank my team member @yuanji1239  for hard work. Thank You to Chris Deotte , MGoksu ,MB and Radek to share the OpenBook technique!  

# RAG 

## Dataset 
1. [Chris Deotte's 60k datasets](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/436383) by @cderotte  

- this dataset with original contexts is split into train, eval, test_offline datasets for training deberta-v3-large model. 


2. [Radek's 6.5k dataset](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/425941)  

- using [tfidf recall codes](https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles) on this dataset. 

- this dataset with tfidf recall contexts is divided into train, eval datasets for training deberta-v3-large. 


## Retrival methods
1. [all-MiniLM-L6-v2  and tfidf methods](https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles)  based on @mbanaei sharing.  

2. [bge-small-faiss by @simjeg ](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/443334) and tfidf methods. 


# Training 
1. [train model code by @cderotte  ](https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1) based on (Dataset 1 - 60k datasets) , and select 6000--24000 samples and set many seeds, and generate models. Then, select best model as my model 1.  

2. Based on 1 code,  we add noise by this codes and retrain my model 1, and generate my model 2. 

```
def calu_lenq85(df, col): 
    '''
    # calu_lenq85(df_train, "context")  ## 930 
    # calu_lenq85(df_valid, "context")  ## 877     
    '''
    ## mask == noise 
    a = df[col].str.split(" ").apply(len) 
    len_q85 = round(np.quantile(a, q=0.85))  
    return len_q85 


def mask_string(s, len_q85=930): 
    s = s.split(" ")
    s = np.array(s) 
    mask = tokenizer.mask_token 
    
    ##  mask for some items 
    if len(s) > len_q85: 
        ## mask ratio 
        num_replacements = np.random.uniform(low=0.0, high=0.08) 
        num_replacements = round(len(s) * num_replacements)

        # mask numbers 
        maskidx = np.random.choice(len(s), size=num_replacements, replace=False)
        s[maskidx] = mask 
        
    ## strings 
    s = " ".join(s.tolist()) 
    return s 
```

3. Training 1 and 2 for (Dataset 2 -  6.5k dataset ) ，and add revised prompts to options by using below code. Train models and generate models 3 and 4.  

```
def generate_prompt_option(df): 
    cond1 = lambda x: True if 'What is' in x["prompt"] else False
    cond1 = df.apply(cond1, axis=1)
    for col in list("ABCDE"): 
        df.loc[cond1, col] = df.loc[cond1, "prompt"].apply(lambda x: x[7:-1]) + ' ' + "is that" + ' ' + df.loc[cond1, col] 
    return df 
```


# Inference 
the ensemble mothod of 4 deberta models and [longformer model](https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles) by @yuanji1239 to infer online test dataset. 


# Thanks for Dr.JI Yuan, all teams and Kaggle 😄
