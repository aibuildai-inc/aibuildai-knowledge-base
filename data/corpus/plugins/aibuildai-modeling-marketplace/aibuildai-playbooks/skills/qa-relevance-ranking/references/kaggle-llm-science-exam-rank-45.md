# 45th Place solutions , learnings , trials and what could have been | 0.919/0.917 submitted too late :D

Competition: kaggle-llm-science-exam
Rank: #45
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/447245

First of all thanks to Kaggle for hosting this competition that open door to trying LLMs and solving problems deemed to be solved by today's llm. Thanks to all the community for generously sharing ideas and code without which most at-least we might not have progressed so far  . 
Also thanks to all the team @urvishp80 @ayaanjang @tonymarkchris as they joined and helped along the way . We were lucky to fix our cv on time but not on time to submit LB score 0.917,0.919 private but still was good figured the former on time :) 

# Datasets

## Train datasets

Below are the datasets which worked best for us
1. Chris 60k with context
2. Generating context1 and context2  with @mbanaei tfidf with 270k data and cohere.

## Validation
Thankful for this dataset https://www.kaggle.com/datasets/yeoyunsianggeremie/validation-500 from @yeoyunsianggeremie . this helped us a lot get correlation between cv and lb .

## Retrieval /Infer

1. Openbook minilm , though mpnet-base-v2 also we tried worked well but was not part of our final submission [0.834 LB with openbk on that]. 
2. MB tfidf technique on wiki data created with https://www.kaggle.com/code/nbroad/create-science-wikipedia-dataset. - worked well
3. MB tfidf technique on cohere by MB  - worked welll
4. we didnt use 270k MB dataset for infer retrival.
5. We found the sorting order in MB notebook was reverse/ascending order based on the scores we just did a simple correction on the same which helped
```python
context1 = f"{retrieved_articles[index][-1][2]}\n{retrieved_articles[index][-2][2]}\n{retrieved_articles[index][-3][2]}\n{retrieved_articles[index][-4][2]}"
```

# Training

- CV 0.893 (on 500 dataset)Deberta on 60k with context
- CV 0.8963 (on 500 dataset)Deberta on context1 and context2 tfidf context data trained [interleaved](https://huggingface.co/docs/datasets/stream#interleave) with hugging face datasets example .. https://www.kaggle.com/code/gauravbrills/customdeberta-w-context-seq-mix/notebook - Our best model seems interleaving helped a lot generalizing 
- CV 0.896 (on 500 dataset)deberta trained only on context 1 generated via tfdidg on 60k
- CV 0.836 (on 500 dataset) deberta with 15 freeze layers on 60k dataset
- CV 0.92 Llama but on 100 dataset not good on LB trained with peft for multiplechoiceseqclassifcation.

**Training graph of our best with interleaved dataset 0.8963 one below 0.8917 cv at train**


Params 
```python
warmup_ratio=0.02, 
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
FREEZE_LAYERS = 15
```

# Final ensemble and infer

We did struggle with ensemble till we had the 500 datasets, we were just mixing things up with context and non context models and sometimes llama but never confident of our submission .
Once we got the 500 dataset we just picked our top 3 models and `average` and `optuna` weighted ensemble them .
https://www.kaggle.com/code/gauravbrills/t2-0-901-openbook-tfidf?scriptVersionId=146318816 

We were lucky to get some in for final submission but missed submitting one last one which could have stored 0.817/0.819 in LB as didnt have time .

| cv | Private LB  |Public LB | description|
| --- | ---  | --- | --- |
| 0.9036 |**0.909**  |0.903|before deadline weighted slightly better|
| 0.9017 |**0.909**  |0.905|before deadline SELECTED|
| 0.91 | **0.919297** |0.917|Missed submitting too late 😃|
| 0.908 |**0.917**  |0.912|Missed submitting too late 😃|


# What tried, did not work or could have worked

- **Rerankers** : We tried a bunch of rerankers using ST marco ones but they did not help in Public LB though seems were good in private .  We also reasearched for colbert and https://huggingface.co/ibm/re2g-reranker-nq which other teams tried but somehow lost confidence
- Training on diff context opebook tfidf , openbook we were also trying to generate context with to check how well that works but we could not finish on time  
- dataset cleaning . we did fiddle around with mwparaserfromhell,LatexNodes2Text and beautiful soup but just lost patience or didnt go ahead in cleaning wiki data from scratch . Seems some top teams did explore this and implemented this better.
```python
from pylatexenc.latex2text import LatexNodes2Text
from bs4 import BeautifulSoup
import mwparserfromhell
#import wikitextparser as wtp
def context_cleaner(text):
    ## Parse Latex
    l2t = LatexNodes2Text()
    #text = l2t.latex_to_text(text)
    ## Parse wiki media
    #text = wtp.parse(text)
    code = mwparserfromhell.parse(text)
    print(code.filter_templates())
    for template in code.filter_templates(): 
        print(template)
        code.replace(template, l2t.latex_to_text(str(template)))
    text = str(code) 
    text = text.replace("==References==", "").replace("==External links==", "")
    return  text
```
- Using faiss with different embeddings and simple consine similarity via ST , diff embedding faiss idx we found wasn't giving as good scores as tfidf so we dropped trying other embeddings though it was way faster . Cosine similarity was bit gpu intensive so we dropped using it though we think its retrival was good. These all were tried on minilm and we could not finish trying on BGE or e5 on time :( .
- **LLMs** : Urvish in our team did crack hhow to use Llama for Multiple choicce sequence classification , our inference unfortunately was taking around 4-5 hours so we could not at the end add to our ensemble . Plus we were not able to improve much on Llama cv. 
Custom llama by @urvishp80 
```python
class CustomLlamaModel(nn.Module):
    def __init__(self, backbone, num_labels, use_gradient_checkpointing=False):
        super(CustomLlamaModel, self).__init__() 
        self.model = backbone
        self.model.config.use_cache = False
        self.config = self.model.config
        self.num_labels = num_labels

        if use_gradient_checkpointing:
                        self.model.gradient_checkpointing_enable() 
        self.pooler = MeanPooling()
        self.dense = nn.Linear(self.config.hidden_size, 1024) 
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.config.hidden_size, 1, bias=False)


    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.model.model(input_ids=flat_input_ids, attention_mask=flat_attention_mask, output_attentions=False,
            output_hidden_states=True)

        # Get the last hidden state from base model
        # last_hidden_state = outputs[0]
        # # Get the last hidden state from base model
        # last_hidden_state = outputs[0]
        # Get the hidden states from all layers
        last_hidden_state = outputs[1][-1]

        x = self.pooler(last_hidden_state, flat_attention_mask)
        x = self.dropout(x)
        logits = self.classifier(x)
        reshaped_logits = logits.view(batch_size, num_choices) 

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if self.model.config.output_attentions:
            attentions = outputs.attentions
        else:
            attentions = None

        if self.model.config.output_hidden_states:
            hidden_states = outputs.hidden_states
        else:
            hidden_states = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
```

At the end was a great learning experience for all of us and still learning a lot still from this comp to use in the real world. Will update this as recall other things we tried or shall write a blog .
