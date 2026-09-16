# 0.577 single model with full code

Competition: feedback-prize-effectiveness
Rank: #14
Source: https://www.kaggle.com/c/feedback-prize-effectiveness/discussion/347713

The first step to achieve this model was to train an mlm on the old dataset. I took all the data from previous feedback competition and fine-tuned deberta-v3-large using `run_mlm.py` script available in the transformers library. The perplexity in the end was around 4.5. I could have trained more but I didn't.

Now comes the more interesting part. I was inspired by @nbroad 's kernel that showed how we can use token classification for this problem. This was the kernel that piqued my interest in this competition and I decided to give it a go. The idea was simple but effective.

Essay texts were represented as follows:

```
[CLS]some text [CLS_{discourse_type}] some valid discourse text [END_{discourse_type}] ..... [SEP]
```

Mapping:

```
disc_types = [
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Evidence",
    "Lead",
    "Position",
    "Rebuttal",
]
cls_tokens_map = {label: f"[CLS_{label.upper()}]" for label in disc_types}
end_tokens_map = {label: f"[END_{label.upper()}]" for label in disc_types}
```

I arranged the data as follows:
- Convert the data to old format (I had already made my kernel public for this part)
- For each token, add on `O` label for discourse types if they are not a valid discourse
- For each token, add -100 as label for discourse effectiveness if they are not a valid discourse
- If the token belongs to one of the discourse cls or end token or is a valid discourse, I added discourse type id as their label
- If the token belongs to one of the discourse cls or end token or is a valid discourse, I added discourse effectiveness id as their label
- discourse effectiveness label was kept only for the CLS_{discourse_type} and END_{discourse_type} tokens

So, now we have for each essay:
- input_ids
- input_types (discourse type ids, one for each input id)
- input_labels (the actual label, most are -100, only start of a discourse and end have an actual label)
- attention_mask

Now it's time to train a model. I made a crucial change in the model's embedding layer that gave a good lift. The change was to add another embedding layer for discourse_types (input_types) and add the result to the original embeddings:

```python
class DebertaV2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)
        self.disc_type_embeddings = nn.Embedding(9, self.embedding_size)
        .
        .
        .

    def forward(
        self, input_ids=None, token_type_ids=None, disc_type_ids=None, position_ids=None, mask=None, inputs_embeds=None
    ):
        .
        .
        if self.config.disc_type_vocab_size > 0:
            disc_type_embeddings = self.disc_type_embeddings(disc_type_ids)
            embeddings += disc_type_embeddings
        .
        .
        .
        return embeddings

```

Other than all this, I used polynomial learning rate scheduler and AdamW optimizer from pytorch.

During inference, I just averaged the probabilities for CLS_{discourse_type} and END_{discourse_type} tokens to get the final label.

That's it. Nothing too complicated to get a good score using a single model (5-fold). Running it with some variations and averaging the models gave us 14th rank on the leaderboard.

The full code to train and infer is available here: https://www.kaggle.com/code/abhishek/0-577-single-model-full-code

I used my own library for training the models: tez. Please give it some love here: https://github.com/abhishekkrthakur/tez

If you have any questions, please feel free to ask. It was a fun competition and I learnt a lot! :)
