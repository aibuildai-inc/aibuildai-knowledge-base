# 42th place: a very simple solution

Competition: uspto-explainable-ai
Rank: #42
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522347

The idea of my solution was very simple: using document frequencies to select the most important tokens and then join top-25 most important tokens with "OR" operator.

Calculating document frequencies for CPC codes was easy and computationally not demanding, so I've made it inline in the submission:
```
code_freq = pl.read_parquet('/kaggle/input/uspto-explainable-ai/patent_metadata.parquet')['cpc_codes'].explode().value_counts().sort('count', descending=True)  

code_freq = code_freq.rows_by_key(key=["cpc_codes"], unique=True)
```

Titles were a little bit more heavy, but still not worth a separate notebook:
```
vectorizer = CountVectorizer(max_df=10000, min_df=10, binary=True)

titles_df = pl.scan_parquet('/kaggle/input/uspto-explainable-ai/patent_data/*')\
        .select(['publication_number', 'title'])\
        .collect()

word_title_freq = vectorizer.fit_transform(titles_df['title'])

word_title_freq = dict(
    zip(
        vectorizer.get_feature_names_out(), 
        np.squeeze(np.asarray(word_title_freq.sum(axis=0)))
    )
)
```

Having these global document frequencies for CPC codes and words in titles, I just calculated local document frequencies for patents who were neighbors and calculated importance of a token as `local_document_frequency / global_document_frequency`:
```
flat_list_codes = [code for codes in row['cpc_codes'] for code in codes ]
codes_counter = Counter(flat_list_codes)
weighted_codes = [(f'cpc:{elem}', count / code_freq.get(elem, 1)) for (elem, count) in codes_counter.items()]

titles = [title for title in row['title'] if title]
flat_list_title_words = [token.text for title in titles for token in analyzer(title)]
title_words_counter = Counter(flat_list_title_words)
weighted_title_words = [(f'ti:{elem}', count / word_title_freq.get(elem, 10000)) for (elem, count) in title_words_counter.items()]
```

Then just sorted all the tokens by their importance and combined via OR:
```
selected_operands = sorted(
            weighted_codes + weighted_title_words,
            key=lambda x: x[1], 
            reverse=True)[:25]

return ' OR '.join(selected_operands)
```

I've tried experimenting with same approach for pairs of codes - it brought just a small increase in score. With claims, abstract and description this approach totally failed. Probably because it ignored frequency within a single document: while codes never repeated within same patent and words in titles were rarely repeating, this was no longer the case with large texts.
