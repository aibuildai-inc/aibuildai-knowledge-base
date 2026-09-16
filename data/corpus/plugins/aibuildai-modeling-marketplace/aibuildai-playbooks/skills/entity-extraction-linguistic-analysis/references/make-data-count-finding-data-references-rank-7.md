# 7th place solution

Competition: make-data-count-finding-data-references
Rank: #7
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/7th-place-solution

Thank you for the hard work of my teammates @zxc123cc , @simingtan , @xuanmingzhang777 , @tonyarobertson 

# Key Points

## 1. Recall

### 1.1 DOI

(1) For V4 data, obtain the DOI from V4 based on article_id, and filter based on whether it appears in the article to identify DOI data.

(2) If not in V4, follow the normal process.

### 1.2 ACC

External data: https://europepmc.org/pub/databases/pmc/TextMinedTerms/

All ACC IDs in the training set are included in this dataset. We obtain the relationship between article ID and ACC ID based on the exist ID column, then retrieve the ACC ID from the external data based on article_id, and filter based on whether it appears in the article to identify ACC ID data. The following data is excluded, with only ACC submitted, type SAMN set to P, and others set to S, resulting in lb=0.783.

```
print('Raw data:', sub_df.shape)
sub_df = sub_df[~sub_df['dataset_id'].str.startswith('GCA_')].reset_index(drop=True)
sub_df = sub_df[~sub_df['dataset_id'].str.contains('/')].reset_index(drop=True)
sub_df = sub_df[~sub_df['dataset_id'].str.startswith('GO:')].reset_index(drop=True)
sub_df = sub_df[~sub_df['dataset_id'].str.startswith('HGNC:')].reset_index(drop=True)
sub_df = sub_df[~sub_df['dataset_id'].str.startswith('RRID:')].reset_index(drop=True)
sub_df = sub_df[~sub_df['dataset_id'].str.contains(':')].reset_index(drop=True)
sub_df = sub_df[~sub_df['dataset_id'].str.match(r'^\\d+$')].reset_index(drop=True)
sub_df = sub_df[~sub_df['dataset_id'].str.startswith('NCT')].reset_index(drop=True)

print('Filtered data:', sub_df.shape)

```

(2) If not in the external data, follow the normal process.

## 2. Type Classification

### 2.1 DOI

(1) Obtain the authors of all article IDs and data DOIs, and use author information for judgment.

(2) For data where author information is not obtained:

1. Use V4 to construct data for training large models, where V4 is labeled using Gemini, thereby distilling a 7B model. Use context to determine type categories.

### 2.2 ACC

1. Use V4 and the training set to construct data for training large models, where V4 is labeled using Gemini, thereby distilling 7B and 14B models.
2. Combine the three models.

Combination strategy 1: Fix SAMN and EPI as P and S respectively, with the rest determined by majority rule.

Strategy 2: For SAMN, if any model predicts P, set as P; for EPI, if any model predicts S, set as S; the rest determined by majority rule.

LB strategy 1 performs better, while PB strategy 2 shows greater improvement.

1. (Not selected, lb=850, pb=771)

Obtain author information for EPI_, and use author information to determine type.
