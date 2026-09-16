# The "Magic" !

Competition: uspto-explainable-ai
Rank: #7
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522199

**TLDR:** You can create queries as long as you want by using a "fake" space character. Whoosh's preprocessing will replace this character by a whitespace, whereas the `count_query_tokens` function will not !

Sometimes a code example is better than a long explanation:
>https://www.kaggle.com/code/theoviel/uspto-magic-with-cudf-pandas-lb-0-8-in-1min

#### Overall idea

Let's say we have two publications ( `publication_1` and `publication_2`) with titles `title_1 =  t1_1 t1_2 t1_3 ...`and `title_2 =  t2_1 t2_2 t2_3 ...`.
The easiest way to query both publications would be to do : 
`query = ti:"title_1" OR ti:"title_2"`
The issue is that titles can be several words long, and with the token limit of 50, you cannot really query enough titles to get a reasonable score.
If only title_1 could count as only one token, that would mean we could query 25 titles, and get a score of 0.8+ effortlessly ! 

#### The magic

The count token function uses spaces:

```
def count_query_tokens(query: str):
    return len([i for i in re.split('[\s+()]', query) if i])
```

However, whoosh uses a more sophisticated preprocessing. If we can find a character than is replaced by a whitespace by whoosh, but not by the `count_query_tokens` regex, then we can query a title for the cost of one token  !

And it turns out, such special character exists: `~` is an example. This query :
```
query = ti:"t1_1~t1_2~t1_3~..." OR ti:"t2_1~t2_2~t2_3~..."
```
Costs 3 tokens and will match the same document as 
```
query = ti:"t1_1 t1_2 t1_3..." OR ti:"t2_1 t2_2 t2_3..."
```

Using this trick to query 25 exact titles alone achieves LB 0.8 which is almost in the gold zone.

Using [cudf-pandas](https://rapids.ai/cudf-pandas/) to process everything on GPU makes the code super-fast. The notebook shared above runs in 1 minute ! Unfortunately there is no efficiency track in this competition 😄

We used slightly fancier heuristics to reach 7th place, but I'm sure top teams found even better hacks !
