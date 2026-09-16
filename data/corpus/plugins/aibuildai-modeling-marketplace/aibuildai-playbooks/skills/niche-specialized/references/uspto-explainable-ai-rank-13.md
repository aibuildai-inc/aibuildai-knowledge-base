# 13th place solution

Competition: uspto-explainable-ai
Rank: #13
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522359

Thanks to the organizers for the fun challenge!

### Summary
- Started in Python and got a public score of 0.53 using frequent pattern mining.
- Switched to C++, built my own searcher, created an ensemble of query generators, picking the best query for each test data row to get a public score of 0.76.

### Original Python solution
I started out in Python, and used FP-Growth algorithm to find combinations of frequent CPC and title terms. I ranked these combinations by the number of targets they would match, and inversely by the estimated number of other non-target patents they would match. This resulted in a public score of 0.53.

### Switching to C++
In experiments, I noticed that looking up the full data for random patents was extremely time-consuming, especially when that data would include descriptions. This seemed to be caused by the Parquet patent data files containing only a single row groups, making it required to read and decompress entire files even when only needing the data for a single patent.

Around the same time I started thinking about the possibility of creating a search index and running queries against it in my submission notebook. This would make it possible to select the best query out of an array of generated queries for each row in the test dataset. However, generating a search index with Whoosh takes quite a lot of time.

Based on those findings I came up with the following plan:
1. Extract the compressed patent data from the Parquet files, tokenize their contents, and store the tokens to disk in a format optimized for my solution. This allow for fast random patent lookup, and fast lookup of a subset of a patent's fields (e.g. only retrieve the CPC and title tokens without parsing the tokens of other fields).
2. At submission time, create an index containing patents that are most likely present in the test index. Assume all targets are present, and then gradually add popular neighbors of patents already in the index until the index contains 200k patents.  
   The index should store a bitset for each possible term denoting the patents that match it. Only consider `<category>:<token>` terms such as "ti:method" or "detd:algorithm" to limit the size of the index. Using such an index we can efficiently execute all queries using such terms with OR, AND, NOT, and XOR operators.  
   This format does not support proximity operators, wildcards, multi-word terms, and uncategorized terms. However, my earlier public score 0.53 submission only used single-word title terms and CPC terms with OR, AND, and XOR operators, so I was confident that the supported subset was enough to get a competitive leaderboard score.
3. Create a bunch of query generators and use them to generate possible queries for each row in the test dataset, with per-query hyperparameter optimization where necessary. Run each generated query against the index created in step 2, and submit the one with the best result.

Along with this plan I decided to switch to C++ for optimal performance. Kaggle notebooks run in an environment with GCC and CMake installed, so we can compile and run C++ code at submission time. Dependencies are pre-installed in a separate notebook and added through a private dataset.

This plan mostly worked out, with some exceptions in step 2. Rather than 200k patents, I eventually switched to an index containing all 13M patents. I dropped support for description terms in this change to reduce the size of the index. I also added counts to the index, indicating how many times each patent matches each term. This resulted in a performance improvement in sorting patents that match the query to pick the top 50.

### Reformatted patent data
I converted the original patent data files to my own format to allow for faster lookups. For each patent, my reformatted patent data contains unordered "token -> #occurrences" mappings for the CPC codes, titles, abstracts, claims, and descriptions of all patents. This data is stored uncompressed in a custom binary format. I stored the mappings in a data file, and created an index file containing a separate mapping from publication number to the offset of its corresponding data in the data file. Together this format allows for fast lookup of random patent data.

### Custom searcher
While building my own searcher I noticed several quirks:
- The XOR parser can be very slow. I've seen queries with just 10 XOR operators get stuck in the query parser in Python. Execution performance is not bad though, just need to make sure not to have too many XOR operators or submissions will time out.
- `query_parser.parse("A OR B OR C") == query_parser.parse("(A OR B) OR C")` but `query_parser.parse("A XOR B XOR C") != query_parser.parse("(A XOR B) XOR C")`. `A XOR B XOR C` gets parsed to `((cpc:A OR cpc:B OR cpc:C) AND NOT (cpc:A AND cpc:B AND cpc:C))`, while `(A XOR B) XOR C` gets parsed to `((((cpc:A OR cpc:B) AND NOT (cpc:A AND cpc:B)) OR cpc:C) AND NOT ((cpc:A OR cpc:B) AND NOT (cpc:A AND cpc:B) AND cpc:C))`.
- If two patents match the query equally well, their order is determined by their Whoosh document id, which seems to be a number that increases every time a document is added to a Whoosh index. I couldn't figure out how to replicate this order in my own searcher.

In the end my searcher was able to execute queries on an index containing 200k patents in less than 0.2ms on average with an in-memory index, and in less than 0.3ms on average with an index stored on disk (in a single thread on my own laptop).

### Query generators
My final ensemble consisted out of four query generators:
1. Single-term generator: this generator was used to test my implementation by generating queries containing a single term by picking the term with the maximum "#targets matching the term / term selectivity" score. Its queries didn't perform well, but as a testbed it was great.
2. FP-Growth generator: this generator is essentially a clone of my original Python-based submission, but now running on abstract terms as well (for claims terms it is too slow).
3. Optimizing generator: this generator supplies the most of my submitted queries. It attempts to find an optimal combination of target groups by starting with all targets in one group and then greedily swapping targets from one group to another (or to a new group). Target groups were converted to a query by creating a group of shared terms ranked by selectivity for each target group, and then equally dividing the number of available tokens across the groups. Terms inside each term group are then joined together by implicit AND operators, and groups of these AND-joined terms are joined together using OR and XOR operators (with a limit of 5 XOR operators in a query).
4. Best-effort generator: this generator iteratively builds the query by constantly adding terms for the highest-priority target that isn't matched by the query yet. For example, if the first 3 targets match a partial query, this generator retrieves the terms of target 4 and adds the one with the lowest selectivity to the query. This generator mostly served as a fallback for cases where the optimizing generator fails to find a good query.

### Local validation
I performed local validation on the first 2,500 rows of the neighbors data in [Devin Anzelmo's validation index](https://www.kaggle.com/datasets/devinanzelmo/uspto-explainable-ai-validation-index). I originally performed this local validation with an in-memory search index containing the 125k targets and 75k related patents, but switched to using a search index containing all 13M patents later as it gave more realistic results.

### Code and data
- Full source code: [github.com/jmerle/uspto-explainable-ai](https://github.com/jmerle/uspto-explainable-ai)
- Offline notebook (competition submission): [kaggle.com/code/jmerle/uspto-explainable-ai-ensemble](https://www.kaggle.com/code/jmerle/uspto-explainable-ai-ensemble)
- Online notebook (dependencies): [kaggle.com/code/jmerle/uspto-explainable-ai-ensemble-dependencies](https://www.kaggle.com/code/jmerle/uspto-explainable-ai-ensemble-dependencies)
- Reformatted competition data dataset: [kaggle.com/datasets/jmerle/uspto-explainable-ai-reformatted-patent-data](https://www.kaggle.com/datasets/jmerle/uspto-explainable-ai-reformatted-patent-data)
- Full search index dataset: [kaggle.com/datasets/jmerle/uspto-explainable-ai-full-search-index](https://www.kaggle.com/datasets/jmerle/uspto-explainable-ai-full-search-index)
- Original submission notebook (before switching to C++): [kaggle.com/code/jmerle/uspto-explainable-ai](https://www.kaggle.com/code/jmerle/uspto-explainable-ai)
