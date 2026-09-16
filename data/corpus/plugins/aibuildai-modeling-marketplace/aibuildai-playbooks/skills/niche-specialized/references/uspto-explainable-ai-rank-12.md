# 12th place - Disjunctions of bigrams, trigrams only

Competition: uspto-explainable-ai
Rank: #12
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522301

My initial solution was unigram combinations of all fields, but after discovering the ‘magic character’* in phrase search and getting good initial results, I pivoted to using bigrams and trigrams only. Ultimately it wasn’t a good decision for the scoring and my only consolation is that the majority of bigrams/trigrams query terms in my final solution are quite meaningful. It’s also quite cool when target patents can be matched by a single bigram/trigram.

I’ll go into detail about 3 main parts of my final solution:

[Preprocessing](#preprocessing)
[Weighted set cover algorithm](#weighted-set-cover-algorithm)
[Parameter optimization](#parameter-optimization)

*Side note:
The ‘magic character’ that I’m referring to is the usage of special characters to join words in phrases, which is the only trick I used in my solution. The character I used to join phrases was ‘-’. The token savings of this trick in my solution was actually not that significant in practice, due to the following reasons:
- Because I want to use meaningful bigrams and trigrams in all fields, it is more important to find ngrams that can target span patent groups. In a significant proportion of patent groups I think there actually aren’t such bigrams/trigrams, so it imposes an upper bound on the possible cover with 50 tokens. In some patent groups however it does work very nicely and you can actually find a bigram/trigram that can cover all 50 neighbors without false positives (in such cases the trick also doesn’t matter since we only need a few tokens)
- Phrase search in description has VERY expensive query time and a major issue I faced in the last 2 weeks after I created my final phrase dataset. I came to realise that some bigrams/trigrams can even take more than 20s on their own and I had to implement multiple changes to get my notebook to score reliably. These findings are documented in a separate post [here](https://www.kaggle.com/competitions/uspto-explainable-ai/discussion/522303)
- Phrase search in Whoosh had some inconsistencies such as the one described [here](https://www.kaggle.com/competitions/uspto-explainable-ai/discussion/519755#2919472), that is complex to account for in the preprocessing pipeline

In summary, phrase search in title, abstract and claims is quite stable, but in description field … *here be dragons*. At the same time description field accounts for majority of the query results for this approach and having only title, claims and abstract would score very poorly.

## Preprocessing
The goal of the preprocessing pipeline is to generate phrase ranks for all patents such that for each target patent, we can fetch ranked phrases for each neighbour and assemble them together quickly. A finite state automaton built using Aho-Corasick is the key part of this pipeline.

###### SpaCy NLP + CountVectorizer
To extract noun chunks I used SpaCy’s nlp pipeline inside a batched CountVectorizer with optimizations like
- disabling lemmatizer and ner in nlp pipeline
- analyze only first 1 million characters in description text (SpaCy’s default maximum length for nlp pipeline is 1 million)
- binary True and max_df of 0.1 for CountVectorizer
- truncating or dropping noun chunks with special characters
- dropping noun chunks with count > 150 after combining batches
- clean the combined vocabulary, check for length, Shannon entropy, etc.  

Final output is a vocabulary of phrases (the count obtained here is an estimate, the true count will be obtained later through the fsa)

###### Build fsa from vocabulary
A finite state automaton using Aho-Corasick (trie with transition links) can get matching substrings in linear time. Whoosh uses some fsa in their indexing as well but I did not investigate using their implementations (Elasticsearch uses a DAFSA under the hood I believe). I used the pyahocorasick library (https://pyahocorasick.readthedocs.io/en/latest/) to build this fsa.
- Note that SpaCy noun chunks have root and extended forms, and do not consistently return the same chunks, but this is inherently resolved by the fsa and we can just extract the matching substrings
- To allow for exact phrase match (instead of substring within word match), pad each phrase with space at the start and end

###### Extract final phrases from parquet data and format into dataset
- Run each description through the automaton and extract all matching phrases with their corresponding counts
- Counts are converted to ranks by scoring among all documents
  - Documents with same phrase counts get same rank but *lowered* e.g. for counts of 10, 5, 5, the ranks are 1, 3, 3 (not 1, 2, 2)
- The phrase index is formatted as an integer list instead of a dictionary. Each stride of 3 represents a (phrase_id, rank, document_frequency)
- Polars hits array limit when saving the final output, I sort the phrase ranks to prioritize larger document frequencies and truncate the list length at 500 phrases per document

## Weighted set cover algorithm
I approached the core problem as a variant of weighted set cover. 2 parts to the problem: 
- calculating the weight of each phrase query (weight of a set is the sum of phrase query weights for each neighbour that contains the phrase)
- algorithm for picking sets

Initially I spent little time on the weight formulation, using a harmonic series based ranking function, and spent the majority of time experimenting on approaches to set picking. Later on I kept to a simple greedy approach and focused on designing a simple, intuitive, and tunable weighting formula.

###### Phrase query weighting
I expressed the query weight as the interpolation of 2 weighting functions - rank based weighting and uniform.
*Rank based weighting*:

Documents with higher rank will have higher weight and the weight of all documents sums to 1.
*Uniform weighting*:

Intuitively this represents the scenario where for any search term, non-relevant documents are never in the index, so rank doesn’t matter since as long as the document has the term, it will always be relevant. As the index size gets smaller compared to the actual global size, this approximation gets better since it becomes increasingly unlikely that any non-relevant document is returned regardless of rank.
*Interpolated function*:

The alpha parameter represents the contribution of ranked and uniform components to the overall weight. Intuitively this represents the probability of false positives given an index size and global size. If the index and global are the same, then rank weights are directly equivalent to search ranks and the weight of a query term is directly equivalent to the ranks of the neighbour documents with that term. If the index is infinitesimally smaller than global, then rank doesn’t matter and the weight of a query term is only related to the number of neighbour documents with that term.

###### Set cover algorithm
Using weights from above, the simple greedy approach is to pick the term that provides the largest increase in total weight in every iteration. I introduce a new parameter, threshold, that represents the ‘saturation’ of weight for an item in the set. For example, if query term a contributes weight 1 to patent x, and threshold is 1, then any query term that also contributes to patent x doesn’t increase the total weight for x, since term a already returns x exactly.

The algorithm loop is:
- Try each query term from all 50 neighbours and calculate the increase in weight for all neighbours, subject to threshold
- Pick the term with highest weight contribution, and remove it from candidates
- Repeat until maximum weight is reached (all neighbour weights reach threshold) or tokens exceed 50

For faster computation I represent set cover as a bitset and only save the highest weight per unique bitset during weight calculation

## Parameter optimization
In addition to alpha and saturation threshold, I have another threshold for filtering out low weightage terms in the initial calculation. These 3 parameters can be easily tuned, and what I tried with the limited time was Bayesian optimization, treating the full scoring against 1 validation index as 1 black box evaluation. In practice the effectiveness was severely limited due to bad validation indexes, query time limit making evaluations noisy, but I do believe that simple tuning is effective if these are resolved. The initial parameter bounds are:
- alpha (0, 0.8)
- weight threshold (0, 0.04)
- combination threshold (saturation) (0, 0.7)

Here is a 3d visualization (credits: https://github.com/Yamanaka-Lab-TUAT/BOXVIA) of one BO set of trials against a validation index, with EI as acquisition function:

Note that for this particular run it looks like fully uniform weighting for a range of combination thresholds is not far from the global maxima, and the weight threshold doesn't seem to matter much. 

## Closing notes
Tldr my solution consists of a heavy preprocessing pipeline to extract meaningful noun phrases, a greedy set cover algorithm with weighting that tries to approximate the sampling of the validation/test index from the global dataset, and some simple tuning. All the data and code are linked below and I look forward to any comments, feedback, suggestions, especially any mistakes or how I can improve. Thanks!


Dataset: https://www.kaggle.com/datasets/dilliontan/uspto-preprocessed/
Submission notebook: https://www.kaggle.com/code/dilliontan/uspto-optimized-submission

###### Query examples

US-5516579-A
23 tokens covers all neighbours
>detd:"polymeric-phenolic-esters" OR detd:"predominant-dicarboxylic-acid" OR detd:"tmac-tma-bpda" OR ti:"solid-phase-polycondensation" OR ab:"rapid-molding-time" OR clm:"particulate-flame-retardants" OR ab:"especially-outstanding-toughness" OR ab:"ether-glycol-phthalate" OR ab:"new-economic-prepolymers" OR ab:"either-aliphatic-diacid" OR clm:"crystalline-polyester-imide" OR ab:"effective-reinforcing-amount"

US-8637833-B2
1 token covers all neighbours
>detd:"proton-delivery-efficiency"

US-11663219-B1
1 token covers all neighbours
>detd:"editable-buckets"

US-8047197-B1
49 tokens covers 29 neighbours
>ab:"tile-roofs" OR clm:"pointed-assembly" OR ab:"tile-roof" OR ab:"coupling-support-structure" OR detd:"screw-attached-mount" OR clm:"said-primary-barriers" OR ab:"phone-stand-assembly" OR detd:"vertically-orientated-bracket" OR ti:"food-cooling-assembly" OR ti:"roof-bracket-apparatus" OR clm:"fixedly-arranged-devices" OR ti:"rain-monitoring-system" OR ab:"tile-support-panels" OR ab:"consecutive-inclined-surfaces" OR ti:"laptop-support-assembly" OR clm:"bent-section-directing" OR clm:"convexly-arcuate-prominence" OR ti:"rod-holding-system" OR clm:"horizontally-orientated-members" OR clm:"telecommunications-rack-structure" OR clm:"middle-telescoping-member" OR ti:"tilt-table-system" OR detd:"first-elongate-spaces" OR ti:"door-painting-assembly" OR ab:"relatively-perpendicular-angle"

US-RE30747-E
49 tokens covers 42 neighbours
>detd:"muszik-et" OR detd:"structuring-substance" OR detd:"hectograph-blankets" OR detd:"fill-adhesive-composition" OR detd:"pregummed-wall-paper" OR detd:"outstanding-writability" OR detd:"decalcomania-coating" OR detd:"thermo-adhesive-tapes" OR detd:"di-dimethylcyclohexyl-adipate" OR detd:"single-plasticizers" OR ti:"polyvinyl-alcohol-deposits" OR ti:"decalcomania-adhesive" OR ti:"marine-dye-composition" OR clm:"water-dispersive-copolymer" OR ti:"pentavalent-vanadium-compound" OR detd:"example-starch-hydrate" OR ab:"medium-moisture-proof" OR ti:"coated-cork-composition" OR detd:"least-limitedsolubility" OR detd:"serial-tolerance" OR ab:"excellent-frangibility" OR detd:"difilcultto-remove" OR detd:"filly-name" OR ab:"stick-application" OR clm:"octyl-terephthalic-acid"

US-10013212-B2
49 tokens covers 38 neighbours
>detd:"intrusion-detection-database" OR detd:"avalon-memory-mapped" OR clm:"programmable-hardware-units" OR ab:"block-rams" OR detd:"reconfigurable-gate-arrays" OR detd:"stabilizing-logic-devices" OR clm:"programmable-logic-processor" OR detd:"partially-reconfigurable-regions" OR ti:"heterogeneous-memory-primitives" OR ab:"guardband-voltages" OR ti:"gpu-virtualisation" OR ab:"least-arbitration-rule" OR ab:"configuration-programming-block" OR clm:"variable-size-ratio" OR ti:"immediate-value-network" OR ab:"iii-cache-data" OR ti:"fast-dynamic-change" OR ti:"matrix-storage-method" OR clm:"asynchronous-handshaking-scheme" OR ti:"hierarchical-sort-acceleration" OR clm:"execution-complete-flag" OR clm:"coherent-interconnection-protocol" OR clm:"intermediate-module-channel" OR ab:"correct-storage-address" OR detd:"problematic-memory-blocks"
