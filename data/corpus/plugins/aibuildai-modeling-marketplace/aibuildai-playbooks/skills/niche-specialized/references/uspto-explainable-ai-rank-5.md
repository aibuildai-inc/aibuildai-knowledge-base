# 5th Place Solution

Competition: uspto-explainable-ai
Rank: #5
Source: https://www.kaggle.com/c/uspto-explainable-ai/discussion/522201

First of all, thanks to the organizers for hosting this competition, and to my teammate @sega1031.

## Overview
* Create many query candidates with minimal false positives.
* Select up to 25 candidates using solver, then concatenate using OR.
* Token count: you can reduce the token count of an AND query to one token (see below).

## Validation Strategy
* We use the published validation index.  
  https://www.kaggle.com/datasets/devinanzelmo/uspto-explainable-ai-validation-index
* The leaderboard score is slightly lower than the validation score but is well correlated.

## The Metric and Basic Strategy
* We believe assumption from @devinanzelmo is correct: https://www.kaggle.com/competitions/uspto-explainable-ai/discussion/499981#2791642
* We confirm this through two submissions. The theoretical and actual leaderboard values are very close.
  1. Select one patent from the 50 neighbors and build a query from first 20 words of its abstract with phrase search. This procedure gives a query with 20 tokens, TP1, and FP0 (= one true positive and zero false positive).
  2. Select two patents from the 50 neighbors and build a query from first 20 words of each abstract for a phrase search. The two queries are concatenated with OR. This results in 41 tokens, TP2, and FP0.

| exp | TP | FP | Theoretical LB | Actual LB |
|-----|----|----|----------------|-----------|
|  1  |  1 |  0 |      0.089     |    0.08   |
|  2  |  2 |  0 |      0.159     |    0.15   |

* In this evaluation metric, it is crucial for TP to rank highly and FP NOT to rank highly.
  * If 10 TPs are followed by 40 FPs -> score: 0.514
  * If 40 FPs are followed by 10 TPs -> score: 0.023
* Therefore, our basic strategy is to collect as many TPs as possible while keeping FPs as close to zero as possible.
* This plot shows the score for N TPs followed by (50-N) FPs.
  * if 41 TPs are followed by 9 FPs -> score: 0.980
  * if 44 TPs are followed by 6 FPs -> score: 0.991



## Creating Candidates
### Global Counter Candidates
* Preparation
  * Count how many times a specific word or n-gram appears in all given patents. For example, "ti:device" may appear 20,000 times in 200,000 patents.
  * We call this "global counter".
  * This global counter is prepared in advance.
* Creating candidates
  * Check the global counter for all words and n-grams appearing in the 50 neighbors we want to retrieve.
  * If "ti:device" appears in 3 out of the 50 neighbors, and the global counter's count is 4, then "ti:device" query results with TP3 and FP1.
  * Create these candidates for ti, ab, clm, detd, and cpc, using unigrams, bigrams, and trigrams.

| candidates    | tp_cover     | tp                   | global_count | fp         |
|------------|--------------|----------------------|--------------|------------|
| ti:device  | {0,1,2}      | len({0,1,2}) = 3     | 4            | 4 - 3 = 1  |
| clm:invention    | {0,2,3,4,5}  | 5                    | 7            | 2          |
| detd:method detd:device   | {6}          | 1                    | 1            | 0          |


* tp_cover = {0,1,2} indicates that this query candidate can retrieve neighbors with indices 0, 1, and 2.
* Example:
  * Concatenate three candidates in this table with OR: `ti:device OR clm:invention OR (detd:method detd:device)`.
  * Since they are concatenated with OR, tp_cover becomes {0,1,2,3,4,5,6}.  
  * Therefore, this query will be 6 tokens, TP7, and FP3.

### 50c2 Candidates

Select two patents from the 50 neighbors.

| publication_number | ti         | abst            |
|--------|---------------|-----------------|
| US-0000-A    | dog cat fox   | pig cow         |
| US-0001-A    | cat fox       | duck frog cow   |

* Extract the common words from these two patents and create an AND query.
  * `ti:cat AND ti:fox AND ab:cow`
* This query will retrieve these two patents, but we need to calculate/estimate how many FPs this query will retrieve.
* We use IDF (inverse document frequency).
  * Since IDF is the logarithm of the inverse of the occurrence probability, summed-up IDF values give the occurrence probability of ANDed candidates.
  * Calculate the IDF sum of the created query, and if it is above a certain threshold, estimate that FP is zero.
  * We set this threshold to 80 using validation data. In this example, we are checking if idf(ti:cat) + idf(ti:fox) + idf(ab:cow) > 80.
* Calculate this for all combinations of 50 choose 2 (= 1225).
  * Each candidate must be 2 TPs and 0 FP.

| candidates    | tp_cover     | tp                   |  IDF_sum | fp         |
|------------|--------------|----------------------|------------|------------|
| ti:cat ti:fox ab:cow  | {0,1}      |  2            | 103.24 |0  |


## Token Count
```python
def count_query_tokens(query: str):
    return len([i for i in re.split('[\s+()]', query) if i])	
```

* The number of tokens is counted by this function,
* Through a particular approach to constructing the query, it is possible to keep the parse result the same while reducing the token count to one.

### Example
Normal AND
```python
query = "ti:dog ti:cat"
num_tokens = whoosh_utils.count_query_tokens(query)
print("num_tokens:", num_tokens)
qp = whoosh_utils.get_query_parser()
qp.parse(query)
```
> num_tokens: 2  
> And([Term('ti', 'dog'), Term('ti', 'cat')])

Special AND
```python
query = 'ti:"dog"ti:"cat"'
```

> num_tokens: 1  
> And([Term('ti', 'dog'), Term('ti', 'cat')])

Normal phrase
```python
query = 'ti:"The quick brown fox jumps over the lazy dog"'
```

> num_tokens: 9  
> Phrase('ti', ['quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog'], slop=1, boost=1.000000)

Special phrase
```python
query = 'ti:"The@quick@brown@fox@jumps@over@the@lazy@dog"'
```

> num_tokens: 1  
> Phrase('ti', ['quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog'], slop=1, boost=1.000000)

### Note
* We would like to apply this technique to OR as well, but we couldn't find a way.
* Even without this technique, we were able to achieve a LB score of 0.85.

## Set Cover Problem
* Up to this point, we have many query candidates created by global counter and 50c2.
* For example, the following candidates may have been obtained.
* Note: With the above token count technique, each token count is always 1.

| candidates                           | tp_covers                 | FP | 
|---------------------------------|---------------------------|----|
| ti:device                       | {0, 1, 2, 3, 4, 5, 6}     | 3 |
| ti:"dog"ti:"cat"                   | {0, 1, 2, 7}                 |  1 |
| clm:"dissociation"ab:"proteins" | {3, 4, 10}                |  0 | 
| ...                             | ...                       | ...| 


* Selecting some of these candidates and concatenate them with OR optimally is difficult, because this is a [set cover problem](https://en.wikipedia.org/wiki/Set_cover_problem), which is known to be NP-hard.
* We solved it as a integer linear programming problem using a python solver.
* Maximize TP and minimize FP with a token count of 50 or less. In practice, we maximized `10xTP − FP`.
* It can also be solved using some kind of greedy algorithm, but the solver provided a slightly better score (+0.01).

## Query Example

<details><summary>Here is a query example with 49 token.</summary>

> (cpc:"B60P1/165"cpc:"E02F3/3483"cpc:"B60P1/165"cpc:"E02F3/3486") OR (cpc:"B60D1/26"cpc:"B62D49/04"cpc:"B60D1/26"cpc:"E02F3/6472"cpc:"B60D1/26"cpc:"E02F3/655"cpc:"B62D49/04"cpc:"E02F3/64"cpc:"B62D49/04"cpc:"E02F3/6472"cpc:"B62D49/04"cpc:"E02F3/655"cpc:"B62D49/04"cpc:"E02F9/2016"cpc:"E02F3/6472"cpc:"E02F9/2016"cpc:"E02F3/655"cpc:"E02F9/2016"detd:"courli"detd:"lgayea") OR (cpc:"B65F2003/0283"cpc:"E02F3/3486"cpc:"B65F3/046"cpc:"E02F3/3486"ti:"end@loader@actuating"ti:"loader@actuating"ti:"loader@actuating@mechanism"ti:"actuating@mechanism@dump"cpc:"B65F2003/0283"cpc:"B65F3/046"detd:"horbe") OR (detd:"wharfpthe"ti:"dumping@scoop"ti:"side@dumping@scoop"detd:"hatchof") OR (detd:"powerswung") OR (detd:"shlppee") OR (detd:"0rneypearce"detd:"schaepcrklaus"ti:"as@asphalt@or"ti:"improvement@therein@laying"ti:"laying@surfacing"ti:"laying@surfacing@material"ti:"machine@and@improvement"ti:"surfacing@material@as"ti:"therein@laying"ti:"therein@laying@surfacing"detd:"comprlsesr") OR (detd:"disswingable"detd:"rdlyonto"detd:"sitionedsymmetrically"detd:"understandingflof"detd:"opjusting") OR (ti:"mechanical@shoveling@machine"ti:"mechanical@shoveling") OR ((detd:"tractors"detd:"ravity"detd:"scrapers"detd:"kick"cpc:"E02F3/6472"ti:"scraper"cpc:"E02F3/656"detd:"apron"detd:"scraper"detd:"hingedly"detd:"tractor"detd:"sheave"detd:"sheaves"detd:"cooperates"detd:"dead"detd:"axle")) OR ((detd:"oor"detd:"exible"detd:"retracting"detd:"trough"detd:"retracted"detd:"turntable"detd:"underground"detd:"jacks"detd:"therealong"detd:"rectilinear"detd:"propelling"detd:"conveyor"detd:"extensible"detd:"slip"detd:"clutch"detd:"mines"detd:"adjustably"detd:"engageable"detd:"elevating"detd:"hydraulic")) OR ((detd:"shovelling"detd:"tom"detd:"cushioned"detd:"compel"detd:"swiveled"detd:"abruptly"detd:"swivelly"detd:"wardly"detd:"hose"detd:"guideway"detd:"swivelled"detd:"undesired"detd:"sup"detd:"ported"detd:"swivel"detd:"osgood"detd:"pile"detd:"muck"detd:"guideways"detd:"dig")) OR ((detd:"planetaries"detd:"payed"detd:"planetary"detd:"mosier"detd:"fiexible"detd:"2li"detd:"conveyer"detd:"tractive"detd:"lil"detd:"ior"detd:"exible"detd:"2s"detd:"yieldable"detd:"tunnels"detd:"compensating"detd:"excepting"detd:"lll"detd:"chute"detd:"illinois"detd:"geared")) OR ((detd:"fioor"detd:"simmons"detd:"overlie"detd:"hydraulically"detd:"attachable"detd:"swivelly"detd:"coal"detd:"jacks"detd:"swivel"detd:"anchor"detd:"joy"detd:"guideways"detd:"unison"detd:"propelling"detd:"pennsylvania"detd:"mining"detd:"extensible"detd:"transporting"detd:"tilted"detd:"elevating")) OR ((detd:"vfiled"detd:"communicable"detd:"eiect"detd:"hydraulically"detd:"sullivan"ti:"material"detd:"propulsion"detd:"machinery"detd:"claremont"detd:"bores"detd:"uid"detd:"pile"detd:"muck"detd:"lin"detd:"dig"detd:"trackway"detd:"conduits"detd:"5i"detd:"massachusetts"detd:"hydraulic")) OR ((detd:"i23"detd:"movementof"detd:"i26"detd:"insides"detd:"encircles"detd:"compactness"detd:"h2"detd:"cushioning"detd:"yieldably"detd:"reversely"detd:"pivoting"detd:"cams"detd:"mucking"detd:"abuts"detd:"lug"detd:"therealong"detd:"teeth"detd:"axles"detd:"scoop"detd:"nuts")) OR ((detd:"foolproof"detd:"selfcentering"detd:"impetus"detd:"i85"detd:"coaction"detd:"rockers"detd:"pinned"detd:"maxson"detd:"i00"detd:"pivotable"detd:"i05"detd:"fork"detd:"inadequate"detd:"dipper"detd:"compelling"detd:"plungers"detd:"incapable"detd:"h5"detd:"abrupt"detd:"tang")) OR ((detd:"evidently"detd:"shank"detd:"receivable"detd:"hoses"detd:"venting"detd:"urges"detd:"interrupting"detd:"vented"detd:"hose"detd:"rolls"detd:"inactive"detd:"claremont"detd:"interrupted"detd:"3i"detd:"joy"detd:"guideways"detd:"trackway"detd:"pennsylvania"detd:"embodies"detd:"assumes")) OR ((detd:"seam"detd:"bevel"detd:"brake"detd:"rocked"detd:"wheeled"detd:"propulsion"detd:"spur"detd:"rst"detd:"coal"detd:"pinion"detd:"withdrawal"detd:"gearing"detd:"shafts"detd:"extensible"detd:"clutch"detd:"keyed"detd:"elevating"detd:"hydraulic")) OR ((detd:"draulic"detd:"hy"detd:"4s"detd:"vfor"detd:"oor"detd:"zontal"detd:"withdrawing"detd:"ie"detd:"andv"detd:"progresses"detd:"mw"detd:"lo"detd:"anchor"detd:"teeth"detd:"conveyor"detd:"extremity"detd:"3l"detd:"hydraulic")) OR ((detd:"isv"detd:"isprovided"detd:"sion"detd:"ropes"detd:"rope"detd:"suddenly"detd:"vof"detd:"relied"detd:"thel"detd:"urge"detd:"anda"detd:"4l"detd:"0f"detd:"gearing"detd:"anchored"detd:"transportation"detd:"mining"detd:"lthe"detd:"transporting")) OR ((detd:"posltion"detd:"motive"detd:"unwind"detd:"drifts"detd:"rearmost"detd:"tunnels"detd:"cated"detd:"segmental"detd:"injury"detd:"imparting"detd:"pile"detd:"ward"detd:"ap"detd:"scoop"detd:"swings"detd:"mines"detd:"reversible")) OR ((detd:"adaptedv"detd:"grooved"detd:"ore"detd:"chine"detd:"vhen"detd:"t0"detd:"opera"detd:"vand"detd:"sidewise"detd:"wheeled"detd:"thev"detd:"meshes"detd:"andv"cpc:"E02F9/022"detd:"movably"detd:"idler"detd:"coal"detd:"pinion"detd:"casting"detd:"bears")) OR ((detd:"compel"detd:"abruptly"detd:"tensioned"detd:"swivelled"detd:"coincident"detd:"assured"detd:"turntable"detd:"claremont"detd:"swivel"detd:"osgood"detd:"fulcrum"detd:"loader"detd:"guideways"cpc:"E02F3/3486"detd:"trackway"detd:"propelling"detd:"rolling"detd:"assumes"detd:"alinement"detd:"swings")) OR ((detd:"hoists"cpc:"E02F3/657"detd:"hoist"cpc:"E02F3/656"detd:"medial"detd:"bumper"detd:"trunnions"detd:"trunnion"detd:"grading"detd:"brake"detd:"steering"detd:"spreading"detd:"tractor"detd:"propulsion"detd:"coacting"detd:"extremities"detd:"gearing"detd:"rail"detd:"dump"detd:"clutch"))


---------------------------
EDIT: 2024/08/06

We pulished our codes.

* with magic
  * https://www.kaggle.com/code/iiyamaiiyama/uspto-5th-place-submission-with-magic?scriptVersionId=190687232
* without magic
  * https://www.kaggle.com/code/iiyamaiiyama/uspto-5th-place-submission-without-magic?scriptVersionId=190688084
* global counter(title)
  * https://www.kaggle.com/code/sega1031/uspto-global-title-word-counter
* global counter as one 
  * https://www.kaggle.com/code/sega1031/uspto-global-counters-limit30
