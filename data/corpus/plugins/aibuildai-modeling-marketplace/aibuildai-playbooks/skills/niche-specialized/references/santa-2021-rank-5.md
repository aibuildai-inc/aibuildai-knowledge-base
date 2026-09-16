# Santa 5th Place Solution (2428) & Insights

Competition: santa-2021
Rank: #5
Source: https://www.kaggle.com/c/santa-2021/discussion/300572

First of congratulations to everyone who achieved a decent score in this competition.  I am happy to get a solo gold.  I solved most of it on paper and then coded it. Here is my solution in detail:

## Name Conventions:
There are 3 types of sequences:
1. Mandatory (M) - sequences starting with '12' - (120)
2. Mandatory Relatives (R) - sequences containing '12' but not starting with it - (120*5)
3. Non Mandatory Non Relatives (N) - All other sequences - (4320)

## Solution:
### 1. Proving lower bound = 2400 (without wildcard, Mathematically) 

Due to additional constraint introduced in the problem, it was easy. The best length/sequence ratio we can have in a cyclic permutation with more than 8 sequences is 7/6 (6 one edges followed by a 2 edge - 2 cycles). There will be 120*3 M in the solution. This enforces additional constraints to have - 120 * 5 * 3 R. So, our permutation must have these 120 * 5 * 2 additional duplicates. 

##### Lets calculate lower bound 
* N - 4320 * 7/6 = 5040
* R - 3 * 120 * 5 = 1800
* M - 120 * 3 = 360
* Total length = 7200
* If we can split this cycle in 3 perms of equal length at 7-edges (if any), we get 2400. But we have relaxed some constraints for best case. May be adding those constraints can get us to 2440 mathematically. 

### 2. Getting to 2440 

* This is similar to what other participants did. I had approximate count of N,R,M in mind and I didn't want to break N-N connections in 2 cycles for the start. This is may be where I get lucky, If I remove all R from a 2 cycle and keep only one M, I get a permutation of length 47 for each cycle (lets call it B, base permutations). Simple concatenating all these 120 cycles. I have permutation of length 120 * 47 = 5640 which contains all N sequences. 

##### Sample base permutation (B) of 47 length: 12345672134567231456723415672345167234561723456

* Now, for M & R -> we can combine pairs of M sequences with swapped chars at end positions '12345671234576'. We can easily make these connections, by appending corresponding sequence at the start of each 120 base perms which makes length of each base perms - 47+7 = 54. Total length = 120 * 54 = 6480.
##### Sample base permutation (B) of 54 length: 123457612345672134567231456723415672345167234561723456

* Only sequence left to be added in each 2 cycle - #12#### (7123456). Now, lets add one M of length 7 with only constraint that it ends with a '7'. Base perm length = 54 + 7 =61. Total length 120*61 = 7320 . 
##### Sample base permutation (B) of 61 length: 1236547123457612345672134567231456723415672345167234561723456

* Use 40 permutations of length 61 in each subpermutations to get 2440. I could prove it visually but after @cpmpml & other's confirmation, I was sure this is lower bound. 

### 3. Getting to 2428 

* Once you know exactly how your solution is created, getting to 2428 is easy. For getting to 2428, we can modify our permutation and add corresponding sequence at the end instead of start of 47 length base permuation. Pairs (swapped at last position) of such wildcard permutations (length 48) will cover all sequences contained in base permutations of length 54. We miss R, but we have extra Ms to make up for it. Remember all we need to stack 2 M to save all Rs (12345671234576). We can stack 3M for these special 6 wildcard cases (124356712345671234765)
##### Sample wildcard permutation (B) of 48 length:12345672134567231456723415672345167234561*234576
* Our solution size -> get reduced from 54 to 48 (save of 6 length). 6 wildcards - 3 pairs - 2428 solution. 

### 4. Why I think 2428 is optimal and we can save only 6 per wildcard in best solution

Assuming we can only use wildcard by breaking and making connections

* We can't save R. R is duplicated on average 3 times. If we use wildcard, to save R at one place, it will still show up at other place. I mean technically we can save R. But, when we save R, we actually remove a redundant edge. 
* Touching base permutation of length 47 (except at the start or end) - Base permutation is saturated. There are only edges of length 1 or 2. (6 ones followed by 2)
* Case 1 - Break and make 2 ones - No benefit - Best you will have 2 ones with new wildcard in place
* Case 2 - Break and make 1 one and 1 two - Best you will have 2 ones with new wildcard in place. After 2 you will be followed by 6 ones in 2 cycle - Again you'll be able to save only 1 distance
* Case 3 - Break and make 3 twos -  Best you will have 2 ones with new wildcard in place - Same logic - you'll be able to save only 1 distance
* At the start or end - here you're not followed by saturated connections - So, here we have best chance of getting benefits. We convert a 7 edge -> 1 edge with wildcard. 7 edges can only exists for M (because of redundant R). 
* More than 6 save is only possible if there are multiple consecutive edges>=2 which is not possible here. If we observe base permutation, the distance between 7 edge and 2 edge is 7 which makes it impossible to use same wildcard to effectively reduce both edges.
* This may not be proper mathematical proof, but, I see it hold true as long as we don't have a 7 and a >=2 edge close to each other.  I tried to bring them close, only to realise that I will be distorting my base permutation (Adding extra penalty) which may give me better wildcard benefits but increase my 2440 base sequence length.

Code here: https://www.kaggle.com/ks2019/getting-2440-in-simpler-way?scriptVersionId=80946172

P.S. If anyone has additional ideas for mathematically proving 2428 as lower bound, contact me. I will love to work on this together.
