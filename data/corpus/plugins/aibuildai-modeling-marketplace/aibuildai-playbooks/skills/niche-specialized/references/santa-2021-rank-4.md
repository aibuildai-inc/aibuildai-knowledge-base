# 4th place solution by hand

Competition: santa-2021
Rank: #4
Source: https://www.kaggle.com/c/santa-2021/discussion/300543

Thanks for the fun competition. The Santa Competition for new-year holidays is my pleasure and I am pleased that this tradition has continued.  

My solution is made by hand. I used a computer to simply enumerate.  


## 2440 without wildcard

In this problem, we need to construct 3 strings such that 
- Every permutation (5040 in total) of [1, 2, 3, 4, 5, 6, 7] is contained in at least 1 string.  
- Every permutation in the form of 12xxxxx (120 in total) is contained in all strings. 
(1: Mr. Santa, 2: Ms. Santa)

Second condition seems to break the symmetry, but in reality it seems that, as a result of this too big break, there is a room to make a solution by symmetry.  

The key idea for me is that 120 permutations of 12xxxxx are not waste.  
For example,  
`1273456 1237456 1234756 1234576 1234567` 
is not only 5 permutations but also it is a part of a 2-cycle  
`7 2345612 7 3456123 7 4561234 7 5612345 7 6123456 7 1234561 (7 2345612)`

By looking for a good string (part of a 2-cycle) that does not overlap with these, I found strings like these:
`1 2345672 1 3456723 1 4567234 1 5672345 1 6723456 1 7234567 (1 2345672)`
By removing the duplicates with 12xxxxx, we obtain 
`345672 1 3456723 1 4567234 1 5672345 1 6723456 1 723456`  (length 45)  
If we fix 1 and 2, there are 5! = 120 sequences like this, so a simple assignment will create 3 strings consists of 40 such sequences and 120 12xxxxx.  
The length is 45 * 40 + 7 * 120 = 2640, Clearly, we can combine 1234567 and 3456721..., and so on, so the length is now 2440.  
More precisely, as 12xxxxx can be part of an important sequence, we need to be careful when combining them. But that's not a critical problem because all 3 strings contains all 12xxxxx. 

This solution by @jpmiller seems similar.  
https://www.kaggle.com/c/santa-2021/discussion/300519


## 2440-> 2428 with wildcard
If we understand the structure of the 2440 solution, it's just a puzzle. 
When the leaderboard is updated, it's time to start a new race. Think without sleeping. I did this twice(2430 -> 2429 -> 2428).

### 2430
We start from 2440 solution, which consists of 40 sequences of the following form: 
`12 345672 1 3456723 1 4567234 1 5672345 1 6723456 1 723456`  (length 47)  
and 80 12xxxxx.   
We should produce 12xxxxx by almighty because it's inefficient, so where to put almighty(denote by 8) should be: 
`12 345672 1 3456723 1 4567234 1 5672345 1 6723456 1 823456 7` (length 48)  
We added 1 character and one 1234567 is newly included, so seemingly this reduces length by 6, 
but no, because we have 1234567 twice. So we add one more character like this: 
`127 345672 1 3456723 1 4567234 1 5672345 1 6723456 1 823456 7`  (length 49)  
thus we could reduce 5 characters by 1 almighty.  

### 2429
Take 2 similar sequence of length 48 with duplicate, such as 1234567 and 1234576,   
`12 345672 1 3456723 1 4567234 1 5672345 1 6723456 1 823456 7`  (length 48)  
`12 345762 1 3457623 1 4576234 1 5762345 1 7623457 1 823457 6`  (length 48)  
and swap the last two characters.
`12 345672 1 3456723 1 4567234 1 5672345 1 6723456 1 823457 6`  (length 48)  
`12 345762 1 3457623 1 4576234 1 5762345 1 7623457 1 823456 7`  (length 48)  

If we joint these two sequences and insert one character, we obtain 
`127 345672 1 3456723 1 4567234 1 5672345 1 6723456 1 823457 6`  (length 49)  
`2 1 3457623 1 4576234 1 5762345 1 7623457 1 823456 7`  (length 41)  
This reduces 11 characters by 2 almighty.  

### 2428
We can put these two sequences in different strings. 
`12 345672 1 3456723 1 4567234 1 5672345 1 6723456 1 823457 6`  (length 48)  
`12 345762 1 3457623 1 4576234 1 5762345 1 7623457 1 823456 7`  (length 48)
