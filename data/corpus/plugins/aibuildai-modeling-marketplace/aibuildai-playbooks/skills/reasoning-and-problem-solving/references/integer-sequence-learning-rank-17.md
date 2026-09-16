# Solutions!

Competition: integer-sequence-learning
Rank: #17
Source: https://www.kaggle.com/c/integer-sequence-learning/discussion/24971

Hi all,

This was an interesting competition, congrats to the winners who managed to resist the urge to [cheat](https://www.youtube.com/watch?v=SrDSqODtEFM)!  

This competition was unusual compared to the other Kaggle competitions I've entered, as it's really 113,845 independent small data problems rather than a single big data problem. This presents the standard ML difficulties - ie. fitting without over-fitting, but with the additional requirement to do it with limited data in a way that can scale back up to work reliably with hundreds of thousands of (varied) sequences.



# Our approach

Our approach was to sequentially test sequences with different solvers, ordered approximately by assumed reliability. This relied on the fundamental assumption that solver effectiveness depends only on the solver (and not the sequence), which I don't think is necessarily optimal.

[Whiz](https://www.kaggle.com/wildwizard) and I teamed to work on different methods. I "unintelligently" implemented the solvers below while Whiz worked on various implementations of [Balzac's prefix lookip](https://www.kaggle.com/balzac/integer-sequence-learning/prefixes-lookup-0-22), which can be incredibly computationally intensive. I'll let Whiz share and describe his code for this if he wants to, as he understands it a lot better than I do!

## **Solvers**
See [here](https://github.com/garethjns/Kaggle-IntegerSequenceLearning) for actual R code. I'm afraid it's not very pretty and some of the variables are rather confusingly named, but I'll include the function names in the solver descriptions, just in case anyone really wants to take a closer look. The numbers are the approximate number of contributions of the solver to the final submission.
 
- [Common differences](http://www.purplemath.com/modules/nextnumb.htm) (**2820**)
- Common differences with variable step size (**2750** reliable and **9246** "dodgy")

- Pattern search (**1785**)
- Pattern search on common difference levels (**435**)
- [Recurrence relation](https://en.wikipedia.org/wiki/Recurrence_relation) (**9849**)
- [Linear fitting using previous points]((https://www.kaggle.com/endintears/integer-sequence-learning/linear-models)) (**37143**)
- Non-linear fitting using previous points (**9000**)
- Borrowed fallbacks (**<4000**)
- [Mode-fallback](https://www.kaggle.com/wcukierski/integer-sequence-learning/mode-benchmark/run/255053/code) (**40196**)

### Common differences (diffTableSolve, diffTablePredict)
This method is pretty simple. Take the difference between each adjacent term in the sequence, if the differences are all the same, the next term can be predicted. It's a [special case](https://en.wikipedia.org/wiki/Recurrence_relation#Relationship_to_difference_equations_narrowly_defined) of the recurrence relation (and presumably produces redundant predictions when both are used).

For example:  
Sequence: **[2, 4, 6, 8, 10]**  
First differences: **[2, 2, 2, 2]**

Next term is **10+2 = 12**

If the differences aren't the same, continue down:  
Sequence: **[2, 4, 7, 11, 16]**  
1st diffs: **[2, 3, 4, 5]**  
2nd diffs: **[1, 1, 1]**  

Next term is **16+5+1 = 22**

False positives are a risk when too few values are available at level to be sure the level is constant.

### Common differences with variable spacing (diffTableSolve2, diffTablePredict2)
An extension of the method of common differences is to take differences over a step size > 1, instead of from only adjacent terms. 

For example, with a step of 2:  
Sequence: **[2, 6, 8, 12]**  
1st diffs with step size 2: **[6, 6]** (ie. 8-2 and 12-4)  
Next term is **((n+1)-step) + diff = 8+6 = 14**  

False positives are a greater risk with this approach as level sizes contract quicker the greater step size used. "Dodgy" solutions were ones that were possible solutions, but lacked enough data to be sure. If no solutions were proposed by pattern search or pattern search on difference levels, the dodgy solution was used.

### Pattern search (patternSearch)

 - Starting with the second half of a sequence, pattern search compares it to the first half. 
 - If the proportion of matched values is greater than some threshold, the match is used to extend the sequence. 
 - If there's no match, one terms are dropped sequentially from the middle of the sequence (not the end) and compared to the (growing) start of the sequence. 

For example
Sequence = **[1, 2, 1, 2, 1, 2]**  
First check: **[1, 2, 1]** vs **[2, 1, 2]** - no match  
Second check: **[ignored, ignored, 1, 2]** vs **[1, 2]** - match, sequence extended with **[1, 2]**

### Pattern search on common difference levels (diffTablePattern)
This does a pattern search on each common difference level. In theory it might be able to find patterns in different levels, even if the difference levels never converge to a constant value... Maybe.

### Recurrence relation (RRSolve)
Based on Nina Chen's [notebook](https://www.kaggle.com/ncchen/integer-sequence-learning/recurrence-relation/notebook) and Enrique Pérez Herrero's R code in the comments. Extended to try every order possible for the length of the sequence available. I think generally, especially for lower order sequences, this method is pretty reliable. However, my implementation began with the lowest order and increased order until a prediction scored above a certain threshold was reached (based on % difference to known and held-out next term), a better approach would be to calculate all possible orders then take the best.

### Linear fitting (fitModPP)
I initially tried fitting linear polynomials as a function of [term position](https://www.kaggle.com/garethjns/integer-sequence-learning/linearfitsr) (fitModDec), however replaced it with linear fitting based on [previous points](https://www.kaggle.com/endintears/integer-sequence-learning/linear-models) (fitModPP). My fitting and scoring was slightly different to most of the linear fit kernels posted. 

 - The fit was done on the sequence minus the last term. All orders were fit between 1 and 12.
 - The last term was predicted for each fit and the fit was scored by % difference between the predicted last term and the known last term. The closeness of the fit to the rest of the points was ignored.
- The winning fit was compared to threshold and if it passed, the fit was redone using all the data (returning the initially held-out, known last point)
 - The next (unknown) term was then predicted.  

I found adding non-linear terms to the fits generally hurt overall performance unless the threshold was extremely restrictive. It was too easy to over-fit to an extent where the false solutions prevented mode-fallback. Seeing as mode-fallback is a bad method anyway, it's a clear demonstration of the dangers of false positives. I didn't get around to adding trigonometric functions, but these perhaps could have captured some of the oscillating fits better than higher order polynomial fits.

### Borrowed-fallbacks
To gain the final ~1% on our submission Whiz and I took publicly posted solutions, debated their possible reliability compared to our implemented methods, and substituted in these values to replace our unreliable estimations. Again we assumed the the general reliability of each method, rather than considering sequence type or properties. Overall we replaced ~4000 mode-fallbacks, linear and non-linear fits with other available predictions (from [Predict OEIS with Markov Chains
](https://www.kaggle.com/enrique1500/integer-sequence-learning/predict-oeis-with-markov-chains), [Real Machine Learning
](https://www.kaggle.com/endintears/integer-sequence-learning/real-machine-learning), [Predict OEIS with Ngrams](https://www.kaggle.com/enrique1500/integer-sequence-learning/predict-oeis-with-ngrams-1)). I'm not entirely sure what proportion of the 4000 replacements were genuine (ie. not just "replacing" one prediction with the same prediction), or which method provided the greatest proportion of alternatives, but it did give us a boost overall.
 

### Mode-fallback
If no solution was proposed by a solver, the mode of the sequence was used, as per the [benchmark](https://www.kaggle.com/wcukierski/integer-sequence-learning/mode-benchmark/run/255053)

# A better approach?

Our approach relied on the assumption that solvers are equally reliable regardless of sequence properties. 

I think a better way would be consider the reliability of a method in context with the properties of a sequence (see this [notebook](https://www.kaggle.com/garethjns/integer-sequence-learning/classifying-tagging-sequences); for example, assuming mode-fallback is less reliable for a monotonic sequence than, say, a binary, oscillating sequence). 

More exhaustive testing of solvers would be a sensible idea too; for example the specificity of a solver could presumably be tested on the training set by iteratively withholding known data for testing. The false positive rates for each solver as a function of n points of sequence used could further inform solver selection. 

Perhaps by combining these two approaches it would be possible to quantify solver reliability for different groups of sequences types and then apply them intelligently depending on sequence type the and amount of data available. Did anyone try anything like this?

# Conclusions
 - False positives were a significant danger. Fit an infinite number of non-linear functions and an infinite number will perfectly describe your sequence, but it doesn't mean the first you find is the correct one, and if it isn't, it's very unlikely to have any predictive value.
 - Looking at the number of mode-fallbacks (~40,000) and linear fits (~37,000) it's clear that the majority of the sequences are constructed by still-unknown functions (unknown in a world where the OEIS doesn't exist, I mean). 
 - Looking at the linear fits, 10496 of the fits were scored perfectly, meaning ~27,000 only represent a polynomial estimations of another function. In addition, an unknown proportion of the 10496 "perfect" fits will be correct by chance, meaning only <1/3rd of the ~37,000 linear fits are likely to be true linear polynomial functions.
 - A guided approach is likely to be more optimal than throwing everything at the wall to see what sticks, whether this be simple [tagging](https://www.kaggle.com/garethjns/integer-sequence-learning/classifying-tagging-sequences), or specifically written solvers for [more sequence types](https://en.wikipedia.org/wiki/Category:Integer_sequences).
