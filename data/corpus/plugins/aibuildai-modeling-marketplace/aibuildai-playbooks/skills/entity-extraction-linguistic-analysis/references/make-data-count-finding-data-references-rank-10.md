# Tricks (10char)

Competition: make-data-count-finding-data-references
Rank: #10
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/tricks-10char

All DOIs from MDC corpus v3
- All accession ids from MDC v4 using only `eupmc` source. If article is not in it just match using known prefixes.
- There is one accession that gives a +0.08 boost in both LB if you remove it from predictions, it's also the most popular new prefix in my sub but I don't know what it is.
- The second most popular new prefix however is a "good" prefix and should be whitelisted i.e. always matched.
- All `SAMN` are primary.

Actually spent a lot of time building different modeling approach but in the end nothing beat probbing. Well I guess it's about the gradients we descended along the way, or something.
