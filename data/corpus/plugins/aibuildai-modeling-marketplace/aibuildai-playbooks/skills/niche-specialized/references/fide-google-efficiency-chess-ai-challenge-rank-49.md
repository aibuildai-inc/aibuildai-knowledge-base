# 49th Place Solution (or how to get silver in one evening)

Competition: fide-google-efficiency-chess-ai-challenge
Rank: #49
Source: https://www.kaggle.com/c/fide-google-efficiency-chess-ai-challenge/discussion/566862

# Solution

My solution based entirely on the [Ethereal 13.00](https://github.com/AndyGrant/Ethereal) engine (ty @agethereal) with minor changes:

- removing everything when associated with nnue, tablebases, unnecessary uci functions
- 12-bit size transposition table
- disable huge pages (ty @jsday96)
- MAX_PLY = 128, MAX_MOVES = 256 (in types.h)
- -Os -s -flto compiler flags
- default 160ms delay, not increment

# What didn't work

- other uci options: ContemptDrawPenalty, ContemptComplexity 
- increment (idk why)

So, in just one evening, making minimal changes to the open source Ethereal engine could get you a silver medal. The solution weighs only 33.1 KB and does not use NNUE. Sadly, I didn't find enough time for this contest and didn't give the cfish engine the attention it deserved.
