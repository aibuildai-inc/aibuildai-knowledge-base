# [9th Place Solution] Cfish + Simple SPSA

Competition: fide-google-efficiency-chess-ai-challenge
Rank: #9
Source: https://www.kaggle.com/c/fide-google-efficiency-chess-ai-challenge/discussion/567106

I'm sharing the solution that earned 9th place. The final code is publicly available at the following link:

[https://github.com/ymgaq/Cfish_kaggle](https://github.com/ymgaq/Cfish_kaggle)

# Selecting the Base Program

After testing multiple chess engines, I finally chose [Cfish](https://github.com/syzygy1/Cfish).

As noted by other top participants, Cfish has the advantage of lower memory consumption due to glibc compared to C++-based engines. Additionally, in Kaggle's single-core environment, it appeared to slightly outperform Stockfish in search speed.

# Memory Optimization
- Removed NNUE, relying exclusively on HCE evaluations.
- Reduced the Transposition Table size to 1MB.
- Counter Move History:
  - Reduced branching by merging conditions (inCheck, CaptureOrPromotion) (1/4 reduction).
  - Changed Counter Move History indexing from pieces to types of pieces (1/2 reduction).
- Reduced Material Table and Pawn Hash Table sizes to 1024.

These adjustments resulted in memory usage of around 4MB.

# Binary Size Compression
- Removed unnecessary functions and classes (NNUE, Bench, TBProbe, PolyBook, NUMA, etc.).
- Removed unnecessary UCI options.
- Optimized build commands:
  - make build -j ARCH=x86-64-bmi2 EXTRACFLAGS="-ffunction-sections -fdata-sections" EXTRALDFLAGS="-Wl,--gc-sections"
- Used `strip` command.
- Compressed binary with `upx --lzma`.

These reductions allowed the binary size to remain under 64KB, even with O3 optimization instead of Os.

# Enhanced Search

Cfish’s search initially matched Stockfish13 but was modified to resemble Stockfish16’s HCE search methods and parameters. This modification improved the engine by roughly +30 elo compared to the original search implementation. By using O3 instead of Os optimization, the average search speed increased by approximately 8-9% in Kaggle's CPU environment.

# SPSA

Chess engines commonly use a powerful black-box optimization method called [SPSA](http://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF) (Simultaneous Perturbation Stochastic Approximation).

Simply put, two versions of the engine with parameters perturbed by +δ and -δ compete against each other, and the winning version’s adjustments are accepted. By randomly perturbing and testing all parameters simultaneously and repeating matches, the optimal parameters are identified.

While robust SPSA libraries such as [fishtest](https://github.com/official-stockfish/fishtest) or [OpenBench](https://github.com/AndyGrant/OpenBench) exist, I opted for a simple custom SPSA script optimized for a single local machine. [My script](https://github.com/ymgaq/Cfish_kaggle/blob/kaggle_update_params_o3/scripts/spsa.py), around 400 lines, was based on literature and an available [Perl implementation](https://github.com/zamar/spsa). It utilizes cutechess-cli and runs SPSA tests concurrently on multiple threads, adopting the competition’s 2000-opening book and 10-second time controls.

This straightforward approach yielded approximately +30 elo improvements, pushing my solution to a Gold Medal.

Considering the randomness of the LB, I submitted two identical binaries as my final submission.

# Other Attempts (Not Adopted)
- Tested Ethereal according to publicly shared notebooks.
- Evaluated memory optimization strategies for all HCE versions of Stockfish 6 to 16. Migrated to Cfish to meet the 5MB memory constraint.
- Attempted training a (768x2)-10-1 NNUE network using bullet, but unfortunately, I could not surpass the performance of HCE.
- SPSA adjustments for time control parameters were not effective in LB testing, hence not adopted.


I found other winning solutions utilizing NNUE and neural networks extremely intriguing and slightly regret limiting my approach exclusively to HCE. Though this was my first experience developing a chess engine, optimizing within constraints akin to embedded systems proved highly engaging. I extend my sincere gratitude to the competition organizers and fellow participants for their extensive and valuable contributions.
