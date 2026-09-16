# Approvers Submission (2nd)

Competition: fide-google-efficiency-chess-ai-challenge
Rank: #2
Source: https://www.kaggle.com/c/fide-google-efficiency-chess-ai-challenge/discussion/569891

## Approvers' Submission

The source code is available at https://github.com/peregrineshahin/Approvers.

### Background

We started out as 2 separate teams, shuffling near the top of the leaderboard. Eventually, we decided to join forces.
Both of us had prior experience as chess engine developers — @peregrineshahin is a Stockfish contributor, a highly skilled professional, and I, @rickonaut, developed my own chess engine as a pet-project.

We came to understand that a highly ranked submission would likely require optimizing all four key aspects of the tournament. With this in mind, our approach was driven by a commitment to developing a chess engine in a dedicated manner, focusing on optimizing for the size limit, memory limit, tested time control (TC), and the opening book.

### Testing

As the gold standard in the chess engine community, we used SPRT (Sequential Probability Ratio Test) to determine whether a change is statistically beneficial and SPSA (Simultaneous Perturbation Stochastic Approximation) for tuning various constants and parameters. In total, we have played around 20M\* games using distributed compute resources. Our GitHub repository has over 1250 branches, each containing different ideas and attempts to improve the submission. We have left all commits and branches intact for historical reference when switching from a private to a public repository. Most functional commits on the `main` branch include descriptions with the result of associated SPRT tests.

Before merging teams, I had my own private local instance of [OpenBench](https://github.com/AndyGrant/OpenBench) up and running, a generic open source chess testing framework, developed by @agethereal, this made it easier to get started rather than debating whether to use [Fishtest](https://github.com/official-stockfish/fishtest) while both can do the job.
Once we joined forces, this setup played a key role in helping us develop and refine the engine together.

\* In comparison, `Fix the bugs?` reported ~38M games.

### Development and Strategies

The starting point is the [Cfish](https://github.com/syzygy1/Cfish), a C port of [Stockfish](https://github.com/official-stockfish/Stockfish).

Unfortunately, although we have over 300 commits in the repository, some commits from before merging the teams were not tracked. However, they might not be particularly relevant to the final submission.

We recognized early on the importance of combining domain-specific knowledge with general development skills.

### Search Features

Due to size limitations, we determined that the most effective strategy for our team was to include four fundamental search features. These features significantly shaped the parameters of our chess engine later on, distinguishing it from conventional engines, generally because these hurt the performance in long time controls:

- Short Time Control (STC) Elo Gainer Optimization – This technique skips root depths at odd plies in Iterative Deepening, a method initially discovered by Shahin while working on Stockfish.
- STC Fail-High Handling – A strategy that moves on to the next root node in the event of a fail-high, originally discovered in the early days of Stockfish.
- Quiescence Search Time Checking – An STC optimization that improves performance in Stockfish-clone engines by monitoring time even during Quiescence Search. This was also identified by Shahin during his previous work on Stockfish.
- Sudden Death Time Control Optimization – A technique developed by the Stockfish team, which scales the Move-To-Go (MTG) parameter dynamically as time control approaches its limit.

Interestingly, we discovered that implementing certain well-known Short Time Control (STC) optimizations, which require a complete retuning of the engine’s hyperparameters, enabled us to incorporate established Very Long Long Time Control (VVLTC) optimizations — techniques that typically do not function under STC conditions!

### Evaluation

For evaluation, we introducd NNUE with a pretty straightforward NNUE architecture adopted in different forms in the chess community — (768x1hm -> 64)x1 -> 1x8.

- 768 input features with horizontal king mirroring. Inputs are flipped along the vertical axis, i.e., a1 becomes h1, b1 becomes g1, etc., based on the position of the friendly king.
- 1 hidden layer with 64 neurons with Squared Clipped Rectified Linear Unit (SCReLU) activation function f(x) = min(max(x, 0), 1)^2.
- 8 output buckets (a layer stack), selected based on the number of pieces remaining on the board (piece_count - 2) / 4.
- It outputs a single number, representing an evaluation of a node in the engine's internal units.

The network training involves 3-stages of progressive training, with each stage restarting from the previous one with modifications, finally followed by an SPSA session. For the full training configuration, see [training/config.rs](https://github.com/peregrineshahin/Approvers/blob/main/training/config.rs) in the repository, compatible with the [Bullet](https://github.com/jw1912/bullet) trainer.

The network is quantized to 8 bits for FT weights/biases and L1 weights, and 16 bits for L1 biases. Also, due to unused features for pawns
(1st and 8th ranks being illegal by the rules of chess) and the mirror squares of kings, the input features are reduced to `704`.

### Size Optimization

To minimize the size of the binary and fit the largest NNUE model while keeping the crucial `-O3` flag for NNUE performance, we did lots of cleanups and simplifications (including functional ones that haven't regressed in our SPRT tests). Additionally, we switched from `gcc` to `clang`, as it produced smaller binaries and at least as fast, later combining with various cflags, `#pragma` directives to disable unrolling on individual loops, and applying a combination of `minsize`, `cold`, and `section(".text.small")` attributes to non-hot functions played a big role for achieving our goal. We also fully removed dependencies on `libm` and `lpthread` by replacing necessary functions with custom implementations and making the application truly single-threaded.

### Local Results

After the source code of all top-3 entries was published, we tested our engine against them. The conditions are as close to Kaggle as possible – 1 thread, 1MB hash, 10s per move (scaled individually based on machine speed to match Kaggle's machine NPS), and the Kaggle opening book. The delay/increment was left unset, as it's unpredictable on Kaggle and causes time losses. One might argue it doesn't even work. So, the only piece missing in our testing was pondering.

In 80K games, there wasn't a single crash or a time loss on any of our machines.

`Linrock` vs. `Approvers`

```
Elo   | -3.95 +- 1.90 (95%)
Conf  | 10.0+0.00s Threads=1 Hash=1MB
Games | N: 40000 W: 7577 L: 8032 D: 24391
Penta | [513, 4465, 10444, 4120, 458]
```

`Fix the bugs?` vs. `Approvers`

```
Elo   | -3.43 +- 1.97 (95%)
Conf  | 10.0+0.00s Threads=1 Hash=1MB
Games | N: 40002 W: 8195 L: 8590 D: 23217
Penta | [576, 4636, 9946, 4293, 550]
```

The top-3 are very close, with a slight edge to our entry. Ultimately, it came down to pure luck due to a highly unstable rating system.
Nevertheless, we had a great time and lots of fun during the competition and hope you did too.

### Bonus

Under the previously mentioned conditions, here's a short match between Approvers and the latest development version of Stockfish at the time of testing (commit `fa6c30af`).

```
Score of Approvers vs Stockfish: 136 - 2012 - 1622  [0.251] 3770
...      Approvers playing White: 100 - 710 - 1075  [0.338] 1885
...      Approvers playing Black: 36 - 1302 - 547  [0.164] 1885
...      White vs Black: 1402 - 746 - 1622  [0.587] 3770
Elo difference: -189.7 +/- 8.4, LOS: 0.0 %, DrawRatio: 43.0 %
```

Please note, Stockfish is optimized for much longer time controls and regresses in such short ones, yet our submission looks quite powerful.
