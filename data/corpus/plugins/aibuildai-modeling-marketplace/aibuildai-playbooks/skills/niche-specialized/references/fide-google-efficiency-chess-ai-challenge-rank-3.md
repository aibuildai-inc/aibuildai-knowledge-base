# "Fix the bugs?" Solution Write-up (3rd)

Competition: fide-google-efficiency-chess-ai-challenge
Rank: #3
Source: https://www.kaggle.com/c/fide-google-efficiency-chess-ai-challenge/discussion/569874

Our solution: https://github.com/AndyGrant/KaggleFish

This post provides an overview of the “Fix the bugs?” team members, and the primary components of our submission, in some technical detail.  For this post, I will be making reference to the following greatly:

- OpenBench (https://github.com/AndyGrant/OpenBench), a distributed chess engine testing framework which is utilized by the vast majority of engine developers
- Sequential Probability Ratio Testing (SPRT) (https://en.wikipedia.org/wiki/Sequential_probability_ratio_test), the primary mechanism behind engine development
- Simultaneous Perturbation Stochastic Approximation (SPSA) (https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation) the primary tuning mechanisms for engines in contexts where simple derivatives do not exist.
- Grapheus (https://github.com/Luecx/Grapheus), a chess-centric neural network training tool written by Finn Eggers, one of the principal authors of the Koivisto and Torch chess engines.
- Stockfish (https://github.com/official-stockfish/Stockfish), as everyone already knows, is the strongest chess engine available in both private or public by a significant margin.
- CFish (https://github.com/syzygy1/Cfish), a port of the Stockfish engine into the C language, which was discontinued in ~2021.


**Section 1: The Team**

Andrew Grant; United States of America; andrew.kaggle@grantnet.us		

I am a software developer for chesscom. My principal work was writing the Torch chess engine as a replacement for use in chess.com's many products and tools. For this competition I made use of my prior experience with my own commercial engine venture Ethereal, my experience developing the OpenBench chess engine testing platform, and just general knowledge from years of contributing to virtually all top engines, including Stockfish, Leela, and Komodo Dragon. The initial plan for this competition was to spend about twenty hours to secure a top-4 finish. However, by the end, there was an interest in trying to win first when it became clear that the winner would be decided by a coin flip. I only became meaningfully involved in the event in the final month, and probably spent ~25 hours per week, using the weekends and evenings. I partnered with my good friend Kimmy, with whom I worked on Torch. We formed the team after I had already done a lot of the boiler plate work. My contributions primarily were stripping down a clone of CFish, designing and training the Neural Network that powered the engine’s evaluation function, managing the compute resources through an OpenBench instance, and just generally working towards adding strength to the engine by back porting known ideas.

Kim Kahre, Finland; kimkahre@outlook.com

Chess engine development has been one of my main hobbies for quite a few years. I also worked on the Torch chess engine for chesscom. All the tinkering I’ve done with Koivisto as well as Torch has certainly been useful. It seemed like an interesting challenge, I thought there was potential for search to behave somewhat differently with such limited resources (possible differences in Transposition Table replacement schemes etc). I very much enjoy the process of chess engine development more generally too. Given the chance to team up with Andrew, I didn’t have to think twice. I probably spent ~20 hours a week but only for the last few weeks of the competition. My contributions are primarily trying variations of as many search ideas/tweaks as possible. 


**Section 2: Initial Efforts In The Competition**

When this competition was announced, I was quick to submit a pruned version of my own commercial engine Ethereal (https://github.com/AndyGrant/Ethereal), in order to establish a baseline metric to compare other submissions to. In the end, many users submitted versions of Ethereal. I took Ethereal, removed the Neural Network, removed all support for multi-threading, tablebases, and other non-essential components. I also shrunk the size of various pre-computed tables and hash tables. With those steps I was able to submit a version of Ethereal using the hand-crafted eval (HCE) thanks to this wrapper script (https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/548061) from Alexander Tian, who is the author of the Altair chess engine, and a contributor to chesscom’s Torch engine. This script allowed ease of access to the competition for engine developers, by establishing a wrapper between Kaggle’s environments and the typical engine communication protocol we are used to.

This submission was able to quickly reach the top of the event, and stayed there for quite some time. As a one-off development effort, I added support for Pondering. Pondering might be obvious to non-engine developers, but it's the act of thinking on your opponent's turn. This is not something that is done really anymore in engine development, since it raises concerns about hardware resource allocation and fairness. The typical pondering mechanism is when you make your move, you guess what the opponent's move will be. Then while you wait, you conduct a search as if your guess was correct. If you were right, then you get a “ponderhit”, and you have saved a lot of time. If you were wrong, then you’ve at least explored some positions that might arise via transpositions. Engines typically manage this by having a dedicated thread in their engines that poll stdin that wait for a “ponderhit” or “stop” signal. Due to the limitations for this event, that now gets implemented by occasionally processing lines from stdin during search. 

After this, there was no more effort spent to work on the Ethereal submission. It sat at the top of the board until various versions of Stockfish forks started to take over. It was quite clear then – and really much earlier to me – that the winning solution(s) would all be CFish forks, with varying degrees of development effort, and varying degrees of gutting.


**Section 3: Steps To Minimize CFish Memory And Size**

One of the principal challenges in this competition was to reduce the size of the binary. Reducing the size of the binary by another kb would allow for another one plus kb of weights in the neural network. The space is so critically important, that I would assert that if any of the top three contenders were to have been allowed 128kb, they would have won the competition by a mile, far beyond the profoundly statistically insignificant results we saw here. 

Most of the initial removals are not worth talking about in detail. But here are some of them: Delete tablebass, delete polyglot opening books, delete NNUE implementation, delete NUMA support, delete threading support, remove large pages, delete multipv search, delete most components of the UCI interface, delete skill levels, delete excess engine output, delete cuckoo tables.

In order to keep our submission usable by the OpenBench framework, we had to start doing preprocessor defines to exclude code only for the Kaggle builds. Using this mechanism, we were able to fully minimize all engine input and output. In the proper submission, the only inputs the engine takes are “position fen … moves …..\n go wtime … btime …”, and the only output produced is “bestmove … ponder …”. 

Our solution was developed on Ubuntu 20.04.6 LTS via WSL. One of the first steps was to build all versions of clang, and see which produced the smallest binary size, without meaningfully changing the engine’s speed. We arrived at clang-10 as the most optimal. As many other competitors found, we used the following additional clang flags to reduce the binary size: -fno-unwind-tables -fno-asynchronous-unwind-tables -fvisibility=hidden -fexperimental-new-pass-manager

Despite best efforts, still more work needed to be done. The next leap in size reduction comes from marking non-critical functions with __attribute__((minsize, cold)), to prompt the compiler to produce significantly smaller versions of those functions. These are applied on essentially all functions that are not in the hotloop for the engine search. Some functions which were inlined were duplicated instead, making a hot inlined version, and a cold noinline version, to further reduce the binary size. 

The python wrapper itself was written in a fairly obfuscated way, and further reduced with the python-minifier utility when building our submission archive. Our actual engine binary was compressed using xz (lzma), and was decompressed inside the python script, as lzma tools were missing from the kaggle environment. We would have preferred to use 7zip, which produced a smaller binary, but despite being allowed to upload a 7zip file, the support for decompressing it inside the kaggle environment did not work. 

There were a number of other efforts to reduce the binary specifically as it concerns the neural network weights. Those will be addressed in the final section, once context has been provided about the weights.

As it concerns memory usage, lowering the amount of RAM used was not much of a hurdle for our submission. We took the obvious first steps of reducing the transposition table. As well as deleted now-obsolete tables like the Pawn Hash table and Material Hash table, after swapping from HCE to an NN.  The major savings came from: using avx2-bitboards for move generation, a method which allows you to save ~200kb in precomputed tables, at the cost of a greater binary size; deleting dimensions of the Continuation History Tables, which accounted for 8MB+ initially; Removing the need to link against libmath; Reducing the max internal search depth to reduce table sizes greatly; Removing the need to link against libpthread. 

Our final submissions made use of a 768kb Transposition Table. According to our measurements using pmap and other tools, we could have used another 1024kb and still been within the 5MB limit, at least locally. We measured doubling the hash table to be worth 14 elo, but decided it was better to stay well under the limit, in case the Kaggle environment was changed, which it was.

**Section 4: Improving The Underlying Engine**

As strong as CFish is, it is lacking a significant number of discoveries made between 2021 and today. We exhaustively tested many of these changes using our OpenBench framework, which was powered by my machines (~144 threads), Kimmy’s machine (~32 threads), machines from Styx who generously donated hardware (~384 threads), and in the final days Google Cloud Compute (~256 threads). Taking a quick look through our testing log, here are some of the things we added to the engine, with a short explanation.

- +11 elo. Pawn Correction History. This was discovered by the author of the Caissa chess engine, Michal Witanowski. What this does is maintains a hash table, indexed by a hash of the underlying pawn+king structure, which keeps track of a running approximation of the difference between the static evaluation for a position, and the search evaluation for a position. This improves the quality of the static evaluation for like-positions over time. https://www.chessprogramming.org/Static_Evaluation_Correction_History
- +11 elo. Non-Pawn Correction History. Same as the above, but indexed by a hash of the pieces and their positions for a single player, excluding their pawns.
- +6 elo. Minor/Major Correction History. Like the above, creates two new hash tables. One is indexed by the hash of the pieces and positions for the minor pieces (Knight + Bishop + King), and the other for the major pieces (Rook + Queen + King)
- +6 elo. Counter Correction History. Like the above, except indexed by the previous played move. 
- +2 elo. Counter Move Pruning using a Depth Based Margin. Counter Move pruning is a mechanism in engines that looks at a history score for a move, and decides to not search it, if the search depth is low, and the history suggests the move is bad. We improve this as is commonplace, by scaling the definition of “bad” as a function of the depth. 
- +6 elo. Various mechanisms that mix the Fail-Soft and Fail-Hard framework. Academically, there are two branches of AlphaBeta pruning, fail-soft and fail-hard. Engines tend to employ fail-soft, as it allows for additional information. But it has been found that mixing the two, in order to temper estimates outside the original search bounds, is an improvement. 
- +7 elo. Late Move Reductions Deeper. This concept applies extensions or reductions inside of typical LMR loops, based on information that is derived from fail-soft about the distance between the move’s score, and the score of our best move thus far. 
- +9 elo. Prior Counter Move Bonus. Applies a very complicated bonus to the history scores for a move, when that move was so good our opponent had no followup.
- +1 elo. Altair Fail-Soft Multicut. Multicut says that if two moves in a position look pretty good, then we can assume one of them is indeed good, and stop searching. This mechanism is improved by using fail-soft results instead of fail-hard results, an idea from the Altair chess engine.

In addition to these patches and many more, a significant amount of elo was obtained using SPSA tuning. Essentially every single constant value in the engine – which can be viewed as a sort of hyperparameters for search, were throwing into an SPSA optimizer on OpenBench, where we would play tens of millions of games with small adjustments to the values, and hone in on something closer to the optima. It is not great in terms of convergence, but reliably improves strength with minimal developer effort. 

We tuned every single search parameter and time management parameter that was applicable. We also tuned the neural network’s weights in the same way, for the L1, L2, and L3 weights. This can be viewed as a sort of post-training, real-world tuning. Training a neural network minimizes a loss function, and we hope that this loss function is a good proxy for elo, but there are clearly some disconnects.

**Section 5: Neural Network**

Due to the nature of this particular competition, the “Model” is only a fraction of the final solution. The Neural Network trained for this event was done using an open source tool named Grapheus, written by another developer of the Torch chess engine. This tool implements your typical suite of tools for training simple feed forward neural  networks, but with a specialization towards chess based inputs. This framework was fed using my personal collection of data generated by playing the Ethereal chess engine against itself and adversaries, and training models based on those game results. That data is not publicly available, but can be published if necessary to satisfy the conditions of the competition. 

Modern chess engines generally use highly over parameterized input sets, but due to the space constraints for this competition, we opted for something simple. We started with 768 inputs to the neural network. 768 is the result of there being 2 players, 6 piece types, and 64 squares. 2x6x64=768. However, we reduced this as many other teams did, by removing the 32 squares which pawns may not occupy (the first and eighth ranks). We also removed another 32 squares by always ensuring the King for the player we were evaluating was mirrored to be on the E-H files. If the King was on the A-D files, we could flip the entire board layout. We believed the Kings and Pawns to be the most important features, so we duplicated them to extract additional value, without having to duplicate the other parameters, saving space. Selecting these features was quite natural, as most chess engine developers start with this exact architecture, due to its small size and ease of training without overfitting. 

The network was trained using a simple batch gradient descent with an ADAM optimizer, fitting a loss function that mapped a sigmoid of the neural networks output to [0..1], matching up with game result labels of 0.0 for a loss, 0.5 for a draw, and 1.0 for a win. Attempts were made to do more interesting things – knowledge distillation with higher quality datasets or networks; efforts to prune and retrain the network in stages; and using more widely used datasets like those made available by the Leela Chess Zero project – but ultimately none of these methods proved any better than the initial attempts. This may be due to a bias towards the existing system, and also the fact that only so much can be done with a mere ~48kb worth of neural network weights. 

Models were trained using two stages of training. Both stages were 500 “super batches” (epoch) of 1024x1024x128 training samples. The first stage started with an LR of 0.001, and had a drop of 40x after the 400th epoch. The second stage picked up with an LR of 0.00025, and again did a 40x drop after the 400th epoch. The primary difference between the stages is that the first stage was trained to fit the model towards an average of sigmoid(label_search) + label_result, commonly referred to as “50/50 eval/WDL” training in chess circles. The second stage was trained purely on the label_result, known as “Pure WDL”. All models were trained on my own personal 3090RTX at home, and took about ~8 hours per stage to reach the desired training steps. Inference rates for the model are almost moot, as it pushes beyond tens of millions of inferences per second. 

The model architecture definition can be seen here: https://github.com/AndyGrant/ClosedGrapheus/blob/kaggle/src/models/ethereal.h#L90-L136

A quick summary, which assumes some understanding of typical chess models:
- We use a 768/PSQBB setup, with king mirroring and pawn square removals
- We use the “half” paradigm, where each side has their own FT accumulator
- There are two FTs per player; one is the usual PSQBB, the other is only the friendly king (mirrored) and all pawns
- Individual FTs use pairwise multiplication activation after a clipped ReLU step
- All FTs are then concat’ed, placing the side-to-move’s FTs first.
- Results are then fed into “L1”, an affine transform producing 8 outputs. 
- Activation using ReLU
- Results are then fed into “L2”, an affine transform producing 16 outputs. There are 8 copies of the L2, depending on the number of pieces on the board. 
- Activation using ReLU
- Results are then fed into “L3”, an affine transform producing 1 output. There are 8 copies of the L2, depending on the number of pieces on the board. 
- Results are joined with a a “skip neuron” that sums up raw material values
- From here a sigmoid is taken for the sole purpose of fitting the loss function

The use of the “dual” FTs, allowing for extra specialization concerning king and pawn structures, is the most meaningful difference between our solution and others. I believe we also had one of, if not the largest neural networks submitted as a result.  This, paired with int8 quantization, and compression tricks like transposing the weights, or factoring out chunks of weights non-functionally, helped provide this edge.

Our other edge was that we followed a rigid, statistically meaningful testing regime. All changes were tested using two rounds of Sequential probability ratio test at a time control comparable to the Kaggle environment. This gives us high confidence that our changes were meaningful. In general, this is a highly important practice for computer chess engine development. We were able to do this thanks to pooled resources, and our decades of experience prior to this event.
