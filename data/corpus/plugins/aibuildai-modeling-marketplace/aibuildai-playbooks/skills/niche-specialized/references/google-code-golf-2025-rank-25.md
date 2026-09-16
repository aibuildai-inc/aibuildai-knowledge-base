# Getting to Rank 25 by Teaching LLMs to Golf

Competition: google-code-golf-2025
Rank: #25
Source: https://www.kaggle.com/c/google-code-golf-2025/writeups/getting-to-rank-25-by-teaching-llms-to-golf

## The Insight That Changed Everything

I finished this competition believing something that most people don't yet realize: **the expert code golfer already lives inside your LLM. You just need to learn how to extract it.**

For three months, I competed in ARC Code Golf using a mix of Claude, GPT-5, and Gemini. Rank 25 wasn't achieved by asking these models to write golf solutions. It was achieved by teaching them to *think like golfers*—to name patterns, remember them, improve iteratively, and discover new tricks I'd never seen before.

The final breakthrough: in the last week of competition, my agent started finding solutions shorter than the leaderboard best on multiple problems. That wasn't random. It was emergence.

## Phase 1: The Foundation (Pre-September)

Before the code golf contest launched, I spent months brainstorming with LLMs on a different problem: building DSLs to generate ARC-like puzzles. This was inspired by Michael Hodel's work on arc-dsl.

```
# Create a grid with 3x3 regions
CREATE GRID 10x10
DRAW DIVIDER AT ROWS 4,8
DRAW DIVIDER AT COLS 3,9

# Task: fill regions
FILL TOP-LEFT REGION BLUE
FILL BOTTOM-RIGHT REGION GREEN
FILL CENTER REGION RED
```

The first three lines setup a generator for a puzzle similar to 55. See image.
In other words some ARC puzzles do a few steps generate that as input image and then do few more steps.
Other puzzles first generate output and then add noise and transformations to generate input.
This difference would help later.

I used Gemini Canvas to build several interactive HTML js utilities to experiment with these mini-DSLs - primarily to generate hundreds of examples of a puzzle and then think about what would a solver do to solve all of them. I did this probably for about 10 or so puzzles from ARC-AGI 2 evaluation set. Only when you build something you begin to understand.

This taught me something crucial—LLMs are excellent at explaining *logic*, not just retrieving knowledge.

When code golf was announced, I realized it split the ARC-AGI problem into two parts:
1. Understanding what a task does
2. Implementing a solver efficiently (assuming understanding is already there)

This realization was liberating. I switched focus entirely. Michael Moffitt's ARC-GEN opened up whole new world of how a framework can be built to create such puzzles.

For the first few months, I used LLMs as **code translators**: feed them a golfed solution, ask what it does; feed them two solutions side-by-side, ask what tricks were used to shrink one. Each chat revealed algorithmic patterns and Python idioms I'd missed. I manually collected good solutions into a baseline notebook, but more importantly, I was building intuition alongside the LLM.

## Phase 2: The False Start (Late September)

I discovered Claude Code and got ambitious. I built a multi-stage agentic pipeline:

- **Analyzer**: Study the task generator, write algorithmic descriptions
- **Solver**: Code a solution from the description
- **Tester**: Validate and rewrite
- **Golfer**: Shrink it
- **Learner**: Document struggles and insights

On paper, perfect. In practice, it collapsed.

The failure was instructive. **Errors in the analyzer couldn't be fixed downstream by the solver.** I was trying to commoditize understanding into sequential stages when understanding *is* the hard part. The pipeline had too many handoff points.

But this forced me to think differently. I unified analyzer + solver + tester. More importantly, this is where I discovered the **ACT framework**:

**Anchors, Clues, Transforms**

The key insight: ARC problems are like solved Sudoku grids corrupted by noise. The author designs a clean output, then adds occlusion and transformation to create the input puzzle. A solver's job isn't to be creative—it's to *recognize invariant objects despite noise and apply specific transformations*.

This reframing changed how I described tasks to LLMs, and it worked. Suddenly we got solutions for all 400 problems.

## Phase 3: The Acceleration (Mid-to-Late October)

Around mid-October, I shifted strategy. Instead of rigid pipelines, I created **feedback loops**.

I manually documented ~40 code-golf tricks:

- Flatten + divmod for 1-D neighbor scans
- Branchless recoloring (color × predicate preserves zeros)
- Slice truthiness as guard + predicate
- Hardcoding block sizes to drop generalization
- Collapsing multi-stage logic into lambda expressions

Then I gave these to the agent as a **checklist** and asked it to review each solution:
1. What invariants does this problem have?
2. Which tricks apply?
3. What bytes can we save?

But here's the key innovation: **I asked the agent to write out *new* tricks it discovered, in the same format**.

The agent started naming tricks I'd never considered. Each new trick got added to the checklist. The list grew: 40 → 80 → 140 tricks. And as it grew, the agent's solutions got better.

## The Breakthrough: Design Review Format

The final leap came when I asked the agent to output a structured **design review** after each optimization pass:

```
== design review:
Task: Task 004 (tilt) — trim solver by ≥5 B (176 B baseline).
Invariants: Rectangular grids; colors 0 + one active hue per cluster; 
every color tilts right when matching hues exist below-right.
Representation: Keep row-major grid, triangular any(...) over g[-~y:] 
wrapped in lambda solver with _ aliasing enumerate.
Checklist hits [33,40,45,66,68,82]: Alias builtins; truthy window any; 
lambda solver; boolean product gate; skip bounds checks; generator suffix.
Plan: Review → Note idioms → Apply checklist → Try unary rewrites → 
Micro-trim → Re-test.
Bytes: 176 B → 119 B
```

Then, after each optimization, the agent would journal its thought process:

> Candidate 7: Checklist pass revisited below/right gate. Realized two predicates collapse into single triangular `any(v in r[j+1:] for r in g[-~y:])`. Dropped row suffix flatten and sum alias for 19 B shave.

This journaling did something magical: **it created a record of reasoning that the agent could learn from**. On the next pass, it could reference earlier sessions. Patterns emerged. The agent started discovering *algorithmic* improvements, not just syntactic ones. (see images for detailed examples)

In the last three days, performance skyrocketed.

## The Construction Principle: Why This Problem Was Solvable

Before I could teach an LLM to golf code, I needed to understand *why* ARC puzzles have a structure worth compressing.

Here's the key insight: **ARC puzzles are constructed like Sudoku, but generalized.**

A Sudoku author first constructs a valid solution grid where every row, column, and region contains 1–9 exactly once. Then they erase most values, leaving you to fill the rest. The solution is *already there*—your job is inverse reconstruction.

Francois Chollet, the ARC author, thinks similarly. He imagines a target grid that is clean, orderly, and follows some hidden rule. Then he *corrupts* it: scattering objects, erasing values, adding noise pixels, punching holes. The puzzle becomes recognizing the objects despite noise and transformation, then applying the intended transformation.

This clicked for me because of my background. My master's thesis was titled "Transformation Invariant Object Recognition from Noisy Data Using Artificial Neural Networks." My advisor, Prof. B. Yegnanarayana at IIT Madras, framed the problem directly from Claude Shannon's noisy channel model—the foundational formulation for cryptography and communication theory. The core question: given corrupted information, how do you recover the original intent?

ARC *is* this problem, posed as visual puzzles.

Understanding this reframed how I described tasks to LLMs. The ACT framework (Anchors, Clues, Transforms) wasn't an invention—it was recognizing the problem structure. I wasn't asking the model to be creative. I was asking it to reverse-engineer the author's compression: what was the clean grid, and what noise was applied?

This understanding was foundational. It made every prompt more precise, every agent more effective.

## Why This Worked: The Real Lesson

Here's what most people misunderstand about LLMs:

LLMs are trained to be verbose because that's what RLHF rewards in chat. But verbosity isn't their ceiling—it's an artifact of training. Underneath, they grasp algorithm internals, language semantics, and pattern compression at a level humans rarely achieve consciously. Let me briefly mention here that when compression is done on discrete symbols it leads to codebooks. When codebooks are contextual, they are powerful. When compression happens on embedding space with overlayed vector superimpositions, it is comprehension.

Language itself is a compression mechanism. When you say "rotate until it touches the edge" instead of "move right 4 cells, now move right 2 cells, now move right 1 cell," you've already compressed the logic. LLMs excel at this kind of abstraction.

**The trick: give them constraints, examples, and feedback loops.**

When I fed the agent:
1. A checklist of 140 named tricks
2. Task descriptions in ACT format  
3. A design review template showing my reasoning
4. Permission to add new tricks it discovered

...it stopped being a code-writing tool and became a *collaborative partner* in a compression game.

### The 10x Multiplier Effect

What surprised me most wasn't the final result—it was the *trajectory*. Each of three components, stacked together, multiplied capabilities by roughly 10x:

1. **LLM Intuition (10x):** When I stopped asking the model to write code and started asking it to *understand* tasks using the ACT framework, solutions improved dramatically. The model had algorithmic comprehension; I just needed to unlock the right abstraction layer.

2. **Reasoning Traces (10x):** When I made the agent write out its reasoning (design reviews, journals, checklist hits), its next iteration was 10x more systematic. Reasoning traces externalize thought, making patterns discoverable and refinable.

3. **Agentic Experimentation (10x):** When I closed the loop—letting the agent discover new tricks, add them to the checklist, and apply them on the next pass—we entered a feedback cycle that felt exponential. The agent wasn't just optimizing; it was learning to optimize.

Stacked: 10x × 10x × 10x ≈ **1000x improvement in solution quality over three months**, though I can't measure it precisely. What I can measure: 40 tricks → 140 tricks, baseline solutions → leaderboard-beating solutions, my manual golfing → consistent agent outperformance.

The insight: these three aren't optional features. They're *multiplicative*. Without reasoning traces, intuition stays buried. Without agentic loops, traces are static documentation. Without intuition, agents write verbose code that no loop can fix.

Another thing to notice is that we got LLM intuition accessible publicly since 2023. Reasoning accessible since October 2024 and Agentic Coders since March-April 2025. We are just getting started!

### The Grammar Breakthrough

Late in the competition, I had a realization that connected centuries of ideas.

Panini was a Sanskrit grammarian who lived around 400 BCE. Rather than writing descriptive grammar with examples and rules, he wrote *generative* grammar—production rules so dense and powerful that running them could generate all possible Sanskrit sentences. He compressed the entire language into ~4,000 rules. Panini recognized verbs as operators and nouns as arguments. His insight was that language can be defined by a finite, ordered set of generative rules with scoping, precedence, and compact metasymbols - the very heart of compiler front-ends - was realized with extraordinary rigor in the Aṣṭādhyāyī two millennia before BNF. Modern formalisms (Post→Chomsky→BNF) gave us the tools compilers use; Pāṇini’s system shows an ancient, remarkably close analogue to that way of thinking.

When I read comments from top competitors (JoKing, Jacekwl) about reusing a handful of code patterns across many problems, something clicked. I spent a night hand-solving the shortest ~40 solutions from the leaderboard, looking for the underlying *form*.

What I found was a generative grammar for ARC solvers:

```python
lambda g: [
  [transform(cell_value) 
   for cell in transformed(row) 
   if condition] 
  for row in expanded_transformed(rowset) 
  if condition
]
```

This single template—with variations in `transform()`, `transformed()`, `expanded_transformed()`, and `condition`—explained most of the compact solutions I saw. It wasn't random brevity. It was a *generative grammar* for ARC problem-solving, discovered by centuries of competitive programming.

When I showed this pattern to the agent and asked it to recognize this grammar in other solutions, something changed. The agent stopped writing ad-hoc code and started *composing from a grammar*. It was no longer optimizing individual solutions; it was learning to think in the language of compression itself. It was using terminology of grid, rowset, rowset expansion, filtering, row extension, cell transformation etc, because I had seeded that in a few fewshot examples. With a grammar to guide, solution search becomes pointed tweaking of parts of the solution. Both the frame of slots and expression lists for each slot type are fixed making the solving (not just the golfing) almost algebraic rearrangement of problem specification.

This is what Panini did for Sanskrit. This is what Shannon did for information. This is what the best code golfers had discovered empirically: reduce your domain to its production rules, and the compression becomes automatic.

In the last days of contest, my obsession with sentential forms went away, and just a list of rules can serve as a grammar system!

## Results

The final submission at Rank 25 used:
- ~30 problems solved manually
- ~370 problems golfed primarily by agents
- Solutions on 3+ problems that beat the leaderboard best (in the final week)

The agent didn't "know" these solutions existed. It learned to *think like a golfer*, which meant:
- Recognizing when a representation shift saves bytes
- Applying chains of micro-optimizations systematically
- Naming emergent patterns and refining them

## A Beautiful Asymptote

Here's something that humbles me: the agent only truly came online a few days before the competition ended. It never attempted ~100 problems. 

Part of me looks at Rank 25 and thinks: *if I'd had one more week, top 10 was reachable.* The trajectory was exponential. The tricks were accumulating. The agent was learning faster than I could keep up.

But then I remember why that doesn't matter.

The humans in the top ranks—JoKing, Jacekwl, Garry Moss, Ken, and dozens of others—spent *years* poring over similar problems character by character, like I myself had done in my long career spawning Fortran to Python via C, C++, VB, Java, Lisp, SAS. They've distilled decades of collective programming and golfing experience and they have generously shared a lot. Their solutions aren't just code; they're crystallized insight.

It is beautiful justice that humans should be the winners.

Everytime an agent goes up a notch in a leaderboard, it discourages a bunch of humans. This is harmful to human society as a whole, since it discourages humans from thinking hard and working hard. This is a serious issue we need to think about and address as AI is adopted widely.

What I'm proud of is something different: I've created a thought calculator that can give them free time. Soon, I hope to make this reproducible from scratch and open-source it. The goal isn't to replace human expertise—it's to amplify it, to let the best minds in the world focus on the breakthroughs that still require human intuition.

Also each one of the solutions is explainable. By implementing a kind of micro descent, the solutions are a series of short rewrites (sometimes rethinking of representation) so one can see the path of the rules used. Hopefully this opens ideas on how to build agents that act in a limited action space, while still being flexible and able to add to it.

I think the solution created will be useful for compiler optimization, new forms of code such as GPU kernel and in general domain-specific skill development.

The agent learned in days what took the best golfers years. But that's not a reason to rank higher—it's a reason to build better tools for the next person who wants to push further.

## The Takeaway

You don't need a smarter LLM to solve harder problems. You need **better abstraction of your reasoning**, fed back as constraints and examples.

Language is logic. By being abstract and vague, it brings expressivity and generality. Teach an LLM your abstractions—not just your goals—and watch it innovate at a speed you can barely track.

The expert golfer was always inside the chatbot. I just had to prompt to use that dialect. And perhaps more importantly: at least in this case, I saw where LLM's intuition had its limit and my intuition can flourish.

---

**P.S.** I'll be releasing the tricks checklist, ACT framework, and design review templates as a notebook as I find more time. I hope to at least walk through some of the agentic setup on some Youtube tutorial. There is so much more to learn from the masters, now that all have shared their solutions! This will give totally new views into what sort of abstractions are totally missing in the agent so far.

** Thanks ** I cannot thank enough Google for Gemini, Anthropic for Claude and above all the OpenAI team for releasing GPT5 codex, which proved to be an uncanny assistant at all levels from helping me organizing code, testing every idea, managing a git repo, troubleshooting potential agonizing situations of score dropping, in a few minutes. Codex on the Web was a game changer for the last leg.
