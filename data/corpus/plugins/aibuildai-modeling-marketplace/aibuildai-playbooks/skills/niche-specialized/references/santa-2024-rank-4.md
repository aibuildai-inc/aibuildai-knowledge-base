# 4th Place Solution

Competition: santa-2024
Rank: #4
Source: https://www.kaggle.com/c/santa-2024/discussion/560536

# Main Solution
Our main solution is loop of followings(ILS). Beamsearch with kick by @daiwakun 
1. Insert optimize
  - Delete randomly selected N words and insert them with beam search



2. Local optimize
  - Repeat followings until No Improvement
    - Shuffle Subsections
    - Swap Words
    - Move Words
    - Exclude Identical Arrangements
    - Remove duplicates



| id | len(words) | score  | time  | Possible score |
|----|-----------|--------|-------|---------------|
| 0  | 10        | **469.77** | 3 m   | 469.77        |
| 1  | 20        | **424.38** | 10 m  | 424.38        |
| 2  | 20        | **298.93** | 10 m  | 298.93        |
| 3  | 30        | 198.93 | 10 h  | 191.73        |
| 4  | 50        | 74.33  | 1 d   | 67.54         |
| 5  | 100       | 36.58  | 1 d   | 28.52         |

## Finding 1
These words increases **valid_length**. We will get an advantage by putting these at the beginning. (**Bold** is id3)
- reindeer **jingle sleigh** gingerbread peppermint **decorations ornament** wreath **magi stocking chimney** fireplace



Regarding id3, if you start with "**magi**," the optimal solution can be discovered in about 3 hours in our algorithm.
FYI, id0 and id1 starts with **reindeer** and id2 starts with **sleigh**.

We also found that consolidating these **function(stop) words** in one place and placing them at the beginning tends to result in lower loss. Optimal answer of id4 is **Function words + Content(other) words**.
- **Function words: and as from have in is it not of that the to we with you**
Id4: 50! → 14! + 36!
Id5: 100! → 20! + 80!

| id | len(words) | score  | time  | Possible score |
|----|-----------|--------|-------|---------------|
| 3  | 30        | **191.73** | 3 h   | 191.73        |
| 4  | 50        | **67.54**  | 6 h   | 67.54         |
| 5  | 100       | 32.4   | ???   | 28.52         |

## Finding 2
Regarding id5, we also found 
- Alphabetical order of content words decreases loss
- Splitting alphabetical content words decreases loss

So we thought the optimal shape of id5 is roughly **Function + sorted(Content1) + sorted(Content2)**
Order would be 20! + 80C40?
32.41 → 28.574555(**LB 246.82532**)

Lastly, we found **and** should be removed from function words.
28.574555 → 28.529695(**LB 246.81784**)


https://gist.github.com/KazukiOnodera/ff74dd5d9171cddac773e03f7d3457f3
