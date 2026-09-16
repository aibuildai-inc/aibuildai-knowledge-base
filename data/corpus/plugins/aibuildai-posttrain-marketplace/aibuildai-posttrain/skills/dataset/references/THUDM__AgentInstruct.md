# THUDM/AgentInstruct

1,866 ReAct-style agent interaction trajectories - thought-and-action dialogues over six real-world tasks - with a per-turn boolean flag marking which assistant turns to train on.

**AgentInstruct** is the training dataset behind "AgentTuning: Enabling Generalized Agent Abilities for LLMs" [1], built by Tsinghua's KEG group (published on the Hub under the THUDM organization, now renamed `zai-org`) to fine-tune Llama-2-chat into the AgentLM series. Each row is a ReAct-format conversation - GPT-4 acting as the agent, interleaving "Think" and "Act" steps against an environment - collected over six tasks (ALFWorld, WebShop, Mind2Web, Knowledge Graph, Operating System, Database) and kept only where GPT-4's trajectory passed a reward-based correctness check [1]. **No licence is stated anywhere on the repository or in the paper.** The paper's own contamination analysis (Table 7) found 6 of 50 ALFWorld and 1 of 200 WebShop test examples "dirty" by its >80% token-overlap definition; it attributes the ALFWorld cases to test tasks differing from training tasks by only a word or two (e.g. a test task asking to cool a mug matching a training task that asks to heat one, otherwise identical), and the single WebShop case to a shared price-constraint phrase appearing across otherwise unrelated tasks, and states its overall conclusion is no evidence of data leakage [1]. The repository's data viewer 404s because its `dataset_info` uses an old loader layout the current server does not parse from the card metadata alone [2][3]; the parquet files and the underlying rows are reachable regardless (below). It lives at https://huggingface.co/datasets/THUDM/AgentInstruct , which redirects to the current canonical id https://huggingface.co/datasets/zai-org/AgentInstruct [2].

**Use it for**: reasoning-trace SFT on agent trajectories - the SFT method card, using the per-turn `loss` boolean to mask which assistant turns are trained on rather than training every `gpt` turn. No licence is stated, so treat reuse as unresolved rather than open. There is no train/test partition in this file: the six "splits" are task categories, not a held-out evaluation slice.

**Licence**: not stated. No `license` field in the repository's card metadata, no `license:` tag on the Hub listing, and no licence text or mention of one anywhere in the README [2][3].

**Shape**: 1,866 rows in one config (`default`), split six ways by task - `alfworld` 336, `webshop` 351, `mind2web` 122, `kg` 324, `os` 195, `db` 538 - two columns (`conversations`, `id`) [3][4].

**Hold out**: nothing within this file itself - there is no train/test partition, only the six task splits. The origin paper's own contamination table found a small number of "dirty" test examples against this training data (6 of 50 in ALFWorld, 1 of 200 in WebShop, over 80% token overlap by its definition); it attributes ALFWorld's to near-duplicate task phrasing and WebShop's single case to a shared price-constraint phrase across unrelated tasks, concluding overall that it found no evidence of data leakage [1]; see Quality.

**Origin**: built by THUDM (Tsinghua KEG group, Hub organization now renamed `zai-org`); every trajectory is a GPT-4 generation, filtered by an automatic reward score rather than by human review [1]. Hub API at the check date: `downloads` 1,135, `likes` 238 [2].

**Trained-on-by**: the origin paper's own AgentLM-7B/13B/70B models, fine-tuned from Llama-2-chat on a mixture of this dataset and the ShareGPT dataset at an AgentInstruct sampling ratio of 0.2 [1]. No adoption by other named models or recipes was found beyond this origin release.

**Introduced by**: [1] (Zeng et al., "AgentTuning").

## Shape

Rows and splits, read from the Hub API's cached `dataset_info` (the `/info` and `/size` datasets-server endpoints, reachable under the renamed id `zai-org/AgentInstruct` even though the shortlist's `THUDM/AgentInstruct` viewer calls 404) [3][4]:

| split | rows | bytes (decoded) |
| --- | --- | --- |
| `alfworld` | 336 | 1,223,363 |
| `webshop` | 351 | 1,602,648 |
| `mind2web` | 122 | 159,590 |
| `kg` | 324 | 2,960,010 |
| `os` | 195 | 660,245 |
| `db` | 538 | 1,436,655 |
| total | 1,866 | 8,042,511 |

One config, `default`, with two columns [3]:

| column | dtype |
| --- | --- |
| `conversations` | list of `{from: string, loss: bool, value: string}` |
| `id` | string |

On-disk size is 1,255,385 bytes as parquet (equal to the repository's declared original-file size, since the repo ships parquet directly) [4]. No source states a token-count statistic for the dataset. The README does state average filtered-trajectory turn counts per task: ALFWorld 13.52, WebShop 3.68, Mind2Web 1.00, Knowledge Graph 6.04, Operating System 3.85, Database 2.06, all-task average 5.24 [1][2].

## Quality

- Filtering signal: trajectories were kept only where GPT-4's run scored a reward `r=1` (fully correct) on every task except Mind2Web, which used a lower threshold of `r ≥ 2/3` because the stricter bar left too few trajectories for that task [1]. The paper's own ablation shows what that filtering buys: at 7B scale, a model trained on unfiltered trajectories scores 1.34 held-in / 0.47 held-out, versus 1.96 held-in / 0.65 held-out for the filtered version used to build this dataset [1].
- Contamination check: the paper ran a token-based check (10-gram match, allowing up to 4 mismatched tokens, following the method used for Llama 2) between the training data and the held-in tasks' test sets, defining a test example "dirty" if over 80% of its tokens are also found in the training data [1]. Its Table 7 reports this per task: ALFWorld 12.00% contamination rate, 6 of 50 test examples dirty; Database 4.72%, 0 of 300 dirty; Knowledge Graph 0.34%, 0 of 150 dirty; Mind2Web 3.40%, 0 of 177 dirty; Operating System 15.95%, 0 of 144 dirty; WebShop 47.18%, 1 of 200 dirty; total across all six 15.58%, 7 of 1,021 dirty [1]. The paper's own summary calls this "no evidence of data leakage" and notes that Database and Operating System - built via task derivation and self-instruct rather than sampled from a benchmark's train split - carry the higher raw contamination rate yet show zero dirty examples; it does not make the same claim for ALFWorld or WebShop, whose test sets are the two with actual dirty examples by its own >80% threshold [1].
- Per-turn training mask: every `human` turn in the served rows carries `loss: null`; `gpt` turns carry `loss: true` or `loss: false`. Reading the first five rows of the `os` split, each begins with a run of `loss: false` `gpt` turns (a fixed few-shot demonstration exchange) followed by a run of `loss: true` turns (the trajectory to actually train on) - e.g. row `os_0`'s nine `gpt` turns run `[False, False, False, True, True, True, True, True, True]` [5]. A collator that trains on every `gpt` turn regardless of `loss` will train on the prepended demonstration exchange as if it were the target trajectory.
- No source states an annotator-agreement figure or a duplicate-row rate for this dataset.

## Load it

The Hub redirects the original `THUDM/AgentInstruct` id to the current `zai-org/AgentInstruct`; both resolve to the same repository and the same commit (`sha` `e252cf78ced8a0ea5f62cfd591784cdbbddbac8a`, last modified 2023-10-23) [2]. There is no `train`/`test` split - pass the task name as `split`:

```python
import datasets

REV = "e252cf78ced8a0ea5f62cfd591784cdbbddbac8a"  # main at the check date
alfworld = datasets.load_dataset("zai-org/AgentInstruct", revision=REV, split="alfworld")  # 336 rows
webshop = datasets.load_dataset("zai-org/AgentInstruct", revision=REV, split="webshop")    # 351 rows
mind2web = datasets.load_dataset("zai-org/AgentInstruct", revision=REV, split="mind2web")  # 122 rows
kg = datasets.load_dataset("zai-org/AgentInstruct", revision=REV, split="kg")              # 324 rows
os_split = datasets.load_dataset("zai-org/AgentInstruct", revision=REV, split="os")        # 195 rows
db = datasets.load_dataset("zai-org/AgentInstruct", revision=REV, split="db")              # 538 rows
```

**Trap**: `datasets-server`'s `/info` and `/size` endpoints 404 under the old id `THUDM/AgentInstruct` (matching the shortlist row's `info_status: 404`) but succeed under the renamed `zai-org/AgentInstruct` [3][4] - the viewer failure is a lookup-key mismatch, not evidence the parquet files are broken; `load_dataset` itself works under either id because the Hub resolves the redirect. A second trap: every `gpt` turn is not training signal - filter to `loss == true` per the Quality note above, or a collator will train on the prepended few-shot demonstration turns too.

## Neighbors

None found from the same builder: THUDM/`zai-org`'s other 31 listed dataset repositories are unrelated benchmarks and SFT sets (e.g. `LongAlign-10k`, `CogVLM-SFT-311K`), none of them a re-release, cleaning, or successor of this one [6]. Two unrelated datasets share the literal name "AgentInstruct" and are easy to confuse with this one: `microsoft/orca-agentinstruct-1M-v1`, a ~1M-row synthetic instruction corpus from a different paper and a different builder [7], and `WangResearchLab/AgentInstruct`, the dataset for "Agent Instructs Large Language Models to be General Zero-Shot Reasoners" (arXiv:2310.03710), also unrelated in content and builder [8]. Neither is a variant of this release; do not substitute one for the other on name alone.

## A row

All six splits share the same two-column schema (`conversations`, `id`), so one row covers the shape served everywhere. From `config="default"`, `split="mind2web"`, row index 0 (datasets-server `/first-rows` under the renamed id) [5]:

```json
{
  "conversations": [
    {
      "from": "human",
      "loss": null,
      "value": "'''\n<html> <div> <header banner> <ul menu> <li menuitem> <a id=0 visit the gift cards page> Gift Cards </a> </li> </ul> </header> <footer contentinfo> <ul navigation> <a id=1> Gift Cards </a> </ul> </footer> </div> </html>\n'''\n\nBased on the HTML webpage above, try to complete the following task:\nTask: Check the balance for a gift card with the number 1000000000000000 and pin 1222\nPrevious actions:\nNone\nWhat should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):\n\nA. None of the above\nB. <a id=0 visit the gift cards page> Gift Cards </a>\nC. <a id=1> Gift Cards </a>"
    },
    {
      "from": "gpt",
      "loss": true,
      "value": "Thought: The previous actions have already completed the task of checking the balance for the gift card. No further action is necessary.\nAnswer: A. None of the above."
    }
  ],
  "id": "mind2web_0"
}
```

Turn count varies sharply by task: Mind2Web rows are single human/gpt exchanges like this one, while ALFWorld and Knowledge Graph rows run to dozens of turns per the average-turn figures in Shape [1][2].

## Where it came from

Built by THUDM (Tsinghua KEG group). For four of the six tasks - ALFWorld, WebShop, Knowledge Graph, and part of Database - instructions were drawn from the training split of an existing benchmark and GPT-4 was run as the interacting agent to produce a trajectory; for the Operating System task and the rest of Database, which have no existing training split, instructions were instead constructed with a task-derivation-and-self-instruct method: GPT-4 first proposes a task, reference solution and evaluation script, and a second GPT-4 instance then attempts to solve it [1]. Mind2Web instructions come from that benchmark's training split, evaluated with a teacher-forcing decomposition into single-step choices rather than full free-form interaction [1]. Every trajectory was then filtered by its automatic reward score, keeping only `r=1` (or `r ≥ 2/3` for Mind2Web) [1]. The resulting 1,866 trajectories were used, mixed with the ShareGPT dataset at a 0.2 sampling ratio, to fine-tune Llama-2-chat into the AgentLM-7B/13B/70B models released alongside this dataset [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-12; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Zeng et al., "AgentTuning: Enabling Generalized Agent Abilities for LLMs", 2023. https://arxiv.org/abs/2310.12823 - task construction, filtering thresholds, contamination analysis, AgentLM training mixture, ablation table. Read via the ar5iv HTML rendering, https://ar5iv.labs.arxiv.org/html/2310.12823 . Fetched 2026-08-12.

[2] THUDM/AgentInstruct dataset card (README) and Hub API record. https://huggingface.co/datasets/THUDM/AgentInstruct - resolves to https://huggingface.co/datasets/zai-org/AgentInstruct after the organization rename; README fetched at the shortlist's pinned commit via `/raw/e252cf78ced8a0ea5f62cfd591784cdbbddbac8a/README.md`, API record via `/api/datasets/THUDM/AgentInstruct?full=true`. Fetched 2026-08-12.

[3] datasets-server info endpoint, under the renamed id (the old id 404s, matching the shortlist row's `info_status`). https://datasets-server.huggingface.co/info?dataset=zai-org%2FAgentInstruct Fetched 2026-08-12.

[4] datasets-server size endpoint, under the renamed id. https://datasets-server.huggingface.co/size?dataset=zai-org%2FAgentInstruct Fetched 2026-08-12.

[5] datasets-server first-rows endpoint, one call per split, under the renamed id: `os`, `db`, `alfworld`, `webshop`, `kg`, `mind2web`. https://datasets-server.huggingface.co/first-rows?dataset=zai-org%2FAgentInstruct&config=default&split=<split> Fetched 2026-08-12.

[6] Hugging Face Hub API listing of datasets by the `zai-org` account (the renamed THUDM organization). https://huggingface.co/api/datasets?author=zai-org&limit=100 Fetched 2026-08-12.

[7] microsoft/orca-agentinstruct-1M-v1 dataset listing, checked only to confirm it is an unrelated, differently-built dataset sharing this one's name. https://huggingface.co/api/datasets?search=agentinstruct&limit=100 Fetched 2026-08-12.

[8] WangResearchLab/AgentInstruct dataset card (README), checked only to confirm it is a third, unrelated dataset sharing this one's name. https://huggingface.co/datasets/WangResearchLab/AgentInstruct/raw/main/README.md Fetched 2026-08-12.

[9] The corpus screening row for `THUDM/AgentInstruct`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-12.

## Appendix: screening record

### Screening verdict

Usable as reasoning-trace SFT data for agent tasks, with no licence stated and no train/test partition within this file. The dataset's own contents support this: it is GPT-4-generated ReAct trajectories, reward-filtered, over six agent tasks, exactly as the screening row's note describes; the paper's own contamination table (cited above) is what lets the card say nothing needs holding out from this file itself, since the paper explains the small dirty-example counts it reports for ALFWorld (near-duplicate task phrasing) and WebShop (a shared price-constraint phrase across unrelated tasks) rather than treating either as leakage.

### The screening row

The row's own note [9]: "1,866 ReAct-style agent interaction trajectories across six real-world tasks (ALFWorld, WebShop, Mind2Web, knowledge graph, operating system and database), built by task derivation and self-instruct and kept only where GPT-4's trajectory passed a strict reward check; the viewer returns 404 because the repo uses an old loader layout." The row carries no flag.
