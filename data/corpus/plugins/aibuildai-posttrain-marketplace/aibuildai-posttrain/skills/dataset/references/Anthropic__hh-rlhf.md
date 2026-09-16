# Anthropic/hh-rlhf

169,352 human-preference pairs - whole assistant dialogues, single- or multi-turn, each ending in a chosen and a rejected reply - over helpfulness and harmlessness, the original Anthropic HH-RLHF release.

**Anthropic/hh-rlhf** is Anthropic's human preference data behind "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback" [1]: crowdworkers held open-ended conversations with Anthropic assistant models and picked one of two candidate replies at the final turn, yielding chosen/rejected transcript pairs that feed preference (reward) model training for RLHF [2]. The same repository carries a separate `red-team-attempts` file from "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned" [3]; that file sits outside the served config, and the card says it is "*not* meant for fine-tuning or preference modeling" [2]. It lives at https://huggingface.co/datasets/Anthropic/hh-rlhf .

**Preference/reward-model data only, never SFT: the card says the pairs "are meant to train preference (or reward) models for subsequent RLHF training", that they "are *not* meant for supervised training of dialogue agents", and that "Training dialogue agents on these data is likely to lead to harmful models and this shold be avoided [sic]" [2]. Hold out the `test` split (8,552 rows) [4].**

**Use it for**: preference-pair training - a reward model, or a preference-optimization method that consumes chosen/rejected pairs - never SFT on the chosen side [2]. The rows are the implicit-prompt preference format: `chosen` and `rejected` both contain the shared dialogue prefix and differ only in the final assistant turn. A reward trainer consumes that directly (trl's RewardTrainer expects "Preference (implicit prompt recommended)"), while DPO-style training wants the explicit-prompt form - extract the shared prompt first (trl ships `extract_prompt()` for exactly this conversion) [5]. See the DPO method card.

**Licence**: MIT (`cardData.license` is `"mit"`; repo tags include `license:mit`), ungated (`"gated": false`, `"private": false`) [6]. The one catch: the not-for-SFT scope above is the card's stated intended use and its safety warning, not a licence term - no source says the MIT grant carries it [2][6].

**Shape**: 169,352 rows in one config (`default`), split `train` 160,800 / `test` 8,552, two string columns (`chosen`, `rejected`) [7][8].

**Hold out**: `test` (8,552 rows); train on `train` (160,800). The screening row's note says so in those words - "train/test, so hold out `test`" [4].

**Origin**: built and released by Anthropic; the two candidate replies are Anthropic-model generations, the preference label is a human crowdworker judgment [2]. Hub API at the check date: `downloads` 33,956, `downloadsAllTime` 1,984,499, `likes` 1,945 [6].

**Trained-on-by**: the origin paper's own preference models, and the RLHF assistants trained against them - this release is that paper's preference data [1][2]. Llama 2's reward models: "We combine our newly collected data with existing open-source preference datasets to form a larger training dataset", naming "Anthropic Helpful and Harmless" among them; its Table 6 lists Anthropic Helpful at 122,387 and Anthropic Harmless at 43,966 comparisons [9]. The DPO paper's single-turn dialogue experiments: "we use the Anthropic Helpful and Harmless dialogue dataset, containing 170k dialogues between a human and an automated assistant" [10].

**Introduced by**: [1] (Bai et al.); the `red-team-attempts` file by [3] (Ganguli et al.).

## Shape

Rows served and splits (datasets-server `/size`) [7]:

| split | rows |
| --- | --- |
| `train` | 160,800 |
| `test` | 8,552 |
| total | 169,352 |

One config, `default`, with two columns (datasets-server `/info`) [8]:

| column | dtype |
| --- | --- |
| `chosen` | string |
| `rejected` | string |

The `default` config is the union of four of the repository's five subdirectories - `harmless-base`, `helpful-base`, `helpful-online`, `helpful-rejection-sampled` - which share this schema; the README's own load instructions present the no-`data_dir` load as loading "all helpfulness/harmless subsets" [2]. The fifth subdirectory, `red-team-attempts`, has a different, transcript-shaped schema and does not appear in `/info`'s config list [2][8].

Sizes (datasets-server `/size`) [7]: 181,576,518 bytes as Parquet, 325,133,436 bytes decoded in memory. The shortlist row's own `bytes` field, 79,254,614, is the original gzipped-JSON download size, not stated by this endpoint; it matches the sum of the four preference-modeling subdirectories' `.jsonl.gz` files (`harmless-base`, `helpful-base`, `helpful-online`, `helpful-rejection-sampled`) in the repository's file tree, excluding `red-team-attempts/red_team_attempts.jsonl.gz` and the top-level `README.md`/`.gitattributes` [11]. No source states sequence-length or token figures for the whole 169,352-row release. Llama 2's Table 6 does state them for the two Anthropic subsets it adopted, whose comparison counts do not add up to this release: Anthropic Helpful 122,387 comparisons, 3.0 average turns per dialogue, 251.5 average tokens per example, 17.7 in the prompt, 88.4 in the response; Anthropic Harmless 43,966, 3.0, 152.5, 15.7, 46.4 [9].

## Quality

- The preference label on each row (which reply is `chosen`) is a human crowdworker judgment, not a model judgment [2][4].
- No source states a measured contamination rate, duplicate rate, or annotator-agreement figure for this dataset; none is invented here.
- The README's only stated quality caveat is the content-safety disclaimer: the data "contain content that may be offensive or upsetting", covering "discriminatory language and discussions of abuse, violence, self-harm, exploitation, and other potentially upsetting subject matter", and it repeats that the data "are *not* intended for training dialogue agents as this will likely lead to harmful model behavior" [2]. It states the data "are intended for research purposes, especially research that can make models *less* harmful" [2].

## Load it

Train on `train`, hold out `test`, pass no `data_dir`, and pin the revision this card's numbers were read at (the Hub API's `sha` for `main` at the check date; the repo was last modified 2023-05-26) [6]:

```python
import datasets

REV = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"  # main at the check date
train = datasets.load_dataset("Anthropic/hh-rlhf", revision=REV, split="train")  # 160,800 rows
test = datasets.load_dataset("Anthropic/hh-rlhf", revision=REV, split="test")    # 8,552 rows - hold out
```

**Trap**: with no `data_dir` argument, `load_dataset` silently merges all four preference-modeling subsets (`harmless-base`, `helpful-base`, `helpful-online`, `helpful-rejection-sampled`) into the `train`/`test` splits above - that merge is what "169,352 rows" and "one config" mean [2][8]. Passing `data_dir="red-team-attempts"` loads an entirely different, transcript-shaped schema (`transcript`, `min_harmlessness_score_transcript`, `rating`, `task_description`, ...) that the card says is "*not* meant for fine-tuning or preference modeling" at all [2]; do not mix it into training data by treating it as another `hh-rlhf` shard.

The `revision=` argument shown above genuinely pins what `load_dataset` fetches, since it names a real git ref on the repo. It does not, however, pin the row counts, byte sizes, or the sampled row quoted throughout this card: those came from the datasets-server `/size`, `/info`, and `/first-rows` endpoints [7][8][12], which accept no revision parameter and always answer against the repository's current state, the same way the neighbor lookups do [13]. Those figures are live reads taken at the check date, not reads pinned to the `sha` above.

## Neighbors

This release has a family of re-releases. Prefer this original when you want the complete pair set at its own splits; every row count below was read live at the check date [13]. Reach for a neighbor only when its format or language is what you need.

- `KHuss/hh-rlhf-formatted` - the same 169,352 pairs at the same 160,800/8,552 split, reformatted with three extra columns (five columns total against two here) [13].
- `Dahoas/full-hh-rlhf` - 124,503 rows reformatted into prompt, response, chosen and rejected columns, split 112,052/12,451; a subset of this release [13].
- `trl-lib/hh-rlhf-helpful-base` - the `helpful-base` part only (43,835 train / 2,354 test, three columns), reformatted for TRL's chat message format [13].
- `PKU-Alignment/processed-hh-rlhf` - this release reshaped into three columns, and 1,160 rows short of it: 159,700 train and 8,492 test against 160,800 and 8,552 here. Its own card says it "processes hh-rlhf into an easy-to-use conversational and human-preference form" and does not account for the missing rows, so treat it as a processed near-copy, not the same pair set [13][14].
- `Unified-Language-Model-Alignment/Anthropic_HH_Golden` - a two-column dataset extending the harmless part of this release, with every chosen response replaced by a GPT-4 rewrite while the rejected side is unchanged (44,849 pairs, split 42,537/2,312), so its preference signal differs from this release on the chosen side alone [13][14].
- Japanese renderings: `kunishou/hh-rlhf-49k-ja-single-turn` (47,763 rows, machine-translated into Japanese, single train split) and `llm-jp/hh-rlhf-12k-ja` (12,000 rows, four columns, single train split) [13].
- The original authors also host a GitHub mirror at https://github.com/anthropics/hh-rlhf [2].

## A row

The repository serves one config and one schema, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [12], with the long cuss-word list truncated:

```json
{
  "chosen": "\n\nHuman: What are some cuss words in english?\n\nAssistant: Here’s an incomplete list.\n\nAss, dick, bugger, crap, fuck, shit, bitch, turd, shithead, [...] mama’s boy, faggot, pervert, queer, scumbag, bitch,\n\nHuman: What's your favorite one?\n\nAssistant: I haven't even thought about it.",
  "rejected": "\n\nHuman: What are some cuss words in english?\n\nAssistant: Here’s an incomplete list.\n\nAss, dick, bugger, crap, fuck, shit, bitch, turd, shithead, [...] mama’s boy, faggot, pervert, queer, scumbag, bitch,\n\nHuman: What's your favorite one?\n\nAssistant: Ass."
}
```

Both fields hold the whole `\n\nHuman: ... \n\nAssistant: ...` transcript up to and including the final turn; `chosen` and `rejected` differ only in that final assistant turn. Dialogue length varies: of the first ten served rows, three carry a single Human turn and the rest carry two to four [12].

## Where it came from

Built and released by Anthropic. The repository bundles two kinds of data [2]:

1. Human preference data over helpfulness and harmlessness, from the paper "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback" [1] - the part served by the `default` config. For **helpfulness**, the replies come in three tranches: from Anthropic's "base models (context-distilled 52B language models)", "via rejection sampling (mostly with best-of-16 sampling) against an early preference model", and "a dataset sampled during our iterated 'online' process" [2]. For **harmlessness**, "the data are only collected for our base models, but otherwise formatted in the same way" [2]. Crowdworker collection details are in section 2 and appendix D of the paper [2].
2. Human-generated and annotated red-teaming dialogues, from "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned" [3] - the separate, not-for-training `red-team-attempts` file. These transcripts "are derived from the harmlessness preference modeling data described above, where only the chosen response is incorporated into the overall transcript", and are "annotated with human and automated measurements of how harmful the overall dialogues are" [2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback", 2022. https://arxiv.org/abs/2204.05862 - the origin paper; current title read from the live abs page. Fetched 2026-08-11.

[2] Anthropic/hh-rlhf dataset card (README). https://huggingface.co/datasets/Anthropic/hh-rlhf/raw/main/README.md - usage scope, tranche descriptions, red-team file description, load instructions, GitHub mirror. Fetched 2026-08-11.

[3] Ganguli et al., "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned", 2022. https://arxiv.org/abs/2209.07858 - the `red-team-attempts` paper. The README links it only as an anthropic.com PDF; this is its arXiv record, current title read from the live abs page. Fetched 2026-08-11.

[4] The corpus screening row for `Anthropic/hh-rlhf`, supplied with this card's request - its `note`, read back in the row's own words in the appendix. Checked 2026-08-11.

[5] trl dataset formats documentation. https://huggingface.co/docs/trl/main/en/dataset_formats - implicit-prompt preference type, `extract_prompt()`, per-trainer expected types. A `main` build, unpinned and mutable. Fetched 2026-08-11.

[6] Hugging Face Hub API record for Anthropic/hh-rlhf. https://huggingface.co/api/datasets/Anthropic/hh-rlhf?full=true - licence, gate, `sha`, `downloads`, `likes`, last-modified date; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[7] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=Anthropic%2Fhh-rlhf - confirmed to accept no revision parameter by comparing this response against the same call with a bogus `revision=` value, which returned an identical body. Fetched 2026-08-11.

[8] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=Anthropic%2Fhh-rlhf - confirmed to accept no revision parameter the same way as [7]. Fetched 2026-08-11.

[9] Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023. https://arxiv.org/abs/2307.09288 - reward-model data mix naming Anthropic Helpful and Harmless; Table 6 comparison counts. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2307.09288). Fetched 2026-08-11.

[10] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - single-turn dialogue experiments on Anthropic HH. Read as the arXiv HTML full text via ar5iv (https://ar5iv.labs.arxiv.org/html/2305.18290). Fetched 2026-08-11.

[11] Hugging Face Hub repository tree listing for Anthropic/hh-rlhf. https://huggingface.co/api/datasets/Anthropic/hh-rlhf/tree/main?recursive=true - per-file byte sizes of the repository's `.jsonl.gz` files, used to confirm the shortlist row's `bytes` figure. Fetched 2026-08-11.

[12] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=Anthropic%2Fhh-rlhf&config=default&split=train - like [7] and [8], this endpoint takes no revision parameter either. Fetched 2026-08-11.

[13] datasets-server size endpoint, one call per neighbor, for every neighbor row count above: `KHuss/hh-rlhf-formatted`, `Dahoas/full-hh-rlhf`, `trl-lib/hh-rlhf-helpful-base`, `PKU-Alignment/processed-hh-rlhf`, `Unified-Language-Model-Alignment/Anthropic_HH_Golden`, `kunishou/hh-rlhf-49k-ja-single-turn`, and `llm-jp/hh-rlhf-12k-ja`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[14] Neighbor dataset cards (READMEs), read for what each re-release says it did: `PKU-Alignment/processed-hh-rlhf` and `Unified-Language-Model-Alignment/Anthropic_HH_Golden`. https://huggingface.co/datasets/<id>/raw/main/README.md Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable, and only as preference/reward-model data: train on `train`, hold out `test`. Two facts decide it, and both are already established above - the dataset's own card restricts the pairs to preference (reward) model training and warns against dialogue-agent SFT (quoted in the opening paragraph) [2], and the screening row's note says to hold out `test` [4].

### The screening row

The row's own note [4]: "Human crowdworker choices between two Anthropic-model replies, for helpfulness and harmlessness; train/test, so hold out `test`." The row carries no flag.
