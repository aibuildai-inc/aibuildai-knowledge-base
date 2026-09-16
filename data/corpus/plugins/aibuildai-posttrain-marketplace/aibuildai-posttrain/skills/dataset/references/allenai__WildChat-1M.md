# allenai/WildChat-1M

837,989 real user-ChatGPT conversations - multi-turn, multi-lingual, with per-turn moderation scores and coarse geo/demographic metadata - the 1M-scale, de-toxified release of the WildChat logs.

**allenai/WildChat-1M** is Allen Institute for AI's release of the WildChat corpus introduced in "WildChat: 1M ChatGPT Interaction Logs in the Wild" [1]: conversations collected by giving online users free access to OpenAI's GPT-3.5 and GPT-4 through a hosted chatbot, logging both the user turns and the model replies together with per-turn moderation scores, hashed IP, and coarsened location fields [2]. **This release has had every conversation flagged as toxic by the OpenAI Moderation API or Detoxify removed (as of the 2024-07-22 update) and every conversation flagged for PII or other sensitive content by a later audit removed (as of the 2024-10-17 update); the toxic-content version is a separate, gated repository, `allenai/WildChat-1M-Full`** [2]. It lives at https://huggingface.co/datasets/allenai/WildChat-1M .

**Use it for**: SFT on real, in-the-wild multi-turn chat, or as a source of naturalistic prompts for other post-training stages - not preference pairs, since each row has one response, not a chosen/rejected pair. Maps to a multi-turn chat SFT format: each row's `conversation` list is an ordered sequence of `role`/`content` turns (user, then assistant, repeating), so it reshapes directly into a chat-template message list; see the SFT method card. The de-toxified restriction above is the usage-shape fact to carry forward - if toxic content is required for the task, the full version needs separate gated access [2].

**Licence**: ODC-BY (`cardData.license` is `"odc-by"`; the Hub API's `gated` field is `false`) [3][4]. The one catch: the README states the licence was changed to ODC-BY on 2024-06-26 and applied retroactively to earlier downloads that had used the ImpACT licence, so a copy fetched before that date may carry stale licence metadata [2].

**Shape**: 837,989 rows, one split (`train`), one config (`default`), 14 columns [4][5].

**Hold out**: nothing found - no source or the screening note flags an evaluation-set overlap for this dataset [6].

**Origin**: built and released by the Allen Institute for AI (AI2); the assistant replies are from OpenAI's `gpt-3.5-turbo` and `gpt-4` model family (both `-0301`/`-0314` variants appear in the sampled rows), the user turns are from real online users of AI2's hosted chatbot [2][7]. Hub API at the check date: `downloads` 19,481, `downloadsAllTime` 295,073, `likes` 453 [3].

**Trained-on-by**: not stated - no source in hand names a specific model or training recipe that trained on this Hub release; none found.

**Introduced by**: [1] (Zhao et al.); a companion visualizer, WildVis, is described in a separate paper referenced from the same card [8].

## Shape

Rows served and columns (datasets-server `/size` and `/info`) [4][5]:

| split | rows |
| --- | --- |
| `train` | 837,989 |

14 top-level columns: `conversation_hash`, `model`, `timestamp`, `conversation` (list of per-turn structs), `turn`, `language`, `openai_moderation` (list of per-turn structs), `detoxify_moderation` (list of per-turn structs), `toxic`, `redacted`, `state`, `country`, `hashed_ip`, `header` [5]. The nested `conversation` struct carries `content`, `country`, `hashed_ip`, `header`, `language`, `redacted`, `role`, `state`, `timestamp`, `toxic`, `turn_identifier` per turn [5].

Sizes (datasets-server `/size`) [4]: 3,360,836,020 bytes as Parquet download, 6,659,534,616 bytes decoded in memory. No source states a token-count or sequence-length statistic for this release; the README gives no such figure [2].

The dataset's own card states 68 languages were detected across the corpus, and that 25.53% of conversations come from the GPT-4 chatbot with the rest from GPT-3.5 [2]. Of the first 23 rows served (`config="default"`, `split="train"`, offset 0), `model` values seen were `gpt-3.5-turbo-0301` and `gpt-4-0314`, `turn` ranged from 1 to 7, and every row had `toxic` equal to `false` [7].

## Quality

- The README states the data were de-identified with Microsoft Presidio plus hand-written rules, and that two content-removal passes were applied after initial release: on 2024-07-22, all conversations flagged toxic by the OpenAI Moderation API or Detoxify were removed, and on 2024-10-17, conversations flagged by an external audit ("Breaking News: Case Studies of Generative AI's Use in Journalism") for containing PII or sensitive information were removed [2]. Consistent with this, the `toxic` field read `false` on all 23 sampled rows [7].
- The README states that a small subset of conversations contain empty user inputs, which the hosted chatbot's UI did not block, and that this sometimes led the assistant to hallucinate a response with no prompt to respond to [2]. It gives no count for this 1M-row release; the sibling 650K-row `allenai/WildChat` card states the equivalent figure for its own smaller corpus as 12,405 of 652,139 conversations [9], but that number describes a different, smaller release and is not read onto this one.
- No source states a measured duplicate-rate or contamination figure for this dataset; none is invented here.

## Load it

```python
import datasets

REV = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"  # main at the check date
ds = datasets.load_dataset("allenai/WildChat-1M", revision=REV, split="train")  # 837,989 rows
```

**Trap**: `conversation_hash` is not a unique row key - the README states different conversations with identical content share the same hash, and that the per-turn `turn_identifier` is the unique identifier within a conversation [2]. Deduplicating or joining on `conversation_hash` alone will silently merge distinct conversations.

## Neighbors

- `allenai/WildChat` - the earlier 650K-conversation release (529,428 rows served live) [10] with a smaller per-turn schema (no `country`, `hashed_ip`, or `header` inside `conversation`) [11]; its own card points readers to `allenai/WildChat-4.8M` as the newer, larger version and does not mention this 1M release as a successor [11].
- `allenai/WildChat-4.8M` - a larger, ungated release with 3,199,860 rows served live and the same 14-column schema as this one [10][12]; prefer it over this release when scale matters more than working with the well-established 1M-row baseline, since its extra rows have not been separately audited in any source read for this card.
- `allenai/WildChat-1M-Full` - the toxic-content counterpart to this release, same row scope but without the toxicity removal pass; the Hub API marks it gated with manual approval required, and the README states that access requires approval, with justification for why the toxic data is needed [2][13]. Use it only if the training task specifically needs the removed toxic conversations; it duplicates this release's non-toxic rows, so mixing both into one training run double-counts data.

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [7], with both turns' `content` truncated:

```json
{
  "conversation_hash": "c9ec5b440fbdd2a269333dd241f32f64",
  "model": "gpt-4-0314",
  "timestamp": "2023-04-09T00:02:53",
  "conversation": [
    {
      "content": "Hey there! Are you familiar with reality shifting? So, I’m refining a foolproof method for reality shifting and want to pick a destination. [...]",
      "country": "United States",
      "hashed_ip": "22fd87ba9b98f3d379b23c7b52961f2d4a8505127e58b3104b227a8ca1ce7260",
      "header": {"accept-language": "en-US,en;q=0.9,es;q=0.8", "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"},
      "language": "English",
      "redacted": false,
      "role": "user",
      "state": "Texas",
      "timestamp": null,
      "toxic": false,
      "turn_identifier": 101001
    },
    {
      "content": "Hey there! I'm more than happy to help you plan your reality-shifting adventure, and I've got just the destination in mind for you [...]",
      "country": null,
      "hashed_ip": null,
      "header": null,
      "language": "English",
      "redacted": false,
      "role": "assistant",
      "state": null,
      "timestamp": "2023-04-09T00:02:53",
      "toxic": false,
      "turn_identifier": 101001
    }
  ],
  "turn": 1,
  "language": "English",
  "toxic": false,
  "redacted": false,
  "state": "Texas",
  "country": "United States",
  "hashed_ip": "22fd87ba9b98f3d379b23c7b52961f2d4a8505127e58b3104b227a8ca1ce7260",
  "header": {"accept-language": "en-US,en;q=0.9,es;q=0.8", "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}
}
```

`openai_moderation` and `detoxify_moderation` are omitted above for brevity; each is a list of one struct per conversation turn, with the same category/score fields shown in the Shape section.

## Where it came from

Built and released by AI2. The README states the underlying WildChat logs were collected by offering online users free access to OpenAI's GPT-3.5 and GPT-4 through a hosted chatbot, capturing both the conversation content and request-time metadata (hashed IP, request headers, and GeoIP-derived state/country) [2]. The user turns are unscripted, real user prompts; the assistant turns are generations from the named OpenAI models, not from any model AI2 trained [2]. The origin paper describes the collection and de-identification pipeline in full [1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision.

[1] Zhao et al., "WildChat: 1M ChatGPT Interaction Logs in the Wild," ICLR 2024. https://arxiv.org/abs/2405.01470 - the origin paper; current title read from the live abs page. Fetched 2026-08-11.

[2] allenai/WildChat-1M dataset card (README). https://huggingface.co/datasets/allenai/WildChat-1M/raw/main/README.md - summary, GPT-4 share, language count, update log, de-identification method, empty-input note, licence history, full-version pointer. Fetched 2026-08-11.

[3] Hugging Face Hub API record for allenai/WildChat-1M. https://huggingface.co/api/datasets/allenai/WildChat-1M?full=true - `sha`, `gated`, `downloads`, `lastModified`; `downloadsAllTime` read through the same endpoint's `expand[]=downloadsAllTime` variant. Fetched 2026-08-11.

[4] datasets-server size endpoint, pinned to the revision above. https://datasets-server.huggingface.co/size?dataset=allenai%2FWildChat-1M&revision=7d6490e462285cf85d91eabea0f9a954fbddcd1f - row counts and byte sizes match the unpinned call to the same endpoint, confirmed by direct comparison. Fetched 2026-08-11.

[5] datasets-server info endpoint, pinned to the revision above. https://datasets-server.huggingface.co/info?dataset=allenai%2FWildChat-1M&revision=7d6490e462285cf85d91eabea0f9a954fbddcd1f Fetched 2026-08-11.

[6] The corpus screening row for `allenai/WildChat-1M`, supplied with this card's request - checked for a `flag` field or a hold-out note; it carries neither. Checked 2026-08-11.

[7] datasets-server first-rows endpoint, pinned to the revision above. https://datasets-server.huggingface.co/first-rows?dataset=allenai%2FWildChat-1M&config=default&split=train&revision=7d6490e462285cf85d91eabea0f9a954fbddcd1f - 23 rows served at offset 0, read for the sample row, `model`/`turn`/`toxic`/`language` distribution; the same rows, in the same order, were confirmed by an unpinned call to the same endpoint. Fetched 2026-08-11.

[8] Deng et al., "WildVis: Open Source Visualizer for Million-Scale Chat Logs in the Wild," EMNLP 2024 Demo. https://arxiv.org/abs/2409.03753 - referenced from the WildChat-1M card as the companion visualizer paper; not itself the origin paper for this dataset. Cited through the WildChat-1M README's link, not independently opened. Fetched 2026-08-11 (README only).

[9] allenai/WildChat dataset card (README). https://huggingface.co/datasets/allenai/WildChat/raw/main/README.md - the 650K-row sibling's empty-input count and pointer to `allenai/WildChat-4.8M`. Fetched 2026-08-11.

[10] datasets-server size endpoint, one call per neighbor: `allenai/WildChat`, `allenai/WildChat-4.8M`. https://datasets-server.huggingface.co/size?dataset=<id> - this endpoint takes no revision parameter, so these counts are live, not pinned. Fetched 2026-08-11.

[11] allenai/WildChat `/info` endpoint (schema comparison against this release). https://datasets-server.huggingface.co/info?dataset=allenai%2FWildChat - implied by the README's `dataset_info.features` block, read from the same fetched README file as [9]. Fetched 2026-08-11.

[12] Hugging Face Hub API record for allenai/WildChat-4.8M. https://huggingface.co/api/datasets/allenai/WildChat-4.8M?full=true - licence, gate status. Fetched 2026-08-11.

[13] Hugging Face Hub API record for allenai/WildChat-1M-Full. https://huggingface.co/api/datasets/allenai/WildChat-1M-Full?full=true - `gated: "manual"`. The datasets-server `/size` endpoint for this repo returned an authentication-required error, confirming it is not openly accessible. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT source data for real multi-turn chat, with the toxic-content caveat already stated above: this release has had toxic and PII-flagged conversations removed, and the toxic-content version sits in a separate gated repository [2]. The screening row's note describes it as real user prompts paired with ChatGPT/GPT-3.5/GPT-4 replies, and identifies it as the 1M-scale release of the same underlying logs [6].

### The screening row

The row's own note [6]: "Real user prompts with ChatGPT/GPT-3.5/GPT-4 replies; the 1M-scale release of the same logs." The row carries no flag.
