# AarushSah/lmsys-chat-1m

1,000,000 real-world, multi-turn conversations between human users and 25 chat models, an ungated re-upload of the LMSYS-Chat-1M release.

**AarushSah/lmsys-chat-1m** carries the full contents of `lmsys/lmsys-chat-1m`, introduced by "LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset" [1]: one million conversations collected from 210K unique IP addresses on the Vicuna demo and Chatbot Arena website between April and August 2023, each row a conversation ID, model name, an OpenAI-API-style message list, a detected language, and per-message OpenAI moderation scores [2]. **The origin paper states the release "may contain questions from MMLU and MT-Bench", so training data drawn from it may contain contaminated samples for those two benchmarks [1]; the README's bundled "LMSYS-Chat-1M Dataset License Agreement" additionally forbids redistributing the dataset to any third party and lets the original authors demand deletion of all copies on request [2].** It lives at https://huggingface.co/datasets/AarushSah/lmsys-chat-1m .

**Use it for**: SFT chat data. Each row's `conversation` field is a list of `{content, role}` turns (`user`/`assistant`), the same shape an SFT chat method card expects; the raw `model` replies are unfiltered LLM output, so a scored MMLU or MT-Bench run needs decontamination against the overlap named above first. See the SFT method card.

**Licence**: no SPDX id in `cardData` - the repository instead reproduces the custom "LMSYS-Chat-1M Dataset License Agreement" in the README body [2]; the one catch is that this agreement forbids redistributing the dataset to third parties and grants the original authors a standing right to demand deletion of all copies [2].

**Shape**: 1,000,000 rows, one config (`default`), one split (`train`) [3][4].

**Hold out**: no benchmark eval split ships with this dataset, but the origin paper's own caveat is a contamination flag, not a split to withhold - decontaminate against MMLU and MT-Bench before scoring on either, per the restriction above [1].

**Origin**: prompts are from human users of the Vicuna demo and Chatbot Arena website; replies are the outputs of 25 chat models served on that site [2]. This mirror repository shows `downloads` 264, `downloadsAllTime` 8,574, `likes` 1 as of the check date [5][6]; it is ungated (`"gated": false`), while the source repository `lmsys/lmsys-chat-1m` is gated (`"gated": "auto"`) [5][7].

**Trained-on-by**: the origin paper's own experiment: fine-tuning Llama2-7B on a 45K-conversation "HighQuality" subset (OpenAI and Anthropic model conversations) produced a model scoring close to Vicuna-7B, while fine-tuning on a 39K-conversation "Upvote" subset (open-model conversations selected by user vote) scored markedly lower [1]. No source found documents a named production model or public recipe trained on this dataset beyond that in-paper experiment.

**Introduced by**: [1] (Zheng et al.), no separate introducing blog found.

## Shape

Rows served and splits (datasets-server `/size`) [3]:

| split | rows |
| --- | --- |
| `train` | 1,000,000 |

One config, `default`, with seven columns (datasets-server `/info`) [4]:

| column | dtype |
| --- | --- |
| `conversation_id` | string |
| `model` | string |
| `conversation` | list\<struct\<content: string, role: string\>\> |
| `turn` | int64 |
| `language` | string |
| `openai_moderation` | list\<struct\<categories: struct[11 bool], category_scores: struct[11 float64], flagged: bool\>\> |
| `redacted` | bool |

The origin paper states the release's own basic statistics: 25 models, 210,479 users, 154 languages, an average of 2.0 turns per sample, an average of 69.5 tokens per prompt, and an average of 214.5 tokens per response [2].

Sizes (datasets-server `/size`) [3]: 1,488,850,250 bytes of original Parquet download, the same 1,488,850,250 bytes as Parquet (this repository stores the data as Parquet natively), and 2,549,703,000 bytes decoded in memory.

## Quality

- The origin paper's own moderation-API tally over the whole release: "a non-negligible portion (5%) of the conversations have potentially harmful content" by the OpenAI moderation API's flagging, and the paper adds that the API's recall may be low, so it expects even more harmful content than that 5% figure captures [1]. Of 38 rows read live from `train` at offset 0, none had any message flagged by `openai_moderation` [8], consistent with 5% being a whole-release rate rather than a rate visible in the first few dozen rows.
- The README states the builders "made our best efforts to remove all conversations that contain personally identifiable information (PII)" via a partnership with the OpaquePrompts team, which replaces detected person names with placeholders like `NAME_1`; each row's `redacted` field marks whether that redaction ran, and the README adds that the process "may impact data quality and occasionally lead to incorrect redactions" [2]. Of the 38 rows read at offset 0, 7 have `redacted` set to true [8].
- The origin paper's own contamination caveat, quoted in the opening paragraph, is a quality signal about the training data itself, not an eval question: the release "may contain questions from MMLU and MT-Bench" [1].
- No source states a measured duplicate-conversation rate for the `train` split as served; the paper's jailbreak-conversation statistics separately note "there can be duplicate or similar jailbreak prompts across different models" in a per-model table that is not deduplicated, but that table is not a whole-dataset duplicate rate [1].

## Load it

```python
import datasets

REV = "c9fce24f9623e40750a061581bdd3a2d1243ff79"  # main at the check date
train = datasets.load_dataset("AarushSah/lmsys-chat-1m", revision=REV, split="train")  # 1,000,000 rows
```

**Trap**: this repository's `cardData` still carries the original `extra_gated_prompt`, `extra_gated_fields`, and `extra_gated_button_content` copied from `lmsys/lmsys-chat-1m`'s README, and the README still requires agreeing to the "LMSYS-Chat-1M Dataset License Agreement" in prose - but the Hub API's `gated` field for this specific repository reads `false`, so `load_dataset` here does not enforce that click-through gate the way it does on the source repository [2][5][7]. The license terms reproduced in the README (no third-party redistribution, deletion on request) still apply to whoever uses the data, gate or no gate [2].

## Neighbors

Every row count below was read live at the check date [9]. Prefer the original `lmsys/lmsys-chat-1m` when gated access is available; reach for a re-processed neighbor only for the specific transformation it names.

- `lmsys/lmsys-chat-1m` - the source repository this mirrors, gated (`"gated": "auto"`), same 1,000,000-row `train` split and column schema [5][7].
- `OpenLeecher/lmsys_chat_1m_clean` - 273,402 rows, described by its own card as deduplicated and reclassified from lmsys-chat-1m; a much smaller, cleaned subset rather than the full release [9][10].
- `openeurollm/lmsys-chat-1m-decontaminated` - 997,682 rows, its own card stating it is "a decontaminated version of lmsys/lmsys-chat-1m" with rows removed for overlap against named math/reasoning eval sets (MATH500, AIME24, AIME25, AMC23, JEEBench, GPQADiamond, and others); pick this over the row above the paper's contamination warning when the training target includes any of those benchmarks [9][11].
- `tokyotech-llm/lmsys-chat-1m-synth` - the Hub dataset viewer is disabled for this repository, so its row count and schema could not be checked live; its card describes a synthetic-response variant [9].

## A row

The repository serves one config and one split, so one row covers it. From `config="default"`, `split="train"`, `row_idx=0` (datasets-server `/first-rows`) [8], with the moderation struct's per-category detail omitted after the first entry and truncated with an ellipsis marker:

```json
{
  "conversation_id": "33f01939a744455c869cb234afca47f1",
  "model": "wizardlm-13b",
  "conversation": [
    {
      "content": "how can identity protection services help protect me against identity theft",
      "role": "user"
    },
    {
      "content": "Identity protection services can help protect you against identity theft in several ways: [...] It's still important to take steps to protect your own identity, such as being cautious with personal information and regularly monitoring your credit reports.",
      "role": "assistant"
    }
  ],
  "turn": 1,
  "language": "English",
  "openai_moderation": [
    {
      "categories": {"harassment": false, "harassment/threatening": false, "hate": false, "hate/threatening": false, "self-harm": false, "self-harm/instructions": false, "self-harm/intent": false, "sexual": false, "sexual/minors": false, "violence": false, "violence/graphic": false},
      "category_scores": {"harassment": 9.212334e-07, "...": "..."},
      "flagged": false
    },
    "..."
  ],
  "redacted": false
}
```

## Where it came from

Built and released by the LMSYS (Large Model Systems) team behind Chatbot Arena. Prompts come from human visitors to the Vicuna demo and Chatbot Arena website, who consented via that site's "Terms of use" section; replies come from the 25 chat models the site served over the collection window, April to August 2023 [1][2]. The paper reports names conversation moderation (via the OpenAI moderation API) and PII redaction (via a partnership with OpaquePrompts) as post-collection processing steps, both reflected in the `openai_moderation` and `redacted` columns [1][2].

## Sources

In-text markers [n] refer to this list, numbered by first appearance. Every source was read on the check date, 2026-08-11; that date covers every number, quote, and corpus row above. Hub repositories are mutable (a repo can be force-pushed), which is why Load it pins the revision. The datasets-server `/size`, `/info`, and `/first-rows` endpoints used below (sources [3], [4], [8]) accept a `revision` query parameter but ignore it: calling `/size` with a nonexistent revision value returns byte-for-byte the same response as calling it with no revision at all, confirmed by direct request on the check date. So the row count, column schema, byte sizes, and sampled row this card draws from those three endpoints are live reads against whatever `main` serves at fetch time, not reads pinned to the `c9fce24f9623e40750a061581bdd3a2d1243ff79` revision that Load it names; that revision only pins what `load_dataset(..., revision=...)` itself resolves.

[1] Zheng et al., "LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset", 2023. https://arxiv.org/abs/2309.11998 - the origin paper; moderation-flag rate, MMLU/MT-Bench contamination caveat, HighQuality/Upvote fine-tuning experiment, basic dataset statistics, collection method. Read via the ar5iv HTML rendering (https://ar5iv.labs.arxiv.org/html/2309.11998); title read from the live arXiv abs page. Fetched 2026-08-11.

[2] AarushSah/lmsys-chat-1m dataset card (README). https://huggingface.co/datasets/AarushSah/lmsys-chat-1m/raw/main/README.md - description, basic-statistics table, PII redaction process, license agreement text, gated-prompt front matter. Fetched 2026-08-11.

[3] datasets-server size endpoint. https://datasets-server.huggingface.co/size?dataset=AarushSah%2Flmsys-chat-1m Fetched 2026-08-11.

[4] datasets-server info endpoint. https://datasets-server.huggingface.co/info?dataset=AarushSah%2Flmsys-chat-1m Fetched 2026-08-11.

[5] Hugging Face Hub API record for AarushSah/lmsys-chat-1m. https://huggingface.co/api/datasets/AarushSah/lmsys-chat-1m?full=true - gate status, `sha`, `downloads`, `cardData` (no `license` key present), last-modified date. Fetched 2026-08-11.

[6] Hugging Face Hub API record for AarushSah/lmsys-chat-1m, all-time downloads variant. https://huggingface.co/api/datasets/AarushSah/lmsys-chat-1m?expand[]=downloadsAllTime Fetched 2026-08-11.

[7] Hugging Face Hub API record for lmsys/lmsys-chat-1m. https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m?full=true - `"gated": "auto"` on the source repository. Fetched 2026-08-11.

[8] datasets-server first-rows endpoint. https://datasets-server.huggingface.co/first-rows?dataset=AarushSah%2Flmsys-chat-1m&config=default&split=train - 38 rows read at offset 0, used for the sampled row, the redacted-flag count, and the flagged-moderation count in Quality. Fetched 2026-08-11.

[9] Hugging Face Hub dataset search API for "lmsys-chat-1m", plus a datasets-server size call per neighbor named above. https://huggingface.co/api/datasets?search=lmsys-chat-1m&limit=100 and https://datasets-server.huggingface.co/size?dataset=<id> - these endpoints take no revision parameter, so the neighbor row counts are live, not pinned. Fetched 2026-08-11.

[10] OpenLeecher/lmsys_chat_1m_clean dataset card (README), read via the Hub API description field. https://huggingface.co/api/datasets/OpenLeecher/lmsys_chat_1m_clean?full=true - describes deduplication and reclassification of lmsys-chat-1m. Fetched 2026-08-11.

[11] openeurollm/lmsys-chat-1m-decontaminated dataset card (README). https://huggingface.co/datasets/openeurollm/lmsys-chat-1m-decontaminated/raw/main/README.md - states it is a decontaminated version of lmsys/lmsys-chat-1m and lists the benchmarks checked against. Fetched 2026-08-11.

## Appendix: screening record

### Screening verdict

Usable as SFT chat data, with two use-time caveats already established above: decontaminate against MMLU and MT-Bench before a scored run on either, since the origin paper itself flags possible overlap [1], and treat the bundled license agreement's no-third-party-redistribution and deletion-on-request terms as binding regardless of this mirror's ungated status [2]. The screening row's note describes the release and this repository's gating gap in the same terms.

### The screening row

The row's own note [as supplied with the shortlist request]: "The LMSYS-Chat-1M release of 1,000,000 real user conversations with 25 chat models, one train split carrying conversation, model, turn, language and the OpenAI moderation scores; the prompts come from human users and the replies are model output, and this re-upload is ungated while the source repo is gated (auto)." The row carries no flag.
