# KoBLEX

KoBLEX: 226 multi-hop questions on Korean statutes (55 one-hop, 125 two-hop, 46 three-hop), each with the provisions that answer it, in Korean with an English statute version [1][2].

**Grades**: token F1 and a reference-anchored judge.

**Score**: token F1 against the answer, and LF-Eval: a GPT-4o judge scoring 1-10 against the reference provisions and answer, reported at Pearson 0.849 with human ratings [1].

**Judge**: GPT-4o.

**Access and licence**: open. The repository states CC BY-NC 4.0; the HF dataset `JihyungL/KoBLEX-koblex` states none. Follow the repository [2][3].

**Use it for**: statute retrieval and multi-hop application in a civil-law jurisdiction.

**Trap**: 226 items is small; the difference between two checkpoints needs a confidence interval before it means anything.

## Sources

Every source was read on 2026-09-29.

[1] KoBLEX paper, EMNLP 2025. https://arxiv.org/abs/2509.01324

[2] KoBLEX repository at commit `1a2d225be33a70ebf91e2b6369acf27551dcfbb4`: LICENSE, README. https://github.com/daehuikim/KoBLEX

[3] Hugging Face Hub API record. https://huggingface.co/api/datasets/JihyungL/KoBLEX-koblex
