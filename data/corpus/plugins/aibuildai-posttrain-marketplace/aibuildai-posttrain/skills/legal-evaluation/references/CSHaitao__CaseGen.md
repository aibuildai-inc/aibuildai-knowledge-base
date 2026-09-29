# CaseGen

CaseGen: 500 Chinese civil cases from China Judgments Online, each split into seven sections, with four generation stages: defence statement, trial facts, legal reasoning and judgment [1][2]. Long-form legal drafting graded by an LLM judge against the real court text.

**Grades**: reference-anchored judge, plus lexical metrics.

**Score**: the judge scores task-specific dimensions and an overall quality on 1-10; the prompts for defence, facts and reasoning say the reference would score 8, leaving 9-10 for output clearly better than it, and the judgment prompt anchors the reference at 10; BLEU, ROUGE-L and BERTScore are reported beside it [1][2].

**Judge**: `gpt-4o-2024-11-20`, told to be as strict as possible. Against human ratings: Spearman 0.750, Pearson 0.726, Kendall 0.667; ROUGE-L and BERTScore agreed less [1].

**Access and licence**: open. The README states CC BY-NC-SA 4.0 for non-commercial academic use; HF `CSHaitao/CaseGen` states CC BY-SA 4.0 [2][3].

**Use it for**: judgment and pleading drafting in a civil-law court style.

**Trap**: the judge anchors on the real court's text, so a legally sound judgment reasoned differently scores lower. The README's non-commercial terms are stricter than the Hub tag; follow the README.

## Sources

Every source was read on 2026-09-29.

[1] CaseGen paper. https://arxiv.org/abs/2502.17943

[2] CaseGen repository at commit `11a468d09e3f3f74c72efbb7feadf9253821cab7`: README, `eval/llm_eval.py`, `eval/template/`. https://github.com/CSHaitao/CaseGen

[3] Hugging Face Hub API record. https://huggingface.co/api/datasets/CSHaitao/CaseGen
