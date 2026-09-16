# Pipeline shapes

Read in steps 8 and 11 when choosing the overall arrangement.

| Pipeline | Structure | Fits which task | Main precondition |
|---|---|---|---|
| **P1: SFT only** | clean the data -> SFT -> decoding | the default start for every task | usable supervised data exists |
| **P2: Iterative SFT** | SFT v1 -> analyze -> change data / LR / epochs -> v2 | rubric-scored, function-calling, multiple-choice, and writing tasks | you can retrain fast |
| **P3: Reasoning SFT + concise anneal** | reasoning SFT -> SFT on short correct traces | math tasks with exact answers | truncation or overthinking is the main problem |
| **P4: SFT + RFT/STaR + SFT** | SFT -> self-sample -> verify -> retrain | test-graded code, math | a reliable verifier exists |
| **P5: SFT + RFT + GRPO** | SFT -> verified self-generated data -> online RL | test-graded code and exact-answer math | the reward is stable and completions vary inside a group |
| **P6: SFT + GRPO** | SFT -> GRPO | verifiable tasks | the SFT model already gives valid answers reliably |
| **P7: SFT + DPO** | SFT -> preference pairs -> DPO | preference-judged writing | legal and trustworthy preference data exists |
| **P8: QLoRA + merge** | 4-bit base -> LoRA -> merge and export | VRAM or experiment throughput is tight | merge and dtype can be checked |
| **P9: SFT + checkpoint soup** | SFT checkpoints -> score-based selection -> averaging | small noisy evaluation sets | the candidates sit in the same basin |
| **P10: teacher-data SFT with trace selection** | distilled dataset or local teacher generation -> verify and keep one shortest correct trace per problem -> SFT -> anneal on hard concise traces | exact-answer math, reasoning, writing and health tasks with no public data | a teacher or trace dataset in the task's format; `34-distillation.md` flow D1 |
| **P11: SFT + on-policy distillation** | SFT -> student samples -> teacher per-token distribution or verifier -> one guarded round | verifiable tasks where the student already succeeds sometimes and a teacher fits beside it | shared tokenizer for token-level methods; a rollback rule; `34-distillation.md` flow D2 |
| **P12: SFT + judge-calibrated preference** | SFT -> calibrate judge against the grader -> rank student candidates -> DPO or RAFT by measured margin -> soup with the parent | preference-judged writing and rubric tasks | a judge that agrees with the grader on held-out pairs; `34-distillation.md` flow D3 |

Public training-run records show that every run starts at P1; only strong runs move on into P4-P7 often. The methodology skill holds one card per method (SFT, RFT, STaR, GRPO, DPO, LoRA, QLoRA); this card keeps only the arrangement choice the workflow makes.
