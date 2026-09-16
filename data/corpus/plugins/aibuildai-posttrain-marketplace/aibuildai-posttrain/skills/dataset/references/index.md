# Every post-training dataset in this skill

This is the full list: 215 datasets, read at 2026-08-11. The SKILL.md body carries only the recommended set, so the common case never opens this file. Each row points at its card, and the card holds the licence, the exact columns and splits, the pinned load line, and a real sample row.

Groups come from the column signature, which is a measurement, not a label anyone chose. `Use` is the judged verdict recorded with the selection: `train` means it may be trained on, `eval` means hold it out, `both` means part of it is a benchmark split. `Origin` says who wrote the text: `human`, `model`, `mixed`, or `unknown`.

**Read this before you build a training mix.** 34 of these 215 datasets are marked `eval`: they are listed so they can be recognised and kept OUT of training, not so they can be trained on. Another 37 are marked `both`, which means one split is safe to train on and another is a live benchmark; the card names which is which.

## Chat dialogues (SFT) (37)

| Dataset | Rows | Licence field | Use | Origin | Card |
|---|---|---|---|---|---|
| `AarushSah/lmsys-chat-1m` | 1,000,000 | none on the card | train | mixed | `AarushSah__lmsys-chat-1m.md` |
| `AGBonnet/augmented-clinical-notes` | 30,000 | mit | train | mixed | `AGBonnet__augmented-clinical-notes.md` |
| `AI-MO/NuminaMath-CoT` | 859,594 | apache-2.0 | both | mixed | `AI-MO__NuminaMath-CoT.md` |
| `AI-MO/NuminaMath-TIR` | 72,540 | apache-2.0 | both | mixed | `AI-MO__NuminaMath-TIR.md` |
| `allenai/tulu-3-sft-mixture` | 939,343 | odc-by | train | mixed | `allenai__tulu-3-sft-mixture.md` |
| `allenai/tulu-v2-sft-mixture` | 326,154 | odc-by | train | mixed | `allenai__tulu-v2-sft-mixture.md` |
| `allenai/WildChat-1M` | 837,989 | odc-by | train | mixed | `allenai__WildChat-1M.md` |
| `BitAgent/tool_calling` | 551,285 | none on the card | train | unknown | `BitAgent__tool_calling.md` |
| `dmis-lab/meerkat-instructions` | 439,834 | cc-by-nc-4.0 | train | mixed | `dmis-lab__meerkat-instructions.md` |
| `FreedomIntelligence/ApolloMoEDataset` | 292,914 | mit | train | mixed | `FreedomIntelligence__ApolloMoEDataset.md` |
| `hkust-nlp/deita-10k-v0` | 10,000 | mit | train | mixed | `hkust-nlp__deita-10k-v0.md` |
| `HuggingFaceH4/Bespoke-Stratos-17k` | 16,710 | none on the card | train | model | `HuggingFaceH4__Bespoke-Stratos-17k.md` |
| `HuggingFaceH4/no_robots` | 10,000 | cc-by-nc-4.0 | train | human | `HuggingFaceH4__no_robots.md` |
| `HuggingFaceH4/ultrachat_200k` | 515,311 | mit | both | model | `HuggingFaceH4__ultrachat_200k.md` |
| `khaimaitien/multi-hop-qa-function-calling-format-V1.0` | 28,733 | none on the card | train | model | `khaimaitien__multi-hop-qa-function-calling-format-V1.0.md` |
| `Magpie-Align/Magpie-Pro-300K-Filtered` | 300,000 | llama3 | train | model | `Magpie-Align__Magpie-Pro-300K-Filtered.md` |
| `Magpie-Align/Magpie-Pro-MT-300K-v0.1` | 300,000 | llama3 | train | model | `Magpie-Align__Magpie-Pro-MT-300K-v0.1.md` |
| `Magpie-Align/Magpie-Qwen2.5-Math-Pro-300K-v0.1` | 300,000 | none on the card | train | model | `Magpie-Align__Magpie-Qwen2.5-Math-Pro-300K-v0.1.md` |
| `Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered` | 300,000 | none on the card | train | model | `Magpie-Align__Magpie-Qwen2.5-Pro-300K-Filtered.md` |
| `mlabonne/FineTome-100k` | 100,000 | none on the card | train | mixed | `mlabonne__FineTome-100k.md` |
| `mlfoundations-dev/4o_annotated_aime` | 3,403 | none on the card | train | mixed | `mlfoundations-dev__4o_annotated_aime.md` |
| `mlfoundations-dev/a1_math_numina_aime` | 31,600 | none on the card | train | mixed | `mlfoundations-dev__a1_math_numina_aime.md` |
| `mlfoundations-dev/a1_math_openmathinstruct_aime` | 31,600 | none on the card | train | mixed | `mlfoundations-dev__a1_math_openmathinstruct_aime.md` |
| `mlfoundations-dev/multiple_samples_all_numina_aime` | 7,500 | none on the card | train | mixed | `mlfoundations-dev__multiple_samples_all_numina_aime.md` |
| `mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify` | 6,659 | none on the card | train | mixed | `mlfoundations-dev__multiple_samples_majority_consensus_numina_aime_math_verify.md` |
| `mlfoundations-dev/r1_annotated_aime` | 3,403 | none on the card | train | mixed | `mlfoundations-dev__r1_annotated_aime.md` |
| `Norquinal/claude_multiround_chat_30k` | 32,170 | none on the card | train | model | `Norquinal__claude_multiround_chat_30k.md` |
| `NousResearch/hermes-function-calling-v1` | 11,578 | apache-2.0 | train | model | `NousResearch__hermes-function-calling-v1.md` |
| `omi-health/medical-dialogue-to-soap-summary` | 10,000 | none on the card | train | model | `omi-health__medical-dialogue-to-soap-summary.md` |
| `Open-Orca/SlimOrca` | 517,982 | mit | train | model | `Open-Orca__SlimOrca.md` |
| `open-r1/Mixture-of-Thoughts` | 698,634 | none on the card | train | model | `open-r1__Mixture-of-Thoughts.md` |
| `open-r1/OpenR1-Math-220k` | 450,258 | apache-2.0 | train | mixed | `open-r1__OpenR1-Math-220k.md` |
| `Post-training-Data-Flywheel/gorilla-openfunctions-v1` | 12,125 | apache-2.0 | train | model | `Post-training-Data-Flywheel__gorilla-openfunctions-v1.md` |
| `prem-research/Funcdex-MT-Function-Calling` | 1,787 | mit | train | model | `prem-research__Funcdex-MT-Function-Calling.md` |
| `teknium/OpenHermes-2.5` | 1,001,551 | none on the card | train | model | `teknium__OpenHermes-2.5.md` |
| `WizardLMTeam/WizardLM_evol_instruct_V2_196k` | 143,000 | mit | train | mixed | `WizardLMTeam__WizardLM_evol_instruct_V2_196k.md` |
| `ZeroAgency/gemma3-pythonic-function-tool-calling-v1` | 299,047 | mit | train | model | `ZeroAgency__gemma3-pythonic-function-tool-calling-v1.md` |

## Instruction and answer pairs (SFT) (85)

| Dataset | Rows | Licence field | Use | Origin | Card |
|---|---|---|---|---|---|
| `agentlans/train-of-thought` | 1,080,020 | cc-by-4.0 | train | model | `agentlans__train-of-thought.md` |
| `AI-MO/aimo-validation-aime` | 90 | apache-2.0 | eval | human | `AI-MO__aimo-validation-aime.md` |
| `AI-MO/aimo-validation-amc` | 83 | apache-2.0 | eval | human | `AI-MO__aimo-validation-amc.md` |
| `AI-MO/aimo-validation-math-level-5` | 721 | none on the card | eval | human | `AI-MO__aimo-validation-math-level-5.md` |
| `allenporter/assist-llm-function-calling` | 2,357 | apache-2.0 | train | model | `allenporter__assist-llm-function-calling.md` |
| `Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn` | 180 | none on the card | eval | model | `Asap7772__aime_gpt-4o-mini_responses_evaluated_flatturn.md` |
| `axiong/pmc_llama_instructions` | 513,999 | openrail | train | human | `axiong__pmc_llama_instructions.md` |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 50,661 | odc-by | train | model | `bigcode__self-oss-instruct-sc2-exec-filter-50k.md` |
| `blue-blues/medical_cot` | 403,269 | apache-2.0 | train | mixed | `blue-blues__medical_cot.md` |
| `cais/mmlu` | 231,400 | mit | both | human | `cais__mmlu.md` |
| `CohereLabs/Global-MMLU` | 601,734 | apache-2.0 | eval | mixed | `CohereLabs__Global-MMLU.md` |
| `databricks/databricks-dolly-15k` | 15,011 | cc-by-sa-3.0 | train | human | `databricks__databricks-dolly-15k.md` |
| `derek-thomas/ScienceQA` | 21,208 | cc-by-sa-4.0 | both | human | `derek-thomas__ScienceQA.md` |
| `di-zhang-fdu/AIME_1983_2024` | 933 | mit | eval | human | `di-zhang-fdu__AIME_1983_2024.md` |
| `dim/competition_math` | 7,500 | none on the card | train | human | `dim__competition_math.md` |
| `EleutherAI/hendrycks_math` | 12,500 | mit | both | human | `EleutherAI__hendrycks_math.md` |
| `evalplus/mbppplus` | 378 | apache-2.0 | eval | mixed | `evalplus__mbppplus.md` |
| `fadodr/mental_health_therapy` | 12,258 | mit | train | mixed | `fadodr__mental_health_therapy.md` |
| `flytech/python-codes-25k` | 49,626 | none on the card | train | unknown | `flytech__python-codes-25k.md` |
| `FreedomIntelligence/medical-o1-reasoning-SFT` | 90,120 | apache-2.0 | train | mixed | `FreedomIntelligence__medical-o1-reasoning-SFT.md` |
| `fzkuji/MedQA` | 272,634 | unknown | both | human | `fzkuji__MedQA.md` |
| `garage-bAInd/Open-Platypus` | 24,926 | none on the card | train | mixed | `garage-bAInd__Open-Platypus.md` |
| `GBaker/MedQA-USMLE-4-options` | 11,451 | cc-by-4.0 | both | human | `GBaker__MedQA-USMLE-4-options.md` |
| `glaiveai/glaive-code-assistant` | 136,109 | apache-2.0 | train | model | `glaiveai__glaive-code-assistant.md` |
| `gneubig/aime-1983-2024` | 933 | cc0-1.0 | eval | human | `gneubig__aime-1983-2024.md` |
| `google-research-datasets/mbpp` | 1,401 | cc-by-4.0 | both | human | `google-research-datasets__mbpp.md` |
| `HuggingFaceH4/aime_2024` | 30 | none on the card | eval | human | `HuggingFaceH4__aime_2024.md` |
| `ise-uiuc/Magicoder-Evol-Instruct-110K` | 111,183 | apache-2.0 | train | model | `ise-uiuc__Magicoder-Evol-Instruct-110K.md` |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | 75,197 | mit | train | model | `ise-uiuc__Magicoder-OSS-Instruct-75K.md` |
| `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768` | 716 | none on the card | train | mixed | `jonathanyin__aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768.md` |
| `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768` | 740 | none on the card | train | mixed | `jonathanyin__aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768.md` |
| `jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-7b_traces_32768` | 830 | none on the card | train | mixed | `jonathanyin__aime_1983_2023_deepseek-r1-distill-qwen-7b_traces_32768.md` |
| `jonathanyin/aime_1983_2023_deepseek-r1_traces_32768` | 877 | none on the card | train | mixed | `jonathanyin__aime_1983_2023_deepseek-r1_traces_32768.md` |
| `jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768` | 866 | none on the card | train | mixed | `jonathanyin__aime_1983_2023_grok-3-mini-high_traces_32768.md` |
| `jonathanyin/aime_1983_2023_qwq-32b_traces_32768` | 889 | none on the card | train | mixed | `jonathanyin__aime_1983_2023_qwq-32b_traces_32768.md` |
| `jondurbin/airoboros-2.2` | 44,838 | other | train | model | `jondurbin__airoboros-2.2.md` |
| `knowrohit07/know_medical_dialogue_v2` | 6,307 | openrail | train | mixed | `knowrohit07__know_medical_dialogue_v2.md` |
| `lavita/AlpaCare-MedInstruct-52k` | 52,002 | none on the card | train | model | `lavita__AlpaCare-MedInstruct-52k.md` |
| `lavita/ChatDoctor-HealthCareMagic-100k` | 112,165 | none on the card | train | human | `lavita__ChatDoctor-HealthCareMagic-100k.md` |
| `lavita/MedQuAD` | 47,441 | none on the card | train | human | `lavita__MedQuAD.md` |
| `m-a-p/CodeFeedback-Filtered-Instruction` | 156,526 | apache-2.0 | train | model | `m-a-p__CodeFeedback-Filtered-Instruction.md` |
| `martim00/math_aime_2023` | 975 | none on the card | train | human | `martim00__math_aime_2023.md` |
| `math-ai/aime24` | 30 | apache-2.0 | eval | human | `math-ai__aime24.md` |
| `math-ai/aime25` | 30 | apache-2.0 | eval | human | `math-ai__aime25.md` |
| `medalpaca/medical_meadow_cord19` | 821,007 | none on the card | train | human | `medalpaca__medical_meadow_cord19.md` |
| `medalpaca/medical_meadow_medical_flashcards` | 33,955 | cc | train | mixed | `medalpaca__medical_meadow_medical_flashcards.md` |
| `medalpaca/medical_meadow_pubmed_causal` | 2,446 | none on the card | train | human | `medalpaca__medical_meadow_pubmed_causal.md` |
| `medalpaca/medical_meadow_wikidoc` | 10,000 | cc | train | mixed | `medalpaca__medical_meadow_wikidoc.md` |
| `medalpaca/medical_meadow_wikidoc_patient_information` | 5,942 | cc | train | mixed | `medalpaca__medical_meadow_wikidoc_patient_information.md` |
| `meta-math/MetaMathQA` | 395,000 | mit | train | mixed | `meta-math__MetaMathQA.md` |
| `microsoft/orca-math-word-problems-200k` | 200,035 | mit | train | model | `microsoft__orca-math-word-problems-200k.md` |
| `mjalg/function-code` | 233 | apache-2.0 | train | unknown | `mjalg__function-code.md` |
| `mlabonne/MedText` | 1,412 | none on the card | train | mixed | `mlabonne__MedText.md` |
| `Nan-Do/instructional_code-search-net-python` | 418,545 | apache-2.0 | train | mixed | `Nan-Do__instructional_code-search-net-python.md` |
| `newfacade/LeetCodeDataset` | 2,869 | apache-2.0 | both | mixed | `newfacade__LeetCodeDataset.md` |
| `Nexusflow/NexusRaven_API_evaluation` | 1,070 | none on the card | eval | mixed | `Nexusflow__NexusRaven_API_evaluation.md` |
| `nickrosh/Evol-Instruct-Code-80k-v1` | 78,264 | cc-by-nc-sa-4.0 | train | model | `nickrosh__Evol-Instruct-Code-80k-v1.md` |
| `nlpie/Llama2-MedTuned-Instructions` | 270,318 | cc-by-nc-4.0 | train | human | `nlpie__Llama2-MedTuned-Instructions.md` |
| `nomic-ai/gpt4all-j-prompt-generations` | 808,812 | apache-2.0 | train | model | `nomic-ai__gpt4all-j-prompt-generations.md` |
| `notbadai/python_functions_reasoning` | 206,299 | apache-2.0 | train | mixed | `notbadai__python_functions_reasoning.md` |
| `nvidia/HelpSteer2` | 21,362 | cc-by-4.0 | train | mixed | `nvidia__HelpSteer2.md` |
| `nvidia/OpenCodeInstruct` | 1,400,000 served of 5,000,000 declared (partial) | cc-by-4.0 | train | model | `nvidia__OpenCodeInstruct.md` |
| `Open-Orca/OpenOrca` | 2,942,029 | mit | train | mixed | `Open-Orca__OpenOrca.md` |
| `openbmb/UltraInteract_sft` | 288,579 | mit | train | mixed | `openbmb__UltraInteract_sft.md` |
| `openmed-community/MedReason-Stenographic` | 31,535 | apache-2.0 | train | model | `openmed-community__MedReason-Stenographic.md` |
| `Pandores/aime-1983-2025` | 1,034 | none on the card | eval | human | `Pandores__aime-1983-2025.md` |
| `Prompt48/AIME_Problem_Set_1983-2024` | 919 | cc0-1.0 | eval | human | `Prompt48__AIME_Problem_Set_1983-2024.md` |
| `qfq/genminiall_onlyqwenwrong_aimegpqatrain_domain_powerlaw_steps` | 1,000 | none on the card | train | mixed | `qfq__genminiall_onlyqwenwrong_aimegpqatrain_domain_powerlaw_steps.md` |
| `qwedsacf/competition_math` | 12,500 | mit | eval | human | `qwedsacf__competition_math.md` |
| `Rock23210/AIME_Deepseek_Clean` | 492 | mit | train | mixed | `Rock23210__AIME_Deepseek_Clean.md` |
| `sahil2801/CodeAlpaca-20k` | 20,022 | cc-by-4.0 | train | model | `sahil2801__CodeAlpaca-20k.md` |
| `Saxo/alpaca_function_calling_dataset` | 112,390 | apache-2.0 | train | model | `Saxo__alpaca_function_calling_dataset.md` |
| `ShenLab/MentalChat16K` | 16,084 | mit | train | mixed | `ShenLab__MentalChat16K.md` |
| `simplescaling/aime_nofigures` | 90 | apache-2.0 | eval | human | `simplescaling__aime_nofigures.md` |
| `starmpcc/Asclepius-Synthetic-Clinical-Notes` | 158,114 | cc-by-nc-sa-4.0 | train | model | `starmpcc__Asclepius-Synthetic-Clinical-Notes.md` |
| `tatsu-lab/alpaca` | 52,002 | cc-by-nc-4.0 | train | model | `tatsu-lab__alpaca.md` |
| `TIGER-Lab/MathInstruct` | 262,039 | mit | train | mixed | `TIGER-Lab__MathInstruct.md` |
| `TIGER-Lab/TheoremQA` | 800 | mit | eval | human | `TIGER-Lab__TheoremQA.md` |
| `TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k` | 2,048 | apache-2.0 | train | unknown | `TigerResearch__tigerbot-kaggle-leetcodesolutions-en-2k.md` |
| `TokenBender/code_instructions_122k_alpaca_style` | 121,959 | apache-2.0 | train | model | `TokenBender__code_instructions_122k_alpaca_style.md` |
| `vikp/python_code_instructions_filtered` | 170,635 | none on the card | train | mixed | `vikp__python_code_instructions_filtered.md` |
| `wics/strategy-qa` | 2,290 | other | eval | human | `wics__strategy-qa.md` |
| `WizardLMTeam/WizardLM_evol_instruct_70k` | 70,000 | mit | train | model | `WizardLMTeam__WizardLM_evol_instruct_70k.md` |
| `yizhongw/self_instruct` | 197,331 | apache-2.0 | both | model | `yizhongw__self_instruct.md` |
| `zuom/AIME-solutions` | 962 | none on the card | eval | human | `zuom__AIME-solutions.md` |

## Preference pairs (10)

| Dataset | Rows | Licence field | Use | Origin | Card |
|---|---|---|---|---|---|
| `allenai/tulu-2.5-preference-data` | 2,122,287 | odc-by | train | mixed | `allenai__tulu-2.5-preference-data.md` |
| `allenai/tulu-3-pref-personas-instruction-following` | 19,890 | none on the card | train | model | `allenai__tulu-3-pref-personas-instruction-following.md` |
| `Anthropic/hh-rlhf` | 169,352 | mit | both | mixed | `Anthropic__hh-rlhf.md` |
| `argilla/distilabel-intel-orca-dpo-pairs` | 12,859 | apache-2.0 | train | model | `argilla__distilabel-intel-orca-dpo-pairs.md` |
| `argilla/ultrafeedback-binarized-preferences-cleaned` | 60,917 | mit | train | mixed | `argilla__ultrafeedback-binarized-preferences-cleaned.md` |
| `BAAI/Infinity-Preference` | 59,438 | apache-2.0 | train | model | `BAAI__Infinity-Preference.md` |
| `HuggingFaceH4/ultrafeedback_binarized` | 187,405 | mit | both | mixed | `HuggingFaceH4__ultrafeedback_binarized.md` |
| `Intel/orca_dpo_pairs` | 12,859 | apache-2.0 | train | model | `Intel__orca_dpo_pairs.md` |
| `jondurbin/gutenberg-dpo-v0.1` | 918 | cc-by-4.0 | train | mixed | `jondurbin__gutenberg-dpo-v0.1.md` |
| `mlabonne/orpo-dpo-mix-40k` | 44,245 | apache-2.0 | train | mixed | `mlabonne__orpo-dpo-mix-40k.md` |

## Multiple-choice question sets (10)

| Dataset | Rows | Licence field | Use | Origin | Card |
|---|---|---|---|---|---|
| `AdaptLLM/medicine-tasks` | 5,383 | none on the card | eval | human | `AdaptLLM__medicine-tasks.md` |
| `allenai/ai2_arc` | 7,787 | cc-by-sa-4.0 | both | human | `allenai__ai2_arc.md` |
| `allenai/openbookqa` | 11,914 | unknown | both | human | `allenai__openbookqa.md` |
| `allenai/qasc` | 9,980 | cc-by-4.0 | both | human | `allenai__qasc.md` |
| `deepmind/aqua_rat` | 195,950 | apache-2.0 | both | human | `deepmind__aqua_rat.md` |
| `dmayhem93/agieval-gaokao-biology` | 210 | mit | eval | human | `dmayhem93__agieval-gaokao-biology.md` |
| `dmayhem93/agieval-gaokao-chemistry` | 207 | mit | eval | human | `dmayhem93__agieval-gaokao-chemistry.md` |
| `dmayhem93/agieval-gaokao-physics` | 200 | mit | eval | human | `dmayhem93__agieval-gaokao-physics.md` |
| `tau/commonsense_qa` | 12,102 | mit | both | human | `tau__commonsense_qa.md` |
| `truthfulqa/truthful_qa` | 1,634 | apache-2.0 | eval | human | `truthfulqa__truthful_qa.md` |

## Plain text (1)

| Dataset | Rows | Licence field | Use | Origin | Card |
|---|---|---|---|---|---|
| `Digital-nimbus/llama-2-oai-function-calling` | 2,243 | mit | train | unknown | `Digital-nimbus__llama-2-oai-function-calling.md` |

## Evaluation sets (4)

These declare an `eval.yaml` at the repository root. Keep them out of training.

| Dataset | Rows | Licence field | Use | Origin | Card |
|---|---|---|---|---|---|
| `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | viewer serves no rows | apache-2.0 | eval | human | `gorilla-llm__Berkeley-Function-Calling-Leaderboard.md` |
| `Idavidrein/gpqa` | 1,252 | cc-by-4.0 | eval | human | `Idavidrein__gpqa.md` |
| `openai/gsm8k` | 17,584 | mit | both | human | `openai__gsm8k.md` |
| `TIGER-Lab/MMLU-Pro` | 12,102 | mit | eval | mixed | `TIGER-Lab__MMLU-Pro.md` |

## Unclassified by column signature (68)

The measured columns do not decide the shape for these, or no columns were served at all. Open the card: its `Use it for` line states the shape.

| Dataset | Rows | Licence field | Use | Origin | Card |
|---|---|---|---|---|---|
| `allenai/sciq` | 13,679 | cc-by-nc-3.0 | both | human | `allenai__sciq.md` |
| `Amod/mental_health_counseling_conversations` | 3,512 | other | train | human | `Amod__mental_health_counseling_conversations.md` |
| `anon8231489123/ShareGPT_Vicuna_unfiltered` | viewer serves no rows | apache-2.0 | train | mixed | `anon8231489123__ShareGPT_Vicuna_unfiltered.md` |
| `berkeley-nest/Nectar` | 182,954 | apache-2.0 | train | model | `berkeley-nest__Nectar.md` |
| `bigcode/commitpackft` | 702,062 | mit | train | human | `bigcode__commitpackft.md` |
| `camel-ai/ai_society` | viewer served 0 rows | cc-by-nc-4.0 | train | model | `camel-ai__ai_society.md` |
| `camel-ai/biology` | 20,000 | cc-by-nc-4.0 | train | model | `camel-ai__biology.md` |
| `camel-ai/chemistry` | viewer served 0 rows | cc-by-nc-4.0 | train | model | `camel-ai__chemistry.md` |
| `camel-ai/math` | viewer served 0 rows | cc-by-nc-4.0 | train | model | `camel-ai__math.md` |
| `camel-ai/physics` | viewer served 0 rows | cc-by-nc-4.0 | train | model | `camel-ai__physics.md` |
| `casey-martin/MedInstruct` | viewer served 0 rows | none on the card | both | mixed | `casey-martin__MedInstruct.md` |
| `chilleD/SVAMP` | 1,000 declared on the card | mit | both | human | `chilleD__SVAMP.md` |
| `codeparrot/apps` | viewer serves no rows | mit | both | human | `codeparrot__apps.md` |
| `cognitivecomputations/dolphin` | viewer serves no rows | apache-2.0 | train | model | `cognitivecomputations__dolphin.md` |
| `cognitivecomputations/dolphin-coder` | viewer serves no rows | apache-2.0 | train | human | `cognitivecomputations__dolphin-coder.md` |
| `CohereForAI/aya_dataset` | 205,568 declared on the card | apache-2.0 | both | human | `CohereForAI__aya_dataset.md` |
| `CohereLabs/m-ArenaHard-v2.0` | 11,454 | none on the card | eval | human | `CohereLabs__m-ArenaHard-v2.0.md` |
| `curaihealth/medical_questions_pairs` | 3,048 | unknown | train | human | `curaihealth__medical_questions_pairs.md` |
| `daman1209arora/jeebench` | 515 | mit | eval | human | `daman1209arora__jeebench.md` |
| `Deepexi/function-calling-small` | 24,608 | cc-by-4.0 | train | unknown | `Deepexi__function-calling-small.md` |
| `deepmind/code_contests` | 4,044 | cc-by-4.0 | both | human | `deepmind__code_contests.md` |
| `Detsutut/MedInstruct` | viewer serves no rows | none on the card | both | model | `Detsutut__MedInstruct.md` |
| `euclaise/writingprompts` | 303,358 | mit | both | human | `euclaise__writingprompts.md` |
| `GAIR/lima` | viewer serves no rows | other | train | human | `GAIR__lima.md` |
| `glaiveai/glaive-function-calling` | 52,893 | apache-2.0 | train | model | `glaiveai__glaive-function-calling.md` |
| `glaiveai/glaive-function-calling-v2` | 112,960 | apache-2.0 | train | model | `glaiveai__glaive-function-calling-v2.md` |
| `greengerong/leetcode` | 2,360 | mit | train | unknown | `greengerong__leetcode.md` |
| `har1/MTS_Dialogue-Clinical_Note` | 1,301 | mit | train | human | `har1__MTS_Dialogue-Clinical_Note.md` |
| `HPAI-BSC/medical-specialities` | 13,775 | none on the card | eval | human | `HPAI-BSC__medical-specialities.md` |
| `HPAI-BSC/Medprompt-MedQA-CoT` | viewer served 0 rows | llama3.1 | train | model | `HPAI-BSC__Medprompt-MedQA-CoT.md` |
| `InclusionAI/Ling-Coder-SFT` | viewer serves no rows | apache-2.0 | train | model | `InclusionAI__Ling-Coder-SFT.md` |
| `jinaai/code_search_net_clean` | 1,914,013 | none on the card | both | human | `jinaai__code_search_net_clean.md` |
| `kaist-ai/CoT-Collection` | 1,837,928 | cc-by-4.0 | train | mixed | `kaist-ai__CoT-Collection.md` |
| `kroshan/BioASQ` | 8,216 | none on the card | train | human | `kroshan__BioASQ.md` |
| `lmsys/chatbot_arena_conversations` | 33,000 declared on the card | cc | train | mixed | `lmsys__chatbot_arena_conversations.md` |
| `mahfoos/Patient-Doctor-Conversation` | 3,325 | none on the card | train | unknown | `mahfoos__Patient-Doctor-Conversation.md` |
| `mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered` | 4,069 | none on the card | train | mixed | `mlfoundations-dev__bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered.md` |
| `Muennighoff/natural-instructions` | viewer serves no rows | none on the card | both | human | `Muennighoff__natural-instructions.md` |
| `nampdn-ai/tiny-codes` | viewer serves no rows | mit | train | model | `nampdn-ai__tiny-codes.md` |
| `Nan-Do/code-search-net-python` | 455,243 | apache-2.0 | train | mixed | `Nan-Do__code-search-net-python.md` |
| `nvidia/HelpSteer3` | 132,937 | cc-by-4.0 | train | mixed | `nvidia__HelpSteer3.md` |
| `nvidia/OpenMathInstruct-1` | 6,879,694 | other | train | mixed | `nvidia__OpenMathInstruct-1.md` |
| `nvidia/OpenMathInstruct-2` | 21,972,791 | cc-by-4.0 | train | mixed | `nvidia__OpenMathInstruct-2.md` |
| `open-r1/codeforces` | 29,286 served of 36,304 declared (partial) | cc-by-4.0 | both | mixed | `open-r1__codeforces.md` |
| `openai/openai_humaneval` | 164 | mit | eval | human | `openai__openai_humaneval.md` |
| `openai/summarize_from_feedback` | 193,841 | none on the card | both | mixed | `openai__summarize_from_feedback.md` |
| `OpenAssistant/oasst2` | 135,174 | apache-2.0 | train | human | `OpenAssistant__oasst2.md` |
| `openbmb/UltraFeedback` | 63,967 | mit | train | mixed | `openbmb__UltraFeedback.md` |
| `openlifescienceai/medmcqa` | 193,155 | apache-2.0 | both | human | `openlifescienceai__medmcqa.md` |
| `PKU-Alignment/PKU-SafeRLHF` | 164,236 | cc-by-nc-4.0 | both | mixed | `PKU-Alignment__PKU-SafeRLHF.md` |
| `PygmalionAI/PIPPA` | viewer serves no rows | apache-2.0 | train | mixed | `PygmalionAI__PIPPA.md` |
| `qiaojin/PubMedQA` | 273,518 | mit | both | mixed | `qiaojin__PubMedQA.md` |
| `redwoodresearch/mbpp_extended` | 38,215 | none on the card | train | model | `redwoodresearch__mbpp_extended.md` |
| `roborovski/synthetic-tool-calls` | 6,008 | none on the card | train | model | `roborovski__synthetic-tool-calls.md` |
| `ruslanmv/ai-medical-chatbot` | 256,916 | none on the card | train | human | `ruslanmv__ai-medical-chatbot.md` |
| `Salesforce/xlam-function-calling-60k` | viewer serves no rows | cc-by-4.0 | train | model | `Salesforce__xlam-function-calling-60k.md` |
| `semeru/text-code-galeras-code-generation-from-docstring-3k-deduped` | 2,937 | none on the card | train | human | `semeru__text-code-galeras-code-generation-from-docstring-3k-deduped.md` |
| `sharkchill-xy/HumanEval_mbpp_format` | 164 | none on the card | eval | human | `sharkchill-xy__HumanEval_mbpp_format.md` |
| `shibing624/medical` | viewer serves no rows | apache-2.0 | both | mixed | `shibing624__medical.md` |
| `stanfordnlp/SHP` | 385,563 | none on the card | both | human | `stanfordnlp__SHP.md` |
| `stellalisy/MediQ_AskDocs_preference` | viewer serves no rows | mit | both | human | `stellalisy__MediQ_AskDocs_preference.md` |
| `stingning/ultrachat` | viewer serves no rows | mit | train | model | `stingning__ultrachat.md` |
| `THUDM/AgentInstruct` | 1,866 declared on the card | none on the card | train | model | `THUDM__AgentInstruct.md` |
| `UCSD26/medical_dialog` | 5,558,895 declared on the card | unknown | train | human | `UCSD26__medical_dialog.md` |
| `Vishal24/function_calling` | 16,746 | none on the card | train | unknown | `Vishal24__function_calling.md` |
| `WNJXYK/AIME_1983_2024-Reasoning-Paths` | viewer serves no rows | mit | train | model | `WNJXYK__AIME_1983_2024-Reasoning-Paths.md` |
| `xw27/scibench` | 692 | none on the card | eval | human | `xw27__scibench.md` |
| `zhuzilin/aime-2024` | 30 | none on the card | eval | human | `zhuzilin__aime-2024.md` |

## How this list was built

Searched at 2026-08-11 across the Hugging Face Hub, GitHub, and the DOI repositories (Zenodo, Harvard Dataverse, figshare, OSF, Dryad, Mendeley). 13 channels ran and produced 2834 candidates in the index; the largest were hub-search (2209), hub-author (726), hub-full-text (637), benchmark-manifest (381), arxiv-inversion (246), readme-link (51), dataverse (38), zenodo (18). A dataset earned a card when its content could be read and judged, and a person then judged every judgeable row for use, origin, and risk.

Two limits a reader must know. Mendeley is unreached, not empty: every endpoint answered 403 behind a bot check, so its zero means nothing. And every number here is a snapshot at one commit; datasets change, so `finding-more.md` holds the live search method for anything newer.
