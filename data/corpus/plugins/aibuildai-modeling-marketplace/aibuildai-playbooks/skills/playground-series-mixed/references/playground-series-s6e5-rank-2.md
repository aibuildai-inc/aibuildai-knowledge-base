# 2nd Place - Autonomous Codex Yolo!

Competition: playground-series-s6e5
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s6e5/writeups/2nd-place-autonomous-codex-yolo

Wow, what a close finish. I missed 1st place by `+0.00001`, that's frustrating. I even had a Private LB `0.95506` selected (as a final submission) in the morning but changed it to a more conservative sub which only got Private LB `0.95502` in the afternoon, ugh! :-( 

Congratulations to @optimistix who won 1st place. And Congratulations (and thank you) to everyone in the Top 10 Public LB. Your hard work made the competition exciting and encouraged everyone to work harder! @unseenuser @mikhailnaumov @milanfx @optimistix @kagglersergio @abisheksrivastav @donmarch14 @mahoganybuttstrings @jerry34 @masayakawamata 

This (May 2026) was a great month's playground competition. The dataset had real signal above and beyond the artifact signal that Kaggle's synthetic data generation procedure adds (i.e. if we train XGB on the original dataset without train.csv, it needs large max depth to find signal). 

This month, we needed to do some new feature engineering and new model building compared to previous playground competitions. Congratulations to everyone who finished in the Top 25 Private LB and beat the top public notebook blenders! You needed to discover and build some good models (and use OOF based ensembling) to beat the high scoring public blender notebooks!

# Autonomous Codex GPT5.5 Yolo!
At the end of April 2026, GPT5.5 was released and the resultant full auto `codex --yolo` is amazing! We can now let Codex run by itself for many hours without us micro-managing it. It's ability to plan and do data science is mind blowing!

# Codex Runs Experiments
For the first time in my life, I watched AI (i.e. LLM Agent) run experiments, perform feature engineering and tune hyperparameters all by itself just like I do, wow! After familiarizing Codex with the data and task, I told Codex to begin with a specific single model public notebook and then repeatedly perform experiments and improve its CV score. I asked Codex to maintain a `local_leaderboard.md` file with the top 10 CV score models. I let Codex to use 4x A100 GPUs and keep running experiments in parallel and keep working and improving CV score until I say stop. 



The result was amazing. Codex did everything! Codex wrote experiments, Codex ran the code on GPU, Codex logged the results (in local leaderboard), Codex analyzed the results, and then Codex decided on more experiments based on previous experiments. Codex kept the GPUs continuously in use, and in a few hours, I opened the file `local_leaderboard.md` and saw CV scores better than any publicly shared single models, wow!

# Diverse Models
Codex spent time improving the single model CV score of the big 6: XGBoost, CatBoost, LightGBM, RealMLP, TabM, and TabICLv2. And then I asked Codex to build other models for diversity. We didn't spend time improving the other models' CV scores. But having the other models improves ensemble diversity and CV score.

| rank | model/class | model count | best model | best CV AUC | public LB | private LB |
|---:|---|---:|---|---:|---:|---:|
| 1 | RealMLP | 40 | `realmlp2_exp147_five_seed_d...` | 0.954426 | 0.95382 | 0.95421 |
| 2 | XGBoost | 36 | `gpt1020_xgb_orighazard` | 0.953553 | 0.95294 | 0.95354 |
| 3 | CatBoost | 37 | `gpt1016_cat_ctrte` | 0.953404 | 0.95105 | 0.95190 |
| 4 | TabM | 11 | `tabm_exp089_wider_artifact_...` | 0.953371 | 0.95304 | 0.95345 |
| 5 | LightGBM | 25 | `lgbm_exp091_slow_lean_origi...` | 0.953023 | 0.95267 | 0.95290 |
| 6 | TabICL | 8 | `pri589_tabicl_v2_original_a...` | 0.950827 | 0.95053 | 0.95085 |
| 7 | FFM | 3 | `pri2_pri515_exp072_full_5_f...` | 0.949178 | 0.95048 | 0.95070 |
| 8 | Custom NN | 9 | `nn_exp022_duplicate_low_card` | 0.948923 | 0.95011 | 0.95050 |
| 9 | RandomForest | 3 | `pri2_pri520_exp125_full_5_f...` | 0.948845 | 0.94856 | 0.94885 |
| 10 | GNN | 3 | `pri2_pri516_exp079_full_5_f...` | 0.947632 | 0.94873 | 0.94927 |
| 11 | HistGB | 1 | `pub007_histboost` | 0.947546 | 0.94742 | 0.94827 |
| 12 | FM | 3 | `pri2_pri518_exp099_full_5_f...` | 0.947381 | 0.94837 | 0.94902 |
| 13 | KNN | 1 | `pri536_knn_7123` | 0.947231 | 0.94704 | 0.94719 |
| 14 | Other | 8 | `pri2_pri511_exp023_full_5_f...` | 0.947160 | 0.94981 | 0.95038 |
| 15 | TabTransformer | 3 | `pri_exp043_tabtran_domain_s...` | 0.947101 | 0.95003 | 0.95059 |
| 16 | ExcelFormer | 1 | `tal005_excelformer` | 0.946912 | 0.94902 | 0.94984 |
| 17 | Cox/survival | 1 | `pri522_cox_8007` | 0.946209 | 0.94717 | 0.94747 |
| 18 | DAE | 1 | `pri545_dae_8906` | 0.946046 | 0.94655 | 0.94717 |
| 19 | AMFormer | 1 | `tal007_amformer` | 0.944542 | 0.94570 | 0.94649 |
| 20 | YDF GBDT | 1 | `pri509_ydf_3000` | 0.943788 | 0.94325 | 0.94420 |
| 21 | MLP-PLR | 1 | `tal009_mlp_plr` | 0.943629 | 0.94428 | 0.94553 |
| 22 | Trompt | 1 | `tal006_trompt` | 0.943453 | 0.94479 | 0.94550 |
| 23 | TabR | 1 | `tal002_tabr` | 0.943445 | 0.94579 | 0.94652 |
| 24 | AutoInt | 1 | `tal011_autoint` | 0.942763 | 0.94485 | 0.94564 |
| 25 | GrowNet | 1 | `tal012_grownet_fixed` | 0.942110 | 0.94304 | 0.94439 |
| 26 | SAINT | 1 | `tal010_saint_fixed` | 0.941141 | 0.94407 | 0.94500 |
| 27 | ModernNCA | 1 | `tal001_modernnca` | 0.941058 | 0.94179 | 0.94296 |
| 28 | Gemini-derived | 2 | `pri555_gemini_9702` | 0.940993 | 0.94081 | 0.94178 |
| 29 | DCN | 1 | `tal008_dcn2` | 0.939530 | 0.94204 | 0.94263 |
| 30 | GPT-derived | 1 | `pri539_gpt_3c` | 0.939197 | 0.94313 | 0.94369 |
| 31 | GANDALF | 1 | `pri553_gandalf_2800` | 0.937605 | 0.93919 | 0.93990 |
| 32 | TabNet | 3 | `pri_exp062_tabnet_combo_te` | 0.937210 | 0.94094 | 0.94166 |
| 33 | NODE | 1 | `tal003_node` | 0.936121 | 0.93559 | 0.93675 |
| 34 | Logistic regression | 1 | `pri512_logreg_7011` | 0.933935 | 0.93381 | 0.93359 |
| 35 | FTTransformer | 1 | `pri517_ftt_6500` | 0.933224 | 0.93579 | 0.93585 |
| 36 | Snap/artifact | 4 | `pri544_snap_3500` | 0.932076 | 0.93169 | 0.93237 |
| 37 | LNN | 1 | `pri554_lnn_4400` | 0.893791 | 0.91096 | 0.91106 |

# Final Solution
My final solution is weighted average of 218 models (via logistic regression). I fit NVIDIA cuML Logistic Regression on the 218 models listed in the above table. The predictions were first converted into logits, and then fed into Logistic Regression. Then Logistic Prediction predicts the `submission.csv` file. This month, I did not train any models on the predictions of other models (i.e. stacking) nor did I use pseudo labels. My final solution is a blend of level 1 models 5 kfold predictions.
