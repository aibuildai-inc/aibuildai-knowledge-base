# 3rd Place Solution — March Machine Learning Mania 2026

Competition: march-machine-learning-mania-2026
Rank: #3
Source: https://www.kaggle.com/c/march-machine-learning-mania-2026/writeups/3rd-place-solution-march-machine-learning-mania

# 3rd Place Solution — March Machine Learning Mania 2026

**Final Score:** 0.1160374 (MSE/Brier, 126 games) | **Rank:** 3rd / 3,485 teams | **Submission ID:** 51062131

## Summary

My approach combined **Logistic Regression on team-level differential features** with a **triple-market probability blend** (ESPN BPI, Vegas moneylines, Kalshi prediction markets). The core insight was that LR dramatically outperforms gradient-boosted trees for Stage 2 (unseen games) despite losing on Stage 1 leaderboard, and that prediction markets provide orthogonal information — especially for Round 1 games where injuries, travel, and matchup-specific factors matter.

Separate models for men's and women's tournaments. 52 rounds of iteration over 4 weeks.

[Tiered Market Blend Architecture]

## What Worked

1. **Logistic Regression over XGBoost for Stage 2** — LR CV Brier 0.124 vs XGB 0.157. With ~650 training rows and differential features, the true relationship is approximately logistic. LR has lower variance and generalizes better to unseen games. XGB won Stage 1 because it memorized ~536 known training games.

2. **Aggressive feature pruning** — Backward elimination from 33→23 men's features and 19→11 women's features was the single biggest CV improvement (-0.007 Brier in one round). Many "intuitively good" basketball features (talent, experience, height, coaching) added multicollinearity noise.

3. **Triple-market blend for Round 1 games** — ESPN BPI game-specific predictions (which account for injuries and matchup details) + Vegas no-vig moneylines + Kalshi championship odds, weighted 60/40 Vegas/BPI for consensus. Market weight: 90% for men's R1, 75% for women's R1, tapering for later rounds.

4. **Carry-over Elo** — Carrying 75% of end-of-season Elo into the next year (vs resetting) improved predictions by capturing multi-year program strength.

5. **Colley Matrix + SRS as custom rating systems** — Computing Colley Matrix rankings and Simple Rating System from regular-season results added orthogonal signal beyond Barttorvik/Massey.

6. **Feature interactions** — `SeedNum × massey_rank`, `ncsos × massey_rank`, `close_win_pct × pt_diff`, `srs × ap_rank` — these captured non-linear relationships that LR can't model directly.

## What Didn't Work

- **XGBoost/LightGBM for Stage 2** — Great for Stage 1 memorization, terrible for generalization on 650 rows
- **Ensembling LR + XGB** — Worsened both CV and LB. Don't blend fundamentally different model types on small data
- **Stacking individually-helpful features** — Adding features that each helped alone often hurt combined (multicollinearity in small-data regime)
- **Recency weighting for LR** — Helped XGB but hurt LR. LR already has low variance; downweighting older seasons reduces effective sample size
- **Extended training data pre-2015** — Barttorvik features have NaN before 2015, creating distributional shift
- **Isotonic calibration** — Trained on OOF distributions which differ from full-model predictions
- **MSE objective for XGBoost** — binary:logistic produces better-calibrated probabilities even when scored on Brier/MSE
- **Polynomial features, ElasticNet, Bagging, KNN blend, Tier-split by seed** — All noise or worse after 10+ rounds of ablation

## Approach Details

### Data Sources

| Source | What | Coverage |
|--------|------|----------|
| Competition data | Regular season results (compact + detailed), tournament results, seeds | 1985-2026 |
| Barttorvik (live CSV) | Adjusted efficiency (O/D), BARTHAG, WAB, SOS, tempo | 2015-2026, all D1 |
| Massey Ordinals | Composite of 28 ranking systems (POM, MOR, KPK, NET, etc.) | 2003-2026, men's only |
| nishaanamin/march-madness-data | KenPom Barttorvik (talent, exp, height, FT%), EvanMiya, AP Poll, Coach Results, Resumes | 2008-2025, tournament teams |
| WNCAA NET Rankings | Women's NET rankings | 2021-2026 |
| ESPN BPI | Game-specific R1 predictions + championship probabilities | 2026 only (scraped Mar 19) |
| Vegas moneylines | No-vig implied probabilities for R1 games | 2026 only (scraped Mar 19) |
| Kalshi | Championship futures odds (Bradley-Terry conversion) | 2026 only |

Team ID mapping files were constructed to link Barttorvik team names, ESPN team IDs, and nishaanamin team names to competition TeamIDs.

### Feature Engineering

All features are computed as **differentials**: `feature_Team1 - feature_Team2` (where Team1 has the lower TeamID). This is the standard approach for pairwise prediction — it ensures the model learns symmetric relationships.

**Men's features (25):**

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | SeedNum | Competition | Tournament seed (1-16) |
| 2 | win_pct | Computed | Regular season win percentage |
| 3 | adjoe | Barttorvik | Adjusted offensive efficiency |
| 4 | adjde | Barttorvik | Adjusted defensive efficiency |
| 5 | WAB | Barttorvik | Wins Above Bubble |
| 6 | sos | Barttorvik | Strength of schedule |
| 7 | adjt | Barttorvik | Adjusted tempo |
| 8 | efg_pct | Computed (box scores) | Effective field goal percentage |
| 9 | tov_pct | Computed (box scores) | Turnover percentage |
| 10 | oreb_pct | Computed (box scores) | Offensive rebound percentage |
| 11 | oreb_pct_d | Computed (box scores) | Opponent offensive rebound pct (defensive rebounding) |
| 12 | fg3_pct | Computed (box scores) | Three-point field goal percentage |
| 13 | ap_rank | AP Poll | Preseason AP ranking (captures "prestige" not in efficiency metrics) |
| 14 | em_change | KenPom Preseason | Preseason → end-of-season efficiency margin change (trajectory) |
| 15 | glm_quality | Computed | GLM team quality from game graph (Raddar method — MLE team strength) |
| 16 | SeedNum_x_pt_diff | Interaction | Seed × point differential |
| 17 | massey_rank_x_barthag | Interaction | Massey composite × Barttorvik win probability |
| 18 | close_win_pct_x_pt_diff | Interaction | Close game win% (≤7pt margin) × point differential |
| 19 | colley_rank | Computed | Colley Matrix ranking (from regular season W/L graph) |
| 20 | ncsos | Computed | Non-conference strength of schedule |
| 21 | SeedNum_x_massey | Interaction | Seed × Massey composite rank |
| 22 | ncsos_x_massey | Interaction | Non-conference SOS × Massey composite |
| 23 | srs | Computed | Simple Rating System: `rating = avg_margin + avg(opponent_ratings)`, iterative |
| 24 | srs_x_ap_rank | Interaction | SRS × AP rank |
| 25 | coach_pase | Coach Results | Coach Performance Above Seed Expectation (historical tournament over/under-performance) |

**Women's features (11):**

| # | Feature | Source |
|---|---------|--------|
| 1 | SeedNum | Competition |
| 2 | win_pct | Computed |
| 3 | oreb_pct_d | Computed (box scores) |
| 4 | blk_pct | Computed (box scores) |
| 5 | last_n_win_pct | Computed (last 10 games) |
| 6 | elo_slope | Computed (Elo trend over season) |
| 7 | net_x_elo | Interaction: NET rank × Elo |
| 8 | elo_x_colley | Interaction: Elo × Colley rank |
| 9 | srs | Computed (same as men's) |
| 10 | last_n_pt_diff_x_elo_x_colley | 3-way interaction |
| 11 | last_n_pt_diff_x_colley_rank | Interaction |

The women's model uses fewer features because women's basketball is more seed-predictable (higher seed wins 76.2% vs 68.9% in men's), and with ~646 training rows, more features = more overfitting.

[LR Feature Importance]

### Custom Rating Systems

**Carry-over Elo:**
- K-factor: 32, home advantage: 100 rating points
- Margin-of-victory multiplier: `ln(|margin| + 1) × 2.2 / (0.001 × |margin| + 2.2)` (diminishing returns for blowouts)
- Season carryover: `0.75 × end_elo + 0.25 × 1500` (captures multi-year program strength)

**Colley Matrix:**
- Solves `(2I + C)r = 1 + (w - l)/2` where C is the connectivity matrix from regular season games
- Produces rankings purely from win/loss record weighted by opponent strength

**SRS (Simple Rating System):**
- Iterative: `rating_i = avg_margin_i + mean(opponent_ratings)`, converged to 1e-6 tolerance
- Mean-centered each iteration

**GLM Quality (Raddar method):**
- Maximum likelihood team strength from the game graph, treating each game outcome as a Bernoulli trial

**Massey Composite:**
- Average of latest rankings from **all Massey Ordinal systems with full season coverage** across the training period (2015-2025 excl. 2020). This dynamically selects ~28 systems rather than a hand-picked subset. Using all available systems rather than a curated subset gave the best CV (-0.00246 vs 15-system subset).

### Model Architecture

```
Pipeline:
  1. SimpleImputer(strategy="median")   — handles NaN from missing external data
  2. StandardScaler()                    — centers/scales differential features
  3. LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
```

**Hyperparameters:**
- Men's: C=100, clip predictions to [0.03, 0.97]
- Women's: C=0.15, clip predictions to [0.005, 0.995]
- Men's C is much higher (less regularization) because 25 well-conditioned differential features benefit from more capacity
- Women's C is low (more regularization) because the model needs noise suppression with 11 features on ~646 rows

### Training / Validation

- **Training data:** Tournament games from 2015-2025 (excluding 2020 — no tournament due to COVID). ~669 men's games, ~646 women's games.
- **Validation:** Leave-One-Season-Out (LOSO) cross-validation using `GroupKFold(n_splits=10)` grouped by season. This simulates the real task: predict a future tournament using only past data.
- **Loss function:** sklearn LogisticRegression optimizes log loss (cross-entropy) natively. Evaluated on Brier score (MSE of probabilities) to match competition metric.
- **No recency weighting** for LR (confirmed through ablation — it hurts by reducing effective sample size).
- **Final CV (LOSO Brier):** Men's 0.11199, Women's 0.13506, Combined **0.12352**.

[Calibration Plot]

### Market Probability Blend (Post-Processing)

The key innovation was blending LR predictions with prediction market probabilities in a **tiered system**:

**Tier 1 — Round 1 game-specific (highest confidence):**
- ESPN BPI provides game-specific win probabilities that factor in injuries, travel, rest, and matchup details
- Vegas moneylines converted to no-vig implied probabilities: `P_fav = |ML| / (|ML| + 100)`, `P_dog = 100 / (ML + 100)`, then normalized
- Market consensus: `60% × Vegas + 40% × BPI` (Vegas has real money at stake → more efficient, reacts faster to injuries)
- Coverage: **30 men's R1 games** (30 with both BPI+Vegas, 2 BPI-only) and **28 women's R1 games**
- **Men's R1:** `10% LR + 90% market_consensus`
- **Women's R1:** `25% LR + 75% market_consensus`

**Tier 2 — Bradley-Terry from championship probabilities:**
- For non-R1 matchups where both teams have BPI championship odds: `P(A beats B) = strength_A / (strength_A + strength_B)`
- **Men's:** `75% LR + 25% BPI_BT`
- **Women's:** `85% LR + 15% BPI_BT`

**Tier 3 — Kalshi futures:**
- Same Bradley-Terry conversion from Kalshi championship futures odds
- **Men's:** `85% LR + 15% Kalshi_BT`
- **Women's:** `85% LR + 15% Kalshi_BT`

**Tier 4 — Pure LR (fallback):**
- For matchups where no market data exists

The rationale: markets aggregate millions of dollars of analysis including real-time injury/lineup information that no pre-tournament statistical model can capture. The tiered approach trusts markets most where they're most informative (game-specific R1 picks) and least where they're thin (deep tournament matchups between low-profile teams).

[LR vs Market Scatter]

[Seed Matchup Heatmap]

### Key Contrarian Calls That Paid Off

- **UConn 70% vs Michigan State (Sweet 16)** — Model was high on UConn despite them being lower-seeded. Correct.
- **Michigan 79% in championship** — Higher than Vegas (~70%). Michigan won. This call alone likely moved us from ~5th to 3rd.

### Progression Summary

[Brier Score Progression]

| Phase | Rounds | CV (Brier) | Key Changes |
|-------|--------|------------|-------------|
| XGBoost baseline | 1-11 | 0.158→0.157 | Feature engineering, Optuna HPO, split M/W, multi-seed |
| Switch to LR | 23-24 | 0.145→0.142 | LR dominates XGB for Stage 2 by -0.013 |
| Feature pruning | 27 | 0.135 | 33→18 M features, 19→8 W features. Biggest single-round gain (-0.007) |
| Feature interactions | 34-46 | 0.135→0.128 | Colley Matrix, SRS, ncsos, GLM quality, interactions, C-tuning |
| SRS + W interactions | 49-50 | 0.126→0.125 | SRS for both genders, feature/C optimization |
| Coach PASE + markets | 51-52 | 0.124 | Triple-market blend (BPI + Vegas + Kalshi) |

### Reproducibility

**Environment:**
- Python 3.9+
- Dependencies: `numpy`, `pandas`, `scikit-learn`
- No GPU required. Full pipeline runs in <30 seconds on any modern machine.

**To reproduce the winning submission:**

```bash
# 1. Clone the repository
git clone https://github.com/kevin1000/march-mania-2026-3rd-place.git
cd march-mania-2026-3rd-place

# 2. Install dependencies
pip install numpy pandas scikit-learn scipy

# 3. Download competition data (requires Kaggle CLI + API token)
mkdir -p data
kaggle competitions download -c march-machine-learning-mania-2026 -p data/
cd data && unzip '*.zip' && rm *.zip && cd ..

# External datasets are already included in the repository (external/ directory).
# These include Barttorvik stats (2015-2026), KenPom, EvanMiya, AP Poll,
# Coach Results, Kalshi odds, WNCAA NET, and team ID mapping files.

# 4. Run the final submission script
#    The submitted version used 90% market weight for M R1 and 75% for W R1.
#    To reproduce the exact winning submission:
python3 -c "
import round52_final as r52
r52.ALPHA_M_R1 = 0.90
r52.ALPHA_W_R1 = 0.75
r52.generate_submission(tag='winning_submission')
"
# Output: submissions/submission_winning_submission.csv
```

**Code structure:**

The solution uses an incremental import chain where each round builds on previous rounds:

| File | Role |
|------|------|
| `round27_pruned.py` | Base: data loading, carry-over Elo, Four Factors, Barttorvik stats, Massey composite, AP Poll, EvanMiya, KenPom Preseason, momentum features |
| `round45_final.py` | Adds Colley Matrix, GLM quality, close game record, all-system Massey, ncsos, feature interactions, Elo slope |
| `round46_final.py` | Adds Kalshi prediction market blend, elo×colley interaction, C re-tuning |
| `round49_final.py` | Adds SRS computation for M and W |
| `round50_final.py` | ESPN BPI data (game-specific R1 + championship BT), final feature/C optimization |
| `round51_final.py` | Coach PASE, Women's BPI R1 predictions |
| `round52_final.py` | **Winning submission.** Triple-market blend (ESPN BPI + Vegas moneylines + Kalshi) |

Import chain: `round27 → round45 → round46 → round49 → round50 → round51 → round52`

All 7 files are required. Each wraps the previous round's `build_m_features()` / `build_w_features()` functions, adding new features incrementally.

**Code repository:** https://github.com/kevin1000/march-mania-2026-3rd-place

### References

- **Competition data:** [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)
- **Barttorvik:** Team ratings downloaded from [barttorvik.com](https://barttorvik.com) (`{year}_team_results.csv`)
- **nishaanamin/march-madness-data:** [Kaggle dataset](https://www.kaggle.com/datasets/nishaanamin/march-madness-data) — KenPom Barttorvik, EvanMiya, AP Poll, Coach Results, KenPom Preseason
- **WNCAA NET Rankings:** [Kaggle dataset](https://www.kaggle.com/datasets/sqlrockstar/wncaa-net)
- **ESPN BPI:** Game-specific predictions scraped from ESPN.com on March 19, 2026
- **Vegas moneylines:** No-vig implied probabilities from ESPN/sportsbooks, March 19, 2026
- **Kalshi:** Championship futures from [kalshi.com](https://kalshi.com) API (`KXMARMAD-26`, `KXNCAWBB-26-CHAMP`), March 15, 2026
- **Colley Matrix:** Based on the [Colley Bias-Free Ranking method](https://www.colleyrankings.com/)
- **Raddar GLM quality:** Inspired by [Raddar's approach](https://www.kaggle.com/raddar) for team strength estimation via game-graph MLE

---

*This was my first entry in March Machine Learning Mania. The journey from a basic XGBoost model to a market-blended LR took 52 rounds of experimentation over 4 weeks, testing 100+ ideas. The biggest lesson: for small-data prediction tasks (~650 rows), simpler models with careful feature selection beat complex models every time.*
