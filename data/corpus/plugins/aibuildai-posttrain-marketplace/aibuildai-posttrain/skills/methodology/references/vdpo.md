# V-DPO

DPO with a classifier-free-guidance term added to the loss so that responses matching the input image, not just the input text, are pushed up during preference training on a vision-language model.

**V-DPO** (Vision-guided Direct Preference Optimization) is an offline preference-optimization method for large vision-language models (LVLMs), introduced by Xie et al. to reduce hallucination by making training-time preference learning pay more attention to the image rather than only the language prior [1]. Its parent is DPO, which reparameterizes the RLHF reward so the optimal policy can be solved in closed form and trained with a simple classification loss instead of RL sampling and a separate reward model [2]. V-DPO's mechanism: it borrows classifier-free guidance (CFG) from diffusion models - originally a way to steer a diffusion sampler toward a class condition without a separate classifier [1] - and adds a KL term $\alpha\,\mathbb{D}_{\mathrm{KL}}[\pi(y\mid v,x)\,\|\,\pi(y\mid x)]$ to DPO's reward-maximization objective, which pushes the vision-conditioned response distribution away from the text-only response distribution [1]. Two reasons the paper gives for existing: LVLM hallucination is linked to over-reliance on the LLM backbone's language priors and insufficient attention to visual context [1]; and prior fixes for this were inference-time decoding penalties (e.g. OPERA), which the paper says require increased inference time and thereby obstruct generalizability and scalability across diverse data domains and sizes, rather than fixing the training objective [1]. The paper's own name search collides with an unrelated "VDPO" (Variational Delayed Policy Optimization, an RL method for delayed rewards, 10 citations) - this card is confirmed against the top-cited match under an exact title search, "V-DPO: Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization" [1].

No landmark production system's own paper or documentation was found citing adoption of V-DPO as of this check (fetch of the paper and its GitHub repository, 2026-08-09) [1][3]; its own experiments compare it only against an SFT baseline, vanilla DPO, and HA-DPO, a separate DPO variant for hallucination trained on 16K style-consistent pairs [1][4]. On AMBER, training with only 5K synthetic V-DPO preference pairs reaches an AMBER Score of 88.4, against 85.7 for HA-DPO trained on 16K pairs - a 2.7-point gap by Table 3's own CHAIR/F1 figures, versus the paper's own prose statement of a 3.7-point gap for the same comparison [1, Table 3]; on MMHal-Bench, V-DPO trained on synthetic data cuts the hallucination rate from 0.59 (vanilla DPO) to 0.53 and raises the overall GPT-4 score from 2.12 to 2.36 [1, Table 4]. Lineage in one line: DPO (2023 [2]) -> V-DPO (2024 [1]); no adopters or named successors were found in this check.

**When to pick it**: offline preference training for an LVLM when hallucination stems from the model ignoring the image and defaulting to text-only priors, and you can construct or reuse image-conditioned preference pairs [1]. Prefer vanilla DPO [2] when the preference signal is purely about response quality and visual grounding is not the failure mode being targeted. The paper's own nearest offline alternative is HA-DPO, a DPO variant trained on a larger (16K) style-consistent hallucination preference set with the same LLaVA-1.5-7B backbone [1][4]. The paper's own alternative outside training-time preference learning is inference-time decoding correction, e.g. OPERA, which penalizes over-trust candidates at generation time instead of updating weights, trading no training cost for the paper's stated downside of increased inference time that obstructs generalizability and scalability across diverse data domains and sizes [1][5].

**Variant of**: DPO [2].

**Data it needs**: image-conditioned preference pairs $(v, x, y_w, y_l)$ - an image, a query, a preferred response, and a dispreferred response - drawn from either "response-contrast" pairs (same image, different responses) or the paper's own "image-contrast" pairs (same response, different images, constructed by swapping visual elements in an image with unexpected substitutes) [1]. The paper trains on 5K synthetic response+image-contrast pairs, or separately on 5K human-annotated response-contrast pairs from RLHF-V, both on top of an LLaVA-v1.5-7B SFT checkpoint [1]. Offline: the loss is computed on a fixed, pre-collected preference dataset, with no sampling from the policy during training [1][2].

**Extra models**: one frozen reference model $\pi_{ref}$, the same as vanilla DPO, needed to compute the log-ratio term in the loss [1][2]. No separate value network or reward model. The visual-guidance term also requires a text-only forward pass of the policy itself (image replaced by zeros, or the frozen SFT checkpoint in the paper's ablation) [1] - not a new model, but an extra forward pass; see Cost.

**Shipped by**: no library implements V-DPO. The paper's own reference code is a standalone research repository (`YuxiXie/V-DPO` on GitHub, Apache-2.0, 60 stars as of this check) built as a fork of the LLaVA codebase, with the loss implemented in `llava_dpo/train/dpo_train.py` and run via `scripts/v1_5/vdpo.sh` - a script-based training pipeline, not an installable package with an exported trainer class [3][6]. trl's own documentation table of contents lists DPO, Online DPO, and SDPO trainers but no vision-guided or V-DPO variant [7]. Building it on top of an existing trainer requires adding the CFG log-ratio term to the DPO loss and, for the two ablation variants, either a second frozen policy for the static text-only distribution or a softmax-normalization step over the guided logits [1] - a loss modification on an existing DPO trainer, not a new sampling loop.

## How it works

Each step: for a pair (preferred $y_w$, dispreferred $y_l$) sharing image $v$ and query $x$, compute a CFG-modified log-ratio for each response against the reference model, then apply the DPO logistic loss to their difference [1].

**Background - DPO's objective**, which V-DPO modifies, maximizes reward under a KL constraint to the reference policy [1, Eq. 1]:

$$ \max_{\pi}\, \mathbb{E}_{(v,x)\sim\mathcal{I}\times\mathcal{P},\, y\sim\pi}\Big[ r(v,x,y) - \beta\, \mathbb{D}_{\mathrm{KL}}\big[\pi(y\mid v,x)\,\|\,\pi_{\mathrm{ref}}(y\mid v,x)\big] \Big] $$

**Classifier-free guidance**, extended by the paper from diffusion models to autoregressive text generation, reweights a conditional distribution by the ratio of conditioned to unconditioned likelihoods raised to a guidance strength $\gamma$ [1, Eq. 5-6]:

$$ \hat{\pi}_\theta(y \mid v, x) \varpropto \pi_\theta(y \mid x) \left( \frac{\pi_\theta(y \mid v, x)}{\pi_\theta(y \mid x)} \right)^{\gamma} $$

**V-DPO's objective** adds a second KL term, weighted by $\alpha > 0$, that pulls the vision-conditioned distribution away from the text-only distribution [1, Eq. 7]:

$$ \max_{\pi}\, \mathbb{E}_{(v,x)\sim\mathcal{I}\times\mathcal{P},\, y\sim\pi}\Big[ r(v,x,y) - \beta\, \mathbb{D}_{\mathrm{KL}}\big[\pi(y\mid v,x)\,\|\,\pi_{\mathrm{ref}}(y\mid v,x)\big] + \alpha\, \mathbb{D}_{\mathrm{KL}}\big[\pi(y\mid v,x)\,\|\,\pi(y\mid x)\big] \Big] $$

Solving this and applying the Bradley-Terry model as in DPO gives the training loss, over the union of response-contrast and image-contrast pairs $\mathcal{D} = \mathcal{D}_y \cup \mathcal{D}_v$ [1, Eq. 9]:

$$ \mathcal{L}_{\mathrm{VDPO}}(\pi_\theta; \pi_{\mathrm{ref}}) = -\, \mathbb{E}_{(w,l)\sim\mathcal{D}} \log \sigma\big(\beta\, u_{\pi_\theta}^{w,l}\big), \qquad u_{\pi_\theta}^{w,l} = f_\theta^w - f_\theta^l $$

$$ f_\theta(v,x,y) = \log \frac{\pi_\theta(y\mid v,x)\, \hat{\varphi}_\theta(v,x,y)}{\pi_{\mathrm{ref}}(y\mid v,x)}, \qquad \varphi_\theta(v,x,y) = \left( \frac{\pi_\theta(y\mid v,x)}{\pi_\theta(y\mid x)} \right)^{\gamma - 1} $$

$\gamma = 1 - \alpha/\beta$: increasing $\alpha$ decreases $\gamma$ below 1, which is the opposite direction from inference-time CFG (which increases $\gamma$ above 1) [1]. At $\gamma = 1$ ($\alpha = 0$) the $\varphi_\theta$ term is 1 and the loss reduces exactly to vanilla DPO's loss [1]. $\hat{\varphi}_\theta$ carries a hat because gradients are stopped through it during backpropagation, so it acts as a training-time reweighting rather than a learned term, and the text-only distribution $\pi_\theta(y\mid x)$ is estimated by feeding zeros in place of the image, $\hat\pi_\theta(\cdot\mid x) = \hat\pi_\theta(\cdot\mid \mathbf{0}, x)$ [1, Eq. 10]. A worked example with $\beta=0.1$, $\gamma=0.75$ (the paper's own synthetic-data setting [1]): if $\log[\pi_\theta(y_w\mid v,x)/\pi_{ref}(y_w\mid v,x)] = 0.5$ and the guidance ratio $\pi_\theta(y_w\mid v,x)/\pi_\theta(y_w\mid x) = 2$, then $\varphi_\theta = 2^{\gamma-1} = 2^{-0.25} \approx 0.841$, so $f_\theta^w \approx 0.5 + \ln(0.841) \approx 0.5 - 0.173 = 0.327$ - the guidance term lowers $f_\theta^w$ whenever the response is less image-specific than text-specific ($\varphi_\theta<1$), and the loss compares this adjusted value against the same computation for $y_l$.

**Two variants the paper itself defines and ablates**: (1) using the current policy's own zeroed-image pass for $\hat\pi(\cdot\mid x)$, versus a "static-lm" variant that instead uses the frozen initial SFT model for that term, which the paper's ablation found improves performance across generative and discriminative tasks [1, Sec. 5.3]; (2) an un-normalized guidance term (Eq. 9, above) versus a softmax-normalized version (Eq. 11) that the paper found further lowers CHAIR hallucination scores (e.g. 6.6->6.2 and 5.6->5.5 on the two data scenarios) but can hurt discriminative-task performance [1, Sec. 5.3]. The paper's main results table (Tables 1-4) reports the un-normalized, non-static-lm version [1].

## Cost

**Theory, from the method's own math:**

- Time: relative to DPO, V-DPO adds one extra forward pass per response per pair - the text-only pass with the image zeroed out, $\pi_\theta(y\mid x)$ - on top of DPO's existing policy and reference forward passes over $(y_w, y_l)$ [1, Eq. 9-10]. No extra backward pass, since gradients through $\hat\varphi_\theta$ are stopped [1].
- Memory: same trained-model count as DPO - one trained policy plus one frozen reference model, no value network or reward model [1][2]. The "static-lm" ablation variant needs a second frozen model held in memory (the initial SFT checkpoint, distinct from the KL reference model, though in the paper's setup both are the same checkpoint) [1, Sec. 5.3].
- A naive reading of Eq. 9 might suggest $\varphi_\theta$ needs its own gradient graph; the paper's own text states gradients are disabled through it, so in practice it only adds forward compute, not backward compute or optimizer state [1].

**In practice, per framework:**

- No shipping framework exists to report framework-specific cost for; the only implementation is the paper's own script-based repository, which does not document throughput or memory numbers beyond stating that all experiments ran on up to 4x40GB A100 GPUs [1, Appendix B][3].

## How to use it

- Data preparation: start from an existing response-contrast preference set (e.g. RLHF-V, human-annotated) or construct image-contrast pairs by (a) extracting object-level captions from an LVLM, (b) using an LLM to propose an "unexpected" element substitution with a stated reason, (c) editing the image (the paper uses Stable Diffusion-based editing) and captioning the edited image, and (d) filtering pairs by CLIPScore [1, Sec. 4.2, Appendix C]. The paper seeds this pipeline from COCO, Visual Genome, and VCR images [1, Appendix C].
- Reward/label convention: no scalar reward model - preference is a binary chosen/rejected label per pair, exactly as in DPO [1][2].
- Key knobs, with each source's own value ("not stated" = the source was checked and does not give it; "not checked" = this card did not verify that source):

| knob | paper's synthetic-data setting [1] | paper's human-annotated (RLHF-V) setting [1] |
| --- | --- | --- |
| $\beta$ (KL-to-reference weight) | 0.1 | 0.1 |
| $\gamma$ (visual-guidance strength) | 0.75 | 0.00 |
| learning rate | 1e-6 | 1e-6 |
| batch size | 64 | 64 |
| epochs | 4 | 4 |
| training pairs | 5K | 5K |
| backbone | LLaVA-v1.5-7B | LLaVA-v1.5-7B |

  $\beta = 0.1$ is stated to follow the DPO paper's own choice directly [1]. $\gamma$ is not a single default: the paper reports it is far more sensitive on synthetic (image-contrast-containing) data, where $\gamma$ below 0.75 (down to $\gamma=0$, i.e. $\gamma-1=-1$) causes divergence from the initial model and degrades hallucination scores, whereas the human-annotated response-contrast-only data tolerates $\gamma=0$ [1, Sec. 5.3]. No framework default exists to compare against (see Shipped by).
- Trade-off a run designer faces: lowering $\gamma$ (raising $\alpha$) strengthens visual guidance and helps up to a data-dependent point, then destabilizes training on image-contrast data specifically, per the paper's own ablation on AMBER [1, Sec. 5.3, Fig. 4] - the paper's guidance is to tune $\gamma$ per data type rather than use one fixed value.

## While it runs

- Signals and their healthy shapes: the paper's own diagnostic is the "distribution gap" - the difference in log-likelihood between vision-conditioned and text-only-conditioned generation for accurate versus hallucinatory samples (Figure 1b); before alignment this gap is dominated by the text-only distribution, and the paper's stated goal of training is to shift the vision-conditioned distribution to be the more discriminative one [1, Sec. 1, Sec. 5.4]. No log-line names or numeric healthy ranges are given, since there is no framework implementation with a logging convention (see Shipped by).
- Published reference runs: the paper's own Tables 1-4, LLaVA-v1.5-7B on POPE, AMBER, HallusionBench, and MMHal-Bench, comparing SFT, HA-DPO, vanilla DPO, and V-DPO on both a 5K synthetic and a 5K RLHF-V preference set [1] - the reference point to compare against, since no other published run exists.
- Degeneracies and defaults: $\gamma = 0$ (maximal guidance, $\gamma-1=-1$) is a documented failure point on image-contrast synthetic data, causing "substantial divergence from the initial model" and degraded hallucination scores, attributed by the paper to the image-contrast pairs deviating further from the SFT model's generation distribution than response-contrast pairs do [1, Sec. 5.3]. There is no framework default to diverge from a paper default, since no framework ships this method.
- Named successors: none found in this check - the fetched paper and its GitHub repository do not name a successor method, and a citation-graph query (Semantic Scholar API) could not be completed at check time (rate-limited, 2026-08-09) [1][3].
- Known failure modes: from the paper's own Limitations section, two are stated - the method does not address cases where language priors are actually needed for a correct answer (e.g. preference for generation fluency is not modeled in its data construction), and the synthetic data-construction pipeline can introduce noise and bias from its automatic generation steps, which may degrade preference-optimization performance [1, Limitations]. The GitHub repository's issue tracker (6 issues total, 3 open/3 closed, at commit `ed27c32` as of 2026-08-09) shows two independently reported problems the authors never resolved: on issue #2, a user reports the released training code does not train at the paper's stated learning rate of 1e-6 (gradient stays `None`, loss does not change) and that raising the learning rate to 0.1 was needed to get gradients flowing; and on issue #1, a user who reconstructed the RLHF-V preference data from `llamafactory/RLHF-V` states the paper's authors never responded about the specific data-construction procedure used in the paper [3].
- What the gain is - and is not: the paper's own results show gains concentrated on hallucination benchmarks (POPE, AMBER, HallusionBench, MMHal-Bench) and, within MMHal-Bench, particularly on comparison and environment question types for vanilla-DPO-trained baselines [1, Sec. 5.2]; the paper positions the method specifically as correcting over-reliance on language priors, not as a general capability improvement, and explicitly leaves fluency-oriented preference and other general alignment scenarios to future work [1, Limitations].

## Sources

[1] Xie, Li, Xu, and Kan, "V-DPO: Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization", 2024. https://arxiv.org/abs/2411.02712 - defines V-DPO: objective, CFG background, data construction, hyperparameters, results, ablations, limitations. Fetched 2026-08-09 (abstract page and full HTML text).

[2] Rafailov, Sharma, Mitchell, Manning, Ermon, and Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023. https://arxiv.org/abs/2305.18290 - DPO, the parent method. Fetched 2026-08-09 (abstract page).

[3] YuxiXie/V-DPO GitHub repository. https://github.com/YuxiXie/V-DPO - reference implementation, README, file tree, license, star count, and issue tracker. Repository metadata, README, file tree, and all 6 issues (via `api.github.com/repos/YuxiXie/V-DPO/issues?state=all` and each issue's `/comments`) fetched 2026-08-09, main branch at commit `ed27c326cc5003a5a3885a8b3b84ced5a1e317c3`; this pin covers commits and issues that existed as of that fetch, not later activity on the live repository.

[4] Zhao, Wang, Ouyang, Dong, Wang, and He, "Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization", 2023. arXiv:2311.16839 - HA-DPO, cited via V-DPO's own reference list and used as its baseline. Not independently fetched; description taken from V-DPO's own text [1].

[5] Huang, Dong, Zhang, Wang, He, Wang, Lin, Zhang, and Yu, "OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation", 2023. arXiv:2311.17911 - decoding-time hallucination correction, cited via V-DPO's own reference list. Not independently fetched; description taken from V-DPO's own text [1].

[6] YuxiXie/V-DPO README.md (raw). https://raw.githubusercontent.com/YuxiXie/V-DPO/main/README.md - training entry point (`llava_dpo/train/dpo_train.py`, `scripts/v1_5/vdpo.sh`), environment setup, attribution to the LLaVA codebase. Fetched 2026-08-09.

[7] trl documentation table of contents. https://raw.githubusercontent.com/huggingface/trl/main/docs/source/_toctree.yml - unpinned `main`-branch build; lists DPO, Online DPO, and SDPO trainers with no vision-guided or V-DPO entry as of this reading. Fetched 2026-08-09; the file's own last-touching commit on `main` is `4f0b9b5ab17cd59dac39484d291f7b54e09e6868`, dated 2026-07-20, so the absence-of-entry claim covers trl's docs tree as of that commit, not any later, unpulled change to `main`.
