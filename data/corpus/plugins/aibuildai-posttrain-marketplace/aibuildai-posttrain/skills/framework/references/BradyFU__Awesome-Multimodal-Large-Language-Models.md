# Awesome-Multimodal-Large-Language-Models

A curated reading list of multimodal-LLM papers, benchmarks, and datasets - not a training library: it ships no code, no trainer classes, and nothing to `pip install`.

**Awesome-Multimodal-Large-Language-Models** is the companion repository to the survey "A Survey on Multimodal Large Language Models" and its follow-ups; its own description reads "Latest Advances on Multimodal Large Language Models" [1]. It is maintained under the GitHub account BradyFU, with the most recent commit authored by contributor xjtupanda [2]. Its "API" is a single long README organized as markdown tables of paper title, venue, date, and links to code/demo, grouped under headings such as Multimodal Instruction Tuning, Multimodal Hallucination, Multimodal RLHF, and several Awesome Datasets sections [3]. It lives at https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models [1].

**When to pick it**: pick this when you want a survey-maintained index of papers, benchmarks, and datasets for multimodal LLMs to read or cite - not when you want a library to run post-training with. It contains no trainer, no Config class, no Python package, and no CLI; every entry is an external link out to a paper, a separate GitHub repo, or a demo page [3]. Because it ships no runnable code of its own, it cannot be compared to trainer-class libraries like trl or verl on scale, launcher, or memory handling - those fields do not apply here.

**Methods it ships**: none - this repository ships no method implementations. It carries an index of papers under a "Multimodal RLHF" heading, including MM-RLHF, RLHF-V, LLaVA-RLHF (Factually Augmented RLHF), Silkie/VLFeedback, RoVRM, and R1-Reward, each linking to that paper's own arXiv page and, where one exists, its own separate code repository [3][4]. The repository tree at this commit contains an `images/mm-rlhf.jpg` file, but a full-text search of the README at this commit finds no line that references that filename or embeds that image - only the unrelated text "MM-RLHF" as a paper title in the Multimodal RLHF table [3]; the pipeline's method-detection hit on that image path does not correspond to any method this repository trains or documents.

**Scale it handles**: not applicable - there is no training code, launcher, or sharding option in this repository to scale [3].

**Install**: there is no package to install. The repository has no `pyproject.toml`, `setup.py`, or PyPI listing surfaced by this card's research; using it means cloning or browsing the README on GitHub. The repository carries no license file, and the GitHub API's license endpoint for it returns "Not Found" [5], so no license is stated.

**Maintained by**: BradyFU (GitHub account) [1], with the repository's most recent push authored by contributor xjtupanda on 2026-08-03 [2], the commit this card cites throughout. The README's own top section is actively updated with dated entries, the most recent being a 2026-08-03 addition [3].

## Quick start

Not applicable. There is no code to run, no model, and no dataset to train against - the repository is a markdown index of external links [3].

## Start it

Not applicable, for the same reason: no launcher, no Config surface, and no GPU/OOM knobs exist in this repository [3].

## Watch it

Not applicable: no training loop, no logger, and no metrics are produced by this repository [3].

## Save it

Not applicable: no checkpoints, adapters, or save/resume calls exist here [3].

## Find it in the docs

- The README itself is the only document; it lives at the repository root and is what renders on the repo's GitHub page [1][3].
- A table of contents near the top links to each section by markdown anchor (e.g. `#multimodal-rlhf`, `#datasets-of-multimodal-rlhf`) [3].
- Every paper row's own links are the actual runnable references: the "Code" column points to that paper's separate GitHub repository when one exists, and the "Demo" column to a hosted demo, e.g. R1-Reward's own repo at github.com/yfzhang114/r1_reward or MM-RLHF's own repo at github.com/Kwai-YuanQi/MM-RLHF [3]. To run anything, follow one of those out-links to the paper's own repository and read that repository's own instructions - this card does not cover them.
- The repository also branches into topic-specific companion trees referenced from the README, such as a `Benchmarks` branch for the MME survey and an `Alignment` branch for the human-preference survey [3]; these are separate branches of the same repo, not separate packages.
- Honest boundary: this is a paper/benchmark/dataset index, not a framework a reader can install, launch, or checkpoint from. A reader looking for a runnable post-training library should treat every row here as a pointer to go read elsewhere, not as an endpoint.

## Sources

[1] BradyFU/Awesome-Multimodal-Large-Language-Models repository home page and description. https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models (metadata such as `description` and `html_url` cross-checked via the GitHub API at https://api.github.com/repos/BradyFU/Awesome-Multimodal-Large-Language-Models). Fetched 2026-08-10.

[2] Commit 5491326535854766073d87d2d40272b5a3c3b544 metadata via GitHub API (author xjtupanda, dated 2026-08-03). https://api.github.com/repos/BradyFU/Awesome-Multimodal-Large-Language-Models/commits/5491326535854766073d87d2d40272b5a3c3b544. Fetched 2026-08-10.

[3] Repository README pinned to commit 5491326535854766073d87d2d40272b5a3c3b544. https://raw.githubusercontent.com/BradyFU/Awesome-Multimodal-Large-Language-Models/5491326535854766073d87d2d40272b5a3c3b544/README.md. Fetched 2026-08-10.

[4] Individual paper entries under the README's Multimodal RLHF section (MM-RLHF, RLHF-V, LLaVA-RLHF, Silkie/VLFeedback, RoVRM, R1-Reward, and "Aligning Multimodal LLM with Human Preference: A Survey"), same source as [3]. Fetched 2026-08-10.

[5] License lookup for the repository via GitHub API, returning "Not Found". https://api.github.com/repos/BradyFU/Awesome-Multimodal-Large-Language-Models/license. Fetched 2026-08-10.
