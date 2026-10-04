# Australia, Canada, India and the European Court of Human Rights

Four more sources, each with one fact that decides whether and how it can be used. Many other jurisdictions publish their law openly and are not covered here; check each one's terms the same way.

| Jurisdiction | Source | What the terms say | Route for synthesis |
|---|---|---|---|
| Australia, legislation | Federal Register of Legislation | "Creative Commons Attribution 4.0 International (the CC BY 4.0 licence)", with a set attribution: "Sourced from the Federal Register of Legislation at [full date of download]", or "Based on content from ..." for changed content [1] | open; store the download date, which the attribution requires |
| Australia, case law | AustLII | "AustLII specifically blocks all spiders and other automated agents from accessing its case-law via the Robots Exclusion Standard", and it "will not act as a data repository or re-supplier of source documents" [2] | not AustLII; use `isaacus/open-australian-legal-corpus` (`legal-dataset`), which records per-source terms |
| Canada | court and tribunal decisions | per court; in `a2aj/canadian-case-law` every sampled row's `upstream_license` includes non-commercial restrictions (`legal-dataset`) | the A2AJ dataset, with its per-row licence field. CanLII's terms page disallows automated readers and was not read |
| India | Supreme Court judgments on the AWS Open Data registry | "CC-BY-4.0"; judgments "from 1950 to 2025", "downloaded from ecourts website", with JSON and parquet metadata, updated "Bi-monthly", managed by Dattam Labs, in the S3 bucket `indian-supreme-court-judgments` [3] | open bulk download |
| Council of Europe | European Court of Human Rights site | texts "may be reproduced provided the source is acknowledged (© ECHR-CEDH)" for private use or for educational purposes related to the Court's activities; "certain information and texts may be protected under intellectual property law" [4] | read the terms against your use; LexGLUE's ECtHR tasks already package these judgments (`legal-dataset`) |

**Use it for**: tasks in a jurisdiction other than the United States, the EU, the UK and China, so that a training mix does not teach one country's law as "the law".

**Trap**: the easiest site to scrape is often the one that forbids it. AustLII blocks automated collection of case law and declines to resupply documents [2], while the same decisions are available from the courts under their own terms, and the Open Australian Legal Corpus has already collected them with those terms recorded.

## What to do

1. Look for a packaged, documented copy in `legal-dataset` before fetching anything.
2. For a new jurisdiction, find three things before writing a task: who publishes the official text, the reuse terms in their own words, and whether automated access is allowed.
3. Store jurisdiction and capture date on every task. A dataset that mixes jurisdictions without a field for it cannot be filtered to the target (`legal-dataset`: `finding-more.md`, stage 2b).

## Sources

Every source was read on 2026-10-03.

[1] Federal Register of Legislation, "Terms of use". https://www.legislation.gov.au/terms-of-use

[2] AustLII, copyright and usage policy page. https://www.austlii.edu.au/austlii/copyright.html

[3] Registry of Open Data on AWS, "Indian Supreme Court Judgments". https://registry.opendata.aws/indian-supreme-court-judgments/

[4] European Court of Human Rights, "Copyright and disclaimer". https://www.echr.coe.int/copyright-and-disclaimer
