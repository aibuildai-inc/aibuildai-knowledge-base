# Personal data and reuse terms in legal source documents

Read before building tasks from court records or filed agreements, and before publishing any synthesized set. Legal documents are public and are also full of named people: litigants, employees, witnesses, defendants. "Public" answers the copyright question in many places. It does not answer the privacy one.

## What the sources themselves do about personal data

| Source | Rule | Where |
|---|---|---|
| U.S. federal court filings | a filing "may include only: (1) the last four digits of the social-security number and taxpayer-identification number; (2) the year of the individual's birth; (3) the minor's initials; and (4) the last four digits of the financial-account number" | Fed. R. Civ. P. 5.2(a) [1] |
| U.S. federal criminal files | "Social Security numbers, names of minor children, financial account numbers, dates of birth, and, in criminal cases, home addresses" are redacted; some documents, such as presentence reports and sealed plea agreements, are not in the public file at all | Judicial Conference privacy policy [2] |
| U.S. Social Security and immigration cases | remote access for non-parties is limited to the docket and the court's dispositions | Fed. R. Civ. P. 5.2(c) [1] |
| EDGAR exhibits | registrants may redact details whose disclosure would be "a clearly unwarranted invasion of personal privacy (e.g., disclosure of bank account numbers, social security numbers, home addresses, and similar information)" | Regulation S-K Item 601(a)(6) [3] |
| UK judgments | the Open Justice Licence is "not a data sharing agreement for personal data"; users must "comply with any judicial decision that restricts the use of personal data" | Open Justice Licence v2.0 [4] |
| Chinese judgments | the Supreme People's Court cut back online publication, citing among its reasons that "Some labor dispute litigants have been repeatedly rejected in job applications" | statement of 2023-12-22 [5] |

Redaction rules are applied by the filer, and filers make mistakes. The rules also leave names in. An opinion or an employment agreement still identifies the person it concerns.

## What follows for synthesized tasks

- **A task built from a real filing carries the real names into training data and into anything the model later generates.** The Chinese court's stated concern is precisely that republished judgments followed people into job applications [5].
- **Published benchmarks avoid the problem by inventing the people.** Harvey LAB requires synthetic people, companies, law firms, addresses and matter facts (`legal-dataset`: `harveyai__harvey-labs.md`).
- **Changing names also blocks memorised answers** (`decontamination-and-splits.md`, rule 7). One step serves both purposes.

## Reuse terms, in one table

Each source card has the detail. This is the summary to check a plan against.

| Source | Copyright or licence, as stated | Limit to notice |
|---|---|---|
| U.S. federal statutes, regulations, opinions | no copyright in U.S. Government works; edicts of government are not copyrightable (`us-federal-law.md`) | unofficial copies are not the legal edition |
| EDGAR exhibits | SEC states access rules, no licence; exhibits are written by filers (`sec-edgar.md`) | 10 requests a second, declared user agent |
| CourtListener bulk data | "free of known copyright restrictions" (`courtlistener-recap.md`) | API rate limits; RECAP is a paid-for sample |
| EU law | Decision 2011/833/EU permits reuse of Commission documents with the source acknowledged and the meaning not distorted (`eur-lex.md`) | EUR-Lex's own notice was not read |
| UK legislation | Open Government Licence v3.0 (`uk-legislation-and-case-law.md`) | |
| UK judgments | Open Justice Licence v2.0 | computational analysis excluded without a further licence |
| Chinese laws and judicial documents | outside copyright under Article 5 (`china-law-and-judgments.md`) | publication has been cut back; crawling named as a harm |
| Australian federal legislation | CC BY 4.0 with set attribution (`other-jurisdictions.md`) | AustLII blocks automated collection of case law |
| Indian Supreme Court judgments (AWS) | CC-BY-4.0 | |

Model-written parts add the generator's terms on top of the source's (`model-written-tasks.md`).

## What to do

1. For tasks built from court filings or agreements about named individuals, replace names, addresses and identifying numbers with invented ones, consistently across every document in the set, and keep the mapping out of the dataset.
2. Leave real names where the name is the law: case captions used as citations, judges as authors of opinions, parties in published appellate decisions cited as authority.
3. Do not build tasks from sealed, expunged or restricted-access material, even where a copy has surfaced.
4. Keep a per-item licence field with the source's terms in its own words, and a per-item generator field.
5. Before publishing a synthesized set, list which sources allow redistribution and under what attribution. Train-only use and publication are different questions.

## Sources

Every source was read on 2026-10-03.

[1] Federal Rule of Civil Procedure 5.2, read in the Legal Information Institute's copy. https://www.law.cornell.edu/rules/frcp/rule_5.2

[2] United States Courts, "Privacy Policy for Electronic Case Files". https://www.uscourts.gov/rules-policies/judiciary-policies/privacy-policy-electronic-case-files

[3] 17 CFR 229.601, read in the Legal Information Institute's copy. https://www.law.cornell.edu/cfr/text/17/229.601

[4] The National Archives, "Open Justice Licence", version 2.0. https://caselaw.nationalarchives.gov.uk/open-justice-licence

[5] Pekingnology, English translation of the Supreme People's Court's question-and-answer statement of 2023-12-22. https://www.pekingnology.com/p/chinas-highest-court-defends-restricting
