# China: statutes, judicial documents and the shrinking judgment record

Chinese law is not protected by copyright, and Chinese court judgments were for some years the largest open judgment collection in the world. Since 2021 far fewer have been published, and the court has named commercial crawling as one reason. A Chinese synthesis pipeline should start from the existing datasets, not from a new crawl.

**Holds**: China Judgments Online (CJO), "First built in 2013", grew to "more than 143 million verdicts" [1]. Statutes, regulations and judicial interpretations are published by the state.

**Terms**: Article 5 of the Copyright Law excludes "Laws, regulations, decisions, orders and other documents of legislative, administrative or judicial nature, and their official translations" from protection [2]. That settles copyright in the text. It does not settle access terms or the personal data inside judgments.

**Access, and why it is narrowing**: uploads fell sharply. One count has annual disclosed verdicts peaking at 23.3 million in 2020, and "In 2022, the total number declined 62%, to 8.9 million"; only 854 administrative cases were uploaded in 2022 against about 670,000 tried [1]. The Supreme People's Court's own figures, in a statement of 2023-12-22, are 19.2 million judgments published for 2020, 14.9 million for 2021, 10.4 million for 2022 and 5.11 million for 2023 up to the date of the statement [3]. The two sources count differently; both show the fall. The court's stated reasons included harm to litigants ("Some labor dispute litigants have been repeatedly rejected in job applications") and that "Some commercial companies have transformed 'crawled' document data into legal search, corporate credit, and AI 'products' for profit" [3]. A separate national judgment database was due online in January 2024 and "will only be accessible to internal staff" [1].

**Already packaged**: `china-ai-law-challenge/cail2018` (criminal judgments with charges, articles and terms), `ShengbinYue/DISC-Law-SFT` and `doolayer/LawBench` have cards in `legal-dataset`; JuDGE, CaseGen, LexEval, PLawBench, LexRubric and LegalAgentBench have grading cards in `legal-evaluation`.

**Use it for**: judgment-prediction tasks where the charge, the articles and the term are in the judgment itself (`labels-from-the-record.md`); document drafting where a real judgment section is the reference; statute application with the article text supplied.

**Feeds these benchmarks**: CAIL2018's training split feeds LawBench tasks 3-1, 3-3 and 3-4, and Chinese judicial-exam questions feed LawBench 1-2 and 3-6 and AGIEval JEC-QA-KD (`legal-dataset` hold-out table). A new task set drawn from the same criminal judgments can repeat those facts.

**Trap**: the public record after 2021 is a selected sample. Case types that stopped being uploaded (administrative cases above all) are nearly absent [1], so tasks synthesized from recent CJO text under-represent them, and a model trained on those tasks sees a skewed picture of practice.

## What to do

1. Start from the carded datasets and their documented splits.
2. Do not build a new crawler for CJO. The court has named commercial crawling as a harm [3], and the record is incomplete anyway.
3. Store the judgment date and the court on every item, and report the date range of the set.
4. Chinese judgments name parties. Follow `personal-data-and-terms.md`.

## Sources

Every source was read on 2026-10-03. The national law database (flk.npc.gov.cn) disallows automated readers and was not read.

[1] MIT Technology Review, "China's judicial system is becoming even more secretive" (2023-12-20). https://www.technologyreview.com/2023/12/20/1085741/china-judgements-online-transparency-government/

[2] Copyright Law of the People's Republic of China (as amended 2020-11-11), Article 5, English text at WIPO Lex. https://www.wipo.int/wipolex/en/legislation/details/21065

[3] Pekingnology, "China's highest court defends restricting online publication of effective judgments" (English translation of the Supreme People's Court's question-and-answer statement of 2023-12-22). https://www.pekingnology.com/p/chinas-highest-court-defends-restricting
