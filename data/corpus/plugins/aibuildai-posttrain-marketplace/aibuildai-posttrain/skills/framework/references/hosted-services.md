# Post-training that runs as a service

Some providers post-train a model for you. There is no repository to clone, no trainer class to read, and no published model that carries a library tag, so not one of the nine search channels behind this skill can see them. They can only be listed by hand.

## The bar for being listed here

A provider is listed only when its own API reference declares the training methods by name, and that reference can be quoted at a commit or a fetch date. A pricing page, a blog post, or a documentation site with no version is not enough. This is the same first-hand-source rule the rest of this plugin runs on.

## What this file can and cannot tell you

**No hosted provider has a card in this skill, and none was surveyed by the channels that built it.** That is a real gap, not a judgement that hosted training is a poor choice. The scope decision recorded while this skill was planned found exactly one provider that meets the bar: OpenAI, whose fine-tuning method names are declared in its own client repository `openai/openai-python`, in `api.md`. This file did not re-read that file, so it states where the source is and not what it currently says. Read it at run time before you plan around it.

The same decision recorded five providers that were checked and left out - Google Vertex, AWS Bedrock, Together, Fireworks, and Predibase - because each publishes no repository, no Hub tag, and no paper, so no first-hand source declares its methods. "Checked and excluded" is what that means. It does not mean the service cannot train a model; it means this skill cannot tell you which method it would run.

| Provider | What was checked | Why it is not listed |
|---|---|---|
| OpenAI | its own client repository `openai/openai-python`, `api.md` | meets the bar for a first-hand source; no card was written, and the file was not re-read for this skill |
| Google Vertex | repository search, Hub tag search, paper search | no first-hand source names its training methods |
| AWS Bedrock | repository search, Hub tag search, paper search | no first-hand source names its training methods |
| Together | repository search, Hub tag search, paper search | no first-hand source names its training methods |
| Fireworks | repository search, Hub tag search, paper search | no first-hand source names its training methods |
| Predibase | repository search, Hub tag search, paper search | no first-hand source names its training methods |

## What to do when a hosted service is the plan

Ask the four questions this file would answer if a provider were listed, and answer them from the provider's own reference before committing a run: which methods it declares by name, what data shape it accepts, what it hands back (a hosted model id, weights, or neither), and whether the evaluator can load that result. The fourth question is the one that ends runs late: a provider that returns only a hosted model id cannot hand a checkpoint to a local evaluator, and `references/loading-the-result.md` is then unusable, because there is nothing on disk to check.
