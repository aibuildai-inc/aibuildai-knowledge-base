<h1 align="center">AIBuildAI Knowledge Base</h1>

---

<p align="center">
  <a href="https://join.slack.com/t/aibuildaicommunity/shared_invite/zt-4a40v9kus-bQr~NIAZSKJkTwYxj5Elww"><img src="https://img.shields.io/badge/Slack-AIBuildAI%20Community-4A154B?logo=slack" alt="Slack"></a>
  <a href="https://discord.gg/JWrbhmkV6k"><img src="https://img.shields.io/badge/Discord-AIBuildAI%20Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

---

The **AIBuildAI Knowledge Base** is a curated library of knowledge for building AI models of every kind, from computer vision and scientific AI to large language models. When an AI agent faces a design decision, such as which model architecture fits the task, which dataset to use, which training method to apply, or how to run a framework, it queries the knowledge base over MCP and reads what has worked, what has failed, and under which conditions. This repository is its source code: the retrieval service, the skill corpus, and the offline pipelines that build the corpus.

The service only retrieves. It returns material, never decisions, and the calling agent designs from what it reads, so building the index needs no LLM, only text splitting and embedding. A fixed snapshot of this corpus is served publicly at `https://32.194.230.84/open`, the default knowledge base address of AIBuildAI agents.

## The corpus

The knowledge base holds one kind of knowledge, the **skill**: a `SKILL.md` with a `description` in its frontmatter and a markdown body, plus the reference files it carries, such as knowledge cards, paper summaries, modeling principles and code examples. A skill lives at `data/corpus/plugins/<marketplace>/<plugin>/skills/<skill>/`, and that triple is its address.

The corpus covers AI model building broadly, and post-training large language models is one area it treats in depth: a dedicated collection walks an agent through a post-training run, from choosing training data and a method to running it on a framework.

| Plugin | Skills | Reference files | What it covers |
|---|---:|---:|---|
| `aibuildai-awesome-lists` | 460 | 110,640 | Domain knowledge from curated awesome lists: papers and tools per research area |
| `aibuildai-playbooks` | 44 | 4,004 | Domain playbooks distilled from top solution write-ups for AI modeling tasks |
| `aibuildai-huggingface` | 11 | 59 | The Hugging Face ecosystem: Hub datasets and models, trainers, evaluations, Gradio apps |
| `aibuildai-posttrain` | 4 | 479 | Post-training a base language model: the workflow of one run, training data, methods, and frameworks |
| `aibuildai-github` | 1 | 9 | Repositories for building AI models for molecular property prediction and drug discovery |

The 460 awesome-list skills span the major areas of AI:

| Area | Skills | Reference files |
|---|---:|---:|
| Computer Vision | 88 | 19,614 |
| Learning Paradigms & Architectures | 60 | 15,676 |
| Multimodal & Embodied AI | 58 | 12,359 |
| Large Language Models, NLP & Speech | 52 | 8,617 |
| Graphs, Time Series & Tabular Data | 44 | 10,919 |
| Generative Models | 41 | 17,104 |
| Efficient AI & ML Systems | 31 | 5,873 |
| AI for Science & Healthcare | 24 | 4,440 |
| Trustworthy AI | 22 | 5,791 |
| Reinforcement Learning | 17 | 3,844 |
| AI Agents & Code | 10 | 4,472 |
| Others | 13 | 1,931 |

## Quick start

The public service at `https://32.194.230.84/open` serves this corpus, so an agent can use the knowledge base without hosting anything.

**AIBuildAI agents.** Mount it in the run config. The [AIBuildAI LLM-Post-Train Agent](https://github.com/aibuildai-inc/aibuildai-llm-posttrain-agent) connects to the public service by default:

```yaml
mcps:
  kb: { type: reference }
```

Agents search it most when they are told to: say so in `llm.system_instructions`, as that repository's examples do.

**Any MCP client.** The service speaks MCP over Streamable HTTP at `https://32.194.230.84/open/mcp`. For example, in Claude Code:

```bash
claude mcp add --transport http kb https://32.194.230.84/open/mcp
```

The agent then gets four tools, named `mcp__kb__*` when the server is mounted as `kb`: `list_plugin` lists the plugins, `search_skills` searches the corpus, `load_skill` returns a skill with its files, and `read_reference` returns one file of a skill.

## Host your own knowledge base (optional)

Build the index and serve it yourself to run on a private network or to serve a corpus you have extended.

```bash
git clone https://github.com/aibuildai-inc/aibuildai-knowledge-base.git && cd aibuildai-knowledge-base
python3.11 -m venv .venv && source .venv/bin/activate
pip install '.[server]'

export AIBUILDAI_KB_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
RUNTIME_ROOT="$(PYTHONPATH=$PWD python -c 'from sourcing.utils.work_dir import runtime_root; print(runtime_root())')"

# build the search index, about 20 minutes for the full corpus on one GPU
python -m aibuildai_mcp.index --plugins-root data/corpus/plugins \
    --index-dir "$RUNTIME_ROOT/index" --model "$AIBUILDAI_KB_EMBED_MODEL"

# serve it at http://127.0.0.1:8000/mcp, with a health check at /healthz
AIBUILDAI_KB_DEVICE=cuda:0 bash deploy/serve_local.sh
```

The index build is incremental: it re-embeds only files whose content changed, and `--force` rebuilds everything. Restart the service after the corpus or the index changes. To point an AIBuildAI agent at your own service, set `AIBUILDAI_KB_BASE_URL` to its address; any other MCP client connects to its `/mcp` endpoint. The service has no authentication, so put it behind your own proxy if it must not be public.

**Docker.** `deploy/deploy.sh` builds the index, bakes it and the embedding model into an image, starts the container `aibuildai-kb` on port 8000, and checks that it answers a search. It needs Docker with the NVIDIA Container Toolkit.

```bash
CUDA_VISIBLE_DEVICES=0 AIBUILDAI_KB_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B bash deploy/deploy.sh
```

| Environment variable | Meaning |
|---|---|
| `AIBUILDAI_KB_EMBED_MODEL` | The embedding model ID |
| `AIBUILDAI_KB_DEVICE` | The torch device for the local service, such as `cuda:0` |
| `AIBUILDAI_KB_RUNTIME_DIR` | Where the index and pipeline outputs go; `runtime/` at the repository root by default |
| `AIBUILDAI_KB_INDEX_BATCH_SIZE` | Chunks embedded per model call, 32 by default; lower it after a CUDA out-of-memory error |

## Growing the corpus

`sourcing/` holds the offline pipelines that write corpus content. They call Claude through the Claude Agent SDK and need a different MCP version than the service, so install them in a separate environment with `pip install '.[producers]'`. See [sourcing/README.md](sourcing/README.md) for each pipeline.

## Project layout

```
aibuildai_mcp/        the retrieval service: index.py builds the index, service.py serves it
data/corpus/plugins/  the skill corpus
sourcing/             offline pipelines that write corpus content
deploy/               Dockerfile, deploy.sh, serve_local.sh
check.sh              pyright and ruff
```

## Citation

If you use this knowledge base in your work, please cite:

```bibtex
@article{zhang2026aibuildai2,
    title={AIBuildAI-2: A Knowledge-Enhanced Agent for Automatically Building AI Models},
    author={Ruiyi Zhang and Peijia Qin and Qi Cao and Li Zhang and Pengtao Xie},
    year={2026},
    journal={arXiv},
    url={https://arxiv.org/abs/2605.27873}
}
```

## License

[Apache License 2.0](LICENSE)

## Community

Questions and discussion: join the [AIBuildAI Community Slack](https://join.slack.com/t/aibuildaicommunity/shared_invite/zt-4a40v9kus-bQr~NIAZSKJkTwYxj5Elww) or the [AIBuildAI Community Discord](https://discord.gg/JWrbhmkV6k). Use the issue tracker for bugs and feature requests.
