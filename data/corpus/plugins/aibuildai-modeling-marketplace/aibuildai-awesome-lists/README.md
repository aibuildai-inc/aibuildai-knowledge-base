# aibuildai-awesome-lists

Academic domain knowledge bases sourced from curated GitHub awesome lists. Each skill is an indexed collection of papers and tools with structured references.

## Skills

| Skill | Entries | Domain |
|-------|---------|--------|
| `awesome-discrete-diffusion` | 46 | Discrete diffusion models |
| `awesome-knowledge-distillation` | 203 | Model compression |
| `awesome-multi-task-learning` | 191 | Multi-task architectures |
| `awesome-pretrain-on-molecules` | 169 | Molecular pre-training |
| `awesome-python-chemistry` | 151 | Computational chemistry tools |
| `awesome-cheminformatics` | 105 | Cheminformatics libraries |
| `awesome-small-molecule-ml` | 35 | Drug discovery ML |

## How It Works

Each knowledge base skill contains:
- `SKILL.md` — index table (title, year, citations, type, category, TLDR)
- `references/` — one markdown file per entry with full details

These knowledge bases are surfaced through the generic `plugin-subagent`: given a problem description, it scans the indices, deep-reads relevant entries, and returns a structured synthesis.

## Adding New Knowledge Bases

Use the knowledge pipeline:
```bash
python -m knowledge.cli run configs/awesome-<name>.yaml
```

Output goes directly to this plugin's `skills/` directory.

## License

MIT
