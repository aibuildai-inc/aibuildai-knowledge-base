# Upstream Provenance

Skills under `skills/` are copied verbatim from https://github.com/huggingface/skills

- Upstream commit: `061ab494cb145f43ae8f218939b99160e2c61c58`
- Upstream commit date: 2026-04-16
- Copied at: 2026-04-17
- License: Apache-2.0

Repository layout diverges from upstream: individual skill dirs are placed
under `skills/` (matching the project's `<marketplace>/<plugin>/skills/<skill>/`
convention); upstream ships multi-agent manifests (`.claude-plugin/`,
`.cursor-plugin/`, `gemini-extension.json`, `hf-mcp/`, `apps/`) which are not
mirrored here — these HF skills are read by the generic `plugin-subagent`
(absorptive), not by a dedicated sub-agent in this plugin.

## Re-sync procedure

```bash
git clone --depth 1 https://github.com/huggingface/skills.git /tmp/hf-skills-new
rsync -a --delete /tmp/hf-skills-new/skills/ ./skills/
rm -rf /tmp/hf-skills-new
# Then update the commit hash above.
```
