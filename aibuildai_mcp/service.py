"""KbService: ONE MCP service mounting the retrieval tools over the Kb's skill
corpus, on an MCPServer streamable-HTTP shell it owns directly (the command-line serve
entry and the `/healthz` route live here).

Boundary: the service ONLY retrieves. Every tool returns the raw retrieved material
(a catalog row, a passage, a skill body, one of its files) and NEVER a decision -- it
never judges your candidates or chooses a model or design for you; a search hit's
relevance score is retrieval bookkeeping, not a recommendation. The calling Agent
is assumed strong: it generalizes, picks models, and designs from this material
itself.

The tool descriptions are the ONE place that teaches an agent how to use the Kb:
clients mount this service like any third-party MCP and follow the interface it
exposes at runtime, so a contract change lands here and must stay additive --
shipped clients cannot be edited.

  list_plugin()
      -> [{marketplace, plugin, description, skills}]
  search_skills(query, top_k=8, marketplace=None, plugin=None)
      -> [{text, marketplace, plugin, skill, path, score}]
  load_skill(marketplace, plugin, skill)
      -> {marketplace, plugin, skill, body, files}
  read_reference(marketplace, plugin, skill, path)
      -> {marketplace, plugin, skill, path, text}

A skill is one SKILL.md plus any other files it carries. Its address is the
triple (marketplace, plugin, skill); the tools that operate on one skill
(load_skill, read_reference) take all three together. The skill's directory is
`<marketplace>/<plugin>/skills/<skill>/` in the corpus, so two different
triples are always two different documents.

The index build stays OFFLINE (`python -m aibuildai_mcp.index`)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mcp.server import MCPServer  # pyright: ignore[reportAttributeAccessIssue]
from opentelemetry import trace
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from aibuildai_mcp.index import SkillCorpus, SkillIndex, _gpu

_TRACE_PROVIDER = TracerProvider(
    resource=Resource.create({"service.name": "aibuildai-mcp"})
)
_TRACE_PROVIDER.add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stderr))
)
trace.set_tracer_provider(_TRACE_PROVIDER)


def _configure_http_logging() -> None:
    """Keep warnings and let OpenTelemetry write request records."""
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# Agent-facing self-description. Explicit string literals (NOT docstrings): a
# Cython-compiled build strips docstrings, which would render a tool description
# empty in production; a literal survives.
_KB_INSTRUCTIONS = (
    "Knowledge base for ML pipeline building. Retrieve curated skill knowledge "
    "(list_plugin / search_skills / load_skill / read_reference). "
    "Every tool returns retrieved material, never a decision -- you generalize, "
    "choose models, and design from it yourself."
)

_LIST_PLUGIN_DESCRIPTION = (
    "List every plugin in the knowledge base. Takes no arguments. Returns "
    "[{marketplace, plugin, description, skills}]: description says what the "
    "plugin covers, skills is how many skills it holds. Call this to see what "
    "knowledge exists, then pass a plugin (or marketplace) name to search_skills "
    "to search inside just that scope."
)

_SEARCH_SKILLS_DESCRIPTION = (
    "Retrieve guidance from curated ML skills via semantic search. Search "
    "with a full description of the problem -- the task, the data "
    "shape, and the constraint -- not bare keywords. Returns up to top_k passages "
    "{text, marketplace, plugin, skill, path, score}: text is the full matched "
    "passage (ready to use, no file to open); marketplace, plugin and skill "
    "together are the address of the skill it came from, and path names the "
    "file inside that skill holding the passage -- SKILL.md for the body, or "
    "one of the skill's other files; read_reference returns any of them. To read "
    "the whole skill, pass the three address values to load_skill. By default "
    "every plugin is searched; pass marketplace and/or plugin (names from "
    "list_plugin) to search only that "
    "scope, so top_k is the best of the scope. An unknown name or a mismatched "
    "marketplace/plugin pair is an error. Prefer this over generic web search "
    "when you need guidance on building or training a model."
)

_LOAD_SKILL_DESCRIPTION = (
    "Load a complete curated skill. A skill's address is the three values "
    "marketplace, plugin and skill, exactly as one search_skills result returned "
    "them; pass all three, never guess an address or mix values from different "
    "results. Returns {marketplace, plugin, skill, body, files}: "
    "body is the full SKILL.md text, and files lists every other file the "
    "skill carries (identifiers such as 'references/cross-validation.md'). "
    "Fetch one with read_reference(marketplace, plugin, skill, path). No filesystem "
    "path is returned -- everything is content the server resolves for you."
)

_READ_REFERENCE_DESCRIPTION = (
    "Return the full content of one file belonging to a curated skill. marketplace, "
    "plugin and skill are that skill's address, the same three values you passed to "
    "load_skill. path is any file the skill holds: one of load_skill's 'files', "
    "or the path a search_skills hit named (SKILL.md included). "
    "Returns {marketplace, plugin, skill, path, text}. path is "
    "an identifier resolved server-side against the corpus, NOT a client filesystem "
    "path."
)


class KbService:
    """The Kb's single MCP service. Owns the MCPServer streamable-HTTP shell -- served
    STATELESS at its configured path (the read-only tools need no per-session state and
    stateless sessions survive horizontal scale) -- mounts the retrieval tools,
    and exposes the serve entry + `/healthz` route."""

    def __init__(
        self,
        *,
        plugins_root: str,
        index_dir: str,
        embed_model: str,
        embed_device: str | None,
        embed_batch_size: int,
        warm: bool,
    ) -> None:
        self.mcp = MCPServer(
            "kb",
            instructions=_KB_INSTRUCTIONS,
        )
        self._skill_corpus = SkillCorpus(plugins_root)
        self._skill_index_dir = Path(index_dir)
        self._skill_corpus.load_all()
        SkillIndex.verify_corpus(self._skill_index_dir, self._skill_corpus)
        self._embed_model = embed_model
        self._embed_device = embed_device
        self._embed_batch_size = embed_batch_size
        self._skill_index: "SkillIndex | None" = None

        @self.mcp.tool(description=_LIST_PLUGIN_DESCRIPTION)
        def list_plugin() -> list[dict]:
            return self._skill_corpus.plugins()

        @self.mcp.tool(description=_SEARCH_SKILLS_DESCRIPTION)
        def search_skills(
            query: str,
            top_k: int = 8,
            marketplace: str | None = None,
            plugin: str | None = None,
        ) -> list[dict]:
            scope = self._skill_corpus.search_scope(marketplace, plugin)
            return self._skill().search(query, top_k=top_k, scope=scope)

        @self.mcp.tool(description=_LOAD_SKILL_DESCRIPTION)
        def load_skill(marketplace: str, plugin: str, skill: str) -> dict:
            return self._skill_corpus.load_skill(marketplace, plugin, skill)

        @self.mcp.tool(description=_READ_REFERENCE_DESCRIPTION)
        def read_reference(
            marketplace: str,
            plugin: str,
            skill: str,
            path: str,
        ) -> dict:
            return self._skill_corpus.read_reference(
                marketplace,
                plugin,
                skill,
                path,
            )

        if warm:
            self._skill()

    # --- lazy index loading ------------------------------------------------- #
    def _skill(self) -> SkillIndex:
        if self._skill_index is None:
            loaded = SkillIndex.load(
                self._skill_index_dir,
                embed_model=self._embed_model,
                device=self._embed_device,
                batch_size=self._embed_batch_size,
            )
            SkillIndex.verify_corpus(self._skill_index_dir, self._skill_corpus)
            self._skill_index = loaded
        return self._skill_index

    # --- MCP HTTP shell: health, ASGI app, serve ---------------------------- #
    def app(
        self,
        *,
        healthz: bool,
        path: str = "/mcp",
        host: str = "127.0.0.1",
    ):  # noqa: ANN201 (Starlette ASGI app)
        """Build the Streamable-HTTP ASGI app. Every route serves openly."""
        if healthz:
            from starlette.responses import JSONResponse

            @self.mcp.custom_route("/healthz", methods=["GET"])
            async def _healthz(request):  # noqa: ANN001, ANN202
                return JSONResponse({"status": "ok"})

        _configure_http_logging()
        app = self.mcp.streamable_http_app(
            streamable_http_path=path,
            json_response=True,
            stateless_http=True,
            host=host,
        )
        app.add_middleware(
            OpenTelemetryMiddleware,
            tracer_provider=_TRACE_PROVIDER,
            exclude_spans=["receive", "send"],
        )
        return app

    def serve(
        self,
        *,
        transport: str,
        host: str,
        port: int,
        path: str,
        healthz: bool,
    ) -> None:
        """Serve over stdio or Streamable HTTP."""
        if transport == "stdio":
            self.mcp.run(transport="stdio")
            return
        import uvicorn

        uvicorn.run(
            self.app(healthz=healthz, path=path, host=host),
            host=host,
            port=port,
            log_config=None,
            log_level="warning",
        )


def main() -> None:
    """Serve the kb service from explicit command-line settings."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--plugins-root",
        required=True,
        default=argparse.SUPPRESS,
        help="Folder that contains marketplace and plugin folders",
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        default=argparse.SUPPRESS,
        help="Folder that contains the built index",
    )
    parser.add_argument(
        "--embed-model",
        required=True,
        default=argparse.SUPPRESS,
        help="Embedding model ID",
    )
    parser.add_argument(
        "--device",
        help="Torch device; omit to use a ready GPU or else the CPU",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="Query texts embedded in one model call",
    )
    parser.add_argument(
        "--transport",
        choices=("http", "stdio"),
        default="http",
        help="MCP transport",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="HTTP listen address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP listen port",
    )
    parser.add_argument(
        "--path",
        default="/mcp",
        help="Streamable HTTP path",
    )
    args = parser.parse_args()
    if args.transport not in {"http", "stdio"}:
        parser.error("--transport must be http or stdio")
    if args.embed_batch_size <= 0:
        parser.error("--embed-batch-size must be greater than zero")
    try:
        _gpu(args.device)
    except ValueError as error:
        parser.error(str(error))
    if args.port <= 0:
        parser.error("--port must be greater than zero")
    service = KbService(
        plugins_root=args.plugins_root,
        index_dir=args.index_dir,
        embed_model=args.embed_model,
        embed_device=args.device,
        embed_batch_size=args.embed_batch_size,
        warm=True,
    )
    service.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        path=args.path,
        healthz=True,
    )


if __name__ == "__main__":
    main()
