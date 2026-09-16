"""Read the skill corpus. Build, load, and search its txtai index."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath

import semchunk
import torch
from txtai import Embeddings

_CORPUS_HASH_FILE = "corpus.sha256"
_AT_FDCWD = -100
_RENAME_EXCHANGE = 2
_RENAMEAT2 = ctypes.CDLL(None, use_errno=True).renameat2
_RENAMEAT2.argtypes = [
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_uint,
]
_RENAMEAT2.restype = ctypes.c_int


Address = tuple[str, str, str]


def _address(relpath: str) -> Address:
    """Return the marketplace, plugin, and skill of a corpus-relative path."""
    parts = PurePosixPath(relpath).parts
    return parts[0], parts[1], parts[3]


def _digest_skill(digest: "hashlib._Hash", address: Address, files: dict[str, bytes]) -> None:
    """Add one skill document's paths and bytes to a running hash."""
    marketplace, plugin, skill = address
    for path, data in files.items():
        digest.update(f"{marketplace}/{plugin}/skills/{skill}/{path}".encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")


class SkillCorpus:
    """Read skill files from the corpus tree.

    A skill document is one directory holding a ``SKILL.md``, at
    ``<marketplace>/<plugin>/skills/<skill>/``, and its address is those three
    names. :meth:`skill_dirs` is the one answer to which documents this corpus
    holds and what each one's address is; indexing, serving, and hashing all
    read the corpus through it.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._loaded: dict[Address, dict[str, str]] | None = None
        self._hash: str | None = None

    def skill_dirs(self) -> list[tuple[Address, Path]]:
        """Return every skill document's address and directory, in path order.

        A dot-prefixed marketplace or plugin is a publish work directory and is
        not part of the corpus. Anything else directly under a plugin's
        ``skills/`` that is not a skill document is a corpus error.
        """
        found: list[tuple[Address, Path]] = []
        for marketplace in sorted(self.root.iterdir()):
            if not marketplace.is_dir() or marketplace.name.startswith("."):
                continue
            for plugin in sorted(marketplace.iterdir()):
                if not plugin.is_dir() or plugin.name.startswith("."):
                    continue
                skills = plugin / "skills"
                if not skills.is_dir():
                    continue
                for entry in sorted(skills.iterdir()):
                    if not entry.is_dir() or not (entry / "SKILL.md").is_file():
                        raise ValueError(
                            "not a skill document: "
                            f"{entry.relative_to(self.root).as_posix()}"
                        )
                    relpath = entry.relative_to(self.root).as_posix()
                    found.append((_address(relpath), entry))
        return found

    def plugins(self) -> list[dict]:
        """Return every plugin's marketplace, name, description, and skill count.

        The description is the ``description`` field of the plugin's
        ``.claude-plugin/plugin.json``; every plugin must carry one.
        """
        counts: dict[tuple[str, str], int] = {}
        for (marketplace, plugin, _), _dir in self.skill_dirs():
            counts[(marketplace, plugin)] = counts.get((marketplace, plugin), 0) + 1
        return [
            {
                "marketplace": marketplace,
                "plugin": plugin,
                "description": json.loads(
                    (self.root / marketplace / plugin / ".claude-plugin" / "plugin.json")
                    .read_text()
                )["description"],
                "skills": count,
            }
            for (marketplace, plugin), count in sorted(counts.items())
        ]

    def search_scope(self, marketplace: str | None, plugin: str | None) -> str | None:
        """Return the source-path prefix a scoped search filters on, or None.

        No name given: None, the global search. Only ``marketplace``: that
        marketplace. Only ``plugin``: the one marketplace holding it. Both: the
        pair must exist together. An unknown name or a mismatched pair raises
        ``ValueError`` so the caller's tool call fails with the reason.
        """
        if marketplace is None and plugin is None:
            return None
        addresses = {(m, p) for (m, p, _), _ in self.skill_dirs()}
        if plugin is None:
            if marketplace not in {m for m, _ in addresses}:
                raise ValueError(
                    f"unknown marketplace {marketplace!r}: call list_plugin "
                    "for the catalog"
                )
            return f"{marketplace}/"
        holders = sorted(m for m, p in addresses if p == plugin)
        if not holders:
            raise ValueError(
                f"unknown plugin {plugin!r}: call list_plugin for the catalog"
            )
        if marketplace is None:
            if len(holders) > 1:
                raise ValueError(
                    f"plugin {plugin!r} exists in more than one marketplace "
                    f"({', '.join(holders)}): pass marketplace too"
                )
        elif marketplace not in holders:
            raise ValueError(
                f"plugin {plugin!r} is not in marketplace {marketplace!r}: "
                f"it lives in {', '.join(holders)}"
            )
        return f"{marketplace or holders[0]}/{plugin}/"

    def iter_files(self) -> Iterator[tuple[Path, str]]:
        """Yield each Markdown file and its corpus-relative path."""
        for _, skill_dir in self.skill_dirs():
            for md in sorted(skill_dir.rglob("*.md")):
                yield md, md.relative_to(self.root).as_posix()

    def _read(self) -> Iterator[tuple[Address, dict[str, bytes]]]:
        """Yield each skill document's address and its files' bytes."""
        for address, skill_dir in self.skill_dirs():
            yield address, {
                path.relative_to(skill_dir).as_posix(): path.read_bytes()
                for path in sorted(
                    path for path in skill_dir.rglob("*") if path.is_file()
                )
            }

    def load_all(self) -> None:
        """Keep one read-only corpus version in memory."""
        loaded: dict[Address, dict[str, str]] = {}
        digest = hashlib.sha256()
        for address, files in self._read():
            _digest_skill(digest, address, files)
            loaded[address] = {
                path: data.decode("utf-8", errors="ignore")
                for path, data in files.items()
            }
        self._loaded = loaded
        self._hash = digest.hexdigest()

    def corpus_hash(self) -> str:
        """Hash every file this corpus holds."""
        if self._hash is None:
            digest = hashlib.sha256()
            for address, files in self._read():
                _digest_skill(digest, address, files)
            self._hash = digest.hexdigest()
        return self._hash

    def _files(self, address: Address) -> dict[str, str]:
        if self._loaded is None:
            raise RuntimeError("load the skill corpus before reading a skill")
        files = self._loaded.get(address)
        if files is None:
            raise ValueError(f"unknown skill: {'/'.join(address)}")
        return files

    def load_skill(self, marketplace: str, plugin: str, skill: str) -> dict:
        """Read one skill body and list every other file the skill carries."""
        files = self._files((marketplace, plugin, skill))
        return {
            "marketplace": marketplace,
            "plugin": plugin,
            "skill": skill,
            "body": files["SKILL.md"],
            "files": sorted(path for path in files if path != "SKILL.md"),
        }

    def read_reference(
        self,
        marketplace: str,
        plugin: str,
        skill: str,
        path: str,
    ) -> dict:
        """Read one file that belongs to a skill."""
        address = (marketplace, plugin, skill)
        files = self._files(address)
        text = files.get(path)
        if text is None:
            raise ValueError(
                f"file not found: {path!r} in skill {'/'.join(address)}"
            )
        return {
            "marketplace": marketplace,
            "plugin": plugin,
            "skill": skill,
            "path": path,
            "text": text,
        }


def _exchange_dirs(first: Path, second: Path) -> None:
    """Swap two directories in one file-system operation."""
    result = _RENAMEAT2(
        _AT_FDCWD,
        os.fsencode(first),
        _AT_FDCWD,
        os.fsencode(second),
        _RENAME_EXCHANGE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{first} <-> {second}")


def _publish_dir(staging: Path, out: Path) -> None:
    """Replace a directory without exposing a partly written version."""
    if not out.exists():
        os.replace(staging, out)
        return
    _exchange_dirs(staging, out)
    shutil.rmtree(staging, ignore_errors=True)


def _gpu(device: str | None) -> bool | int:
    """Translate a torch device name to txtai's GPU setting."""
    if device is None:
        return True
    if device == "cpu":
        return False
    if device == "cuda":
        index = 0
    elif device.startswith("cuda:") and device[5:].isdigit():
        index = int(device[5:])
    else:
        raise ValueError("device must be cpu, cuda, or cuda:<number>")
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} is not ready")
    return index


class SkillIndex:
    """Search the skill passages stored by txtai."""

    CHUNK_MAX_TOKENS = 512
    CHUNK_OVERLAP_TOKENS = 64

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        scope: str | None = None,
    ) -> list[dict]:
        """Search all passages, or only those whose source starts with *scope*.

        A scoped search compares the query against every indexed passage before
        the SQL filter keeps the scope, so its top_k is the best of the scope,
        not a filtered slice of the global top.
        """
        if top_k <= 0 or not self.embeddings.count():
            return []
        if scope is None:
            depth = min(max(top_k * 20, 200), self.embeddings.count())
            sql = "select text, source, score from txtai where similar(:query)"
            parameters = {"query": query}
        else:
            depth = self.embeddings.count()
            sql = (
                f"select text, source, score from txtai "
                f"where similar(:query, {depth}) and source like :scope"
            )
            parameters = {"query": query, "scope": f"{scope}%"}
        rows = self.embeddings.search(sql, depth, parameters=parameters)
        found: list[dict] = []
        sources: set[str] = set()
        for row in rows:
            if row["source"] not in sources:
                sources.add(row["source"])
                marketplace, plugin, skill = _address(row["source"])
                found.append(
                    {
                        "text": row["text"],
                        "marketplace": marketplace,
                        "plugin": plugin,
                        "skill": skill,
                        "path": "/".join(PurePosixPath(row["source"]).parts[4:]),
                        "score": round(float(row["score"]), 6),
                    }
                )
            if len(found) == top_k:
                break
        return found

    @classmethod
    def _config(
        cls,
        embed_model: str,
        device: str | None,
        batch_size: int,
    ) -> dict:
        return {
            "path": embed_model,
            "content": True,
            "objects": True,
            "gpu": _gpu(device),
            "encodebatch": batch_size,
            "maxlength": cls.CHUNK_MAX_TOKENS,
            "chunk_overlap_tokens": cls.CHUNK_OVERLAP_TOKENS,
        }

    @classmethod
    def build(
        cls,
        index_dir: str | Path,
        corpus: SkillCorpus,
        *,
        embed_model: str,
        token_counter: Callable[[str], int],
        device: str | None,
        batch_size: int,
        reuse: bool,
    ) -> dict:
        """Save changed corpus files in txtai and remove files that disappeared."""
        index_dir = Path(index_dir)
        config = cls._config(embed_model, device, batch_size)
        corpus_hash = corpus.corpus_hash()
        embeddings = Embeddings()
        existing: dict[str, dict] = {}
        loaded = False
        if reuse and embeddings.exists(str(index_dir)):
            embeddings.load(
                str(index_dir),
                config={"gpu": config["gpu"], "encodebatch": batch_size},
            )
            if all(
                embeddings.config.get(key) == value
                for key, value in config.items()
                if key not in {"gpu", "encodebatch"}
            ):
                for row in embeddings.search(
                    "select id, source, doc_hash from txtai",
                    embeddings.count(),
                ):
                    entry = existing.setdefault(
                        row["source"],
                        {"doc_hash": row["doc_hash"], "ids": []},
                    )
                    entry["ids"].append(row["id"])
                loaded = True
            else:
                embeddings.close()
                embeddings = Embeddings(config)
        else:
            embeddings = Embeddings(config)

        chunker = semchunk.chunkerify(token_counter, cls.CHUNK_MAX_TOKENS)
        current_sources: set[str] = set()
        changed_ids: list[str] = []
        documents: list[tuple[str, dict, None]] = []
        files = reused_files = embedded_files = 0
        for md, relpath in corpus.iter_files():
            files += 1
            current_sources.add(relpath)
            with md.open("rb") as source:
                doc_hash = hashlib.file_digest(source, "sha256").hexdigest()
            saved = existing.get(relpath)
            if saved is not None and saved["doc_hash"] == doc_hash:
                reused_files += 1
                continue
            embedded_files += 1
            if saved is not None:
                changed_ids.extend(saved["ids"])
            text = md.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            for part, piece in enumerate(
                chunker(text, overlap=cls.CHUNK_OVERLAP_TOKENS)
            ):
                documents.append(
                    (
                        f"{relpath}#{part}",
                        {
                            "text": piece,
                            "source": relpath,
                            "doc_hash": doc_hash,
                        },
                        None,
                    )
                )

        for source, saved in existing.items():
            if source not in current_sources:
                changed_ids.extend(saved["ids"])

        if changed_ids:
            embeddings.delete(changed_ids)
        if documents:
            embeddings.upsert(documents)

        stats = {
            "files": files,
            "chunks": embeddings.count(),
            "reused_files": reused_files,
            "embedded_files": embedded_files,
            "embedded_chunks": len(documents),
        }
        saved_hash_path = index_dir / _CORPUS_HASH_FILE
        saved_hash = (
            saved_hash_path.read_text(encoding="utf-8").strip()
            if saved_hash_path.is_file()
            else None
        )
        if (
            loaded
            and not changed_ids
            and not documents
            and not embedded_files
            and saved_hash == corpus_hash
        ):
            return stats

        index_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(dir=index_dir.parent, prefix=f"{index_dir.name}.tmp-")
        )
        try:
            embeddings.save(str(staging))
            (staging / _CORPUS_HASH_FILE).write_text(
                f"{corpus_hash}\n",
                encoding="utf-8",
            )
            _publish_dir(staging, index_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return stats

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        *,
        embed_model: str,
        device: str | None,
        batch_size: int,
    ) -> SkillIndex:
        config = cls._config(embed_model, device, batch_size)
        embeddings = Embeddings()
        embeddings.load(
            str(index_dir),
            config={"gpu": config["gpu"], "encodebatch": batch_size},
        )
        for key in ("path", "maxlength", "chunk_overlap_tokens"):
            if embeddings.config.get(key) != config[key]:
                raise RuntimeError(
                    f"kb index {key} does not match the service; rebuild the index"
                )
        return cls(embeddings)

    @staticmethod
    def exists(index_dir: str | Path) -> bool:
        return Embeddings().exists(str(index_dir))

    @staticmethod
    def verify_corpus(index_dir: str | Path, corpus: SkillCorpus) -> None:
        """Stop when the search index belongs to a different corpus."""
        saved_hash_path = Path(index_dir, _CORPUS_HASH_FILE)
        if not saved_hash_path.is_file():
            raise RuntimeError("kb index has no corpus hash; rebuild the index")
        saved_hash = saved_hash_path.read_text(encoding="utf-8").strip()
        if saved_hash != corpus.corpus_hash():
            raise RuntimeError("kb index does not match the corpus; rebuild the index")

    @staticmethod
    def model(index_dir: str | Path) -> str:
        config = json.loads(Path(index_dir, "config.json").read_text(encoding="utf-8"))
        return config["path"]


def main(argv: list[str] | None = None) -> None:
    """Build the index from explicit command-line settings."""
    parser = argparse.ArgumentParser(
        prog="aibuildai_mcp.index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--plugins-root",
        type=Path,
        required=True,
        default=argparse.SUPPRESS,
        help="Folder that contains marketplace and plugin folders",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        default=argparse.SUPPRESS,
        help="Folder to write or update",
    )
    parser.add_argument(
        "--model",
        required=True,
        default=argparse.SUPPRESS,
        help="Embedding model ID",
    )
    parser.add_argument(
        "--device",
        help="cpu, cuda, or cuda:<number>; omit to use a ready GPU",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Chunks embedded in one model call; without it the exported"
        " AIBUILDAI_KB_INDEX_BATCH_SIZE, and without that 32",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Embed the whole corpus and ignore the saved index",
    )
    args = parser.parse_args(argv)
    # Where the batch size came from, so the line printed below states the one
    # thing the reader can change, and a bad host value is refused the way a bad
    # flag is instead of raising out of the parser.
    declared = os.environ.get("AIBUILDAI_KB_INDEX_BATCH_SIZE", "").strip()
    if hasattr(args, "batch_size"):
        source = "--batch-size"
    elif not declared:
        args.batch_size, source = 32, "the built-in default"
    else:
        source = "AIBUILDAI_KB_INDEX_BATCH_SIZE"
        if not re.fullmatch(r"[+-]?[0-9]+", declared):
            parser.error(f"{source} is {declared!r}, which is not a whole number")
        args.batch_size = int(declared)
    if args.batch_size <= 0:
        parser.error(f"{source} gives {args.batch_size}; it must be greater than zero")
    try:
        _gpu(args.device)
    except ValueError as error:
        parser.error(str(error))

    print(
        f"embedding {args.batch_size} chunks of at most"
        f" {SkillIndex.CHUNK_MAX_TOKENS} tokens per call, from {source}"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stats = SkillIndex.build(
        args.index_dir,
        SkillCorpus(args.plugins_root),
        embed_model=args.model,
        token_counter=lambda text: len(tokenizer(text)["input_ids"]),
        device=args.device,
        batch_size=args.batch_size,
        reuse=not args.force,
    )
    print(f"skill index built: {stats}")


if __name__ == "__main__":
    main()
