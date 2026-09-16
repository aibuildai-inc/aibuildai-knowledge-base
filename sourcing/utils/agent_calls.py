"""Run sourcing agents through the Claude Agent SDK and record what each one did."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    Message,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from pydantic import BaseModel

from sourcing.utils.work_dir import atomic_write_text

DEFAULT_BATCH_SIZE = 50
DEFAULT_WORKERS = 4
_LIVE_TRANSCRIPT_TOOLS = {"Agent", "Task", "Write", "Edit", "NotebookEdit"}
# Provider answers that a repeat of the same request cannot change: an invalid
# request, a failed login, a billing refusal, a permission refusal.
_PERMANENT_API_STATUSES = {400, 401, 402, 403}


class PermanentError(RuntimeError):
    """The provider refused the request for a reason that a retry cannot fix."""


def sum_costs(*costs: float | None) -> float | None:
    """An unknown component makes the total unknown, not zero."""
    return None if None in costs else sum(cost for cost in costs if cost is not None)


@contextmanager
def _preserve_failure(cleanup: Callable[[], None]) -> Iterator[None]:
    """Keep this operation's failure when its cleanup also fails."""
    failure: BaseException | None = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            cleanup()
        except BaseException as cleanup_error:
            if failure is not None and cleanup_error is not failure:
                raise failure from cleanup_error
            raise


def _rejected(reason: str) -> str:
    """The message that sends a rejected structured result back to its own session."""
    return (
        "Your submitted output has been REJECTED by the verification that runs "
        "AFTER submission. The earlier 'Structured output provided successfully' "
        "tool result only confirmed the schema; that submission is now void. "
        f"Fix the problem and call StructuredOutput again. Reason: {reason}"
    )


def _render_markdown(name: str, messages: list[Message]) -> str:
    """Write one ask's messages as a readable transcript."""
    lines = [f"# {name}", ""]
    for message in messages:
        if isinstance(message, UserMessage) and isinstance(message.content, str):
            lines += ["## user", "", message.content, ""]
        elif isinstance(message, UserMessage):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    text = block.content if isinstance(block.content, str) else json.dumps(block.content, ensure_ascii=False)
                    lines += [f"### tool result {block.tool_use_id}", "", "```", text, "```", ""]
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    lines += ["## assistant", "", block.text, ""]
                elif isinstance(block, ToolUseBlock):
                    lines += [f"### tool call {block.name} {block.id}", "", "```json", json.dumps(block.input, ensure_ascii=False, indent=2), "```", ""]
        elif isinstance(message, ResultMessage):
            lines += ["## result", "", f"subtype={message.subtype} turns={message.num_turns} cost_usd={message.total_cost_usd}", ""]
    return "\n".join(lines)


@dataclass
class CallRecord:
    result: dict | str
    cost_usd: float | None
    step_dir: Path


class AgentConversation:
    """One agent conversation that answers several prompts, one ask at a time.

    The SDK client holds the provider session across asks: later asks carry the
    earlier turns as the model's own context. Each ask records its own transcript
    under its own step dir, so every round stays inspectable on disk.
    """

    def __init__(self, calls: "AgentCalls", client: ClaudeSDKClient, *, name: str, instructions: str, structured: bool) -> None:
        self._calls = calls
        self._client = client
        self._name = name
        self._instructions = instructions
        self._structured = structured
        # ResultMessage.total_cost_usd counts everything this client has spent,
        # so one request's cost is the difference from the previous total.
        self._client_cost: float | None = 0.0

    async def _send(self, prompt: str, messages: list[Message], started: float) -> ResultMessage:
        await self._client.query(prompt)
        result: ResultMessage | None = None
        async for message in self._client.receive_response():
            messages.append(message)
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock) and block.name in _LIVE_TRANSCRIPT_TOOLS:
                        scope = "sub-agent" if message.parent_tool_use_id else "agent"
                        print(f"[{scope}] +{time.monotonic() - started:.0f}s: {block.name}", flush=True)
            elif isinstance(message, ResultMessage):
                result = message
        assert result is not None, f"{self._name}: the response ended without a result"
        return result

    def _request_cost(self, result: ResultMessage) -> float | None:
        previous = self._client_cost
        self._client_cost = result.total_cost_usd
        if previous is None or result.total_cost_usd is None:
            return None
        return result.total_cost_usd - previous

    async def ask(
        self,
        *,
        user_prompt: str,
        step_dir: Path,
        verify: Callable[[dict], Awaitable[str | None]] | None = None,
    ) -> CallRecord:
        """Send one prompt and return its declared structured result or text.

        ``verify`` returns None to accept, or a sentence saying what is wrong, and
        that sentence goes back into THIS session, so the agent repairs its own
        answer with everything it already read still in context.
        """
        if verify is not None and not self._structured:
            raise AssertionError(f"{self._name}: verify needs an output_schema to check")
        step_dir = Path(step_dir)
        self._calls.spent.setdefault(step_dir, 0.0)
        started = time.monotonic()
        messages: list[Message] = []
        cost: float | None = 0.0
        candidate: dict | None = None

        def record() -> None:
            self._calls.spent[step_dir] = sum_costs(self._calls.spent[step_dir], cost)
            step_dir.mkdir(parents=True, exist_ok=True)
            trajectory = [{"type": type(message).__name__, **dataclasses.asdict(message)} for message in messages]
            atomic_write_text(step_dir / "trajectory.json", json.dumps(trajectory, ensure_ascii=False, indent=2, default=str))
            atomic_write_text(step_dir / "execution.md", _render_markdown(self._name, messages))
            atomic_write_text(step_dir / "system_prompt.md", self._instructions)
            atomic_write_text(step_dir / "user_message.md", user_prompt)
            if candidate is not None:
                atomic_write_text(step_dir / "output_1.json", json.dumps(candidate, indent=2))
            else:
                (step_dir / "output_1.json").unlink(missing_ok=True)

        with _preserve_failure(record):
            prompt = user_prompt
            while True:
                result = await self._send(prompt, messages, started)
                cost = sum_costs(cost, self._request_cost(result))
                if result.is_error:
                    failure = PermanentError if result.api_error_status in _PERMANENT_API_STATUSES else RuntimeError
                    raise failure(f"{self._name}: {result.subtype}: {result.errors or result.result}")
                if not self._structured:
                    assert result.result is not None, f"{self._name}: conversation produced no text"
                    answer: dict | str = result.result
                    break
                if not isinstance(result.structured_output, dict):
                    raise AssertionError(f"{self._name}: conversation produced no structured result")
                candidate = result.structured_output
                answer = candidate
                if verify is None:
                    break
                reason = await verify(candidate)
                if reason is None:
                    break
                candidate = None
                prompt = _rejected(reason)
        return CallRecord(result=answer, cost_usd=cost, step_dir=step_dir)


@dataclass(slots=True)
class AgentCalls:
    """Agent calls made by one sourcing command.

    It also owns what was spent at each step dir. A caller that reads the
    cost off the record it got back misses every call that raised, and
    those calls spent real money before they failed. Calls that share a
    step dir add up, because a retry into the same dir spends again; the
    card writer gives every call its own dir, so there each figure is one
    call's, whether or not that call returned. Two callers writing
    different dirs cannot read each other's spend.
    """

    spent: dict[Path, float | None] = field(default_factory=dict)

    @asynccontextmanager
    async def conversation(
        self,
        *,
        name: str,
        model: str,
        cwd: str,
        instructions: str = "",
        tools: tuple[str, ...] = (),
        output_schema: dict | None = None,
        max_turns: int | None = None,
        add_dirs: tuple[str, ...] | list[str] = (),
        env: dict[str, str] | None = None,
        effort: str | None = None,
    ) -> AsyncIterator[AgentConversation]:
        """Hold one SDK client until the conversation it serves has closed."""
        options = ClaudeAgentOptions(
            system_prompt={"type": "preset", "preset": "claude_code", "append": instructions},
            tools=list(tools),
            allowed_tools=list(tools),
            model=model,
            cwd=cwd,
            add_dirs=list(add_dirs),
            env=dict(env or {}),
            permission_mode="bypassPermissions",
            setting_sources=[],
            max_turns=max_turns,
            effort=effort,  # pyright: ignore[reportArgumentType]
            output_format=None if output_schema is None else {"type": "json_schema", "schema": output_schema},
        )
        async with ClaudeSDKClient(options) as client:
            yield AgentConversation(self, client, name=name, instructions=instructions, structured=output_schema is not None)

    async def run_batch(
        self,
        *,
        items: list[dict],
        make_prompt: Callable[[list[dict]], str],
        key_of: Callable[[dict], str],
        output_schema: type[BaseModel],
        system_prompt: str,
        model: str,
        calls_dir: Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
        workers: int = DEFAULT_WORKERS,
        verify_verdict: Callable[[dict, dict], "str | None"] | None = None,
        on_batch_complete: Callable[[list[dict]], None] | None = None,
        max_batches: int | None = None,
    ) -> list[dict]:
        """Run one agent call per batch and return one verdict per item.

        A submission that misses a requested item, or that ``verify_verdict``
        rejects, goes back to the SAME session until the batch is complete.
        ``verify_verdict`` takes (item, verdict) and returns None to accept or a
        sentence saying what is wrong. Verdicts come back in item order, one per
        item of every processed batch (``max_batches`` bounds which batches are
        processed, never which items of a batch return).
        """
        batches = [
            items[index : index + batch_size]
            for index in range(0, len(items), batch_size)
        ]
        if max_batches is not None:
            batches = batches[:max_batches]
        semaphore = asyncio.Semaphore(workers)

        async def process(index: int, batch: list) -> list[dict]:
            async with semaphore:
                by_item = {key_of(item): item for item in batch}

                async def verify(result: dict) -> "str | None":
                    submitted = {
                        key_of(verdict): verdict
                        for verdict in result.get("verdicts", [])
                        if key_of(verdict) in by_item
                    }
                    reasons = []
                    missing = [key for key in by_item if key not in submitted]
                    if missing:
                        reasons.append(
                            "these requested items have no verdict; return "
                            "one verdict per item, echoing its key exactly: "
                            + ", ".join(missing)
                        )
                    if verify_verdict is not None:
                        for key, verdict in submitted.items():
                            reason = verify_verdict(by_item[key], verdict)
                            if reason is not None:
                                reasons.append(f"[{key}] {reason}")
                    return "\n".join(reasons) if reasons else None

                # No max_turns bound: the verify rounds need later turns, and a
                # session that is not converging is for the person watching the
                # run to stop.
                call_dir = calls_dir / f"batch_{index:03d}"
                call_dir.mkdir(parents=True, exist_ok=True)
                async with self.conversation(
                    name="agent",
                    instructions=system_prompt,
                    model=model,
                    cwd=str(call_dir.resolve()),
                    output_schema=output_schema.model_json_schema(),
                ) as session:
                    record = await session.ask(
                        user_prompt=make_prompt(batch),
                        step_dir=call_dir,
                        verify=verify,
                    )
                assert isinstance(record.result, dict)
                by_key = {
                    key_of(verdict): verdict
                    for verdict in record.result["verdicts"]
                    if key_of(verdict) in by_item
                }
                verdicts = [by_key[key_of(item)] for item in batch]
                if on_batch_complete is not None:
                    on_batch_complete(verdicts)
                return verdicts

        results = await asyncio.gather(
            *(process(index, batch) for index, batch in enumerate(batches))
        )
        return [verdict for result in results for verdict in result]
