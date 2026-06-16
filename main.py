"""Corax Telegram Connector capability.

A standalone connector that gives a Corax agent a Telegram chat surface,
inspired by the Hermes agent's gateway: token-streamed replies, session and
runtime commands, model selection, and clean message formatting (no stray
``**`` markdown artifacts).

It implements the public ``agent_core.Capability`` contract and carries an
``agent_sdk`` manifest, so it installs and loads through the SDK without any
change to ``corax-core`` / ``corax-sdk`` / ``corax-agent``.

The capability is the Telegram *I/O + formatting + command brain*; an agent
loop drives it:

* ``poll``           -- long-poll ``getUpdates`` and tag incoming slash commands
* ``parse_command``  -- recognise ``/new``, ``/reload``, ``/model``, ``/help`` …
* ``send`` / ``edit``-- deliver a (formatted) message or edit one
* ``stream``         -- the throttled token-streaming step (edit-in-place with a
                        cursor), so the model's answer appears live
* ``format``         -- markdown -> Telegram HTML (renders bold/italic/code,
                        never shows literal ``**``)
* ``describe``       -- self-describe operations, commands and formats

The agent decides *what* the commands do (start a new session, reload the
runtime, switch the active model); this connector only recognises them and
returns a ready-to-send confirmation. The bot token is read from the
environment and is never echoed back to the caller.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from agent_core import (
    Capability,
    CapabilityRequest,
    CoreError,
    ErrorCode,
    HealthStatus,
    PermissionLevel,
    Result,
    ResultStatus,
    RiskLevel,
    SideEffect,
)
from agent_core import schema as core_schema
from agent_sdk import capability
from agent_sdk.manifests.models import CapabilityType

CAPABILITY_ID = "telegram.connector"
CAPABILITY_NAME = "Telegram Connector"

DEFAULT_BASE_URL = "https://api.telegram.org"
DEFAULT_FORMAT = "html"
SUPPORTED_FORMATS = ("html", "plain", "auto")

MAX_MESSAGE_CHARS = 4096
# Split the *source* below this so the rendered HTML stays under the hard limit.
SAFE_SOURCE_CHARS = 3500
STREAM_CURSOR = "▌"

DEFAULT_EDIT_INTERVAL_MS = 700
DEFAULT_BUFFER_THRESHOLD = 60
DEFAULT_POLL_TIMEOUT = 30
MAX_POLL_TIMEOUT = 60
DEFAULT_REQUEST_TIMEOUT = 65.0

_TOKEN_ENV = ("CORAX_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN")
_ALLOWED_CHATS_ENV = "CORAX_TELEGRAM_ALLOWED_CHATS"

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string"},
        "chat_id": {},
        "text": {"type": "string"},
        "message_id": {"type": "integer"},
        "format": {"type": "string"},
        "reply_to": {"type": "integer"},
        "disable_preview": {"type": "boolean"},
        "offset": {"type": "integer"},
        "limit": {"type": "integer"},
        "timeout": {"type": "integer"},
        "last_sent_text": {"type": "string"},
        "elapsed_ms": {"type": "number"},
        "done": {"type": "boolean"},
        "edit_interval_ms": {"type": "integer"},
        "buffer_threshold": {"type": "integer"},
        "base_url": {"type": "string"},
        "request_timeout": {"type": "number"},
        "mock": {"type": "boolean"},
        "mock_updates": {"type": "array"},
        "state_key": {"type": "string"},
    },
    "required": ["operation"],
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string"},
        "ok": {"type": "boolean"},
        "chat_id": {},
        "format": {"type": "string"},
        "parse_mode": {"type": "string"},
        "text": {"type": "string"},
        "message_id": {"type": "integer"},
        "message_ids": {"type": "array"},
        "messages": {"type": "array"},
        "edited": {"type": "boolean"},
        "should_edit": {"type": "boolean"},
        "done": {"type": "boolean"},
        "updates": {"type": "array"},
        "count": {"type": "integer"},
        "next_offset": {"type": "integer"},
        "command": {"type": "object"},
        "commands": {"type": "array"},
        "operations": {"type": "array"},
        "formats": {"type": "array"},
    },
    "required": ["operation", "ok"],
}

# Slash command vocabulary. Each maps to a normalized action the *agent* runs,
# plus a ready-to-send confirmation for the simple ones.
_COMMANDS = {
    "start": ("help", None),
    "help": ("help", None),
    "new": ("new_session", "🆕 Started a new session."),
    "reset": ("new_session", "🆕 Started a new session."),
    "reload": ("reload_agent", "♻️ Reloading the agent…"),
    "restart": ("reload_agent", "♻️ Reloading the agent…"),
    "model": ("set_model", None),
    "stop": ("cancel", "🛑 Cancelled."),
    "cancel": ("cancel", "🛑 Cancelled."),
}

_HELP_TEXT = (
    "Commands:\n"
    "/new — start a new session\n"
    "/reload — reload the agent\n"
    "/model <name> — switch model (no name shows the current one)\n"
    "/help — show this help"
)


class _SecurityError(Exception):
    """A request was refused by the capability's security rules."""


# --------------------------------------------------------------------------- #
# Result helpers
# --------------------------------------------------------------------------- #
def _fail(
    request: CapabilityRequest,
    code: ErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
    status: ResultStatus = ResultStatus.ERROR,
) -> Result:
    return Result.fail(
        CoreError(code=code, message=message, details=details or {}, retryable=retryable),
        session_id=request.session_id,
        task_id=request.task_id,
        status=status,
    )


def _ok(request: CapabilityRequest, payload: dict[str, Any]) -> Result:
    return Result.ok(payload, session_id=request.session_id, task_id=request.task_id)


# --------------------------------------------------------------------------- #
# Formatting: markdown -> Telegram HTML / plain (never leaves literal ``**``)
# --------------------------------------------------------------------------- #
def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_html(text: str) -> str:
    """Convert common markdown to Telegram-HTML. Code is protected from escaping."""
    if not text:
        return ""

    placeholders: dict[str, str] = {}
    counter = [0]

    def ph(value: str) -> str:
        key = f"\x00{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = value
        return key

    t = text
    # Fenced code blocks -> <pre>
    t = re.sub(
        r"```[^\n]*\n?([\s\S]*?)```",
        lambda m: ph(f"<pre>{_escape_html(m.group(1))}</pre>"),
        t,
    )
    # Inline code -> <code>
    t = re.sub(
        r"`([^`\n]+)`",
        lambda m: ph(f"<code>{_escape_html(m.group(1))}</code>"),
        t,
    )
    # Links [text](url) -> <a>
    t = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda m: ph(f'<a href="{_escape_html(m.group(2))}">{_escape_html(m.group(1))}</a>'),
        t,
    )
    # Headers (#..######) -> bold
    t = re.sub(
        r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$",
        lambda m: ph(f"<b>{_escape_html(m.group(1))}</b>"),
        t,
    )
    # Bold **text** / __text__
    t = re.sub(r"\*\*(.+?)\*\*", lambda m: ph(f"<b>{_escape_html(m.group(1))}</b>"), t, flags=re.S)
    t = re.sub(r"__(.+?)__", lambda m: ph(f"<b>{_escape_html(m.group(1))}</b>"), t, flags=re.S)
    # Strikethrough ~~text~~
    t = re.sub(r"~~(.+?)~~", lambda m: ph(f"<s>{_escape_html(m.group(1))}</s>"), t, flags=re.S)
    # Italic *text* / _text_ (single, same line; bullets like "* item" are left alone)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", lambda m: ph(f"<i>{_escape_html(m.group(1))}</i>"), t)
    t = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", lambda m: ph(f"<i>{_escape_html(m.group(1))}</i>"), t)
    # Drop any stray markers so a literal ``**`` / ``__`` can never reach the user.
    t = t.replace("**", "").replace("__", "")
    # Escape everything that is left, then restore the protected fragments.
    t = _escape_html(t)
    for key, value in placeholders.items():
        t = t.replace(key, value)
    return t


def to_plain(text: str) -> str:
    """Strip markdown to clean plain text (also removes stray ``**``)."""
    if not text:
        return ""
    t = text
    t = re.sub(r"```[^\n]*\n?([\s\S]*?)```", r"\1", t)
    t = re.sub(r"`([^`\n]+)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r"\1 (\2)", t)
    t = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+", "", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)
    t = re.sub(r"__(.+?)__", r"\1", t, flags=re.S)
    t = re.sub(r"~~(.+?)~~", r"\1", t, flags=re.S)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", t)
    t = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", t)
    return t.replace("**", "").replace("__", "")


def render(text: str, fmt: str) -> tuple[str, str | None]:
    """Return ``(rendered_text, parse_mode)`` for the requested format."""
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format {fmt!r} (use: {', '.join(SUPPORTED_FORMATS)})")
    if fmt == "plain":
        return to_plain(text), None
    return to_html(text), "HTML"


def _split_source(text: str, limit: int = SAFE_SOURCE_CHARS) -> list[str]:
    """Split source text into chunks at line boundaries, under ``limit`` chars."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if current and len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = line
        else:
            current = line if not current else f"{current}\n{line}"
    chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def parse_command(text: object) -> dict[str, Any]:
    """Recognise a leading slash command. Pure; returns a normalized record."""
    if not isinstance(text, str):
        return {"is_command": False}
    stripped = text.strip()
    if not stripped.startswith("/"):
        return {"is_command": False}
    head, _, rest = stripped.partition(" ")
    name = head[1:].split("@", 1)[0].lower()
    args = rest.strip()
    if name not in _COMMANDS:
        return {"is_command": True, "command": "unknown", "name": name, "args": args, "reply": None}
    action, reply = _COMMANDS[name]
    if action == "help":
        reply = _HELP_TEXT
    elif action == "set_model" and not args:
        reply = "Send /model <name> to switch the model."
    return {"is_command": True, "command": action, "name": name, "args": args, "reply": reply}


# --------------------------------------------------------------------------- #
# Environment-backed setup
# --------------------------------------------------------------------------- #
def _resolve_token(mock: bool) -> str:
    for name in _TOKEN_ENV:
        token = os.getenv(name)
        if token:
            return token
    if mock:
        return ""
    raise ValueError(
        "Telegram bot token not configured (set CORAX_TELEGRAM_BOT_TOKEN)"
    )


def _check_chat_allowed(chat_id: object) -> None:
    raw = os.getenv(_ALLOWED_CHATS_ENV)
    if not raw:
        return
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    if str(chat_id) not in allowed:
        raise _SecurityError("chat_id is not in the configured allow-list")


def _resolve_base_url(data: dict[str, Any]) -> str:
    return str(data.get("base_url") or os.getenv("CORAX_TELEGRAM_BASE_URL") or DEFAULT_BASE_URL)


def _require_chat_id(data: dict[str, Any]) -> object:
    if "chat_id" not in data or data["chat_id"] in (None, ""):
        raise ValueError("chat_id is required")
    return data["chat_id"]


# --------------------------------------------------------------------------- #
# Capability
# --------------------------------------------------------------------------- #
@capability(
    id=CAPABILITY_ID,
    name=CAPABILITY_NAME,
    description=(
        "Telegram chat connector for Corax: token-streamed replies, session and "
        "model commands, and clean HTML formatting (no stray ** markdown)."
    ),
    version="1.0.0",
    author="Corax",
    license="MIT",
    tags=["telegram", "connector", "messaging", "streaming", "bot"],
    permission_level=PermissionLevel.CONFIRM,
    risk_level=RiskLevel.MEDIUM,
    side_effects=[SideEffect.NETWORK_REQUEST, SideEffect.SEND_MESSAGE],
    input_schema=INPUT_SCHEMA,
    output_schema=OUTPUT_SCHEMA,
    entrypoint="main:TelegramConnector",
    capability_type=CapabilityType.CONNECTOR,
    min_core_version="0.1.0",
    sdk_version="0.1.0",
)
class TelegramConnector(Capability):
    """Drive a Telegram bot chat: poll, format, send, and stream replies."""

    async def execute(self, request: CapabilityRequest) -> Result:
        try:
            data = request.input
            errors = core_schema.validate(data, INPUT_SCHEMA)
            if errors:
                raise ValueError("; ".join(errors))

            operation = data["operation"]
            handler = {
                "send": self._send,
                "edit": self._edit,
                "stream": self._stream,
                "poll": self._poll,
                "parse_command": self._parse_command,
                "format": self._format,
                "describe": self._describe,
            }.get(operation)
            if handler is None:
                raise ValueError(f"unsupported operation {operation!r}")
            result = handler(request, data)
            # Optional: mirror the payload into core session state so a
            # kernel-driven caller (the agent gateway) can read it back via the
            # StateManager. Off unless a non-empty ``state_key`` is given.
            state_key = data.get("state_key")
            if isinstance(state_key, str) and state_key and result.is_success:
                result.state_patch = {state_key: result.payload}
            return result
        except ValueError as exc:
            return _fail(request, ErrorCode.INVALID_INPUT, str(exc))
        except _SecurityError as exc:
            return _fail(
                request,
                ErrorCode.POLICY_DENIED,
                f"request rejected by capability security rules: {exc}",
                status=ResultStatus.POLICY_DENIED,
            )
        except Exception:
            return _fail(
                request,
                ErrorCode.CAPABILITY_FAILED,
                "telegram connector failed before completing the request",
            )

    # -- operations ------------------------------------------------------ #
    def _send(self, request: CapabilityRequest, data: dict[str, Any]) -> Result:
        mock = bool(data.get("mock", False))
        token = _resolve_token(mock)
        chat_id = _require_chat_id(data)
        _check_chat_allowed(chat_id)
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        fmt = str(data.get("format", DEFAULT_FORMAT))
        _, parse_mode = render("", fmt)  # validates fmt early
        base_url = _resolve_base_url(data)
        timeout = float(data.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))

        messages: list[dict[str, Any]] = []
        reply_to = data.get("reply_to")
        for source_chunk in _split_source(text):
            rendered, parse_mode = render(source_chunk, fmt)
            params: dict[str, Any] = {"chat_id": chat_id, "text": rendered}
            if parse_mode:
                params["parse_mode"] = parse_mode
            if data.get("disable_preview"):
                params["disable_web_page_preview"] = True
            if reply_to is not None:
                params["reply_to_message_id"] = reply_to
                reply_to = None  # thread only the first chunk
            result = self._dispatch_message(
                request, mock, token, "sendMessage", params, base_url, timeout
            )
            if isinstance(result, Result):
                return result
            messages.append({"message_id": result.get("message_id"), "text": rendered})

        return _ok(
            request,
            {
                "operation": "send",
                "ok": True,
                "chat_id": chat_id,
                "format": fmt,
                "parse_mode": parse_mode,
                "message_ids": [m["message_id"] for m in messages],
                "messages": messages,
                "count": len(messages),
            },
        )

    def _edit(self, request: CapabilityRequest, data: dict[str, Any]) -> Result:
        mock = bool(data.get("mock", False))
        token = _resolve_token(mock)
        chat_id = _require_chat_id(data)
        _check_chat_allowed(chat_id)
        message_id = data.get("message_id")
        if not isinstance(message_id, int):
            raise ValueError("message_id is required for edit")
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        fmt = str(data.get("format", DEFAULT_FORMAT))
        rendered, parse_mode = render(text, fmt)
        base_url = _resolve_base_url(data)
        timeout = float(data.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))
        params: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": rendered}
        if parse_mode:
            params["parse_mode"] = parse_mode

        result = self._dispatch_message(
            request, mock, token, "editMessageText", params, base_url, timeout
        )
        if isinstance(result, Result):
            return result
        return _ok(
            request,
            {
                "operation": "edit",
                "ok": True,
                "chat_id": chat_id,
                "message_id": result.get("message_id", message_id),
                "format": fmt,
                "parse_mode": parse_mode,
                "text": rendered,
            },
        )

    def _stream(self, request: CapabilityRequest, data: dict[str, Any]) -> Result:
        mock = bool(data.get("mock", False))
        chat_id = _require_chat_id(data)
        _check_chat_allowed(chat_id)
        text = str(data.get("text", ""))
        last_sent = str(data.get("last_sent_text", ""))
        done = bool(data.get("done", False))
        elapsed_ms = float(data.get("elapsed_ms", 0))
        interval = int(data.get("edit_interval_ms", DEFAULT_EDIT_INTERVAL_MS))
        threshold = int(data.get("buffer_threshold", DEFAULT_BUFFER_THRESHOLD))
        message_id = data.get("message_id")
        fmt = str(data.get("format", DEFAULT_FORMAT))

        changed = text != last_sent
        pending = max(0, len(text) - len(last_sent))
        should_edit = done or (changed and (elapsed_ms >= interval or pending >= threshold))

        if not should_edit or (done and not text.strip() and message_id is None):
            return _ok(
                request,
                {
                    "operation": "stream",
                    "ok": True,
                    "edited": False,
                    "should_edit": should_edit,
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "done": done,
                },
            )

        token = _resolve_token(mock)
        base_url = _resolve_base_url(data)
        timeout = float(data.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))
        display = text if done else f"{text}{STREAM_CURSOR}"
        rendered, parse_mode = render(display, fmt)

        if message_id is None:
            params = {"chat_id": chat_id, "text": rendered}
            if parse_mode:
                params["parse_mode"] = parse_mode
            method = "sendMessage"
        else:
            params = {"chat_id": chat_id, "message_id": message_id, "text": rendered}
            if parse_mode:
                params["parse_mode"] = parse_mode
            method = "editMessageText"

        result = self._dispatch_message(request, mock, token, method, params, base_url, timeout)
        if isinstance(result, Result):
            return result
        return _ok(
            request,
            {
                "operation": "stream",
                "ok": True,
                "edited": True,
                "should_edit": True,
                "chat_id": chat_id,
                "message_id": result.get("message_id", message_id),
                "sent_text": text,
                "done": done,
            },
        )

    def _poll(self, request: CapabilityRequest, data: dict[str, Any]) -> Result:
        mock = bool(data.get("mock", False))
        token = _resolve_token(mock)
        timeout = int(data.get("timeout", DEFAULT_POLL_TIMEOUT))
        if timeout < 0 or timeout > MAX_POLL_TIMEOUT:
            raise ValueError(f"timeout must be between 0 and {MAX_POLL_TIMEOUT}")

        if mock:
            raw_updates = list(data.get("mock_updates", []))
        else:
            params: dict[str, Any] = {"timeout": timeout}
            if "offset" in data:
                params["offset"] = data["offset"]
            if "limit" in data:
                params["limit"] = data["limit"]
            base_url = _resolve_base_url(data)
            request_timeout = float(data.get("request_timeout", timeout + 5))
            response = self._call_api_safe(
                request, token, "getUpdates", params, base_url, request_timeout
            )
            if isinstance(response, Result):
                return response
            raw_updates = response.get("result", [])

        updates = []
        next_offset = None
        for update in raw_updates:
            message = update.get("message") or update.get("edited_message") or {}
            text = message.get("text")
            chat = message.get("chat") or {}
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                next_offset = update_id + 1
            updates.append(
                {
                    "update_id": update_id,
                    "chat_id": chat.get("id"),
                    "text": text,
                    "command": parse_command(text),
                }
            )

        return _ok(
            request,
            {
                "operation": "poll",
                "ok": True,
                "count": len(updates),
                "updates": updates,
                "next_offset": next_offset,
            },
        )

    def _parse_command(self, request: CapabilityRequest, data: dict[str, Any]) -> Result:
        return _ok(
            request,
            {
                "operation": "parse_command",
                "ok": True,
                "command": parse_command(data.get("text")),
            },
        )

    def _format(self, request: CapabilityRequest, data: dict[str, Any]) -> Result:
        # ``text`` is schema-guaranteed to be a string when present.
        text = data.get("text", "")
        fmt = str(data.get("format", DEFAULT_FORMAT))
        rendered, parse_mode = render(text, fmt)
        return _ok(
            request,
            {
                "operation": "format",
                "ok": True,
                "format": fmt,
                "parse_mode": parse_mode,
                "text": rendered,
            },
        )

    def _describe(self, request: CapabilityRequest, data: dict[str, Any]) -> Result:
        commands = []
        for name in ("new", "reload", "model", "help"):
            action, _ = _COMMANDS[name]
            commands.append({"command": f"/{name}", "action": action})
        return _ok(
            request,
            {
                "operation": "describe",
                "ok": True,
                "operations": ["send", "edit", "stream", "poll", "parse_command", "format", "describe"],
                "commands": commands,
                "formats": list(SUPPORTED_FORMATS),
                "defaults": {
                    "edit_interval_ms": DEFAULT_EDIT_INTERVAL_MS,
                    "buffer_threshold": DEFAULT_BUFFER_THRESHOLD,
                    "cursor": STREAM_CURSOR,
                    "max_message_chars": MAX_MESSAGE_CHARS,
                },
                "token_env": list(_TOKEN_ENV),
            },
        )

    # -- network --------------------------------------------------------- #
    def _dispatch_message(
        self,
        request: CapabilityRequest,
        mock: bool,
        token: str,
        method: str,
        params: dict[str, Any],
        base_url: str,
        timeout: float,
    ) -> dict[str, Any] | Result:
        """Send/edit one message. Returns the Telegram ``result`` dict or a Result fail."""
        if mock:
            return {"message_id": params.get("message_id", 1), "mock": True}
        return self._call_api_safe(request, token, method, params, base_url, timeout)

    def _call_api_safe(
        self,
        request: CapabilityRequest,
        token: str,
        method: str,
        params: dict[str, Any],
        base_url: str,
        timeout: float,
    ) -> dict[str, Any] | Result:
        try:
            response = self._call_api(
                token=token, method=method, params=params, base_url=base_url, timeout=timeout
            )
        except urllib.error.HTTPError as exc:
            return _fail(
                request,
                ErrorCode.CAPABILITY_FAILED,
                "telegram API returned an HTTP error",
                details={"status": exc.code, "method": method},
                retryable=True,
            )
        except (urllib.error.URLError, TimeoutError, OSError):
            return _fail(
                request,
                ErrorCode.CAPABILITY_FAILED,
                "telegram API request failed",
                details={"method": method},
                retryable=True,
            )
        except json.JSONDecodeError:
            return _fail(
                request,
                ErrorCode.CAPABILITY_FAILED,
                "telegram API returned invalid JSON",
                details={"method": method},
                retryable=True,
            )
        if not response.get("ok", False):
            return _fail(
                request,
                ErrorCode.CAPABILITY_FAILED,
                "telegram API rejected the request",
                details={"method": method, "description": response.get("description")},
                retryable=False,
            )
        result = response.get("result")
        return result if isinstance(result, dict) else response

    def _call_api(
        self,
        *,
        token: str,
        method: str,
        params: dict[str, Any],
        base_url: str,
        timeout: float,
    ) -> dict[str, Any]:
        endpoint = f"{base_url.rstrip('/')}/bot{token}/{method}"
        body = json.dumps(params).encode("utf-8")
        http_request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    async def health_check(self) -> HealthStatus:
        return HealthStatus.HEALTHY
