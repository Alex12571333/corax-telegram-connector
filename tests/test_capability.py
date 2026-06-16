from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import urllib.error

from agent_core import (
    Capability,
    CapabilityRequest,
    ErrorCode,
    PermissionLevel,
    ResultStatus,
    RiskLevel,
    SideEffect,
)
from agent_sdk import CapabilityManifest, load_instance, validate_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Anchor coverage's source=["main"] to this package's main.py (the SDK loader
# re-executes the same file under an isolated name; coverage aggregates by path).
sys.path.insert(0, str(PROJECT_ROOT))
import main  # noqa: E402,F401

_ENV_KEYS = (
    "CORAX_TELEGRAM_BOT_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "CORAX_TELEGRAM_ALLOWED_CHATS",
    "CORAX_TELEGRAM_BASE_URL",
)


def request(payload: dict) -> CapabilityRequest:
    return CapabilityRequest(task_id="task-1", session_id="session-1", input=payload)


def _scrub_env() -> None:
    for key in _ENV_KEYS:
        os.environ.pop(key, None)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Pure formatting (module-level helpers)
# --------------------------------------------------------------------------- #
class FormattingTests(unittest.TestCase):
    def test_bold_becomes_html_no_asterisks(self) -> None:
        out = main.to_html("hello **world**")
        self.assertEqual(out, "hello <b>world</b>")
        self.assertNotIn("**", out)

    def test_full_markdown_to_html(self) -> None:
        src = "# Title\n**b** _i_ ~~s~~ `c` and __b2__\n```py\nx<1 & y\n```\n[t](http://h)"
        out = main.to_html(src)
        self.assertIn("<b>Title</b>", out)
        self.assertIn("<b>b</b>", out)
        self.assertIn("<i>i</i>", out)
        self.assertIn("<s>s</s>", out)
        self.assertIn("<code>c</code>", out)
        self.assertIn("<b>b2</b>", out)
        self.assertIn("<pre>x&lt;1 &amp; y\n</pre>", out)
        self.assertIn('<a href="http://h">t</a>', out)
        self.assertNotIn("**", out)

    def test_escapes_html_specials_outside_code(self) -> None:
        self.assertEqual(main.to_html("a < b & c > d"), "a &lt; b &amp; c &gt; d")

    def test_stray_double_asterisk_removed(self) -> None:
        self.assertNotIn("**", main.to_html("dangling ** here"))
        self.assertEqual(main.to_html(""), "")

    def test_to_plain_strips_markdown(self) -> None:
        out = main.to_plain("# H\n**b** _i_ `c` [t](http://h)\n```\ncode\n```")
        self.assertNotIn("**", out)
        self.assertNotIn("`", out)
        self.assertIn("t (http://h)", out)
        self.assertIn("code", out)
        self.assertEqual(main.to_plain(""), "")

    def test_render_formats(self) -> None:
        self.assertEqual(main.render("**x**", "html"), ("<b>x</b>", "HTML"))
        self.assertEqual(main.render("**x**", "auto"), ("<b>x</b>", "HTML"))
        self.assertEqual(main.render("**x**", "plain"), ("x", None))

    def test_render_rejects_unknown_format(self) -> None:
        with self.assertRaises(ValueError):
            main.render("x", "rtf")

    def test_split_source_keeps_short_text(self) -> None:
        self.assertEqual(main._split_source("short"), ["short"])

    def test_split_source_breaks_long_text(self) -> None:
        text = "\n".join(["line"] * 2000)
        chunks = main._split_source(text, limit=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 100 for c in chunks))

    def test_split_source_hard_splits_a_huge_line(self) -> None:
        chunks = main._split_source("x" * 250, limit=100)
        self.assertEqual(chunks, ["x" * 100, "x" * 100, "x" * 50])

    def test_split_source_flushes_before_huge_line(self) -> None:
        chunks = main._split_source("ab\n" + "y" * 150, limit=100)
        self.assertEqual(chunks[0], "ab")
        self.assertEqual(chunks[1], "y" * 100)


class CommandTests(unittest.TestCase):
    def test_not_a_command(self) -> None:
        self.assertFalse(main.parse_command("hello")["is_command"])
        self.assertFalse(main.parse_command(123)["is_command"])

    def test_new_and_reload(self) -> None:
        self.assertEqual(main.parse_command("/new")["command"], "new_session")
        self.assertEqual(main.parse_command("/reset")["command"], "new_session")
        self.assertEqual(main.parse_command("/reload")["command"], "reload_agent")

    def test_model_with_and_without_args(self) -> None:
        with_arg = main.parse_command("/model gemma-4")
        self.assertEqual(with_arg["command"], "set_model")
        self.assertEqual(with_arg["args"], "gemma-4")
        no_arg = main.parse_command("/model")
        self.assertEqual(no_arg["command"], "set_model")
        self.assertIn("/model", no_arg["reply"])

    def test_help_and_botname_suffix(self) -> None:
        out = main.parse_command("/help@corax_bot")
        self.assertEqual(out["command"], "help")
        self.assertIn("/new", out["reply"])

    def test_unknown_command(self) -> None:
        out = main.parse_command("/frobnicate now")
        self.assertEqual(out["command"], "unknown")
        self.assertEqual(out["args"], "now")


class ManifestLoaderTests(unittest.TestCase):
    def test_manifest_is_sdk_valid(self) -> None:
        manifest = CapabilityManifest.load(PROJECT_ROOT)
        result = validate_manifest(manifest, core_version="0.1.0")
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.warnings, [])
        self.assertEqual(manifest.id, "telegram.connector")
        self.assertEqual(manifest.name, "Telegram Connector")
        self.assertEqual(manifest.permission_level, PermissionLevel.CONFIRM)
        self.assertEqual(manifest.risk_level, RiskLevel.MEDIUM)
        self.assertEqual(manifest.entrypoint, "main:TelegramConnector")
        self.assertEqual(manifest.capability_type.value, "connector")

    def test_manifest_json_identity(self) -> None:
        data = json.loads((PROJECT_ROOT / "capability.json").read_text())
        self.assertEqual(data["id"], "telegram.connector")
        self.assertEqual(data["author"], "Corax")
        self.assertEqual(data["license"], "MIT")
        self.assertEqual(set(data["side_effects"]), {"network_request", "send_message"})
        self.assertNotIn("required_scopes", data)

    def test_loads_through_sdk_loader(self) -> None:
        manifest = CapabilityManifest.load(PROJECT_ROOT)
        cap = load_instance(manifest, PROJECT_ROOT, core_version="0.1.0")
        self.assertIsInstance(cap, Capability)
        self.assertEqual(cap.id, "telegram.connector")
        self.assertEqual(cap.required_scopes, set())
        self.assertEqual(
            cap.side_effects, {SideEffect.NETWORK_REQUEST, SideEffect.SEND_MESSAGE}
        )


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        _scrub_env()
        manifest = CapabilityManifest.load(PROJECT_ROOT)
        self.cap = load_instance(manifest, PROJECT_ROOT, core_version="0.1.0")

    async def asyncTearDown(self) -> None:
        _scrub_env()

    # -- dispatch / schema ---------------------------------------------- #
    async def test_missing_operation(self) -> None:
        result = await self.cap.execute(request({}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    async def test_unsupported_operation(self) -> None:
        result = await self.cap.execute(request({"operation": "fly"}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)
        self.assertIn("unsupported operation", result.error.message)

    async def test_schema_type_error(self) -> None:
        result = await self.cap.execute(request({"operation": "send", "message_id": "no"}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    # -- send ----------------------------------------------------------- #
    async def test_send_mock(self) -> None:
        result = await self.cap.execute(
            request({"operation": "send", "chat_id": 5, "text": "**hi** there", "mock": True})
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["parse_mode"], "HTML")
        self.assertEqual(result.payload["messages"][0]["text"], "<b>hi</b> there")
        self.assertEqual(result.payload["count"], 1)

    async def test_send_plain_with_reply_and_preview(self) -> None:
        result = await self.cap.execute(
            request(
                {
                    "operation": "send",
                    "chat_id": 5,
                    "text": "**hi**",
                    "format": "plain",
                    "reply_to": 10,
                    "disable_preview": True,
                    "mock": True,
                }
            )
        )
        self.assertIsNone(result.payload["parse_mode"])
        self.assertEqual(result.payload["messages"][0]["text"], "hi")

    async def test_send_requires_chat_id(self) -> None:
        result = await self.cap.execute(request({"operation": "send", "text": "hi", "mock": True}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)
        self.assertIn("chat_id", result.error.message)

    async def test_send_requires_text(self) -> None:
        result = await self.cap.execute(
            request({"operation": "send", "chat_id": 5, "text": "  ", "mock": True})
        )
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    async def test_send_missing_token(self) -> None:
        result = await self.cap.execute(request({"operation": "send", "chat_id": 5, "text": "hi"}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)
        self.assertIn("token", result.error.message)

    async def test_send_splits_long_text(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        long_text = "\n".join(["paragraph"] * 800)
        with patch.object(self.cap, "_call_api", MagicMock(return_value={"ok": True, "result": {"message_id": 1}})):
            result = await self.cap.execute(
                request({"operation": "send", "chat_id": 5, "text": long_text})
            )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertGreater(result.payload["count"], 1)

    # -- security ------------------------------------------------------- #
    async def test_chat_allowlist_denies(self) -> None:
        os.environ["CORAX_TELEGRAM_ALLOWED_CHATS"] = "100, 200"
        result = await self.cap.execute(
            request({"operation": "send", "chat_id": 5, "text": "hi", "mock": True})
        )
        self.assertEqual(result.status, ResultStatus.POLICY_DENIED)
        self.assertEqual(result.error.code, ErrorCode.POLICY_DENIED)

    async def test_chat_allowlist_allows(self) -> None:
        os.environ["CORAX_TELEGRAM_ALLOWED_CHATS"] = "5, 200"
        result = await self.cap.execute(
            request({"operation": "send", "chat_id": 5, "text": "hi", "mock": True})
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)

    async def test_token_never_leaks_in_output(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "SUPER-SECRET-TOKEN"
        with patch.object(self.cap, "_call_api", MagicMock(return_value={"ok": True, "result": {"message_id": 7}})) as api:
            result = await self.cap.execute(
                request({"operation": "send", "chat_id": 5, "text": "hi"})
            )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertNotIn("SUPER-SECRET-TOKEN", json.dumps(result.payload))
        self.assertEqual(api.call_args.kwargs["token"], "SUPER-SECRET-TOKEN")

    # -- edit ----------------------------------------------------------- #
    async def test_edit_mock(self) -> None:
        result = await self.cap.execute(
            request(
                {"operation": "edit", "chat_id": 5, "message_id": 9, "text": "**x**", "mock": True}
            )
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["message_id"], 9)
        self.assertEqual(result.payload["text"], "<b>x</b>")

    async def test_edit_plain_no_parse_mode(self) -> None:
        result = await self.cap.execute(
            request(
                {
                    "operation": "edit",
                    "chat_id": 5,
                    "message_id": 9,
                    "text": "**x**",
                    "format": "plain",
                    "mock": True,
                }
            )
        )
        self.assertIsNone(result.payload["parse_mode"])
        self.assertEqual(result.payload["text"], "x")

    async def test_edit_api_error_returns_fail(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        with patch.object(self.cap, "_call_api", MagicMock(side_effect=urllib.error.URLError("x"))):
            result = await self.cap.execute(
                request({"operation": "edit", "chat_id": 5, "message_id": 9, "text": "hi"})
            )
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)

    async def test_edit_requires_message_id(self) -> None:
        result = await self.cap.execute(
            request({"operation": "edit", "chat_id": 5, "text": "x", "mock": True})
        )
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    async def test_edit_requires_text(self) -> None:
        result = await self.cap.execute(
            request({"operation": "edit", "chat_id": 5, "message_id": 1, "text": "", "mock": True})
        )
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    # -- stream --------------------------------------------------------- #
    async def test_stream_first_chunk_sends(self) -> None:
        result = await self.cap.execute(
            request(
                {
                    "operation": "stream",
                    "chat_id": 5,
                    "text": "hello",
                    "elapsed_ms": 1000,
                    "mock": True,
                }
            )
        )
        self.assertTrue(result.payload["edited"])
        self.assertEqual(result.payload["sent_text"], "hello")
        self.assertEqual(result.payload["message_id"], 1)  # mock new-message id

    async def test_stream_skips_when_not_due(self) -> None:
        result = await self.cap.execute(
            request(
                {
                    "operation": "stream",
                    "chat_id": 5,
                    "text": "hello world",
                    "last_sent_text": "hello",
                    "elapsed_ms": 10,
                    "edit_interval_ms": 700,
                    "buffer_threshold": 1000,
                    "message_id": 3,
                    "mock": True,
                }
            )
        )
        self.assertFalse(result.payload["edited"])
        self.assertFalse(result.payload["should_edit"])

    async def test_stream_flushes_on_buffer_threshold(self) -> None:
        result = await self.cap.execute(
            request(
                {
                    "operation": "stream",
                    "chat_id": 5,
                    "message_id": 3,
                    "text": "x" * 80,
                    "last_sent_text": "",
                    "elapsed_ms": 0,
                    "buffer_threshold": 60,
                    "mock": True,
                }
            )
        )
        self.assertTrue(result.payload["edited"])
        self.assertEqual(result.payload["message_id"], 3)

    async def test_stream_done_finalizes_without_cursor(self) -> None:
        captured = {}

        def fake_call(*, token, method, params, base_url, timeout):
            captured["text"] = params["text"]
            return {"ok": True, "result": {"message_id": params.get("message_id", 4)}}

        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        with patch.object(self.cap, "_call_api", fake_call):
            result = await self.cap.execute(
                request(
                    {
                        "operation": "stream",
                        "chat_id": 5,
                        "message_id": 4,
                        "text": "final answer",
                        "last_sent_text": "final answer",
                        "done": True,
                    }
                )
            )
        self.assertTrue(result.payload["edited"])
        self.assertNotIn(main.STREAM_CURSOR, captured["text"])

    async def test_stream_plain_new_message(self) -> None:
        result = await self.cap.execute(
            request(
                {
                    "operation": "stream",
                    "chat_id": 5,
                    "text": "hello",
                    "format": "plain",
                    "elapsed_ms": 1000,
                    "mock": True,
                }
            )
        )
        self.assertTrue(result.payload["edited"])
        self.assertEqual(result.payload["message_id"], 1)

    async def test_stream_plain_edit_existing(self) -> None:
        result = await self.cap.execute(
            request(
                {
                    "operation": "stream",
                    "chat_id": 5,
                    "message_id": 7,
                    "text": "x" * 80,
                    "format": "plain",
                    "buffer_threshold": 60,
                    "mock": True,
                }
            )
        )
        self.assertTrue(result.payload["edited"])
        self.assertEqual(result.payload["message_id"], 7)

    async def test_stream_api_error_returns_fail(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        with patch.object(self.cap, "_call_api", MagicMock(side_effect=urllib.error.URLError("x"))):
            result = await self.cap.execute(
                request({"operation": "stream", "chat_id": 5, "text": "hi", "done": True})
            )
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)

    async def test_stream_done_empty_is_noop(self) -> None:
        result = await self.cap.execute(
            request({"operation": "stream", "chat_id": 5, "text": "   ", "done": True, "mock": True})
        )
        self.assertFalse(result.payload["edited"])

    # -- poll ----------------------------------------------------------- #
    async def test_poll_mock_updates_tags_commands(self) -> None:
        updates = [
            {"update_id": 11, "message": {"text": "/new", "chat": {"id": 5}}},
            {"update_id": 12, "edited_message": {"text": "hi", "chat": {"id": 5}}},
        ]
        result = await self.cap.execute(
            request({"operation": "poll", "mock": True, "mock_updates": updates})
        )
        self.assertEqual(result.payload["count"], 2)
        self.assertEqual(result.payload["updates"][0]["command"]["command"], "new_session")
        self.assertEqual(result.payload["next_offset"], 13)

    async def test_poll_mock_update_without_update_id(self) -> None:
        updates = [{"message": {"text": "hi", "chat": {"id": 5}}}]  # no update_id
        result = await self.cap.execute(
            request({"operation": "poll", "mock": True, "mock_updates": updates})
        )
        self.assertEqual(result.payload["count"], 1)
        self.assertIsNone(result.payload["next_offset"])

    async def test_poll_real_without_offset_or_limit(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        with patch.object(self.cap, "_call_api", MagicMock(return_value={"ok": True, "result": []})):
            result = await self.cap.execute(request({"operation": "poll", "timeout": 0}))
        self.assertEqual(result.payload["count"], 0)

    async def test_poll_api_error_returns_fail(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        with patch.object(self.cap, "_call_api", MagicMock(side_effect=urllib.error.URLError("x"))):
            result = await self.cap.execute(request({"operation": "poll", "timeout": 0}))
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)

    async def test_poll_invalid_timeout(self) -> None:
        result = await self.cap.execute(request({"operation": "poll", "timeout": 999, "mock": True}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    async def test_poll_real_via_api(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        payload = {"ok": True, "result": [{"update_id": 1, "message": {"text": "hey", "chat": {"id": 9}}}]}
        with patch.object(self.cap, "_call_api", MagicMock(return_value=payload)):
            result = await self.cap.execute(
                request({"operation": "poll", "offset": 1, "limit": 10, "timeout": 0})
            )
        self.assertEqual(result.payload["count"], 1)
        self.assertEqual(result.payload["updates"][0]["chat_id"], 9)

    # -- parse_command / format / describe ------------------------------ #
    async def test_parse_command_op(self) -> None:
        result = await self.cap.execute(
            request({"operation": "parse_command", "text": "/model gemma-4"})
        )
        self.assertEqual(result.payload["command"]["command"], "set_model")

    async def test_format_op(self) -> None:
        result = await self.cap.execute(
            request({"operation": "format", "text": "**bold**", "format": "html"})
        )
        self.assertEqual(result.payload["text"], "<b>bold</b>")

    async def test_format_rejects_non_string(self) -> None:
        result = await self.cap.execute(request({"operation": "format", "text": 5}))
        self.assertEqual(result.error.code, ErrorCode.INVALID_INPUT)

    async def test_describe_op(self) -> None:
        result = await self.cap.execute(request({"operation": "describe"}))
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertIn("stream", result.payload["operations"])
        self.assertIn("html", result.payload["formats"])
        names = {c["command"] for c in result.payload["commands"]}
        self.assertEqual(names, {"/new", "/reload", "/model", "/help"})

    # -- real network path + error handling (via _call_api) ------------- #
    async def test_real_call_api_via_urlopen(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "SECRET123"
        fake = _FakeResponse({"ok": True, "result": {"message_id": 55, "text": "x"}})
        with patch("urllib.request.urlopen", return_value=fake) as urlopen:
            result = await self.cap.execute(
                request({"operation": "send", "chat_id": 5, "text": "hi there"})
            )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.payload["message_ids"], [55])
        sent = urlopen.call_args.args[0]
        self.assertIn("/botSECRET123/sendMessage", sent.full_url)
        self.assertNotIn("SECRET123", json.dumps(result.payload))

    async def test_api_not_ok(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        with patch.object(self.cap, "_call_api", MagicMock(return_value={"ok": False, "description": "bad chat"})):
            result = await self.cap.execute(request({"operation": "send", "chat_id": 5, "text": "hi"}))
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)
        self.assertEqual(result.error.details["description"], "bad chat")

    async def test_api_result_not_dict_falls_back(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        with patch.object(self.cap, "_call_api", MagicMock(return_value={"ok": True, "result": True})):
            result = await self.cap.execute(request({"operation": "send", "chat_id": 5, "text": "hi"}))
        self.assertEqual(result.status, ResultStatus.SUCCESS)

    async def test_http_error(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        exc = urllib.error.HTTPError("http://x", 502, "bad", {}, None)
        with patch.object(self.cap, "_call_api", MagicMock(side_effect=exc)):
            result = await self.cap.execute(request({"operation": "send", "chat_id": 5, "text": "hi"}))
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)
        self.assertEqual(result.error.details["status"], 502)

    async def test_url_error(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        with patch.object(self.cap, "_call_api", MagicMock(side_effect=urllib.error.URLError("down"))):
            result = await self.cap.execute(request({"operation": "send", "chat_id": 5, "text": "hi"}))
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)
        self.assertIn("request failed", result.error.message)

    async def test_json_decode_error(self) -> None:
        os.environ["CORAX_TELEGRAM_BOT_TOKEN"] = "T"
        exc = json.JSONDecodeError("bad", "", 0)
        with patch.object(self.cap, "_call_api", MagicMock(side_effect=exc)):
            result = await self.cap.execute(request({"operation": "send", "chat_id": 5, "text": "hi"}))
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)
        self.assertIn("invalid JSON", result.error.message)

    async def test_no_raw_exceptions_leak(self) -> None:
        with patch.object(
            self.cap, "_send", MagicMock(side_effect=RuntimeError("raw secret failure"))
        ):
            result = await self.cap.execute(
                request({"operation": "send", "chat_id": 5, "text": "hi", "mock": True})
            )
        self.assertEqual(result.error.code, ErrorCode.CAPABILITY_FAILED)
        self.assertNotIn("raw secret failure", result.error.message)

    async def test_health_check(self) -> None:
        status = await self.cap.health_check()
        self.assertEqual(status.value, "healthy")


if __name__ == "__main__":
    unittest.main()
