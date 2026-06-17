# Corax Telegram Connector

A standalone [Corax](https://github.com/Alex12571333/corax-agent) capability that
gives an agent a **Telegram chat surface** — inspired by the Hermes agent's
gateway: **token-streamed replies**, **session / runtime commands**, **model
selection**, and **clean message formatting** (renders bold/italic/code, never
shows literal `**`).

It is a pure capability package. It uses only the public contracts of
`agent_core` (`Capability` / `Result`) and the `agent_sdk` manifest + loader, and
does **not** modify or depend on the internals of `corax-core`, `corax-sdk`, or
`corax-agent`. It installs by pointing a `capabilities.available` entry at this
directory.

| | |
|---|---|
| id | `telegram.connector` |
| entrypoint | `main:TelegramConnector` |
| type | `connector` |
| permission level | `confirm` |
| risk level | `medium` |
| side effects | `network_request`, `send_message` |

## Design

The connector is the Telegram **I/O + formatting + command brain**; an agent
loop drives it and decides *what* commands do. It never runs its own bot loop,
so it stays a well-behaved request/response capability.

| operation | purpose |
|---|---|
| `poll` | long-poll `getUpdates`; each update is tagged with its parsed slash command |
| `parse_command` | recognise `/new`, `/reload`, `/model`, `/help`, … (pure) |
| `send` | deliver a formatted message (auto-splits past Telegram's 4096 limit) |
| `edit` | edit an existing message |
| `stream` | token streaming via Telegram drafts when available, with edit-in-place fallback |
| `format` | markdown → Telegram HTML / plain text |
| `describe` | self-describe operations, commands, formats and defaults |

### Streaming

Call `stream` repeatedly as the model emits tokens, passing the growing `text`.
With `"transport": "auto"`, private chats use Telegram's native
`sendMessageDraft` preview for smoother animation; groups and failed draft
frames fall back to `sendMessage` + `editMessageText`.

```jsonc
{ "operation": "stream", "chat_id": 123, "message_id": null,
  "text": "partial answer…", "last_sent_text": "partial",
  "elapsed_ms": 800, "edit_interval_ms": 700, "buffer_threshold": 60,
  "transport": "auto", "chat_type": "private", "draft_id": 12345,
  "done": false }
```

It updates only when it's worth it — `done`, or `elapsed_ms ≥ edit_interval_ms`,
or `≥ buffer_threshold` new chars. Draft frames return `message_id: null`; the
final `done: true` call sends the real Telegram message that stays in history.
Edit fallback returns the `message_id` to reuse and appends a `▌` cursor until
the final call.

### Commands

`parse_command` / `poll` recognise:

| command | normalized action (the agent performs it) |
|---|---|
| `/new`, `/reset` | `new_session` |
| `/reload`, `/restart` | `reload_agent` |
| `/model <name>` | `set_model` (args = model name) |
| `/help`, `/start` | `help` (connector returns the help text) |
| `/stop`, `/cancel` | `cancel` |

The connector recognises the command and returns a ready-to-send `reply`; the
agent wires `new_session` / `reload_agent` / `set_model` to its runtime (e.g.
`runtime.reload_config`, a fresh session, or `CORAX_LLM_MODEL`). Pairs naturally
with [corax-llm-local-connector](https://github.com/Alex12571333/corax-llm-local-connector).

### Formatting (no stray `**`)

`format` (and every `send`/`edit`/`stream`) converts markdown to **Telegram
HTML**: `**b**`→`<b>b</b>`, `*i*`/`_i_`→`<i>…</i>`, `` `c` ``→`<code>…</code>`,
fenced blocks→`<pre>…</pre>`, `# H`→bold, `[t](u)`→`<a>`. Any leftover `**` is
stripped, so the model's bold never reaches the user as literal asterisks. Use
`"format": "plain"` for markdown-free plain text.

## Configuration

| Env var | Meaning |
|---|---|
| `CORAX_TELEGRAM_BOT_TOKEN` (or `TELEGRAM_BOT_TOKEN`) | bot token; read from env, **never** echoed in output |
| `CORAX_TELEGRAM_ALLOWED_CHATS` | optional comma list of allowed chat ids; others are refused (`POLICY_DENIED`) |
| `CORAX_TELEGRAM_BASE_URL` | override the API base (default `https://api.telegram.org`) |

Pass `"mock": true` in any request to skip the network (used by the tests).

## Security

- The bot token is read only from the environment and never placed in a result.
- Optional per-chat allow-list; never reads `.env` / `~/.ssh`.
- No raw exception leaks — every failure is a structured `Result.fail`.

## Install into a Corax Agent (no agent code change)

```yaml
capabilities:
  enabled: [..., telegram.connector]
  available:
    telegram.connector:
      enabled: true
      type: connector
      description: Telegram chat connector
      path: ../corax-telegram-connector
```

## Tests

```bash
python -m unittest discover -s tests -v
# coverage (100%)
python -m coverage run -m unittest discover -s tests && python -m coverage report -m
```
