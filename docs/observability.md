# tars observability — how to see what's happening

Four surfaces, ordered from highest level (agent decisions) to lowest
(raw tracing). Pick the one that matches the question you're trying
to answer; you don't usually need all four at once.

| Surface | Granularity | Storage | When to use |
|---|---|---|---|
| **Trajectory** | Agent decision tree (steps, retries, branches) | SQLite, XDG data dir | "What did the agent decide to do, in what order, and how did it end?" |
| **Pipeline events DB** | One row per LLM call | `~/.tars/events/pipeline_events.db` (+ `bodies.db`) | "Which calls were slow? Which model? Cache hit? What did the prompt/response actually look like?" |
| **tracing stream** | Live events to stderr as they happen | stderr (RAM) | "I'm running it now, what's the latency on this call?" |
| **`arc.log.jsonl`** | arc-specific (if you also run arc) | Project root JSONL file | arc-side debugging — separate from tars |

Don't confuse the trajectory store and the pipeline events DB. They
share an SQLite-flavored vibe but answer different questions and live
in different files.

---

## 1. Trajectory — agent decision tree

A single `tars run` (or `tars run-task`) produces one **trajectory**:
a tree of steps representing what the agent actually did, in order,
including failures and recoveries.

```bash
# List every trajectory the runtime knows about, with terminal status.
tars trajectory list
```

```
ID                                 EVENTS  STATUS
6ae96718ef7e4a32802e6fb8bd37bea2        5  completed
8588e0d6762a4fa898369a52df6d4071        5  abandoned
87ce2c7f66a44a3fa9d6bf928a743aff        5  abandoned
```

```bash
# Dump every event for one trajectory as JSON lines.
tars trajectory show 6ae96718ef7e4a32802e6fb8bd37bea2
```

The events are typed (`AgentEvent` in
`crates/tars-runtime/src/event.rs`); the common ones you'll see:

| Event | When |
|---|---|
| `trajectory_started` | A new run begins (root or a branch — `parent` is set on branches) |
| `step_started` | One agent invocation about to run (`agent`, `input_summary`, `idempotency_key`) |
| `llm_call_captured` | An LLM call inside a step — links to its `LlmCallFinished` record in the events DB |
| `step_completed` | The step succeeded (carries `output_summary` + per-step token `usage`) |
| `step_failed` | The step failed (carries `classification` so recovery can branch on it) |
| `trajectory_completed` / `_suspended` / `_abandoned` | Terminal — exactly one per trajectory |

### Filter for one step

```bash
tars trajectory show <id> \
  | jq -c 'select(.step_seq == 3)'
```

### Find every failed step across all trajectories

```bash
for id in $(tars trajectory list | awk 'NR>1 {print $1}'); do
  tars trajectory show "$id" | jq -c "select(.type==\"step_failed\") | . + {traj:\"$id\"}"
done
```

### Storage location

The trajectory store lives at `$XDG_DATA_HOME/tars/events.sqlite`
(macOS: `~/Library/Application Support/tars/events.sqlite`). Override
with `--events-path` or `TARS_EVENTS_PATH`.

---

## 2. Pipeline events DB — one row per LLM call

Each `Pipeline.complete` / `Pipeline.stream` lands one
`LlmCallFinished` event in `~/.tars/events/pipeline_events.db`, with
the full request and response bytes stashed in a content-addressed
`bodies.db` next to it.

This is the right place to answer **"why did that call take so long?"**
or **"what was the prompt that produced this output?"**.

### List recent calls

```bash
tars events list                       # default: last 7d, 50 rows
tars events list --since 1d --tag dogfood
tars events list --since all --limit 200 --json | jq -c .
```

```
event_id                              timestamp            tenant          model                         result  tags
------------------------------------------------------------------------------------------------------------------------
d6d01554-6dc0-429d-88d2-d1c13f376b28  2026-05-08 11:47:44  tenant-test     qwen/qwen3-coder-30b:2        ok
b6ee9aa7-86c3-4430-ba2a-a446e9208c54  2026-05-08 11:53:02  tenant-test     gemini-3-flash-preview        ok
```

### See one call in full

```bash
tars events show <event_id>            # JSON metadata only
tars events show <event_id> --with-bodies   # plus request/response payloads from CAS
```

The `--with-bodies` form resolves the `request_ref` / `response_ref`
hashes into `bodies.db` and prints the actual prompt and completion.
This is how to recover the full input/output of an old call without
re-running it.

### Useful one-liners

These are the queries that come up over and over again:

```bash
# Slowest 10 calls in the last day.
tars events list --since 1d --json \
  | jq -s 'sort_by(-.telemetry.provider_latency_ms)
           | .[:10]
           | .[] | {model: .actual_model,
                    latency_ms: .telemetry.provider_latency_ms,
                    out_tok: .usage.output_tokens}'

# Cache-hit rate by model.
tars events list --since 7d --json \
  | jq -s 'group_by(.actual_model)
           | map({model: .[0].actual_model,
                  total: length,
                  hits: ([.[] | select(.telemetry.cache_hit)] | length)})
           | map(. + {hit_rate: (.hits/.total)})'

# All errors and their kinds.
tars events list --since 7d --json \
  | jq -c 'select(.result.result=="error") | {ts: .timestamp, model: .actual_model, kind: .result.kind}'
```

### What's inside `LlmCallFinished`

The fields that matter most when debugging:

| Field | What |
|---|---|
| `telemetry.provider_latency_ms` | **Real wall time** of the underlying provider call. Trust this over `usage.output_tokens` if you're chasing latency. |
| `telemetry.cache_hit` | Was this served from the prompt cache? |
| `telemetry.retry_count`, `retry_attempts` | How many retries; each with its `error_kind` and backoff |
| `usage.{input,output,cached_input,thinking}_tokens` | Provider-reported usage. **Sanity-check against response body size** — CLI-backend providers like `claude_cli` can report inflated counts (see [providers/claude-cli.md §1](./providers/claude-cli.md)). |
| `actual_model` | Post-routing model, not the `ModelHint` the caller asked for |
| `request_fingerprint` | sha256 of canonical request body — same prompt across tenants hashes the same |
| `request_ref` / `response_ref` | CAS pointers into `bodies.db` |
| `result.result` + `result.kind` | `"ok"` or `"error"` + the error class string |
| `validation_summary` | Output validators that ran + their outcomes |

### Tags

If you set `tags` on a `RequestContext`, every call within that scope
gets tagged. Use them to slice — common patterns: `tags: ["dogfood"]`,
`tags: ["batch_2026_05_19"]`, `tags: ["experiment:tools_disabled"]`.

```bash
tars events list --tag dogfood --since 1d
```

### Forward compatibility

The event schema is `non_exhaustive`. Today only `LlmCallFinished` is
actively emitted; `EvaluationScored` is defined in the schema and will
land in a future phase (offline evaluators); `Other` is a catchall so
old readers don't break when a new variant ships. So your `jq` scripts
should always check `.type` before deep field access.

---

## 3. tracing stream — live, on stderr

While `tars` is running, the `TelemetryMiddleware` emits a `tracing`
event for every LLM call. These go to **stderr**, not stdout — `stdout`
stays clean for piping protocol output.

```bash
tars run ... -v              # info-level (one line per LLM call)
tars run ... -vv             # debug
tars run ... -vvv            # trace (very chatty)

# Fine-grained targeting via the standard RUST_LOG env (wins over -v).
RUST_LOG=tars_pipeline::telemetry=info,tars_provider=debug tars run ...
```

### JSON output — for log aggregators

By default the stderr stream is human-readable (ANSI colours, indented
fields). To pipe into Datadog / Loki / ELK / a JSONL log file, switch
formats with `--log-format json`:

```bash
tars run ... --log-format json 2>tars.jsonl

# Or via env var, useful for containerised deploys:
TARS_LOG_FORMAT_FLAG=json tars run ...
```

Sample JSON record (one event per line, `\n`-delimited):

```json
{"level":"INFO","message":"llm.call.finished","service":"tars-cli",
 "model":"claude-opus-4-7","elapsed_ms":847,"input_tokens":1234,
 "output_tokens":567,"cached_input_tokens":0,"stop_reason":"EndTurn",
 "trace_id":"…","tenant_id":"…"}
```

### Events you'll see

| Event | When | Key fields |
|---|---|---|
| `llm.call.start` | Request reaches the telemetry layer | `model`, `messages`, `tools`, `trace_id`, `tenant_id` |
| `llm.call.opened` | First byte from the provider | `elapsed_ms` (=TTFB / spawn-to-first-byte for CLI backends) |
| `llm.call.finished` | Stream terminated normally | `elapsed_ms`, full `usage`, `stop_reason` |
| `llm.call.failed` | Provider call failed to open | `elapsed_ms`, `error_class`, `error` message |
| `llm.call.stream_error` | Mid-stream error | `elapsed_ms`, `error_class`, `error` message |

### Combine with `jq` for real-time slicing

```bash
tars run ... --log-format json 2>&1 \
  | jq -c 'select(.message=="llm.call.finished")
           | {model, latency: .elapsed_ms, out_tok: .output_tokens}'
```

---

## 4. `arc.log.jsonl` — arc-specific (if you also use arc)

If your project is driven by [arc](https://github.com/leocaolab/arc),
arc writes its own JSONL log next to the project — typically
`./arc.log.jsonl`. This is **arc's** structured event stream, not
tars's; it covers what arc decided to do (fix rounds, scan results,
patch attempts) and which sessions it spawned.

```bash
# What did arc do in this run?
jq -r 'select(.event|test("fix_round|scan|patch|claude_invoke"))
       | [.timestamp, .event, .turn // "-"] | @tsv' \
  ./arc.log.jsonl | head
```

If you're not using arc, ignore this file — tars doesn't write to it.

---

## How the layers relate

```
trajectory                                ← tars trajectory show
  ├─ step 1 (agent="orchestrator")
  │    └─ llm_call_captured  ─────→  LlmCallFinished  ← tars events show
  │                                        (with body in bodies.db)
  ├─ step 2 (agent="worker:search")
  │    ├─ llm_call_captured  ─────→  LlmCallFinished
  │    └─ tool_call (e.g. fs.list_dir)
  └─ step_completed

(throughout: tracing events on stderr — controlled by RUST_LOG / -v)
(externally: arc.log.jsonl — if arc is the driver)
```

A trajectory references LLM calls by id; calls reference trajectories
by `trace_id`. So you can pivot either way:

```bash
# From a trajectory, find every LLM call it made:
tars trajectory show <traj_id> \
  | jq -r 'select(.type=="llm_call_captured") | .event_id' \
  | xargs -I{} tars events show {}

# From a slow LLM call, find which trajectory it was part of:
tars events show <event_id> | jq -r '.trace_id'
# then:
tars trajectory show <that-trace-id>
```

---

## Practical "I want to debug X → look at Y"

| Symptom | First place to look |
|---|---|
| "This call is slow" | `tars events list --since 1d --json \| jq 'sort_by(-.telemetry.provider_latency_ms)'` |
| "Why did the agent give up?" | `tars trajectory show <id>` — look for the final `trajectory_abandoned` event's `cause` |
| "Show me the actual prompt that produced this output" | `tars events show <event_id> --with-bodies` |
| "Was this cached?" | `tars events show <event_id> \| jq .telemetry.cache_hit` |
| "How many retries did this call need?" | `tars events show <event_id> \| jq .telemetry.retry_count` |
| "I want to grep across runs" | `--log-format json 2>tars.jsonl`, then `jq` or `rg` over `tars.jsonl` |
| "Are token counts wildly off from response size?" | You're probably using a `*_cli` backend with default flags. See [providers/claude-cli.md](./providers/claude-cli.md) — the agent loop is reporting inflated `usage`. |

---

## See also

- [`USER-GUIDE.md`](./USER-GUIDE.md) — calling shapes, validators, errors
- [`providers/claude-cli.md`](./providers/claude-cli.md) — token-bloat caveats for the CLI backend
- [`architecture/08-melt-observability.md`](./architecture/08-melt-observability.md) — internals of the tracing layer
- [`architecture/17-pipeline-event-store.md`](./architecture/17-pipeline-event-store.md) — schema spec for `pipeline_events.db`
