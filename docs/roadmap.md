# tars roadmap — cost & reliability for production agent serving

> Status: live planning doc. Features below are scoped to land in
> sequence; PRs reference this doc by section number.
> Last updated 2026-05-20.

The features in this doc share one motivation: **make tars usable
for safety-critical production agent serving at predictable cost**.
The driving use case is Cando Rail's "Peter the Safety Agent"
(LangChain + GPT-4.1, <$0.05/draft, expert-in-the-loop) — a
representative production agent that we currently don't fully support.

Order of work is bottom-up: type/error infra first, then middleware
that uses it, then the bigger access-pattern shift (batch).

| # | Feature | Effort | Status | Tracks |
|---|---|---|---|---|
| 1 | [Rate-limit max_wait + cancel propagation](#1-rate-limit-max-wait--cancel-propagation) | 1-2 days | next up | small |
| 2 | [Fallback / degrade middleware](#2-fallback--degrade-middleware) | 1-2 weeks | designed | small |
| 3 | [Per-call budget middleware](#3-per-call-budget-middleware) | 1 week | designed | small |
| 4 | [Tenant budget middleware (stateful)](#4-tenant-budget-middleware-stateful) | 2-3 weeks | sketched | medium |
| 5 | [Batch mode (BatchSubmitter trait)](#5-batch-mode-batchsubmitter-trait) | 3-4 weeks | sketched | medium |

Features 1+2 are paired — `max_wait` without fallback creates dead
paths, and fallback without `max_wait` makes "wait 30 minutes" a
plausible default. Ship them in the same release.

---

## Scope discipline (applies to everything below)

tars is an **agent runtime**. These features are intentionally narrow.

| In scope | Out of scope |
|---|---|
| Type-safe error classes for runtime decisions | Billing, invoicing, payment |
| Pre-call estimation, post-call true-up | Forecasting, financial planning |
| Middleware that fails fast or switches providers | Auto-scaling policy, capacity planning |
| Telemetry events so callers can observe | Job orchestration / cron / scheduling |
| Traits with reference impls | Persistence of operational state (caller's DB) |

If a feature requires a database, a UI, or a policy engine to be useful
at the runtime layer, it doesn't belong in tars-pipeline. Examples of
the right factoring: `BudgetStore` is a trait — callers plug in Redis
or Postgres. Same for `BatchJobStore`.

---

## 1. Rate-limit max_wait + cancel propagation

### Motivation

`Retry-After` from Anthropic/OpenAI can be hours during outages. The
current `RetryMiddleware` will sleep that long. An agent should never
sleep 30 minutes inside one call — better to bubble the error to a
fallback layer (§2) or to the caller.

### Current state (verified 2026-05-20)

- ✅ `ProviderError::RateLimited { retry_after: Option<Duration> }` in `tars-types/src/error.rs:22`
- ✅ `RetryMiddleware` parses `Retry-After` (`retry.rs:21` doc-comment) and respects it
- ✅ Anthropic + OpenAI HTTP backends extract the header
- ❌ No upper cap on sleep duration
- ❓ Cancel propagation during sleep — needs verification (must be `tokio::select!` against `ctx.cancel`)

### Design

```rust
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_backoff: Duration,
    pub backoff_multiplier: f64,
    pub max_backoff: Duration,
    pub max_wait: Duration,            // ← NEW: per-attempt cap on Retry-After respect
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(500),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(8),
            max_wait: Duration::from_secs(30),   // anything longer = bubble error
        }
    }
}
```

Logic:

```rust
if let Some(retry_after) = err.retry_after() {
    if retry_after > cfg.max_wait {
        // Bubble the error unchanged — outer FallbackMiddleware (§2)
        // may switch provider; if no fallback, caller decides.
        return Err(err);
    }
    cancellable_sleep(retry_after, &ctx.cancel).await;
} else {
    cancellable_sleep(exponential_backoff(attempt, &cfg), &ctx.cancel).await;
}
```

`cancellable_sleep` wraps `tokio::select! { _ = sleep => (), _ = cancel.cancelled() => () }` so a caller-side cancel terminates the sleep immediately.

### Tests

- Retry-After 5s, max_wait 30s → sleeps 5s, retries
- Retry-After 5min, max_wait 30s → returns `RateLimited` error unchanged on first attempt (does not retry)
- Cancel during sleep → returns immediately with `Cancelled` error (or whatever the existing cancel path produces)

### Not doing

- Adaptive `max_wait` based on tenant tier — caller passes static config
- Cross-call learning ("this provider rate-limited a lot, lower max_wait") — that's an analytics concern

---

## 2. Fallback / degrade middleware

### Motivation

Today, `RoutingPolicy` picks a provider **once** at request open. If
that provider returns `BudgetExceeded`, `RateLimited` (with long
retry-after), `ContextTooLong`, or `ModelOverloaded`, the call dies
even if another configured provider would have succeeded. Peter
explicitly wants Opus → Sonnet → Haiku → local degradation; this is
the standard cost+availability strategy in production.

Routing handles "which provider for *this* request shape." Fallback
handles "what to do when that provider failed in a typed, retryable
way." They are different concerns and should be separate middlewares.

### Design

```rust
use tars_pipeline::{FallbackMiddleware, FallbackTrigger};
use tars_types::ErrorClass;

let mw = FallbackMiddleware::builder(registry.clone())
    .primary(ProviderId::new("anthropic_opus"))
    .fallback_to(
        ProviderId::new("anthropic_sonnet"),
        FallbackTrigger::on(&[
            ErrorClass::BudgetExceeded,
            ErrorClass::ContextTooLong,
        ]),
    )
    .fallback_to(
        ProviderId::new("vllm_local"),
        FallbackTrigger::on(&[
            ErrorClass::RateLimited,
            ErrorClass::ModelOverloaded,
            ErrorClass::Network,
        ]),
    )
    .build();

let pipeline = Pipeline::builder(initial_provider)
    .layer(TelemetryMiddleware::new())
    .layer(mw)                              // Fallback OUTSIDE
    .layer(RetryMiddleware::default())      // Retry INSIDE — same-provider attempts first
    .build();
```

### Composition decisions

| Decision | Choice | Reason |
|---|---|---|
| Fallback layer position | **Outside Retry** | Retry handles short-term flakes on one provider; Fallback handles long-term capacity / cost / context problems. Reverse would burn fallback slots on every 429. |
| Trigger spec | Per-hop set of `ErrorClass` | Different errors warrant different fallback strategies (cost vs availability). Explicit is better than a single global trigger list. |
| `Permanent` error class | **Never triggers fallback** | 400 Bad Request means the request is wrong. Trying the same request on another provider fails the same way. |
| Each fallback hop runs Retry? | Yes — every hop is an independent "primary attempt" | Otherwise a transient flake on the fallback provider terminates the call. |
| Fingerprint stability across hops | `request_fingerprint` is provider-agnostic by construction | Free — schema already does this. Enables cross-provider analytics ("how often does this prompt fall back?"). |
| Cooperation with Budget MW | Pre-check on Opus rejects → Fallback catches → re-checks on Sonnet (different pricing) | Two middlewares, one shared concept (typed error) — no special-casing needed. |

### Telemetry

Each fallback hop emits:

```
tracing::warn!(
    event = "fallback.triggered",
    from = %from_provider_id,
    to = %to_provider_id,
    error_class = ?err.class(),
    trace_id = %ctx.trace_id,
)
```

These show up in the existing `--log-format json` stream and (Phase 2) in `pipeline_events.db` as a new `PipelineEvent::FallbackTriggered` variant.

### Not doing

- **Sticky session** ("always use the same provider for the same conversation_id") — caller sets `RequestContext.attributes.preferred_provider_id` and a custom RoutingPolicy reads it. Out of fallback scope.
- **Cost-aware ordering** ("dynamically pick the cheapest provider that meets capability"). Phase 2 if at all; for now an explicit ordered list is enough and transparent.
- **Cross-trajectory memory** ("provider X failed last hour, skip it this time"). That's a circuit-breaker concern — `CircuitBreaker` middleware already exists at a lower layer.
- **Auto-discovery of capable fallback providers** — caller spells out the chain. Magic here would hide real cost decisions.

### Tests

- BudgetExceeded on primary → switches to sonnet → succeeds
- RateLimited on primary with retry_after > max_wait → bubbles past Retry → caught by Fallback → switches to vllm_local
- Permanent error on primary → does NOT fall back; error surfaces immediately
- All hops fail → final error includes the *last* error from the chain plus a `hops_tried` list in the error message
- Fallback chain order respected (sonnet attempted before vllm_local for budget errors)
- Cancel mid-fallback → terminates immediately

---

## 3. Per-call budget middleware

### Motivation

Cando wants `<$0.05/draft`. Without runtime enforcement this is a
post-hoc analytics target — too late. The check needs to happen
before the network round-trip.

### Current state

- ✅ `ProviderError::BudgetExceeded` in `error.rs:27`
- ✅ `Capabilities.pricing` carries per-million-token rates
- ✅ Architecture doc 02 reserves the Budget layer position in the pipeline diagram
- ❌ No middleware implementation

### Design

```rust
pub struct PerCallBudgetMiddleware {
    cap_usd: f64,
}

impl PerCallBudgetMiddleware {
    pub fn new(cap_usd: f64) -> Self { Self { cap_usd } }
}
```

Pre-call:
1. `chars_in / 4` → estimated input tokens (consistent with arch §15 anti-pattern #1 — no real tokenizer on hot path)
2. `req.max_output_tokens` (or `Capabilities.max_output_tokens` fallback) → output token cap
3. Multiply by `Capabilities.pricing` → upper-bound USD cost
4. If `>= cap_usd` → return `ProviderError::BudgetExceeded` immediately, no provider call

Post-call (after stream drains): no debit — `PerCallBudgetMiddleware` is stateless. `TenantBudgetMiddleware` (§4) does the debit.

### Pricing-zero handling

`Pricing::default()` is all zeros — what the `claude_cli` / `gemini_cli` / `codex_cli` subscription backends use. Upper bound is always 0, so the cap is always satisfied. Behavior: `tracing::warn!` once per provider, then pass through.

```rust
if capabilities.pricing.is_zero() {
    tracing::warn!(
        provider = %req_ctx_provider,
        "per-call budget cap is a no-op on subscription-billed providers"
    );
    // pass through
}
```

### Composition

Cache should sit **outside** Budget — a cache hit should not pre-check budget at all (it's free). The current arch doc has Budget outside Cache; **this PR will propose flipping that** in arch §02.

```rust
Pipeline::builder(provider)
    .layer(TelemetryMiddleware::new())
    .layer(CacheLookupMiddleware::new(cache))
    .layer(PerCallBudgetMiddleware::new(0.05))   // ← inside Cache
    .layer(RetryMiddleware::default())
    .build();
```

### Not doing

- Token-precise estimation — `chars/4` is the rule across all middleware
- Per-feature budgets ("planner has $0.10, executor has $0.05") — caller composes by building separate Pipelines
- Cost attribution analytics — that's `tars events` + `jq` post-hoc
- Dynamic pricing tables — pricing comes from `Capabilities`; for live pricing, override Capabilities

### Tests

- Estimated cost > cap → BudgetExceeded
- Estimated cost ≤ cap → passes, real call goes through
- Subscription backend (zero pricing) → passes with one tracing warn
- Cache hit → does not consult budget at all (composition test)
- Estimation matches arch §15 anti-pattern (no tokenizer load on call path — assert via timing test or absence of dependency)

---

## 4. Tenant budget middleware (stateful)

### Motivation

`PerCallBudgetMiddleware` (§3) is stateless — perfect for the "one
call can't exceed X" Peter requirement, useless for "tenant-X gets
$100/day." Real multi-tenant deployments need the second one.

### Design

Trait + reference impls, store pluggable:

```rust
#[async_trait]
pub trait BudgetStore: Send + Sync + 'static {
    async fn remaining(&self, tenant: &TenantId) -> Result<Option<f64>, BudgetStoreError>;
    async fn debit(&self, tenant: &TenantId, amount_usd: f64) -> Result<f64, BudgetStoreError>;
    async fn refund(&self, tenant: &TenantId, amount_usd: f64) -> Result<(), BudgetStoreError>;
}

pub struct TenantBudgetMiddleware {
    store: Arc<dyn BudgetStore>,
}
```

Reference impls in tars-pipeline:
- `InMemoryBudgetStore` (testing, single-process deploys)
- (later) `RedisBudgetStore` behind a feature flag — or in a downstream crate

Pre-call: `store.remaining(tenant) >= estimated_cost` else `BudgetExceeded`.
Post-call: `store.debit(tenant, real_cost)` where `real_cost = (input + output × pricing_multiplier)` from `usage`.
Refund path: if the call failed mid-stream (no real usage), the pre-call optimistic debit should be refunded. Decision: **don't pre-debit, only post-debit.** Slight race risk (two concurrent calls both pass pre-check, both go through, briefly negative balance) — acceptable for runtime layer; strict accounting belongs in a billing system.

### Concurrency

Pre-check uses `remaining(tenant)` (read), so two concurrent calls can both pass. This is by design — the budget runtime is **soft** (best-effort cap), not a financial ledger. If callers need hard caps, layer their own ledger in the store impl with `WATCH` / row locks.

Documented as a tradeoff in the middleware's doc-comment.

### Not doing

- Quota allocation / provisioning — store is owned by caller
- Multi-currency — USD only; non-USD providers convert at config load
- Persistence semantics — store impl's choice (Redis TTL, Postgres `UPDATE … WHERE balance >= cost`, etc.)

---

## 5. Batch mode (`BatchSubmitter` trait)

### Motivation

Anthropic and OpenAI both offer batch APIs at ~50% sync pricing with
up to 24 h latency. Many of Peter's pre-work draft generations are
overnight-acceptable — half the cost is significant at scale.

### Why not just slot batch into `LlmProvider`

Different access pattern, different latency assumptions:
- No streaming; result file ready hours later
- Submit + poll + fetch — three calls, not one
- Caller must persist a `BatchJobId` across submit and fetch

Forcing this through `LlmEventStream` would break both APIs. Solution: separate trait, separate facade, **same request/response types** (so tooling and analytics work uniformly).

### Design

```rust
#[async_trait]
pub trait BatchSubmitter: Send + Sync + 'static {
    async fn submit(
        &self,
        items: Vec<(BatchItemId, ChatRequest)>,
    ) -> Result<BatchJobId, ProviderError>;

    async fn status(&self, id: &BatchJobId) -> Result<BatchStatus, ProviderError>;

    async fn results(
        &self,
        id: &BatchJobId,
    ) -> Result<Vec<BatchResultItem>, ProviderError>;
}

pub enum BatchStatus {
    Submitted,
    InProgress { processed: u32, total: u32, eta: Option<SystemTime> },
    Completed,
    Failed { kind: String, message: String },
    Expired,
}

pub struct BatchResultItem {
    pub item_id: BatchItemId,
    pub result: Result<ChatResponse, ProviderError>,
}
```

Provider impls:
- `AnthropicProvider` impls `BatchSubmitter` (message-batches API)
- `OpenAiProvider` impls `BatchSubmitter` (batch API)
- CLI backends do **not** impl — `try_as_batch_submitter` returns `None`

### Middleware coverage

What batch keeps from the sync pipeline:
- ✅ Telemetry — `tracing` events on submit/status/fetch
- ✅ Event store — fetch path translates each item into a `LlmCallFinished` event with `actual_model`, `usage`, real cost. Same downstream analytics work for batch and sync.
- ❌ Cache — assume miss (batch API has its own pricing semantics)
- ❌ Retry — failed batches are operator decisions, not auto-retry
- ❌ Routing / Fallback — batch is explicit; caller picks the provider
- ⚠️ Budget — Phase 2; pre-check can estimate from item list, post-tally from results

### Caller flow

```rust
let submitter: Arc<dyn BatchSubmitter> = registry
    .get(&ProviderId::new("anthropic"))
    .and_then(try_as_batch_submitter)
    .ok_or("provider doesn't support batch")?;

let job_id = submitter.submit(items).await?;
my_db.save_batch_job(job_id);                  // caller persists

// background worker polls
loop {
    match submitter.status(&job_id).await? {
        BatchStatus::Completed => {
            for item in submitter.results(&job_id).await? { /* … */ }
            break;
        }
        BatchStatus::InProgress { .. } => {
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
        BatchStatus::Failed { .. } | BatchStatus::Expired => break,
        _ => continue,
    }
}
```

### Not doing

- Cron / scheduling — caller-owned
- Job state DB — caller-owned (we expose opaque IDs)
- Auto-retry of failed jobs — caller decides on per-item retry policy
- Mixing batch + sync in one logical call — two APIs, caller routes
- Item-level cancellation — vendor API constraint, not ours

---

## What this roadmap is NOT

- **Not a billing system** — pricing is `Capabilities.pricing`; ledger is your DB
- **Not a job scheduler** — `tokio` is in your app, not in tars-pipeline
- **Not an evaluator** — `EvaluationScored` event exists; runner is Doc 16's job, not this roadmap
- **Not a vector store** — RAG primitives are documented as caller's concern (see `comparison.md` §LangChain)
- **Not a Realtime/voice transport** — `Modality::Audio` is reserved but unimplemented; separate roadmap when there's a concrete user

These exclusions are part of the agent-runtime scope discipline. If a
feature pushes any of these boundaries, it doesn't land here.
