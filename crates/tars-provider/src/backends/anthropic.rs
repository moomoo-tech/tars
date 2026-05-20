//! Anthropic (Claude) HTTP backend.
//!
//! Wire format reference: <https://docs.anthropic.com/en/api/messages>
//!
//! Differences from OpenAI worth noting:
//!
//! - **Auth**: `x-api-key` header (not `Authorization: Bearer`).
//! - **Versioning**: `anthropic-version: 2023-06-01` mandatory header.
//! - **System**: separate top-level `system` field, not a message role.
//! - **Tool calls**: `tool_use` content blocks; no JSON-string nesting
//!   (args arrive as a real object — easier than OpenAI in this regard).
//! - **Caching**: explicit `cache_control: {type: "ephemeral"}` markers
//!   inserted on specific blocks. We attach to the system prompt and
//!   to the *last* message when [`CacheDirective::MarkBoundary`] is set.
//! - **Thinking**: a `thinking` content block + `thinking` config; we
//!   surface the deltas as [`ChatEvent::ThinkingDelta`].
//! - **Structured output**: emulated via a forced `tool_choice` (Doc
//!   01 §9). The "tool" is a synthetic schema-only call.
//! - **Streaming events**: SSE with named events (`message_start`,
//!   `content_block_start`, `content_block_delta`, `message_delta`,
//!   `message_stop`, `ping`, `error`). The named events are key — we
//!   route on `raw.event`, not just `data`.

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::StatusCode;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::{Value, json};
use url::Url;

use tars_types::{
    BatchItemId, BatchJobId, BatchResultItem, BatchStatus, CacheDirective, Capabilities,
    ChatEvent, ChatRequest, ChatResponse, ChatResponseBuilder, ContentBlock, ImageData,
    Message, Modality, PromptCacheKind, ProviderError, ProviderId, RequestContext,
    StopReason, StructuredOutputMode, Usage,
};

use crate::auth::{Auth, AuthResolver, ResolvedAuth};
use crate::batch::BatchSubmitter;
use crate::http_base::{
    HttpAdapter, HttpProviderBase, HttpProviderExtras, SseEvent, stream_via_adapter,
};
use crate::provider::{LlmEventStream, LlmProvider};
use crate::tool_buffer::ToolCallBuffer;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_API_VERSION: &str = "2023-06-01";

/// Synthetic tool name used to emulate structured output (Doc 01 §9).
const STRUCTURED_OUTPUT_TOOL: &str = "__respond_with__";

/// Builder.
#[derive(Clone, Debug)]
pub struct AnthropicProviderBuilder {
    id: ProviderId,
    base_url: String,
    api_version: String,
    auth: Auth,
    capabilities: Option<Capabilities>,
    extras: HttpProviderExtras,
}

impl AnthropicProviderBuilder {
    pub fn new(id: impl Into<ProviderId>, auth: Auth) -> Self {
        Self {
            id: id.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_version: DEFAULT_API_VERSION.to_string(),
            auth,
            capabilities: None,
            extras: HttpProviderExtras::default(),
        }
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn api_version(mut self, v: impl Into<String>) -> Self {
        self.api_version = v.into();
        self
    }

    pub fn capabilities(mut self, c: Capabilities) -> Self {
        self.capabilities = Some(c);
        self
    }

    pub fn extras(mut self, extras: HttpProviderExtras) -> Self {
        self.extras = extras;
        self
    }

    pub fn build(
        self,
        http: Arc<HttpProviderBase>,
        auth_resolver: Arc<dyn AuthResolver>,
    ) -> Arc<AnthropicProvider> {
        let caps = self.capabilities.unwrap_or_else(default_capabilities);
        let adapter = Arc::new(AnthropicAdapter {
            base_url: self.base_url,
            api_version: self.api_version,
            extras: self.extras,
        });
        Arc::new(AnthropicProvider {
            id: self.id,
            http,
            auth_resolver,
            auth: self.auth,
            adapter,
            capabilities: caps,
        })
    }
}

fn default_capabilities() -> Capabilities {
    use std::collections::HashSet;
    let mut modalities = HashSet::new();
    modalities.insert(Modality::Text);
    modalities.insert(Modality::Image);
    Capabilities {
        max_context_tokens: 200_000,
        max_output_tokens: 8_192,
        supports_tool_use: true,
        supports_parallel_tool_calls: true,
        supports_structured_output: StructuredOutputMode::ToolUseEmulation,
        supports_vision: true,
        supports_thinking: true,
        supports_cancel: true,
        prompt_cache: PromptCacheKind::ExplicitMarker,
        streaming: true,
        modalities_in: modalities.clone(),
        modalities_out: HashSet::from([Modality::Text]),
        pricing: tars_types::Pricing::default(),
    }
}

pub struct AnthropicProvider {
    id: ProviderId,
    http: Arc<HttpProviderBase>,
    auth_resolver: Arc<dyn AuthResolver>,
    auth: Auth,
    adapter: Arc<AnthropicAdapter>,
    capabilities: Capabilities,
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }
    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
    async fn stream(
        self: Arc<Self>,
        req: ChatRequest,
        ctx: RequestContext,
    ) -> Result<LlmEventStream, ProviderError> {
        let auth = self.auth_resolver.resolve(&self.auth, &ctx).await?;
        stream_via_adapter(self.http.clone(), self.adapter.clone(), auth, req, ctx).await
    }

    fn as_batch_submitter(self: Arc<Self>) -> Option<Arc<dyn BatchSubmitter>> {
        Some(self)
    }
}

// ─── BatchSubmitter — Anthropic Message Batches API ────────────────
//
// Reference: <https://docs.anthropic.com/en/api/creating-message-batches>
//
// One-step submission (no separate file upload): the request body
// inlines all items under `requests[]`. Vendor SLAs say up to 24 h,
// usually faster. Pricing is ~50% of sync. Per-item failures surface
// in `results()` while the overall job stays `Completed`.

#[async_trait]
impl BatchSubmitter for AnthropicProvider {
    async fn submit(
        &self,
        items: Vec<(BatchItemId, ChatRequest)>,
    ) -> Result<BatchJobId, ProviderError> {
        if items.is_empty() {
            return Err(ProviderError::InvalidRequest(
                "batch submit: items list must not be empty".into(),
            ));
        }

        // Reuse the streaming adapter's translate_request to build each
        // line's `params` — same body shape the synchronous endpoint
        // would have accepted.
        let mut requests = Vec::with_capacity(items.len());
        for (item_id, req) in items {
            let params = self.adapter.translate_request(&req)?;
            requests.push(json!({
                "custom_id": item_id.as_str(),
                "params": params,
            }));
        }
        let body = json!({ "requests": requests });

        let auth = self
            .auth_resolver
            .resolve(&self.auth, &RequestContext::test_default())
            .await?;
        let headers = self.adapter.build_headers(&auth)?;
        let url = self.adapter.batch_url("")?;

        let resp = self
            .http
            .client
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(ProviderError::from)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let h = resp.headers().clone();
            let text = resp.text().await.unwrap_or_default();
            return Err(self.adapter.classify_error(status, &h, &text));
        }

        let v: Value = resp
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("batch submit: response not JSON: {e}")))?;
        let id = v
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ProviderError::Parse("batch submit: response missing `id`".into()))?;
        Ok(BatchJobId::new(id))
    }

    async fn status(&self, id: &BatchJobId) -> Result<BatchStatus, ProviderError> {
        let auth = self
            .auth_resolver
            .resolve(&self.auth, &RequestContext::test_default())
            .await?;
        let headers = self.adapter.build_headers(&auth)?;
        let url = self.adapter.batch_url(&format!("/{}", id.as_str()))?;

        let resp = self
            .http
            .client
            .get(url)
            .headers(headers)
            .send()
            .await
            .map_err(ProviderError::from)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let h = resp.headers().clone();
            let text = resp.text().await.unwrap_or_default();
            return Err(self.adapter.classify_error(status, &h, &text));
        }

        let v: Value = resp
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("batch status: response not JSON: {e}")))?;
        translate_anthropic_batch_status(&v)
    }

    async fn results(
        &self,
        id: &BatchJobId,
    ) -> Result<Vec<BatchResultItem>, ProviderError> {
        // Anthropic's results endpoint 404s on non-terminal jobs; we
        // pre-check here so the error path is uniform across vendors
        // (see trait doc — `results()` on non-terminal is a caller bug,
        // not a backend error).
        let st = self.status(id).await?;
        if !st.is_terminal() {
            return Err(ProviderError::InvalidRequest(format!(
                "batch results: job {id} is not yet terminal (status: {st:?})"
            )));
        }

        let auth = self
            .auth_resolver
            .resolve(&self.auth, &RequestContext::test_default())
            .await?;
        let headers = self.adapter.build_headers(&auth)?;
        let url = self
            .adapter
            .batch_url(&format!("/{}/results", id.as_str()))?;

        let resp = self
            .http
            .client
            .get(url)
            .headers(headers)
            .send()
            .await
            .map_err(ProviderError::from)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let h = resp.headers().clone();
            let text = resp.text().await.unwrap_or_default();
            return Err(self.adapter.classify_error(status, &h, &text));
        }

        let text = resp
            .text()
            .await
            .map_err(ProviderError::from)?;
        parse_anthropic_batch_results(&text)
    }

    async fn cancel(&self, id: &BatchJobId) -> Result<(), ProviderError> {
        let auth = self
            .auth_resolver
            .resolve(&self.auth, &RequestContext::test_default())
            .await?;
        let headers = self.adapter.build_headers(&auth)?;
        let url = self
            .adapter
            .batch_url(&format!("/{}/cancel", id.as_str()))?;

        let resp = self
            .http
            .client
            .post(url)
            .headers(headers)
            .send()
            .await
            .map_err(ProviderError::from)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let h = resp.headers().clone();
            let text = resp.text().await.unwrap_or_default();
            return Err(self.adapter.classify_error(status, &h, &text));
        }
        Ok(())
    }
}

/// Translate Anthropic's batch status JSON into our vendor-neutral
/// [`BatchStatus`]. The vendor reports `processing_status` plus a
/// `request_counts` breakdown — we collapse "ended" into one of
/// Completed / Cancelled / Expired based on the count distribution.
fn translate_anthropic_batch_status(v: &Value) -> Result<BatchStatus, ProviderError> {
    let status = v
        .get("processing_status")
        .and_then(|s| s.as_str())
        .ok_or_else(|| {
            ProviderError::Parse("batch status: missing `processing_status`".into())
        })?;

    let counts = v.get("request_counts").cloned().unwrap_or_else(|| json!({}));
    let get = |k: &str| {
        counts
            .get(k)
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as u32
    };
    let processing = get("processing");
    let succeeded = get("succeeded");
    let errored = get("errored");
    let canceled = get("canceled");
    let expired = get("expired");
    let processed = succeeded + errored + canceled + expired;
    let total = Some(processed + processing);

    match status {
        "in_progress" => Ok(BatchStatus::InProgress {
            processed,
            total,
            eta: None,
        }),
        "canceling" => Ok(BatchStatus::InProgress {
            processed,
            total,
            eta: None,
        }),
        "ended" => {
            // Collapse the count distribution into one terminal state.
            // Per-item issues surface in results() — the overall job
            // is Completed unless every item ended the same non-success
            // way (all cancelled / all expired).
            if processed > 0 && canceled == processed {
                Ok(BatchStatus::Cancelled)
            } else if processed > 0 && expired == processed {
                Ok(BatchStatus::Expired)
            } else {
                Ok(BatchStatus::Completed)
            }
        }
        other => Err(ProviderError::Parse(format!(
            "batch status: unknown `processing_status` value: {other:?}"
        ))),
    }
}

/// Parse Anthropic's results JSONL into [`BatchResultItem`]s. Each
/// line has `custom_id` + `result.type` ∈ {succeeded, errored,
/// canceled, expired}; we translate to a per-item `Result<ChatResponse, ProviderError>`.
fn parse_anthropic_batch_results(text: &str) -> Result<Vec<BatchResultItem>, ProviderError> {
    let mut items = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| {
            ProviderError::Parse(format!(
                "batch results line {}: not JSON: {e}",
                idx + 1
            ))
        })?;
        let custom_id = v
            .get("custom_id")
            .and_then(|s| s.as_str())
            .ok_or_else(|| {
                ProviderError::Parse(format!(
                    "batch results line {}: missing custom_id",
                    idx + 1
                ))
            })?
            .to_string();
        let result_val = v.get("result").ok_or_else(|| {
            ProviderError::Parse(format!(
                "batch results line {}: missing result",
                idx + 1
            ))
        })?;
        let result_type = result_val
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| {
                ProviderError::Parse(format!(
                    "batch results line {}: missing result.type",
                    idx + 1
                ))
            })?;

        let outcome: Result<ChatResponse, ProviderError> = match result_type {
            "succeeded" => {
                let message = result_val.get("message").ok_or_else(|| {
                    ProviderError::Parse(format!(
                        "batch results line {}: succeeded but missing message",
                        idx + 1
                    ))
                })?;
                anthropic_message_to_chat_response(message)
            }
            "errored" => {
                let err = result_val
                    .get("error")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let err_type = err
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("error");
                let err_msg = err
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("(no message)");
                Err(match err_type {
                    "invalid_request_error" => {
                        ProviderError::InvalidRequest(err_msg.to_string())
                    }
                    "authentication_error" => ProviderError::Auth(err_msg.to_string()),
                    "rate_limit_error" => {
                        ProviderError::RateLimited { retry_after: None }
                    }
                    "overloaded_error" => ProviderError::ModelOverloaded,
                    _ => ProviderError::Internal(format!(
                        "anthropic batch item error ({err_type}): {err_msg}"
                    )),
                })
            }
            "canceled" => Err(ProviderError::Internal("item cancelled".into())),
            "expired" => Err(ProviderError::Internal("item expired".into())),
            other => Err(ProviderError::Parse(format!(
                "batch results line {}: unknown result.type {other:?}",
                idx + 1
            ))),
        };

        items.push(BatchResultItem {
            item_id: BatchItemId::new(custom_id),
            result: outcome,
        });
    }
    Ok(items)
}

/// Convert one Anthropic message-shape JSON into a [`ChatResponse`] by
/// replaying it through [`ChatResponseBuilder`]. Text content blocks
/// become `Delta` events; we set the terminal `Finished` from
/// `stop_reason` + `usage`.
///
/// **Known gap (Phase 2)**: `tool_use` content blocks are skipped.
/// Batch consumers that need tool calls in batch responses can either
/// (a) parse the raw `message` JSON themselves, or (b) wait for V2
/// when we extend `ChatEvent::ToolCallStart/Args/End` replay here.
fn anthropic_message_to_chat_response(msg: &Value) -> Result<ChatResponse, ProviderError> {
    let model = msg
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("anthropic");
    let mut acc = ChatResponseBuilder::new();
    acc.apply(ChatEvent::started(model));

    if let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) {
        for block in blocks {
            if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    acc.apply(ChatEvent::Delta {
                        text: text.to_string(),
                    });
                }
            }
            // tool_use blocks: see fn doc-comment.
        }
    }

    let stop_reason = match msg.get("stop_reason").and_then(|s| s.as_str()) {
        Some("end_turn") => StopReason::EndTurn,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some("tool_use") => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    };

    let u = msg.get("usage").cloned().unwrap_or_else(|| json!({}));
    let usage_u64 = |k: &str| u.get(k).and_then(|n| n.as_u64()).unwrap_or(0);
    let usage = Usage {
        input_tokens: usage_u64("input_tokens"),
        output_tokens: usage_u64("output_tokens"),
        cached_input_tokens: usage_u64("cache_read_input_tokens"),
        cache_creation_tokens: usage_u64("cache_creation_input_tokens"),
        thinking_tokens: 0,
    };
    acc.apply(ChatEvent::Finished { stop_reason, usage });
    Ok(acc.finish())
}

pub struct AnthropicAdapter {
    base_url: String,
    api_version: String,
    extras: HttpProviderExtras,
}

impl AnthropicAdapter {
    /// Translate one of our content blocks into Anthropic's content shape.
    fn translate_block(b: &ContentBlock) -> Value {
        match b {
            ContentBlock::Text { text } => json!({"type": "text", "text": text}),
            ContentBlock::Image { mime, data } => {
                let source = match data {
                    ImageData::Url(u) => json!({"type": "url", "url": u}),
                    ImageData::Base64(b) => json!({
                        "type": "base64",
                        "media_type": mime,
                        "data": b,
                    }),
                };
                json!({"type": "image", "source": source})
            }
        }
    }

    fn translate_content(blocks: &[ContentBlock]) -> Value {
        Value::Array(blocks.iter().map(Self::translate_block).collect())
    }

    fn translate_message(m: &Message) -> Value {
        match m {
            Message::User { content } => json!({
                "role": "user",
                "content": Self::translate_content(content),
            }),
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let mut blocks: Vec<Value> = content.iter().map(Self::translate_block).collect();
                for tc in tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }));
                }
                json!({"role": "assistant", "content": blocks})
            }
            // Anthropic doesn't have a `tool` role — tool results are
            // user-role messages with `tool_result` content blocks.
            Message::Tool {
                tool_call_id,
                content,
                is_error,
            } => {
                let mut result_block = json!({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": Self::translate_content(content),
                });
                if *is_error {
                    result_block["is_error"] = json!(true);
                }
                json!({
                    "role": "user",
                    "content": [result_block],
                })
            }
            // Anthropic's `system` is top-level, not a message role.
            // If a System message arrives here it's typically because
            // a caller serialized a transcript verbatim — flatten it
            // into a user-role text block prefixed with "[system]" so
            // it isn't indistinguishable from a real user turn.
            Message::System { content } => {
                let mut blocks: Vec<Value> = content.iter().map(Self::translate_block).collect();
                blocks.insert(0, json!({"type": "text", "text": "[system]"}));
                json!({
                    "role": "user",
                    "content": Value::Array(blocks),
                })
            }
        }
    }

    /// Apply [`CacheDirective::MarkBoundary`] markers. Anthropic accepts
    /// up to 4 cache_control markers; we attach to system + last
    /// content block per directive (in order). The translation here is
    /// best-effort — callers wanting precise placement should construct
    /// messages with the markers already on specific blocks.
    fn apply_cache_directives(body: &mut Value, directives: &[CacheDirective]) {
        let want_marker = directives
            .iter()
            .any(|d| matches!(d, CacheDirective::MarkBoundary { .. }));
        if !want_marker {
            return;
        }
        // Add marker to the system prompt (cheapest cache placement).
        if let Some(system_blocks) = body.get_mut("system").and_then(|s| s.as_array_mut()) {
            if let Some(last) = system_blocks.last_mut() {
                last["cache_control"] = json!({"type": "ephemeral"});
            }
        }
        // Add marker to the last block of the last *user* message
        // (covers RAG context use cases). If the conversation ends on
        // an assistant turn, attaching cache_control there would
        // cache assistant output instead of user-supplied context,
        // wasting the budget; walk back to the most recent user msg.
        if let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
            if let Some(last_user) = messages
                .iter_mut()
                .rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
            {
                if let Some(blocks) = last_user.get_mut("content").and_then(|c| c.as_array_mut()) {
                    if let Some(last_block) = blocks.last_mut() {
                        last_block["cache_control"] = json!({"type": "ephemeral"});
                    }
                }
            }
        }
    }
}

impl AnthropicAdapter {
    /// Build a `messages/batches` URL with the given suffix. Used by the
    /// `BatchSubmitter` impl on `AnthropicProvider` — `""` is the
    /// collection (POST submit), `/{id}` is one job, `/{id}/results` and
    /// `/{id}/cancel` are sub-resources.
    pub(crate) fn batch_url(&self, suffix: &str) -> Result<Url, ProviderError> {
        Url::parse(&format!(
            "{}/v1/messages/batches{suffix}",
            self.base_url.trim_end_matches('/')
        ))
        .map_err(|e| ProviderError::Internal(format!("bad anthropic batch url: {e}")))
    }
}

#[async_trait]
impl HttpAdapter for AnthropicAdapter {
    fn build_url(&self, _model: &str) -> Result<Url, ProviderError> {
        Url::parse(&format!(
            "{}/v1/messages",
            self.base_url.trim_end_matches('/')
        ))
        .map_err(|e| ProviderError::Internal(format!("bad anthropic base_url: {e}")))
    }

    fn build_headers(&self, auth: &ResolvedAuth) -> Result<HeaderMap, ProviderError> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert(
            "anthropic-version",
            HeaderValue::from_str(&self.api_version)
                .map_err(|e| ProviderError::Internal(format!("bad version header: {e}")))?,
        );
        match auth {
            ResolvedAuth::ApiKey(k) | ResolvedAuth::Bearer(k) => {
                h.insert(
                    "x-api-key",
                    HeaderValue::from_str(k).map_err(|e| {
                        // Invalid header chars in the key are an auth-
                        // credential problem, not a backend bug.
                        ProviderError::Auth(format!("malformed x-api-key value: {e}"))
                    })?,
                );
            }
            ResolvedAuth::None => {
                return Err(ProviderError::Auth(
                    "Anthropic requires an x-api-key; got Auth::None".into(),
                ));
            }
        }
        Ok(h)
    }

    fn translate_request(&self, req: &ChatRequest) -> Result<Value, ProviderError> {
        let model = req
            .model
            .explicit()
            .ok_or_else(|| ProviderError::InvalidRequest("model must be explicit".into()))?;

        if req.messages.is_empty() {
            return Err(ProviderError::InvalidRequest(
                "anthropic: messages array must contain at least one message".into(),
            ));
        }

        // Anthropic rejects requests with duplicate tool names with a
        // 400; surface a clear error before the round-trip.
        let mut seen = std::collections::HashSet::new();
        for t in &req.tools {
            if !seen.insert(t.name.as_str()) {
                return Err(ProviderError::InvalidRequest(format!(
                    "anthropic: duplicate tool name `{}` in request",
                    t.name
                )));
            }
        }
        // Reserved synthetic tool name used for structured-output
        // emulation must not collide with a caller-supplied tool.
        if req.structured_output.is_some()
            && req.tools.iter().any(|t| t.name == STRUCTURED_OUTPUT_TOOL)
        {
            return Err(ProviderError::InvalidRequest(format!(
                "anthropic: tool name `{STRUCTURED_OUTPUT_TOOL}` is reserved for structured-output emulation"
            )));
        }

        let messages: Vec<Value> = req.messages.iter().map(Self::translate_message).collect();

        let max_tokens = req.max_output_tokens.unwrap_or(4096);
        if max_tokens == 0 {
            return Err(ProviderError::InvalidRequest(
                "anthropic: max_output_tokens must be > 0".into(),
            ));
        }

        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": true,
        });

        if let Some(sys) = &req.system {
            // Always emit `system` as an array of blocks so cache_control
            // can be attached uniformly.
            body["system"] = json!([{"type": "text", "text": sys}]);
        }

        if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }
        if !req.stop_sequences.is_empty() {
            body["stop_sequences"] = json!(req.stop_sequences);
        }

        // Tools.
        let mut tools_to_send: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema.schema,
                })
            })
            .collect();

        // Structured output emulation (Doc 01 §9): inject a hidden tool
        // and force its use.
        if let Some(schema) = &req.structured_output {
            tools_to_send.push(json!({
                "name": STRUCTURED_OUTPUT_TOOL,
                "description": "Return the response strictly conforming to the schema.",
                "input_schema": schema.schema,
            }));
            body["tool_choice"] = json!({
                "type": "tool",
                "name": STRUCTURED_OUTPUT_TOOL,
            });
        } else if !tools_to_send.is_empty() {
            // Apply caller's tool_choice only when not overridden by
            // structured output.
            body["tool_choice"] = match &req.tool_choice {
                tars_types::ToolChoice::Auto => json!({"type": "auto"}),
                tars_types::ToolChoice::None => json!({"type": "none"}),
                tars_types::ToolChoice::Required => json!({"type": "any"}),
                tars_types::ToolChoice::Specific(name) => {
                    if !req.tools.iter().any(|t| &t.name == name) {
                        return Err(ProviderError::InvalidRequest(format!(
                            "anthropic: tool_choice references unknown tool `{name}`"
                        )));
                    }
                    json!({"type": "tool", "name": name})
                }
            };
        }
        if !tools_to_send.is_empty() {
            body["tools"] = Value::Array(tools_to_send);
        }

        // Thinking.
        match req.thinking {
            tars_types::ThinkingMode::Off => {}
            tars_types::ThinkingMode::Auto => {
                body["thinking"] = json!({"type": "enabled", "budget_tokens": 4096});
            }
            tars_types::ThinkingMode::Budget(b) => {
                body["thinking"] = json!({"type": "enabled", "budget_tokens": b});
            }
        }

        Self::apply_cache_directives(&mut body, &req.cache_directives);

        Ok(body)
    }

    fn parse_event(
        &self,
        raw: &SseEvent,
        buf: &mut ToolCallBuffer,
    ) -> Result<Vec<ChatEvent>, ProviderError> {
        if raw.data.is_empty() {
            return Ok(Vec::new());
        }
        // `ping` events carry no business payload.
        if raw.event == "ping" {
            return Ok(Vec::new());
        }
        if raw.event == "error" {
            // Provider-emitted error mid-stream (rare). Surface as ProviderError.
            let v: Value = serde_json::from_str(&raw.data).unwrap_or(Value::Null);
            let msg = v
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("anthropic mid-stream error")
                .to_string();
            return Err(ProviderError::Internal(msg));
        }

        let v: Value = serde_json::from_str(&raw.data).map_err(|e| {
            ProviderError::Parse(format!(
                "anthropic sse: {e} (raw: {})",
                truncate(&raw.data, 200)
            ))
        })?;

        let mut out = Vec::new();
        match raw.event.as_str() {
            "message_start" => {
                let model = v
                    .pointer("/message/model")
                    .and_then(|m| m.as_str())
                    .unwrap_or("")
                    .to_string();
                let cache_hit = v
                    .pointer("/message/usage")
                    .and_then(|u| u.as_object())
                    .map(|u| tars_types::CacheHitInfo {
                        cached_input_tokens: u
                            .get("cache_read_input_tokens")
                            .and_then(|x| x.as_u64())
                            .unwrap_or(0),
                        used_explicit_handle: false,
                        replayed_from_cache: false,
                    })
                    .unwrap_or_default();
                out.push(ChatEvent::Started {
                    actual_model: model,
                    cache_hit,
                });
            }
            "content_block_start" => {
                let index = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let cb = v.get("content_block").cloned().unwrap_or(Value::Null);
                match cb.get("type").and_then(|t| t.as_str()) {
                    Some("tool_use") => {
                        let id = cb
                            .get("id")
                            .and_then(|s| s.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = cb
                            .get("name")
                            .and_then(|s| s.as_str())
                            .unwrap_or("")
                            .to_string();
                        out.push(ChatEvent::ToolCallStart {
                            index,
                            id: id.clone(),
                            name: name.clone(),
                        });
                        buf.on_start(index, id, name);
                    }
                    Some("text") => {
                        // Anthropic occasionally emits a `text` block with
                        // initial text already populated. Forward it.
                        if let Some(t) = cb.get("text").and_then(|s| s.as_str()) {
                            if !t.is_empty() {
                                out.push(ChatEvent::Delta {
                                    text: t.to_string(),
                                });
                            }
                        }
                    }
                    Some("thinking") => {
                        if let Some(t) = cb.get("thinking").and_then(|s| s.as_str()) {
                            if !t.is_empty() {
                                out.push(ChatEvent::ThinkingDelta {
                                    text: t.to_string(),
                                });
                            }
                        }
                    }
                    _ => {} // unknown block types silently ignored
                }
            }
            "content_block_delta" => {
                let index = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let delta = v.get("delta").cloned().unwrap_or(Value::Null);
                match delta.get("type").and_then(|t| t.as_str()) {
                    Some("text_delta") => {
                        if let Some(t) = delta.get("text").and_then(|s| s.as_str()) {
                            out.push(ChatEvent::Delta {
                                text: t.to_string(),
                            });
                        }
                    }
                    Some("input_json_delta") => {
                        // Tool args fragment.
                        if let Some(p) = delta.get("partial_json").and_then(|s| s.as_str()) {
                            out.push(ChatEvent::ToolCallArgsDelta {
                                index,
                                args_delta: p.to_string(),
                            });
                            buf.on_delta(index, p);
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(t) = delta.get("thinking").and_then(|s| s.as_str()) {
                            out.push(ChatEvent::ThinkingDelta {
                                text: t.to_string(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                // Finalize any tool call at this index. Non-tool
                // content blocks (text / thinking) return an error
                // here because they were never registered with the
                // buffer; that's the expected path, log at trace.
                match buf.finalize(index) {
                    Ok((id, _name, parsed)) => {
                        out.push(ChatEvent::ToolCallEnd {
                            index,
                            id,
                            parsed_args: parsed,
                        });
                    }
                    Err(e) => {
                        tracing::trace!(
                            index,
                            error = %e,
                            "anthropic: content_block_stop finalize miss (likely text/thinking block)",
                        );
                    }
                }
            }
            "message_delta" => {
                // Carries `delta.stop_reason` and updated `usage`.
                let stop = v
                    .pointer("/delta/stop_reason")
                    .and_then(|s| s.as_str())
                    .map(map_stop_reason);
                let usage = v.get("usage").and_then(|u| u.as_object()).cloned();
                if let (Some(stop), Some(u)) = (stop, usage) {
                    out.push(ChatEvent::Finished {
                        stop_reason: stop,
                        usage: parse_usage(&u),
                    });
                    buf.mark_finished();
                }
            }
            // Authoritative end. message_delta may have failed to emit
            // Finished (missing stop_reason or usage in the delta
            // payload, mid-stream provider quirk), which would leave
            // consumers waiting forever. Emit a synthetic Finished as a
            // last resort.
            "message_stop" if !buf.finished_emitted() => {
                tracing::warn!(
                    "anthropic: message_stop without prior Finished; emitting synthetic terminator",
                );
                out.push(ChatEvent::Finished {
                    stop_reason: StopReason::Other,
                    usage: Usage::default(),
                });
                buf.mark_finished();
            }
            _ => {} // unknown events are tolerated
        }

        Ok(out)
    }

    fn classify_error(
        &self,
        status: StatusCode,
        headers: &reqwest::header::HeaderMap,
        body: &str,
    ) -> ProviderError {
        let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
        let message = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| truncate(body, 300));

        match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => ProviderError::Auth(message),
            StatusCode::TOO_MANY_REQUESTS => ProviderError::RateLimited {
                retry_after: crate::http_base::parse_retry_after(headers),
            },
            StatusCode::SERVICE_UNAVAILABLE | StatusCode::GATEWAY_TIMEOUT => {
                ProviderError::ModelOverloaded
            }
            StatusCode::BAD_REQUEST => {
                let lower = message.to_lowercase();
                if lower.contains("max_tokens") || lower.contains("context") {
                    ProviderError::ContextTooLong {
                        limit: 0,
                        requested: 0,
                    }
                } else {
                    ProviderError::InvalidRequest(message)
                }
            }
            s if s.is_server_error() => ProviderError::Internal(format!("status {s}: {message}")),
            _ => ProviderError::InvalidRequest(format!("status {status}: {message}")),
        }
    }

    fn extras(&self) -> &HttpProviderExtras {
        &self.extras
    }
}

fn map_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::Other,
    }
}

fn parse_usage(u: &serde_json::Map<String, Value>) -> Usage {
    // Anthropic reports input_tokens DISJOINT from cache_read /
    // cache_creation. Our canonical Usage struct is OpenAI-style:
    // input_tokens is the total prompt size and includes the cached
    // and creation subsets. Normalize at the boundary so cost_for and
    // total_tokens work uniformly across providers.
    let api_input = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let cached = u
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let creation = u
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    // Anthropic doesn't currently break out thinking tokens in the
    // usage block (they're folded into output_tokens) but probe a
    // couple of likely spellings to future-proof.
    let thinking = u
        .get("thinking_tokens")
        .or_else(|| u.get("output_thinking_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    Usage {
        input_tokens: api_input.saturating_add(cached).saturating_add(creation),
        output_tokens: output,
        cached_input_tokens: cached,
        cache_creation_tokens: creation,
        thinking_tokens: thinking,
    }
}

fn truncate(s: &str, max: usize) -> String {
    let trimmed = crate::http_base::truncate_utf8(s, max);
    if trimmed.len() == s.len() {
        s.to_string()
    } else {
        format!("{trimmed}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_401_is_auth() {
        let a = AnthropicAdapter {
            base_url: DEFAULT_BASE_URL.into(),
            api_version: DEFAULT_API_VERSION.into(),
            extras: HttpProviderExtras::default(),
        };
        let err = a.classify_error(
            StatusCode::UNAUTHORIZED,
            &reqwest::header::HeaderMap::new(),
            r#"{"error":{"message":"invalid"}}"#,
        );
        assert!(matches!(err, ProviderError::Auth(_)));
    }

    #[test]
    fn classify_429_with_retry_after_ms_populates_field() {
        // Anthropic uses retry-after-ms (millisecond precision).
        let a = AnthropicAdapter {
            base_url: DEFAULT_BASE_URL.into(),
            api_version: DEFAULT_API_VERSION.into(),
            extras: HttpProviderExtras::default(),
        };
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after-ms", "1500".parse().unwrap());
        let err = a.classify_error(StatusCode::TOO_MANY_REQUESTS, &headers, "");
        match err {
            ProviderError::RateLimited { retry_after } => {
                assert_eq!(retry_after, Some(std::time::Duration::from_millis(1500)));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn translate_request_promotes_system_to_top_level() {
        let a = AnthropicAdapter {
            base_url: DEFAULT_BASE_URL.into(),
            api_version: DEFAULT_API_VERSION.into(),
            extras: HttpProviderExtras::default(),
        };
        let req = ChatRequest::user(
            tars_types::ModelHint::Explicit("claude-opus-4-7".into()),
            "hello",
        )
        .with_system("you are concise");
        let body = a.translate_request(&req).unwrap();
        assert!(body["system"].is_array());
        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(body["system"][0]["text"], "you are concise");
        assert_eq!(body["model"], "claude-opus-4-7");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn cache_marker_attaches_to_last_message_block() {
        let a = AnthropicAdapter {
            base_url: DEFAULT_BASE_URL.into(),
            api_version: DEFAULT_API_VERSION.into(),
            extras: HttpProviderExtras::default(),
        };
        let mut req = ChatRequest::user(
            tars_types::ModelHint::Explicit("claude-opus-4-7".into()),
            "context",
        )
        .with_system("sys");
        req.cache_directives.push(CacheDirective::MarkBoundary {
            ttl: std::time::Duration::from_secs(300),
        });
        let body = a.translate_request(&req).unwrap();
        // Last user content block carries cache_control.
        let last_block = &body["messages"][0]["content"][0];
        assert_eq!(last_block["cache_control"]["type"], "ephemeral");
        // System block likewise.
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn structured_output_injects_forced_tool() {
        let a = AnthropicAdapter {
            base_url: DEFAULT_BASE_URL.into(),
            api_version: DEFAULT_API_VERSION.into(),
            extras: HttpProviderExtras::default(),
        };
        let mut req = ChatRequest::user(
            tars_types::ModelHint::Explicit("claude-opus-4-7".into()),
            "give json",
        );
        req.structured_output = Some(tars_types::JsonSchema::strict(
            "Resp",
            serde_json::json!({"type":"object"}),
        ));
        let body = a.translate_request(&req).unwrap();
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], STRUCTURED_OUTPUT_TOOL);
        let tools = body["tools"].as_array().unwrap();
        assert!(tools.iter().any(|t| t["name"] == STRUCTURED_OUTPUT_TOOL));
    }

    #[test]
    fn build_headers_requires_api_key() {
        let a = AnthropicAdapter {
            base_url: DEFAULT_BASE_URL.into(),
            api_version: DEFAULT_API_VERSION.into(),
            extras: HttpProviderExtras::default(),
        };
        let err = a.build_headers(&ResolvedAuth::None).unwrap_err();
        assert!(matches!(err, ProviderError::Auth(_)));

        let h = a
            .build_headers(&ResolvedAuth::ApiKey("sk-ant-x".into()))
            .unwrap();
        assert_eq!(h.get("x-api-key").unwrap(), "sk-ant-x");
        assert_eq!(h.get("anthropic-version").unwrap(), DEFAULT_API_VERSION);
    }

    #[test]
    fn map_stop_reasons() {
        assert_eq!(map_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(map_stop_reason("max_tokens"), StopReason::MaxTokens);
        assert_eq!(map_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(map_stop_reason("stop_sequence"), StopReason::StopSequence);
        assert_eq!(map_stop_reason("???"), StopReason::Other);
    }
}
