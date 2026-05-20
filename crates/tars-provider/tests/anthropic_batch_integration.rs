//! End-to-end Anthropic batch tests against a wiremock-backed server.
//!
//! Covers submit / status / results / cancel — the four BatchSubmitter
//! trait methods — plus the `as_batch_submitter` LlmProvider override.

use std::sync::Arc;

use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use tars_provider::auth::{Auth, basic};
use tars_provider::backends::anthropic::AnthropicProviderBuilder;
use tars_provider::http_base::HttpProviderBase;
use tars_provider::provider::LlmProvider;
use tars_types::{
    BatchItemId, BatchJobId, BatchStatus, ChatRequest, ModelHint, ProviderError,
};

fn build_provider(server: &MockServer) -> Arc<dyn LlmProvider> {
    let http = HttpProviderBase::default_arc().unwrap();
    let provider = AnthropicProviderBuilder::new("anthropic_test", Auth::inline("test-key"))
        .base_url(server.uri())
        .build(http, basic());
    provider
}

#[tokio::test]
async fn submit_posts_requests_array_and_returns_job_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        // Confirm body is shape-compatible — `requests` array with custom_ids.
        .and(body_partial_json(serde_json::json!({
            "requests": [
                {"custom_id": "draft-1"},
                {"custom_id": "draft-2"},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msgbatch_01abc",
            "type": "message_batch",
            "processing_status": "in_progress",
            "request_counts": {
                "processing": 2,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = build_provider(&server);
    let submitter = provider.as_batch_submitter().expect("anthropic supports batch");

    let id = submitter
        .submit(vec![
            (
                BatchItemId::new("draft-1"),
                ChatRequest::user(ModelHint::Explicit("claude-opus-4-7".into()), "draft one"),
            ),
            (
                BatchItemId::new("draft-2"),
                ChatRequest::user(ModelHint::Explicit("claude-opus-4-7".into()), "draft two"),
            ),
        ])
        .await
        .unwrap();
    assert_eq!(id.as_str(), "msgbatch_01abc");
}

#[tokio::test]
async fn submit_empty_items_is_invalid_request() {
    // No wiremock — should reject before HTTP.
    let server = MockServer::start().await;
    let provider = build_provider(&server);
    let submitter = provider.as_batch_submitter().unwrap();

    let err = submitter.submit(vec![]).await.err().expect("must reject");
    assert!(matches!(err, ProviderError::InvalidRequest(_)));
}

#[tokio::test]
async fn status_translates_in_progress() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/msgbatch_01abc"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msgbatch_01abc",
            "processing_status": "in_progress",
            "request_counts": {
                "processing": 8,
                "succeeded": 2,
                "errored": 0,
                "canceled": 0,
                "expired": 0
            }
        })))
        .mount(&server)
        .await;

    let provider = build_provider(&server);
    let submitter = provider.as_batch_submitter().unwrap();
    let st = submitter
        .status(&BatchJobId::new("msgbatch_01abc"))
        .await
        .unwrap();
    match st {
        BatchStatus::InProgress {
            processed,
            total,
            eta: _,
        } => {
            assert_eq!(processed, 2);
            assert_eq!(total, Some(10));
        }
        other => panic!("expected InProgress, got {other:?}"),
    }
}

#[tokio::test]
async fn status_translates_ended_to_completed() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/msgbatch_done"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msgbatch_done",
            "processing_status": "ended",
            "request_counts": {
                "processing": 0,
                "succeeded": 9,
                "errored": 1,
                "canceled": 0,
                "expired": 0
            }
        })))
        .mount(&server)
        .await;

    let provider = build_provider(&server);
    let submitter = provider.as_batch_submitter().unwrap();
    assert_eq!(
        submitter
            .status(&BatchJobId::new("msgbatch_done"))
            .await
            .unwrap(),
        BatchStatus::Completed,
    );
}

#[tokio::test]
async fn status_all_expired_maps_to_expired() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/msgbatch_exp"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msgbatch_exp",
            "processing_status": "ended",
            "request_counts": {
                "processing": 0,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 5
            }
        })))
        .mount(&server)
        .await;

    let provider = build_provider(&server);
    let submitter = provider.as_batch_submitter().unwrap();
    assert_eq!(
        submitter
            .status(&BatchJobId::new("msgbatch_exp"))
            .await
            .unwrap(),
        BatchStatus::Expired,
    );
}

#[tokio::test]
async fn results_parses_jsonl_with_mixed_outcomes() {
    let server = MockServer::start().await;
    // First call: status (results() pre-checks terminality).
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/msgbatch_done"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msgbatch_done",
            "processing_status": "ended",
            "request_counts": {
                "processing": 0,
                "succeeded": 1,
                "errored": 1,
                "canceled": 0,
                "expired": 0
            }
        })))
        .mount(&server)
        .await;

    // Then: actual results JSONL with one succeeded + one errored.
    let jsonl = r#"{"custom_id":"draft-1","result":{"type":"succeeded","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-7","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2}}}}
{"custom_id":"draft-2","result":{"type":"errored","error":{"type":"invalid_request_error","message":"bad prompt"}}}"#;
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/msgbatch_done/results"))
        .respond_with(ResponseTemplate::new(200).set_body_string(jsonl))
        .mount(&server)
        .await;

    let provider = build_provider(&server);
    let submitter = provider.as_batch_submitter().unwrap();
    let results = submitter
        .results(&BatchJobId::new("msgbatch_done"))
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].item_id.as_str(), "draft-1");
    let succ = results[0].result.as_ref().expect("draft-1 should succeed");
    assert!(succ.text.contains("hello"));
    assert_eq!(succ.usage.input_tokens, 10);
    assert_eq!(succ.usage.output_tokens, 2);

    assert_eq!(results[1].item_id.as_str(), "draft-2");
    let err = results[1].result.as_ref().expect_err("draft-2 should fail");
    assert!(matches!(err, ProviderError::InvalidRequest(_)));
}

#[tokio::test]
async fn results_on_non_terminal_returns_invalid_request_without_fetching() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/msgbatch_pending"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "processing_status": "in_progress",
            "request_counts": {
                "processing": 5,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0
            }
        })))
        .mount(&server)
        .await;
    // /results endpoint NOT mocked — if results() tries to fetch, the
    // 404 from wiremock would mask the actual contract we want to test.
    // Instead, results() should pre-check status() and refuse before hitting it.

    let provider = build_provider(&server);
    let submitter = provider.as_batch_submitter().unwrap();
    let err = submitter
        .results(&BatchJobId::new("msgbatch_pending"))
        .await
        .err()
        .expect("must refuse on non-terminal");
    assert!(matches!(err, ProviderError::InvalidRequest(_)));
}

#[tokio::test]
async fn cancel_posts_to_cancel_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/batches/msgbatch_01abc/cancel"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msgbatch_01abc",
            "processing_status": "canceling"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = build_provider(&server);
    let submitter = provider.as_batch_submitter().unwrap();
    submitter
        .cancel(&BatchJobId::new("msgbatch_01abc"))
        .await
        .unwrap();
}

#[tokio::test]
async fn submit_propagates_http_error_via_classifier() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .respond_with(
            ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "type": "error",
                "error": {"type": "authentication_error", "message": "invalid x-api-key"}
            })),
        )
        .mount(&server)
        .await;

    let provider = build_provider(&server);
    let submitter = provider.as_batch_submitter().unwrap();
    let err = submitter
        .submit(vec![(
            BatchItemId::new("x"),
            ChatRequest::user(ModelHint::Explicit("claude-opus-4-7".into()), "hi"),
        )])
        .await
        .err()
        .expect("401 should error");
    // Adapter's classify_error maps 401 → Auth. We don't pin the kind to
    // avoid coupling to the adapter's classifier; just confirm it's a
    // typed error and not a Network/Parse fallthrough.
    assert!(
        !matches!(err, ProviderError::Network(_) | ProviderError::Parse(_)),
        "401 should not bubble as Network/Parse, got {err:?}"
    );
}
