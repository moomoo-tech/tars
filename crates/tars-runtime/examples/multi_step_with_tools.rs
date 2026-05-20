//! A complex multi-agent demo, no live LLM required.
//!
//! Plan shape (3 steps, emitted by a scripted Orchestrator):
//!
//! ```text
//!   s1 [enumerate]    Worker calls fs.list_dir, then fs.read_file,
//!                     summarises what it found. Critic approves.
//!   s2 [cross_check]  Worker reads a second file. Critic refines once
//!                     ("be more concrete"). Worker retries with the
//!                     suggestion threaded in. Critic approves.
//!   s3 [report]       Worker writes a final summary, no tools. Critic
//!                     approves.
//! ```
//!
//! Run:
//! ```bash
//! cargo run -p tars-runtime --example multi_step_with_tools
//! ```
//!
//! Output is the full event log (`tars trajectory show`-equivalent),
//! pretty-printed. Total ≈ 29 events across 9 agent invocations.
//!
//! The LLM is a `ScriptedProvider` that pops a `Vec<ChatEvent>` per
//! call from a FIFO — i.e. exactly what a real provider would stream
//! if a model honoured the strict schemas. Tools (`fs.list_dir`,
//! `fs.read_file`) are real and jailed to a tempdir.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;

use tars_pipeline::{LlmService, Pipeline, ProviderService};
use tars_provider::{LlmEventStream, LlmProvider};
use tars_runtime::{
    AgentEvent, CriticAgent, LocalRuntime, OrchestratorAgent, RunTaskConfig, Runtime, WorkerAgent,
    run_task,
};
use tars_storage::{EventStore, SqliteEventStore, SqliteEventStoreConfig};
use tars_tools::{
    ToolRegistry,
    builtins::{ListDirTool, ReadFileTool},
};
use tars_types::{
    AgentId, Capabilities, ChatEvent, ChatRequest, Pricing, ProviderError, ProviderId,
    RequestContext, StopReason, Usage,
};
use tokio_util::sync::CancellationToken;

// ── Provider that pops a canned event sequence per call ────────────────

struct ScriptedProvider {
    id: ProviderId,
    capabilities: Capabilities,
    queue: Mutex<VecDeque<Vec<ChatEvent>>>,
}

impl ScriptedProvider {
    fn new(sequences: Vec<Vec<ChatEvent>>) -> Arc<Self> {
        Arc::new(Self {
            id: ProviderId::new("scripted"),
            capabilities: Capabilities::text_only_baseline(Pricing::default()),
            queue: Mutex::new(sequences.into_iter().collect()),
        })
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }
    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
    async fn stream(
        self: Arc<Self>,
        _req: ChatRequest,
        _ctx: RequestContext,
    ) -> Result<LlmEventStream, ProviderError> {
        let next = self.queue.lock().unwrap().pop_front().ok_or_else(|| {
            ProviderError::Internal("ScriptedProvider: queue empty — script underran".into())
        })?;
        let mapped: Vec<Result<ChatEvent, ProviderError>> = next.into_iter().map(Ok).collect();
        Ok(Box::pin(stream::iter(mapped)))
    }
}

// ── Canned event-stream builders ──────────────────────────────────────

fn text_stream(json: impl Into<String>) -> Vec<ChatEvent> {
    let text = json.into();
    let out_tokens = (text.len() / 4) as u64;
    vec![
        ChatEvent::started("scripted-model"),
        ChatEvent::Delta { text },
        ChatEvent::Finished {
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 50,
                output_tokens: out_tokens,
                ..Default::default()
            },
        },
    ]
}

fn tool_stream(call_id: &str, name: &str, args: serde_json::Value) -> Vec<ChatEvent> {
    vec![
        ChatEvent::started("scripted-model"),
        ChatEvent::ToolCallStart {
            index: 0,
            id: call_id.to_string(),
            name: name.to_string(),
        },
        ChatEvent::ToolCallEnd {
            index: 0,
            id: call_id.to_string(),
            parsed_args: args,
        },
        ChatEvent::Finished {
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                input_tokens: 50,
                output_tokens: 5,
                ..Default::default()
            },
        },
    ]
}

fn plan_json(s1_file: &str, s2_file: &str) -> String {
    serde_json::json!({
        "plan_id": "p-demo",
        "goal": "Enumerate the workspace and cross-check two files",
        "steps": [
            {
                "id": "s1",
                "worker_role": "enumerate",
                "instruction":
                    format!("List the demo dir, then read {s1_file}, summarise contents."),
                "depends_on": []
            },
            {
                "id": "s2",
                "worker_role": "cross_check",
                "instruction":
                    format!("Read {s2_file} and cross-check it against the s1 summary."),
                "depends_on": ["s1"]
            },
            {
                "id": "s3",
                "worker_role": "report",
                "instruction": "Write a one-line final report combining s1 and s2.",
                "depends_on": ["s2"]
            }
        ]
    })
    .to_string()
}

fn worker_final(summary: &str, confidence: f64) -> String {
    serde_json::json!({ "summary": summary, "confidence": confidence }).to_string()
}

fn approve() -> String {
    r#"{"kind":"approve","reason":"looks good","suggestions":[]}"#.to_string()
}

fn refine(suggestion: &str) -> String {
    serde_json::json!({
        "kind": "refine",
        "reason": "needs more concrete detail",
        "suggestions": [suggestion],
    })
    .to_string()
}

// ── Main ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Tempdir + two real files for the tools to read.
    let dir = tempfile::tempdir()?;
    let f1 = dir.path().join("alpha.txt");
    let f2 = dir.path().join("beta.txt");
    tokio::fs::write(&f1, b"alpha: hello world from alpha").await?;
    tokio::fs::write(&f2, b"beta: cross-reference data point").await?;

    // 2. Tool registry — jailed to the tempdir.
    let mut reg = ToolRegistry::new();
    reg.register_owned(
        ListDirTool::with_root(dir.path()).expect("list_dir root accepted"),
    )?;
    reg.register_owned(
        ReadFileTool::with_root(dir.path()).expect("read_file root accepted"),
    )?;
    let registry = Arc::new(reg);

    // 3. Scripted LLM responses, in call order.
    //
    //   1. Orchestrator: 3-step plan
    //   2. s1 Worker call A: tool_call fs.list_dir
    //   3. s1 Worker call B: tool_call fs.read_file(alpha.txt)
    //   4. s1 Worker call C: final JSON
    //   5. s1 Critic: approve
    //   6. s2 Worker attempt 1, call A: tool_call fs.read_file(beta.txt)
    //   7. s2 Worker attempt 1, call B: final JSON (vague)
    //   8. s2 Critic: refine
    //   9. s2 Worker attempt 2, call A: tool_call fs.read_file(beta.txt)
    //  10. s2 Worker attempt 2, call B: final JSON (concrete)
    //  11. s2 Critic: approve
    //  12. s3 Worker: final JSON (no tool call)
    //  13. s3 Critic: approve
    let provider = ScriptedProvider::new(vec![
        // 1. Orchestrator
        text_stream(plan_json(
            f1.to_str().unwrap(),
            f2.to_str().unwrap(),
        )),
        // 2-4. s1 Worker (3 internal LLM calls; one trajectory step)
        tool_stream(
            "c1",
            "fs.list_dir",
            serde_json::json!({ "path": dir.path().to_str().unwrap() }),
        ),
        tool_stream(
            "c2",
            "fs.read_file",
            serde_json::json!({ "path": f1.to_str().unwrap() }),
        ),
        text_stream(worker_final(
            "alpha.txt: greeting payload, 30 bytes",
            0.85,
        )),
        // 5. s1 Critic
        text_stream(approve()),
        // 6-7. s2 Worker attempt 1
        tool_stream(
            "c3",
            "fs.read_file",
            serde_json::json!({ "path": f2.to_str().unwrap() }),
        ),
        text_stream(worker_final("beta.txt has some data", 0.5)),
        // 8. s2 Critic refine
        text_stream(refine("cite byte length and topic, like the s1 summary did")),
        // 9-10. s2 Worker attempt 2
        tool_stream(
            "c4",
            "fs.read_file",
            serde_json::json!({ "path": f2.to_str().unwrap() }),
        ),
        text_stream(worker_final(
            "beta.txt: cross-reference data, 32 bytes; aligns with alpha greeting",
            0.9,
        )),
        // 11. s2 Critic approve
        text_stream(approve()),
        // 12. s3 Worker (no tool call — straight to final)
        text_stream(worker_final(
            "alpha + beta: greeting paired with cross-reference; workspace is consistent",
            0.95,
        )),
        // 13. s3 Critic approve
        text_stream(approve()),
    ]);

    // 4. Pipeline + Runtime (SQLite event store on tempdir).
    let provider_svc: Arc<dyn LlmService> = ProviderService::new(provider);
    let llm: Arc<dyn LlmService> =
        Arc::new(Pipeline::builder_with_inner(provider_svc).build());
    let events_path = dir.path().join("events.sqlite");
    let store: Arc<dyn EventStore> =
        SqliteEventStore::open(SqliteEventStoreConfig::new(&events_path))?;
    let runtime = LocalRuntime::new(store);

    // 5. Agent triad. Worker uses tools; Orchestrator + Critic don't.
    let orch = OrchestratorAgent::new(AgentId::new("orch"), "scripted-model");
    let worker = WorkerAgent::with_tools(
        AgentId::new("worker"),
        "scripted-model",
        "multi_role",
        registry,
    );
    let critic = CriticAgent::new(AgentId::new("critic"), "scripted-model");

    // 6. Drive the loop.
    let outcome = run_task(
        runtime.clone() as Arc<dyn Runtime>,
        llm,
        orch,
        worker,
        critic,
        "Enumerate the workspace and cross-check two files",
        RunTaskConfig::default(),
        CancellationToken::new(),
    )
    .await?;

    // 7. Print the high-level outcome.
    println!("╭─ TaskOutcome ─────────────────────────────────────────────");
    println!("│ trajectory: {}", outcome.trajectory_id);
    println!("│ plan_id:    {}", outcome.plan.plan_id);
    println!("│ steps:      {}", outcome.steps.len());
    for s in &outcome.steps {
        // worker_role is on the plan, indexed by step_id.
        let role = outcome
            .plan
            .steps
            .iter()
            .find(|ps| ps.id == s.step_id)
            .map(|ps| ps.worker_role.as_str())
            .unwrap_or("?");
        println!(
            "│   - {} (worker_role={}, refine_attempts={})",
            s.step_id, role, s.refinement_attempts,
        );
    }
    println!("╰───────────────────────────────────────────────────────────");
    println!();

    // 8. Replay the trajectory event log.
    let events = runtime.replay(&outcome.trajectory_id).await?;
    println!("╭─ Trajectory event log ({} events) ──────────────", events.len());
    for (i, ev) in events.iter().enumerate() {
        let line = match ev {
            AgentEvent::TrajectoryStarted { reason, .. } => {
                format!("TrajectoryStarted   reason={reason:?}")
            }
            AgentEvent::StepStarted {
                step_seq,
                agent,
                idempotency_key,
                input_summary,
                ..
            } => format!(
                "StepStarted  #{step_seq} agent={agent} idem={} input={input_summary:.80}",
                &idempotency_key.as_str()[..8],
            ),
            AgentEvent::LlmCallCaptured {
                step_seq,
                provider,
                usage,
                ..
            } => format!(
                "LlmCallCaptured #{step_seq} provider={provider} in/out={}/{} tokens",
                usage.input_tokens, usage.output_tokens,
            ),
            AgentEvent::StepCompleted {
                step_seq,
                output_summary,
                usage,
                ..
            } => format!(
                "StepCompleted #{step_seq} out={output_summary:.60} usage=in/out={}/{}",
                usage.input_tokens, usage.output_tokens,
            ),
            AgentEvent::StepFailed {
                step_seq,
                error,
                classification,
                ..
            } => format!(
                "StepFailed   #{step_seq} cls={classification:?} err={error}",
            ),
            AgentEvent::TrajectorySuspended { reason, .. } => {
                format!("TrajectorySuspended reason={reason}")
            }
            AgentEvent::TrajectoryAbandoned { cause, .. } => {
                format!("TrajectoryAbandoned cause={cause}")
            }
            AgentEvent::TrajectoryCompleted { summary, .. } => {
                format!("TrajectoryCompleted summary={summary:?}")
            }
        };
        println!("│ [{i:>2}] {line}");
    }
    println!("╰────────────────────────────────────────────────────────────");

    Ok(())
}
