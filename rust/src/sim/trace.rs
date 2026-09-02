//! Verbose per-solve trace channel for the paper sim.
//!
//! This is separate from the console logger and runs on a *much* tighter
//! interval than `platform_log_interval!` (one line every 100 steps). It feeds
//! the on-page sim console, which should scroll steadily — like an
//! `npm install` / `cargo build` log — with a mix of lines: a per-frame
//! header, fold progress, per-substep predict / solve / velocity stats, and
//! the inner-loop hinge residual.
//!
//! Each call site passes a `tag`; at most `TRACE_MAX_PER_STEP` lines per tag
//! per emitting step get through, so the once-per-substep lines all appear
//! while the hinge line (called thousands of times) is clamped.
//!
//! Lines are buffered in a bounded ring and handed to JS in one batch per
//! frame via `trace_drain` (see the `drain_paper_trace` wasm export).

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

use crate::platform_context::PlatformContext;

/// Emit trace lines only on every Nth sim step. 1 = every step.
const TRACE_INTERVAL: usize = 5;
/// Per-tag cap on lines kept per emitting step. Bump to see more substeps.
const TRACE_MAX_PER_STEP: usize = 1;
/// Hard cap on buffered lines; the oldest are dropped past this.
const MAX_BUFFER: usize = 8192;

/// Randomly drop otherwise-eligible trace lines to give the scroll some
/// jitter. Set to `false` for a fully deterministic stream.
const TRACE_RANDOMIZE: bool = true;
/// Probability in [0, 1) that an eligible line is dropped when
/// `TRACE_RANDOMIZE` is on. Dropped lines don't consume the per-tag budget.
const TRACE_DROP_PROB: f32 = 0.35;

thread_local! {
    static TRACE_RNG: Cell<u32> = const { Cell::new(0) };
}

/// xorshift32 in [0, 1). Cheap, no deps; lazily seeded from the wall clock so
/// the drop pattern varies between runs.
fn rand_unit() -> f32 {
    TRACE_RNG.with(|r| {
        let mut x = r.get();
        if x == 0 {
            x = (PlatformContext::now_ms().to_bits() as u32) | 1;
        }
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        r.set(x);
        (x >> 8) as f32 / (1u32 << 24) as f32
    })
}

struct TraceState {
    buf:  VecDeque<String>,
    tags: Vec<(&'static str, usize)>,
    step: Option<usize>,
    emit: bool,
}

thread_local! {
    static PAPER_TRACE: RefCell<TraceState> = RefCell::new(TraceState {
        buf:  VecDeque::new(),
        tags: Vec::new(),
        step: None,
        emit: false,
    });
}

/// Start a new sim step: reset the per-tag budgets, cache the step number for
/// line prefixes, and decide whether this step emits at all. Call once at the
/// top of each sim step.
pub fn trace_begin_frame() {
    let step = PlatformContext::with_logger(|l| l.step());
    PAPER_TRACE.with(|t| {
        let mut t = t.borrow_mut();
        t.tags.clear();
        t.step = step;
        t.emit = step.map_or(true, |s| TRACE_INTERVAL != 0 && s % TRACE_INTERVAL == 0);
    });
}

/// Whether the current step emits trace lines. Guard expensive trace-only
/// computations (norms, energy sums) with this.
pub fn trace_emitting() -> bool {
    PAPER_TRACE.with(|t| t.borrow().emit)
}

/// Push one trace line under `tag`, prefixed with the current sim step. No-ops
/// on non-emitting steps and once this tag's per-step budget is spent.
pub fn trace_push(tag: &'static str, msg: &str) {
    PAPER_TRACE.with(|t| {
        let mut t = t.borrow_mut();
        if !t.emit {
            return;
        }
        if TRACE_RANDOMIZE && rand_unit() < TRACE_DROP_PROB {
            return;
        }
        let idx = match t.tags.iter().position(|(k, _)| *k == tag) {
            Some(i) => i,
            None => {
                t.tags.push((tag, 0));
                t.tags.len() - 1
            }
        };
        if t.tags[idx].1 >= TRACE_MAX_PER_STEP {
            return;
        }
        t.tags[idx].1 += 1;

        let line = match t.step {
            Some(s) => format!("[step={}] {}", s, msg),
            None    => msg.to_string(),
        };
        t.buf.push_back(line);
        while t.buf.len() > MAX_BUFFER {
            t.buf.pop_front();
        }
    });
}

/// Drain every buffered line, joined by '\n'. Empty string when nothing is
/// queued.
pub fn trace_drain() -> String {
    PAPER_TRACE.with(|t| {
        let mut t = t.borrow_mut();
        if t.buf.is_empty() {
            return String::new();
        }
        Vec::from_iter(t.buf.drain(..)).join("\n")
    })
}

/// Emit a verbose trace line for the on-page sim console:
/// `paper_trace!("tag", "fmt {}", value)`. Formats like `format!`. Throttled
/// per tag by `TRACE_INTERVAL` / `TRACE_MAX_PER_STEP`, then randomly thinned by
/// `TRACE_RANDOMIZE` / `TRACE_DROP_PROB`.
#[macro_export]
macro_rules! paper_trace {
    ($tag:expr, $($arg:tt)*) => {
        $crate::sim::trace::trace_push($tag, &format!($($arg)*))
    };
}
