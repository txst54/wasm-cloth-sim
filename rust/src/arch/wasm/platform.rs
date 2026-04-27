//! WASM platform implementation using web_sys.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::platform::{Logger, Timer};
use crate::platform_context::PlatformContext;

pub struct WasmLogger {
    step: AtomicUsize,
    interval_step: AtomicUsize,
    interval_count: AtomicUsize,
}

impl WasmLogger {
    pub fn new() -> Self {
        Self {
            step: AtomicUsize::new(usize::MAX),
            interval_step: AtomicUsize::new(usize::MAX),
            interval_count: AtomicUsize::new(0),
        }
    }
}

impl Default for WasmLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl Logger for WasmLogger {
    fn log(&self, msg: &str) {
        let s = self.step.load(Ordering::Relaxed);
        if s == usize::MAX {
            web_sys::console::log_1(&msg.into());
        } else {
            web_sys::console::log_1(&format!("[step={}] {}", s, msg).into());
        }
    }
    fn warn(&self, msg: &str) {
        let s = self.step.load(Ordering::Relaxed);
        if s == usize::MAX {
            web_sys::console::warn_1(&msg.into());
        } else {
            web_sys::console::warn_1(&format!("[step={}] {}", s, msg).into());
        }
    }
    fn error(&self, msg: &str) {
        let s = self.step.load(Ordering::Relaxed);
        if s == usize::MAX {
            web_sys::console::error_1(&msg.into());
        } else {
            web_sys::console::error_1(&format!("[step={}] {}", s, msg).into());
        }
    }
    fn set_step(&self, step: usize) {
        self.step.store(step, Ordering::Relaxed);
    }
    fn step(&self) -> Option<usize> {
        let s = self.step.load(Ordering::Relaxed);
        if s == usize::MAX { None } else { Some(s) }
    }
    fn log_interval(&self, msg: &str, interval: usize, max_per_step: i32) {
        let s = self.step.load(Ordering::Relaxed);
        if s == usize::MAX || interval == 0 { return; }
        if s % interval != 0 { return; }

        let last_step = self.interval_step.load(Ordering::Relaxed);
        if last_step != s {
            self.interval_step.store(s, Ordering::Relaxed);
            self.interval_count.store(0, Ordering::Relaxed);
        }

        if max_per_step >= 0 {
            let count = self.interval_count.fetch_add(1, Ordering::Relaxed);
            if count >= max_per_step as usize { return; }
        }

        web_sys::console::log_1(&format!("[step={}] {}", s, msg).into());
    }
}

pub struct WasmTimer;

impl Timer for WasmTimer {
    fn now_ms(&self) -> f64 {
        web_sys::window()
            .expect("no window")
            .performance()
            .expect("no performance")
            .now()
    }
}

/// Initialize the WASM platform context. Call this early in WASM entry points.
pub fn init_platform() {
    PlatformContext::init(PlatformContext {
        logger: Arc::new(WasmLogger::new()),
        timer: Arc::new(WasmTimer),
    });
}
