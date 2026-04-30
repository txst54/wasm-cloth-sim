//! Native platform implementation using std.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use crate::platform::{Logger, Timer};
use crate::platform_context::PlatformContext;

pub struct NativeLogger {
    step: AtomicUsize,
    interval_step: AtomicUsize,
    interval_count: AtomicUsize,
}

impl NativeLogger {
    pub fn new() -> Self {
        Self {
            step: AtomicUsize::new(usize::MAX),
            interval_step: AtomicUsize::new(usize::MAX),
            interval_count: AtomicUsize::new(0),
        }
    }
}

impl Default for NativeLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl Logger for NativeLogger {
    fn log(&self, msg: &str) {
        let s = self.step.load(Ordering::Relaxed);
        if s == usize::MAX {
            println!("[LOG] {}", msg);
        } else {
            println!("[LOG step={}] {}", s, msg);
        }
    }
    fn warn(&self, msg: &str) {
        let s = self.step.load(Ordering::Relaxed);
        if s == usize::MAX {
            eprintln!("[WARN] {}", msg);
        } else {
            eprintln!("[WARN step={}] {}", s, msg);
        }
    }
    fn error(&self, msg: &str) {
        let s = self.step.load(Ordering::Relaxed);
        if s == usize::MAX {
            eprintln!("[ERROR] {}", msg);
        } else {
            eprintln!("[ERROR step={}] {}", s, msg);
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

        println!("[LOG step={}] {}", s, msg);
    }
}

pub struct NativeTimer {
    start: Instant,
}

impl NativeTimer {
    pub fn new() -> Self {
        Self { start: Instant::now() }
    }
}

impl Default for NativeTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer for NativeTimer {
    fn now_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

/// Initialize the native platform context. Call this early in main().
pub fn init_platform() {
    PlatformContext::init(PlatformContext {
        logger: Arc::new(NativeLogger::new()),
        timer: Arc::new(NativeTimer::new()),
    });
}
