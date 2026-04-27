//! Native platform implementation using std.

use std::sync::Arc;
use std::time::Instant;
use crate::platform::{Logger, Timer};
use crate::platform_context::PlatformContext;

pub struct NativeLogger;

impl Logger for NativeLogger {
    fn log(&self, msg: &str) {
        println!("[LOG] {}", msg);
    }
    fn warn(&self, msg: &str) {
        eprintln!("[WARN] {}", msg);
    }
    fn error(&self, msg: &str) {
        eprintln!("[ERROR] {}", msg);
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
        logger: Arc::new(NativeLogger),
        timer: Arc::new(NativeTimer::new()),
    });
}
