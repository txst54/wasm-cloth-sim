//! WASM platform implementation using web_sys.

use std::sync::Arc;
use crate::platform::{Logger, Timer};
use crate::platform_context::PlatformContext;

pub struct WasmLogger;

impl Logger for WasmLogger {
    fn log(&self, msg: &str) {
        web_sys::console::log_1(&msg.into());
    }
    fn warn(&self, msg: &str) {
        web_sys::console::warn_1(&msg.into());
    }
    fn error(&self, msg: &str) {
        web_sys::console::error_1(&msg.into());
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
        logger: Arc::new(WasmLogger),
        timer: Arc::new(WasmTimer),
    });
}
