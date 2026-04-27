//! Platform abstraction layer - traits for OS/runtime-specific services.
//!
//! Core simulation code uses these traits via dependency injection or
//! thread-local storage, never importing web_sys or native-specific code.

/// Logging abstraction - replaces web_sys::console
pub trait Logger: Send + Sync {
    fn log(&self, msg: &str);
    fn warn(&self, msg: &str);
    fn error(&self, msg: &str);
}

/// Timer abstraction - replaces web_sys::window().performance().now()
pub trait Timer: Send + Sync {
    /// Returns current time in milliseconds (monotonic).
    fn now_ms(&self) -> f64;
}

/// No-op logger for headless/testing
pub struct NullLogger;

impl Logger for NullLogger {
    fn log(&self, _: &str) {}
    fn warn(&self, _: &str) {}
    fn error(&self, _: &str) {}
}

/// No-op timer for testing
pub struct NullTimer;

impl Timer for NullTimer {
    fn now_ms(&self) -> f64 {
        0.0
    }
}
