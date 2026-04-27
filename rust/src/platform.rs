//! Platform abstraction layer - traits for OS/runtime-specific services.
//!
//! Core simulation code uses these traits via dependency injection or
//! thread-local storage, never importing web_sys or native-specific code.

/// Logging abstraction - replaces web_sys::console
pub trait Logger: Send + Sync {
    fn log(&self, msg: &str);
    fn warn(&self, msg: &str);
    fn error(&self, msg: &str);
    fn set_step(&self, step: usize);
    fn step(&self) -> Option<usize>;
    fn log_interval(&self, msg: &str, interval: usize, max_per_step: i32);
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
    fn set_step(&self, _: usize) {}
    fn step(&self) -> Option<usize> { None }
    fn log_interval(&self, _: &str, _: usize, _: i32) {}
}

/// No-op timer for testing
pub struct NullTimer;

impl Timer for NullTimer {
    fn now_ms(&self) -> f64 {
        0.0
    }
}
