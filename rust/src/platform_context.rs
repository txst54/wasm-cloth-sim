//! Thread-local platform services context.

use std::cell::RefCell;
use std::sync::Arc;
use crate::platform::{Logger, Timer, NullLogger, NullTimer};

thread_local! {
    static PLATFORM: RefCell<Option<PlatformContext>> = const { RefCell::new(None) };
}

pub struct PlatformContext {
    pub logger: Arc<dyn Logger>,
    pub timer: Arc<dyn Timer>,
}

impl PlatformContext {
    pub fn init(ctx: PlatformContext) {
        PLATFORM.with(|p| *p.borrow_mut() = Some(ctx));
    }

    pub fn with_logger<F, R>(f: F) -> R
    where
        F: FnOnce(&dyn Logger) -> R,
    {
        PLATFORM.with(|p| {
            let guard = p.borrow();
            match guard.as_ref() {
                Some(ctx) => f(ctx.logger.as_ref()),
                None => f(&NullLogger),
            }
        })
    }

    pub fn now_ms() -> f64 {
        PLATFORM.with(|p| {
            let guard = p.borrow();
            guard.as_ref().map_or(0.0, |ctx| ctx.timer.now_ms())
        })
    }
}

/// Log an informational message via the platform logger.
#[macro_export]
macro_rules! platform_log {
    ($($arg:tt)*) => {
        $crate::platform_context::PlatformContext::with_logger(|l| l.log(&format!($($arg)*)))
    };
}

/// Log a warning message via the platform logger.
#[macro_export]
macro_rules! platform_warn {
    ($($arg:tt)*) => {
        $crate::platform_context::PlatformContext::with_logger(|l| l.warn(&format!($($arg)*)))
    };
}

/// Log an error message via the platform logger.
#[macro_export]
macro_rules! platform_error {
    ($($arg:tt)*) => {
        $crate::platform_context::PlatformContext::with_logger(|l| l.error(&format!($($arg)*)))
    };
}
