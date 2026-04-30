//! Architecture-specific implementations.
//!
//! This module is the ONLY place cfg attributes appear outside arch/.

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

#[cfg(target_arch = "wasm32")]
pub use wasm::*;

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;
