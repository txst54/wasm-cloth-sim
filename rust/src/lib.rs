//! Core simulation library for cloth and paper physics.
//!
//! This crate provides platform-independent simulation code. Architecture-specific
//! code (WASM rendering, native CLI) lives in the `arch` module.

// Platform abstraction
pub mod platform;
pub mod platform_context;

// Core simulation
pub mod params;
pub mod sim;
pub mod bvh;
pub mod rigid_body;

// WASM-only triangle demo (depends on GPU)
#[cfg(target_arch = "wasm32")]
mod triangle;

// Architecture-specific code
pub mod arch;

// Re-exports for convenience
pub use params::SimParams;
pub use sim::{
    ClothSim, ClothSimCore, CreasePattern, CreaseType,
    FoldDirection, FoldSpec, PaperSim, Positions, Faces,
    RigidSimCore, RigidSimParams, MeshSim,
};
pub use rigid_body::{RigidBodyInstance, RigidBodyTemplate};
