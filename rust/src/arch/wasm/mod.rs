//! WASM architecture implementation.

pub mod platform;
pub mod gpu;
pub mod pipeline;
pub mod light;
pub mod camera;
pub mod cloth;
pub mod scene;
pub mod input;
pub mod fps;
pub mod harness;

pub use gpu::GpuContext;
pub use light::Lighting;
pub use camera::Camera;
pub use cloth::{Cloth, Material};
pub use harness::*;
