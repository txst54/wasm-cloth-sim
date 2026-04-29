pub mod shared;
pub mod traits;
pub mod cloth_sim;
pub mod paper_sim;
pub mod rigid_sim;
pub mod crease;
pub mod particle_hash;
pub mod sdf;
pub mod particle_cloth_sim;
pub mod particle_paper_sim;

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub mod particle_cloth_sim_gpu;
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub mod particle_paper_sim_gpu;

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
compile_error!("the `gpu` feature requires target_arch = \"wasm32\"");

pub use cloth_sim::ClothSim;
pub use paper_sim::{FoldDirection, FoldSpec, HingeConstraint, PaperSim};
pub use rigid_sim::{RigidSimCore, RigidSimParams};
pub use shared::{closest_point_on_triangle, ClothSimCore, Faces, Positions, SimCore};
pub use traits::MeshSim;
pub use crease::{CreasePattern, CreaseType};
pub use particle_hash::ParticleHash;
pub use sdf::{SdfObstacle, SignedDistanceField};
pub use particle_cloth_sim::ParticleClothSim;
pub use particle_paper_sim::{ParticleHinge, ParticlePaperSim};

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub use particle_cloth_sim_gpu::ParticleClothSimGpu;
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub use particle_paper_sim_gpu::ParticlePaperSimGpu;
