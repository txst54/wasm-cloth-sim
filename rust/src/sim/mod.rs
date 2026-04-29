pub mod shared;
pub mod traits;
pub mod cloth_sim;
pub mod paper_sim;
pub mod rigid_sim;
pub mod crease;
pub mod particle_hash;
pub mod sdf;
pub mod particle_cloth_sim;

pub use cloth_sim::ClothSim;
pub use paper_sim::{FoldDirection, FoldSpec, HingeConstraint, PaperSim};
pub use rigid_sim::{RigidSimCore, RigidSimParams};
pub use shared::{closest_point_on_triangle, ClothSimCore, Faces, Positions, SimCore};
pub use traits::MeshSim;
pub use crease::{CreasePattern, CreaseType};
pub use particle_hash::ParticleHash;
pub use sdf::{SdfObstacle, SignedDistanceField};
pub use particle_cloth_sim::ParticleClothSim;
