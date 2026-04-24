pub mod shared;
pub mod traits;
pub mod cloth_sim;
pub mod paper_sim;
pub mod rigid_sim;
pub mod crease;

pub use cloth_sim::ClothSim;
pub use paper_sim::{FoldDirection, FoldSpec, HingeConstraint, PaperSim};
pub use rigid_sim::{RigidSimCore, RigidSimParams};
pub use shared::{closest_point_on_triangle, ClothSimCore, Faces, Positions, SimCore};
pub use traits::MeshSim;
pub use crease::{CreasePattern, CreaseType};
