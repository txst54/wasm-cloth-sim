pub mod shared;
pub mod traits;
pub mod cloth_sim;
pub mod paper_sim;

pub use cloth_sim::ClothSim;
pub use paper_sim::{FoldSpec, HingeConstraint, PaperSim};
pub use shared::{closest_point_on_triangle, Faces, Positions};
pub use traits::MeshSim;
