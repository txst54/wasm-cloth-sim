pub mod shared;
pub mod traits;
pub mod cloth_sim;
pub mod paper_sim;
pub mod crease;

pub use cloth_sim::ClothSim;
pub use paper_sim::{FoldDirection, FoldSpec, HingeConstraint, PaperSim};
pub use shared::{closest_point_on_triangle, Faces, Positions};
pub use traits::MeshSim;
pub use crease::{CreasePattern, CreaseType};
