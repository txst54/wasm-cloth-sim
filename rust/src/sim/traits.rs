use crate::params::SimParams;
use super::shared::Positions;

/// Common interface for fabric-type PBD simulations.
///
/// Both `ClothSim` and `PaperSim` expose these methods so they can be used
/// interchangeably behind a `dyn MeshSim` trait object when needed.
pub trait MeshSim {
    fn step(&mut self, params: &SimParams);
    fn positions(&self) -> &Positions;
    fn set_clicked_vertex(&mut self, vi: Option<usize>);
    fn set_mouse_pos(&mut self, pos: [f32; 3]);
}
