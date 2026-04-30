use std::collections::HashSet;
use std::ops::{Deref, DerefMut};

use crate::params::SimParams;
use super::shared::{Positions, ClothSimCore};
use super::traits::MeshSim;

/// Plain cloth simulation — full PBD with stretch, bending, pin, pull, and
/// self-collision.  All state lives in the inner [`SimCore`]; this struct adds
/// no extra data, but `Deref` / `DerefMut` let callers access `q`,
/// `clicked_vertex`, `mouse_pos`, etc. directly (maintaining the old API).
pub struct ClothSim {
    pub core: ClothSimCore,
}

impl Deref for ClothSim {
    type Target = ClothSimCore;
    fn deref(&self) -> &ClothSimCore { &self.core }
}

impl DerefMut for ClothSim {
    fn deref_mut(&mut self) -> &mut ClothSimCore { &mut self.core }
}

impl ClothSim {
    /// Create a ClothSim from an NxN grid.
    pub fn from_grid(resolution: usize) -> Self {
        let n = resolution;
        let pinned = vec![(n-1)*n, (n-1)*n + (n-1)]; // upper-left, upper-right
        Self { core: ClothSimCore::from_grid(n, &pinned) }
    }

    pub fn step(&mut self, params: &SimParams) {
        self.core.step(params, &HashSet::new());
    }
}

impl MeshSim for ClothSim {
    fn step(&mut self, params: &SimParams)              { self.core.step(params, &HashSet::new()); }
    fn positions(&self) -> &Positions                   { &self.core.q }
    fn set_clicked_vertex(&mut self, vi: Option<usize>) { self.core.clicked_vertex = vi; }
    fn set_mouse_pos(&mut self, pos: [f32; 3])          { self.core.mouse_pos = pos; }
}
