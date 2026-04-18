use std::collections::HashSet;
use std::ops::{Deref, DerefMut};

use crate::{cloth::Cloth, gpu::GpuContext, params::SimParams};
use super::shared::{Positions, SimCore};
use super::traits::MeshSim;

/// Plain cloth simulation — full PBD with stretch, bending, pin, pull, and
/// self-collision.  All state lives in the inner [`SimCore`]; this struct adds
/// no extra data, but `Deref` / `DerefMut` let callers access `q`,
/// `clicked_vertex`, `mouse_pos`, etc. directly (maintaining the old API).
pub struct ClothSim {
    pub core: SimCore,
}

impl Deref for ClothSim {
    type Target = SimCore;
    fn deref(&self) -> &SimCore { &self.core }
}

impl DerefMut for ClothSim {
    fn deref_mut(&mut self) -> &mut SimCore { &mut self.core }
}

impl ClothSim {
    pub fn from_cloth(cloth: &Cloth) -> Self {
        let n = cloth.resolution as usize;
        let pinned = vec![(n-1)*n, (n-1)*n + (n-1)]; // upper-left, upper-right
        Self { core: SimCore::from_cloth(cloth, &pinned) }
    }

    pub fn step(&mut self, params: &SimParams) {
        self.core.step(params, &HashSet::new(), |_, _| {});
    }

    pub fn write_to_cloth(&self, cloth: &mut Cloth, ctx: &GpuContext) {
        self.core.write_to_cloth(cloth, ctx);
    }
}

impl MeshSim for ClothSim {
    fn step(&mut self, params: &SimParams)                        { self.core.step(params, &HashSet::new(), |_, _| {}); }
    fn write_to_cloth(&self, cloth: &mut Cloth, ctx: &GpuContext) { self.core.write_to_cloth(cloth, ctx); }
    fn positions(&self) -> &Positions                             { &self.core.q }
    fn set_clicked_vertex(&mut self, vi: Option<usize>)           { self.core.clicked_vertex = vi; }
    fn set_mouse_pos(&mut self, pos: [f32; 3])                    { self.core.mouse_pos = pos; }
}
