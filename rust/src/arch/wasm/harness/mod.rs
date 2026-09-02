//! WASM entry points and render loops.
//!
//! Each `run_*` function wires up a canvas, a sim, and a render loop. Shared
//! input handling lives in [`super::input`], scene constants and procedural
//! mesh builders in [`super::scene`], and the FPS overlay in [`super::fps`].
//!
//! This module is split by sim: [`basic_cloth`] and [`combined`] hold the
//! plain cloth-only and cloth+rigid-cube demos, [`paper`] and
//! [`particle_paper`] the rigid-facet paper folding sims (mass-spring and
//! particle respectively), [`rigid`] the standalone rigid-body demo, and
//! [`particle_cloth`] / [`head_cloth`] the particle cloth sim and its
//! OBJ-mesh-obstacle variant. This file (`mod.rs`) holds only plumbing shared
//! across all of them.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use super::camera::Camera;
use super::gpu::GpuContext;
use super::light::Lighting;
use super::scene;
use crate::params::SimParams;

mod basic_cloth;
mod combined;
mod head_cloth;
mod particle_cloth;
mod particle_paper;
mod paper;
mod rigid;

pub use basic_cloth::*;
pub use combined::*;
pub use head_cloth::*;
pub use particle_cloth::*;
pub use particle_paper::*;
pub use paper::*;
pub use rigid::*;

// ── Thread-locals ─────────────────────────────────────────────────────────────

thread_local! {
    pub(super) static PARAMS: Rc<RefCell<SimParams>> = Rc::new(RefCell::new(SimParams::default()));
}

// ── Common scene-construction helpers ─────────────────────────────────────────

pub(super) fn get_canvas(canvas_id: &str) -> Result<(web_sys::Window, HtmlCanvasElement), JsValue> {
    let window = web_sys::window().unwrap();
    let canvas = window
        .document().unwrap()
        .get_element_by_id(canvas_id).unwrap()
        .dyn_into::<HtmlCanvasElement>()?;
    Ok((window, canvas))
}

/// `(ctx, light, camera)` — every entry point starts the same way.
pub(super) async fn build_basics(canvas: HtmlCanvasElement) -> Result<(GpuContext, Lighting, Camera), JsValue> {
    let ctx    = GpuContext::new(canvas).await?;
    let light  = scene::make_light(&ctx);
    let camera = Camera::new(&ctx);
    Ok((ctx, light, camera))
}

/// Schedule `body` as a `requestAnimationFrame` loop. `body` is invoked once
/// per frame; this helper handles the self-rescheduling boilerplate.
pub(super) fn run_raf<F: FnMut() + 'static>(window: &web_sys::Window, mut body: F) -> Result<(), JsValue> {
    let loop_fn: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let loop_fn_inner = loop_fn.clone();
    *loop_fn.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        body();
        web_sys::window().unwrap().request_animation_frame(
            loop_fn_inner.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
        ).unwrap();
    }) as Box<dyn FnMut()>));
    window.request_animation_frame(
        loop_fn.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
    )?;
    Ok(())
}

// ── PickHost impl helper ──────────────────────────────────────────────────────

/// Boilerplate for `PickHost` impls that delegate to a sim field.
/// `$path` names the sim path (e.g. `sim` or `sim.core`); `clears_drag` is
/// `true` for sims that own a `dragging_vertices` field to reset on pick.
macro_rules! impl_pickhost {
    ($state:ty, $($path:ident).+, clears_drag = $drag:tt) => {
        impl crate::arch::wasm::input::PickHost for $state {
            fn camera(&self) -> &Camera             { &self.camera }
            fn canvas(&self) -> &HtmlCanvasElement   { &self.canvas }
            fn keys_mut(&mut self) -> &mut [bool; 8] { &mut self.keys }
            fn positions(&self) -> &crate::sim::shared::Positions { &self.$($path).+.q }
            fn clicked_vertex(&self) -> Option<usize> { self.$($path).+.clicked_vertex }
            fn set_clicked(&mut self, idx: Option<usize>) {
                self.$($path).+.clicked_vertex = idx;
                impl_pickhost!(@drag self, $($path).+, $drag);
            }
            fn set_mouse(&mut self, pos: [f32; 3]) { self.$($path).+.mouse_pos = pos; }
        }
    };
    (@drag $self:ident, $($path:ident).+, true)  => {
        if $self.$($path).+.clicked_vertex.is_some() {
            $self.$($path).+.dragging_vertices = None;
        }
    };
    (@drag $self:ident, $($path:ident).+, false) => {};
}
pub(super) use impl_pickhost;

// ── Param setters (called from JS panel) ──────────────────────────────────────

macro_rules! param_setter {
    ($name:ident, $field:ident, $ty:ty) => {
        #[wasm_bindgen] pub fn $name(v: $ty) { PARAMS.with(|p| p.borrow_mut().$field = v); }
    };
    ($name:ident, $field:ident, $ty:ty, |$v:ident| $expr:expr) => {
        #[wasm_bindgen] pub fn $name($v: $ty) { PARAMS.with(|p| p.borrow_mut().$field = $expr); }
    };
}

param_setter!(set_time_step,                       time_step,                       f64);
param_setter!(set_constraint_iters,                constraint_iters,                u32);
param_setter!(set_num_substeps,                    num_substeps,                    u32, |v| v.max(1));
param_setter!(set_stretch_compliance,              stretch_compliance,              f64);
param_setter!(set_bend_compliance,                 bend_compliance,                 f64);
param_setter!(set_gravity_enabled,                 gravity_enabled,                 bool);
param_setter!(set_gravity_g,                       gravity_g,                       f64);
param_setter!(set_pin_enabled,                     pin_enabled,                     bool);
param_setter!(set_pin_weight,                      pin_weight,                      f64);
param_setter!(set_stretch_enabled,                 stretch_enabled,                 bool);
param_setter!(set_stretch_weight,                  stretch_weight,                  f64);
param_setter!(set_bending_enabled,                 bending_enabled,                 bool);
param_setter!(set_bending_weight,                  bending_weight,                  f64);
param_setter!(set_pulling_enabled,                 pulling_enabled,                 bool);
param_setter!(set_pulling_weight,                  pulling_weight,                  f64);
param_setter!(set_self_collision_enabled,          self_collision_enabled,          bool);
param_setter!(set_self_collision_threshold,        self_collision_threshold,        f64);
param_setter!(set_self_collision_recompute_iters,  self_collision_recompute_iters,  u16);
param_setter!(set_use_distance_constraints,        use_distance_constraints,        bool);
param_setter!(set_pulling_area,                    pulling_area,                    u32);
param_setter!(set_damping,                         damping,                         f64);
param_setter!(set_friction_enabled,                friction_enabled,                bool);
param_setter!(set_friction_mu,                     friction_mu,                     f64);
param_setter!(set_cloth_friction_enabled,          cloth_friction_enabled,          bool);
param_setter!(set_cloth_friction_d,                cloth_friction_d,                f64);
