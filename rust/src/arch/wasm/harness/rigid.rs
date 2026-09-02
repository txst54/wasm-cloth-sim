//! Standalone rigid-body demo: a tumbling cube, no cloth interaction.

use std::cell::RefCell;
use std::rc::Rc;

use nalgebra as na;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use super::{build_basics, get_canvas, run_raf};
use crate::arch::wasm::cloth::Cloth;
use crate::arch::wasm::gpu::GpuContext;
use crate::arch::wasm::input::{apply_camera_keys, install_keyboard_handlers, PickHost};
use crate::arch::wasm::light::Lighting;
use crate::arch::wasm::scene;
use crate::arch::wasm::Camera;
use crate::rigid_body::{RigidBodyInstance, RigidBodyTemplate};
use crate::sim::shared::Positions;
use crate::sim::{RigidSimCore, RigidSimParams};

thread_local! {
    pub(super) static RIGID_APP_STATE: RefCell<Option<Rc<RefCell<RigidAppState>>>> = RefCell::new(None);
}

pub(super) struct RigidAppState {
    ctx:    GpuContext,
    cloth:  Cloth,
    light:  Lighting,
    camera: Camera,
    sim:    RigidSimCore,
    params: RigidSimParams,
    canvas: HtmlCanvasElement,
    keys:   [bool; 8],
}

// Rigid demo: keyboard only (camera controls). Picking is unreachable.
impl PickHost for RigidAppState {
    fn camera(&self) -> &Camera               { &self.camera }
    fn canvas(&self) -> &HtmlCanvasElement     { &self.canvas }
    fn keys_mut(&mut self) -> &mut [bool; 8]   { &mut self.keys }
    fn positions(&self) -> &Positions          { unreachable!() }
    fn clicked_vertex(&self) -> Option<usize>  { None }
    fn set_clicked(&mut self, _idx: Option<usize>) {}
    fn set_mouse(&mut self, _pos: [f32; 3])    {}
}

#[wasm_bindgen]
pub async fn run_rigid(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let (window, canvas) = get_canvas(canvas_id)?;
    let (ctx, light, camera) = build_basics(canvas.clone()).await?;

    let template = RigidBodyTemplate::new(&scene::CUBE_VERTS, &scene::CUBE_FACES);
    let body = RigidBodyInstance::new(
        template,
        na::Vector3::zeros(),
        na::Vector3::zeros(),
        na::Vector3::zeros(),
        na::Vector3::new(1.0, 2.0, 0.5),
        1000.0,
    );
    let sim    = RigidSimCore::new(vec![body]);
    let params = RigidSimParams { gravity_enabled: false, ..Default::default() };
    let cloth  = scene::cube_cloth(&ctx, &light);

    let state = Rc::new(RefCell::new(RigidAppState {
        ctx, cloth, light, camera, sim, params,
        canvas: canvas.clone(), keys: [false; 8],
    }));
    RIGID_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    install_keyboard_handlers(state.clone(), &window)?;

    let state_inner = state.clone();
    run_raf(&window, move || {
        let mut s = state_inner.borrow_mut();
        let RigidAppState { sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        apply_camera_keys(camera, keys, ctx);
        sim.step(params);

        let world = sim.bodies[0].world_vertices();
        for (i, pos) in cloth.positions.iter_mut().enumerate() {
            *pos = [world[i][0], world[i][1], world[i][2]];
        }
        cloth.upload(ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            light.clear_shadow(ctx);
            cloth.render_shadow(ctx, light);
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }
    })
}
