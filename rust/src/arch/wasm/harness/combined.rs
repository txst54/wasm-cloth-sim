//! Combined cloth + rigid cube demo: a swinging cube collides with a cloth sheet.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use nalgebra as na;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use super::{build_basics, get_canvas, run_raf, PARAMS};
use crate::arch::wasm::cloth::Cloth;
use crate::arch::wasm::fps::FpsOverlay;
use crate::arch::wasm::gpu::GpuContext;
use crate::arch::wasm::input::{apply_camera_keys, install_handlers, PickHost};
use crate::arch::wasm::light::Lighting;
use crate::arch::wasm::platform::init_platform;
use crate::arch::wasm::scene;
use crate::arch::wasm::Camera;
use crate::params::SimParams;
use crate::rigid_body::{RigidBodyInstance, RigidBodyTemplate};
use crate::sim::shared::Positions;
use crate::sim::{ClothSim, RigidSimCore, RigidSimParams};

thread_local! {
    pub(super) static COMBINED_APP_STATE: RefCell<Option<Rc<RefCell<CombinedAppState>>>> = RefCell::new(None);
}

pub(super) struct CombinedAppState {
    pub(super) ctx:   GpuContext,
    pub(super) cloth: Cloth,
    cube_cloth:   Cloth,
    pub(super) light: Lighting,
    camera:       Camera,
    pub(super) cloth_sim: ClothSim,
    rigid_sim:    RigidSimCore,
    rigid_params: RigidSimParams,
    params:       Rc<RefCell<SimParams>>,
    canvas:       HtmlCanvasElement,
    keys:         [bool; 8],
    paused:       bool,
    step_once:    bool,
}

// Hand-rolled impl so spacebar / N drive paused/step.
impl PickHost for CombinedAppState {
    fn camera(&self) -> &Camera               { &self.camera }
    fn canvas(&self) -> &HtmlCanvasElement     { &self.canvas }
    fn keys_mut(&mut self) -> &mut [bool; 8]   { &mut self.keys }
    fn positions(&self) -> &Positions          { &self.cloth_sim.q }
    fn clicked_vertex(&self) -> Option<usize>  { self.cloth_sim.clicked_vertex }
    fn set_clicked(&mut self, idx: Option<usize>) {
        self.cloth_sim.clicked_vertex = idx;
        if idx.is_some() { self.cloth_sim.dragging_vertices = None; }
    }
    fn set_mouse(&mut self, pos: [f32; 3]) { self.cloth_sim.mouse_pos = pos; }
    fn on_extra_key(&mut self, key: &str) -> bool {
        match key {
            " "       => { self.paused = !self.paused; true }
            "n" | "N" => { if self.paused { self.step_once = true; } true }
            _         => false,
        }
    }
}

#[wasm_bindgen]
pub async fn run(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let (window, canvas) = get_canvas(canvas_id)?;
    let (ctx, light, camera) = build_basics(canvas.clone()).await?;

    let cloth     = Cloth::new(&ctx, 32, &light);
    let cloth_sim = ClothSim::from_grid(32);

    // Cube: travels through origin from front-left to back-right of camera.
    let template = RigidBodyTemplate::new(&scene::CUBE_VERTS, &scene::CUBE_FACES);
    let body = RigidBodyInstance::new(
        template,
        na::Vector3::new(-1.2, 0.0, 1.2),
        na::Vector3::zeros(),
        na::Vector3::new(0.4, 0.0, -0.4),
        na::Vector3::new(1.0, 2.0, 0.5),
        1000.0,
    );
    let rigid_sim    = RigidSimCore::new(vec![body]);
    let rigid_params = RigidSimParams { gravity_enabled: false, ..Default::default() };
    let cube_cloth   = scene::cube_cloth(&ctx, &light);

    let state = Rc::new(RefCell::new(CombinedAppState {
        ctx, cloth, cube_cloth, light, camera,
        cloth_sim, rigid_sim, rigid_params,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(), keys: [false; 8],
        paused: false, step_once: false,
    }));
    COMBINED_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    install_handlers(state.clone(), &canvas, &window)?;

    let mut overlay = FpsOverlay::new(&window);
    let state_inner = state.clone();
    run_raf(&window, move || {
        let mut s = state_inner.borrow_mut();
        let CombinedAppState {
            cloth_sim, cloth, rigid_sim, rigid_params, cube_cloth,
            ctx, light, camera, params, keys, paused, step_once, ..
        } = &mut *s;

        apply_camera_keys(camera, keys, ctx);

        let should_step = !*paused || *step_once;
        *step_once = false;

        if should_step {
            rigid_sim.step(rigid_params);
            let prev = rigid_sim.bodies[0].prev_world_vertices();
            let curr = rigid_sim.bodies[0].world_vertices();

            const CLOTH_RIGID_THRESHOLD: f32 = 0.01;
            let faces: &[[u32; 3]] = &rigid_sim.bodies[0].get_template().faces;
            let bvh = crate::bvh::Bvh::build_from_verts(
                &curr, &prev, faces, CLOTH_RIGID_THRESHOLD,
            );
            if let Some(ref bvh) = bvh {
                cloth_sim.step_with_rigid(
                    &params.borrow(),
                    &HashSet::new(),
                    Some((bvh, &prev, &curr, faces, CLOTH_RIGID_THRESHOLD)),
                );
            } else {
                cloth_sim.step(&params.borrow());
            }
            cloth.sync_from_sim(&cloth_sim.q, ctx);

            for (i, pos) in cube_cloth.positions.iter_mut().enumerate() {
                *pos = [curr[i].x, curr[i].y, curr[i].z];
            }
            cube_cloth.upload(ctx);
        }

        if let Ok((frame, view)) = ctx.begin_frame() {
            light.clear_shadow(ctx);
            cloth.render_shadow(ctx, light);
            cube_cloth.render_shadow(ctx, light);
            cloth.render(ctx, &view, light, camera);
            cube_cloth.render_over(ctx, &view, light, camera);
            frame.present();
        }
        overlay.tick();
    })
}
