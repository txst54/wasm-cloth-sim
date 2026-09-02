//! Paper sim (mass-spring hinges): grid with a central fold, or a
//! crease-pattern-driven mesh.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use super::{build_basics, get_canvas, impl_pickhost, run_raf, PARAMS};
use crate::arch::wasm::cloth::{Cloth, Material};
use crate::arch::wasm::gpu::GpuContext;
use crate::arch::wasm::input::{apply_camera_keys, install_handlers};
use crate::arch::wasm::light::Lighting;
use crate::arch::wasm::platform::init_platform;
use crate::arch::wasm::Camera;
use crate::params::SimParams;
use crate::platform_context::PlatformContext;
use crate::sim::{CreasePattern, FoldDirection, FoldSpec, PaperSim};

thread_local! {
    pub(super) static PAPER_APP_STATE: RefCell<Option<Rc<RefCell<PaperAppState>>>> = RefCell::new(None);
}

pub(super) struct PaperAppState {
    ctx:        GpuContext,
    cloth:      Cloth,
    light:      Lighting,
    camera:     Camera,
    sim:        PaperSim,
    params:     Rc<RefCell<SimParams>>,
    canvas:     HtmlCanvasElement,
    keys:       [bool; 8],
    cp_data:    Option<String>,
    resolution: usize,
}

impl_pickhost!(PaperAppState, sim, clears_drag = true);

fn central_vertical_fold(n: usize) -> HashMap<(u32, u32), FoldSpec> {
    let mut map = HashMap::new();
    let col = n / 2;
    for row in 0..(n - 1) {
        let a = (row * n + col) as u32;
        let b = ((row + 1) * n + col) as u32;
        let (lo, hi) = (a.min(b), a.max(b));
        map.insert((lo, hi), FoldSpec {
            target_angle: std::f32::consts::PI,
            compliance:   1e-4,
            direction:    FoldDirection::Mountain,
            damping:      0.5,
        });
    }
    map
}

#[wasm_bindgen]
pub async fn run_paper(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let (window, canvas) = get_canvas(canvas_id)?;
    let resolution = PARAMS.with(|p| p.borrow().resolution as usize);
    let (ctx, light, camera) = build_basics(canvas.clone()).await?;
    let mut cloth = Cloth::new(&ctx, resolution as u32, &light);
    cloth.set_material(&ctx, Material::Paper);
    let mut sim = PaperSim::from_grid(resolution);
    sim.set_fold_map(central_vertical_fold(resolution));

    let state = Rc::new(RefCell::new(PaperAppState {
        ctx, cloth, light, camera, sim,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(),
        keys: [false; 8],
        cp_data: None,
        resolution,
    }));
    PAPER_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));
    install_handlers(state.clone(), &canvas, &window)?;
    spawn_paper_loop(state, &window)
}

#[wasm_bindgen]
pub async fn run_paper_with_cp(canvas_id: &str, cp_data: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let (window, canvas) = get_canvas(canvas_id)?;
    let resolution = PARAMS.with(|p| p.borrow().resolution as usize);
    let (ctx, light, camera) = build_basics(canvas.clone()).await?;

    let cp = CreasePattern::parse(cp_data).map_err(|e| JsValue::from_str(&e))?;
    let (sim, positions, faces, colors, edge_colors) =
        PaperSim::from_crease_pattern(&cp, resolution);
    let mut cloth = Cloth::from_mesh(&ctx, positions, faces, colors, edge_colors, &light);
    cloth.set_material(&ctx, Material::Paper);

    let state = Rc::new(RefCell::new(PaperAppState {
        ctx, cloth, light, camera, sim,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(),
        keys: [false; 8],
        cp_data: Some(cp_data.to_string()),
        resolution,
    }));
    PAPER_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));
    install_handlers(state.clone(), &canvas, &window)?;
    spawn_paper_loop(state, &window)
}

fn spawn_paper_loop(
    state: Rc<RefCell<PaperAppState>>,
    window: &web_sys::Window,
) -> Result<(), JsValue> {
    let mut frame_idx: usize = 0;
    let state_inner = state.clone();
    run_raf(window, move || {
        let mut s = state_inner.borrow_mut();
        let PaperAppState { sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        apply_camera_keys(camera, keys, ctx);
        PlatformContext::set_step(frame_idx);
        frame_idx += 1;
        sim.step(&params.borrow());
        cloth.sync_from_sim(&sim.q, ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            light.clear_shadow(ctx);
            cloth.render_shadow(ctx, light);
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }
    })
}

// ── Paper-sim setters ─────────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn set_paper_hinge_compliance(alpha: f64) {
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        for h in &mut state.borrow_mut().sim.hinges {
            h.compliance = (alpha as f32) / h.rest_edge_len.max(1e-12);
        }
    });
}

#[wasm_bindgen]
pub fn set_paper_hinge_damping(beta: f64) {
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        for h in &mut state.borrow_mut().sim.hinges { h.damping = beta as f32; }
    });
}

#[wasm_bindgen]
pub fn set_paper_fold_speed(rads_per_sec: f64) {
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        state.borrow_mut().sim.fold_speed = rads_per_sec as f32;
    });
}

#[wasm_bindgen]
pub fn set_paper_fold_angle(degrees: f64) {
    let d = (degrees as f32).to_radians();
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        for h in &mut state.borrow_mut().sim.hinges {
            h.target_angle = match h.direction {
                FoldDirection::Mountain => -d,
                FoldDirection::Valley   =>  d,
            };
        }
    });
}

#[wasm_bindgen]
pub fn set_paper_fold_amount(degrees: f64) {
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        state.borrow_mut().sim.set_fold_angle(degrees as f32);
    });
}

#[wasm_bindgen]
pub fn set_wireframe_enabled(enabled: bool) {
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        state.borrow_mut().cloth.wireframe_enabled = enabled;
    });
}

#[wasm_bindgen]
pub fn set_paper_resolution(v: u32) {
    PARAMS.with(|p| p.borrow_mut().resolution = v);
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        let resolution = v as usize;

        if let Some(ref cp_data) = s.cp_data {
            if let Ok(cp) = CreasePattern::parse(cp_data) {
                let (sim, positions, faces, colors, edge_colors) =
                    PaperSim::from_crease_pattern(&cp, resolution);
                let mut new_cloth = Cloth::from_mesh(&s.ctx, positions, faces, colors, edge_colors, &s.light);
                new_cloth.set_material(&s.ctx, Material::Paper);
                s.cloth = new_cloth;
                s.sim = sim;
                s.resolution = resolution;
            }
        } else {
            let mut sim = PaperSim::from_grid(resolution);
            sim.set_fold_map(central_vertical_fold(resolution));
            let mut new_cloth = Cloth::new(&s.ctx, resolution as u32, &s.light);
            new_cloth.set_material(&s.ctx, Material::Paper);
            s.cloth = new_cloth;
            s.sim = sim;
            s.resolution = resolution;
        }
    });
}
