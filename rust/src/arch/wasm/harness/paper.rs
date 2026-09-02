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
    /// Auto-orbit rate in rad/s. 0 = off. When non-zero, the render loop
    /// recentres the camera on the model and spins it about the vertical axis.
    spin_speed: f32,
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
        spin_speed: 0.0,
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
        spin_speed: 0.0,
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
    let mut last_ms = PlatformContext::now_ms();
    let state_inner = state.clone();
    run_raf(window, move || {
        let mut s = state_inner.borrow_mut();
        let PaperAppState { sim, cloth, ctx, light, camera, params, keys, spin_speed, .. } = &mut *s;

        let now_ms = PlatformContext::now_ms();
        let dt = (((now_ms - last_ms) / 1000.0) as f32).clamp(0.0, 0.1);
        last_ms = now_ms;

        apply_camera_keys(camera, keys, ctx);

        // Auto-orbit: recentre on the model's centroid and advance yaw so the
        // paper appears to spin about a vertical axis at screen centre.
        if *spin_speed != 0.0 {
            let q = &sim.q;
            let n = q.nrows().max(1) as f32;
            let (mut cx, mut cy, mut cz) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..q.nrows() {
                cx += q[(i, 0)];
                cy += q[(i, 1)];
                cz += q[(i, 2)];
            }
            camera.target = [cx / n, cy / n, cz / n];
            camera.yaw += *spin_speed * dt;
            camera.update(&ctx.queue);
        }

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
pub fn set_surface_enabled(enabled: bool) {
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        state.borrow_mut().cloth.surface_enabled = enabled;
    });
}

// Wireframe edge colors. `kind`: 0 = regular, 1 = mountain crease, 2 = valley
// crease. Components are RGB in 0..1.
fn set_wireframe_color(kind: u32, r: f64, g: f64, b: f64) {
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        let PaperAppState { cloth, ctx, .. } = &mut *s;
        cloth.set_wireframe_color(ctx, kind, [r as f32, g as f32, b as f32]);
    });
}

#[wasm_bindgen]
pub fn set_wireframe_regular_color(r: f64, g: f64, b: f64) {
    set_wireframe_color(0, r, g, b);
}

#[wasm_bindgen]
pub fn set_wireframe_mountain_color(r: f64, g: f64, b: f64) {
    set_wireframe_color(1, r, g, b);
}

#[wasm_bindgen]
pub fn set_wireframe_valley_color(r: f64, g: f64, b: f64) {
    set_wireframe_color(2, r, g, b);
}

/// Drain the verbose paper-sim trace buffer as a newline-joined string.
/// The on-page sim console calls this once per animation frame. Separate from
/// the throttled browser console; returns "" when nothing is queued.
#[wasm_bindgen]
pub fn drain_paper_trace() -> String {
    crate::sim::trace::trace_drain()
}

/// Continuously orbit the camera about a vertical axis through the model's
/// centroid. `degrees_per_sec` = 0 stops the spin (default).
#[wasm_bindgen]
pub fn set_paper_spin_speed(degrees_per_sec: f64) {
    let rads = (degrees_per_sec as f32).to_radians();
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        state.borrow_mut().spin_speed = rads;
    });
}

#[wasm_bindgen]
pub fn set_paper_resolution(v: u32) {
    PARAMS.with(|p| p.borrow_mut().resolution = v);
    PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        let resolution = v as usize;
        // Rebuilding the cloth resets the wireframe palette to its default;
        // carry the current colors across so a resolution change doesn't
        // clobber a custom palette.
        let wf_colors = s.cloth.wireframe_colors;

        if let Some(ref cp_data) = s.cp_data {
            if let Ok(cp) = CreasePattern::parse(cp_data) {
                let (sim, positions, faces, colors, edge_colors) =
                    PaperSim::from_crease_pattern(&cp, resolution);
                let mut new_cloth = Cloth::from_mesh(&s.ctx, positions, faces, colors, edge_colors, &s.light);
                new_cloth.set_material(&s.ctx, Material::Paper);
                new_cloth.set_wireframe_colors(&s.ctx, wf_colors);
                s.cloth = new_cloth;
                s.sim = sim;
                s.resolution = resolution;
            }
        } else {
            let mut sim = PaperSim::from_grid(resolution);
            sim.set_fold_map(central_vertical_fold(resolution));
            let mut new_cloth = Cloth::new(&s.ctx, resolution as u32, &s.light);
            new_cloth.set_material(&s.ctx, Material::Paper);
            new_cloth.set_wireframe_colors(&s.ctx, wf_colors);
            s.cloth = new_cloth;
            s.sim = sim;
            s.resolution = resolution;
        }
    });
}
