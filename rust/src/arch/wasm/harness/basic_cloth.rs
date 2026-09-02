//! Cloth-only sim: a plain grid of cloth falling under gravity.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use super::combined::COMBINED_APP_STATE;
use super::{build_basics, get_canvas, impl_pickhost, run_raf, PARAMS};
use crate::arch::wasm::cloth::Cloth;
use crate::arch::wasm::fps::FpsOverlay;
use crate::arch::wasm::gpu::GpuContext;
use crate::arch::wasm::input::{apply_camera_keys, install_handlers};
use crate::arch::wasm::light::Lighting;
use crate::arch::wasm::platform::init_platform;
use crate::arch::wasm::Camera;
use crate::params::SimParams;
use crate::sim::ClothSim;

thread_local! {
    pub(super) static APP_STATE: RefCell<Option<Rc<RefCell<AppState>>>> = RefCell::new(None);
}

pub(super) struct AppState {
    pub(super) ctx:    GpuContext,
    pub(super) cloth:  Cloth,
    pub(super) light:  Lighting,
    pub(super) camera: Camera,
    pub(super) sim:    ClothSim,
    pub(super) params: Rc<RefCell<SimParams>>,
    pub(super) canvas: HtmlCanvasElement,
    /// Held key state: `[←, →, ↑, ↓, A, D, W, S]`.
    pub(super) keys:   [bool; 8],
}

impl_pickhost!(AppState, sim, clears_drag = true);

#[wasm_bindgen]
pub async fn run_cloth(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let (window, canvas) = get_canvas(canvas_id)?;
    let (ctx, light, camera) = build_basics(canvas.clone()).await?;

    let cloth = Cloth::new(&ctx, 32, &light);
    let sim   = ClothSim::from_grid(32);

    let state = Rc::new(RefCell::new(AppState {
        ctx, cloth, light, camera, sim,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(),
        keys: [false; 8],
    }));
    APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    install_handlers(state.clone(), &canvas, &window)?;

    let mut overlay = FpsOverlay::new(&window);
    let state_inner = state.clone();
    run_raf(&window, move || {
        let mut s = state_inner.borrow_mut();
        let AppState { sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        apply_camera_keys(camera, keys, ctx);
        sim.step(&params.borrow());
        cloth.sync_from_sim(&sim.q, ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            light.clear_shadow(ctx);
            cloth.render_shadow(ctx, light);
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }
        overlay.tick();
    })
}

#[wasm_bindgen]
pub fn set_resolution(v: u32) {
    APP_STATE.with(|a| {
        if let Some(state) = a.borrow().as_ref() {
            let mut s = state.borrow_mut();
            s.cloth = Cloth::new(&s.ctx, v, &s.light);
            s.sim   = ClothSim::from_grid(v as usize);
        }
    });
    COMBINED_APP_STATE.with(|a| {
        if let Some(state) = a.borrow().as_ref() {
            let mut s = state.borrow_mut();
            s.cloth     = Cloth::new(&s.ctx, v, &s.light);
            s.cloth_sim = ClothSim::from_grid(v as usize);
        }
    });
}
