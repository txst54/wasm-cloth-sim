//! Particle paper sim: grid or crease-pattern-driven mesh, particle-based
//! hinge folding (vs. the mass-spring [`super::paper`] sim).

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use super::{build_basics, get_canvas, impl_pickhost, run_raf, PARAMS};
use crate::arch::wasm::cloth::{Cloth, Material};
use crate::arch::wasm::fps::FpsOverlay;
use crate::arch::wasm::gpu::GpuContext;
use crate::arch::wasm::input::{apply_camera_keys, install_handlers};
use crate::arch::wasm::light::Lighting;
use crate::arch::wasm::platform::init_platform;
use crate::arch::wasm::Camera;
use crate::params::SimParams;
use crate::sim::{CreasePattern, FoldDirection, ParticlePaperSim};

thread_local! {
    pub(super) static PARTICLE_PAPER_APP_STATE: RefCell<Option<Rc<RefCell<ParticlePaperAppState>>>> = RefCell::new(None);
}

pub(super) struct ParticlePaperAppState {
    ctx:        GpuContext,
    cloth:      Cloth,
    light:      Lighting,
    camera:     Camera,
    sim:        ParticlePaperSim,
    #[cfg(feature = "gpu")]
    gpu_sim:    crate::sim::ParticlePaperSimGpu,
    params:     Rc<RefCell<SimParams>>,
    canvas:     HtmlCanvasElement,
    keys:       [bool; 8],
    cp_data:    Option<String>,
    resolution: usize,
}

impl_pickhost!(ParticlePaperAppState, sim.core, clears_drag = false);

#[wasm_bindgen]
pub async fn run_particle_paper(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let (window, canvas) = get_canvas(canvas_id)?;
    let resolution = PARAMS.with(|p| p.borrow().resolution as usize);
    let (ctx, light, camera) = build_basics(canvas.clone()).await?;
    let mut cloth = Cloth::new(&ctx, resolution as u32, &light);
    cloth.set_material(&ctx, Material::Paper);
    let sim   = ParticlePaperSim::from_grid(resolution);

    #[cfg(feature = "gpu")]
    let gpu_sim = crate::sim::ParticlePaperSimGpu::from_cpu(
        ctx.device.clone(), ctx.queue.clone(), &sim,
    );

    let state = Rc::new(RefCell::new(ParticlePaperAppState {
        ctx, cloth, light, camera, sim,
        #[cfg(feature = "gpu")]
        gpu_sim,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(),
        keys: [false; 8],
        cp_data: None,
        resolution,
    }));
    PARTICLE_PAPER_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    install_handlers(state.clone(), &canvas, &window)?;
    spawn_particle_paper_loop(state, &window)
}

#[wasm_bindgen]
pub async fn run_particle_paper_with_cp(canvas_id: &str, cp_data: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let (window, canvas) = get_canvas(canvas_id)?;
    let resolution = PARAMS.with(|p| p.borrow().resolution as usize);
    let (ctx, light, camera) = build_basics(canvas.clone()).await?;

    let cp = CreasePattern::parse(cp_data).map_err(|e| JsValue::from_str(&e))?;
    let (sim, positions, faces, colors, edge_colors) =
        ParticlePaperSim::from_crease_pattern(&cp, resolution);
    let mut cloth = Cloth::from_mesh(&ctx, positions, faces, colors, edge_colors, &light);
    cloth.set_material(&ctx, Material::Paper);

    #[cfg(feature = "gpu")]
    let gpu_sim = crate::sim::ParticlePaperSimGpu::from_cpu(
        ctx.device.clone(), ctx.queue.clone(), &sim,
    );

    let state = Rc::new(RefCell::new(ParticlePaperAppState {
        ctx, cloth, light, camera, sim,
        #[cfg(feature = "gpu")]
        gpu_sim,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(),
        keys: [false; 8],
        cp_data: Some(cp_data.to_string()),
        resolution,
    }));
    PARTICLE_PAPER_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    install_handlers(state.clone(), &canvas, &window)?;
    spawn_particle_paper_loop(state, &window)
}

fn spawn_particle_paper_loop(
    state: Rc<RefCell<ParticlePaperAppState>>,
    window: &web_sys::Window,
) -> Result<(), JsValue> {
    let mut overlay = FpsOverlay::new(window);
    let state_inner = state.clone();
    run_raf(window, move || {
        let mut s = state_inner.borrow_mut();
        #[cfg(feature = "gpu")]
        let ParticlePaperAppState { sim, gpu_sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;
        #[cfg(not(feature = "gpu"))]
        let ParticlePaperAppState { sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        apply_camera_keys(camera, keys, ctx);

        #[cfg(feature = "gpu")]
        {
            gpu_sim.poll_readback();
            {
                use crate::sim::traits::MeshSim;
                sim.core.q.copy_from(gpu_sim.positions());
                gpu_sim.set_clicked_vertex(sim.core.clicked_vertex);
                gpu_sim.set_mouse_pos(sim.core.mouse_pos);
            }

            let mut enc = gpu_sim.cloth().device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("ppaper_step+sync+normals") },
            );
            gpu_sim.encode_step(&mut enc, &params.borrow());
            cloth.encode_position_sync(&mut enc, gpu_sim.cloth().q_buffer());
            cloth.encode_normals(&mut enc);
            let kick = gpu_sim.cloth().encode_readback_kick(&mut enc);
            gpu_sim.cloth_mut().finalize_submit(enc, kick);
        }
        #[cfg(not(feature = "gpu"))]
        {
            sim.step(&params.borrow());
            cloth.sync_from_sim(&sim.core.q, ctx);
        }

        if let Ok((frame, view)) = ctx.begin_frame() {
            light.clear_shadow(ctx);
            cloth.render_shadow(ctx, light);
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }
        overlay.tick();
    })
}

// ── Particle-paper setters ────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn set_particle_paper_resolution(v: u32) {
    PARAMS.with(|p| p.borrow_mut().resolution = v);
    PARTICLE_PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        let resolution = v as usize;
        if let Some(ref cp_data) = s.cp_data {
            if let Ok(cp) = CreasePattern::parse(cp_data) {
                let (sim, positions, faces, colors, edge_colors) =
                    ParticlePaperSim::from_crease_pattern(&cp, resolution);
                let mut new_cloth = Cloth::from_mesh(&s.ctx, positions, faces, colors, edge_colors, &s.light);
                new_cloth.set_material(&s.ctx, Material::Paper);
                s.cloth = new_cloth;
                #[cfg(feature = "gpu")]
                {
                    s.gpu_sim = crate::sim::ParticlePaperSimGpu::from_cpu(
                        s.ctx.device.clone(), s.ctx.queue.clone(), &sim,
                    );
                }
                s.sim = sim;
                s.resolution = resolution;
            }
        } else {
            let sim = ParticlePaperSim::from_grid(resolution);
            let mut new_cloth = Cloth::new(&s.ctx, resolution as u32, &s.light);
            new_cloth.set_material(&s.ctx, Material::Paper);
            s.cloth = new_cloth;
            #[cfg(feature = "gpu")]
            {
                s.gpu_sim = crate::sim::ParticlePaperSimGpu::from_cpu(
                    s.ctx.device.clone(), s.ctx.queue.clone(), &sim,
                );
            }
            s.sim = sim;
            s.resolution = resolution;
        }
    });
}

#[wasm_bindgen]
pub fn set_particle_paper_fold_angle(degrees: f64) {
    let d = (degrees as f32).to_radians();
    PARTICLE_PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        for h in s.sim.hinges.iter_mut() {
            h.target_angle = match h.direction {
                FoldDirection::Mountain => -d,
                FoldDirection::Valley   =>  d,
            };
        }
        #[cfg(feature = "gpu")]
        {
            let s_mut = &mut *s;
            for (gh, sh) in s_mut.gpu_sim.hinges.iter_mut().zip(s_mut.sim.hinges.iter()) {
                gh.target_angle = sh.target_angle;
            }
        }
    });
}

#[wasm_bindgen]
pub fn set_particle_paper_fold_amount(degrees: f64) {
    let d = degrees as f32;
    PARTICLE_PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        s.sim.set_fold_angle(d);
        #[cfg(feature = "gpu")]
        { s.gpu_sim.set_fold_angle(d); }
    });
}

#[wasm_bindgen]
pub fn set_particle_paper_fold_speed(rads_per_sec: f64) {
    let r = rads_per_sec as f32;
    PARTICLE_PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        s.sim.fold_speed = r;
        #[cfg(feature = "gpu")]
        { s.gpu_sim.set_fold_speed(r); }
    });
}

#[wasm_bindgen]
pub fn set_particle_paper_hinge_compliance(alpha: f64) {
    let a_val = alpha as f32;
    PARTICLE_PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        for h in s.sim.hinges.iter_mut() {
            h.compliance = a_val / h.rest_edge_len.max(1e-12);
        }
        #[cfg(feature = "gpu")]
        { s.gpu_sim.set_hinge_compliance(a_val); }
    });
}

#[wasm_bindgen]
pub fn set_particle_paper_hinge_damping(beta: f64) {
    let b = beta as f32;
    PARTICLE_PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        for h in s.sim.hinges.iter_mut() { h.damping = b; }
        #[cfg(feature = "gpu")]
        { s.gpu_sim.set_hinge_damping(b); }
    });
}

#[wasm_bindgen]
pub fn set_particle_paper_wireframe_enabled(enabled: bool) {
    PARTICLE_PAPER_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        state.borrow_mut().cloth.wireframe_enabled = enabled;
    });
}
