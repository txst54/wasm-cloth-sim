//! Particle cloth sim with SDF obstacles (sphere + ground plane).
//!
//! [`ParticleAppState`] and [`PARTICLE_APP_STATE`] are also reused by
//! [`super::head_cloth`], which bootstraps the same sim with an OBJ-mesh
//! obstacle instead of a sphere.

use std::cell::RefCell;
use std::rc::Rc;

use nalgebra as na;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use super::{build_basics, get_canvas, impl_pickhost, run_raf, PARAMS};
use crate::arch::wasm::cloth::Cloth;
use crate::arch::wasm::fps::FpsOverlay;
use crate::arch::wasm::gpu::GpuContext;
use crate::arch::wasm::input::{apply_camera_keys, install_handlers};
use crate::arch::wasm::light::Lighting;
use crate::arch::wasm::platform::init_platform;
use crate::arch::wasm::scene;
use crate::arch::wasm::Camera;
use crate::params::SimParams;
use crate::sim::{ParticleClothSim, SdfObstacle};

const PARTICLE_RESOLUTION: usize = 120;
const SPHERE_CENTER: [f32; 3] = [0.0, 0.0, 0.0];
const SPHERE_RADIUS: f32      = 0.4;

thread_local! {
    pub(super) static PARTICLE_APP_STATE: RefCell<Option<Rc<RefCell<ParticleAppState>>>> = RefCell::new(None);
}

pub(super) struct ParticleAppState {
    pub(super) ctx:    GpuContext,
    pub(super) cloth:  Cloth,
    pub(super) light:  Lighting,
    pub(super) camera: Camera,
    pub(super) sim:    ParticleClothSim,
    #[cfg(feature = "gpu")]
    pub(super) gpu_sim: crate::sim::ParticleClothSimGpu,
    pub(super) sphere_cloth:  Option<Cloth>,
    pub(super) ground_cloth:  Option<Cloth>,
    pub(super) params:        Rc<RefCell<SimParams>>,
    pub(super) canvas:        HtmlCanvasElement,
    pub(super) keys:          [bool; 8],
    pub(super) resolution:    usize,
    pub(super) sphere_center: [f32; 3],
    pub(super) sphere_radius: f32,
    /// Optional baked mesh SDF volume — preserved across resolution
    /// rebuilds so `set_particle_resolution` can re-bind it on the new
    /// GPU sim. `None` for scenes without a mesh obstacle.
    pub(super) mesh_sdf_vol:  Option<crate::sim::MeshSdfVolume>,
}

impl_pickhost!(ParticleAppState, sim, clears_drag = false);

pub(super) fn particle_obstacles(sim: &mut ParticleClothSim, center: [f32; 3], radius: f32) {
    sim.obstacles.clear();
    sim.add_obstacle(SdfObstacle::sphere(
        na::Vector3::new(center[0], center[1], center[2]),
        radius,
    ));
    let ground_y = center[1] - radius;
    sim.add_obstacle(SdfObstacle::plane(na::Vector3::new(0.0, 1.0, 0.0), ground_y));
}

#[wasm_bindgen]
pub async fn run_particle_cloth(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let (window, canvas) = get_canvas(canvas_id)?;
    let (ctx, light, camera) = build_basics(canvas.clone()).await?;

    let cloth   = Cloth::new(&ctx, PARTICLE_RESOLUTION as u32, &light);
    let mut sim = ParticleClothSim::from_grid(PARTICLE_RESOLUTION, &[]);
    particle_obstacles(&mut sim, SPHERE_CENTER, SPHERE_RADIUS);

    #[cfg(feature = "gpu")]
    let gpu_sim = {
        let mut g = crate::sim::ParticleClothSimGpu::from_cpu(
            ctx.device.clone(), ctx.queue.clone(), &sim,
        );
        g.set_obstacles(&sim.obstacles);
        web_sys::console::log_1(&"[psim] GPU compute path enabled".into());
        g
    };

    let sphere_cloth = Some(scene::sphere_cloth(&ctx, &light, SPHERE_CENTER, SPHERE_RADIUS));
    let ground_cloth = Some(scene::ground_cloth(&ctx, &light, SPHERE_CENTER[1] - SPHERE_RADIUS, 10.0));

    let state = Rc::new(RefCell::new(ParticleAppState {
        ctx, cloth, light, camera, sim,
        #[cfg(feature = "gpu")]
        gpu_sim,
        sphere_cloth, ground_cloth,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(), keys: [false; 8],
        resolution: PARTICLE_RESOLUTION,
        sphere_center: SPHERE_CENTER,
        sphere_radius: SPHERE_RADIUS,
        mesh_sdf_vol:  None,
    }));
    PARTICLE_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    install_handlers(state.clone(), &canvas, &window)?;

    let mut overlay = FpsOverlay::new(&window);
    let state_inner = state.clone();
    run_raf(&window, move || {
        let mut s = state_inner.borrow_mut();
        #[cfg(feature = "gpu")]
        let ParticleAppState { sim, gpu_sim, cloth, sphere_cloth, ground_cloth, ctx, light, camera, params, keys, .. } = &mut *s;
        #[cfg(not(feature = "gpu"))]
        let ParticleAppState { sim, cloth, sphere_cloth, ground_cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        apply_camera_keys(camera, keys, ctx);

        #[cfg(feature = "gpu")]
        {
            // Drain readback into sim.q for picking. Picking can tolerate
            // 1–3 frame stale positions; rendering cannot, which is why
            // positions go directly into the vertex buffer below.
            gpu_sim.poll_readback();
            {
                use crate::sim::traits::MeshSim;
                sim.q.copy_from(gpu_sim.positions());
                gpu_sim.set_clicked_vertex(sim.clicked_vertex);
                gpu_sim.set_mouse_pos(sim.mouse_pos);
            }

            // One submission: sim step + GPU→VB position copy + GPU normals.
            // Positions and normals land in the vertex buffer in the same
            // submission as the sim step that produced them — no async
            // readback latency, no shading lag.
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("particle_cloth_step+sync+normals"),
            });
            gpu_sim.encode_step(&mut enc, &params.borrow());
            cloth.encode_position_sync(&mut enc, gpu_sim.q_buffer());
            cloth.encode_normals(&mut enc);
            // Kick staging readback for next frame's picking.
            let kick = gpu_sim.encode_readback_kick(&mut enc);
            gpu_sim.finalize_submit(enc, kick);
        }
        #[cfg(not(feature = "gpu"))]
        {
            sim.step(&params.borrow());
            cloth.sync_from_sim(&sim.q, ctx);
        }

        if let Ok((frame, view)) = ctx.begin_frame() {
            light.clear_shadow(ctx);
            cloth.render_shadow(ctx, light);
            if let Some(sc) = sphere_cloth.as_ref() { sc.render_shadow(ctx, light); }
            if let Some(gc) = ground_cloth.as_ref() { gc.render_shadow(ctx, light); }
            cloth.render(ctx, &view, light, camera);
            if let Some(gc) = ground_cloth { gc.render_over(ctx, &view, light, camera); }
            if let Some(sc) = sphere_cloth { sc.render_over(ctx, &view, light, camera); }
            frame.present();
        }
        overlay.tick();
    })
}

// ── Particle-cloth setters ────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn set_particle_resolution(v: u32) {
    PARTICLE_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        let n = v.max(2) as usize;
        let new_cloth = Cloth::new(&s.ctx, n as u32, &s.light);
        let mut new_sim = ParticleClothSim::from_grid(n, &[]);
        // Carry over whatever obstacle setup the current scene has — sphere
        // + plane for the cloth canvas, mesh + plane for the head canvas,
        // etc. The CPU sim owns the source of truth.
        new_sim.obstacles = s.sim.obstacles.clone();
        #[cfg(feature = "gpu")]
        {
            let mut g = crate::sim::ParticleClothSimGpu::from_cpu(
                s.ctx.device.clone(), s.ctx.queue.clone(), &new_sim,
            );
            if let Some(vol) = s.mesh_sdf_vol.as_ref() {
                g.set_mesh_sdf(vol);
            }
            g.set_obstacles(&new_sim.obstacles);
            s.gpu_sim = g;
        }
        s.cloth = new_cloth;
        s.sim   = new_sim;
        s.resolution = n;
    });
}

#[wasm_bindgen]
pub fn set_particle_sphere(cx: f32, cy: f32, cz: f32, radius: f32) {
    PARTICLE_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        s.sphere_center = [cx, cy, cz];
        s.sphere_radius = radius;
        particle_obstacles(&mut s.sim, [cx, cy, cz], radius);
        s.sphere_cloth = Some(scene::sphere_cloth(&s.ctx, &s.light, [cx, cy, cz], radius));
        s.ground_cloth = Some(scene::ground_cloth(&s.ctx, &s.light, cy - radius, 10.0));
    });
}

#[wasm_bindgen]
pub fn set_particle_radius_scale(scale: f32) {
    PARTICLE_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        let avg: f32 = if s.sim.edge_rest.is_empty() { 0.05 } else {
            s.sim.edge_rest.iter().sum::<f32>() / s.sim.edge_rest.len() as f32
        };
        let r_val = scale.max(0.0) * avg;
        for r in s.sim.r.iter_mut() { *r = r_val; }
        s.sim.r_max = r_val;
    });
}
