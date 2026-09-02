//! Head: particle cloth falling onto an OBJ mesh (baked to an SDF obstacle).
//!
//! Reuses [`super::particle_cloth::ParticleAppState`] verbatim — this is the
//! same sim as the plain particle-cloth demo, just bootstrapped with a
//! mesh-SDF obstacle instead of a sphere.

use std::cell::RefCell;
use std::rc::Rc;

use nalgebra as na;
use wasm_bindgen::prelude::*;

use super::particle_cloth::{ParticleAppState, PARTICLE_APP_STATE};
use super::{build_basics, get_canvas, run_raf, PARAMS};
use crate::arch::wasm::cloth::Cloth;
use crate::arch::wasm::fps::FpsOverlay;
use crate::arch::wasm::input::{apply_camera_keys, install_handlers};
use crate::arch::wasm::platform::init_platform;
use crate::arch::wasm::scene;
use crate::sim::mesh_sdf::bake as bake_mesh_sdf;
use crate::sim::obj_loader::ObjMesh;
use crate::sim::{ParticleClothSim, SdfObstacle};

const HEAD_PARTICLE_RESOLUTION: usize = 120;
const HEAD_SDF_RESOLUTION: u32        = 64;
const HEAD_SDF_PADDING: f32           = 0.10;
/// How far above the head's top bound the cloth spawns.
const HEAD_CLOTH_DROP: f32            = 0.5;

#[wasm_bindgen]
pub async fn run_head_cloth(canvas_id: &str, obj_text: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let (window, canvas) = get_canvas(canvas_id)?;
    let (ctx, light, mut camera) = build_basics(canvas.clone()).await?;

    let mut mesh = ObjMesh::parse(obj_text).map_err(|e| JsValue::from_str(&e))?;
    mesh.recenter_unit();
    let (lo, hi) = mesh.bounds();
    web_sys::console::log_1(&format!(
        "[psim] head OBJ: {} verts / {} faces, bbox [{:.2},{:.2},{:.2}]–[{:.2},{:.2},{:.2}]",
        mesh.positions.len(), mesh.faces.len(),
        lo[0], lo[1], lo[2], hi[0], hi[1], hi[2],
    ).into());

    web_sys::console::log_1(&"[psim] baking mesh SDF…".into());
    let vol = bake_mesh_sdf(&mesh, HEAD_SDF_RESOLUTION, HEAD_SDF_PADDING);
    web_sys::console::log_1(&format!(
        "[psim] mesh SDF baked: {}^3 cells, bbox [{:.3},{:.3},{:.3}]–[{:.3},{:.3},{:.3}]",
        HEAD_SDF_RESOLUTION,
        vol.bounds_min[0], vol.bounds_min[1], vol.bounds_min[2],
        vol.bounds_max[0], vol.bounds_max[1], vol.bounds_max[2],
    ).into());
    {
        // Sanity probes: corners (should be > 0, far from mesh), bbox
        // center (deep inside body → should be < 0), top of head (just
        // above mesh, ~ padding distance).
        let n = HEAD_SDF_RESOLUTION as usize;
        let at = |ix: usize, iy: usize, iz: usize| vol.data[iz * n * n + iy * n + ix];
        let mid = n / 2;
        let last = n - 1;
        web_sys::console::log_1(&format!(
            "[psim] SDF probes: bmin={:.4} bmax={:.4} center={:.4} \
             top_center={:.4} bot_center={:.4} side_mid={:.4}",
            at(0, 0, 0), at(last, last, last),
            at(mid, mid, mid),
            at(mid, last, mid), at(mid, 0, mid),
            at(last, mid, mid),
        ).into());
    }

    // Cloth + obstacles.
    let mut sim = ParticleClothSim::from_grid(HEAD_PARTICLE_RESOLUTION, &[]);
    // Lift cloth so it falls cleanly onto the head's top.
    let cloth_y = hi[1] + HEAD_CLOTH_DROP;
    let dy = cloth_y - 1.5; // from_grid spawns at y = 1.5
    if dy != 0.0 {
        for i in 0..sim.q.nrows() {
            sim.q[(i, 1)]      += dy;
            sim.q_prev[(i, 1)] += dy;
            sim.q_pred[(i, 1)] += dy;
            sim.q_rest[(i, 1)] += dy;
        }
    }

    sim.add_obstacle(SdfObstacle::mesh(
        na::Vector3::zeros(),
        na::Vector3::new(vol.bounds_min[0], vol.bounds_min[1], vol.bounds_min[2]),
        na::Vector3::new(vol.bounds_max[0], vol.bounds_max[1], vol.bounds_max[2]),
    ));
    let ground_y = lo[1] - 0.05;
    sim.add_obstacle(SdfObstacle::plane(na::Vector3::new(0.0, 1.0, 0.0), ground_y));

    let cloth = Cloth::new(&ctx, HEAD_PARTICLE_RESOLUTION as u32, &light);

    #[cfg(feature = "gpu")]
    let gpu_sim = {
        let mut g = crate::sim::ParticleClothSimGpu::from_cpu(
            ctx.device.clone(), ctx.queue.clone(), &sim,
        );
        g.set_mesh_sdf(&vol);
        g.set_obstacles(&sim.obstacles);
        web_sys::console::log_1(&"[psim] head GPU compute path enabled".into());
        g
    };

    let head_render = Some(scene::mesh_cloth(
        &ctx, &light, mesh.positions.clone(), mesh.faces.clone(), [0.85, 0.72, 0.62],
    ));
    let half_extent = (hi[0] - lo[0]).abs().max((hi[2] - lo[2]).abs()) * 4.0 + 5.0;
    let ground_render = Some(scene::ground_cloth(&ctx, &light, ground_y, half_extent));

    // Frame the head: distance ~ 3 × max half-extent.
    let max_half = ((hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2])) * 0.5;
    camera.dist   = max_half * 3.5;
    camera.target = [0.0, 0.5 * (lo[1] + hi[1]), 0.0];
    camera.update(&ctx.queue);

    let state = Rc::new(RefCell::new(ParticleAppState {
        ctx, cloth, light, camera, sim,
        #[cfg(feature = "gpu")]
        gpu_sim,
        sphere_cloth: head_render,
        ground_cloth: ground_render,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(),
        keys: [false; 8],
        resolution: HEAD_PARTICLE_RESOLUTION,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 0.0,
        mesh_sdf_vol:  Some(vol),
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
            gpu_sim.poll_readback();
            {
                use crate::sim::traits::MeshSim;
                sim.q.copy_from(gpu_sim.positions());
                gpu_sim.set_clicked_vertex(sim.clicked_vertex);
                gpu_sim.set_mouse_pos(sim.mouse_pos);
            }
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("head_cloth_step+sync+normals"),
            });
            gpu_sim.encode_step(&mut enc, &params.borrow());
            cloth.encode_position_sync(&mut enc, gpu_sim.q_buffer());
            cloth.encode_normals(&mut enc);
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
