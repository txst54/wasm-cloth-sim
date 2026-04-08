mod camera;
mod gpu;
mod pipeline;
mod triangle;
mod cloth;
mod light;
mod params;
mod sim;

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent};

use camera::Camera;
use cloth::Cloth;
use gpu::GpuContext;
use light::Light;
use params::SimParams;
use sim::ClothSim;

struct AppState {
    ctx:    GpuContext,
    cloth:  Cloth,
    light:  Light,
    camera: Camera,
    sim:    ClothSim,
    params: SimParams,
    canvas: HtmlCanvasElement,
    /// [left, right, up, down] arrow key held state
    keys:   [bool; 4],
}

#[wasm_bindgen]
pub async fn run(canvas_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let canvas = window
        .document().unwrap()
        .get_element_by_id(canvas_id).unwrap()
        .dyn_into::<HtmlCanvasElement>().unwrap();

    let ctx    = GpuContext::new(canvas.clone()).await?;
    let light  = Light::new(&ctx, [2.0, 0.0, 0.5]);
    let camera = Camera::new(&ctx);
    let cloth  = Cloth::new(&ctx, 64, &light);
    let sim    = ClothSim::from_cloth(&cloth);

    let state = Rc::new(RefCell::new(AppState {
        ctx,
        cloth,
        light,
        camera,
        sim,
        params: SimParams::default(),
        canvas: canvas.clone(),
        keys: [false; 4],
    }));

    // Convert a MouseEvent's offset coordinates to NDC [-1, 1].
    fn to_ndc(event: &MouseEvent, canvas: &HtmlCanvasElement) -> (f32, f32) {
        let w = canvas.offset_width()  as f32;
        let h = canvas.offset_height() as f32;
        let nx =  (event.offset_x() as f32 / w) * 2.0 - 1.0;
        let ny = -(event.offset_y() as f32 / h) * 2.0 + 1.0; // flip Y
        (nx, ny)
    }

    // mousedown — find closest vertex and begin drag
    let state_md = state.clone();
    let mousedown = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut s = state_md.borrow_mut();
        let (nx, ny) = to_ndc(&e, &s.canvas);

        let mut best_idx  = 0usize;
        let mut best_dist = f32::MAX;
        for i in 0..s.sim.q.nrows() {
            // Project vertex to NDC for picking
            let wp = [s.sim.q[(i, 0)], s.sim.q[(i, 1)], s.sim.q[(i, 2)]];
            let (px, py) = project_to_ndc(wp, &s.camera);
            let dx = px - nx;
            let dy = py - ny;
            let d2 = dx * dx + dy * dy;
            if d2 < best_dist {
                best_dist = d2;
                best_idx  = i;
            }
        }
        let vertex_world = [
            s.sim.q[(best_idx, 0)],
            s.sim.q[(best_idx, 1)],
            s.sim.q[(best_idx, 2)],
        ];
        s.sim.clicked_vertex = Some(best_idx);
        s.sim.mouse_pos = ray_plane_intersect(nx, ny, vertex_world, &s.camera)
            .unwrap_or(vertex_world);
    }));
    canvas.add_event_listener_with_callback("mousedown", mousedown.as_ref().unchecked_ref())?;
    mousedown.forget();

    // mousemove — update drag target
    let state_mm = state.clone();
    let mousemove = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut s = state_mm.borrow_mut();
        if let Some(v) = s.sim.clicked_vertex {
            let (nx, ny) = to_ndc(&e, &s.canvas);
            let current_pos = [s.sim.q[(v, 0)], s.sim.q[(v, 1)], s.sim.q[(v, 2)]];
            if let Some(world) = ray_plane_intersect(nx, ny, current_pos, &s.camera) {
                s.sim.mouse_pos = world;
            }
        }
    }));
    canvas.add_event_listener_with_callback("mousemove", mousemove.as_ref().unchecked_ref())?;
    mousemove.forget();

    // mouseup — release drag
    let state_mu = state.clone();
    let mouseup = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |_: MouseEvent| {
        state_mu.borrow_mut().sim.clicked_vertex = None;
    }));
    canvas.add_event_listener_with_callback("mouseup", mouseup.as_ref().unchecked_ref())?;
    mouseup.forget();

    // keydown — record arrow key press and suppress page scroll
    let state_kd = state.clone();
    let keydown = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            e.prevent_default();
            state_kd.borrow_mut().keys[idx] = true;
        }
    }));
    window.add_event_listener_with_callback("keydown", keydown.as_ref().unchecked_ref())?;
    keydown.forget();

    // keyup — release arrow key
    let state_ku = state.clone();
    let keyup = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            state_ku.borrow_mut().keys[idx] = false;
        }
    }));
    window.add_event_listener_with_callback("keyup", keyup.as_ref().unchecked_ref())?;
    keyup.forget();

    // RAF loop
    let loop_fn: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let loop_fn_inner = loop_fn.clone();
    let state_inner   = state.clone();

    *loop_fn.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mut s = state_inner.borrow_mut();
        let AppState { sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        // Rotate camera from held arrow keys (~60 fps → 0.02 rad/frame ≈ 1.2 rad/s)
        const SPEED: f32 = 0.02;
        if keys[0] { camera.yaw   -= SPEED; }
        if keys[1] { camera.yaw   += SPEED; }
        if keys[2] { camera.pitch += SPEED; }
        if keys[3] { camera.pitch -= SPEED; }
        camera.pitch = camera.pitch.clamp(-1.5, 1.5);

        if keys.iter().any(|&k| k) {
            camera.update(&ctx.queue);
        }

        sim.step(params);
        sim.write_to_cloth(cloth, ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }

        web_sys::window()
            .unwrap()
            .request_animation_frame(
                loop_fn_inner.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
            )
            .unwrap();
    }) as Box<dyn FnMut()>));

    window.request_animation_frame(
        loop_fn.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
    )?;

    Ok(())
}

/// Maps "ArrowLeft/Right/Up/Down" to indices 0/1/2/3.
fn arrow_key_index(key: &str) -> Option<usize> {
    match key {
        "ArrowLeft"  => Some(0),
        "ArrowRight" => Some(1),
        "ArrowUp"    => Some(2),
        "ArrowDown"  => Some(3),
        _            => None,
    }
}

/// Project a world-space point to NDC using the camera's view-projection.
/// Returns (ndc_x, ndc_y).
fn project_to_ndc(world: [f32; 3], camera: &Camera) -> (f32, f32) {
    // Reconstruct column-major view_proj from inv_view_proj is messy;
    // instead re-derive from the uniform buffer is not accessible here.
    // Simple approach: use the fact that the camera always looks at the origin.
    // We recompute the forward projection from eye + stored inv_vp (inverted).
    // Since we store inv_vp row-major, invert it back via a 4-vec multiply:
    // clip = vp * world_h  →  use the column-major vp stored in the uniform.
    // For picking we just need approximate 2-D distance in NDC, so we re-derive
    // from inv_vp: world = inv_vp * clip  ⟹  clip = inv_vp⁻¹ * world.
    // Simpler: manually compute clip via view then proj.
    let e = camera.eye;
    // View: translate to camera-relative, then rotate.
    // We don't have the basis vectors cached; instead directly use the geometry.
    // (This mirrors build() in camera.rs without extra storage.)
    let forward = normalize3(neg3(e)); // looking at origin
    let right   = normalize3(cross3(forward, [0.0, 1.0, 0.0]));
    let up      = cross3(right, forward);

    let d = sub3(world, e);
    let vx = dot3(d, right);
    let vy = dot3(d, up);
    let vz = dot3(d, neg3(forward)); // camera looks down -Z (right-handed)

    // Perspective divide (fov_y = PI/4, so f = cot(PI/8))
    let f = 1.0 / (std::f32::consts::FRAC_PI_4 * 0.5).tan();
    let nx = (f / camera.aspect) * vx / (-vz);
    let ny = f * vy / (-vz);
    (nx, ny)
}

/// Cast a ray from the camera through NDC (nx, ny), intersect with the plane
/// that is parallel to the camera's image plane and passes through `plane_point`.
/// Returns the world-space intersection, or None if the ray is parallel to the plane.
fn ray_plane_intersect(
    nx: f32, ny: f32,
    plane_point: [f32; 3],
    camera: &Camera,
) -> Option<[f32; 3]> {
    // Unproject two NDC points to get a world-space ray direction.
    let near = unproject(nx, ny, 0.0, &camera.inv_view_proj);
    let far  = unproject(nx, ny, 1.0, &camera.inv_view_proj);
    let dir  = normalize3(sub3(far, near));

    // Plane normal = camera forward = normalize(-eye)  (looking at origin)
    let plane_normal = normalize3(neg3(camera.eye));

    let denom = dot3(plane_normal, dir);
    if denom.abs() < 1e-6 { return None; }

    let t = dot3(plane_normal, sub3(plane_point, camera.eye)) / denom;
    Some([
        camera.eye[0] + t * dir[0],
        camera.eye[1] + t * dir[1],
        camera.eye[2] + t * dir[2],
    ])
}

/// Unproject an NDC point (nx, ny, nz) to world space using the row-major inverse VP.
fn unproject(nx: f32, ny: f32, nz: f32, inv_vp: &[[f32; 4]; 4]) -> [f32; 3] {
    let clip = [nx, ny, nz, 1.0f32];
    let mut w = [0.0f32; 4];
    for row in 0..4 {
        for col in 0..4 {
            w[row] += inv_vp[row][col] * clip[col];
        }
    }
    let inv_w = 1.0 / w[3];
    [w[0] * inv_w, w[1] * inv_w, w[2] * inv_w]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
fn neg3(a: [f32; 3]) -> [f32; 3] { [-a[0], -a[1], -a[2]] }
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
    if len > 1e-8 { [v[0]/len, v[1]/len, v[2]/len] } else { v }
}
