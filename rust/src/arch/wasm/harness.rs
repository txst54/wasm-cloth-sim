//! WASM entry points and render loop.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent, TouchEvent};

use nalgebra as na;

use super::platform::init_platform;
use super::camera::Camera;
use super::cloth::Cloth;
use super::gpu::GpuContext;
use super::light::Light;
use crate::params::SimParams;
use crate::rigid_body::{RigidBodyInstance, RigidBodyTemplate};
use crate::sim::{ClothSim, CreasePattern, CreaseType, FoldDirection, FoldSpec, PaperSim, RigidSimCore, RigidSimParams};

thread_local! {
    static PARAMS: Rc<RefCell<SimParams>> = Rc::new(RefCell::new(SimParams::default()));
    static APP_STATE: RefCell<Option<Rc<RefCell<AppState>>>> = RefCell::new(None);
    static PAPER_APP_STATE: RefCell<Option<Rc<RefCell<PaperAppState>>>> = RefCell::new(None);
    static RIGID_APP_STATE: RefCell<Option<Rc<RefCell<RigidAppState>>>> = RefCell::new(None);
    static COMBINED_APP_STATE: RefCell<Option<Rc<RefCell<CombinedAppState>>>> = RefCell::new(None);
}

struct AppState {
    ctx:    GpuContext,
    cloth:  Cloth,
    light:  Light,
    camera: Camera,
    sim:    ClothSim,
    params: Rc<RefCell<SimParams>>,
    canvas: HtmlCanvasElement,
    /// Held key state: [←, →, ↑, ↓, A, D, W, S].
    /// Arrows orbit the camera; WASD translates the look-at target.
    keys:   [bool; 8],
}

struct PaperAppState {
    ctx:    GpuContext,
    cloth:  Cloth,
    light:  Light,
    camera: Camera,
    sim:    PaperSim,
    params: Rc<RefCell<SimParams>>,
    canvas: HtmlCanvasElement,
    keys:   [bool; 8],
    cp_data: Option<String>,
    resolution: usize,
}

struct RigidAppState {
    ctx:    GpuContext,
    cloth:  Cloth,
    light:  Light,
    camera: Camera,
    sim:    RigidSimCore,
    params: RigidSimParams,
    canvas: HtmlCanvasElement,
    keys:   [bool; 8],
}

struct CombinedAppState {
    ctx:          GpuContext,
    cloth:        Cloth,
    cube_cloth:   Cloth,
    light:        Light,
    camera:       Camera,
    cloth_sim:    ClothSim,
    rigid_sim:    RigidSimCore,
    rigid_params: RigidSimParams,
    params:       Rc<RefCell<SimParams>>,
    canvas:       HtmlCanvasElement,
    keys:         [bool; 8],
}

/// Run cloth and a spinning cube together on the same canvas.
/// They are simulated independently (no interaction yet).
/// Mouse drag picks cloth vertices; arrow keys / WASD orbit the camera.
#[wasm_bindgen]
pub async fn run(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let window = web_sys::window().unwrap();
    let canvas = window
        .document().unwrap()
        .get_element_by_id(canvas_id).unwrap()
        .dyn_into::<HtmlCanvasElement>().unwrap();

    let ctx    = GpuContext::new(canvas.clone()).await?;
    let light  = Light::new(&ctx, [2.0, 0.0, 0.5]);
    let camera = Camera::new(&ctx);

    // ── Cloth sim ──────────────────────────────────────────────────────────
    let cloth     = Cloth::new(&ctx, 32, &light);
    let cloth_sim = ClothSim::from_grid(32);

    // ── Rigid body: spinning cube ──────────────────────────────────────────
    const H: f32 = 0.125;
    let cube_verts: &[[f32; 3]] = &[
        [-H, -H, -H], [ H, -H, -H], [ H,  H, -H], [-H,  H, -H],
        [-H, -H,  H], [ H, -H,  H], [ H,  H,  H], [-H,  H,  H],
    ];
    let cube_faces: &[[u32; 3]] = &[
        [4, 5, 6], [4, 6, 7], // +Z
        [1, 0, 3], [1, 3, 2], // -Z
        [0, 4, 7], [0, 7, 3], // -X
        [5, 1, 2], [5, 2, 6], // +X
        [7, 6, 2], [7, 2, 3], // +Y
        [0, 1, 5], [0, 5, 4], // -Y
    ];
    let template = RigidBodyTemplate::new(cube_verts, cube_faces);
    let body = RigidBodyInstance::new(
        template,
        na::Vector3::zeros(),                // same origin as cloth
        na::Vector3::zeros(),
        na::Vector3::zeros(),
        na::Vector3::new(1.0, 2.0, 0.5),    // initial angular velocity (rad/s)
        1000.0,
    );
    let rigid_sim    = RigidSimCore::new(vec![body]);
    let rigid_params = RigidSimParams { gravity_enabled: false, ..Default::default() };
    let cube_cloth   = Cloth::from_mesh(
        &ctx,
        cube_verts.to_vec(),
        cube_faces.to_vec(),
        vec![[0.72, 0.53, 0.30]; 8],
        HashMap::new(),
        &light,
    );

    let fps_div = window.document().unwrap().create_element("div").unwrap();
    fps_div.set_inner_html("FPS: 0");
    fps_div.set_attribute("style",
        "position:fixed; top:10px; left:10px; color:white; font-family:monospace; z-index:1000;"
    ).unwrap();
    window.document().unwrap().body().unwrap().append_child(&fps_div).unwrap();

    let state = Rc::new(RefCell::new(CombinedAppState {
        ctx, cloth, cube_cloth, light, camera,
        cloth_sim, rigid_sim, rigid_params,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(), keys: [false; 8],
    }));
    COMBINED_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    fn to_ndc(event: &MouseEvent, canvas: &HtmlCanvasElement) -> (f32, f32) {
        let w = canvas.offset_width()  as f32;
        let h = canvas.offset_height() as f32;
        ( (event.offset_x() as f32 / w) * 2.0 - 1.0,
         -(event.offset_y() as f32 / h) * 2.0 + 1.0)
    }

    fn to_ndc_touch(touch: &web_sys::Touch, canvas: &HtmlCanvasElement) -> (f32, f32) {
        let rect = canvas.get_bounding_client_rect();
        let ox = touch.client_x() as f32 - rect.left() as f32;
        let oy = touch.client_y() as f32 - rect.top()  as f32;
        ( (ox / rect.width()  as f32) * 2.0 - 1.0,
         -(oy / rect.height() as f32) * 2.0 + 1.0)
    }

    // Mouse drag — picks cloth vertices only
    let state_md = state.clone();
    let mousedown = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut s = state_md.borrow_mut();
        let (nx, ny) = to_ndc(&e, &s.canvas);
        let mut best_idx = 0usize; let mut best_dist = f32::MAX;
        for i in 0..s.cloth_sim.q.nrows() {
            let wp = [s.cloth_sim.q[(i,0)], s.cloth_sim.q[(i,1)], s.cloth_sim.q[(i,2)]];
            let (px, py) = project_to_ndc(wp, &s.camera);
            let d2 = (px-nx)*(px-nx) + (py-ny)*(py-ny);
            if d2 < best_dist { best_dist = d2; best_idx = i; }
        }
        let vw = [s.cloth_sim.q[(best_idx,0)], s.cloth_sim.q[(best_idx,1)], s.cloth_sim.q[(best_idx,2)]];
        s.cloth_sim.clicked_vertex    = Some(best_idx);
        s.cloth_sim.dragging_vertices = None;
        s.cloth_sim.mouse_pos = ray_plane_intersect(nx, ny, vw, &s.camera).unwrap_or(vw);
    }));
    canvas.add_event_listener_with_callback("mousedown", mousedown.as_ref().unchecked_ref())?;
    mousedown.forget();

    let state_mm = state.clone();
    let mousemove = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut s = state_mm.borrow_mut();
        if let Some(v) = s.cloth_sim.clicked_vertex {
            let (nx, ny) = to_ndc(&e, &s.canvas);
            let cp = [s.cloth_sim.q[(v,0)], s.cloth_sim.q[(v,1)], s.cloth_sim.q[(v,2)]];
            if let Some(world) = ray_plane_intersect(nx, ny, cp, &s.camera) {
                s.cloth_sim.mouse_pos = world;
            }
        }
    }));
    canvas.add_event_listener_with_callback("mousemove", mousemove.as_ref().unchecked_ref())?;
    mousemove.forget();

    let state_mu = state.clone();
    let mouseup = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |_: MouseEvent| {
        state_mu.borrow_mut().cloth_sim.clicked_vertex = None;
    }));
    canvas.add_event_listener_with_callback("mouseup", mouseup.as_ref().unchecked_ref())?;
    mouseup.forget();

    let state_ts = state.clone();
    let touchstart = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |e: TouchEvent| {
        e.prevent_default();
        let touch = match e.touches().get(0) { Some(t) => t, None => return };
        let mut s = state_ts.borrow_mut();
        let (nx, ny) = to_ndc_touch(&touch, &s.canvas);
        let mut best_idx = 0usize; let mut best_dist = f32::MAX;
        for i in 0..s.cloth_sim.q.nrows() {
            let wp = [s.cloth_sim.q[(i,0)], s.cloth_sim.q[(i,1)], s.cloth_sim.q[(i,2)]];
            let (px, py) = project_to_ndc(wp, &s.camera);
            let d2 = (px-nx)*(px-nx) + (py-ny)*(py-ny);
            if d2 < best_dist { best_dist = d2; best_idx = i; }
        }
        let vw = [s.cloth_sim.q[(best_idx,0)], s.cloth_sim.q[(best_idx,1)], s.cloth_sim.q[(best_idx,2)]];
        s.cloth_sim.clicked_vertex = Some(best_idx);
        s.cloth_sim.mouse_pos = ray_plane_intersect(nx, ny, vw, &s.camera).unwrap_or(vw);
    }));
    canvas.add_event_listener_with_callback("touchstart", touchstart.as_ref().unchecked_ref())?;
    touchstart.forget();

    let state_tm = state.clone();
    let touchmove = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |e: TouchEvent| {
        e.prevent_default();
        let touch = match e.touches().get(0) { Some(t) => t, None => return };
        let mut s = state_tm.borrow_mut();
        if let Some(v) = s.cloth_sim.clicked_vertex {
            let (nx, ny) = to_ndc_touch(&touch, &s.canvas);
            let cp = [s.cloth_sim.q[(v,0)], s.cloth_sim.q[(v,1)], s.cloth_sim.q[(v,2)]];
            if let Some(world) = ray_plane_intersect(nx, ny, cp, &s.camera) {
                s.cloth_sim.mouse_pos = world;
            }
        }
    }));
    canvas.add_event_listener_with_callback("touchmove", touchmove.as_ref().unchecked_ref())?;
    touchmove.forget();

    let state_te = state.clone();
    let touchend = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |_: TouchEvent| {
        state_te.borrow_mut().cloth_sim.clicked_vertex = None;
    }));
    canvas.add_event_listener_with_callback("touchend", touchend.as_ref().unchecked_ref())?;
    touchend.forget();

    let state_kd = state.clone();
    let keydown = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            e.prevent_default();
            state_kd.borrow_mut().keys[idx] = true;
        }
    }));
    window.add_event_listener_with_callback("keydown", keydown.as_ref().unchecked_ref())?;
    keydown.forget();

    let state_ku = state.clone();
    let keyup = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            state_ku.borrow_mut().keys[idx] = false;
        }
    }));
    window.add_event_listener_with_callback("keyup", keyup.as_ref().unchecked_ref())?;
    keyup.forget();

    let loop_fn: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let loop_fn_inner = loop_fn.clone();
    let state_inner   = state.clone();
    let last_time = Rc::new(RefCell::new(0.0f64));
    let fps       = Rc::new(RefCell::new(0.0f64));

    *loop_fn.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mut s = state_inner.borrow_mut();
        let CombinedAppState {
            cloth_sim, cloth, rigid_sim, rigid_params, cube_cloth,
            ctx, light, camera, params, keys, ..
        } = &mut *s;

        const ROT_SPEED: f32 = 0.02;
        const MOV_SPEED: f32 = 0.03;
        if keys[0] { camera.yaw   -= ROT_SPEED; }
        if keys[1] { camera.yaw   += ROT_SPEED; }
        if keys[2] { camera.pitch += ROT_SPEED; }
        if keys[3] { camera.pitch -= ROT_SPEED; }
        camera.pitch = camera.pitch.clamp(-1.5, 1.5);
        let fwd   = camera.forward();
        let right = camera.right();
        if keys[4] { for i in 0..3 { camera.target[i] -= right[i] * MOV_SPEED; } }
        if keys[5] { for i in 0..3 { camera.target[i] += right[i] * MOV_SPEED; } }
        if keys[6] { for i in 0..3 { camera.target[i] += fwd[i]   * MOV_SPEED; } }
        if keys[7] { for i in 0..3 { camera.target[i] -= fwd[i]   * MOV_SPEED; } }
        if keys.iter().any(|&k| k) { camera.update(&ctx.queue); }

        // Step cloth
        cloth_sim.step(&params.borrow());
        cloth.sync_from_sim(&cloth_sim.q, ctx);

        // Step rigid body and push updated world-space vertices to render mesh
        rigid_sim.step(rigid_params);
        let world_verts = rigid_sim.bodies[0].world_vertices();
        for (i, pos) in cube_cloth.positions.iter_mut().enumerate() {
            let v = world_verts[i];
            *pos = [v[0], v[1], v[2]];
        }
        cube_cloth.upload(ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            cloth.render(ctx, &view, light, camera);
            cube_cloth.render_over(ctx, &view, light, camera);
            frame.present();
        }

        let now = web_sys::window().unwrap().performance().unwrap().now();
        let mut last = last_time.borrow_mut();
        let dt = now - *last;
        *last = now;
        if dt > 0.0 { *fps.borrow_mut() = 1000.0 / dt; }
        fps_div.set_inner_html(&format!("FPS: {:.1}", *fps.borrow()));

        web_sys::window().unwrap()
            .request_animation_frame(
                loop_fn_inner.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
            ).unwrap();
    }) as Box<dyn FnMut()>));

    window.request_animation_frame(
        loop_fn.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
    )?;

    Ok(())
}

#[wasm_bindgen]
pub async fn run_cloth(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let window = web_sys::window().unwrap();
    let canvas = window
        .document().unwrap()
        .get_element_by_id(canvas_id).unwrap()
        .dyn_into::<HtmlCanvasElement>().unwrap();

    let ctx    = GpuContext::new(canvas.clone()).await?;
    let light  = Light::new(&ctx, [2.0, 0.0, 0.5]);
    let camera = Camera::new(&ctx);
    let cloth  = Cloth::new(&ctx, 32, &light);
    let sim    = ClothSim::from_grid(32);

    let state = Rc::new(RefCell::new(AppState {
        ctx,
        cloth,
        light,
        camera,
        sim,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(),
        keys: [false; 8],
    }));
    APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    let fps_div = window.document().unwrap()
        .create_element("div").unwrap();

    fps_div.set_inner_html("FPS: 0");
    fps_div.set_attribute("style",
                          "position:fixed; top:10px; left:10px; color:white; font-family:monospace; z-index:1000;"
    ).unwrap();

    window.document().unwrap().body().unwrap().append_child(&fps_div).unwrap();

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
        s.sim.dragging_vertices = None;
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

    // Convert a Touch's client coordinates to NDC [-1, 1] relative to the canvas.
    fn to_ndc_touch(touch: &web_sys::Touch, canvas: &HtmlCanvasElement) -> (f32, f32) {
        let rect = canvas.get_bounding_client_rect();
        let ox = touch.client_x() as f32 - rect.left() as f32;
        let oy = touch.client_y() as f32 - rect.top()  as f32;
        let w  = rect.width()  as f32;
        let h  = rect.height() as f32;
        let nx =  (ox / w) * 2.0 - 1.0;
        let ny = -(oy / h) * 2.0 + 1.0;
        (nx, ny)
    }

    // touchstart — same as mousedown
    let state_ts = state.clone();
    let touchstart = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |e: TouchEvent| {
        e.prevent_default();
        let touch = match e.touches().get(0) { Some(t) => t, None => return };
        let mut s = state_ts.borrow_mut();
        let (nx, ny) = to_ndc_touch(&touch, &s.canvas);

        let mut best_idx  = 0usize;
        let mut best_dist = f32::MAX;
        for i in 0..s.sim.q.nrows() {
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
    canvas.add_event_listener_with_callback("touchstart", touchstart.as_ref().unchecked_ref())?;
    touchstart.forget();

    // touchmove — same as mousemove
    let state_tm = state.clone();
    let touchmove = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |e: TouchEvent| {
        e.prevent_default();
        let touch = match e.touches().get(0) { Some(t) => t, None => return };
        let mut s = state_tm.borrow_mut();
        if let Some(v) = s.sim.clicked_vertex {
            let (nx, ny) = to_ndc_touch(&touch, &s.canvas);
            let current_pos = [s.sim.q[(v, 0)], s.sim.q[(v, 1)], s.sim.q[(v, 2)]];
            if let Some(world) = ray_plane_intersect(nx, ny, current_pos, &s.camera) {
                s.sim.mouse_pos = world;
            }
        }
    }));
    canvas.add_event_listener_with_callback("touchmove", touchmove.as_ref().unchecked_ref())?;
    touchmove.forget();

    // touchend — same as mouseup
    let state_te = state.clone();
    let touchend = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |_: TouchEvent| {
        state_te.borrow_mut().sim.clicked_vertex = None;
    }));
    canvas.add_event_listener_with_callback("touchend", touchend.as_ref().unchecked_ref())?;
    touchend.forget();

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
    let last_time = Rc::new(RefCell::new(0.0));
    let fps = Rc::new(RefCell::new(0.0));

    *loop_fn.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mut s = state_inner.borrow_mut();
        let AppState { sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        // Arrows orbit (~0.02 rad/frame); WASD translate (~0.03 units/frame).
        const ROT_SPEED: f32 = 0.02;
        const MOV_SPEED: f32 = 0.03;
        if keys[0] { camera.yaw   -= ROT_SPEED; }
        if keys[1] { camera.yaw   += ROT_SPEED; }
        if keys[2] { camera.pitch += ROT_SPEED; }
        if keys[3] { camera.pitch -= ROT_SPEED; }
        camera.pitch = camera.pitch.clamp(-1.5, 1.5);
        let fwd = camera.forward();
        let right = camera.right();
        if keys[4] { for i in 0..3 { camera.target[i] -= right[i] * MOV_SPEED; } }
        if keys[5] { for i in 0..3 { camera.target[i] += right[i] * MOV_SPEED; } }
        if keys[6] { for i in 0..3 { camera.target[i] += fwd[i]   * MOV_SPEED; } }
        if keys[7] { for i in 0..3 { camera.target[i] -= fwd[i]   * MOV_SPEED; } }

        if keys.iter().any(|&k| k) {
            camera.update(&ctx.queue);
        }

        sim.step(&params.borrow());
        cloth.sync_from_sim(&sim.q, ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }

        let now = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();

        let mut last = last_time.borrow_mut();
        let dt = now - *last;
        *last = now;

        if dt > 0.0 {
            *fps.borrow_mut() = 1000.0 / dt;
        }
        fps_div.set_inner_html(&format!("FPS: {:.1}", *fps.borrow()));

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

/// Start the paper simulation on the given canvas.
///
/// Sets up a center vertical fold using all interior edges along column `n/2`.
/// All other behaviour (dragging, camera, self-collision, etc.) is identical
/// to the cloth simulation.
#[wasm_bindgen]
pub async fn run_paper(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let window  = web_sys::window().unwrap();
    let canvas  = window
        .document().unwrap()
        .get_element_by_id(canvas_id).unwrap()
        .dyn_into::<HtmlCanvasElement>().unwrap();

    let resolution = PARAMS.with(|p| p.borrow().resolution as usize);
    let ctx    = GpuContext::new(canvas.clone()).await?;
    let light  = Light::new(&ctx, [2.0, 0.0, 0.5]);
    let camera = Camera::new(&ctx);
    let cloth  = Cloth::new(&ctx, resolution as u32, &light);

    let mut sim = PaperSim::from_grid(resolution);

    // Register a center vertical fold: all interior edges at column n/2.
    let n   = resolution;
    let col = n / 2;
    let mut fold_map: HashMap<(u32, u32), FoldSpec> = HashMap::new();
    for row in 0..(n - 1) {
        let a = (row * n + col) as u32;
        let b = ((row + 1) * n + col) as u32;
        let lo = a.min(b); let hi = a.max(b);
        fold_map.insert((lo, hi), FoldSpec { target_angle: std::f32::consts::PI, compliance: 1e-4, direction: FoldDirection::Mountain, damping: 0.5 });
    }
    sim.set_fold_map(fold_map);

    let state = Rc::new(RefCell::new(PaperAppState {
        ctx, cloth, light, camera, sim,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(),
        keys: [false; 8],
        cp_data: None,
        resolution,
    }));
    PAPER_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    // ── Event handlers (identical pattern to run(), but via PAPER_APP_STATE) ──

    fn to_ndc_p(event: &MouseEvent, canvas: &HtmlCanvasElement) -> (f32, f32) {
        let w = canvas.offset_width()  as f32;
        let h = canvas.offset_height() as f32;
        let nx =  (event.offset_x() as f32 / w) * 2.0 - 1.0;
        let ny = -(event.offset_y() as f32 / h) * 2.0 + 1.0;
        (nx, ny)
    }

    let state_md = state.clone();
    let mousedown = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut s = state_md.borrow_mut();
        let (nx, ny) = to_ndc_p(&e, &s.canvas);
        let mut best_idx = 0usize; let mut best_dist = f32::MAX;
        for i in 0..s.sim.q.nrows() {
            let wp = [s.sim.q[(i,0)], s.sim.q[(i,1)], s.sim.q[(i,2)]];
            let (px, py) = project_to_ndc(wp, &s.camera);
            let d2 = (px-nx)*(px-nx) + (py-ny)*(py-ny);
            if d2 < best_dist { best_dist = d2; best_idx = i; }
        }
        let vw = [s.sim.q[(best_idx,0)], s.sim.q[(best_idx,1)], s.sim.q[(best_idx,2)]];
        s.sim.clicked_vertex   = Some(best_idx);
        s.sim.dragging_vertices = None;
        s.sim.mouse_pos = ray_plane_intersect(nx, ny, vw, &s.camera).unwrap_or(vw);
    }));
    canvas.add_event_listener_with_callback("mousedown", mousedown.as_ref().unchecked_ref())?;
    mousedown.forget();

    let state_mm = state.clone();
    let mousemove = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut s = state_mm.borrow_mut();
        if let Some(v) = s.sim.clicked_vertex {
            let (nx, ny) = to_ndc_p(&e, &s.canvas);
            let cp = [s.sim.q[(v,0)], s.sim.q[(v,1)], s.sim.q[(v,2)]];
            if let Some(world) = ray_plane_intersect(nx, ny, cp, &s.camera) {
                s.sim.mouse_pos = world;
            }
        }
    }));
    canvas.add_event_listener_with_callback("mousemove", mousemove.as_ref().unchecked_ref())?;
    mousemove.forget();

    let state_mu = state.clone();
    let mouseup = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |_: MouseEvent| {
        state_mu.borrow_mut().sim.clicked_vertex = None;
    }));
    canvas.add_event_listener_with_callback("mouseup", mouseup.as_ref().unchecked_ref())?;
    mouseup.forget();

    fn to_ndc_touch_p(touch: &web_sys::Touch, canvas: &HtmlCanvasElement) -> (f32, f32) {
        let rect = canvas.get_bounding_client_rect();
        let ox = touch.client_x() as f32 - rect.left() as f32;
        let oy = touch.client_y() as f32 - rect.top()  as f32;
        let nx =  (ox / rect.width()  as f32) * 2.0 - 1.0;
        let ny = -(oy / rect.height() as f32) * 2.0 + 1.0;
        (nx, ny)
    }

    let state_ts = state.clone();
    let touchstart = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |e: TouchEvent| {
        e.prevent_default();
        let touch = match e.touches().get(0) { Some(t) => t, None => return };
        let mut s = state_ts.borrow_mut();
        let (nx, ny) = to_ndc_touch_p(&touch, &s.canvas);
        let mut best_idx = 0usize; let mut best_dist = f32::MAX;
        for i in 0..s.sim.q.nrows() {
            let wp = [s.sim.q[(i,0)], s.sim.q[(i,1)], s.sim.q[(i,2)]];
            let (px, py) = project_to_ndc(wp, &s.camera);
            let d2 = (px-nx)*(px-nx) + (py-ny)*(py-ny);
            if d2 < best_dist { best_dist = d2; best_idx = i; }
        }
        let vw = [s.sim.q[(best_idx,0)], s.sim.q[(best_idx,1)], s.sim.q[(best_idx,2)]];
        s.sim.clicked_vertex = Some(best_idx);
        s.sim.mouse_pos = ray_plane_intersect(nx, ny, vw, &s.camera).unwrap_or(vw);
    }));
    canvas.add_event_listener_with_callback("touchstart", touchstart.as_ref().unchecked_ref())?;
    touchstart.forget();

    let state_tm = state.clone();
    let touchmove = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |e: TouchEvent| {
        e.prevent_default();
        let touch = match e.touches().get(0) { Some(t) => t, None => return };
        let mut s = state_tm.borrow_mut();
        if let Some(v) = s.sim.clicked_vertex {
            let (nx, ny) = to_ndc_touch_p(&touch, &s.canvas);
            let cp = [s.sim.q[(v,0)], s.sim.q[(v,1)], s.sim.q[(v,2)]];
            if let Some(world) = ray_plane_intersect(nx, ny, cp, &s.camera) {
                s.sim.mouse_pos = world;
            }
        }
    }));
    canvas.add_event_listener_with_callback("touchmove", touchmove.as_ref().unchecked_ref())?;
    touchmove.forget();

    let state_te = state.clone();
    let touchend = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |_: TouchEvent| {
        state_te.borrow_mut().sim.clicked_vertex = None;
    }));
    canvas.add_event_listener_with_callback("touchend", touchend.as_ref().unchecked_ref())?;
    touchend.forget();

    let state_kd = state.clone();
    let keydown = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            e.prevent_default();
            state_kd.borrow_mut().keys[idx] = true;
        }
    }));
    window.add_event_listener_with_callback("keydown", keydown.as_ref().unchecked_ref())?;
    keydown.forget();

    let state_ku = state.clone();
    let keyup = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            state_ku.borrow_mut().keys[idx] = false;
        }
    }));
    window.add_event_listener_with_callback("keyup", keyup.as_ref().unchecked_ref())?;
    keyup.forget();

    // ── RAF loop ──────────────────────────────────────────────────────────────
    let loop_fn: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let loop_fn_inner = loop_fn.clone();
    let state_inner   = state.clone();

    *loop_fn.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mut s = state_inner.borrow_mut();
        let PaperAppState { sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        const ROT_SPEED: f32 = 0.02;
        const MOV_SPEED: f32 = 0.03;
        if keys[0] { camera.yaw   -= ROT_SPEED; }
        if keys[1] { camera.yaw   += ROT_SPEED; }
        if keys[2] { camera.pitch += ROT_SPEED; }
        if keys[3] { camera.pitch -= ROT_SPEED; }
        camera.pitch = camera.pitch.clamp(-1.5, 1.5);
        let fwd = camera.forward();
        let right = camera.right();
        if keys[4] { for i in 0..3 { camera.target[i] -= right[i] * MOV_SPEED; } }
        if keys[5] { for i in 0..3 { camera.target[i] += right[i] * MOV_SPEED; } }
        if keys[6] { for i in 0..3 { camera.target[i] += fwd[i]   * MOV_SPEED; } }
        if keys[7] { for i in 0..3 { camera.target[i] -= fwd[i]   * MOV_SPEED; } }
        if keys.iter().any(|&k| k) { camera.update(&ctx.queue); }

        sim.step(&params.borrow());
        cloth.sync_from_sim(&sim.q, ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }

        web_sys::window().unwrap()
            .request_animation_frame(
                loop_fn_inner.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
            ).unwrap();
    }) as Box<dyn FnMut()>));

    window.request_animation_frame(
        loop_fn.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
    )?;

    Ok(())
}

/// Start the paper simulation with a crease pattern (.cp file contents).
#[wasm_bindgen]
pub async fn run_paper_with_cp(canvas_id: &str, cp_data: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let window = web_sys::window().unwrap();
    let canvas = window
        .document().unwrap()
        .get_element_by_id(canvas_id).unwrap()
        .dyn_into::<HtmlCanvasElement>().unwrap();

    let resolution = PARAMS.with(|p| p.borrow().resolution as usize);
    let ctx = GpuContext::new(canvas.clone()).await?;
    let light = Light::new(&ctx, [2.0, 0.0, 0.5]);
    let camera = Camera::new(&ctx);

    // Parse creasepattern and build mesh
    let cp = CreasePattern::parse(cp_data)
        .map_err(|e| JsValue::from_str(&e))?;
    let (sim, positions, faces, colors, edge_colors) = PaperSim::from_crease_pattern(&cp, resolution);
    let cloth = Cloth::from_mesh(&ctx, positions, faces, colors, edge_colors, &light);

    let state = Rc::new(RefCell::new(PaperAppState {
        ctx, cloth, light, camera, sim,
        params: PARAMS.with(|p| p.clone()),
        canvas: canvas.clone(),
        keys: [false; 8],
        cp_data: Some(cp_data.to_string()),
        resolution,
    }));
    PAPER_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    // ── Event handlers ────────────────────────────────────────────────────────

    fn to_ndc_p(event: &MouseEvent, canvas: &HtmlCanvasElement) -> (f32, f32) {
        let w = canvas.offset_width() as f32;
        let h = canvas.offset_height() as f32;
        let nx = (event.offset_x() as f32 / w) * 2.0 - 1.0;
        let ny = -(event.offset_y() as f32 / h) * 2.0 + 1.0;
        (nx, ny)
    }

    let state_md = state.clone();
    let mousedown = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut s = state_md.borrow_mut();
        let (nx, ny) = to_ndc_p(&e, &s.canvas);
        let mut best_idx = 0usize; let mut best_dist = f32::MAX;
        for i in 0..s.sim.q.nrows() {
            let wp = [s.sim.q[(i,0)], s.sim.q[(i,1)], s.sim.q[(i,2)]];
            let (px, py) = project_to_ndc(wp, &s.camera);
            let d2 = (px-nx)*(px-nx) + (py-ny)*(py-ny);
            if d2 < best_dist { best_dist = d2; best_idx = i; }
        }
        let vw = [s.sim.q[(best_idx,0)], s.sim.q[(best_idx,1)], s.sim.q[(best_idx,2)]];
        s.sim.clicked_vertex = Some(best_idx);
        s.sim.dragging_vertices = None;
        s.sim.mouse_pos = ray_plane_intersect(nx, ny, vw, &s.camera).unwrap_or(vw);
    }));
    canvas.add_event_listener_with_callback("mousedown", mousedown.as_ref().unchecked_ref())?;
    mousedown.forget();

    let state_mm = state.clone();
    let mousemove = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut s = state_mm.borrow_mut();
        if let Some(v) = s.sim.clicked_vertex {
            let (nx, ny) = to_ndc_p(&e, &s.canvas);
            let cp = [s.sim.q[(v,0)], s.sim.q[(v,1)], s.sim.q[(v,2)]];
            if let Some(world) = ray_plane_intersect(nx, ny, cp, &s.camera) {
                s.sim.mouse_pos = world;
            }
        }
    }));
    canvas.add_event_listener_with_callback("mousemove", mousemove.as_ref().unchecked_ref())?;
    mousemove.forget();

    let state_mu = state.clone();
    let mouseup = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |_: MouseEvent| {
        state_mu.borrow_mut().sim.clicked_vertex = None;
    }));
    canvas.add_event_listener_with_callback("mouseup", mouseup.as_ref().unchecked_ref())?;
    mouseup.forget();

    fn to_ndc_touch_p(touch: &web_sys::Touch, canvas: &HtmlCanvasElement) -> (f32, f32) {
        let rect = canvas.get_bounding_client_rect();
        let ox = touch.client_x() as f32 - rect.left() as f32;
        let oy = touch.client_y() as f32 - rect.top() as f32;
        let nx = (ox / rect.width() as f32) * 2.0 - 1.0;
        let ny = -(oy / rect.height() as f32) * 2.0 + 1.0;
        (nx, ny)
    }

    let state_ts = state.clone();
    let touchstart = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |e: TouchEvent| {
        e.prevent_default();
        let touch = match e.touches().get(0) { Some(t) => t, None => return };
        let mut s = state_ts.borrow_mut();
        let (nx, ny) = to_ndc_touch_p(&touch, &s.canvas);
        let mut best_idx = 0usize; let mut best_dist = f32::MAX;
        for i in 0..s.sim.q.nrows() {
            let wp = [s.sim.q[(i,0)], s.sim.q[(i,1)], s.sim.q[(i,2)]];
            let (px, py) = project_to_ndc(wp, &s.camera);
            let d2 = (px-nx)*(px-nx) + (py-ny)*(py-ny);
            if d2 < best_dist { best_dist = d2; best_idx = i; }
        }
        let vw = [s.sim.q[(best_idx,0)], s.sim.q[(best_idx,1)], s.sim.q[(best_idx,2)]];
        s.sim.clicked_vertex = Some(best_idx);
        s.sim.mouse_pos = ray_plane_intersect(nx, ny, vw, &s.camera).unwrap_or(vw);
    }));
    canvas.add_event_listener_with_callback("touchstart", touchstart.as_ref().unchecked_ref())?;
    touchstart.forget();

    let state_tm = state.clone();
    let touchmove = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |e: TouchEvent| {
        e.prevent_default();
        let touch = match e.touches().get(0) { Some(t) => t, None => return };
        let mut s = state_tm.borrow_mut();
        if let Some(v) = s.sim.clicked_vertex {
            let (nx, ny) = to_ndc_touch_p(&touch, &s.canvas);
            let cp = [s.sim.q[(v,0)], s.sim.q[(v,1)], s.sim.q[(v,2)]];
            if let Some(world) = ray_plane_intersect(nx, ny, cp, &s.camera) {
                s.sim.mouse_pos = world;
            }
        }
    }));
    canvas.add_event_listener_with_callback("touchmove", touchmove.as_ref().unchecked_ref())?;
    touchmove.forget();

    let state_te = state.clone();
    let touchend = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |_: TouchEvent| {
        state_te.borrow_mut().sim.clicked_vertex = None;
    }));
    canvas.add_event_listener_with_callback("touchend", touchend.as_ref().unchecked_ref())?;
    touchend.forget();

    let state_kd = state.clone();
    let keydown = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            e.prevent_default();
            state_kd.borrow_mut().keys[idx] = true;
        }
    }));
    window.add_event_listener_with_callback("keydown", keydown.as_ref().unchecked_ref())?;
    keydown.forget();

    let state_ku = state.clone();
    let keyup = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            state_ku.borrow_mut().keys[idx] = false;
        }
    }));
    window.add_event_listener_with_callback("keyup", keyup.as_ref().unchecked_ref())?;
    keyup.forget();

    // ── RAF loop ──────────────────────────────────────────────────────────────
    let loop_fn: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let loop_fn_inner = loop_fn.clone();
    let state_inner = state.clone();

    *loop_fn.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mut s = state_inner.borrow_mut();
        let PaperAppState { sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        const ROT_SPEED: f32 = 0.02;
        const MOV_SPEED: f32 = 0.03;
        if keys[0] { camera.yaw -= ROT_SPEED; }
        if keys[1] { camera.yaw += ROT_SPEED; }
        if keys[2] { camera.pitch += ROT_SPEED; }
        if keys[3] { camera.pitch -= ROT_SPEED; }
        camera.pitch = camera.pitch.clamp(-1.5, 1.5);
        let fwd = camera.forward();
        let right = camera.right();
        if keys[4] { for i in 0..3 { camera.target[i] -= right[i] * MOV_SPEED; } }
        if keys[5] { for i in 0..3 { camera.target[i] += right[i] * MOV_SPEED; } }
        if keys[6] { for i in 0..3 { camera.target[i] += fwd[i]   * MOV_SPEED; } }
        if keys[7] { for i in 0..3 { camera.target[i] -= fwd[i]   * MOV_SPEED; } }
        if keys.iter().any(|&k| k) { camera.update(&ctx.queue); }

        sim.step(&params.borrow());
        cloth.sync_from_sim(&sim.q, ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }

        web_sys::window().unwrap()
            .request_animation_frame(
                loop_fn_inner.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
            ).unwrap();
    }) as Box<dyn FnMut()>));

    window.request_animation_frame(
        loop_fn.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
    )?;

    Ok(())
}

/// Set the XPBD compliance α for all paper hinges.
/// Smaller = stiffer / faster fold (1e-6 … 1e-2 typical range).
#[wasm_bindgen]
pub fn set_paper_hinge_compliance(alpha: f64) {
    PAPER_APP_STATE.with(|a| {
        if let Some(state) = a.borrow().as_ref() {
            let mut s = state.borrow_mut();
            for hinge in &mut s.sim.hinges {
                hinge.compliance = alpha as f32;
            }
        }
    });
}

/// Set XPBD constraint damping for all paper hinges.
/// Higher = more damping of angular velocity along constraint gradient (0 … 10 typical).
/// Default 0.5; set to 0 for no additional constraint damping.
#[wasm_bindgen]
pub fn set_paper_hinge_damping(beta: f64) {
    PAPER_APP_STATE.with(|a| {
        if let Some(state) = a.borrow().as_ref() {
            let mut s = state.borrow_mut();
            for hinge in &mut s.sim.hinges {
                hinge.damping = beta as f32;
            }
        }
    });
}

/// Set how fast the hinge goal angle tracks the user target (radians / second).
/// Higher = more responsive but risks instability on large jumps.
/// Default 5.0 rad/s (≈286 °/s; a 90° fold takes ~0.3 s).
#[wasm_bindgen]
pub fn set_paper_fold_speed(rads_per_sec: f64) {
    PAPER_APP_STATE.with(|a| {
        if let Some(state) = a.borrow().as_ref() {
            state.borrow_mut().sim.fold_speed = rads_per_sec as f32;
        }
    });
}

/// Set the target dihedral angle for all paper hinges.
///
/// `degrees` is the dihedral angle in degrees:
/// - 180° → flat (no fold)
/// - 90°  → 90° fold
/// - 0°   → panels face each other
///
/// Mountain and valley folds fold in opposite directions.
#[wasm_bindgen]
pub fn set_paper_fold_angle(degrees: f64) {
    let desired_dihedral = degrees as f32 * std::f32::consts::PI / 180.0;
    // target_angle is the offset from rest_angle (π).
    // For mountain: goal_dihedral = rest + target, we want goal = desired_dihedral
    //   so target = desired - π (e.g., 0° → target = -π, 180° → target = 0)
    // For valley: fold in opposite direction, so target = π - desired
    PAPER_APP_STATE.with(|a| {
        if let Some(state) = a.borrow().as_ref() {
            let mut s = state.borrow_mut();
            for hinge in &mut s.sim.hinges {
                hinge.target_angle = match hinge.direction {
                    crate::sim::FoldDirection::Mountain => desired_dihedral - std::f32::consts::PI,
                    crate::sim::FoldDirection::Valley => std::f32::consts::PI - desired_dihedral,
                };
            }
        }
    });
}

/// Set the fold amount for all paper hinges (direction-aware).
///
/// `degrees` is the fold amount:
/// - 0°   → flat (no fold)
/// - 180° → fully folded
///
/// Mountain folds decrease dihedral angle, valley folds increase it.
#[wasm_bindgen]
pub fn set_paper_fold_amount(degrees: f64) {
    PAPER_APP_STATE.with(|a| {
        if let Some(state) = a.borrow().as_ref() {
            state.borrow_mut().sim.set_fold_angle(degrees as f32);
        }
    });
}

/// Enable or disable wireframe overlay rendering.
#[wasm_bindgen]
pub fn set_wireframe_enabled(enabled: bool) {
    PAPER_APP_STATE.with(|a| {
        if let Some(state) = a.borrow().as_ref() {
            state.borrow_mut().cloth.wireframe_enabled = enabled;
        }
    });
}

/// Set the resolution for paper simulation, reloading the mesh/crease pattern.
#[wasm_bindgen]
pub fn set_paper_resolution(v: u32) {
    PARAMS.with(|p| p.borrow_mut().resolution = v);
    PAPER_APP_STATE.with(|a| {
        if let Some(state) = a.borrow().as_ref() {
            let mut s = state.borrow_mut();
            let resolution = v as usize;

            if let Some(ref cp_data) = s.cp_data {
                // Rebuild from crease pattern
                if let Ok(cp) = CreasePattern::parse(cp_data) {
                    let (sim, positions, faces, colors, edge_colors) =
                        PaperSim::from_crease_pattern(&cp, resolution);
                    s.cloth = Cloth::from_mesh(&s.ctx, positions, faces, colors, edge_colors, &s.light);
                    s.sim = sim;
                    s.resolution = resolution;
                }
            } else {
                // Rebuild simple grid
                let mut sim = PaperSim::from_grid(resolution);

                // Re-register center vertical fold
                let n = resolution;
                let col = n / 2;
                let mut fold_map: HashMap<(u32, u32), FoldSpec> = HashMap::new();
                for row in 0..(n - 1) {
                    let a = (row * n + col) as u32;
                    let b = ((row + 1) * n + col) as u32;
                    let lo = a.min(b); let hi = a.max(b);
                    fold_map.insert((lo, hi), FoldSpec {
                        target_angle: std::f32::consts::PI,
                        compliance: 1e-4,
                        direction: FoldDirection::Mountain,
                        damping: 0.5,
                    });
                }
                sim.set_fold_map(fold_map);

                s.cloth = Cloth::new(&s.ctx, resolution as u32, &s.light);
                s.sim = sim;
                s.resolution = resolution;
            }
        }
    });
}

/// Simulate a single rigid cube (0.25 × 0.25 × 0.25) spinning freely.
///
/// The cube is centered at the origin with no translational velocity and a
/// small initial angular velocity.  No inter-body gravity is applied (single
/// body).  Camera controls (arrow keys / WASD) work as in the other sims.
#[wasm_bindgen]
pub async fn run_rigid(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let window = web_sys::window().unwrap();
    let canvas = window
        .document().unwrap()
        .get_element_by_id(canvas_id).unwrap()
        .dyn_into::<HtmlCanvasElement>().unwrap();

    let ctx    = GpuContext::new(canvas.clone()).await?;
    let light  = Light::new(&ctx, [2.0, 0.0, 0.5]);
    let camera = Camera::new(&ctx);

    // ── Cube mesh: 8 vertices, 2 triangles per face, outward-CCW winding ──
    const H: f32 = 0.125; // half-side → 0.25 × 0.25 × 0.25 cube
    let verts: &[[f32; 3]] = &[
        [-H, -H, -H], [ H, -H, -H], [ H,  H, -H], [-H,  H, -H], // 0-3: -Z face
        [-H, -H,  H], [ H, -H,  H], [ H,  H,  H], [-H,  H,  H], // 4-7: +Z face
    ];
    let face_indices: &[[u32; 3]] = &[
        [4, 5, 6], [4, 6, 7], // +Z
        [1, 0, 3], [1, 3, 2], // -Z
        [0, 4, 7], [0, 7, 3], // -X
        [5, 1, 2], [5, 2, 6], // +X
        [7, 6, 2], [7, 2, 3], // +Y
        [0, 1, 5], [0, 5, 4], // -Y
    ];

    let template = RigidBodyTemplate::new(verts, face_indices);
    let body = RigidBodyInstance::new(
        template,
        na::Vector3::zeros(),                    // c: centered at origin
        na::Vector3::zeros(),                    // theta: no initial rotation
        na::Vector3::zeros(),                    // cvel: no translational velocity
        na::Vector3::new(1.0, 2.0, 0.5),        // w: gentle spin (rad/s)
        1000.0,                                  // density (kg/m³)
    );
    let sim    = RigidSimCore::new(vec![body]);
    let params = RigidSimParams { gravity_enabled: false, ..Default::default() };

    // Cloth is used purely for rendering — positions are updated each frame
    // from the body's current world-space vertices.
    let cloth = Cloth::from_mesh(
        &ctx,
        verts.to_vec(),
        face_indices.to_vec(),
        vec![[0.72, 0.53, 0.30]; 8], // warm wood colour
        HashMap::new(),
        &light,
    );

    let state = Rc::new(RefCell::new(RigidAppState {
        ctx, cloth, light, camera, sim, params,
        canvas: canvas.clone(), keys: [false; 8],
    }));
    RIGID_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    let state_kd = state.clone();
    let keydown = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            e.prevent_default();
            state_kd.borrow_mut().keys[idx] = true;
        }
    }));
    window.add_event_listener_with_callback("keydown", keydown.as_ref().unchecked_ref())?;
    keydown.forget();

    let state_ku = state.clone();
    let keyup = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            state_ku.borrow_mut().keys[idx] = false;
        }
    }));
    window.add_event_listener_with_callback("keyup", keyup.as_ref().unchecked_ref())?;
    keyup.forget();

    let loop_fn: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let loop_fn_inner = loop_fn.clone();
    let state_inner   = state.clone();

    *loop_fn.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mut s = state_inner.borrow_mut();
        let RigidAppState { sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        const ROT_SPEED: f32 = 0.02;
        const MOV_SPEED: f32 = 0.03;
        if keys[0] { camera.yaw   -= ROT_SPEED; }
        if keys[1] { camera.yaw   += ROT_SPEED; }
        if keys[2] { camera.pitch += ROT_SPEED; }
        if keys[3] { camera.pitch -= ROT_SPEED; }
        camera.pitch = camera.pitch.clamp(-1.5, 1.5);
        let fwd   = camera.forward();
        let right = camera.right();
        if keys[4] { for i in 0..3 { camera.target[i] -= right[i] * MOV_SPEED; } }
        if keys[5] { for i in 0..3 { camera.target[i] += right[i] * MOV_SPEED; } }
        if keys[6] { for i in 0..3 { camera.target[i] += fwd[i]   * MOV_SPEED; } }
        if keys[7] { for i in 0..3 { camera.target[i] -= fwd[i]   * MOV_SPEED; } }
        if keys.iter().any(|&k| k) { camera.update(&ctx.queue); }

        sim.step(params);

        // Push updated world-space vertex positions into the render mesh.
        let world_verts = sim.bodies[0].world_vertices();
        for (i, pos) in cloth.positions.iter_mut().enumerate() {
            let v = world_verts[i];
            *pos = [v[0], v[1], v[2]];
        }
        cloth.upload(ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }

        web_sys::window().unwrap()
            .request_animation_frame(
                loop_fn_inner.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
            ).unwrap();
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
        "a" | "A"    => Some(4),
        "d" | "D"    => Some(5),
        "w" | "W"    => Some(6),
        "s" | "S"    => Some(7),
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

// ── Param setters (called from JS panel) ──────────────────────────────────────

#[wasm_bindgen] pub fn set_time_step(v: f64)         { PARAMS.with(|p| p.borrow_mut().time_step = v); }
#[wasm_bindgen] pub fn set_constraint_iters(v: u32)  { PARAMS.with(|p| p.borrow_mut().constraint_iters = v); }
#[wasm_bindgen] pub fn set_gravity_enabled(v: bool)  { PARAMS.with(|p| p.borrow_mut().gravity_enabled = v); }
#[wasm_bindgen] pub fn set_gravity_g(v: f64)         { PARAMS.with(|p| p.borrow_mut().gravity_g = v); }
#[wasm_bindgen] pub fn set_pin_enabled(v: bool)      { PARAMS.with(|p| p.borrow_mut().pin_enabled = v); }
#[wasm_bindgen] pub fn set_pin_weight(v: f64)        { PARAMS.with(|p| p.borrow_mut().pin_weight = v); }
#[wasm_bindgen] pub fn set_stretch_enabled(v: bool)  { PARAMS.with(|p| p.borrow_mut().stretch_enabled = v); }
#[wasm_bindgen] pub fn set_stretch_weight(v: f64)    { PARAMS.with(|p| p.borrow_mut().stretch_weight = v); }
#[wasm_bindgen] pub fn set_bending_enabled(v: bool)  { PARAMS.with(|p| p.borrow_mut().bending_enabled = v); }
#[wasm_bindgen] pub fn set_bending_weight(v: f64)    { PARAMS.with(|p| p.borrow_mut().bending_weight = v); }
#[wasm_bindgen] pub fn set_pulling_enabled(v: bool)  { PARAMS.with(|p| p.borrow_mut().pulling_enabled = v); }
#[wasm_bindgen] pub fn set_pulling_weight(v: f64)    { PARAMS.with(|p| p.borrow_mut().pulling_weight = v); }
#[wasm_bindgen] pub fn set_self_collision_enabled(v: bool)          { PARAMS.with(|p| p.borrow_mut().self_collision_enabled = v); }
#[wasm_bindgen] pub fn set_self_collision_threshold(v: f64)         { PARAMS.with(|p| p.borrow_mut().self_collision_threshold = v); }
#[wasm_bindgen] pub fn set_self_collision_recompute_iters(v: u16)  { PARAMS.with(|p| p.borrow_mut().self_collision_recompute_iters = v); }
#[wasm_bindgen] pub fn set_use_distance_constraints(v: bool)        { PARAMS.with(|p| p.borrow_mut().use_distance_constraints = v); }
#[wasm_bindgen] pub fn set_pulling_area(v: u32)    { PARAMS.with(|p| p.borrow_mut().pulling_area = v); }
#[wasm_bindgen] pub fn set_damping(v: f64)         { PARAMS.with(|p| p.borrow_mut().damping = v); }

#[wasm_bindgen]
pub fn set_resolution(v: u32) {
    APP_STATE.with(|a| {
        if let Some(state) = a.borrow().as_ref() {
            let mut s = state.borrow_mut();
            let new_cloth = Cloth::new(&s.ctx, v, &s.light);
            let new_sim   = ClothSim::from_grid(v as usize);
            s.cloth = new_cloth;
            s.sim   = new_sim;
        }
    });
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
