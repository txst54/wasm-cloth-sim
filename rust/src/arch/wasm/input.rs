//! Shared input helpers for the WASM harness:
//! NDC conversion, picking, ray/plane math, camera key controls, and
//! a generic pick-handler installer driven by the `PickHost` trait.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent, Touch, TouchEvent};

use super::camera::Camera;
use super::gpu::GpuContext;
use crate::sim::shared::Positions;

// ── NDC conversion ────────────────────────────────────────────────────────────

pub fn ndc_from_mouse(e: &MouseEvent, canvas: &HtmlCanvasElement) -> (f32, f32) {
    let w = canvas.offset_width()  as f32;
    let h = canvas.offset_height() as f32;
    ( (e.offset_x() as f32 / w) * 2.0 - 1.0,
     -(e.offset_y() as f32 / h) * 2.0 + 1.0)
}

pub fn ndc_from_touch(t: &Touch, canvas: &HtmlCanvasElement) -> (f32, f32) {
    let rect = canvas.get_bounding_client_rect();
    let ox = t.client_x() as f32 - rect.left() as f32;
    let oy = t.client_y() as f32 - rect.top()  as f32;
    ( (ox / rect.width()  as f32) * 2.0 - 1.0,
     -(oy / rect.height() as f32) * 2.0 + 1.0)
}

// ── Vec3 helpers ──────────────────────────────────────────────────────────────

#[inline] fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline] fn neg3(a: [f32; 3]) -> [f32; 3] { [-a[0], -a[1], -a[2]] }
#[inline] fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
#[inline] fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
#[inline] fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
    if len > 1e-8 { [v[0]/len, v[1]/len, v[2]/len] } else { v }
}

// ── Projection / picking ──────────────────────────────────────────────────────

/// Project a world-space point to NDC using the camera's view-projection.
pub fn project_to_ndc(world: [f32; 3], camera: &Camera) -> (f32, f32) {
    let e = camera.eye;
    let forward = normalize3(neg3(e)); // camera always looks at the origin
    let right   = normalize3(cross3(forward, [0.0, 1.0, 0.0]));
    let up      = cross3(right, forward);

    let d  = sub3(world, e);
    let vx = dot3(d, right);
    let vy = dot3(d, up);
    let vz = dot3(d, neg3(forward));

    let f = 1.0 / (std::f32::consts::FRAC_PI_4 * 0.5).tan();
    ((f / camera.aspect) * vx / (-vz), f * vy / (-vz))
}

/// Cast a ray from the camera through NDC (nx, ny), intersect with the plane
/// parallel to the image plane through `plane_point`.
pub fn ray_plane_intersect(
    nx: f32, ny: f32,
    plane_point: [f32; 3],
    camera: &Camera,
) -> Option<[f32; 3]> {
    let near = unproject(nx, ny, 0.0, &camera.inv_view_proj);
    let far  = unproject(nx, ny, 1.0, &camera.inv_view_proj);
    let dir  = normalize3(sub3(far, near));
    let n    = normalize3(neg3(camera.eye));
    let denom = dot3(n, dir);
    if denom.abs() < 1e-6 { return None; }
    let t = dot3(n, sub3(plane_point, camera.eye)) / denom;
    Some([
        camera.eye[0] + t * dir[0],
        camera.eye[1] + t * dir[1],
        camera.eye[2] + t * dir[2],
    ])
}

fn unproject(nx: f32, ny: f32, nz: f32, inv_vp: &[[f32; 4]; 4]) -> [f32; 3] {
    let clip = [nx, ny, nz, 1.0f32];
    let mut w = [0.0f32; 4];
    for row in 0..4 {
        for col in 0..4 { w[row] += inv_vp[row][col] * clip[col]; }
    }
    let inv_w = 1.0 / w[3];
    [w[0] * inv_w, w[1] * inv_w, w[2] * inv_w]
}

/// Read vertex `i` from a Positions matrix as `[f32; 3]`.
#[inline]
pub fn vertex_at(q: &Positions, i: usize) -> [f32; 3] {
    [q[(i, 0)], q[(i, 1)], q[(i, 2)]]
}

/// Find the vertex closest to NDC `(nx, ny)`. Returns `(index, world_pos)`.
pub fn pick_closest(q: &Positions, camera: &Camera, nx: f32, ny: f32) -> (usize, [f32; 3]) {
    let mut best_idx = 0usize;
    let mut best_d2  = f32::MAX;
    for i in 0..q.nrows() {
        let wp = vertex_at(q, i);
        let (px, py) = project_to_ndc(wp, camera);
        let dx = px - nx; let dy = py - ny;
        let d2 = dx*dx + dy*dy;
        if d2 < best_d2 { best_d2 = d2; best_idx = i; }
    }
    (best_idx, vertex_at(q, best_idx))
}

// ── Keyboard / camera controls ────────────────────────────────────────────────

/// Maps arrow keys / WASD to indices `[←, →, ↑, ↓, A, D, W, S]`.
pub fn arrow_key_index(key: &str) -> Option<usize> {
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

const ROT_SPEED: f32 = 0.02;
const MOV_SPEED: f32 = 0.03;

/// Apply held-key state to the camera (orbit + target translate) and upload
/// the new view-projection if anything changed.
pub fn apply_camera_keys(camera: &mut Camera, keys: &[bool; 8], ctx: &GpuContext) {
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
}

// ── Generic input-handler installation ────────────────────────────────────────

/// Implemented by app-state structs to expose the fields needed by the input
/// system. Sequential read-then-write keeps the borrow checker happy without
/// requiring split borrows through nested fields.
pub trait PickHost {
    fn camera(&self)  -> &Camera;
    fn canvas(&self)  -> &HtmlCanvasElement;
    fn keys_mut(&mut self) -> &mut [bool; 8];

    fn positions(&self) -> &Positions;
    fn clicked_vertex(&self) -> Option<usize>;
    /// Set (or clear) the clicked vertex. Implementors with `dragging_vertices`
    /// should clear it here when a new pick starts.
    fn set_clicked(&mut self, idx: Option<usize>);
    fn set_mouse(&mut self, pos: [f32; 3]);

    /// Optional hook for spacebar / step-once-style gestures. Returning `true`
    /// consumes the event.
    fn on_extra_key(&mut self, _key: &str) -> bool { false }
}

fn begin_pick<H: PickHost>(host: &mut H, nx: f32, ny: f32) {
    let (idx, vw) = pick_closest(host.positions(), host.camera(), nx, ny);
    let target = ray_plane_intersect(nx, ny, vw, host.camera()).unwrap_or(vw);
    host.set_clicked(Some(idx));
    host.set_mouse(target);
}

fn drag_to<H: PickHost>(host: &mut H, nx: f32, ny: f32) {
    let v = match host.clicked_vertex() { Some(v) => v, None => return };
    let cp = vertex_at(host.positions(), v);
    if let Some(world) = ray_plane_intersect(nx, ny, cp, host.camera()) {
        host.set_mouse(world);
    }
}

fn end_pick<H: PickHost>(host: &mut H) {
    host.set_clicked(None);
}

/// Install mouse + touch + keyboard handlers for any `PickHost`.
pub fn install_handlers<H: PickHost + 'static>(
    state: Rc<RefCell<H>>,
    canvas: &HtmlCanvasElement,
    window: &web_sys::Window,
) -> Result<(), JsValue> {
    // mousedown
    let s = state.clone();
    let cb = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut h = s.borrow_mut();
        let (nx, ny) = ndc_from_mouse(&e, h.canvas());
        begin_pick(&mut *h, nx, ny);
    }));
    canvas.add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref())?;
    cb.forget();

    // mousemove
    let s = state.clone();
    let cb = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut h = s.borrow_mut();
        let (nx, ny) = ndc_from_mouse(&e, h.canvas());
        drag_to(&mut *h, nx, ny);
    }));
    canvas.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())?;
    cb.forget();

    // mouseup
    let s = state.clone();
    let cb = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |_: MouseEvent| {
        end_pick(&mut *s.borrow_mut());
    }));
    canvas.add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref())?;
    cb.forget();

    // touchstart
    let s = state.clone();
    let cb = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |e: TouchEvent| {
        e.prevent_default();
        let touch = match e.touches().get(0) { Some(t) => t, None => return };
        let mut h = s.borrow_mut();
        let (nx, ny) = ndc_from_touch(&touch, h.canvas());
        begin_pick(&mut *h, nx, ny);
    }));
    canvas.add_event_listener_with_callback("touchstart", cb.as_ref().unchecked_ref())?;
    cb.forget();

    // touchmove
    let s = state.clone();
    let cb = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |e: TouchEvent| {
        e.prevent_default();
        let touch = match e.touches().get(0) { Some(t) => t, None => return };
        let mut h = s.borrow_mut();
        let (nx, ny) = ndc_from_touch(&touch, h.canvas());
        drag_to(&mut *h, nx, ny);
    }));
    canvas.add_event_listener_with_callback("touchmove", cb.as_ref().unchecked_ref())?;
    cb.forget();

    // touchend
    let s = state.clone();
    let cb = Closure::<dyn FnMut(TouchEvent)>::wrap(Box::new(move |_: TouchEvent| {
        end_pick(&mut *s.borrow_mut());
    }));
    canvas.add_event_listener_with_callback("touchend", cb.as_ref().unchecked_ref())?;
    cb.forget();

    install_keyboard_handlers(state, window)
}

/// Install only the keyboard handlers (camera controls + `on_extra_key`).
pub fn install_keyboard_handlers<H: PickHost + 'static>(
    state: Rc<RefCell<H>>,
    window: &web_sys::Window,
) -> Result<(), JsValue> {
    let s = state.clone();
    let cb = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        let key = e.key();
        if let Some(idx) = arrow_key_index(&key) {
            e.prevent_default();
            s.borrow_mut().keys_mut()[idx] = true;
        } else if s.borrow_mut().on_extra_key(&key) {
            e.prevent_default();
        }
    }));
    window.add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())?;
    cb.forget();

    let s = state.clone();
    let cb = Closure::<dyn FnMut(KeyboardEvent)>::wrap(Box::new(move |e: KeyboardEvent| {
        if let Some(idx) = arrow_key_index(&e.key()) {
            s.borrow_mut().keys_mut()[idx] = false;
        }
    }));
    window.add_event_listener_with_callback("keyup", cb.as_ref().unchecked_ref())?;
    cb.forget();
    Ok(())
}
