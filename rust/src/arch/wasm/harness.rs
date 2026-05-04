//! WASM entry points and render loops.
//!
//! Each `run_*` function wires up a canvas, a sim, and a render loop. Shared
//! input handling lives in [`super::input`], scene constants and procedural
//! mesh builders in [`super::scene`], and the FPS overlay in [`super::fps`].

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use nalgebra as na;

use super::camera::Camera;
use super::cloth::Cloth;
use super::fps::FpsOverlay;
use super::gpu::GpuContext;
use super::input::{
    apply_camera_keys, install_handlers, install_keyboard_handlers, PickHost,
};
use crate::sim::shared::Positions;
use super::light::Light;
use super::platform::init_platform;
use super::scene::{self, octa_sphere_mesh};
use crate::params::SimParams;
use crate::platform_context::PlatformContext;
use crate::rigid_body::{RigidBodyInstance, RigidBodyTemplate};
use crate::sim::{
    ClothSim, CreasePattern, FoldDirection, FoldSpec, PaperSim,
    ParticleClothSim, ParticlePaperSim, RigidSimCore, RigidSimParams,
    SdfObstacle,
};

// ── Thread-locals ─────────────────────────────────────────────────────────────

thread_local! {
    static PARAMS:                   Rc<RefCell<SimParams>> = Rc::new(RefCell::new(SimParams::default()));
    static APP_STATE:                RefCell<Option<Rc<RefCell<AppState>>>>             = RefCell::new(None);
    static PAPER_APP_STATE:          RefCell<Option<Rc<RefCell<PaperAppState>>>>        = RefCell::new(None);
    static RIGID_APP_STATE:          RefCell<Option<Rc<RefCell<RigidAppState>>>>        = RefCell::new(None);
    static COMBINED_APP_STATE:       RefCell<Option<Rc<RefCell<CombinedAppState>>>>     = RefCell::new(None);
    static PARTICLE_APP_STATE:       RefCell<Option<Rc<RefCell<ParticleAppState>>>>     = RefCell::new(None);
    static PARTICLE_PAPER_APP_STATE: RefCell<Option<Rc<RefCell<ParticlePaperAppState>>>> = RefCell::new(None);
}

// ── Common scene-construction helpers ─────────────────────────────────────────

fn get_canvas(canvas_id: &str) -> Result<(web_sys::Window, HtmlCanvasElement), JsValue> {
    let window = web_sys::window().unwrap();
    let canvas = window
        .document().unwrap()
        .get_element_by_id(canvas_id).unwrap()
        .dyn_into::<HtmlCanvasElement>()?;
    Ok((window, canvas))
}

/// `(ctx, light, camera)` — every entry point starts the same way.
async fn build_basics(canvas: HtmlCanvasElement) -> Result<(GpuContext, Light, Camera), JsValue> {
    let ctx    = GpuContext::new(canvas).await?;
    let light  = scene::make_light(&ctx);
    let camera = Camera::new(&ctx);
    Ok((ctx, light, camera))
}

/// Schedule `body` as a `requestAnimationFrame` loop. `body` is invoked once
/// per frame; this helper handles the self-rescheduling boilerplate.
fn run_raf<F: FnMut() + 'static>(window: &web_sys::Window, mut body: F) -> Result<(), JsValue> {
    let loop_fn: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let loop_fn_inner = loop_fn.clone();
    *loop_fn.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        body();
        web_sys::window().unwrap().request_animation_frame(
            loop_fn_inner.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
        ).unwrap();
    }) as Box<dyn FnMut()>));
    window.request_animation_frame(
        loop_fn.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
    )?;
    Ok(())
}

// ── App states ────────────────────────────────────────────────────────────────

struct AppState {
    ctx:    GpuContext,
    cloth:  Cloth,
    light:  Light,
    camera: Camera,
    sim:    ClothSim,
    params: Rc<RefCell<SimParams>>,
    canvas: HtmlCanvasElement,
    /// Held key state: `[←, →, ↑, ↓, A, D, W, S]`.
    keys:   [bool; 8],
}

struct PaperAppState {
    ctx:        GpuContext,
    cloth:      Cloth,
    light:      Light,
    camera:     Camera,
    sim:        PaperSim,
    params:     Rc<RefCell<SimParams>>,
    canvas:     HtmlCanvasElement,
    keys:       [bool; 8],
    cp_data:    Option<String>,
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

struct ParticleAppState {
    ctx:           GpuContext,
    cloth:         Cloth,
    light:         Light,
    camera:        Camera,
    sim:           ParticleClothSim,
    #[cfg(feature = "gpu")]
    gpu_sim:       crate::sim::ParticleClothSimGpu,
    sphere_cloth:  Option<Cloth>,
    ground_cloth:  Option<Cloth>,
    params:        Rc<RefCell<SimParams>>,
    canvas:        HtmlCanvasElement,
    keys:          [bool; 8],
    resolution:    usize,
    sphere_center: [f32; 3],
    sphere_radius: f32,
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
    paused:       bool,
    step_once:    bool,
}

struct ParticlePaperAppState {
    ctx:        GpuContext,
    cloth:      Cloth,
    light:      Light,
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

// ── PickHost impls ────────────────────────────────────────────────────────────

/// Boilerplate for `PickHost` impls that delegate to a sim field.
/// `$path` names the sim path (e.g. `sim` or `sim.core`); `clears_drag` is
/// `true` for sims that own a `dragging_vertices` field to reset on pick.
macro_rules! impl_pickhost {
    ($state:ty, $($path:ident).+, clears_drag = $drag:tt) => {
        impl PickHost for $state {
            fn camera(&self) -> &Camera             { &self.camera }
            fn canvas(&self) -> &HtmlCanvasElement   { &self.canvas }
            fn keys_mut(&mut self) -> &mut [bool; 8] { &mut self.keys }
            fn positions(&self) -> &Positions        { &self.$($path).+.q }
            fn clicked_vertex(&self) -> Option<usize> { self.$($path).+.clicked_vertex }
            fn set_clicked(&mut self, idx: Option<usize>) {
                self.$($path).+.clicked_vertex = idx;
                impl_pickhost!(@drag self, $($path).+, $drag);
            }
            fn set_mouse(&mut self, pos: [f32; 3]) { self.$($path).+.mouse_pos = pos; }
        }
    };
    (@drag $self:ident, $($path:ident).+, true)  => {
        if $self.$($path).+.clicked_vertex.is_some() {
            $self.$($path).+.dragging_vertices = None;
        }
    };
    (@drag $self:ident, $($path:ident).+, false) => {};
}

impl_pickhost!(AppState,              sim,      clears_drag = true);
impl_pickhost!(PaperAppState,         sim,      clears_drag = true);
impl_pickhost!(ParticleAppState,      sim,      clears_drag = false);
impl_pickhost!(ParticlePaperAppState, sim.core, clears_drag = false);

// CombinedAppState gets a hand-rolled impl so spacebar / N drive paused/step.
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

// Rigid demo: keyboard only (camera controls). Picking is unreachable.
impl PickHost for RigidAppState {
    fn camera(&self) -> &Camera               { &self.camera }
    fn canvas(&self) -> &HtmlCanvasElement     { &self.canvas }
    fn keys_mut(&mut self) -> &mut [bool; 8]   { &mut self.keys }
    fn positions(&self) -> &Positions          { unreachable!() }
    fn clicked_vertex(&self) -> Option<usize>  { None }
    fn set_clicked(&mut self, _idx: Option<usize>) {}
    fn set_mouse(&mut self, _pos: [f32; 3])    {}
}

// ── Cloth-only sim ────────────────────────────────────────────────────────────

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
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }
        overlay.tick();
    })
}

// ── Combined cloth + rigid cube ───────────────────────────────────────────────

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
            cloth.render(ctx, &view, light, camera);
            cube_cloth.render_over(ctx, &view, light, camera);
            frame.present();
        }
        overlay.tick();
    })
}

// ── Paper sim (grid + central fold) ───────────────────────────────────────────

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
    let cloth = Cloth::new(&ctx, resolution as u32, &light);
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
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }
    })
}

// ── Rigid-body demo ───────────────────────────────────────────────────────────

#[wasm_bindgen]
pub async fn run_rigid(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let (window, canvas) = get_canvas(canvas_id)?;
    let (ctx, light, camera) = build_basics(canvas.clone()).await?;

    let template = RigidBodyTemplate::new(&scene::CUBE_VERTS, &scene::CUBE_FACES);
    let body = RigidBodyInstance::new(
        template,
        na::Vector3::zeros(),
        na::Vector3::zeros(),
        na::Vector3::zeros(),
        na::Vector3::new(1.0, 2.0, 0.5),
        1000.0,
    );
    let sim    = RigidSimCore::new(vec![body]);
    let params = RigidSimParams { gravity_enabled: false, ..Default::default() };
    let cloth  = scene::cube_cloth(&ctx, &light);

    let state = Rc::new(RefCell::new(RigidAppState {
        ctx, cloth, light, camera, sim, params,
        canvas: canvas.clone(), keys: [false; 8],
    }));
    RIGID_APP_STATE.with(|a| *a.borrow_mut() = Some(state.clone()));

    install_keyboard_handlers(state.clone(), &window)?;

    let state_inner = state.clone();
    run_raf(&window, move || {
        let mut s = state_inner.borrow_mut();
        let RigidAppState { sim, cloth, ctx, light, camera, params, keys, .. } = &mut *s;

        apply_camera_keys(camera, keys, ctx);
        sim.step(params);

        let world = sim.bodies[0].world_vertices();
        for (i, pos) in cloth.positions.iter_mut().enumerate() {
            *pos = [world[i][0], world[i][1], world[i][2]];
        }
        cloth.upload(ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }
    })
}

// ── Particle cloth + SDF obstacles ────────────────────────────────────────────

const PARTICLE_RESOLUTION: usize = 120;
const SPHERE_CENTER: [f32; 3] = [0.0, 0.0, 0.0];
const SPHERE_RADIUS: f32      = 0.4;

fn particle_obstacles(sim: &mut ParticleClothSim, center: [f32; 3], radius: f32) {
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
                gpu_sim.step(&params.borrow());
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            sim.step(&params.borrow());
        }
        cloth.sync_from_sim(&sim.q, ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            cloth.render(ctx, &view, light, camera);
            if let Some(gc) = ground_cloth { gc.render_over(ctx, &view, light, camera); }
            if let Some(sc) = sphere_cloth { sc.render_over(ctx, &view, light, camera); }
            frame.present();
        }
        overlay.tick();
    })
}

// ── Particle paper sim ────────────────────────────────────────────────────────

#[wasm_bindgen]
pub async fn run_particle_paper(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    init_platform();
    let (window, canvas) = get_canvas(canvas_id)?;
    let resolution = PARAMS.with(|p| p.borrow().resolution as usize);
    let (ctx, light, camera) = build_basics(canvas.clone()).await?;
    let cloth = Cloth::new(&ctx, resolution as u32, &light);
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
    let cloth = Cloth::from_mesh(&ctx, positions, faces, colors, edge_colors, &light);

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
                gpu_sim.step(&params.borrow());
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            sim.step(&params.borrow());
        }
        cloth.sync_from_sim(&sim.core.q, ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            cloth.render(ctx, &view, light, camera);
            frame.present();
        }
        overlay.tick();
    })
}

// ── Param setters (called from JS panel) ──────────────────────────────────────

macro_rules! param_setter {
    ($name:ident, $field:ident, $ty:ty) => {
        #[wasm_bindgen] pub fn $name(v: $ty) { PARAMS.with(|p| p.borrow_mut().$field = v); }
    };
    ($name:ident, $field:ident, $ty:ty, |$v:ident| $expr:expr) => {
        #[wasm_bindgen] pub fn $name($v: $ty) { PARAMS.with(|p| p.borrow_mut().$field = $expr); }
    };
}

param_setter!(set_time_step,                       time_step,                       f64);
param_setter!(set_constraint_iters,                constraint_iters,                u32);
param_setter!(set_num_substeps,                    num_substeps,                    u32, |v| v.max(1));
param_setter!(set_stretch_compliance,              stretch_compliance,              f64);
param_setter!(set_bend_compliance,                 bend_compliance,                 f64);
param_setter!(set_gravity_enabled,                 gravity_enabled,                 bool);
param_setter!(set_gravity_g,                       gravity_g,                       f64);
param_setter!(set_pin_enabled,                     pin_enabled,                     bool);
param_setter!(set_pin_weight,                      pin_weight,                      f64);
param_setter!(set_stretch_enabled,                 stretch_enabled,                 bool);
param_setter!(set_stretch_weight,                  stretch_weight,                  f64);
param_setter!(set_bending_enabled,                 bending_enabled,                 bool);
param_setter!(set_bending_weight,                  bending_weight,                  f64);
param_setter!(set_pulling_enabled,                 pulling_enabled,                 bool);
param_setter!(set_pulling_weight,                  pulling_weight,                  f64);
param_setter!(set_self_collision_enabled,          self_collision_enabled,          bool);
param_setter!(set_self_collision_threshold,        self_collision_threshold,        f64);
param_setter!(set_self_collision_recompute_iters,  self_collision_recompute_iters,  u16);
param_setter!(set_use_distance_constraints,        use_distance_constraints,        bool);
param_setter!(set_pulling_area,                    pulling_area,                    u32);
param_setter!(set_damping,                         damping,                         f64);
param_setter!(set_friction_enabled,                friction_enabled,                bool);
param_setter!(set_friction_mu,                     friction_mu,                     f64);
param_setter!(set_cloth_friction_enabled,          cloth_friction_enabled,          bool);
param_setter!(set_cloth_friction_d,                cloth_friction_d,                f64);

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
                s.cloth = Cloth::from_mesh(&s.ctx, positions, faces, colors, edge_colors, &s.light);
                s.sim = sim;
                s.resolution = resolution;
            }
        } else {
            let mut sim = PaperSim::from_grid(resolution);
            sim.set_fold_map(central_vertical_fold(resolution));
            s.cloth = Cloth::new(&s.ctx, resolution as u32, &s.light);
            s.sim = sim;
            s.resolution = resolution;
        }
    });
}

// ── Particle-cloth setters ────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn set_particle_resolution(v: u32) {
    PARTICLE_APP_STATE.with(|a| if let Some(state) = a.borrow().as_ref() {
        let mut s = state.borrow_mut();
        let n = v.max(2) as usize;
        let new_cloth = Cloth::new(&s.ctx, n as u32, &s.light);
        let mut new_sim = ParticleClothSim::from_grid(n, &[]);
        particle_obstacles(&mut new_sim, s.sphere_center, s.sphere_radius);
        #[cfg(feature = "gpu")]
        {
            let mut g = crate::sim::ParticleClothSimGpu::from_cpu(
                s.ctx.device.clone(), s.ctx.queue.clone(), &new_sim,
            );
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
                s.cloth = Cloth::from_mesh(&s.ctx, positions, faces, colors, edge_colors, &s.light);
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
            s.cloth = Cloth::new(&s.ctx, resolution as u32, &s.light);
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

