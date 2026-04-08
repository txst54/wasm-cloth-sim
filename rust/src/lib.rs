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
use web_sys::{HtmlCanvasElement, MouseEvent};

use cloth::Cloth;
use gpu::GpuContext;
use light::Light;
use params::SimParams;
use sim::ClothSim;

struct AppState {
    ctx:    GpuContext,
    cloth:  Cloth,
    light:  Light,
    sim:    ClothSim,
    params: SimParams,
    canvas: HtmlCanvasElement,
}

#[wasm_bindgen]
pub async fn run(canvas_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let canvas = window
        .document().unwrap()
        .get_element_by_id(canvas_id).unwrap()
        .dyn_into::<HtmlCanvasElement>().unwrap();

    let ctx   = GpuContext::new(canvas.clone()).await?;
    let light = Light::new(&ctx, [2.0, 0.0, 0.5]);
    let cloth = Cloth::new(&ctx, 64, &light);
    let sim   = ClothSim::from_cloth(&cloth);

    let state = Rc::new(RefCell::new(AppState {
        ctx,
        cloth,
        light,
        sim,
        params: SimParams::default(),
        canvas: canvas.clone(),
    }));

    // Convert a MouseEvent's offset coordinates to clip space [-1, 1].
    fn to_clip(event: &MouseEvent, canvas: &HtmlCanvasElement) -> (f32, f32) {
        let w = canvas.offset_width()  as f32;
        let h = canvas.offset_height() as f32;
        let cx =  (event.offset_x() as f32 / w) * 2.0 - 1.0;
        let cy = -(event.offset_y() as f32 / h) * 2.0 + 1.0; // flip Y
        (cx, cy)
    }

    // mousedown — find closest vertex and begin drag
    let state_md = state.clone();
    let mousedown = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut s = state_md.borrow_mut();
        let (cx, cy) = to_clip(&e, &s.canvas);

        let mut best_idx  = 0usize;
        let mut best_dist = f32::MAX;
        for i in 0..s.sim.q.nrows() {
            let dx = s.sim.q[(i, 0)] - cx;
            let dy = s.sim.q[(i, 1)] - cy;
            let d2 = dx * dx + dy * dy;
            if d2 < best_dist {
                best_dist = d2;
                best_idx  = i;
            }
        }
        s.sim.clicked_vertex = Some(best_idx);
        s.sim.mouse_pos = [cx, cy, s.sim.q[(best_idx, 2)]];
    }));
    canvas.add_event_listener_with_callback("mousedown", mousedown.as_ref().unchecked_ref())?;
    mousedown.forget();

    // mousemove — update drag target
    let state_mm = state.clone();
    let mousemove = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |e: MouseEvent| {
        let mut s = state_mm.borrow_mut();
        if s.sim.clicked_vertex.is_some() {
            let (cx, cy) = to_clip(&e, &s.canvas);
            s.sim.mouse_pos[0] = cx;
            s.sim.mouse_pos[1] = cy;
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

    // RAF loop
    let loop_fn: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let loop_fn_inner = loop_fn.clone();
    let state_inner   = state.clone();

    *loop_fn.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mut s = state_inner.borrow_mut();
        let AppState { sim, cloth, ctx, light, params, .. } = &mut *s;

        sim.step(params);
        sim.write_to_cloth(cloth, ctx);

        if let Ok((frame, view)) = ctx.begin_frame() {
            cloth.render(ctx, &view, light);
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
