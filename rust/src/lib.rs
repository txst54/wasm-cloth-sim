mod gpu;
mod pipeline;
mod triangle;

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use gpu::GpuContext;
use triangle::TriangleRenderer;

#[wasm_bindgen]
pub async fn run(canvas_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document
        .get_element_by_id(canvas_id)
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();

    let ctx = GpuContext::new(canvas).await?;
    let renderer = TriangleRenderer::new(&ctx);

    let (frame, view) = ctx.begin_frame()?;
    renderer.render(&ctx, &view);
    frame.present();

    Ok(())
}
