//! On-screen FPS counter shared by the WASM render loops.

use web_sys::Element;

pub struct FpsOverlay {
    div:       Element,
    last_time: f64,
    fps:       f64,
}

impl FpsOverlay {
    pub fn new(window: &web_sys::Window) -> Self {
        let doc = window.document().unwrap();
        let div = doc.create_element("div").unwrap();
        div.set_inner_html("FPS: 0");
        div.set_attribute(
            "style",
            "position:fixed; top:10px; left:10px; color:white; font-family:monospace; z-index:1000;",
        ).unwrap();
        doc.body().unwrap().append_child(&div).unwrap();
        Self { div, last_time: 0.0, fps: 0.0 }
    }

    /// Sample the current time, update the running FPS estimate, and refresh
    /// the on-page display. Call once per frame.
    pub fn tick(&mut self) {
        let now = web_sys::window().unwrap().performance().unwrap().now();
        let dt  = now - self.last_time;
        self.last_time = now;
        if dt > 0.0 { self.fps = 1000.0 / dt; }
        self.div.set_inner_html(&format!("FPS: {:.1}", self.fps));
    }
}
