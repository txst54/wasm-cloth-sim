use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;

fn alert(msg: &str) {
    if let Some(win) = web_sys::window() {
        let _ = win.alert_with_message(msg);
    }
    web_sys::console::error_1(&JsValue::from_str(msg));
}

fn webgpu_supported() -> bool {
    web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w.navigator(), &JsValue::from_str("gpu")).ok())
        .map(|v| !v.is_undefined() && !v.is_null())
        .unwrap_or(false)
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_view: wgpu::TextureView,
}

fn make_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

impl GpuContext {
    pub async fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        let width = canvas.width();
        let height = canvas.height();

        if !webgpu_supported() {
            let msg = "WebGPU is not available in this browser. \
                       Please use a recent version of Chrome, Edge, or Firefox Nightly \
                       on a system with a supported GPU.";
            alert(msg);
            return Err(JsValue::from_str(msg));
        }

        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(desc);

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| {
                let msg = format!("Failed to create WebGPU surface: {e}");
                alert(&msg);
                JsValue::from_str(&msg)
            })?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                let msg = format!(
                    "No compatible GPU adapter found ({e}). \
                     This simulation requires a system with a WebGPU-capable GPU."
                );
                alert(&msg);
                JsValue::from_str(&msg)
            })?;

        let info = adapter.get_info();
        if matches!(info.device_type, wgpu::DeviceType::Cpu) {
            let msg = format!(
                "Only a software (CPU) GPU adapter is available ({} — {}). \
                 Performance will be unusably slow. \
                 Please run this on a system with a discrete or integrated GPU.",
                info.name, info.backend.to_str()
            );
            alert(&msg);
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    ..wgpu::Limits::default()
                },
                label: None,
                ..Default::default()
            })
            .await
            .map_err(|e| {
                let msg = format!(
                    "GPU does not meet the minimum requirements for this simulation \
                     (need max_storage_buffers_per_shader_stage >= 16): {e}"
                );
                alert(&msg);
                JsValue::from_str(&msg)
            })?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.first().copied()
            .ok_or_else(|| JsValue::from_str("No supported surface formats"))?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let depth_view = make_depth_view(&device, width, height);

        Ok(Self { device, queue, surface, config, depth_view })
    }

    pub fn begin_frame(&self) -> Result<(wgpu::SurfaceTexture, wgpu::TextureView), JsValue> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return Err(JsValue::from_str("Failed to acquire surface texture")),
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok((frame, view))
    }
}
