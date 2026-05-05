use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::gpu::GpuContext;

pub const SHADOW_SIZE: u32 = 4096;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LightingUniform {
    key_dir:    [f32; 4],
    key_color:  [f32; 4],
    fill_dir:   [f32; 4],
    fill_color: [f32; 4],
    rim_dir:    [f32; 4],
    rim_color:  [f32; 4],
    ambient:    [f32; 4],
    light_view_proj: [[f32; 4]; 4],
}

fn n3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    [v[0]/l, v[1]/l, v[2]/l]
}

fn ortho_light_view_proj(dir_to_light: [f32; 3], half: f32) -> [[f32; 4]; 4] {
    // Place light camera looking from +dir at origin.
    let dist = half * 3.0;
    let eye  = [dir_to_light[0] * dist, dir_to_light[1] * dist, dir_to_light[2] * dist];
    let f    = n3([-eye[0], -eye[1], -eye[2]]); // forward = toward origin
    // Robust up: avoid colinear with f.
    let up0  = if f[1].abs() > 0.95 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let r    = n3(cross(f, up0));
    let u    = cross(r, f);

    // View matrix (column-major output rows).
    let view = [
        [ r[0],  r[1],  r[2], -dot(r, eye)],
        [ u[0],  u[1],  u[2], -dot(u, eye)],
        [-f[0], -f[1], -f[2],  dot(f, eye)],
        [ 0.0,   0.0,   0.0,   1.0        ],
    ];

    // Orthographic projection mapping x,y in [-half,half] -> [-1,1]; z in [near,far] -> [0,1].
    let near = 0.1f32;
    let far  = dist * 2.5;
    let proj = [
        [1.0/half, 0.0,       0.0,                0.0                       ],
        [0.0,      1.0/half,  0.0,                0.0                       ],
        [0.0,      0.0,       1.0/(near - far),   near/(near - far)         ],
        [0.0,      0.0,       0.0,                1.0                       ],
    ];

    let m = mat_mul_row(proj, view);
    transpose4(m)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn mat_mul_row(a: [[f32;4];4], b: [[f32;4];4]) -> [[f32;4];4] {
    let mut c = [[0.0f32;4];4];
    for r in 0..4 { for col in 0..4 { for k in 0..4 { c[r][col] += a[r][k]*b[k][col]; } } }
    c
}
fn transpose4(m: [[f32;4];4]) -> [[f32;4];4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

pub struct Lighting {
    buffer: wgpu::Buffer,
    pub bind_group_layout:        wgpu::BindGroupLayout,
    pub bind_group:               wgpu::BindGroup,
    pub shadow_bind_group_layout: wgpu::BindGroupLayout,
    pub shadow_bind_group:        wgpu::BindGroup,
    pub shadow_view:              wgpu::TextureView,
}

impl Lighting {
    pub fn new(ctx: &GpuContext) -> Self {
        // Default 3-point rig.
        let key_dir   = n3([1.0, 1.5, 0.8]);
        let fill_dir  = n3([-0.7, 0.4, 0.5]);
        let rim_dir   = n3([0.0, 0.3, -1.0]);

        let half = 3.0f32;
        let lvp = ortho_light_view_proj(key_dir, half);

        let uniform = LightingUniform {
            key_dir:    [key_dir[0],  key_dir[1],  key_dir[2],  0.0],
            key_color:  [1.00, 0.97, 0.92, 0.0],
            fill_dir:   [fill_dir[0], fill_dir[1], fill_dir[2], 0.0],
            fill_color: [0.22, 0.27, 0.35, 0.0],
            rim_dir:    [rim_dir[0],  rim_dir[1],  rim_dir[2],  0.0],
            rim_color:  [0.60, 0.50, 0.38, 0.0],
            ambient:    [0.18, 0.20, 0.24, 0.0],
            light_view_proj: lvp,
        };

        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lighting Buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Shadow depth texture.
        let shadow_tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map"),
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let shadow_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Main bind group: uniform + shadow texture + comparison sampler.
        let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
            ],
        });

        // Shadow-pass bind group: just the uniform (so the shadow texture can be used as attachment).
        let shadow_bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lighting Shadow BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let shadow_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lighting Shadow BG"),
            layout: &shadow_bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });

        Self {
            buffer, bind_group_layout, bind_group,
            shadow_bind_group_layout, shadow_bind_group, shadow_view,
        }
    }

    /// Clear the shadow map to far depth (1.0). Call once per frame before rendering shadow casters.
    pub fn clear_shadow(&self, ctx: &GpuContext) {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Shadow Clear Encoder"),
        });
        {
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    #[allow(dead_code)]
    pub fn upload(&self, _ctx: &GpuContext) {}
    #[allow(dead_code)]
    pub fn buffer(&self) -> &wgpu::Buffer { &self.buffer }
}
