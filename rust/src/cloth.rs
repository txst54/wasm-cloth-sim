use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{camera::Camera, gpu::GpuContext, light::Light, pipeline::PipelineBuilder};

// ── Vertex layout ────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal:   [f32; 3],
    color:    [f32; 3],
}

impl Vertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,  // position
            1 => Float32x3,  // normal
            2 => Float32x3,  // color
        ],
    };
}

// ── Shader ───────────────────────────────────────────────────────────────────

const SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec3<f32>,
}

struct VertexOutput {
    @builtin(position)              clip_position:  vec4<f32>,
    @location(0)                    world_position: vec3<f32>,
    @location(1)                    normal:         vec3<f32>,
    @location(2) @interpolate(flat) color:          vec3<f32>,
}

struct LightUniform {
    position: vec3<f32>,
}
@group(0) @binding(0) var<uniform> light: LightUniform;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec3<f32>,
}
@group(1) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position  = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_position = in.position;
    out.normal         = in.normal;
    out.color          = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let light_dir = normalize(light.position - in.world_position);
    let ambient = 0.15;
    let diffuse       = max(dot(n, light_dir), 0.0);

    // Blinn-Phong specular — use actual camera position
    let view_dir = normalize(camera.position - in.world_position);
    let half_dir = normalize(light_dir + view_dir);
    let specular = pow(max(dot(n, half_dir), 0.0), 32.0) * 0.5;

    let lit = in.color * (ambient + diffuse * 0.85) + vec3<f32>(specular);
    // return vec4<f32>(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    return vec4<f32>((n.x + 1.0) / 2.0, (n.y + 1.0) / 2.0, (n.z + 1.0) / 2.0, 1.0);
}
"#;

// ── Math helpers ─────────────────────────────────────────────────────────────

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if len > 1e-8 { [v[0]/len, v[1]/len, v[2]/len] } else { [0.0, 0.0, 1.0] }
}

/// Computes vertex normals as the area-weighted average of adjacent face normals.
/// Each face normal is the (unnormalized) cross product of its edges, whose magnitude
/// equals twice the triangle area, giving natural area weighting when accumulated.
fn compute_vertex_normals(positions: &[[f32; 3]], n: usize) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; n * n];
    for row in 0..(n - 1) {
        for col in 0..(n - 1) {
            let tl = row * n + col;
            let tr = tl + 1;
            let bl = (row + 1) * n + col;
            let br = bl + 1;
            // Triangle 1: tl, tr, br
            let fn1 = cross(sub(positions[tr], positions[tl]), sub(positions[br], positions[tl]));
            // Triangle 2: tl, br, bl
            let fn2 = cross(sub(positions[br], positions[tl]), sub(positions[bl], positions[tl]));
            for &vi in &[tl, tr, br] { normals[vi] = add(normals[vi], fn1); }
            for &vi in &[tl, br, bl] { normals[vi] = add(normals[vi], fn2); }
        }
    }
    normals.into_iter().map(normalize).collect()
}

// ── Grid color ────────────────────────────────────────────────────────────────

fn grid_color(row: usize, col: usize) -> [f32; 3] {
    if (row + col) % 2 == 0 { [0.75, 0.88, 1.00] } else { [0.75, 0.88, 1.00] }
}

// ── Cloth ────────────────────────────────────────────────────────────────────

/// NxN cloth mesh. Mutate `positions` then call `upload()` before rendering.
pub struct Cloth {
    pub resolution: u32,
    /// Flat array of NxN vertex positions in row-major order.
    /// Index vertex (row, col) as `positions[row * resolution + col]`.
    pub positions: Vec<[f32; 3]>,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl Cloth {
    pub fn new(ctx: &GpuContext, resolution: u32, light: &Light) -> Self {
        let n = resolution as usize;
        assert!(n >= 2, "resolution must be at least 2");

        let mut positions = Vec::with_capacity(n * n);
        let mut vertices = Vec::with_capacity(n * n);

        for row in 0..n {
            for col in 0..n {
                let x = (col as f32 / (n - 1) as f32) * 2.0 - 1.0;
                let y = (row as f32 / (n - 1) as f32) * 2.0 - 1.0;
                let pos = [x * 0.9, y * 0.9, 0.0f32];
                positions.push(pos);
                vertices.push(Vertex {
                    position: pos,
                    normal: [0.0, 0.0, 1.0],
                    color: grid_color(row, col),
                });
            }
        }

        let mut indices: Vec<u32> = Vec::with_capacity((n - 1) * (n - 1) * 6);
        for row in 0..(n - 1) {
            for col in 0..(n - 1) {
                let tl = (row * n + col) as u32;
                let tr = tl + 1;
                let bl = ((row + 1) * n + col) as u32;
                let br = bl + 1;
                indices.extend_from_slice(&[tl, tr, br, tl, br, bl]);
            }
        }

        let index_count = indices.len() as u32;

        let vertex_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloth VB"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloth IB"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let camera_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloth Camera BGL placeholder"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline = PipelineBuilder::new(&ctx.device, ctx.config.format)
            .label("Cloth Pipeline")
            .shader(SHADER)
            .bind_group_layout(&light.bind_group_layout)
            .bind_group_layout(&camera_bgl)
            .vertex_layout(Vertex::LAYOUT)
            .build();

        Self { resolution, positions, pipeline, vertex_buffer, index_buffer, index_count }
    }

    /// Re-upload positions (and recomputed normals) to the GPU.
    /// Call after any simulation step.
    pub fn upload(&self, ctx: &GpuContext) {
        let n = self.resolution as usize;
        let normals = compute_vertex_normals(&self.positions, n);
        let vertices: Vec<Vertex> = self.positions.iter().enumerate().map(|(i, &position)| {
            let row = i / n;
            let col = i % n;
            Vertex { position, normal: normals[i], color: grid_color(row, col) }
        }).collect();
        ctx.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
    }

    pub fn render(&self, ctx: &GpuContext, view: &wgpu::TextureView, light: &Light, camera: &Camera) {
        let mut encoder = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Cloth Encoder") },
        );
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Cloth Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.05, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &ctx.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &light.bind_group, &[]);
            rpass.set_bind_group(1, &camera.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }
}
