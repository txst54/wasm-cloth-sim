use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{camera::Camera, gpu::GpuContext, light::Light, pipeline::PipelineBuilder};
use crate::sim::crease::CreaseType;
use crate::sim::Positions;

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
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x3,
        ],
    };
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WireframeVertex {
    position: [f32; 3],
    color:    [f32; 3],
}

impl WireframeVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<WireframeVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
        ],
    };
}

const WIREFRAME_SHADER: &str = r#"
struct LightUniform {
    position: vec3<f32>,
}
@group(0) @binding(0) var<uniform> light: LightUniform;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec3<f32>,
}
@group(1) @binding(0) var<uniform> camera: CameraUniform;

struct WireframeIn {
    @location(0) position: vec3<f32>,
    @location(1) color:    vec3<f32>,
}

struct WireframeOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       color:         vec3<f32>,
}

@vertex
fn vs_main(in: WireframeIn) -> WireframeOut {
    var out: WireframeOut;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: WireframeOut) -> @location(0) vec4<f32> {
    let dummy = light.position.x * 0.0;
    return vec4<f32>(in.color + vec3<f32>(dummy, 0.0, 0.0), 1.0);
}
"#;

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
    @location(2)  @interpolate(flat)                  color:          vec3<f32>,
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
    let ambient = 0.45;
    let diffuse       = max(dot(n, light_dir), 0.0);

    let view_dir = normalize(camera.position - in.world_position);
    let half_dir = normalize(light_dir + view_dir);
    let specular = pow(max(dot(n, half_dir), 0.0), 32.0) * 0.5;

    let lit = in.color * (ambient + diffuse * 0.85);
    // return vec4<f32>((n.x + 1.0) / 2.0, (n.y + 1.0) / 2.0, (n.z + 1.0) / 2.0, 1.0);
    return vec4<f32>(lit, 1.0);
}
"#;

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

fn compute_vertex_normals(positions: &[[f32; 3]], n: usize) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; n * n];
    for row in 0..(n - 1) {
        for col in 0..(n - 1) {
            let tl = row * n + col;
            let tr = tl + 1;
            let bl = (row + 1) * n + col;
            let br = bl + 1;
            let fn1 = cross(sub(positions[tr], positions[tl]), sub(positions[br], positions[tl]));
            let fn2 = cross(sub(positions[br], positions[tl]), sub(positions[bl], positions[tl]));
            for &vi in &[tl, tr, br] { normals[vi] = add(normals[vi], fn1); }
            for &vi in &[tl, br, bl] { normals[vi] = add(normals[vi], fn2); }
        }
    }
    normals.into_iter().map(normalize).collect()
}

fn compute_mesh_normals(positions: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; positions.len()];
    for &[i0, i1, i2] in faces {
        let p0 = positions[i0 as usize];
        let p1 = positions[i1 as usize];
        let p2 = positions[i2 as usize];
        let fn_vec = cross(sub(p1, p0), sub(p2, p0));
        normals[i0 as usize] = add(normals[i0 as usize], fn_vec);
        normals[i1 as usize] = add(normals[i1 as usize], fn_vec);
        normals[i2 as usize] = add(normals[i2 as usize], fn_vec);
    }
    normals.into_iter().map(normalize).collect()
}

fn grid_color(row: usize, col: usize) -> [f32; 3] {
    if (row + col) % 2 == 0 { [0.85, 0.85, 0.85] } else { [0.35, 0.35, 0.35] }
}

pub struct Cloth {
    pub resolution: u32,
    pub positions: Vec<[f32; 3]>,
    pub faces: Option<Vec<[u32; 3]>>,
    pub colors: Option<Vec<[f32; 3]>>,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    wireframe_pipeline: wgpu::RenderPipeline,
    wireframe_vertex_buffer: wgpu::Buffer,
    wireframe_vertex_count: u32,
    wireframe_edges: Vec<(u32, u32, [f32; 3])>,
    pub wireframe_enabled: bool,
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

        let gray = [0.2f32, 0.2, 0.2];
        let mut wireframe_edges: Vec<(u32, u32, [f32; 3])> = Vec::new();
        for row in 0..n {
            for col in 0..n {
                let v = (row * n + col) as u32;
                if col < n - 1 {
                    wireframe_edges.push((v, v + 1, gray));
                }
                if row < n - 1 {
                    wireframe_edges.push((v, v + n as u32, gray));
                }
            }
        }
        let wireframe_verts: Vec<WireframeVertex> = wireframe_edges.iter()
            .flat_map(|&(a, b, color)| [
                WireframeVertex { position: positions[a as usize], color },
                WireframeVertex { position: positions[b as usize], color },
            ])
            .collect();
        let wireframe_vertex_count = wireframe_verts.len() as u32;

        let wireframe_vertex_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wireframe VB"),
            contents: bytemuck::cast_slice(&wireframe_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let wireframe_pipeline = PipelineBuilder::new(&ctx.device, ctx.config.format)
            .label("Wireframe Pipeline")
            .shader(WIREFRAME_SHADER)
            .bind_group_layout(&light.bind_group_layout)
            .bind_group_layout(&camera_bgl)
            .vertex_layout(WireframeVertex::LAYOUT)
            .topology(wgpu::PrimitiveTopology::LineList)
            .build();

        Self {
            resolution, positions, faces: None, colors: None,
            pipeline, vertex_buffer, index_buffer, index_count,
            wireframe_pipeline, wireframe_vertex_buffer, wireframe_vertex_count,
            wireframe_edges,
            wireframe_enabled: false,
        }
    }

    pub fn from_mesh(
        ctx: &GpuContext,
        positions: Vec<[f32; 3]>,
        faces: Vec<[u32; 3]>,
        colors: Vec<[f32; 3]>,
        edge_types: HashMap<(u32, u32), CreaseType>,
        light: &Light,
    ) -> Self {
        let num_verts = positions.len();
        let normals = compute_mesh_normals(&positions, &faces);

        let vertices: Vec<Vertex> = positions.iter().enumerate().map(|(i, &position)| {
            Vertex {
                position,
                normal: normals[i],
                color: colors.get(i).copied().unwrap_or([0.75, 0.88, 1.0]),
            }
        }).collect();

        let indices: Vec<u32> = faces.iter().flat_map(|f| f.iter().copied()).collect();
        let index_count = indices.len() as u32;

        let vertex_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh VB"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh IB"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let camera_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh Camera BGL"),
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
            .label("Mesh Pipeline")
            .shader(SHADER)
            .bind_group_layout(&light.bind_group_layout)
            .bind_group_layout(&camera_bgl)
            .vertex_layout(Vertex::LAYOUT)
            .build();

        let mut edge_set = std::collections::HashSet::new();
        for &[i0, i1, i2] in &faces {
            let mut add_edge = |a: u32, b: u32| {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edge_set.insert((lo, hi));
            };
            add_edge(i0, i1);
            add_edge(i1, i2);
            add_edge(i2, i0);
        }

        let wireframe_edges: Vec<(u32, u32, [f32; 3])> = edge_set.iter()
            .map(|&(a, b)| {
                let color = match edge_types.get(&(a, b)) {
                    Some(CreaseType::Mountain) => [0.9, 0.1, 0.1],
                    Some(CreaseType::Valley)   => [0.1, 0.3, 1.0],
                    _ => [0.25, 0.25, 0.25],
                };
                (a, b, color)
            })
            .collect();

        let wireframe_verts: Vec<WireframeVertex> = wireframe_edges.iter()
            .flat_map(|&(a, b, color)| [
                WireframeVertex { position: positions[a as usize], color },
                WireframeVertex { position: positions[b as usize], color },
            ])
            .collect();
        let wireframe_vertex_count = wireframe_verts.len() as u32;

        let wireframe_vertex_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wireframe VB"),
            contents: bytemuck::cast_slice(&wireframe_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let wireframe_pipeline = PipelineBuilder::new(&ctx.device, ctx.config.format)
            .label("Wireframe Pipeline")
            .shader(WIREFRAME_SHADER)
            .bind_group_layout(&light.bind_group_layout)
            .bind_group_layout(&camera_bgl)
            .vertex_layout(WireframeVertex::LAYOUT)
            .topology(wgpu::PrimitiveTopology::LineList)
            .build();

        Self {
            resolution: num_verts as u32,
            positions,
            faces: Some(faces),
            colors: Some(colors),
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            wireframe_pipeline,
            wireframe_vertex_buffer,
            wireframe_vertex_count,
            wireframe_edges,
            wireframe_enabled: false,
        }
    }

    /// Sync positions from simulation and upload to GPU.
    pub fn sync_from_sim(&mut self, sim_positions: &Positions, ctx: &GpuContext) {
        for (i, pos) in self.positions.iter_mut().enumerate() {
            pos[0] = sim_positions[(i, 0)];
            pos[1] = sim_positions[(i, 1)];
            pos[2] = sim_positions[(i, 2)];
        }
        self.upload(ctx);
    }

    pub fn upload(&self, ctx: &GpuContext) {
        let vertices: Vec<Vertex> = if let Some(ref faces) = self.faces {
            let normals = compute_mesh_normals(&self.positions, faces);
            self.positions.iter().enumerate().map(|(i, &position)| {
                Vertex {
                    position,
                    normal: normals[i],
                    color: self.colors.as_ref()
                        .and_then(|c| c.get(i).copied())
                        .unwrap_or([0.75, 0.88, 1.0]),
                }
            }).collect()
        } else {
            let n = self.resolution as usize;
            let normals = compute_vertex_normals(&self.positions, n);
            self.positions.iter().enumerate().map(|(i, &position)| {
                let row = i / n;
                let col = i % n;
                Vertex { position, normal: normals[i], color: grid_color(row, col) }
            }).collect()
        };
        ctx.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));

        if self.wireframe_enabled && !self.wireframe_edges.is_empty() {
            let wf_verts: Vec<WireframeVertex> = self.wireframe_edges.iter()
                .flat_map(|&(a, b, color)| [
                    WireframeVertex { position: self.positions[a as usize], color },
                    WireframeVertex { position: self.positions[b as usize], color },
                ])
                .collect();
            ctx.queue.write_buffer(&self.wireframe_vertex_buffer, 0, bytemuck::cast_slice(&wf_verts));
        }
    }

    pub fn render(&self, ctx: &GpuContext, view: &wgpu::TextureView, light: &Light, camera: &Camera) {
        self.render_impl(ctx, view, light, camera, true);
    }

    pub fn render_over(&self, ctx: &GpuContext, view: &wgpu::TextureView, light: &Light, camera: &Camera) {
        self.render_impl(ctx, view, light, camera, false);
    }

    fn render_impl(&self, ctx: &GpuContext, view: &wgpu::TextureView, light: &Light, camera: &Camera, clear: bool) {
        let color_load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.05, a: 1.0 })
        } else {
            wgpu::LoadOp::Load
        };
        let depth_load = if clear { wgpu::LoadOp::Clear(1.0) } else { wgpu::LoadOp::Load };

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
                    ops: wgpu::Operations { load: color_load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &ctx.depth_view,
                    depth_ops: Some(wgpu::Operations { load: depth_load, store: wgpu::StoreOp::Store }),
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

            if self.wireframe_enabled {
                rpass.set_pipeline(&self.wireframe_pipeline);
                rpass.set_bind_group(0, &light.bind_group, &[]);
                rpass.set_bind_group(1, &camera.bind_group, &[]);
                rpass.set_vertex_buffer(0, self.wireframe_vertex_buffer.slice(..));
                rpass.draw(0..self.wireframe_vertex_count, 0..1);
            }
        }
        ctx.queue.submit(Some(encoder.finish()));
    }
}
