use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{camera::Camera, gpu::GpuContext, light::Lighting, pipeline::PipelineBuilder};
use crate::sim::crease::CreaseType;
use crate::sim::Positions;

#[derive(Copy, Clone, Debug)]
pub enum Material {
    Cloth = 0,
    Paper = 1,
    Rigid = 2,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaterialUniform {
    kind: u32,
    _p0:  u32,
    _p1:  u32,
    _p2:  u32,
}

// Positions live in their own vertex buffer with vec4 stride so it has the
// same memory layout as the GPU sim's `q_buf` (array<vec4<f32>>). This lets
// the GPU sim path copy_buffer_to_buffer directly into `position_buffer`
// without a CPU round-trip — eliminating the readback-induced stutter.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PosVertex {
    position: [f32; 4],
}

impl PosVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
        ],
    };
}

// Normals + colors share a buffer because both are CPU-driven and updated at
// the same cadence. Splitting them further would not buy anything.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AttrVertex {
    normal: [f32; 3],
    color:  [f32; 3],
}

impl AttrVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<AttrVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            1 => Float32x3,
            2 => Float32x3,
        ],
    };
}

// One entry per line-list vertex. `vidx` indexes into the GPU position buffer
// (`array<vec4<f32>>`), which the wireframe vertex shader fetches at draw time
// — that way the wireframe always reflects live GPU positions without any CPU
// readback. `color` is per line vertex so each edge can carry its own color
// (mountain/valley/regular) even though edges share mesh vertices.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WireframeVertex {
    color: [f32; 3],
    vidx:  u32,
}

impl WireframeVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<WireframeVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Uint32,
        ],
    };
}

const WIREFRAME_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec3<f32>,
}
@group(1) @binding(0) var<uniform> camera: CameraUniform;

struct WireframeIn {
    @location(0) color: vec3<f32>,
    @location(1) vidx:  u32,
}

struct WireframeOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       color:         vec3<f32>,
}

@vertex
fn vs_main(in: WireframeIn) -> WireframeOut {
    let p = positions[in.vidx].xyz;
    var out: WireframeOut;
    out.clip_position = camera.view_proj * vec4<f32>(p, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: WireframeOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
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
    @location(2) @interpolate(flat) color:          vec3<f32>,
}

struct Lights {
    key_dir:    vec4<f32>,
    key_color:  vec4<f32>,
    fill_dir:   vec4<f32>,
    fill_color: vec4<f32>,
    rim_dir:    vec4<f32>,
    rim_color:  vec4<f32>,
    ambient:    vec4<f32>,
    light_view_proj: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> lights: Lights;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_samp: sampler_comparison;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position:  vec3<f32>,
}
@group(1) @binding(0) var<uniform> camera: CameraUniform;

struct Material { kind: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(2) @binding(0) var<uniform> material: Material;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position  = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_position = in.position;
    out.normal         = in.normal;
    out.color          = in.color;
    return out;
}

fn shadow_factor(world_pos: vec3<f32>, n_dot_l: f32) -> f32 {
    let lp = lights.light_view_proj * vec4<f32>(world_pos, 1.0);
    let p  = lp.xyz / lp.w;
    let uv_raw = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    let uv = clamp(uv_raw, vec2<f32>(0.0), vec2<f32>(1.0));
    let bias = max(0.0025 * (1.0 - n_dot_l), 0.0006);
    let ref_z = clamp(p.z - bias, 0.0, 1.0);

    var acc = 0.0;
    let ts = 1.0 / 4096.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let o = vec2<f32>(f32(dx), f32(dy)) * ts;
            acc = acc + textureSampleCompareLevel(shadow_map, shadow_samp, uv + o, ref_z);
        }
    }
    let pcf = acc / 9.0;

    // Outside the shadow frustum (or behind the far plane): no shadowing.
    let in_bounds = f32(uv_raw.x >= 0.0 && uv_raw.x <= 1.0 &&
                        uv_raw.y >= 0.0 && uv_raw.y <= 1.0 &&
                        p.z >= 0.0 && p.z <= 1.0);
    return mix(1.0, pcf, in_bounds);
}

fn brdf(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, lc: vec3<f32>, base: vec3<f32>, kind: u32) -> vec3<f32> {
    let nl = max(dot(n, l), 0.0);
    var diff = base * lc * nl;
    var spec = vec3<f32>(0.0);
    // if (kind == 1u) {
    //     // Paper: medium specular (matches prior behaviour).
    //     let h = normalize(l + v);
    //     let s = pow(max(dot(n, h), 0.0), 32.0) * 0.5;
    //     spec = lc * s;
    // } else
    if (kind == 2u) {
        // Rigid / marble: tight, strong specular.
        let h = normalize(l + v);
        let s = pow(max(dot(n, h), 0.0), 128.0) * 1.4;
        spec = lc * s;
    }
    return diff + spec;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let v = normalize(camera.position - in.world_position);
    let n_geo = normalize(in.normal);
    var n = n_geo;
    if (dot(n, v) < 0.0) { n = -n; }
    let base = in.color;
    let kind = material.kind;

    // Shadow bias must be camera-independent: use |n_geo · l|, not the
    // view-flipped normal, otherwise the bias (and thus the shadow edge)
    // shifts as the camera orbits.
    let nl_key_geo = abs(dot(n_geo, lights.key_dir.xyz));
    let sh = shadow_factor(in.world_position, nl_key_geo);

    var color = lights.ambient.rgb * base;
    color = color + brdf(n, v, lights.key_dir.xyz,  lights.key_color.rgb,  base, kind) * sh;
    color = color + brdf(n, v, lights.fill_dir.xyz, lights.fill_color.rgb, base, kind);

    // Rim: silhouette-only soft highlight from behind.
    let rim_n    = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    let rim_face = max(dot(n, lights.rim_dir.xyz), 0.0);
    color = color + lights.rim_color.rgb * rim_n * rim_face * base;

    return vec4<f32>(color, 1.0);
}
"#;

const SHADOW_SHADER: &str = r#"
struct Lights {
    key_dir:    vec4<f32>,
    key_color:  vec4<f32>,
    fill_dir:   vec4<f32>,
    fill_color: vec4<f32>,
    rim_dir:    vec4<f32>,
    rim_color:  vec4<f32>,
    ambient:    vec4<f32>,
    light_view_proj: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> lights: Lights;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return lights.light_view_proj * vec4<f32>(in.position, 1.0);
}
"#;

// GPU normals: per-face area-weighted scatter using i32 atomics, then
// per-vertex normalize. Writes the normal slots of `attr_buffer` in place,
// leaving the color slots untouched. Matches `compute_vertex_normals`
// (area-weighted) used by the grid path. The mesh path's CPU code uses
// angle-weighted normals; for animated meshes we accept the approximation
// — area-weighted is fast and visually close.
const NORMALS_SHADER: &str = r#"
struct Params {
    n_faces: u32,
    n_verts: u32,
    scale:   f32,
    _pad:    u32,
}

@group(0) @binding(0) var<storage, read>       positions : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       indices   : array<u32>;
@group(0) @binding(2) var<storage, read_write> n_acc     : array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> attrs     : array<f32>;
@group(0) @binding(4) var<uniform>             params    : Params;

@compute @workgroup_size(64)
fn cs_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_verts * 3u) { return; }
    atomicStore(&n_acc[i], 0);
}

@compute @workgroup_size(64)
fn cs_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fi = gid.x;
    if (fi >= params.n_faces) { return; }
    let i0 = indices[fi*3u + 0u];
    let i1 = indices[fi*3u + 1u];
    let i2 = indices[fi*3u + 2u];
    let p0 = positions[i0].xyz;
    let p1 = positions[i1].xyz;
    let p2 = positions[i2].xyz;
    let face_n = cross(p1 - p0, p2 - p0);
    let nx = i32(face_n.x * params.scale);
    let ny = i32(face_n.y * params.scale);
    let nz = i32(face_n.z * params.scale);
    atomicAdd(&n_acc[i0*3u + 0u], nx);
    atomicAdd(&n_acc[i0*3u + 1u], ny);
    atomicAdd(&n_acc[i0*3u + 2u], nz);
    atomicAdd(&n_acc[i1*3u + 0u], nx);
    atomicAdd(&n_acc[i1*3u + 1u], ny);
    atomicAdd(&n_acc[i1*3u + 2u], nz);
    atomicAdd(&n_acc[i2*3u + 0u], nx);
    atomicAdd(&n_acc[i2*3u + 1u], ny);
    atomicAdd(&n_acc[i2*3u + 2u], nz);
}

@compute @workgroup_size(64)
fn cs_finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vi = gid.x;
    if (vi >= params.n_verts) { return; }
    let nx = f32(atomicLoad(&n_acc[vi*3u + 0u])) / params.scale;
    let ny = f32(atomicLoad(&n_acc[vi*3u + 1u])) / params.scale;
    let nz = f32(atomicLoad(&n_acc[vi*3u + 2u])) / params.scale;
    let len = sqrt(nx*nx + ny*ny + nz*nz);
    var nrm = vec3<f32>(0.0, 0.0, 1.0);
    if (len > 1e-8) { nrm = vec3<f32>(nx, ny, nz) / len; }
    // attr stride 6 floats: [0..3) normal, [3..6) color. Write normal only.
    attrs[vi*6u + 0u] = nrm.x;
    attrs[vi*6u + 1u] = nrm.y;
    attrs[vi*6u + 2u] = nrm.z;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct NormalsParams {
    n_faces: u32,
    n_verts: u32,
    scale:   f32,
    _pad:    u32,
}

#[allow(dead_code)] // _acc_buf and _params_buf are kept alive for the bind group.
struct NormalsResources {
    n_verts: u32,
    n_faces: u32,
    _acc_buf: wgpu::Buffer,
    _params_buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
    pl_clear: wgpu::ComputePipeline,
    pl_scatter: wgpu::ComputePipeline,
    pl_finalize: wgpu::ComputePipeline,
}

fn build_normals_resources(
    ctx: &GpuContext,
    position_buffer: &wgpu::Buffer,
    index_buffer:    &wgpu::Buffer,
    attr_buffer:     &wgpu::Buffer,
    n_verts: u32,
    n_faces: u32,
) -> NormalsResources {
    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Normals BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            },
        ],
    });

    let acc_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Normals Accumulator"),
        size: (n_verts as u64) * 3 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Picked so face-normal magnitudes (cross of edge vectors, ~edge_len^2)
    // stay well within i32 range even when summed over an entire one-ring.
    const SCALE: f32 = 65536.0;
    let params = NormalsParams { n_faces, n_verts, scale: SCALE, _pad: 0 };
    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Normals Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Normals BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: position_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: index_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: acc_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: attr_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
        ],
    });

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Normals Shader"),
        source: wgpu::ShaderSource::Wgsl(NORMALS_SHADER.into()),
    });
    let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Normals PL"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let mk = |entry: &str| ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });

    NormalsResources {
        n_verts, n_faces,
        _acc_buf: acc_buf,
        _params_buf: params_buf,
        bg,
        pl_clear: mk("cs_clear"),
        pl_scatter: mk("cs_scatter"),
        pl_finalize: mk("cs_finalize"),
    }
}

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

// Angle-weighted vertex normals (Max 1999). Each face contributes its UNIT
// normal scaled by the interior angle the face subtends at the vertex. This
// removes area bias — without it, irregular CDT strips along creases produce
// alternating bright/dark stripes because the largest neighbor dominates each
// on-crease vertex's normal.
fn compute_mesh_normals(positions: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; positions.len()];
    for &[i0, i1, i2] in faces {
        let p0 = positions[i0 as usize];
        let p1 = positions[i1 as usize];
        let p2 = positions[i2 as usize];

        let e01 = sub(p1, p0);
        let e02 = sub(p2, p0);
        let face_n = normalize(cross(e01, e02));

        let e10 = sub(p0, p1);
        let e12 = sub(p2, p1);
        let e20 = sub(p0, p2);
        let e21 = sub(p1, p2);

        let a0 = angle_between(e01, e02);
        let a1 = angle_between(e10, e12);
        let a2 = angle_between(e20, e21);

        let w0 = [face_n[0] * a0, face_n[1] * a0, face_n[2] * a0];
        let w1 = [face_n[0] * a1, face_n[1] * a1, face_n[2] * a1];
        let w2 = [face_n[0] * a2, face_n[1] * a2, face_n[2] * a2];

        normals[i0 as usize] = add(normals[i0 as usize], w0);
        normals[i1 as usize] = add(normals[i1 as usize], w1);
        normals[i2 as usize] = add(normals[i2 as usize], w2);
    }
    normals.into_iter().map(normalize).collect()
}

fn angle_between(a: [f32; 3], b: [f32; 3]) -> f32 {
    let la = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt();
    let lb = (b[0]*b[0] + b[1]*b[1] + b[2]*b[2]).sqrt();
    let denom = la * lb;
    if denom < 1e-12 { return 0.0; }
    let c = ((a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) / denom).clamp(-1.0, 1.0);
    c.acos()
}

fn grid_color(row: usize, col: usize) -> [f32; 3] {
    if (row + col) % 2 == 0 { [0.85, 0.85, 0.85] } else { [0.75, 0.75, 0.75] }
}

fn build_material_resources(
    ctx: &GpuContext,
    material: Material,
) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
    let uniform = MaterialUniform { kind: material as u32, _p0: 0, _p1: 0, _p2: 0 };
    let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Material UB"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Material BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Material BG"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
    });
    (buffer, bgl, bg)
}

pub struct Cloth {
    pub resolution: u32,
    pub positions: Vec<[f32; 3]>,
    pub faces: Option<Vec<[u32; 3]>>,
    pub colors: Option<Vec<[f32; 3]>>,
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    position_buffer: wgpu::Buffer,
    attr_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    num_verts: u32,
    wireframe_pipeline: wgpu::RenderPipeline,
    wireframe_vertex_buffer: wgpu::Buffer,
    wireframe_vertex_count: u32,
    wireframe_positions_bind_group: wgpu::BindGroup,
    pub wireframe_enabled: bool,
    material_buffer: wgpu::Buffer,
    material_bind_group: wgpu::BindGroup,
    pub material: Material,
    normals: NormalsResources,
}

impl Cloth {
    pub fn new(ctx: &GpuContext, resolution: u32, lighting: &Lighting) -> Self {
        let n = resolution as usize;
        assert!(n >= 2, "resolution must be at least 2");

        let mut positions = Vec::with_capacity(n * n);
        let mut pos_verts: Vec<PosVertex> = Vec::with_capacity(n * n);
        let mut attr_verts: Vec<AttrVertex> = Vec::with_capacity(n * n);

        for row in 0..n {
            for col in 0..n {
                let x = (col as f32 / (n - 1) as f32) * 2.0 - 1.0;
                let y = (row as f32 / (n - 1) as f32) * 2.0 - 1.0;
                let pos = [x * 0.9, y * 0.9, 0.0f32];
                positions.push(pos);
                pos_verts.push(PosVertex { position: [pos[0], pos[1], pos[2], 0.0] });
                attr_verts.push(AttrVertex {
                    normal: [0.0, 0.0, 1.0],
                    color:  grid_color(row, col),
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
        let num_verts = pos_verts.len() as u32;

        let position_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloth Position VB"),
            contents: bytemuck::cast_slice(&pos_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let attr_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloth Attr VB"),
            contents: bytemuck::cast_slice(&attr_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let index_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloth IB"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
        });

        let camera_bgl = make_camera_bgl(ctx);
        let (material_buffer, material_bgl, material_bind_group) =
            build_material_resources(ctx, Material::Cloth);

        let pipeline = PipelineBuilder::new(&ctx.device, ctx.config.format)
            .label("Cloth Pipeline")
            .shader(SHADER)
            .bind_group_layout(&lighting.bind_group_layout)
            .bind_group_layout(&camera_bgl)
            .bind_group_layout(&material_bgl)
            .vertex_layout(PosVertex::LAYOUT)
            .vertex_layout(AttrVertex::LAYOUT)
            .build();

        let shadow_pipeline = build_shadow_pipeline(ctx, &lighting.shadow_bind_group_layout);

        let gray = [0.2f32, 0.2, 0.2];
        let mut wireframe_verts: Vec<WireframeVertex> = Vec::new();
        for row in 0..n {
            for col in 0..n {
                let v = (row * n + col) as u32;
                if col < n - 1 {
                    wireframe_verts.push(WireframeVertex { color: gray, vidx: v });
                    wireframe_verts.push(WireframeVertex { color: gray, vidx: v + 1 });
                }
                if row < n - 1 {
                    wireframe_verts.push(WireframeVertex { color: gray, vidx: v });
                    wireframe_verts.push(WireframeVertex { color: gray, vidx: v + n as u32 });
                }
            }
        }
        let wireframe_vertex_count = wireframe_verts.len() as u32;

        let wireframe_vertex_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wireframe VB"),
            contents: bytemuck::cast_slice(&wireframe_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let wf_pos_bgl = wireframe_positions_bgl(ctx);
        let wireframe_positions_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Wireframe Positions BG"),
            layout: &wf_pos_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: position_buffer.as_entire_binding(),
            }],
        });

        let wireframe_pipeline = PipelineBuilder::new(&ctx.device, ctx.config.format)
            .label("Wireframe Pipeline")
            .shader(WIREFRAME_SHADER)
            .bind_group_layout(&wf_pos_bgl)
            .bind_group_layout(&camera_bgl)
            .vertex_layout(WireframeVertex::LAYOUT)
            .topology(wgpu::PrimitiveTopology::LineList)
            .build();

        let n_faces = (n.saturating_sub(1) * n.saturating_sub(1) * 2) as u32;
        let normals = build_normals_resources(
            ctx, &position_buffer, &index_buffer, &attr_buffer, num_verts, n_faces,
        );

        Self {
            resolution, positions, faces: None, colors: None,
            pipeline, shadow_pipeline,
            position_buffer, attr_buffer,
            index_buffer, index_count, num_verts,
            wireframe_pipeline, wireframe_vertex_buffer, wireframe_vertex_count,
            wireframe_positions_bind_group,
            wireframe_enabled: false,
            material_buffer, material_bind_group, material: Material::Cloth,
            normals,
        }
    }

    pub fn from_mesh(
        ctx: &GpuContext,
        positions: Vec<[f32; 3]>,
        faces: Vec<[u32; 3]>,
        colors: Vec<[f32; 3]>,
        edge_types: HashMap<(u32, u32), CreaseType>,
        lighting: &Lighting,
    ) -> Self {
        let num_verts = positions.len();
        let normals = compute_mesh_normals(&positions, &faces);

        let pos_verts: Vec<PosVertex> = positions.iter().map(|p| {
            PosVertex { position: [p[0], p[1], p[2], 0.0] }
        }).collect();
        let attr_verts: Vec<AttrVertex> = (0..num_verts).map(|i| AttrVertex {
            normal: normals[i],
            color:  colors.get(i).copied().unwrap_or([0.75, 0.88, 1.0]),
        }).collect();

        let indices: Vec<u32> = faces.iter().flat_map(|f| f.iter().copied()).collect();
        let index_count = indices.len() as u32;

        let position_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Position VB"),
            contents: bytemuck::cast_slice(&pos_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let attr_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Attr VB"),
            contents: bytemuck::cast_slice(&attr_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let index_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh IB"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
        });

        let camera_bgl = make_camera_bgl(ctx);
        let (material_buffer, material_bgl, material_bind_group) =
            build_material_resources(ctx, Material::Cloth);

        let pipeline = PipelineBuilder::new(&ctx.device, ctx.config.format)
            .label("Mesh Pipeline")
            .shader(SHADER)
            .bind_group_layout(&lighting.bind_group_layout)
            .bind_group_layout(&camera_bgl)
            .bind_group_layout(&material_bgl)
            .vertex_layout(PosVertex::LAYOUT)
            .vertex_layout(AttrVertex::LAYOUT)
            .build();

        let shadow_pipeline = build_shadow_pipeline(ctx, &lighting.shadow_bind_group_layout);

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

        let wireframe_verts: Vec<WireframeVertex> = edge_set.iter()
            .flat_map(|&(a, b)| {
                let color = match edge_types.get(&(a, b)) {
                    Some(CreaseType::Mountain) => [0.9, 0.1, 0.1],
                    Some(CreaseType::Valley)   => [0.1, 0.3, 1.0],
                    _ => [0.25, 0.25, 0.25],
                };
                [
                    WireframeVertex { color, vidx: a },
                    WireframeVertex { color, vidx: b },
                ]
            })
            .collect();
        let wireframe_vertex_count = wireframe_verts.len() as u32;

        let wireframe_vertex_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wireframe VB"),
            contents: bytemuck::cast_slice(&wireframe_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let wf_pos_bgl = wireframe_positions_bgl(ctx);
        let wireframe_positions_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Wireframe Positions BG"),
            layout: &wf_pos_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: position_buffer.as_entire_binding(),
            }],
        });

        let wireframe_pipeline = PipelineBuilder::new(&ctx.device, ctx.config.format)
            .label("Wireframe Pipeline")
            .shader(WIREFRAME_SHADER)
            .bind_group_layout(&wf_pos_bgl)
            .bind_group_layout(&camera_bgl)
            .vertex_layout(WireframeVertex::LAYOUT)
            .topology(wgpu::PrimitiveTopology::LineList)
            .build();

        let n_faces = (index_count / 3) as u32;
        let normals = build_normals_resources(
            ctx, &position_buffer, &index_buffer, &attr_buffer,
            num_verts as u32, n_faces,
        );

        Self {
            resolution: num_verts as u32,
            positions,
            faces: Some(faces),
            colors: Some(colors),
            pipeline,
            shadow_pipeline,
            position_buffer,
            attr_buffer,
            index_buffer,
            index_count,
            num_verts: num_verts as u32,
            wireframe_pipeline,
            wireframe_vertex_buffer,
            wireframe_vertex_count,
            wireframe_positions_bind_group,
            wireframe_enabled: false,
            material_buffer, material_bind_group, material: Material::Cloth,
            normals,
        }
    }

    pub fn set_material(&mut self, ctx: &GpuContext, m: Material) {
        self.material = m;
        let u = MaterialUniform { kind: m as u32, _p0: 0, _p1: 0, _p2: 0 };
        ctx.queue.write_buffer(&self.material_buffer, 0, bytemuck::bytes_of(&u));
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
        // Position buffer (vec4-padded) — written from CPU `self.positions`.
        let pos_verts: Vec<PosVertex> = self.positions.iter()
            .map(|p| PosVertex { position: [p[0], p[1], p[2], 0.0] })
            .collect();
        ctx.queue.write_buffer(&self.position_buffer, 0, bytemuck::cast_slice(&pos_verts));

        self.upload_attrs(ctx);
        // Wireframe positions are fetched from `position_buffer` directly in
        // the wireframe vertex shader, so there's nothing to re-upload here.
    }

    /// Recompute normals from `self.positions` and upload normal+color attrs.
    /// Use this on the GPU sim path: the position buffer is filled from
    /// `q_buf` directly, but normals still come from CPU positions (they
    /// trail GPU positions by however many frames the async readback is
    /// behind — typically 1–3 — which is imperceptible for shading).
    pub fn upload_attrs(&self, ctx: &GpuContext) {
        let attrs: Vec<AttrVertex> = if let Some(ref faces) = self.faces {
            let normals = compute_mesh_normals(&self.positions, faces);
            (0..self.num_verts as usize).map(|i| AttrVertex {
                normal: normals[i],
                color:  self.colors.as_ref()
                    .and_then(|c| c.get(i).copied())
                    .unwrap_or([0.75, 0.88, 1.0]),
            }).collect()
        } else {
            let n = self.resolution as usize;
            let normals = compute_vertex_normals(&self.positions, n);
            (0..self.num_verts as usize).map(|i| {
                let row = i / n;
                let col = i % n;
                AttrVertex { normal: normals[i], color: grid_color(row, col) }
            }).collect()
        };
        ctx.queue.write_buffer(&self.attr_buffer, 0, bytemuck::cast_slice(&attrs));
    }

    /// Copy positions from a source storage buffer (e.g. the GPU sim's
    /// `q_buf`, which is `array<vec4<f32>>`) into our position vertex buffer
    /// in the caller's command encoder. No CPU round-trip, no readback
    /// latency — positions land in the vertex buffer in the same submission
    /// as the sim step that produced them.
    pub fn encode_position_sync(&self, enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer) {
        let bytes = (self.num_verts as u64) * 16;
        enc.copy_buffer_to_buffer(src, 0, &self.position_buffer, 0, bytes);
    }

    /// Recompute per-vertex normals on the GPU and write them into the
    /// normal slots of `attr_buffer`. Color slots are untouched. Three
    /// compute passes: clear accumulator → per-face scatter → per-vertex
    /// normalize. Reads positions from `position_buffer`, so call this
    /// AFTER `encode_position_sync` in the same encoder.
    pub fn encode_normals(&self, enc: &mut wgpu::CommandEncoder) {
        let n_verts = self.normals.n_verts;
        let n_faces = self.normals.n_faces;
        if n_verts == 0 || n_faces == 0 { return; }

        let groups = |total: u32, ws: u32| (total + ws - 1) / ws;

        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("normals_clear"), timestamp_writes: None });
            p.set_pipeline(&self.normals.pl_clear);
            p.set_bind_group(0, &self.normals.bg, &[]);
            p.dispatch_workgroups(groups(n_verts * 3, 64), 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("normals_scatter"), timestamp_writes: None });
            p.set_pipeline(&self.normals.pl_scatter);
            p.set_bind_group(0, &self.normals.bg, &[]);
            p.dispatch_workgroups(groups(n_faces, 64), 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("normals_finalize"), timestamp_writes: None });
            p.set_pipeline(&self.normals.pl_finalize);
            p.set_bind_group(0, &self.normals.bg, &[]);
            p.dispatch_workgroups(groups(n_verts, 64), 1, 1);
        }
    }

    /// Render this mesh into the lighting's shadow map (depth-only pass).
    pub fn render_shadow(&self, ctx: &GpuContext, lighting: &Lighting) {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Shadow Encoder"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &lighting.shadow_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.shadow_pipeline);
            rpass.set_bind_group(0, &lighting.shadow_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.position_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn render(&self, ctx: &GpuContext, view: &wgpu::TextureView, lighting: &Lighting, camera: &Camera) {
        self.render_impl(ctx, view, lighting, camera, true);
    }

    pub fn render_over(&self, ctx: &GpuContext, view: &wgpu::TextureView, lighting: &Lighting, camera: &Camera) {
        self.render_impl(ctx, view, lighting, camera, false);
    }

    fn render_impl(&self, ctx: &GpuContext, view: &wgpu::TextureView, lighting: &Lighting, camera: &Camera, clear: bool) {
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
            rpass.set_bind_group(0, &lighting.bind_group, &[]);
            rpass.set_bind_group(1, &camera.bind_group, &[]);
            rpass.set_bind_group(2, &self.material_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.position_buffer.slice(..));
            rpass.set_vertex_buffer(1, self.attr_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.index_count, 0, 0..1);

            if self.wireframe_enabled {
                rpass.set_pipeline(&self.wireframe_pipeline);
                rpass.set_bind_group(0, &self.wireframe_positions_bind_group, &[]);
                rpass.set_bind_group(1, &camera.bind_group, &[]);
                rpass.set_vertex_buffer(0, self.wireframe_vertex_buffer.slice(..));
                rpass.draw(0..self.wireframe_vertex_count, 0..1);
            }
        }
        ctx.queue.submit(Some(encoder.finish()));
    }
}

fn make_camera_bgl(ctx: &GpuContext) -> wgpu::BindGroupLayout {
    ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    })
}

/// @group(0) for the wireframe pipeline: a storage binding over the live GPU
/// position buffer so the vertex shader can fetch each line endpoint at draw
/// time. Removes the need to keep a CPU mirror of positions in sync with the
/// GPU sim's `q_buf`.
fn wireframe_positions_bgl(ctx: &GpuContext) -> wgpu::BindGroupLayout {
    ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Wireframe Positions BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn build_shadow_pipeline(ctx: &GpuContext, shadow_bgl: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shadow Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
    });
    let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Shadow Pipeline Layout"),
        bind_group_layouts: &[Some(shadow_bgl)],
        immediate_size: 0,
    });
    ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Shadow Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[PosVertex::LAYOUT],
            compilation_options: Default::default(),
        },
        fragment: None,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        cache: None,
        multiview_mask: None,
    })
}
