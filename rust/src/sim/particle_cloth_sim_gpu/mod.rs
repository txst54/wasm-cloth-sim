//! GPU compute path for `ParticleClothSim`. WASM-only.
//!
//! Mirrors the CPU implementation step-for-step but runs each phase as a
//! WebGPU compute dispatch. Distance constraints (stretch + bend) use greedy
//! graph coloring so that within a color class no two constraints share a
//! vertex — the kernel can therefore write both endpoints with plain stores.
//! Self-collision uses a Jacobi accumulator into a fixed-point i32 atomic
//! `dq` buffer because contact pairs change every substep and coloring them
//! on-GPU is not worth the complexity.
//!
//! Construction takes a CPU `ParticleClothSim` snapshot and uploads its
//! buffers; thereafter the GPU owns the simulation state. `positions()`
//! returns a CPU-side cached copy that can be refreshed via
//! `sync_positions().await`.

pub mod coloring;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use nalgebra as na;
use wgpu::util::DeviceExt;

use crate::params::SimParams;

use super::particle_cloth_sim::ParticleClothSim;
use super::sdf::{SdfObstacle, SignedDistanceField};
use super::shared::Positions;
use super::traits::MeshSim;

use coloring::Coloring;

// ── Uniform / GPU types ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct PredictParams { h: f32, g: f32, damping: f32, n: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct CountParams { n: u32, _p0: u32, _p1: u32, _p2: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct DistanceParams { h: f32, alpha: f32, n_color: u32, base: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct HashCountParams { inv_h: f32, table_size: u32, n: u32, _p: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct HashScanMeta { table_size: u32, _p0: u32, _p1: u32, _p2: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct ScatterMeta { n: u32, _p0: u32, _p1: u32, _p2: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct ContactParams {
    h: f32, alpha: f32, mu: f32, n: u32,
    inv_h: f32, ts: u32, scale: f32, _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct ContactApplyParams { n: u32, inv_scale: f32, _p0: u32, _p1: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct SdfParams { alpha: f32, mu: f32, n: u32, num_obs: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct PinVelParams {
    inv_h: f32, pin_enabled: u32, mouse_idx: u32, n: u32,
    mouse_pos: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct ObstacleGpu {
    center:  [f32; 4],
    rot0:    [f32; 4],
    rot1:    [f32; 4],
    rot2:    [f32; 4],
    a:       [f32; 4],
    b:       [f32; 4],
    kind:    u32,
    _p:      [u32; 3],
}

const FIXED_POINT_SCALE: f32 = 1_048_576.0; // 2^20

// ── Helpers ──────────────────────────────────────────────────────────────────

fn vec3_buf(positions: &Positions) -> Vec<[f32; 4]> {
    let n = positions.nrows();
    let mut out = vec![[0.0; 4]; n];
    for i in 0..n {
        out[i] = [positions[(i, 0)], positions[(i, 1)], positions[(i, 2)], 0.0];
    }
    out
}

fn next_prime(mut n: u32) -> u32 {
    fn is_prime(n: u32) -> bool {
        if n < 2 { return false; }
        if n % 2 == 0 { return n == 2; }
        let mut d = 3u32;
        while d.saturating_mul(d) <= n {
            if n % d == 0 { return false; }
            d += 2;
        }
        true
    }
    if n < 2 { return 2; }
    if n % 2 == 0 { n += 1; }
    loop { if is_prime(n) { return n; } n += 2; }
}

fn build_one_ring_csr(one_ring: &[HashSet<u32>]) -> (Vec<u32>, Vec<u32>) {
    let n = one_ring.len();
    let mut row = Vec::with_capacity(n + 1);
    let mut col = Vec::new();
    let mut acc: u32 = 0;
    row.push(0);
    for nb in one_ring {
        let mut sorted: Vec<u32> = nb.iter().copied().collect();
        sorted.sort_unstable();
        col.extend_from_slice(&sorted);
        acc += sorted.len() as u32;
        row.push(acc);
    }
    (row, col)
}

fn obstacle_to_gpu(o: &SdfObstacle) -> ObstacleGpu {
    let r = &o.rotation;
    let row = |i: usize| [r[(i, 0)], r[(i, 1)], r[(i, 2)], 0.0];
    let (kind, a, b) = match o.sdf {
        SignedDistanceField::Sphere  { radius } =>
            (0u32, [radius, 0.0, 0.0, 0.0], [0.0; 4]),
        SignedDistanceField::Plane   { n, d } =>
            (1u32, [n.x, n.y, n.z, 0.0], [d, 0.0, 0.0, 0.0]),
        SignedDistanceField::Box     { half_ext } =>
            (2u32, [half_ext.x, half_ext.y, half_ext.z, 0.0], [0.0; 4]),
        SignedDistanceField::Capsule { half_h, radius } =>
            (3u32, [half_h, radius, 0.0, 0.0], [0.0; 4]),
    };
    ObstacleGpu {
        center: [o.center.x, o.center.y, o.center.z, 0.0],
        rot0: row(0), rot1: row(1), rot2: row(2),
        a, b, kind, _p: [0; 3],
    }
}

// ── Bind-group / pipeline plumbing ──────────────────────────────────────────

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_bgl(
    device: &wgpu::Device,
    label: &str,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn make_pipeline(
    device: &wgpu::Device,
    label: &str,
    bgls: &[&wgpu::BindGroupLayout],
    src: &str,
    entry: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let bgls_opt: Vec<Option<&wgpu::BindGroupLayout>> = bgls.iter().map(|b| Some(*b)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bgls_opt,
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

// ── Public type ─────────────────────────────────────────────────────────────

pub struct ParticleClothSimGpu {
    device: wgpu::Device,
    queue:  wgpu::Queue,

    n:        u32,
    n_edges:  u32,
    n_bends:  u32,
    table_size: u32,
    r_max:    f32,

    // Persistent buffers.
    pub q_buf:       wgpu::Buffer,
    q_prev_buf:      wgpu::Buffer,
    q_pred_buf:      wgpu::Buffer,
    q_rest_buf:      wgpu::Buffer,
    v_buf:           wgpu::Buffer,
    w_inv_buf:       wgpu::Buffer,
    radius_buf:      wgpu::Buffer,

    edges_buf:        wgpu::Buffer,
    edge_rest_buf:    wgpu::Buffer,
    stretch_lambda_buf: wgpu::Buffer,
    bend_pairs_buf:   wgpu::Buffer,
    bend_rest_buf:    wgpu::Buffer,
    bend_lambda_buf:  wgpu::Buffer,
    color_idx_stretch_buf: wgpu::Buffer,
    color_idx_bend_buf:    wgpu::Buffer,

    one_ring_row_buf: wgpu::Buffer,
    one_ring_col_buf: wgpu::Buffer,

    cell_count_buf:   wgpu::Buffer,
    cell_start_buf:   wgpu::Buffer,
    cell_cursor_buf:  wgpu::Buffer,
    particle_cell_buf: wgpu::Buffer,
    particle_id_buf:   wgpu::Buffer,

    dq_buf:           wgpu::Buffer,

    obstacles_buf:    wgpu::Buffer,
    obs_capacity:     u32,
    num_obstacles:    u32,

    // Per-pass uniform buffers (rewritten each substep).
    predict_u:    wgpu::Buffer,
    copy_u:       wgpu::Buffer,
    // One uniform buffer per color class. Separate buffers are required
    // because all `queue.write_buffer` calls before a submit are flushed
    // ahead of any encoded command — overwriting the same uniform across
    // colors leaves only the last write visible to every dispatch.
    distance_us_stretch: Vec<wgpu::Buffer>,
    distance_us_bend:    Vec<wgpu::Buffer>,
    zero_lambda_stretch_u: wgpu::Buffer,
    zero_lambda_bend_u:    wgpu::Buffer,
    zero_dq_u:    wgpu::Buffer,
    zero_cellcount_u: wgpu::Buffer,
    hash_count_u: wgpu::Buffer,
    hash_scan_u:  wgpu::Buffer,
    scatter_u:    wgpu::Buffer,
    contact_u:    wgpu::Buffer,
    contact_apply_u: wgpu::Buffer,
    sdf_u:        wgpu::Buffer,
    pin_vel_u:    wgpu::Buffer,

    // Pipelines.
    pl_predict:       wgpu::ComputePipeline,
    pl_copy_pred:     wgpu::ComputePipeline,
    pl_zero_f32:      wgpu::ComputePipeline,
    pl_zero_u32:      wgpu::ComputePipeline,
    pl_distance:      wgpu::ComputePipeline,
    pl_hash_count:    wgpu::ComputePipeline,
    pl_hash_scan:     wgpu::ComputePipeline,
    pl_hash_scatter:  wgpu::ComputePipeline,
    pl_contact:       wgpu::ComputePipeline,
    pl_contact_apply: wgpu::ComputePipeline,
    pl_sdf:           wgpu::ComputePipeline,
    pl_pin_vel:       wgpu::ComputePipeline,

    // Bind groups (pre-built where stable).
    bg_predict:       wgpu::BindGroup,
    bg_copy_pred:     wgpu::BindGroup,
    bg_zero_lambda_s: wgpu::BindGroup,
    bg_zero_lambda_b: wgpu::BindGroup,
    bg_zero_dq:       wgpu::BindGroup,
    bg_zero_cellcount: wgpu::BindGroup,
    // One bg_distance_core per color (each binds its own uniform buffer).
    bg_distance_core_stretch: Vec<wgpu::BindGroup>,
    bg_distance_core_bend:    Vec<wgpu::BindGroup>,
    bg_distance_stretch: wgpu::BindGroup,
    bg_distance_bend: wgpu::BindGroup,
    bg_hash_count:    wgpu::BindGroup,
    bg_hash_scan:     wgpu::BindGroup,
    bg_hash_scatter:  wgpu::BindGroup,
    bg_contact_core:  wgpu::BindGroup,
    bg_contact_hash:  wgpu::BindGroup,
    bg_contact_dq:    wgpu::BindGroup,
    bg_contact_apply: wgpu::BindGroup,
    bg_sdf:           wgpu::BindGroup,
    bg_pin_vel:       wgpu::BindGroup,

    // Coloring metadata (CPU side).
    stretch_coloring: Coloring,
    bend_coloring:    Coloring,

    // Mouse / pin state.
    clicked_vertex: Option<usize>,
    mouse_pos:      [f32; 3],

    // Position cache + non-blocking readback flags. The renderer reads
    // `positions_cache` synchronously every frame; `poll_readback` drains the
    // staging buffer once `map_async`'s callback has flipped `ready`.
    positions_cache: Positions,
    staging_q:       wgpu::Buffer,
    map_flags:       Arc<Mutex<MapFlags>>,

    pub resolution: usize,
}

impl ParticleClothSimGpu {
    pub fn from_cpu(
        device: wgpu::Device,
        queue:  wgpu::Queue,
        cpu: &ParticleClothSim,
    ) -> Self {
        let n        = cpu.q.nrows() as u32;
        let n_edges  = cpu.edges.len() as u32;
        let n_bends  = cpu.diamonds.len() as u32;

        // Coloring (debug-validated).
        let stretch_coloring = coloring::color_edges(&cpu.edges, n as usize);
        let bend_coloring    = coloring::color_bend(&cpu.diamonds, n as usize);
        #[cfg(debug_assertions)]
        {
            let edge_refs: Vec<&[u32]> = cpu.edges.iter().map(|e| &e[..]).collect();
            coloring::validate(&stretch_coloring, &edge_refs, n as usize);
            let bend_pairs: Vec<[u32; 2]> = cpu.diamonds.iter().map(|d| [d[2], d[3]]).collect();
            let bend_refs: Vec<&[u32]> = bend_pairs.iter().map(|p| &p[..]).collect();
            coloring::validate(&bend_coloring, &bend_refs, n as usize);
        }

        // Buffers.
        let q_data       = vec3_buf(&cpu.q);
        let q_rest_data  = vec3_buf(&cpu.q_rest);
        let v_data       = vec![[0f32; 4]; n as usize];

        let mk_storage = |label: &str, contents: &[u8], copy_src: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            if copy_src { usage |= wgpu::BufferUsages::COPY_SRC; }
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };

        let q_buf       = mk_storage("q",       bytemuck::cast_slice(&q_data), true);
        let q_prev_buf  = mk_storage("q_prev",  bytemuck::cast_slice(&q_data), false);
        let q_pred_buf  = mk_storage("q_pred",  bytemuck::cast_slice(&q_data), false);
        let q_rest_buf  = mk_storage("q_rest",  bytemuck::cast_slice(&q_rest_data), false);
        let v_buf       = mk_storage("v",       bytemuck::cast_slice(&v_data), false);
        let w_inv_buf   = mk_storage("w_inv",   bytemuck::cast_slice(cpu.w.as_slice()), false);
        let radius_buf  = mk_storage("radius",  bytemuck::cast_slice(&cpu.r), false);

        // Edges and bends — flatten.
        let edges_flat: Vec<[u32; 2]> = cpu.edges.clone();
        let bend_pairs: Vec<[u32; 2]> = cpu.diamonds.iter().map(|d| [d[2], d[3]]).collect();

        let edges_buf       = mk_storage("edges",       bytemuck::cast_slice(&edges_flat), false);
        let edge_rest_buf   = mk_storage("edge_rest",   bytemuck::cast_slice(&cpu.edge_rest), false);
        let stretch_lambda_buf = mk_storage("stretch_lambda", bytemuck::cast_slice(&vec![0f32; n_edges as usize]), false);
        let bend_pairs_buf  = mk_storage("bend_pairs",  bytemuck::cast_slice(&bend_pairs), false);
        let bend_rest_buf   = mk_storage("bend_rest",   bytemuck::cast_slice(&cpu.diamond_rest), false);
        let bend_lambda_buf = mk_storage("bend_lambda", bytemuck::cast_slice(&vec![0f32; n_bends as usize]), false);

        let color_idx_stretch_buf = mk_storage("color_idx_stretch",
            bytemuck::cast_slice(&stretch_coloring.idx), false);
        let color_idx_bend_buf    = mk_storage("color_idx_bend",
            bytemuck::cast_slice(&bend_coloring.idx), false);

        let (or_row, or_col) = build_one_ring_csr(&cpu.one_ring);
        let one_ring_row_buf = mk_storage("one_ring_row", bytemuck::cast_slice(&or_row), false);
        let one_ring_col_buf = mk_storage("one_ring_col",
            bytemuck::cast_slice(if or_col.is_empty() { &[0u32][..] } else { &or_col[..] }), false);

        let table_size = next_prime((n as u32 * 2).max(31));
        let cell_count_buf  = mk_storage("cell_count",
            bytemuck::cast_slice(&vec![0u32; table_size as usize]), false);
        let cell_start_buf  = mk_storage("cell_start",
            bytemuck::cast_slice(&vec![0u32; table_size as usize + 1]), false);
        let cell_cursor_buf = mk_storage("cell_cursor",
            bytemuck::cast_slice(&vec![0u32; table_size as usize + 1]), false);
        let particle_cell_buf = mk_storage("particle_cell",
            bytemuck::cast_slice(&vec![0u32; n as usize]), false);
        let particle_id_buf   = mk_storage("particle_id",
            bytemuck::cast_slice(&vec![0u32; n as usize]), false);

        let dq_buf = mk_storage("dq",
            bytemuck::cast_slice(&vec![0i32; n as usize * 3]), false);

        // Obstacles. Reserve a small capacity that grows on demand.
        let obs_capacity: u32 = 8.max(cpu.obstacles.len() as u32);
        let obs_init: Vec<ObstacleGpu> = cpu.obstacles.iter().map(obstacle_to_gpu).collect();
        let mut obs_padded = obs_init.clone();
        obs_padded.resize(obs_capacity as usize, ObstacleGpu::default());
        let obstacles_buf = mk_storage("obstacles", bytemuck::cast_slice(&obs_padded), false);
        let num_obstacles = cpu.obstacles.len() as u32;

        // Uniform buffers (small, mutable).
        let mk_uniform = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label), size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let predict_u    = mk_uniform("predict_u",    16);
        let copy_u       = mk_uniform("copy_u",       16);
        let distance_us_stretch: Vec<wgpu::Buffer> = (0..stretch_coloring.num_colors())
            .map(|k| mk_uniform(&format!("distance_u_s{}", k), 16))
            .collect();
        let distance_us_bend: Vec<wgpu::Buffer> = (0..bend_coloring.num_colors())
            .map(|k| mk_uniform(&format!("distance_u_b{}", k), 16))
            .collect();
        let zero_lambda_stretch_u = mk_uniform("zero_lambda_s_u", 16);
        let zero_lambda_bend_u    = mk_uniform("zero_lambda_b_u", 16);
        let zero_dq_u    = mk_uniform("zero_dq_u",    16);
        let zero_cellcount_u = mk_uniform("zero_cellcount_u", 16);
        let hash_count_u = mk_uniform("hash_count_u", 16);
        let hash_scan_u  = mk_uniform("hash_scan_u",  16);
        let scatter_u    = mk_uniform("scatter_u",    16);
        let contact_u    = mk_uniform("contact_u",    32);
        let contact_apply_u = mk_uniform("contact_apply_u", 16);
        let sdf_u        = mk_uniform("sdf_u",        16);
        let pin_vel_u    = mk_uniform("pin_vel_u",    32);

        // Pre-write static-size meta uniforms.
        queue.write_buffer(&zero_lambda_stretch_u, 0, bytemuck::bytes_of(&CountParams { n: n_edges, ..Default::default() }));
        queue.write_buffer(&zero_lambda_bend_u,    0, bytemuck::bytes_of(&CountParams { n: n_bends, ..Default::default() }));
        queue.write_buffer(&zero_dq_u,             0, bytemuck::bytes_of(&CountParams { n: n * 3, ..Default::default() }));
        queue.write_buffer(&zero_cellcount_u,      0, bytemuck::bytes_of(&CountParams { n: table_size, ..Default::default() }));
        queue.write_buffer(&copy_u,                0, bytemuck::bytes_of(&CountParams { n, ..Default::default() }));
        queue.write_buffer(&hash_scan_u,           0, bytemuck::bytes_of(&HashScanMeta { table_size, ..Default::default() }));
        queue.write_buffer(&scatter_u,             0, bytemuck::bytes_of(&ScatterMeta { n, ..Default::default() }));

        // Bind-group layouts.
        let bgl_predict = make_bgl(&device, "bgl_predict", vec![
            uniform_entry(0),
            storage_entry(1, false), storage_entry(2, false),
            storage_entry(3, false), storage_entry(4, true),
        ]);
        let bgl_copy_pred = make_bgl(&device, "bgl_copy_pred", vec![
            uniform_entry(0), storage_entry(1, true), storage_entry(2, false),
        ]);
        let bgl_zero_f32 = make_bgl(&device, "bgl_zero_f32", vec![
            uniform_entry(0), storage_entry(1, false),
        ]);
        let bgl_zero_u32 = make_bgl(&device, "bgl_zero_u32", vec![
            uniform_entry(0), storage_entry(1, false),
        ]);
        let bgl_distance_core = make_bgl(&device, "bgl_distance_core", vec![
            uniform_entry(0), storage_entry(1, false), storage_entry(2, true),
        ]);
        let bgl_distance_pair = make_bgl(&device, "bgl_distance_pair", vec![
            storage_entry(0, true), storage_entry(1, true),
            storage_entry(2, false), storage_entry(3, true),
        ]);
        let bgl_hash_count = make_bgl(&device, "bgl_hash_count", vec![
            uniform_entry(0), storage_entry(1, true),
            storage_entry(2, false), storage_entry(3, false),
        ]);
        let bgl_hash_scan = make_bgl(&device, "bgl_hash_scan", vec![
            uniform_entry(0), storage_entry(1, true),
            storage_entry(2, false), storage_entry(3, false),
        ]);
        let bgl_hash_scatter = make_bgl(&device, "bgl_hash_scatter", vec![
            uniform_entry(0), storage_entry(1, true),
            storage_entry(2, false), storage_entry(3, false),
        ]);
        let bgl_contact_core = make_bgl(&device, "bgl_contact_core", vec![
            uniform_entry(0),
            storage_entry(1, true), storage_entry(2, true),
            storage_entry(3, true), storage_entry(4, true),
            storage_entry(5, true),
        ]);
        let bgl_contact_hash = make_bgl(&device, "bgl_contact_hash", vec![
            storage_entry(0, true), storage_entry(1, true),
            storage_entry(2, true), storage_entry(3, true),
        ]);
        let bgl_contact_dq = make_bgl(&device, "bgl_contact_dq", vec![
            storage_entry(0, false),
        ]);
        let bgl_contact_apply = make_bgl(&device, "bgl_contact_apply", vec![
            uniform_entry(0),
            storage_entry(1, false), storage_entry(2, false),
            storage_entry(3, true),
        ]);
        let bgl_sdf = make_bgl(&device, "bgl_sdf", vec![
            uniform_entry(0),
            storage_entry(1, false), storage_entry(2, true),
            storage_entry(3, true),  storage_entry(4, true),
            storage_entry(5, true),
        ]);
        let bgl_pin_vel = make_bgl(&device, "bgl_pin_vel", vec![
            uniform_entry(0),
            storage_entry(1, false), storage_entry(2, true),
            storage_entry(3, false), storage_entry(4, true),
        ]);

        // Pipelines.
        let pl_predict       = make_pipeline(&device, "pl_predict",
            &[&bgl_predict], include_str!("shaders/predict.wgsl"), "main");
        let pl_copy_pred     = make_pipeline(&device, "pl_copy_pred",
            &[&bgl_copy_pred], include_str!("shaders/copy_q_to_pred.wgsl"), "main");
        let pl_zero_f32      = make_pipeline(&device, "pl_zero_f32",
            &[&bgl_zero_f32], include_str!("shaders/zero_lambdas.wgsl"), "main");
        let pl_zero_u32      = make_pipeline(&device, "pl_zero_u32",
            &[&bgl_zero_u32], include_str!("shaders/zero_u32.wgsl"), "main");
        let pl_distance      = make_pipeline(&device, "pl_distance",
            &[&bgl_distance_core, &bgl_distance_pair],
            include_str!("shaders/distance.wgsl"), "main");
        let pl_hash_count    = make_pipeline(&device, "pl_hash_count",
            &[&bgl_hash_count], include_str!("shaders/hash_count.wgsl"), "main");
        let pl_hash_scan     = make_pipeline(&device, "pl_hash_scan",
            &[&bgl_hash_scan], include_str!("shaders/hash_scan.wgsl"), "main");
        let pl_hash_scatter  = make_pipeline(&device, "pl_hash_scatter",
            &[&bgl_hash_scatter], include_str!("shaders/hash_scatter.wgsl"), "main");
        let pl_contact       = make_pipeline(&device, "pl_contact",
            &[&bgl_contact_core, &bgl_contact_hash, &bgl_contact_dq],
            include_str!("shaders/contact_jacobi.wgsl"), "main");
        let pl_contact_apply = make_pipeline(&device, "pl_contact_apply",
            &[&bgl_contact_apply], include_str!("shaders/contact_apply.wgsl"), "main");
        let pl_sdf           = make_pipeline(&device, "pl_sdf",
            &[&bgl_sdf], include_str!("shaders/sdf_contact.wgsl"), "main");
        let pl_pin_vel       = make_pipeline(&device, "pl_pin_vel",
            &[&bgl_pin_vel], include_str!("shaders/pin_velocity.wgsl"), "main");

        // Bind groups.
        let mk_bg = |label: &str, layout: &wgpu::BindGroupLayout, entries: Vec<wgpu::BindGroupEntry>| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label), layout, entries: &entries,
            })
        };
        fn entry(b: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() }
        }

        let bg_predict = mk_bg("bg_predict", &bgl_predict, vec![
            entry(0, &predict_u), entry(1, &q_buf), entry(2, &q_prev_buf),
            entry(3, &v_buf),     entry(4, &w_inv_buf),
        ]);
        let bg_copy_pred = mk_bg("bg_copy_pred", &bgl_copy_pred, vec![
            entry(0, &copy_u), entry(1, &q_buf), entry(2, &q_pred_buf),
        ]);
        let bg_zero_lambda_s = mk_bg("bg_zero_lambda_s", &bgl_zero_f32, vec![
            entry(0, &zero_lambda_stretch_u), entry(1, &stretch_lambda_buf),
        ]);
        let bg_zero_lambda_b = mk_bg("bg_zero_lambda_b", &bgl_zero_f32, vec![
            entry(0, &zero_lambda_bend_u), entry(1, &bend_lambda_buf),
        ]);
        let bg_zero_dq = mk_bg("bg_zero_dq", &bgl_zero_u32, vec![
            entry(0, &zero_dq_u), entry(1, &dq_buf),
        ]);
        let bg_zero_cellcount = mk_bg("bg_zero_cellcount", &bgl_zero_u32, vec![
            entry(0, &zero_cellcount_u), entry(1, &cell_count_buf),
        ]);
        let bg_distance_core_stretch: Vec<wgpu::BindGroup> = distance_us_stretch.iter().enumerate()
            .map(|(k, u)| mk_bg(&format!("bg_distance_core_s{}", k), &bgl_distance_core, vec![
                entry(0, u), entry(1, &q_buf), entry(2, &w_inv_buf),
            ])).collect();
        let bg_distance_core_bend: Vec<wgpu::BindGroup> = distance_us_bend.iter().enumerate()
            .map(|(k, u)| mk_bg(&format!("bg_distance_core_b{}", k), &bgl_distance_core, vec![
                entry(0, u), entry(1, &q_buf), entry(2, &w_inv_buf),
            ])).collect();
        let bg_distance_stretch = mk_bg("bg_distance_stretch", &bgl_distance_pair, vec![
            entry(0, &edges_buf), entry(1, &edge_rest_buf),
            entry(2, &stretch_lambda_buf), entry(3, &color_idx_stretch_buf),
        ]);
        let bg_distance_bend = mk_bg("bg_distance_bend", &bgl_distance_pair, vec![
            entry(0, &bend_pairs_buf), entry(1, &bend_rest_buf),
            entry(2, &bend_lambda_buf), entry(3, &color_idx_bend_buf),
        ]);
        let bg_hash_count = mk_bg("bg_hash_count", &bgl_hash_count, vec![
            entry(0, &hash_count_u), entry(1, &q_buf),
            entry(2, &cell_count_buf), entry(3, &particle_cell_buf),
        ]);
        let bg_hash_scan = mk_bg("bg_hash_scan", &bgl_hash_scan, vec![
            entry(0, &hash_scan_u), entry(1, &cell_count_buf),
            entry(2, &cell_start_buf), entry(3, &cell_cursor_buf),
        ]);
        let bg_hash_scatter = mk_bg("bg_hash_scatter", &bgl_hash_scatter, vec![
            entry(0, &scatter_u), entry(1, &particle_cell_buf),
            entry(2, &cell_cursor_buf), entry(3, &particle_id_buf),
        ]);
        let bg_contact_core = mk_bg("bg_contact_core", &bgl_contact_core, vec![
            entry(0, &contact_u), entry(1, &q_buf), entry(2, &q_pred_buf),
            entry(3, &q_rest_buf), entry(4, &w_inv_buf), entry(5, &radius_buf),
        ]);
        let bg_contact_hash = mk_bg("bg_contact_hash", &bgl_contact_hash, vec![
            entry(0, &cell_start_buf), entry(1, &particle_id_buf),
            entry(2, &one_ring_row_buf), entry(3, &one_ring_col_buf),
        ]);
        let bg_contact_dq = mk_bg("bg_contact_dq", &bgl_contact_dq, vec![
            entry(0, &dq_buf),
        ]);
        let bg_contact_apply = mk_bg("bg_contact_apply", &bgl_contact_apply, vec![
            entry(0, &contact_apply_u), entry(1, &q_buf),
            entry(2, &dq_buf), entry(3, &w_inv_buf),
        ]);
        let bg_sdf = mk_bg("bg_sdf", &bgl_sdf, vec![
            entry(0, &sdf_u), entry(1, &q_buf), entry(2, &q_pred_buf),
            entry(3, &w_inv_buf), entry(4, &radius_buf), entry(5, &obstacles_buf),
        ]);
        let bg_pin_vel = mk_bg("bg_pin_vel", &bgl_pin_vel, vec![
            entry(0, &pin_vel_u), entry(1, &q_buf), entry(2, &q_prev_buf),
            entry(3, &v_buf),     entry(4, &w_inv_buf),
        ]);

        // Staging for readback.
        let staging_q = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_q"),
            size: (n as u64) * 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let r_max = cpu.r_max;
        let positions_cache = cpu.q.clone();

        Self {
            device, queue,
            n, n_edges, n_bends, table_size, r_max,
            q_buf, q_prev_buf, q_pred_buf, q_rest_buf, v_buf, w_inv_buf, radius_buf,
            edges_buf, edge_rest_buf, stretch_lambda_buf,
            bend_pairs_buf, bend_rest_buf, bend_lambda_buf,
            color_idx_stretch_buf, color_idx_bend_buf,
            one_ring_row_buf, one_ring_col_buf,
            cell_count_buf, cell_start_buf, cell_cursor_buf,
            particle_cell_buf, particle_id_buf,
            dq_buf,
            obstacles_buf, obs_capacity, num_obstacles,
            predict_u, copy_u,
            distance_us_stretch, distance_us_bend,
            zero_lambda_stretch_u, zero_lambda_bend_u,
            zero_dq_u, zero_cellcount_u,
            hash_count_u, hash_scan_u, scatter_u,
            contact_u, contact_apply_u, sdf_u, pin_vel_u,
            pl_predict, pl_copy_pred, pl_zero_f32, pl_zero_u32, pl_distance,
            pl_hash_count, pl_hash_scan, pl_hash_scatter,
            pl_contact, pl_contact_apply, pl_sdf, pl_pin_vel,
            bg_predict, bg_copy_pred,
            bg_zero_lambda_s, bg_zero_lambda_b, bg_zero_dq, bg_zero_cellcount,
            bg_distance_core_stretch, bg_distance_core_bend,
            bg_distance_stretch, bg_distance_bend,
            bg_hash_count, bg_hash_scan, bg_hash_scatter,
            bg_contact_core, bg_contact_hash, bg_contact_dq, bg_contact_apply,
            bg_sdf, bg_pin_vel,
            stretch_coloring, bend_coloring,
            clicked_vertex: None, mouse_pos: [0.0; 3],
            positions_cache, staging_q,
            map_flags: Arc::new(Mutex::new(MapFlags::default())),
            resolution: cpu.resolution,
        }
    }

    /// Replace the obstacle list. Resizes the GPU buffer if capacity is too
    /// small. Cheap; called only when the scene changes.
    pub fn set_obstacles(&mut self, obstacles: &[SdfObstacle]) {
        let need = obstacles.len() as u32;
        if need > self.obs_capacity {
            let new_cap = need.next_power_of_two().max(8);
            let zero = vec![ObstacleGpu::default(); new_cap as usize];
            self.obstacles_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("obstacles"),
                contents: bytemuck::cast_slice(&zero),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            self.obs_capacity = new_cap;
            // Recreate the SDF bind group to point at the new buffer.
            // (The pipeline layout is unchanged.)
        }
        let gpu: Vec<ObstacleGpu> = obstacles.iter().map(obstacle_to_gpu).collect();
        if !gpu.is_empty() {
            self.queue.write_buffer(&self.obstacles_buf, 0, bytemuck::cast_slice(&gpu));
        }
        self.num_obstacles = need;
    }

    pub fn dispatch_count(invocations: u32, group: u32) -> u32 {
        if invocations == 0 { 0 } else { (invocations + group - 1) / group }
    }

    // ── Public accessors used by ParticlePaperSimGpu ──────────────────────────

    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self)  -> &wgpu::Queue  { &self.queue }
    pub fn q_buffer(&self)     -> &wgpu::Buffer { &self.q_buf }
    pub fn q_prev_buffer(&self) -> &wgpu::Buffer { &self.q_prev_buf }
    pub fn w_inv_buffer(&self) -> &wgpu::Buffer { &self.w_inv_buf }
    pub fn n_particles(&self) -> u32 { self.n }

    // ── Per-substep encoding stages (composable from outer sims) ─────────────

    pub fn encode_predict(&self, enc: &mut wgpu::CommandEncoder, h: f32, g: f32, damping: f32) {
        let n = self.n;
        self.queue.write_buffer(&self.predict_u, 0,
            bytemuck::bytes_of(&PredictParams { h, g, damping, n }));
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("predict"), timestamp_writes: None });
        p.set_pipeline(&self.pl_predict);
        p.set_bind_group(0, &self.bg_predict, &[]);
        p.dispatch_workgroups(Self::dispatch_count(n, 256), 1, 1);
    }

    pub fn encode_copy_q_to_pred(&self, enc: &mut wgpu::CommandEncoder) {
        let n = self.n;
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("copy_q_to_pred"), timestamp_writes: None });
        p.set_pipeline(&self.pl_copy_pred);
        p.set_bind_group(0, &self.bg_copy_pred, &[]);
        p.dispatch_workgroups(Self::dispatch_count(n, 256), 1, 1);
    }

    pub fn encode_zero_lambdas(&self, enc: &mut wgpu::CommandEncoder) {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("zero_lambda"), timestamp_writes: None });
        p.set_pipeline(&self.pl_zero_f32);
        p.set_bind_group(0, &self.bg_zero_lambda_s, &[]);
        p.dispatch_workgroups(Self::dispatch_count(self.n_edges, 256), 1, 1);
        p.set_bind_group(0, &self.bg_zero_lambda_b, &[]);
        p.dispatch_workgroups(Self::dispatch_count(self.n_bends, 256), 1, 1);
    }

    /// Distance constraints, one color class at a time.
    pub fn encode_distance_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        h: f32,
        alpha: f32,
        stretch: bool,
    ) {
        let coloring = if stretch { &self.stretch_coloring } else { &self.bend_coloring };
        let bg_pair  = if stretch { &self.bg_distance_stretch } else { &self.bg_distance_bend };
        let us       = if stretch { &self.distance_us_stretch } else { &self.distance_us_bend };
        let cores    = if stretch { &self.bg_distance_core_stretch } else { &self.bg_distance_core_bend };
        let label    = if stretch { "distance_stretch" } else { "distance_bend" };

        for k in 0..coloring.num_colors() {
            let base = coloring.offsets[k];
            let size = coloring.offsets[k + 1] - base;
            if size == 0 { continue; }
            self.queue.write_buffer(&us[k], 0,
                bytemuck::bytes_of(&DistanceParams { h, alpha, n_color: size, base }));
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label), timestamp_writes: None });
            p.set_pipeline(&self.pl_distance);
            p.set_bind_group(0, &cores[k], &[]);
            p.set_bind_group(1, bg_pair, &[]);
            p.dispatch_workgroups(Self::dispatch_count(size, 64), 1, 1);
        }
    }

    pub fn encode_self_collision(
        &self,
        enc: &mut wgpu::CommandEncoder,
        h: f32,
        alpha_c: f32,
        mu: f32,
    ) {
        let n = self.n;
        let n_workgroups_n = Self::dispatch_count(n, 256);
        let inv_hash_h = 1.0 / (2.0 * self.r_max);
        // Zero cell_count + dq.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_hash_dq"), timestamp_writes: None });
            p.set_pipeline(&self.pl_zero_u32);
            p.set_bind_group(0, &self.bg_zero_cellcount, &[]);
            p.dispatch_workgroups(Self::dispatch_count(self.table_size, 256), 1, 1);
            p.set_bind_group(0, &self.bg_zero_dq, &[]);
            p.dispatch_workgroups(Self::dispatch_count(n * 3, 256), 1, 1);
        }
        self.queue.write_buffer(&self.hash_count_u, 0,
            bytemuck::bytes_of(&HashCountParams {
                inv_h: inv_hash_h, table_size: self.table_size, n, _p: 0,
            }));
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash_count"), timestamp_writes: None });
            p.set_pipeline(&self.pl_hash_count);
            p.set_bind_group(0, &self.bg_hash_count, &[]);
            p.dispatch_workgroups(n_workgroups_n, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash_scan"), timestamp_writes: None });
            p.set_pipeline(&self.pl_hash_scan);
            p.set_bind_group(0, &self.bg_hash_scan, &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash_scatter"), timestamp_writes: None });
            p.set_pipeline(&self.pl_hash_scatter);
            p.set_bind_group(0, &self.bg_hash_scatter, &[]);
            p.dispatch_workgroups(n_workgroups_n, 1, 1);
        }
        self.queue.write_buffer(&self.contact_u, 0,
            bytemuck::bytes_of(&ContactParams {
                h, alpha: alpha_c, mu, n,
                inv_h: inv_hash_h, ts: self.table_size,
                scale: FIXED_POINT_SCALE, _pad: 0,
            }));
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("contact_jacobi"), timestamp_writes: None });
            p.set_pipeline(&self.pl_contact);
            p.set_bind_group(0, &self.bg_contact_core, &[]);
            p.set_bind_group(1, &self.bg_contact_hash, &[]);
            p.set_bind_group(2, &self.bg_contact_dq, &[]);
            p.dispatch_workgroups(Self::dispatch_count(n, 64), 1, 1);
        }
        self.queue.write_buffer(&self.contact_apply_u, 0,
            bytemuck::bytes_of(&ContactApplyParams {
                n, inv_scale: 1.0 / FIXED_POINT_SCALE, _p0: 0, _p1: 0,
            }));
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("contact_apply"), timestamp_writes: None });
            p.set_pipeline(&self.pl_contact_apply);
            p.set_bind_group(0, &self.bg_contact_apply, &[]);
            p.dispatch_workgroups(n_workgroups_n, 1, 1);
        }
    }

    pub fn encode_sdf(&self, enc: &mut wgpu::CommandEncoder, alpha_c: f32, mu: f32) {
        if self.num_obstacles == 0 { return; }
        let n = self.n;
        self.queue.write_buffer(&self.sdf_u, 0,
            bytemuck::bytes_of(&SdfParams {
                alpha: alpha_c, mu, n, num_obs: self.num_obstacles,
            }));
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sdf_contact"), timestamp_writes: None });
        p.set_pipeline(&self.pl_sdf);
        p.set_bind_group(0, &self.bg_sdf, &[]);
        p.dispatch_workgroups(Self::dispatch_count(n, 256), 1, 1);
    }

    pub fn encode_pin_velocity(
        &self,
        enc: &mut wgpu::CommandEncoder,
        h: f32,
        pin_enabled: bool,
    ) {
        let n = self.n;
        let (mouse_idx, pin_flag) = match (pin_enabled, self.clicked_vertex) {
            (true, Some(vi)) if vi < n as usize => (vi as u32, 1u32),
            (true, _)                            => (u32::MAX, 1u32),
            _                                    => (u32::MAX, 0u32),
        };
        self.queue.write_buffer(&self.pin_vel_u, 0,
            bytemuck::bytes_of(&PinVelParams {
                inv_h: 1.0 / h, pin_enabled: pin_flag, mouse_idx, n,
                mouse_pos: [self.mouse_pos[0], self.mouse_pos[1], self.mouse_pos[2], 0.0],
            }));
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pin_velocity"), timestamp_writes: None });
        p.set_pipeline(&self.pl_pin_vel);
        p.set_bind_group(0, &self.bg_pin_vel, &[]);
        p.dispatch_workgroups(Self::dispatch_count(n, 256), 1, 1);
    }

    /// Encode the staging-buffer copy if a previous map isn't pending. Returns
    /// whether the copy was encoded — pass that to `finalize_submit`.
    pub fn encode_readback_kick(&self, enc: &mut wgpu::CommandEncoder) -> bool {
        let kick = !self.map_flags.lock().unwrap().inflight;
        if kick {
            enc.copy_buffer_to_buffer(&self.q_buf, 0, &self.staging_q, 0, (self.n as u64) * 16);
        }
        kick
    }

    /// Submit the encoder and arm the async map_async if `kicked`.
    pub fn finalize_submit(&mut self, enc: wgpu::CommandEncoder, kicked: bool) {
        self.queue.submit([enc.finish()]);
        if kicked {
            self.map_flags.lock().unwrap().inflight = true;
            let flags = self.map_flags.clone();
            self.staging_q.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                let mut f = flags.lock().unwrap();
                if res.is_ok() { f.ready = true; } else { f.inflight = false; }
            });
        }
    }

    /// Run one full `step` (n_substeps internal substeps) on the GPU.
    pub fn step_gpu(&mut self, params: &SimParams) {
        let dt = params.time_step as f32;
        let n_sub = params.num_substeps.max(1);
        let h = dt / n_sub as f32;
        let damping = params.damping as f32;
        let g = if params.gravity_enabled { params.gravity_g as f32 } else { 0.0 };

        let alpha_s = if params.stretch_enabled {
            (params.stretch_compliance as f32) / (h * h)
        } else { 1e30 };
        let alpha_b = if params.bending_enabled {
            (params.bend_compliance as f32) / (h * h)
        } else { 1e30 };
        let alpha_c = 0.0f32;
        let mu = if params.friction_enabled { params.friction_mu as f32 } else { 0.0 };

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("psim_gpu_step"),
        });

        for _sub in 0..n_sub {
            self.encode_predict(&mut enc, h, g, damping);
            self.encode_copy_q_to_pred(&mut enc);
            self.encode_zero_lambdas(&mut enc);
            if params.stretch_enabled {
                self.encode_distance_pass(&mut enc, h, alpha_s, /*stretch=*/ true);
            }
            if params.bending_enabled {
                self.encode_distance_pass(&mut enc, h, alpha_b, /*stretch=*/ false);
            }
            if params.self_collision_enabled {
                self.encode_self_collision(&mut enc, h, alpha_c, mu);
            }
            self.encode_sdf(&mut enc, alpha_c, mu);
            self.encode_pin_velocity(&mut enc, h, params.pin_enabled);
        }

        let kick = self.encode_readback_kick(&mut enc);
        self.finalize_submit(enc, kick);
    }

    /// Drain a completed staging map into `positions_cache`, then schedule a
    /// fresh readback. Call this once per frame *before* reading
    /// `positions()`. Non-blocking: if the previous map hasn't completed,
    /// the cache stays as-is.
    pub fn poll_readback(&mut self) {
        // Drain if ready.
        let drain = {
            let f = self.map_flags.lock().unwrap();
            f.inflight && f.ready
        };
        if drain {
            let n = self.n as usize;
            {
                let view = self.staging_q.slice(..).get_mapped_range();
                let raw: &[[f32; 4]] = bytemuck::cast_slice(&view);
                for i in 0..n {
                    self.positions_cache[(i, 0)] = raw[i][0];
                    self.positions_cache[(i, 1)] = raw[i][1];
                    self.positions_cache[(i, 2)] = raw[i][2];
                }
            }
            self.staging_q.unmap();
            let mut f = self.map_flags.lock().unwrap();
            f.inflight = false;
            f.ready    = false;
        }
        // Note: kicking the next map_async is now done inside step_gpu after
        // submit. poll_readback only drains completed maps.
    }
}

#[derive(Default)]
struct MapFlags {
    inflight: bool, // map_async called, awaiting GPU
    ready:    bool, // callback fired, mapped bytes available
}

impl MeshSim for ParticleClothSimGpu {
    fn step(&mut self, params: &SimParams)              { self.step_gpu(params); }
    fn positions(&self) -> &Positions                   { &self.positions_cache }
    fn set_clicked_vertex(&mut self, vi: Option<usize>) { self.clicked_vertex = vi; }
    fn set_mouse_pos(&mut self, pos: [f32; 3])          { self.mouse_pos = pos; }
}

