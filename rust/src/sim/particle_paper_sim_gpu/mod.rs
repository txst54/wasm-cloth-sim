//! GPU compute path for `ParticlePaperSim`. WASM-only.
//!
//! Composes a `ParticleClothSimGpu` (which owns the per-particle state and the
//! cloth/distance/contact pipelines) with a dihedral-hinge pipeline. Hinges are
//! greedily graph-colored on their 4-tuple endpoints so each color class is an
//! independent set; the kernel can then write all four endpoint slots without
//! atomics.
//!
//! Per-substep encoding order (mirrors the CPU `ParticlePaperSim::step`):
//!   predict → copy_q_to_pred → zero λ (cloth + hinges)
//!   → distance(stretch) → distance(bend) → dihedral
//!   → self_collision → sdf → pin/velocity.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::params::SimParams;

use super::particle_cloth_sim_gpu::coloring::{color_constraints, Coloring};
use super::particle_cloth_sim_gpu::ParticleClothSimGpu;
use super::particle_paper_sim::{ParticleHinge, ParticlePaperSim};
use super::shared::Positions;
use super::traits::MeshSim;
use super::FoldDirection;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct DihedralParams { h_dt: f32, n_color: u32, base: u32, _pad: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct HingeAbcd { abcd: [u32; 4] }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct HingeMeta {
    compliance: f32,
    damping:    f32,
    goal_angle: f32,
    _pad:       f32,
}

fn make_pipeline(
    device: &wgpu::Device,
    label:  &str,
    bgls:   &[&wgpu::BindGroupLayout],
    src:    &str,
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
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

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

pub struct ParticlePaperSimGpu {
    pub cloth: ParticleClothSimGpu,

    // CPU-side hinge state. Lives here (not on cloth) so we can rate-limit
    // current_angle and re-upload meta when fold/compliance/damping changes.
    pub hinges: Vec<ParticleHinge>,
    pub fold_speed: f32,

    n_hinges: u32,
    coloring: Coloring,

    hinges_buf:    wgpu::Buffer,
    meta_buf:      wgpu::Buffer,
    lambda_buf:    wgpu::Buffer,
    color_idx_buf: wgpu::Buffer,

    dihedral_us: Vec<wgpu::Buffer>,             // one per color
    bg_core_per_color: Vec<wgpu::BindGroup>,    // one per color (each binds its own uniform)
    bg_pair: wgpu::BindGroup,
    pl_dihedral: wgpu::ComputePipeline,
}

impl ParticlePaperSimGpu {
    pub fn from_cpu(
        device: wgpu::Device,
        queue:  wgpu::Queue,
        cpu:    &ParticlePaperSim,
    ) -> Self {
        let cloth = ParticleClothSimGpu::from_cpu(device.clone(), queue.clone(), &cpu.core);
        let n_verts = cpu.core.q.nrows();
        let n_hinges = cpu.hinges.len() as u32;

        // 4-tuple coloring.
        let endpoint_storage: Vec<[u32; 4]> = cpu.hinges.iter()
            .map(|h| [h.a, h.b, h.c, h.d])
            .collect();
        let endpoint_refs: Vec<&[u32]> = endpoint_storage.iter().map(|t| &t[..]).collect();
        let coloring = color_constraints(&endpoint_refs, n_verts);
        #[cfg(debug_assertions)]
        super::particle_cloth_sim_gpu::coloring::validate(&coloring, &endpoint_refs, n_verts);

        // Initial buffers.
        let hinges_data: Vec<HingeAbcd> = cpu.hinges.iter()
            .map(|h| HingeAbcd { abcd: [h.a, h.b, h.c, h.d] })
            .collect();
        let meta_data: Vec<HingeMeta> = cpu.hinges.iter()
            .map(|h| HingeMeta {
                compliance: h.compliance, damping: h.damping,
                goal_angle: h.current_angle, _pad: 0.0,
            })
            .collect();
        // Always allocate at least 1 slot to keep buffers non-empty.
        let pad_hinges = if hinges_data.is_empty() { vec![HingeAbcd::default()] } else { hinges_data.clone() };
        let pad_meta   = if meta_data.is_empty()   { vec![HingeMeta::default()]   } else { meta_data.clone() };
        let pad_idx: Vec<u32> = if coloring.idx.is_empty() { vec![0u32] } else { coloring.idx.clone() };

        let mk_storage = |label: &str, bytes: &[u8], extra_dst: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE;
            if extra_dst { usage |= wgpu::BufferUsages::COPY_DST; }
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytes, usage,
            })
        };

        let hinges_buf    = mk_storage("hinges",    bytemuck::cast_slice(&pad_hinges), false);
        let meta_buf      = mk_storage("hinge_meta", bytemuck::cast_slice(&pad_meta),  true);
        let lambda_buf    = mk_storage("hinge_lambda",
            bytemuck::cast_slice(&vec![0f32; pad_hinges.len()]), true);
        let color_idx_buf = mk_storage("hinge_color_idx", bytemuck::cast_slice(&pad_idx), false);

        // BGLs.
        let bgl_core = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_dihedral_core"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, false), // q
                storage_entry(2, true),  // q_prev
                storage_entry(3, true),  // w_inv
            ],
        });
        let bgl_pair = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_dihedral_pair"),
            entries: &[
                storage_entry(0, true),  // hinges
                storage_entry(1, true),  // meta
                storage_entry(2, false), // lambda
                storage_entry(3, true),  // color_idx
            ],
        });

        let pl_dihedral = make_pipeline(&device, "pl_dihedral",
            &[&bgl_core, &bgl_pair], include_str!("shaders/dihedral.wgsl"));

        // Per-color uniform buffers + core bind groups.
        let n_colors = coloring.num_colors().max(1);
        let dihedral_us: Vec<wgpu::Buffer> = (0..n_colors)
            .map(|k| device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dihedral_u_{}", k)),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
            .collect();

        let mk_bg = |label: &str, layout: &wgpu::BindGroupLayout, entries: Vec<wgpu::BindGroupEntry>| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label), layout, entries: &entries,
            })
        };
        fn entry(b: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() }
        }

        let bg_core_per_color: Vec<wgpu::BindGroup> = dihedral_us.iter().enumerate()
            .map(|(k, u)| mk_bg(&format!("bg_dihedral_core_{}", k), &bgl_core, vec![
                entry(0, u),
                entry(1, cloth.q_buffer()),
                entry(2, cloth.q_prev_buffer()),
                entry(3, cloth.w_inv_buffer()),
            ]))
            .collect();

        let bg_pair = mk_bg("bg_dihedral_pair", &bgl_pair, vec![
            entry(0, &hinges_buf),
            entry(1, &meta_buf),
            entry(2, &lambda_buf),
            entry(3, &color_idx_buf),
        ]);

        Self {
            cloth,
            hinges: cpu.hinges.clone(),
            fold_speed: cpu.fold_speed,
            n_hinges,
            coloring,
            hinges_buf, meta_buf, lambda_buf, color_idx_buf,
            dihedral_us, bg_core_per_color, bg_pair, pl_dihedral,
        }
    }

    /// Re-upload `HingeMeta` for all hinges. Call after fold-angle / compliance
    /// / damping setters mutate `self.hinges`.
    pub fn upload_meta(&self) {
        if self.n_hinges == 0 { return; }
        let meta: Vec<HingeMeta> = self.hinges.iter()
            .map(|h| HingeMeta {
                compliance: h.compliance, damping: h.damping,
                goal_angle: h.current_angle, _pad: 0.0,
            })
            .collect();
        self.cloth.queue().write_buffer(&self.meta_buf, 0, bytemuck::cast_slice(&meta));
    }

    /// Update target_angle for every hinge from a single fold-amount input.
    pub fn set_fold_angle(&mut self, degrees: f32) {
        let amt = degrees.to_radians();
        for h in self.hinges.iter_mut() {
            h.target_angle = match h.direction {
                FoldDirection::Mountain => -amt,
                FoldDirection::Valley   =>  amt,
            };
        }
    }

    pub fn set_hinge_compliance(&mut self, alpha: f32) {
        for h in self.hinges.iter_mut() {
            h.compliance = alpha / h.rest_edge_len.max(1e-12);
        }
        self.upload_meta();
    }
    pub fn set_hinge_damping(&mut self, beta: f32) {
        for h in self.hinges.iter_mut() { h.damping = beta; }
        self.upload_meta();
    }
    pub fn set_fold_speed(&mut self, rps: f32) { self.fold_speed = rps; }

    /// Run one full `step` on the GPU.
    pub fn step_gpu(&mut self, params: &SimParams) {
        // Rate-limit current_angle (one tick per outer step, like PaperSim).
        let dt = params.time_step as f32;
        let max_delta = self.fold_speed * dt;
        for h in self.hinges.iter_mut() {
            let delta = (h.target_angle - h.current_angle).clamp(-max_delta, max_delta);
            h.current_angle += delta;
        }
        self.upload_meta();

        let n_sub = params.num_substeps.max(1);
        let h_dt = dt / n_sub as f32;
        let damping = params.damping as f32;
        let g = if params.gravity_enabled { params.gravity_g as f32 } else { 0.0 };

        let alpha_s = if params.stretch_enabled {
            (params.stretch_compliance as f32) / (h_dt * h_dt)
        } else { 1e30 };
        let alpha_b = if params.bending_enabled {
            (params.bend_compliance as f32) / (h_dt * h_dt)
        } else { 1e30 };
        let alpha_c = 0.0f32;
        let mu = if params.friction_enabled { params.friction_mu as f32 } else { 0.0 };

        let mut enc = self.cloth.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ppaper_gpu_step"),
        });

        for _sub in 0..n_sub {
            self.cloth.encode_predict(&mut enc, h_dt, g, damping);
            self.cloth.encode_copy_q_to_pred(&mut enc);
            self.cloth.encode_zero_lambdas(&mut enc);
            // Zero hinge lambdas via queue.write_buffer (small).
            if self.n_hinges > 0 {
                let zeros = vec![0f32; self.n_hinges as usize];
                self.cloth.queue().write_buffer(&self.lambda_buf, 0, bytemuck::cast_slice(&zeros));
            }

            if params.stretch_enabled {
                self.cloth.encode_distance_pass(&mut enc, h_dt, alpha_s, /*stretch=*/ true);
            }
            if params.bending_enabled {
                self.cloth.encode_distance_pass(&mut enc, h_dt, alpha_b, /*stretch=*/ false);
            }
            if self.n_hinges > 0 {
                self.encode_dihedral(&mut enc, h_dt);
            }
            if params.self_collision_enabled {
                self.cloth.encode_self_collision(&mut enc, h_dt, alpha_c, mu);
            }
            self.cloth.encode_sdf(&mut enc, alpha_c, mu);
            self.cloth.encode_pin_velocity(&mut enc, h_dt, params.pin_enabled);
        }

        let kick = self.cloth.encode_readback_kick(&mut enc);
        self.cloth.finalize_submit(enc, kick);
    }

    fn encode_dihedral(&self, enc: &mut wgpu::CommandEncoder, h_dt: f32) {
        for k in 0..self.coloring.num_colors() {
            let base = self.coloring.offsets[k];
            let size = self.coloring.offsets[k + 1] - base;
            if size == 0 { continue; }
            self.cloth.queue().write_buffer(&self.dihedral_us[k], 0,
                bytemuck::bytes_of(&DihedralParams {
                    h_dt, n_color: size, base, _pad: 0,
                }));
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dihedral"), timestamp_writes: None });
            p.set_pipeline(&self.pl_dihedral);
            p.set_bind_group(0, &self.bg_core_per_color[k], &[]);
            p.set_bind_group(1, &self.bg_pair, &[]);
            p.dispatch_workgroups(ParticleClothSimGpu::dispatch_count(size, 64), 1, 1);
        }
    }

    pub fn poll_readback(&mut self) { self.cloth.poll_readback(); }
}

impl MeshSim for ParticlePaperSimGpu {
    fn step(&mut self, params: &SimParams)              { self.step_gpu(params); }
    fn positions(&self) -> &Positions                   { self.cloth.positions() }
    fn set_clicked_vertex(&mut self, vi: Option<usize>) { self.cloth.set_clicked_vertex(vi); }
    fn set_mouse_pos(&mut self, pos: [f32; 3])          { self.cloth.set_mouse_pos(pos); }
}
