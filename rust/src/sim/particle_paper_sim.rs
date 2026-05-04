//! Particle-based paper simulation.
//!
//! Composes a `ParticleClothSim` (XPBD distance + self-collision + SDF) with
//! the dihedral hinge / fold-speed machinery from `PaperSim`. Hinge edges are
//! removed from the cloth's bend distance constraint set at construction time
//! so the dihedral constraint is the sole driver of fold dynamics.

use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};

use crate::params::SimParams;
use crate::{platform_log, platform_warn};

use super::crease::{CreasePattern, CreaseType, find_edges_on_creases, find_overlapping_edges};
use super::paper_sim::{apply_hinge_xpbd, dihedral_angle, normalise_edge};
use super::particle_cloth_sim::ParticleClothSim;
use super::shared::{apply_distance_constraint_xpbd, build_diamonds, build_edges, Faces, Positions};
use super::traits::MeshSim;
use super::{FoldDirection, FoldSpec};

/// Standalone hinge that owns its 4 vertex indices (no dependency on a
/// `ClothSimCore.diamonds` slot, since we filter those for the cloth bend pass).
#[derive(Clone, Debug)]
pub struct ParticleHinge {
    pub a: u32, pub b: u32, pub c: u32, pub d: u32,
    pub rest_angle:    f32,
    pub target_angle:  f32,
    pub current_angle: f32,
    pub compliance:    f32,
    pub rest_edge_len: f32,
    pub direction:     FoldDirection,
    pub damping:       f32,
    pub lambda:        f32,
}

pub struct ParticlePaperSim {
    pub core: ParticleClothSim,
    pub hinges: Vec<ParticleHinge>,
    pub fold_speed: f32,
    pub crease_chains: Vec<Vec<u32>>,
}

impl Deref for ParticlePaperSim {
    type Target = ParticleClothSim;
    fn deref(&self) -> &ParticleClothSim { &self.core }
}
impl DerefMut for ParticlePaperSim {
    fn deref_mut(&mut self) -> &mut ParticleClothSim { &mut self.core }
}

impl ParticlePaperSim {
    /// Simple square paper grid with no creases registered.
    pub fn from_grid(resolution: usize) -> Self {
        let core = ParticleClothSim::from_grid(resolution, &[]);
        Self {
            core,
            hinges: Vec::new(),
            fold_speed: 5.0,
            crease_chains: Vec::new(),
        }
    }

    /// Build from a crease pattern. Mirrors `PaperSim::from_crease_pattern`
    /// but constructs a `ParticleClothSim` for the substrate and stores
    /// hinges as standalone `ParticleHinge`s.
    ///
    /// Returns (sim, positions, faces, vertex_colors, edge_colors) so the
    /// renderer can build a Cloth mesh.
    pub fn from_crease_pattern(
        cp: &CreasePattern,
        resolution: usize,
    ) -> (Self, Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<[f32; 3]>, HashMap<(u32, u32), CreaseType>) {
        let (positions, faces, fold_edges, crease_chains, split_creases) = cp.build_mesh(resolution);

        // Pin top-right corner.
        let mut pin_idx = 0usize;
        let mut best_dist = f32::MAX;
        for (i, &[x, y, _]) in positions.iter().enumerate() {
            let d = (x - 0.9).abs() + (y - 0.9).abs();
            if d < best_dist { best_dist = d; pin_idx = i; }
        }

        // Build a Faces matrix for diamond / overlap analysis.
        let mut faces_mat = Faces::zeros(faces.len());
        for (fi, f) in faces.iter().enumerate() {
            faces_mat[(fi, 0)] = f[0]; faces_mat[(fi, 1)] = f[1]; faces_mat[(fi, 2)] = f[2];
        }
        // Build the *full* (unfiltered) diamond list once so we can map fold
        // edges → diamond [a,b,c,d] for hinge construction.
        let diamonds_all = build_diamonds(&faces_mat);
        let edge_to_diamond: HashMap<(u32, u32), [u32; 4]> = diamonds_all.iter()
            .map(|&[a, b, c, d]| (normalise_edge(a, b), [a, b, c, d]))
            .collect();

        // Build a positions matrix for dihedral_angle().
        let mut q_for_angle = Positions::zeros(positions.len());
        for (i, p) in positions.iter().enumerate() {
            q_for_angle[(i, 0)] = p[0]; q_for_angle[(i, 1)] = p[1]; q_for_angle[(i, 2)] = p[2];
        }

        // Diagnostics matching PaperSim.
        let edges_for_check: Vec<[u32; 2]> = build_edges(&faces_mat);
        let overlapping = find_overlapping_edges(&positions, &edges_for_check, 1e-6);
        if !overlapping.is_empty() {
            platform_warn!("WARNING: Found {} overlapping edge pairs in mesh!", overlapping.len());
        }
        let edges_on_creases = find_edges_on_creases(&positions, &edges_for_check, &split_creases, 1e-6);

        // Merge fold_edges (CDT) with edges_on_creases (geometric).
        let mut all_fold_edges: HashMap<(u32, u32), CreaseType> = fold_edges.clone();
        let mut extra_edges = 0usize;
        for (edge, crease_type) in &edges_on_creases {
            if !all_fold_edges.contains_key(edge) {
                all_fold_edges.insert(*edge, *crease_type);
                extra_edges += 1;
            }
        }
        if extra_edges > 0 {
            platform_log!("Found {} additional edges lying on crease lines", extra_edges);
        }

        // Vertex colors.
        let mut vertex_colors = vec![[0.85f32, 0.85, 0.85]; positions.len()];
        // for (&(a, b), &crease_type) in &all_fold_edges {
        //     let color = match crease_type {
        //         CreaseType::Mountain => [1.0, 0.6, 0.6],
        //         CreaseType::Valley   => [0.6, 0.6, 1.0],
        //         CreaseType::Boundary => [0.85, 0.85, 0.85],
        //     };
        //     vertex_colors[a as usize] = color;
        //     vertex_colors[b as usize] = color;
        // }

        // Skip set for cloth bending: any fold edge that has a corresponding diamond.
        let skip_bend: HashSet<(u32, u32)> = all_fold_edges.keys()
            .filter(|k| edge_to_diamond.contains_key(k))
            .copied()
            .collect();

        let core = ParticleClothSim::from_mesh(&positions, &faces, &[pin_idx], &skip_bend);

        // Build crease chains as u32 (already given by build_mesh).
        let crease_chains_owned = crease_chains;

        let mut sim = Self {
            core,
            hinges: Vec::new(),
            fold_speed: 5.0,
            crease_chains: crease_chains_owned,
        };

        // Build hinges from all_fold_edges that have a diamond.
        let edge_colors = all_fold_edges.clone();
        let mut fold_map: HashMap<(u32, u32), FoldSpec> = HashMap::new();
        for ((a, b), crease_type) in all_fold_edges {
            let direction = match crease_type {
                CreaseType::Mountain => FoldDirection::Mountain,
                CreaseType::Valley   => FoldDirection::Valley,
                CreaseType::Boundary => continue,
            };
            fold_map.insert((a, b), FoldSpec {
                target_angle: 0.0,
                compliance: 1e-4,
                direction,
                damping: 0.5,
            });
        }
        sim.set_fold_map(fold_map, &edge_to_diamond, &q_for_angle);

        (sim, positions, faces, vertex_colors, edge_colors)
    }

    /// Register hinges from a fold map, using the supplied `edge_to_diamond`
    /// (built from the *full* face mesh, not the cloth's filtered diamonds).
    pub fn set_fold_map(
        &mut self,
        fold_map:        HashMap<(u32, u32), FoldSpec>,
        edge_to_diamond: &HashMap<(u32, u32), [u32; 4]>,
        q_rest:          &Positions,
    ) {
        self.hinges.clear();
        let mut dropped = 0usize;
        for ((ea, eb), spec) in fold_map {
            let key = normalise_edge(ea, eb);
            if let Some(&[a, b, c, d]) = edge_to_diamond.get(&key) {
                let rest_angle = dihedral_angle(q_rest, a, b, c, d);
                let edge_len = (q_rest.row(b as usize) - q_rest.row(a as usize)).norm();
                self.hinges.push(ParticleHinge {
                    a, b, c, d,
                    rest_angle,
                    target_angle: spec.target_angle,
                    current_angle: 0.0,
                    compliance: spec.compliance / edge_len.max(1e-12),
                    rest_edge_len: edge_len,
                    direction: spec.direction,
                    damping: spec.damping,
                    lambda: 0.0,
                });
            } else {
                dropped += 1;
            }
        }
        if dropped > 0 {
            platform_warn!("ParticlePaperSim: dropped {} fold edges with no diamond", dropped);
        }
        let m = self.hinges.iter().filter(|h| h.direction == FoldDirection::Mountain).count();
        let v = self.hinges.iter().filter(|h| h.direction == FoldDirection::Valley).count();
        platform_log!(
            "ParticlePaperSim hinges: {} total ({} mountain, {} valley)",
            self.hinges.len(), m, v
        );
    }

    /// Set the desired fold amount in degrees, applied per-hinge with
    /// direction sign. 0 = flat.
    pub fn set_fold_angle(&mut self, angle_degrees: f32) {
        let fold_amount = angle_degrees.to_radians();
        for h in self.hinges.iter_mut() {
            h.target_angle = match h.direction {
                FoldDirection::Mountain => -fold_amount,
                FoldDirection::Valley   =>  fold_amount,
            };
        }
    }

    pub fn step(&mut self, params: &SimParams) {
        let dt = params.time_step as f32;
        let max_delta = self.fold_speed * dt;
        for h in self.hinges.iter_mut() {
            let delta = (h.target_angle - h.current_angle).clamp(-max_delta, max_delta);
            h.current_angle += delta;
        }

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

        let core = &mut self.core;
        let n = core.q.nrows();

        for _ in 0..n_sub {
            // Predict
            core.q_prev.copy_from(&core.q);
            for i in 0..n {
                if core.w[i] == 0.0 {
                    core.v[(i, 0)] = 0.0; core.v[(i, 1)] = 0.0; core.v[(i, 2)] = 0.0;
                    continue;
                }
                core.v[(i, 1)] += h_dt * g;
                let d = (1.0 - damping).max(0.0);
                core.v[(i, 0)] *= d; core.v[(i, 1)] *= d; core.v[(i, 2)] *= d;
                core.q[(i, 0)] += h_dt * core.v[(i, 0)];
                core.q[(i, 1)] += h_dt * core.v[(i, 1)];
                core.q[(i, 2)] += h_dt * core.v[(i, 2)];
            }
            core.q_pred.copy_from(&core.q);
            core.contact_pairs.clear();

            for l in core.stretch_lambda.iter_mut() { *l = 0.0; }
            for l in core.bend_lambda.iter_mut()    { *l = 0.0; }
            for h in self.hinges.iter_mut() { h.lambda = 0.0; }

            // Stretch
            if params.stretch_enabled {
                for (ei, &[a, b]) in core.edges.iter().enumerate() {
                    apply_distance_constraint_xpbd(
                        &mut core.q, &core.w,
                        a as usize, b as usize,
                        core.edge_rest[ei], alpha_s,
                        &mut core.stretch_lambda[ei],
                    );
                }
            }
            // Bend (cloth, on filtered diamonds — fold edges already excluded)
            if params.bending_enabled {
                for (di, &[_, _, c, d]) in core.diamonds.iter().enumerate() {
                    apply_distance_constraint_xpbd(
                        &mut core.q, &core.w,
                        c as usize, d as usize,
                        core.diamond_rest[di], alpha_b,
                        &mut core.bend_lambda[di],
                    );
                }
            }

            // Hinge dihedral XPBD
            for hin in self.hinges.iter_mut() {
                let alpha_tilde = hin.compliance / (h_dt * h_dt);
                let gamma       = hin.compliance * hin.damping / h_dt;
                let goal_dihedral = hin.current_angle;
                apply_hinge_xpbd(
                    &mut core.q, &core.q_prev, &core.w,
                    hin.a as usize, hin.b as usize, hin.c as usize, hin.d as usize,
                    goal_dihedral, alpha_tilde, gamma, &mut hin.lambda,
                );
            }

            // Self-collision
            if params.self_collision_enabled {
                core.hash.set_cell_size(2.0 * core.r_max);
                core.hash.rebuild(&core.q);
                core.project_self_contact(alpha_c, mu);
            }
            if !core.obstacles.is_empty() {
                core.project_sdf_contact(alpha_c, mu);
            }

            // Pins
            if params.pin_enabled {
                if let Some(vi) = core.clicked_vertex {
                    if vi < n {
                        core.q[(vi, 0)] = core.mouse_pos[0];
                        core.q[(vi, 1)] = core.mouse_pos[1];
                        core.q[(vi, 2)] = core.mouse_pos[2];
                    }
                }
                for i in 0..n {
                    if core.w[i] == 0.0 {
                        core.q[(i, 0)] = core.q_prev[(i, 0)];
                        core.q[(i, 1)] = core.q_prev[(i, 1)];
                        core.q[(i, 2)] = core.q_prev[(i, 2)];
                    }
                }
            }

            // Velocity update
            let inv_h = 1.0 / h_dt;
            for i in 0..n {
                if core.w[i] == 0.0 { continue; }
                core.v[(i, 0)] = (core.q[(i, 0)] - core.q_prev[(i, 0)]) * inv_h;
                core.v[(i, 1)] = (core.q[(i, 1)] - core.q_prev[(i, 1)]) * inv_h;
                core.v[(i, 2)] = (core.q[(i, 2)] - core.q_prev[(i, 2)]) * inv_h;
            }
        }

    }
}

impl MeshSim for ParticlePaperSim {
    fn step(&mut self, params: &SimParams)              { ParticlePaperSim::step(self, params); }
    fn positions(&self) -> &Positions                   { &self.core.q }
    fn set_clicked_vertex(&mut self, vi: Option<usize>) { self.core.clicked_vertex = vi; }
    fn set_mouse_pos(&mut self, pos: [f32; 3])          { self.core.mouse_pos = pos; }
}
