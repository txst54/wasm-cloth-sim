//! Particle-based cloth + rigid SDF collision under XPBD with substepping.
//!
//! Differences from `ClothSim`:
//!  - Self-collision is particle-vs-particle via a flat-array spatial hash,
//!    not vertex-vs-triangle / edge-vs-edge.
//!  - Rigid contact is sampled from analytic SDFs (sphere / plane / box),
//!    not triangle BVH.
//!  - Stretch and bending are *both* distance constraints (no shape matching).
//!    Bending uses the diamond diagonal `c↔d`.
//!
//! The XPBD loop is the standard Macklin form: many cheap substeps with a
//! single Gauss–Seidel pass each.

use std::collections::HashSet;

use nalgebra as na;

use crate::params::SimParams;

use super::shared::{
    apply_distance_constraint_xpbd,
    build_diamonds, build_edges, build_vertex_neighbors,
    Faces, Positions,
};
use super::particle_hash::ParticleHash;
use super::sdf::SdfObstacle;
use super::traits::MeshSim;

pub struct ParticleClothSim {
    pub q:        Positions,
    pub q_prev:   Positions,
    /// Rest (initial) positions, used to compute pairwise rest distance for
    /// self-contact thresholds (`d_coll = min(2r, d_rest)`).
    pub q_rest:   Positions,
    pub v:        Positions,
    pub w:        na::DVector<f32>,
    /// Per-particle contact radius.
    pub r:        Vec<f32>,
    pub r_max:    f32,

    pub faces:    Faces,
    pub edges:    Vec<[u32; 2]>,
    pub edge_rest:    Vec<f32>,
    /// Diamond `[a, b, c, d]` with bending constraint on `c↔d`.
    pub diamonds: Vec<[u32; 4]>,
    pub diamond_rest: Vec<f32>,

    pub stretch_lambda: Vec<f32>,
    pub bend_lambda:    Vec<f32>,

    /// Topological 1-ring per vertex. Used to skip adjacent particles in
    /// self-contact (otherwise structural neighbors would constantly collide).
    pub one_ring:   Vec<HashSet<u32>>,

    pub hash:       ParticleHash,
    pub obstacles:  Vec<SdfObstacle>,

    pub clicked_vertex: Option<usize>,
    pub mouse_pos:      [f32; 3],

    /// Resolution of the source grid (for renderer compatibility).
    pub resolution: usize,
}

impl ParticleClothSim {
    pub fn from_grid(resolution: usize, pinned: &[usize]) -> Self {
        let n = resolution;
        let num_verts = n * n;

        let mut q = Positions::zeros(num_verts);
        for row in 0..n {
            for col in 0..n {
                let i = row * n + col;
                let x = (col as f32 / (n - 1) as f32) * 2.0 - 1.0;
                let z = (row as f32 / (n - 1) as f32) * 2.0 - 1.0;
                // Lay the cloth horizontally above the sphere so it drapes under gravity.
                q[(i, 0)] = x * 0.9;
                q[(i, 1)] = 1.5;
                q[(i, 2)] = z * 0.9;
            }
        }
        let q_prev = q.clone();
        let q_rest = q.clone();
        let v = Positions::zeros(num_verts);

        let mut w = na::DVector::from_element(num_verts, 1.0f32);
        for &idx in pinned {
            if idx < num_verts { w[idx] = 0.0; }
        }

        let num_quads = (n - 1) * (n - 1);
        let mut faces = Faces::zeros(num_quads * 2);
        let mut fi = 0usize;
        for row in 0..(n - 1) {
            for col in 0..(n - 1) {
                let tl = (row * n + col) as u32;
                let tr = tl + 1;
                let bl = ((row + 1) * n + col) as u32;
                let br = bl + 1;
                faces[(fi, 0)] = tl; faces[(fi, 1)] = tr; faces[(fi, 2)] = br; fi += 1;
                faces[(fi, 0)] = tl; faces[(fi, 1)] = br; faces[(fi, 2)] = bl; fi += 1;
            }
        }

        let edges    = build_edges(&faces);
        let diamonds = build_diamonds(&faces);
        let one_ring = build_vertex_neighbors(&faces, num_verts);

        let edge_rest: Vec<f32> = edges.iter()
            .map(|&[a, b]| (q.row(a as usize) - q.row(b as usize)).norm())
            .collect();
        let diamond_rest: Vec<f32> = diamonds.iter()
            .map(|&[_, _, c, d]| (q.row(c as usize) - q.row(d as usize)).norm())
            .collect();

        let avg_edge = if edge_rest.is_empty() { 0.05 } else {
            edge_rest.iter().sum::<f32>() / edge_rest.len() as f32
        };
        // Particle radius = 0.45 * edge so that 1-ring neighbors at rest
        // (distance = edge) are not in contact (2r ≈ 0.9·edge).
        let r_val = 0.45 * avg_edge;
        let r = vec![r_val; num_verts];
        let r_max = r_val;

        let cell_size = 2.0 * r_max;
        let hash = ParticleHash::new(num_verts, cell_size);

        let stretch_lambda = vec![0.0f32; edges.len()];
        let bend_lambda    = vec![0.0f32; diamonds.len()];

        Self {
            q, q_prev, q_rest, v, w, r, r_max,
            faces, edges, edge_rest, diamonds, diamond_rest,
            stretch_lambda, bend_lambda, one_ring,
            hash, obstacles: Vec::new(),
            clicked_vertex: None, mouse_pos: [0.0; 3],
            resolution: n,
        }
    }

    pub fn add_obstacle(&mut self, o: SdfObstacle) { self.obstacles.push(o); }

    pub fn step(&mut self, params: &SimParams) {
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
        let alpha_c = 0.0f32; // hard contact

        let n = self.q.nrows();

        for _ in 0..n_sub {
            // Predict
            self.q_prev.copy_from(&self.q);
            for i in 0..n {
                if self.w[i] == 0.0 {
                    self.v[(i, 0)] = 0.0; self.v[(i, 1)] = 0.0; self.v[(i, 2)] = 0.0;
                    continue;
                }
                self.v[(i, 1)] += h * g;
                let d = (1.0 - damping).max(0.0);
                self.v[(i, 0)] *= d; self.v[(i, 1)] *= d; self.v[(i, 2)] *= d;
                self.q[(i, 0)] += h * self.v[(i, 0)];
                self.q[(i, 1)] += h * self.v[(i, 1)];
                self.q[(i, 2)] += h * self.v[(i, 2)];
            }

            for l in self.stretch_lambda.iter_mut() { *l = 0.0; }
            for l in self.bend_lambda.iter_mut()    { *l = 0.0; }

            // 1) Distance constraints — stretch (edges)
            if params.stretch_enabled {
                for (ei, &[a, b]) in self.edges.iter().enumerate() {
                    apply_distance_constraint_xpbd(
                        &mut self.q, &self.w,
                        a as usize, b as usize,
                        self.edge_rest[ei], alpha_s,
                        &mut self.stretch_lambda[ei],
                    );
                }
            }

            // 2) Distance constraints — bending (diamond diagonals)
            if params.bending_enabled {
                for (di, &[_, _, c, d]) in self.diamonds.iter().enumerate() {
                    apply_distance_constraint_xpbd(
                        &mut self.q, &self.w,
                        c as usize, d as usize,
                        self.diamond_rest[di], alpha_b,
                        &mut self.bend_lambda[di],
                    );
                }
            }

            // 3) Self-collision via particle hash (inequality)
            if params.self_collision_enabled {
                self.hash.set_cell_size(2.0 * self.r_max);
                self.hash.rebuild(&self.q);
                self.project_self_contact(alpha_c);
            }

            // 4) Rigid SDF contact
            if !self.obstacles.is_empty() {
                self.project_sdf_contact(alpha_c);
            }

            // 5) Pin constraint via mouse drag (snap clicked vertex to mouse_pos)
            if params.pin_enabled {
                if let Some(vi) = self.clicked_vertex {
                    if vi < n {
                        self.q[(vi, 0)] = self.mouse_pos[0];
                        self.q[(vi, 1)] = self.mouse_pos[1];
                        self.q[(vi, 2)] = self.mouse_pos[2];
                    }
                }
                // pinned (w=0) particles are kept at their initial position by predict
                for i in 0..n {
                    if self.w[i] == 0.0 {
                        self.q[(i, 0)] = self.q_prev[(i, 0)];
                        self.q[(i, 1)] = self.q_prev[(i, 1)];
                        self.q[(i, 2)] = self.q_prev[(i, 2)];
                    }
                }
            }

            // 6) Update velocities
            let inv_h = 1.0 / h;
            for i in 0..n {
                if self.w[i] == 0.0 { continue; }
                self.v[(i, 0)] = (self.q[(i, 0)] - self.q_prev[(i, 0)]) * inv_h;
                self.v[(i, 1)] = (self.q[(i, 1)] - self.q_prev[(i, 1)]) * inv_h;
                self.v[(i, 2)] = (self.q[(i, 2)] - self.q_prev[(i, 2)]) * inv_h;
            }
        }
    }

    fn project_self_contact(&mut self, alpha: f32) {
        let n = self.q.nrows();
        let mut neigh: Vec<u32> = Vec::with_capacity(64);
        for i in 0..n {
            let pi = [self.q[(i, 0)], self.q[(i, 1)], self.q[(i, 2)]];
            let ri = self.r[i];
            neigh.clear();
            self.hash.for_each_neighbor(pi, |j| { neigh.push(j); });
            for &ju in neigh.iter() {
                let j = ju as usize;
                if j <= i { continue; }
                if self.one_ring[i].contains(&(j as u32)) { continue; }
                let dx = self.q[(j, 0)] - self.q[(i, 0)];
                let dy = self.q[(j, 1)] - self.q[(i, 1)];
                let dz = self.q[(j, 2)] - self.q[(i, 2)];
                // d_coll = min(2r, d_rest) so contact threshold never exceeds
                // the pair's rest distance — otherwise distance constraints
                // and contact constraints fight each other.
                let r_sum = ri + self.r[j];
                let rxd = self.q_rest[(j, 0)] - self.q_rest[(i, 0)];
                let ryd = self.q_rest[(j, 1)] - self.q_rest[(i, 1)];
                let rzd = self.q_rest[(j, 2)] - self.q_rest[(i, 2)];
                let d_rest = (rxd * rxd + ryd * ryd + rzd * rzd).sqrt();
                let d_coll = r_sum.min(d_rest);
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 >= d_coll * d_coll || d2 < 1e-12 { continue; }
                let d = d2.sqrt();
                let c_val = d - d_coll; // < 0
                let wi = self.w[i]; let wj = self.w[j];
                let denom = wi + wj + alpha;
                if denom < 1e-12 { continue; }
                let dl = -c_val / denom;
                let inv_d = 1.0 / d;
                let nx = dx * inv_d; let ny = dy * inv_d; let nz = dz * inv_d;
                if wi > 0.0 {
                    self.q[(i, 0)] -= wi * dl * nx;
                    self.q[(i, 1)] -= wi * dl * ny;
                    self.q[(i, 2)] -= wi * dl * nz;
                }
                if wj > 0.0 {
                    self.q[(j, 0)] += wj * dl * nx;
                    self.q[(j, 1)] += wj * dl * ny;
                    self.q[(j, 2)] += wj * dl * nz;
                }
            }
        }
    }

    fn project_sdf_contact(&mut self, alpha: f32) {
        let n = self.q.nrows();
        for i in 0..n {
            let wi = self.w[i];
            if wi == 0.0 { continue; }
            let pi = na::Vector3::new(self.q[(i, 0)], self.q[(i, 1)], self.q[(i, 2)]);
            for obs in &self.obstacles {
                let (d, n_world) = obs.query(pi);
                let phi = d - self.r[i];
                if phi >= 0.0 { continue; }
                let denom = wi + alpha;
                if denom < 1e-12 { continue; }
                let dl = -phi / denom;
                self.q[(i, 0)] += wi * dl * n_world.x;
                self.q[(i, 1)] += wi * dl * n_world.y;
                self.q[(i, 2)] += wi * dl * n_world.z;
            }
        }
    }
}

impl MeshSim for ParticleClothSim {
    fn step(&mut self, params: &SimParams)              { ParticleClothSim::step(self, params); }
    fn positions(&self) -> &Positions                   { &self.q }
    fn set_clicked_vertex(&mut self, vi: Option<usize>) { self.clicked_vertex = vi; }
    fn set_mouse_pos(&mut self, pos: [f32; 3])          { self.mouse_pos = pos; }
}
