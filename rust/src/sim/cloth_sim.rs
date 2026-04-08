use std::collections::HashMap;

use nalgebra as na;

use crate::{cloth::Cloth, gpu::GpuContext, params::SimParams};

/// Nx3 matrix of vertex positions / velocities.
pub type Positions = na::OMatrix<f32, na::Dyn, na::Const<3>>;

/// Mx3 matrix of triangle vertex indices (matches cloth.rs triangulation).
pub type Faces = na::OMatrix<u32, na::Dyn, na::Const<3>>;

// ── Diamond builder ───────────────────────────────────────────────────────────

/// For each interior edge shared by two triangles, collect the 4 vertices
/// (the 2 edge endpoints + 1 opposite vertex from each triangle).
fn build_diamonds(faces: &Faces) -> Vec<[u32; 4]> {
    let m = faces.nrows();
    // edge (min, max) → [(face_idx, opposite_vertex), ...]
    let mut edge_map: HashMap<(u32, u32), [(u32, u32); 2]> = HashMap::with_capacity(m * 3);
    let mut counts: HashMap<(u32, u32), u8> = HashMap::with_capacity(m * 3);

    for fi in 0..m {
        let v = [faces[(fi, 0)], faces[(fi, 1)], faces[(fi, 2)]];
        for e in 0..3usize {
            let a   = v[e];
            let b   = v[(e + 1) % 3];
            let opp = v[(e + 2) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            let cnt = counts.entry(key).or_insert(0);
            if *cnt == 0 {
                edge_map.insert(key, [(opp, 0), (0, 0)]);
            } else if *cnt == 1 {
                edge_map.get_mut(&key).unwrap()[1] = (opp, 0);
            }
            *cnt += 1;
        }
    }

    edge_map
        .into_iter()
        .filter(|&(ref k, _)| *counts.get(k).unwrap() == 2)
        .map(|((a, b), adj)| [a, b, adj[0].0, adj[1].0])
        .collect()
}

// ── Shape-matching constraint (Müller et al.) ─────────────────────────────────
//
// Finds the best-fit rotation R between the current cluster and the rest
// cluster, then blends each vertex toward its rotated rest target.
//
// Optimization: M = Σ aᵢ bᵢᵀ is built in one pass (no A/B matrices),
// keeping extra allocation to zero regardless of cluster size.

fn apply_constraint(q: &mut Positions, q_rest: &Positions, idx: &[u32], weight: f32) {
    let n = idx.len();
    let inv_n = 1.0 / n as f32;

    // Centroids
    let mut c  = na::Vector3::<f32>::zeros();
    let mut c0 = na::Vector3::<f32>::zeros();
    for &i in idx {
        c  += q.row(i as usize).transpose();
        c0 += q_rest.row(i as usize).transpose();
    }
    c  *= inv_n;
    c0 *= inv_n;

    // M = AB^T, cols of A contain xi - c, cols of B contain xi0 - c0
    let mut m = na::Matrix3::<f32>::zeros();
    for &i in idx {
        let a = q.row(i as usize).transpose() - &c;
        let b = q_rest.row(i as usize).transpose() - &c0;
        m += a * b.transpose();
    }

    let svd = na::SVD::new(m, true, true);
    let u   = svd.u.unwrap();
    let mut v_t = svd.v_t.unwrap();   // nalgebra gives Vᵀ directly

    // Fix improper rotation (reflection)
    if (u * v_t).determinant() < 0.0 {
        v_t.row_mut(2).scale_mut(-1.0); // flip last row of Vᵀ ≡ flip last col of V
    }
    let r = u * v_t;

    // Project: gᵢ = R(q_restᵢ - c0) + c, then blend
    for &i in idx {
        let i  = i as usize;
        let g  = r * (q_rest.row(i).transpose() - &c0) + &c;
        let qi = q.row(i).transpose();
        let updated = (1.0 - weight) * qi + weight * g;
        q.row_mut(i).copy_from(&updated.transpose());
    }
}

// ── ClothSim ──────────────────────────────────────────────────────────────────

pub struct ClothSim {
    /// Current positions, Nx3.
    pub q: Positions,
    /// Rest positions, Nx3. Set once at construction, never mutated.
    pub q_rest: Positions,
    /// Positions at the start of the current step, Nx3.
    pub q_prev: Positions,
    /// Velocities, Nx3.
    pub v: Positions,
    /// Inverse masses, length N. Zero means pinned.
    pub w: na::DVector<f32>,
    /// Triangle index matrix, Mx3.
    pub faces: Faces,
    /// Bending diamond clusters: each entry is [a, b, opp0, opp1] where
    /// (a,b) is the shared edge and opp* are the opposite vertices.
    pub diamonds: Vec<[u32; 4]>,
    /// Index of the vertex currently being dragged, if any.
    pub clicked_vertex: Option<usize>,
    /// Target position for the dragged vertex in clip space.
    pub mouse_pos: [f32; 3],
}

impl ClothSim {
    pub fn from_cloth(cloth: &Cloth) -> Self {
        let n = cloth.resolution as usize;
        let num_verts = n * n;

        // ── Positions ──────────────────────────────────────────────────────────
        let mut q = Positions::zeros(num_verts);
        for (i, pos) in cloth.positions.iter().enumerate() {
            q[(i, 0)] = pos[0];
            q[(i, 1)] = pos[1];
            q[(i, 2)] = pos[2];
        }
        let q_rest = q.clone();
        let q_prev = q.clone();
        let v = Positions::zeros(num_verts);

        // ── Inverse masses ─────────────────────────────────────────────────────
        let mut w = na::DVector::from_element(num_verts, 1.0f32);
        w[(n - 1) * n] = 0.0;           // upper-left  pinned
        w[(n - 1) * n + (n - 1)] = 0.0; // upper-right pinned

        // ── Face matrix (matches cloth.rs: [tl,tr,br] + [tl,br,bl]) ───────────
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

        let diamonds = build_diamonds(&faces);

        Self {
            q, q_rest, q_prev, v, w, faces, diamonds,
            clicked_vertex: None,
            mouse_pos: [0.0; 3],
        }
    }

    pub fn step(&mut self, params: &SimParams) {
        let dt = params.time_step as f32;
        let n  = self.q.nrows();

        self.q_prev.copy_from(&self.q);

        // ── External forces → velocity ─────────────────────────────────────────
        if params.gravity_enabled {
            let g = params.gravity_g as f32;
            for i in 0..n {
                if self.w[i] > 0.0 {
                    self.v[(i, 1)] += g * dt;
                }
            }
        }

        // ── Predict positions ──────────────────────────────────────────────────
        for i in 0..n {
            if self.w[i] > 0.0 {
                self.q[(i, 0)] += self.v[(i, 0)] * dt;
                self.q[(i, 1)] += self.v[(i, 1)] * dt;
                self.q[(i, 2)] += self.v[(i, 2)] * dt;
            }
        }

        // ── Constraint projection ──────────────────────────────────────────────
        for _ in 0..params.constraint_iters {
            if params.stretch_enabled {
                let sw = params.stretch_weight as f32;
                for fi in 0..self.faces.nrows() {
                    // Copy indices first so q borrow is separate from faces borrow
                    let idx = [
                        self.faces[(fi, 0)],
                        self.faces[(fi, 1)],
                        self.faces[(fi, 2)],
                    ];
                    apply_constraint(&mut self.q, &self.q_rest, &idx, sw);
                }
            }

            if params.bending_enabled {
                let bw = params.bending_weight as f32;
                for di in 0..self.diamonds.len() {
                    let idx = self.diamonds[di]; // [u32;4] is Copy
                    apply_constraint(&mut self.q, &self.q_rest, &idx, bw);
                }
            }

            if params.pin_enabled {
                for i in 0..n {
                    if self.w[i] == 0.0 {
                        self.q.row_mut(i).copy_from(&self.q_rest.row(i));
                    }
                }
            }

            if params.pulling_enabled {
                if let Some(v) = self.clicked_vertex {
                    let w = params.pulling_weight as f32;
                    let mp = na::Vector3::new(self.mouse_pos[0], self.mouse_pos[1], self.mouse_pos[2]);
                    let qi = self.q.row(v).transpose();
                    let updated = (1.0 - w) * qi + w * mp;
                    self.q.row_mut(v).copy_from(&updated.transpose());
                }
            }
        }

        // ── Velocity from position delta ───────────────────────────────────────
        let inv_dt = 1.0 / dt;
        for i in 0..n {
            if self.w[i] > 0.0 {
                self.v[(i, 0)] = (self.q[(i, 0)] - self.q_prev[(i, 0)]) * inv_dt;
                self.v[(i, 1)] = (self.q[(i, 1)] - self.q_prev[(i, 1)]) * inv_dt;
                self.v[(i, 2)] = (self.q[(i, 2)] - self.q_prev[(i, 2)]) * inv_dt;
            }
        }
    }

    /// Copy `q` back into `cloth.positions` and upload to the GPU.
    pub fn write_to_cloth(&self, cloth: &mut Cloth, ctx: &GpuContext) {
        for (i, pos) in cloth.positions.iter_mut().enumerate() {
            pos[0] = self.q[(i, 0)];
            pos[1] = self.q[(i, 1)];
            pos[2] = self.q[(i, 2)];
        }
        cloth.upload(ctx);
    }
}
