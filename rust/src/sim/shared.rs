use std::collections::{HashMap, HashSet, VecDeque};

use nalgebra as na;

use crate::{cloth::Cloth, gpu::GpuContext, params::SimParams};

/// Nx3 matrix of vertex positions / velocities.
pub type Positions = na::OMatrix<f32, na::Dyn, na::Const<3>>;

/// Mx3 matrix of triangle vertex indices.
pub type Faces = na::OMatrix<u32, na::Dyn, na::Const<3>>;

// ── Spatial hash ─────────────────────────────────────────────────────────────

pub struct TriangleSpatialHash {
    pub cell_size: f32,
    inv_cell_size: f32,
    pub cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl TriangleSpatialHash {
    pub fn new(cell_size: f32) -> Self {
        Self { cell_size, inv_cell_size: 1.0 / cell_size, cells: HashMap::new() }
    }

    #[inline]
    fn cell_of(&self, x: f32, y: f32, z: f32) -> (i32, i32, i32) {
        (
            (x * self.inv_cell_size).floor() as i32,
            (y * self.inv_cell_size).floor() as i32,
            (z * self.inv_cell_size).floor() as i32,
        )
    }

    pub fn rebuild(&mut self, q: &Positions, faces: &Faces) {
        self.cells.clear();
        for fi in 0..faces.nrows() {
            let v0 = faces[(fi, 0)] as usize;
            let v1 = faces[(fi, 1)] as usize;
            let v2 = faces[(fi, 2)] as usize;
            let min_x = q[(v0,0)].min(q[(v1,0)]).min(q[(v2,0)]);
            let min_y = q[(v0,1)].min(q[(v1,1)]).min(q[(v2,1)]);
            let min_z = q[(v0,2)].min(q[(v1,2)]).min(q[(v2,2)]);
            let max_x = q[(v0,0)].max(q[(v1,0)]).max(q[(v2,0)]);
            let max_y = q[(v0,1)].max(q[(v1,1)]).max(q[(v2,1)]);
            let max_z = q[(v0,2)].max(q[(v1,2)]).max(q[(v2,2)]);
            let (cx0,cy0,cz0) = self.cell_of(min_x, min_y, min_z);
            let (cx1,cy1,cz1) = self.cell_of(max_x, max_y, max_z);
            for cx in cx0..=cx1 {
                for cy in cy0..=cy1 {
                    for cz in cz0..=cz1 {
                        self.cells.entry((cx,cy,cz)).or_default().push(fi as u32);
                    }
                }
            }
        }
    }

    pub fn query_aabb(&self, min: [f32; 3], max: [f32; 3]) -> HashSet<u32> {
        let (cx0,cy0,cz0) = self.cell_of(min[0], min[1], min[2]);
        let (cx1,cy1,cz1) = self.cell_of(max[0], max[1], max[2]);
        let mut result = HashSet::new();
        for cx in cx0..=cx1 {
            for cy in cy0..=cy1 {
                for cz in cz0..=cz1 {
                    if let Some(tris) = self.cells.get(&(cx,cy,cz)) {
                        result.extend(tris);
                    }
                }
            }
        }
        result
    }
}

// ── Mesh topology helpers ─────────────────────────────────────────────────────

pub(super) fn build_edges(faces: &Faces) -> Vec<[u32; 2]> {
    let mut seen: HashSet<(u32,u32)> = HashSet::new();
    let mut edges = Vec::new();
    for fi in 0..faces.nrows() {
        let v = [faces[(fi,0)], faces[(fi,1)], faces[(fi,2)]];
        for e in 0..3 {
            let a = v[e]; let b = v[(e+1)%3];
            let key = if a < b { (a,b) } else { (b,a) };
            if seen.insert(key) { edges.push([key.0, key.1]); }
        }
    }
    edges
}

pub(super) fn build_vertex_neighbors(faces: &Faces, num_verts: usize) -> Vec<HashSet<u32>> {
    let mut nb = vec![HashSet::new(); num_verts];
    for fi in 0..faces.nrows() {
        let v = [faces[(fi,0)], faces[(fi,1)], faces[(fi,2)]];
        for e in 0..3 {
            let a = v[e] as usize; let b = v[(e+1)%3] as usize;
            nb[a].insert(b as u32); nb[b].insert(a as u32);
        }
    }
    nb
}

/// For each interior edge shared by two triangles, collect the 4 vertices
/// `[a, b, opp0, opp1]` where `(a, b)` is the shared edge.
/// Build diamond quads `[a, b, c, d]` for every interior edge.
///
/// Ordering guarantee derived from face winding (assumes CCW faces):
/// - `c` is the opposite vertex from the triangle where edge goes a→b
///   (forward traversal), i.e. triangle (a, b, c) is CCW.
/// - `d` is the opposite vertex from the triangle where edge goes b→a
///   (backward traversal), i.e. triangle (b, a, d) is CCW.
///
/// This means for a flat CCW mesh the dihedral angle atan2(sin, cos) of
/// `(b-a)×(c-a)` vs `(b-a)×(d-a)` is exactly π, and deviations from π
/// have a consistent sign across all edges.
pub(super) fn build_diamonds(faces: &Faces) -> Vec<[u32; 4]> {
    let m = faces.nrows();
    // For each normalised edge key (min,max), store (opposite_vertex, forward)
    // where forward = true when the edge was traversed min→max in that face.
    let mut edge_map: HashMap<(u32, u32), Vec<(u32, bool)>> = HashMap::with_capacity(m * 3);
    for fi in 0..m {
        let v = [faces[(fi,0)], faces[(fi,1)], faces[(fi,2)]];
        for e in 0..3usize {
            let a = v[e]; let b = v[(e+1)%3]; let opp = v[(e+2)%3];
            let key = if a < b { (a, b) } else { (b, a) };
            let forward = a < b;
            edge_map.entry(key).or_default().push((opp, forward));
        }
    }
    edge_map.into_iter()
        .filter(|(_, adj)| adj.len() == 2)
        .map(|((a, b), adj)| {
            // c = opp from forward triangle (edge a→b), d = opp from backward
            let (c, d) = if adj[0].1 {
                (adj[0].0, adj[1].0)
            } else {
                (adj[1].0, adj[0].0)
            };
            [a, b, c, d]
        })
        .collect()
}

// ── Constraint helpers ────────────────────────────────────────────────────────

pub(super) fn gaussian_weight(d: f32, sigma: f32) -> f32 {
    (-(d * d) / (2.0 * sigma * sigma)).exp()
}

/// Returns the closest point on triangle (a, b, c) to point p.
/// Uses Ericson's barycentric region method.
pub fn closest_point_on_triangle(
    p: na::Vector3<f32>,
    a: na::Vector3<f32>,
    b: na::Vector3<f32>,
    c: na::Vector3<f32>,
) -> na::Vector3<f32> {
    let ab = b - a; let ac = c - a; let ap = p - a;
    let d1 = ab.dot(&ap); let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 { return a; }
    let bp = p - b; let d3 = ab.dot(&bp); let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 { return b; }
    let cp = p - c; let d5 = ab.dot(&cp); let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 { return c; }
    let vc = d1*d4 - d3*d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 { return a + (d1/(d1-d3)) * ab; }
    let vb = d5*d2 - d1*d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 { return a + (d2/(d2-d6)) * ac; }
    let va = d3*d6 - d5*d4;
    if va <= 0.0 && (d4-d3) >= 0.0 && (d5-d6) >= 0.0 {
        return b + ((d4-d3)/((d4-d3)+(d5-d6))) * (c - b);
    }
    let denom = 1.0 / (va + vb + vc);
    a + vb*denom * ab + vc*denom * ac
}

/// Shape-matching constraint via SVD polar decomposition.
pub(super) fn apply_constraint(q: &mut Positions, q_rest: &Positions, idx: &[u32], weight: f32) {
    let n = idx.len();
    let inv_n = 1.0 / n as f32;
    let mut c  = na::Vector3::<f32>::zeros();
    let mut c0 = na::Vector3::<f32>::zeros();
    for &i in idx {
        c  += q.row(i as usize).transpose();
        c0 += q_rest.row(i as usize).transpose();
    }
    c *= inv_n; c0 *= inv_n;
    let mut m = na::Matrix3::<f32>::zeros();
    for &i in idx {
        let a = q.row(i as usize).transpose() - &c;
        let b = q_rest.row(i as usize).transpose() - &c0;
        m += a * b.transpose();
    }
    let svd = na::SVD::new(m, true, true);
    let u = svd.u.unwrap();
    let mut v_t = svd.v_t.unwrap();
    if (u * v_t).determinant() < 0.0 { v_t.row_mut(2).scale_mut(-1.0); }
    let r = u * v_t;
    for &i in idx {
        let i = i as usize;
        let g = r * (q_rest.row(i).transpose() - &c0) + &c;
        let qi = q.row(i).transpose();
        let updated = (1.0 - weight) * qi + weight * g;
        q.row_mut(i).copy_from(&updated.transpose());
    }
}

/// XPBD distance constraint between vertices `a` and `b`.
/// `alpha_tilde` = compliance / dt².  Use 0 for infinitely stiff (pure projection).
#[inline]
pub(super) fn apply_distance_constraint_xpbd(
    q: &mut Positions,
    w: &na::DVector<f32>,
    a: usize, b: usize,
    rest_len: f32,
    alpha_tilde: f32,
    lambda: &mut f32,
) {
    let dx = q[(b,0)]-q[(a,0)]; let dy = q[(b,1)]-q[(a,1)]; let dz = q[(b,2)]-q[(a,2)];
    let len_sq = dx*dx + dy*dy + dz*dz;
    if len_sq < 1e-12 { return; }
    let len = len_sq.sqrt();
    let c_val = len - rest_len;
    if c_val.abs() < 1e-8 { return; }
    let wa = w[a]; let wb = w[b];
    let denom = wa + wb + alpha_tilde;
    if denom < 1e-12 { return; }
    let dl = -(c_val + alpha_tilde * *lambda) / denom;
    *lambda += dl;
    let correction = dl / len;
    if wa > 0.0 { q[(a,0)] -= wa*correction*dx; q[(a,1)] -= wa*correction*dy; q[(a,2)] -= wa*correction*dz; }
    if wb > 0.0 { q[(b,0)] += wb*correction*dx; q[(b,1)] += wb*correction*dy; q[(b,2)] += wb*correction*dz; }
}

/// Convert UI stretch_weight (0..1, higher = stiffer) to XPBD compliance.
/// weight=1 → compliance≈0 (rigid), weight→0 → large compliance (soft).
fn stretch_compliance_from_weight(w: f32) -> f32 {
    if w >= 1.0 { return 0.0; }
    if w <= 0.0 { return 1.0; }
    (1.0 - w) * 1e-2
}

/// Convert UI bending_weight to XPBD compliance.
fn bend_compliance_from_weight(w: f32) -> f32 {
    if w >= 1.0 { return 0.0; }
    if w <= 0.0 { return 1.0; }
    (1.0 - w) * 1e-1
}

// ── DragInfluence ─────────────────────────────────────────────────────────────

pub struct DragInfluence {
    pub vi: usize,
    pub alpha: f32,
    pub offset: na::Vector3<f32>,
}

// ── SimCore ───────────────────────────────────────────────────────────────────

/// Shared XPBD state for all fabric-type simulations.
pub struct SimCore {
    pub q:                  Positions,
    pub q_rest:             Positions,
    pub q_prev:             Positions,
    pub v:                  Positions,
    pub w:                  na::DVector<f32>,
    pub faces:              Faces,
    pub diamonds:           Vec<[u32; 4]>,
    pub vertex_neighbors:   Vec<HashSet<u32>>,
    pub edges:              Vec<[u32; 2]>,
    pub edge_rest_lengths:  Vec<f32>,
    pub diamond_rest_diag:  Vec<f32>,
    pub stretch_lambdas:    Vec<f32>,
    pub bend_lambdas:       Vec<f32>,
    pub clicked_vertex:     Option<usize>,
    pub dragging_vertices:  Option<Vec<DragInfluence>>,
    pub mouse_pos:          [f32; 3],
    pub triangle_hash:      TriangleSpatialHash,
}

impl SimCore {
    pub fn from_cloth(cloth: &Cloth, pinned: &[usize]) -> Self {
        let n = cloth.resolution as usize;
        let num_verts = n * n;

        let mut q = Positions::zeros(num_verts);
        for (i, pos) in cloth.positions.iter().enumerate() {
            q[(i,0)] = pos[0]; q[(i,1)] = pos[1]; q[(i,2)] = pos[2];
        }
        let q_rest = q.clone();
        let q_prev = q.clone();
        let v = Positions::zeros(num_verts);

        let mut w = na::DVector::from_element(num_verts, 1.0f32);
        for &idx in pinned {
            w[idx] = 0.0;
        }

        let num_quads = (n-1)*(n-1);
        let mut faces = Faces::zeros(num_quads * 2);
        let mut fi = 0usize;
        for row in 0..(n-1) {
            for col in 0..(n-1) {
                let tl = (row*n + col) as u32;
                let tr = tl + 1;
                let bl = ((row+1)*n + col) as u32;
                let br = bl + 1;
                faces[(fi,0)]=tl; faces[(fi,1)]=tr; faces[(fi,2)]=br; fi+=1;
                faces[(fi,0)]=tl; faces[(fi,1)]=br; faces[(fi,2)]=bl; fi+=1;
            }
        }

        let diamonds        = build_diamonds(&faces);
        let vertex_neighbors = build_vertex_neighbors(&faces, num_verts);
        let edges           = build_edges(&faces);
        let edge_rest_lengths: Vec<f32> = edges.iter().map(|&[a,b]| {
            (q_rest.row(a as usize) - q_rest.row(b as usize)).norm()
        }).collect();
        let diamond_rest_diag: Vec<f32> = diamonds.iter().map(|&[_,_,c,d]| {
            (q_rest.row(c as usize) - q_rest.row(d as usize)).norm()
        }).collect();

        let cell_size = 3.6 / (n as f32 - 1.0);
        let mut triangle_hash = TriangleSpatialHash::new(cell_size);
        triangle_hash.rebuild(&q, &faces);

        let stretch_lambdas = vec![0.0f32; edges.len()];
        let bend_lambdas = vec![0.0f32; diamonds.len()];

        Self {
            q, q_rest, q_prev, v, w, faces, diamonds, vertex_neighbors,
            edges, edge_rest_lengths, diamond_rest_diag,
            stretch_lambdas, bend_lambdas,
            clicked_vertex: None, dragging_vertices: None, mouse_pos: [0.0; 3],
            triangle_hash,
        }
    }

    /// Create SimCore from arbitrary mesh positions and faces.
    pub fn from_mesh(positions: &[[f32; 3]], face_indices: &[[u32; 3]], pinned: &[usize]) -> Self {
        let num_verts = positions.len();
        let num_faces = face_indices.len();

        let mut q = Positions::zeros(num_verts);
        for (i, pos) in positions.iter().enumerate() {
            q[(i,0)] = pos[0]; q[(i,1)] = pos[1]; q[(i,2)] = pos[2];
        }
        let q_rest = q.clone();
        let q_prev = q.clone();
        let v = Positions::zeros(num_verts);

        let mut w = na::DVector::from_element(num_verts, 1.0f32);
        for &idx in pinned {
            if idx < num_verts { w[idx] = 0.0; }
        }

        let mut faces = Faces::zeros(num_faces);
        for (fi, &[i0, i1, i2]) in face_indices.iter().enumerate() {
            faces[(fi, 0)] = i0;
            faces[(fi, 1)] = i1;
            faces[(fi, 2)] = i2;
        }

        let diamonds = build_diamonds(&faces);
        let vertex_neighbors = build_vertex_neighbors(&faces, num_verts);
        let edges = build_edges(&faces);
        let edge_rest_lengths: Vec<f32> = edges.iter().map(|&[a, b]| {
            (q_rest.row(a as usize) - q_rest.row(b as usize)).norm()
        }).collect();
        let diamond_rest_diag: Vec<f32> = diamonds.iter().map(|&[_, _, c, d]| {
            (q_rest.row(c as usize) - q_rest.row(d as usize)).norm()
        }).collect();

        let stretch_lambdas = vec![0.0f32; edges.len()];
        let bend_lambdas = vec![0.0f32; diamonds.len()];

        let avg_edge_len = if edge_rest_lengths.is_empty() {
            0.1
        } else {
            edge_rest_lengths.iter().sum::<f32>() / edge_rest_lengths.len() as f32
        };
        let cell_size = avg_edge_len * 2.0;
        let mut triangle_hash = TriangleSpatialHash::new(cell_size);
        triangle_hash.rebuild(&q, &faces);

        Self {
            q, q_rest, q_prev, v, w, faces, diamonds, vertex_neighbors,
            edges, edge_rest_lengths, diamond_rest_diag,
            stretch_lambdas, bend_lambdas,
            clicked_vertex: None, dragging_vertices: None, mouse_pos: [0.0; 3],
            triangle_hash,
        }
    }

    /// Predict positions: apply forces and integrate velocity.
    pub fn predict(&mut self, params: &SimParams) {
        let dt = params.time_step as f32;
        let n  = self.q.nrows();

        self.q_prev.copy_from(&self.q);

        if params.gravity_enabled {
            let g = params.gravity_g as f32;
            for i in 0..n {
                if self.w[i] > 0.0 { self.v[(i,1)] += g * dt; }
            }
        }

        if let Some(v) = self.clicked_vertex {
            self.dragging_vertices =
                Some(self.build_drag_influences(v, params.pulling_area as usize));
        }

        for i in 0..n {
            if self.w[i] > 0.0 {
                self.q[(i,0)] += self.v[(i,0)] * dt;
                self.q[(i,1)] += self.v[(i,1)] * dt;
                self.q[(i,2)] += self.v[(i,2)] * dt;
            }
        }
    }

    /// Reset all XPBD lambda accumulators (call once per substep before iterations).
    pub fn reset_lambdas(&mut self) {
        self.stretch_lambdas.fill(0.0);
        self.bend_lambdas.fill(0.0);
    }

    /// Solve stretch constraints (XPBD distance constraints on edges).
    /// `compliance` maps from the UI weight: lower compliance = stiffer.
    pub fn solve_stretch(&mut self, params: &SimParams) {
        if !params.stretch_enabled { return; }
        let dt = params.time_step as f32;
        let alpha_tilde = stretch_compliance_from_weight(params.stretch_weight as f32) / (dt * dt);
        if params.use_distance_constraints {
            for ei in 0..self.edges.len() {
                let [a,b] = self.edges[ei];
                apply_distance_constraint_xpbd(
                    &mut self.q, &self.w,
                    a as usize, b as usize,
                    self.edge_rest_lengths[ei], alpha_tilde,
                    &mut self.stretch_lambdas[ei],
                );
            }
        } else {
            let sw = params.stretch_weight as f32;
            for fi in 0..self.faces.nrows() {
                let idx = [self.faces[(fi,0)], self.faces[(fi,1)], self.faces[(fi,2)]];
                apply_constraint(&mut self.q, &self.q_rest, &idx, sw);
            }
        }
    }

    /// Solve bend constraints (XPBD distance constraints on diamond diagonals).
    pub fn solve_bend(&mut self, params: &SimParams, skip: &HashSet<usize>) {
        if !params.bending_enabled { return; }
        let dt = params.time_step as f32;
        let alpha_tilde = bend_compliance_from_weight(params.bending_weight as f32) / (dt * dt);
        if params.use_distance_constraints {
            for di in 0..self.diamonds.len() {
                if skip.contains(&di) { continue; }
                let [_,_,c,d] = self.diamonds[di];
                apply_distance_constraint_xpbd(
                    &mut self.q, &self.w,
                    c as usize, d as usize,
                    self.diamond_rest_diag[di], alpha_tilde,
                    &mut self.bend_lambdas[di],
                );
            }
        } else {
            let bw = params.bending_weight as f32;
            for di in 0..self.diamonds.len() {
                if skip.contains(&di) { continue; }
                let idx = self.diamonds[di];
                apply_constraint(&mut self.q, &self.q_rest, &idx, bw);
            }
        }
    }

    /// Enforce pin constraints (infinite-mass vertices snap to rest position).
    pub fn solve_pins(&mut self, params: &SimParams) {
        if !params.pin_enabled { return; }
        let n = self.q.nrows();
        for i in 0..n {
            if self.w[i] == 0.0 { self.q.row_mut(i).copy_from(&self.q_rest.row(i)); }
        }
    }

    /// Apply mouse-drag pulling constraints.
    pub fn solve_pulling(&mut self, params: &SimParams) {
        if !params.pulling_enabled { return; }
        if self.clicked_vertex.is_none() { return; }
        let mp = na::Vector3::new(
            self.mouse_pos[0], self.mouse_pos[1], self.mouse_pos[2],
        );
        let base = params.pulling_weight as f32;
        if let Some(verts) = &self.dragging_vertices {
            for inf in verts {
                let target = mp + inf.offset;
                let qi = self.q.row(inf.vi).transpose();
                let wt = base * inf.alpha;
                let updated = (1.0 - wt) * qi + wt * target;
                self.q.row_mut(inf.vi).copy_from(&updated.transpose());
            }
        }
    }

    /// Pre-compute self-collision broad-phase pairs (call once before iteration loop).
    pub fn precompute_collision_pairs(&mut self, params: &SimParams) -> Vec<(usize, u32)> {
        if params.self_collision_enabled && !params.self_collision_recompute_pairs {
            self.triangle_hash.rebuild(&self.q, &self.faces);
            self.close_vertex_triangle_pairs(params.self_collision_threshold as f32)
        } else {
            Vec::new()
        }
    }

    /// Solve self-collision constraints for one iteration.
    pub fn solve_self_collision(&mut self, params: &SimParams, precomputed_pairs: &[(usize, u32)]) {
        if !params.self_collision_enabled { return; }
        let threshold = params.self_collision_threshold as f32;
        let t2 = threshold * threshold;
        let iter_pairs: Vec<(usize, u32)>;
        let sc_pairs: &[(usize, u32)] = if params.self_collision_recompute_pairs {
            self.triangle_hash.rebuild(&self.q, &self.faces);
            iter_pairs = self.close_vertex_triangle_pairs(threshold);
            &iter_pairs
        } else {
            precomputed_pairs
        };
        for &(vi, fi) in sc_pairs {
            let i0 = self.faces[(fi as usize,0)] as usize;
            let i1 = self.faces[(fi as usize,1)] as usize;
            let i2 = self.faces[(fi as usize,2)] as usize;
            let p = na::Vector3::new(self.q[(vi,0)], self.q[(vi,1)], self.q[(vi,2)]);
            let a = na::Vector3::new(self.q[(i0,0)], self.q[(i0,1)], self.q[(i0,2)]);
            let b = na::Vector3::new(self.q[(i1,0)], self.q[(i1,1)], self.q[(i1,2)]);
            let c = na::Vector3::new(self.q[(i2,0)], self.q[(i2,1)], self.q[(i2,2)]);
            let closest = closest_point_on_triangle(p, a, b, c);
            let d = p - closest;
            let dist_sq = d.norm_squared();
            if dist_sq >= t2 || dist_sq < 1e-12 { continue; }
            let dist = dist_sq.sqrt();
            let n_hat = d / dist;
            let pen = threshold - dist;
            let wv = self.w[vi];
            let w0 = self.w[i0]/3.0; let w1 = self.w[i1]/3.0; let w2 = self.w[i2]/3.0;
            let total_w = wv + w0 + w1 + w2;
            if total_w < 1e-12 { continue; }
            if wv > 0.0 {
                let delta = (wv/total_w) * pen;
                self.q[(vi,0)] += n_hat[0]*delta;
                self.q[(vi,1)] += n_hat[1]*delta;
                self.q[(vi,2)] += n_hat[2]*delta;
            }
            for (ti, wi) in [(i0,w0),(i1,w1),(i2,w2)] {
                if self.w[ti] > 0.0 {
                    let delta = (wi/total_w) * pen;
                    self.q[(ti,0)] -= n_hat[0]*delta;
                    self.q[(ti,1)] -= n_hat[1]*delta;
                    self.q[(ti,2)] -= n_hat[2]*delta;
                }
            }
        }
    }

    /// Derive velocity from position change and apply damping.
    pub fn update_velocity(&mut self, params: &SimParams) {
        let dt = params.time_step as f32;
        let inv_dt = 1.0 / dt;
        let damp = 1.0 - (params.damping as f32).clamp(0.0, 1.0);
        let n = self.q.nrows();
        for i in 0..n {
            if self.w[i] > 0.0 {
                self.v[(i,0)] = (self.q[(i,0)] - self.q_prev[(i,0)]) * inv_dt * damp;
                self.v[(i,1)] = (self.q[(i,1)] - self.q_prev[(i,1)]) * inv_dt * damp;
                self.v[(i,2)] = (self.q[(i,2)] - self.q_prev[(i,2)]) * inv_dt * damp;
            }
        }
    }

    /// Remove rigid-body rotation from position corrections.
    /// Computes the angular displacement introduced by the constraint solve
    /// (q vs q_prev) and counter-rotates q so only deformation remains.
    /// Must be called after constraint solving, before update_velocity.
    pub fn remove_rigid_rotation(&mut self) {
        let n = self.q.nrows();
        let dt = 1.0f32; // we work in displacement space, dt cancels out

        // COM of current and previous positions (mass = 1/w)
        let mut total_mass = 0.0f32;
        let mut com = na::Vector3::<f32>::zeros();
        let mut com_prev = na::Vector3::<f32>::zeros();
        for i in 0..n {
            if self.w[i] > 0.0 {
                let m = 1.0 / self.w[i];
                com += m * na::Vector3::new(self.q[(i,0)], self.q[(i,1)], self.q[(i,2)]);
                com_prev += m * na::Vector3::new(self.q_prev[(i,0)], self.q_prev[(i,1)], self.q_prev[(i,2)]);
                total_mass += m;
            }
        }
        if total_mass < 1e-12 { return; }
        com /= total_mass;
        com_prev /= total_mass;

        // Compute angular velocity of the displacement field (q - q_prev)
        // using r relative to current COM, displacement as "velocity"
        let mut ang_mom = na::Vector3::<f32>::zeros();
        let mut inertia = na::Matrix3::<f32>::zeros();
        for i in 0..n {
            if self.w[i] > 0.0 {
                let m = 1.0 / self.w[i];
                let r = na::Vector3::new(
                    self.q[(i,0)] - com[0],
                    self.q[(i,1)] - com[1],
                    self.q[(i,2)] - com[2],
                );
                let disp = na::Vector3::new(
                    self.q[(i,0)] - self.q_prev[(i,0)],
                    self.q[(i,1)] - self.q_prev[(i,1)],
                    self.q[(i,2)] - self.q_prev[(i,2)],
                );
                ang_mom += m * r.cross(&disp);
                let r2 = r.norm_squared();
                inertia += m * (r2 * na::Matrix3::identity() - r * r.transpose());
            }
        }

        let Some(inv_inertia) = inertia.try_inverse() else { return };
        let omega = inv_inertia * ang_mom;

        // Counter-rotate positions: subtract ω × r from each vertex position
        for i in 0..n {
            if self.w[i] > 0.0 {
                let r = na::Vector3::new(
                    self.q[(i,0)] - com[0],
                    self.q[(i,1)] - com[1],
                    self.q[(i,2)] - com[2],
                );
                let rot_disp = omega.cross(&r);
                self.q[(i,0)] -= rot_disp[0];
                self.q[(i,1)] -= rot_disp[1];
                self.q[(i,2)] -= rot_disp[2];
            }
        }
    }

    /// Convenience: full XPBD step for simple sims (cloth) that don't need
    /// custom constraints injected into the loop.
    pub fn step(&mut self, params: &SimParams, skip_bending: &HashSet<usize>) {
        self.predict(params);
        let sc_pairs = self.precompute_collision_pairs(params);
        self.reset_lambdas();
        for _ in 0..params.constraint_iters {
            self.solve_stretch(params);
            self.solve_bend(params, skip_bending);
            self.solve_pins(params);
            self.solve_pulling(params);
            self.solve_self_collision(params, &sc_pairs);
        }
        self.update_velocity(params);
    }

    /// Broad-phase filter: returns `(vertex_idx, face_idx)` pairs where the vertex
    /// AABB (swept from q_prev to q, padded by threshold) overlaps the triangle AABB,
    /// and the vertex is not topologically adjacent to the triangle.
    /// Narrow-phase distance check happens in `step`.
    pub fn close_vertex_triangle_pairs(&self, threshold: f32) -> Vec<(usize, u32)> {
        let mut pairs = Vec::new();
        for vi in 0..self.q.nrows() {
            let px0=self.q_prev[(vi,0)]; let py0=self.q_prev[(vi,1)]; let pz0=self.q_prev[(vi,2)];
            let px1=self.q[(vi,0)];     let py1=self.q[(vi,1)];     let pz1=self.q[(vi,2)];
            let min = [px0.min(px1)-threshold, py0.min(py1)-threshold, pz0.min(pz1)-threshold];
            let max = [px0.max(px1)+threshold, py0.max(py1)+threshold, pz0.max(pz1)+threshold];
            let candidates = self.triangle_hash.query_aabb(min, max);
            for fi in candidates {
                let i0=self.faces[(fi as usize,0)] as usize;
                let i1=self.faces[(fi as usize,1)] as usize;
                let i2=self.faces[(fi as usize,2)] as usize;
                if vi==i0 || vi==i1 || vi==i2 { continue; }
                let nb = &self.vertex_neighbors[vi];
                if nb.contains(&(i0 as u32)) || nb.contains(&(i1 as u32)) || nb.contains(&(i2 as u32)) { continue; }
                pairs.push((vi, fi));
            }
        }
        pairs
    }

    pub fn build_drag_influences(&self, center: usize, max_hops: usize) -> Vec<DragInfluence> {
        let center_pos = self.q.row(center).transpose();
        let mut dist = vec![usize::MAX; self.q.nrows()];
        let mut queue = VecDeque::new();
        dist[center] = 0;
        queue.push_back(center);
        let mut out = Vec::new();
        while let Some(v) = queue.pop_front() {
            let d = dist[v];
            if d > max_hops { continue; }
            let alpha = gaussian_weight(d as f32, 2.0);
            if alpha > 0.0 {
                out.push(DragInfluence {
                    vi: v,
                    alpha,
                    offset: self.q.row(v).transpose() - center_pos,
                });
            }
            if d == max_hops { continue; }
            for &nb in &self.vertex_neighbors[v] {
                let nb = nb as usize;
                if dist[nb] == usize::MAX { dist[nb] = d + 1; queue.push_back(nb); }
            }
        }
        out
    }

    pub fn write_to_cloth(&self, cloth: &mut Cloth, ctx: &GpuContext) {
        for (i, pos) in cloth.positions.iter_mut().enumerate() {
            pos[0] = self.q[(i,0)]; pos[1] = self.q[(i,1)]; pos[2] = self.q[(i,2)];
        }
        cloth.upload(ctx);
    }
}
