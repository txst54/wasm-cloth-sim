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
pub(super) fn build_diamonds(faces: &Faces) -> Vec<[u32; 4]> {
    let m = faces.nrows();
    let mut edge_map: HashMap<(u32,u32), [(u32,u32); 2]> = HashMap::with_capacity(m * 3);
    let mut counts:   HashMap<(u32,u32), u8>              = HashMap::with_capacity(m * 3);
    for fi in 0..m {
        let v = [faces[(fi,0)], faces[(fi,1)], faces[(fi,2)]];
        for e in 0..3usize {
            let a = v[e]; let b = v[(e+1)%3]; let opp = v[(e+2)%3];
            let key = if a < b { (a,b) } else { (b,a) };
            let cnt = counts.entry(key).or_insert(0);
            if *cnt == 0 { edge_map.insert(key, [(opp,0),(0,0)]); }
            else if *cnt == 1 { edge_map.get_mut(&key).unwrap()[1] = (opp,0); }
            *cnt += 1;
        }
    }
    edge_map.into_iter()
        .filter(|(k, _)| *counts.get(k).unwrap() == 2)
        .map(|((a,b), adj)| [a, b, adj[0].0, adj[1].0])
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

/// Distance constraint between vertices `a` and `b`.
#[inline]
pub(super) fn apply_distance_constraint(
    q: &mut Positions,
    w: &na::DVector<f32>,
    a: usize, b: usize,
    rest_len: f32, weight: f32,
) {
    let dx = q[(b,0)]-q[(a,0)]; let dy = q[(b,1)]-q[(a,1)]; let dz = q[(b,2)]-q[(a,2)];
    let len_sq = dx*dx + dy*dy + dz*dz;
    if len_sq < 1e-12 { return; }
    let len = len_sq.sqrt();
    let wa = w[a]; let wb = w[b];
    let total_w = wa + wb;
    if total_w < 1e-12 { return; }
    let c = weight * (len - rest_len) / (total_w * len);
    if wa > 0.0 { q[(a,0)] += wa*c*dx; q[(a,1)] += wa*c*dy; q[(a,2)] += wa*c*dz; }
    if wb > 0.0 { q[(b,0)] -= wb*c*dx; q[(b,1)] -= wb*c*dy; q[(b,2)] -= wb*c*dz; }
}

// ── DragInfluence ─────────────────────────────────────────────────────────────

pub struct DragInfluence {
    pub vi: usize,
    pub alpha: f32,
    pub offset: na::Vector3<f32>,
}

// ── SimCore ───────────────────────────────────────────────────────────────────

/// Shared PBD state for all fabric-type simulations.
///
/// `step` is parameterised by two hooks so subclasses can override bending:
/// - `skip_bending_diamonds`: diamond indices that skip the default bending
///   constraint (e.g. replaced by a hinge dihedral constraint).
/// - `extra_per_iter`: called once per constraint iteration after all built-in
///   constraints. Receives `(&mut q, &w)` so callers can apply custom constraints
///   (e.g. hinge dihedral angles).
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
    pub clicked_vertex:     Option<usize>,
    pub dragging_vertices:  Option<Vec<DragInfluence>>,
    pub mouse_pos:          [f32; 3],
    pub triangle_hash:      TriangleSpatialHash,
}

impl SimCore {
    pub fn from_cloth(cloth: &Cloth) -> Self {
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
        w[(n-1)*n]         = 0.0; // upper-left  pinned
        w[(n-1)*n + (n-1)] = 0.0; // upper-right pinned

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

        Self {
            q, q_rest, q_prev, v, w, faces, diamonds, vertex_neighbors,
            edges, edge_rest_lengths, diamond_rest_diag,
            clicked_vertex: None, dragging_vertices: None, mouse_pos: [0.0; 3],
            triangle_hash,
        }
    }

    /// Full PBD step.
    ///
    /// `skip_bending_diamonds` — indices of diamonds that should skip the default
    /// bending constraint (use an empty set for plain cloth behaviour).
    ///
    /// `extra_per_iter` — closure called once per constraint iteration with
    /// `(&mut q, &w)`. Use this to inject hinge/dihedral constraints.
    pub fn step<F>(
        &mut self,
        params: &SimParams,
        skip_bending_diamonds: &HashSet<usize>,
        mut extra_per_iter: F,
    ) where F: FnMut(&mut Positions, &na::DVector<f32>) {
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

        let sc_pairs_initial = if params.self_collision_enabled
            && !params.self_collision_recompute_pairs
        {
            self.triangle_hash.rebuild(&self.q, &self.faces);
            self.close_vertex_triangle_pairs(params.self_collision_threshold as f32)
        } else {
            Vec::new()
        };

        for _ in 0..params.constraint_iters {
            if params.stretch_enabled {
                let sw = params.stretch_weight as f32;
                if params.use_distance_constraints {
                    for ei in 0..self.edges.len() {
                        let [a,b] = self.edges[ei];
                        apply_distance_constraint(
                            &mut self.q, &self.w,
                            a as usize, b as usize,
                            self.edge_rest_lengths[ei], sw,
                        );
                    }
                } else {
                    for fi in 0..self.faces.nrows() {
                        let idx = [self.faces[(fi,0)], self.faces[(fi,1)], self.faces[(fi,2)]];
                        apply_constraint(&mut self.q, &self.q_rest, &idx, sw);
                    }
                }
            }

            if params.bending_enabled {
                let bw = params.bending_weight as f32;
                if params.use_distance_constraints {
                    for di in 0..self.diamonds.len() {
                        if skip_bending_diamonds.contains(&di) { continue; }
                        let [_,_,c,d] = self.diamonds[di];
                        apply_distance_constraint(
                            &mut self.q, &self.w,
                            c as usize, d as usize,
                            self.diamond_rest_diag[di], bw,
                        );
                    }
                } else {
                    for di in 0..self.diamonds.len() {
                        if skip_bending_diamonds.contains(&di) { continue; }
                        let idx = self.diamonds[di];
                        apply_constraint(&mut self.q, &self.q_rest, &idx, bw);
                    }
                }
            }

            if params.pin_enabled {
                for i in 0..n {
                    if self.w[i] == 0.0 { self.q.row_mut(i).copy_from(&self.q_rest.row(i)); }
                }
            }

            if params.pulling_enabled {
                if let Some(_v) = self.clicked_vertex {
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
            }

            // Extra per-iteration constraints (e.g. hinge dihedral angles).
            extra_per_iter(&mut self.q, &self.w);

            if params.self_collision_enabled {
                let threshold = params.self_collision_threshold as f32;
                let t2 = threshold * threshold;
                let iter_pairs: Vec<(usize, u32)>;
                let sc_pairs: &[(usize, u32)] = if params.self_collision_recompute_pairs {
                    self.triangle_hash.rebuild(&self.q, &self.faces);
                    iter_pairs = self.close_vertex_triangle_pairs(threshold);
                    &iter_pairs
                } else {
                    &sc_pairs_initial
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
        }

        let inv_dt = 1.0 / dt;
        let damp = 1.0 - (params.damping as f32).clamp(0.0, 1.0);
        for i in 0..n {
            if self.w[i] > 0.0 {
                self.v[(i,0)] = (self.q[(i,0)] - self.q_prev[(i,0)]) * inv_dt * damp;
                self.v[(i,1)] = (self.q[(i,1)] - self.q_prev[(i,1)]) * inv_dt * damp;
                self.v[(i,2)] = (self.q[(i,2)] - self.q_prev[(i,2)]) * inv_dt * damp;
            }
        }
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
