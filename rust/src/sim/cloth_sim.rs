use std::collections::{HashMap, HashSet};

use nalgebra as na;

use crate::{cloth::Cloth, gpu::GpuContext, params::SimParams};

/// Nx3 matrix of vertex positions / velocities.
pub type Positions = na::OMatrix<f32, na::Dyn, na::Const<3>>;

/// Mx3 matrix of triangle vertex indices (matches cloth.rs triangulation).
pub type Faces = na::OMatrix<u32, na::Dyn, na::Const<3>>;

/// A uniform-grid spatial hash over cloth triangles.
///
/// Each frame, call `rebuild` with the current positions and face list.
/// Then call `query_aabb` to retrieve candidate triangle indices for any region.
pub struct TriangleSpatialHash {
    /// Side length of each cubic cell in world units.
    pub cell_size: f32,
    inv_cell_size: f32,
    /// Cell key → triangle (face) indices that overlap that cell.
    pub cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl TriangleSpatialHash {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
        }
    }

    #[inline]
    fn cell_of(&self, x: f32, y: f32, z: f32) -> (i32, i32, i32) {
        (
            (x * self.inv_cell_size).floor() as i32,
            (y * self.inv_cell_size).floor() as i32,
            (z * self.inv_cell_size).floor() as i32,
        )
    }

    /// Recompute the hash from the current cloth positions. Called every frame.
    pub fn rebuild(&mut self, q: &Positions, q_prev: &Positions, faces: &Faces) {
        self.cells.clear();

        let m = faces.nrows();
        for fi in 0..m {
            let v0 = faces[(fi, 0)] as usize;
            let v1 = faces[(fi, 1)] as usize;
            let v2 = faces[(fi, 2)] as usize;

            let min_x = q[(v0, 0)].min(q[(v1, 0)]).min(q[(v2, 0)])
                .min(q_prev[(v0, 0)]).min(q_prev[(v1, 0)]).min(q_prev[(v2, 0)]);
            let min_y = q[(v0, 1)].min(q[(v1, 1)]).min(q[(v2, 1)])
                .min(q_prev[(v0, 1)]).min(q_prev[(v1, 1)]).min(q_prev[(v2, 1)]);
            let min_z = q[(v0, 2)].min(q[(v1, 2)]).min(q[(v2, 2)])
                .min(q_prev[(v0, 2)]).min(q_prev[(v1, 2)]).min(q_prev[(v2, 2)]);

            let max_x = q[(v0, 0)].max(q[(v1, 0)]).max(q[(v2, 0)])
                .max(q_prev[(v0, 0)]).max(q_prev[(v1, 0)]).max(q_prev[(v2, 0)]);
            let max_y = q[(v0, 1)].max(q[(v1, 1)]).max(q[(v2, 1)])
                .max(q_prev[(v0, 1)]).max(q_prev[(v1, 1)]).max(q_prev[(v2, 1)]);
            let max_z = q[(v0, 2)].max(q[(v1, 2)]).max(q[(v2, 2)])
                .max(q_prev[(v0, 2)]).max(q_prev[(v1, 2)]).max(q_prev[(v2, 2)]);

            let (cx0, cy0, cz0) = self.cell_of(min_x, min_y, min_z);
            let (cx1, cy1, cz1) = self.cell_of(max_x, max_y, max_z);

            for cx in cx0..=cx1 {
                for cy in cy0..=cy1 {
                    for cz in cz0..=cz1 {
                        self.cells.entry((cx, cy, cz)).or_default().push(fi as u32);
                    }
                }
            }
        }
    }

    /// Returns the unique set of triangle indices whose cells overlap the given AABB.
    pub fn query_aabb(&self, min: [f32; 3], max: [f32; 3]) -> HashSet<u32> {
        let (cx0, cy0, cz0) = self.cell_of(min[0], min[1], min[2]);
        let (cx1, cy1, cz1) = self.cell_of(max[0], max[1], max[2]);
        let mut result = HashSet::new();
        for cx in cx0..=cx1 {
            for cy in cy0..=cy1 {
                for cz in cz0..=cz1 {
                    if let Some(tris) = self.cells.get(&(cx, cy, cz)) {
                        result.extend(tris);
                    }
                }
            }
        }
        result
    }
}

/// Build the unique edge list from a face matrix, returning [a, b] pairs with a < b.
fn build_edges(faces: &Faces) -> Vec<[u32; 2]> {
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let mut edges = Vec::new();
    for fi in 0..faces.nrows() {
        let v = [faces[(fi, 0)], faces[(fi, 1)], faces[(fi, 2)]];
        for e in 0..3 {
            let a = v[e];
            let b = v[(e + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            if seen.insert(key) {
                edges.push([key.0, key.1]);
            }
        }
    }
    edges
}

/// Build a per-vertex set of directly connected (1-ring) neighbor vertex indices.
/// Used to exclude adjacent triangles from self-collision detection.
fn build_vertex_neighbors(faces: &Faces, num_verts: usize) -> Vec<HashSet<u32>> {
    let mut neighbors = vec![HashSet::new(); num_verts];
    for fi in 0..faces.nrows() {
        let v = [faces[(fi, 0)], faces[(fi, 1)], faces[(fi, 2)]];
        for e in 0..3 {
            let a = v[e] as usize;
            let b = v[(e + 1) % 3] as usize;
            neighbors[a].insert(b as u32);
            neighbors[b].insert(a as u32);
        }
    }
    neighbors
}

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

/// Finds the best-fit rotation R between the current cluster and the rest
/// cluster, then blends each vertex toward its rotated rest target.
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
        v_t.row_mut(2).scale_mut(-1.0); // flip last row of VT ≡ flip last col of V
    }
    let r = u * v_t;

    // Project: gi = R(q_resti - c0) + c, then blend
    for &i in idx {
        let i  = i as usize;
        let g  = r * (q_rest.row(i).transpose() - &c0) + &c;
        let qi = q.row(i).transpose();
        let updated = (1.0 - weight) * qi + weight * g;
        q.row_mut(i).copy_from(&updated.transpose());
    }
}

/// Enforce a single distance constraint between vertices `a` and `b`.
/// Moves each vertex along the edge direction, weighted by inverse mass,
/// so the pair converges toward `rest_len`. `weight` ∈ [0, 1] scales the correction.
#[inline]
fn apply_distance_constraint(
    q: &mut Positions,
    w: &na::DVector<f32>,
    a: usize,
    b: usize,
    rest_len: f32,
    weight: f32,
) {
    let dx = q[(b, 0)] - q[(a, 0)];
    let dy = q[(b, 1)] - q[(a, 1)];
    let dz = q[(b, 2)] - q[(a, 2)];
    let len_sq = dx * dx + dy * dy + dz * dz;
    if len_sq < 1e-12 { return; }
    let len = len_sq.sqrt();
    let wa = w[a];
    let wb = w[b];
    let total_w = wa + wb;
    if total_w < 1e-12 { return; }
    // c folds in both the normalisation (1/len) and the mass-weighted correction magnitude.
    let c = weight * (len - rest_len) / (total_w * len);
    if wa > 0.0 {
        q[(a, 0)] += wa * c * dx;
        q[(a, 1)] += wa * c * dy;
        q[(a, 2)] += wa * c * dz;
    }
    if wb > 0.0 {
        q[(b, 0)] -= wb * c * dx;
        q[(b, 1)] -= wb * c * dy;
        q[(b, 2)] -= wb * c * dz;
    }
}

fn gaussian_weight(d: f32, sigma: f32) -> f32 {
    (- (d * d) / (2.0 * sigma * sigma)).exp()
}

/// Evaluate the cubic f(t) = (p(t) - a(t)) · [(b(t) - a(t)) × (c(t) - a(t))]
/// which is zero when p, a, b, c are coplanar.
/// All positions are linearly interpolated from prev to curr.
#[inline]
fn coplanarity(
    p0: na::Vector3<f32>, p1: na::Vector3<f32>, // vertex
    a0: na::Vector3<f32>, a1: na::Vector3<f32>, // triangle verts
    b0: na::Vector3<f32>, b1: na::Vector3<f32>,
    c0: na::Vector3<f32>, c1: na::Vector3<f32>,
    t: f32,
) -> f32 {
    let p = p0 + t * (p1 - p0);
    let a = a0 + t * (a1 - a0);
    let b = b0 + t * (b1 - b0);
    let c = c0 + t * (c1 - c0);
    (p - a).dot(&((b - a).cross(&(c - a))))
}

/// Find the earliest t in [t0, t1] where f(t) = 0, using derivative roots
/// to subdivide into monotone intervals then bisecting each.
/// Returns None if no root exists in the interval.
fn find_earliest_root(
    f: impl Fn(f32) -> f32,
    df: impl Fn(f32) -> f32,
    t0: f32,
    t1: f32,
    tol: f32,
) -> Option<f32> {
    // Find the two roots of the derivative (quadratic) to get monotone sub-intervals.
    // We sample df at a few points to find sign changes as a simple robust fallback.
    // For a true cubic, df is quadratic — solve it analytically via the quadratic formula
    // applied to sampled coefficients. Here we use interval subdivision for robustness.
    let n_intervals = 8; // subdivide [t0,t1] to find sign changes of f
    let step = (t1 - t0) / n_intervals as f32;

    let mut earliest: Option<f32> = None;

    let mut fa = f(t0);
    for i in 0..n_intervals {
        let ta = t0 + i as f32 * step;
        let tb = ta + step;
        let fb = f(tb);

        if fa * fb <= 0.0 {
            // Sign change — bisect to find root.
            let mut lo = ta;
            let mut hi = tb;
            let mut flo = fa;
            for _ in 0..32 {
                if (hi - lo) < tol { break; }
                let mid = (lo + hi) * 0.5;
                let fmid = f(mid);
                if flo * fmid <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fmid;
                }
            }
            let root = (lo + hi) * 0.5;
            if earliest.map_or(true, |e| root < e) {
                earliest = Some(root);
            }
            // We want the earliest root so stop after finding one
            // (intervals are ordered left to right).
            break;
        }

        fa = fb;
    }

    earliest
}

/// Returns barycentric coords (u, v, w) of point p projected onto plane of triangle (a,b,c).
/// Returns None if p is outside the triangle.
fn barycentric_in_triangle(
    p: na::Vector3<f32>,
    a: na::Vector3<f32>,
    b: na::Vector3<f32>,
    c: na::Vector3<f32>,
) -> Option<(f32, f32, f32)> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let n = ab.cross(&ac);
    let denom = n.dot(&n);
    if denom < 1e-12 { return None; } // degenerate triangle

    let u = n.dot(&(ab.cross(&ap))) / denom; // weight for c... wait, let's be explicit:
    // Standard: p = a*wa + b*wb + c*wc
    let wc = n.dot(&(ab.cross(&ap))) / denom;
    let wb = n.dot(&(ap.cross(&ac))) / denom;
    let wa = 1.0 - wb - wc;

    if wa >= -1e-4 && wb >= -1e-4 && wc >= -1e-4 {
        Some((wa, wb, wc))
    } else {
        None
    }
}

/// CCD vertex-triangle test.
/// Returns Some((t, wa, wb, wc)) — collision time and barycentric coords at t —
/// or None if no collision occurs in [0, 1].
pub fn ccd_vertex_triangle(
    p0: na::Vector3<f32>, p1: na::Vector3<f32>,
    a0: na::Vector3<f32>, a1: na::Vector3<f32>,
    b0: na::Vector3<f32>, b1: na::Vector3<f32>,
    c0: na::Vector3<f32>, c1: na::Vector3<f32>,
    thickness: f32,
) -> Option<(f32, f32, f32, f32)> {
    let f  = |t: f32| coplanarity(p0, p1, a0, a1, b0, b1, c0, c1, t);
    let df = |t: f32| {
        let eps = 1e-5;
        (f(t + eps) - f(t - eps)) / (2.0 * eps)
    };

    let t_coplanar = find_earliest_root(f, df, 0.0, 1.0, 1e-6)?;

    // At t_coplanar, check if p is actually inside the triangle.
    let p = p0 + t_coplanar * (p1 - p0);
    let a = a0 + t_coplanar * (a1 - a0);
    let b = b0 + t_coplanar * (b1 - b0);
    let c = c0 + t_coplanar * (c1 - c0);

    // Also enforce thickness — reject if they never get within thickness distance.
    let closest = closest_point_on_triangle(p, a, b, c);
    if (p - closest).norm() > thickness * 2.0 {
        return None;
    }

    barycentric_in_triangle(p, a, b, c)
        .map(|(wa, wb, wc)| (t_coplanar, wa, wb, wc))
}

// ── Closest point on triangle ─────────────────────────────────────────────────

/// Returns the closest point on triangle (a, b, c) to point p.
/// Uses Ericson's barycentric region method (Real-Time Collision Detection §5.1.5).
pub fn closest_point_on_triangle(
    p: na::Vector3<f32>,
    a: na::Vector3<f32>,
    b: na::Vector3<f32>,
    c: na::Vector3<f32>,
) -> na::Vector3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a; // vertex region A
    }

    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b; // vertex region B
    }

    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c; // vertex region C
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + v * ab; // edge region AB
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + w * ac; // edge region AC
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b); // edge region BC
    }

    // Inside face region
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + v * ab + w * ac
}

pub struct DragInfluence {
    vi: usize,
    alpha: f32,
    offset: na::Vector3<f32>,
}

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
    /// 1-ring neighbor sets: vertex_neighbors[i] is the set of vertex indices
    /// directly connected to vertex i by an edge. Used to exclude adjacent
    /// triangles from self-collision broad-phase.
    pub vertex_neighbors: Vec<HashSet<u32>>,
    /// Unique mesh edges [a, b] (a < b). Used by distance-constraint solver.
    pub edges: Vec<[u32; 2]>,
    /// Rest length for each entry in `edges`.
    pub edge_rest_lengths: Vec<f32>,
    /// Rest length of the cross-diagonal (opp0 ↔ opp1) for each bending diamond.
    pub diamond_rest_diag: Vec<f32>,
    /// Index of the vertex currently being dragged, if any.
    pub clicked_vertex: Option<usize>,
    pub dragging_vertices: Option<Vec<DragInfluence>>,
    /// Target position for the dragged vertex in clip space.
    pub mouse_pos: [f32; 3],
    /// Spatial hash over triangles, rebuilt each frame after constraint projection.
    pub triangle_hash: TriangleSpatialHash,

}

impl ClothSim {
    pub fn from_cloth(cloth: &Cloth) -> Self {
        let n = cloth.resolution as usize;
        let num_verts = n * n;

        // Set up configurational matrices
        let mut q = Positions::zeros(num_verts);
        for (i, pos) in cloth.positions.iter().enumerate() {
            q[(i, 0)] = pos[0];
            q[(i, 1)] = pos[1];
            q[(i, 2)] = pos[2];
        }
        let q_rest = q.clone();
        let q_prev = q.clone();
        let v = Positions::zeros(num_verts);

        // Compute inverse masses
        let mut w = na::DVector::from_element(num_verts, 1.0f32);
        w[(n - 1) * n] = 0.0;           // upper-left  pinned
        w[(n - 1) * n + (n - 1)] = 0.0; // upper-right pinned

        // Compute Face matrices ([tl,tr,br] + [tl,br,bl])
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
        let vertex_neighbors = build_vertex_neighbors(&faces, num_verts);

        let edges = build_edges(&faces);
        let edge_rest_lengths: Vec<f32> = edges.iter().map(|&[a, b]| {
            let d = q_rest.row(a as usize) - q_rest.row(b as usize);
            d.norm()
        }).collect();
        let diamond_rest_diag: Vec<f32> = diamonds.iter().map(|&[_, _, c, d]| {
            let delta = q_rest.row(c as usize) - q_rest.row(d as usize);
            delta.norm()
        }).collect();

        // Cloth spans [-0.9, 0.9] = 1.8 units; rest edge = 1.8/(n-1).
        // Cell size ≈ 2× rest edge so each triangle spans ~1 cell on average.
        let cell_size = 3.6 / (n as f32 - 1.0);
        let mut triangle_hash = TriangleSpatialHash::new(cell_size);
        triangle_hash.rebuild(&q, &q_prev, &faces);

        Self {
            q, q_rest, q_prev, v, w, faces, diamonds, vertex_neighbors,
            edges, edge_rest_lengths, diamond_rest_diag,
            clicked_vertex: None,
            dragging_vertices: None,
            mouse_pos: [0.0; 3],
            triangle_hash,
        }
    }

    pub fn step(&mut self, params: &SimParams) {
        let dt = params.time_step as f32;
        let n  = self.q.nrows();

        // ── 1. Save state, apply forces, integrate ────────────────────────────
        self.q_prev.copy_from(&self.q);

        if params.gravity_enabled {
            let g = params.gravity_g as f32;
            for i in 0..n {
                if self.w[i] > 0.0 {
                    self.v[(i, 1)] += g * dt;
                }
            }
        }

        if let Some(v) = self.clicked_vertex {
            self.dragging_vertices = Some(self.build_drag_influences(v, params.pulling_area as usize));
        }

        for i in 0..n {
            if self.w[i] > 0.0 {
                self.q[(i, 0)] += self.v[(i, 0)] * dt;
                self.q[(i, 1)] += self.v[(i, 1)] * dt;
                self.q[(i, 2)] += self.v[(i, 2)] * dt;
            }
        }

        // ── 2. Constraint loop (no collision detection here) ──────────────────
        for _ in 0..params.constraint_iters {
            if params.stretch_enabled {
                let sw = params.stretch_weight as f32;
                if params.use_distance_constraints {
                    for ei in 0..self.edges.len() {
                        let [a, b] = self.edges[ei];
                        apply_distance_constraint(
                            &mut self.q, &self.w,
                            a as usize, b as usize,
                            self.edge_rest_lengths[ei], sw,
                        );
                    }
                } else {
                    for fi in 0..self.faces.nrows() {
                        let idx = [
                            self.faces[(fi, 0)],
                            self.faces[(fi, 1)],
                            self.faces[(fi, 2)],
                        ];
                        apply_constraint(&mut self.q, &self.q_rest, &idx, sw);
                    }
                }
            }

            if params.bending_enabled {
                let bw = params.bending_weight as f32;
                if params.use_distance_constraints {
                    for di in 0..self.diamonds.len() {
                        let [_, _, c, d] = self.diamonds[di];
                        apply_distance_constraint(
                            &mut self.q, &self.w,
                            c as usize, d as usize,
                            self.diamond_rest_diag[di], bw,
                        );
                    }
                } else {
                    for di in 0..self.diamonds.len() {
                        let idx = self.diamonds[di];
                        apply_constraint(&mut self.q, &self.q_rest, &idx, bw);
                    }
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
                if let Some(_) = self.clicked_vertex {
                    let mp = na::Vector3::new(self.mouse_pos[0], self.mouse_pos[1], self.mouse_pos[2]);
                    let base = params.pulling_weight as f32;
                    if let Some(verts) = &self.dragging_vertices {
                        for inf in verts {
                            let target = mp + inf.offset;
                            let qi = self.q.row(inf.vi).transpose();
                            let w = base * inf.alpha;
                            let updated = (1.0 - w) * qi + w * target;
                            self.q.row_mut(inf.vi).copy_from(&updated.transpose());
                        }
                    }
                }
            }
        }

        // ── 3. CCD self-collision pass ────────────────────────────────────────────
        if params.self_collision_enabled {
            let threshold = params.self_collision_threshold as f32;

            self.triangle_hash.rebuild(&self.q, &self.q_prev, &self.faces);
            let sc_pairs = self.close_vertex_triangle_pairs(threshold);

            // Collect and sort by earliest collision time so we resolve in order.
            let mut events: Vec<(f32, usize, u32, f32, f32, f32)> = Vec::new(); // (t, vi, fi, wa, wb, wc)

            for (vi, fi) in sc_pairs {
                let i0 = self.faces[(fi as usize, 0)] as usize;
                let i1 = self.faces[(fi as usize, 1)] as usize;
                let i2 = self.faces[(fi as usize, 2)] as usize;

                let p0 = na::Vector3::new(self.q_prev[(vi,0)], self.q_prev[(vi,1)], self.q_prev[(vi,2)]);
                let p1 = na::Vector3::new(self.q[(vi,0)],      self.q[(vi,1)],      self.q[(vi,2)]);
                let a0 = na::Vector3::new(self.q_prev[(i0,0)], self.q_prev[(i0,1)], self.q_prev[(i0,2)]);
                let a1 = na::Vector3::new(self.q[(i0,0)],      self.q[(i0,1)],      self.q[(i0,2)]);
                let b0 = na::Vector3::new(self.q_prev[(i1,0)], self.q_prev[(i1,1)], self.q_prev[(i1,2)]);
                let b1 = na::Vector3::new(self.q[(i1,0)],      self.q[(i1,1)],      self.q[(i1,2)]);
                let c0 = na::Vector3::new(self.q_prev[(i2,0)], self.q_prev[(i2,1)], self.q_prev[(i2,2)]);
                let c1 = na::Vector3::new(self.q[(i2,0)],      self.q[(i2,1)],      self.q[(i2,2)]);

                if let Some((t, wa, wb, wc)) = ccd_vertex_triangle(p0, p1, a0, a1, b0, b1, c0, c1, threshold) {
                    events.push((t, vi, fi, wa, wb, wc));
                }
            }

            // Resolve earliest collisions first.
            events.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

            for (_, vi, fi, wa, wb, wc) in events {
                let i0 = self.faces[(fi as usize, 0)] as usize;
                let i1 = self.faces[(fi as usize, 1)] as usize;
                let i2 = self.faces[(fi as usize, 2)] as usize;

                // Compute normal from triangle at t=1 (post-constraint positions).
                let a = na::Vector3::new(self.q[(i0,0)], self.q[(i0,1)], self.q[(i0,2)]);
                let b = na::Vector3::new(self.q[(i1,0)], self.q[(i1,1)], self.q[(i1,2)]);
                let c = na::Vector3::new(self.q[(i2,0)], self.q[(i2,1)], self.q[(i2,2)]);
                let p = na::Vector3::new(self.q[(vi, 0)], self.q[(vi, 1)], self.q[(vi, 2)]);

                let n_hat = (b - a).cross(&(c - a));
                let n_len = n_hat.norm();
                if n_len < 1e-12 { continue; }
                let n_hat = n_hat / n_len;

                // Orient normal toward the vertex.
                let n_hat = if n_hat.dot(&(p - a)) < 0.0 { -n_hat } else { n_hat };

                let pen = threshold - n_hat.dot(&(p - a));
                if pen <= 0.0 { continue; }

                // Mass-weighted correction using barycentric coords.
                let wv = self.w[vi];
                let w0 = self.w[i0] * wa;
                let w1 = self.w[i1] * wb;
                let w2 = self.w[i2] * wc;
                let total_w = wv + w0 + w1 + w2;
                if total_w < 1e-12 { continue; }

                if wv > 0.0 {
                    let delta = (wv / total_w) * pen;
                    self.q[(vi, 0)] += n_hat[0] * delta;
                    self.q[(vi, 1)] += n_hat[1] * delta;
                    self.q[(vi, 2)] += n_hat[2] * delta;
                }
                for (ti, wi) in [(i0, w0), (i1, w1), (i2, w2)] {
                    if self.w[ti] > 0.0 {
                        let delta = (wi / total_w) * pen;
                        self.q[(ti, 0)] -= n_hat[0] * delta;
                        self.q[(ti, 1)] -= n_hat[1] * delta;
                        self.q[(ti, 2)] -= n_hat[2] * delta;
                    }
                }
            }
        }

        // ── 4. Velocity update ────────────────────────────────────────────────
        // Collision responses are baked in automatically: displaced vertices
        // get corrected velocities for free from (q - q_prev) / dt.
        let inv_dt = 1.0 / dt;
        for i in 0..n {
            if self.w[i] > 0.0 {
                self.v[(i, 0)] = (self.q[(i, 0)] - self.q_prev[(i, 0)]) * inv_dt;
                self.v[(i, 1)] = (self.q[(i, 1)] - self.q_prev[(i, 1)]) * inv_dt;
                self.v[(i, 2)] = (self.q[(i, 2)] - self.q_prev[(i, 2)]) * inv_dt;
            }
        }
    }

    /// Returns all (vertex_index, face_index) pairs where the vertex lies within
    /// `threshold` world units of a triangle it does not belong to.
    ///
    /// Uses `triangle_hash` for broad-phase culling
    pub fn close_vertex_triangle_pairs(&self, threshold: f32) -> Vec<(usize, u32)> {
        let mut pairs = Vec::new();
        let t2 = threshold * threshold;

        for vi in 0..self.q.nrows() {
            let px0 = self.q_prev[(vi, 0)];
            let py0 = self.q_prev[(vi, 1)];
            let pz0 = self.q_prev[(vi, 2)];

            let px1 = self.q[(vi, 0)];
            let py1 = self.q[(vi, 1)];
            let pz1 = self.q[(vi, 2)];

            let min = [
                px0.min(px1) - threshold,
                py0.min(py1) - threshold,
                pz0.min(pz1) - threshold,
            ];

            let max = [
                px0.max(px1) + threshold,
                py0.max(py1) + threshold,
                pz0.max(pz1) + threshold,
            ];

            let candidates = self.triangle_hash.query_aabb(min, max);

            for fi in candidates {
                let i0 = self.faces[(fi as usize, 0)] as usize;
                let i1 = self.faces[(fi as usize, 1)] as usize;
                let i2 = self.faces[(fi as usize, 2)] as usize;

                // Skip triangles that own this vertex or share an edge with it.
                // Without this, topologically adjacent triangles generate phantom
                // repulsion forces whenever the cloth folds and they come within threshold.
                if vi == i0 || vi == i1 || vi == i2 {
                    continue;
                }
                let nb = &self.vertex_neighbors[vi];
                if nb.contains(&(i0 as u32)) || nb.contains(&(i1 as u32)) || nb.contains(&(i2 as u32)) {
                    continue;
                }
                pairs.push((vi, fi));
            }
        }

        pairs
    }

    pub fn build_drag_influences(
        &self,
        center: usize,
        max_hops: usize,
    ) -> Vec<DragInfluence> {
        use std::collections::VecDeque;

        let center_pos = self.q.row(center).transpose();
        let mut dist = vec![usize::MAX; self.q.nrows()];
        let mut q = VecDeque::new();

        dist[center] = 0;
        q.push_back(center);

        let mut out = Vec::new();

        while let Some(v) = q.pop_front() {
            let d = dist[v];
            if d > max_hops {
                continue;
            }

            let alpha = gaussian_weight(d as f32, 2.0);

            if alpha > 0.0 {
                let pos = self.q.row(v).transpose();
                out.push(DragInfluence {
                    vi: v,
                    alpha,
                    offset: pos - center_pos,
                });
            }

            if d == max_hops {
                continue;
            }

            for &nb in &self.vertex_neighbors[v] {
                let nb = nb as usize;
                if dist[nb] == usize::MAX {
                    dist[nb] = d + 1;
                    q.push_back(nb);
                }
            }
        }

        out
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
