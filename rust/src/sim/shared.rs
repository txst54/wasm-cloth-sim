use std::collections::{HashMap, HashSet, VecDeque};
use web_sys::console;

use nalgebra as na;

use crate::{cloth::Cloth, gpu::GpuContext, params::SimParams};

/// Nx3 matrix of vertex positions / velocities.
pub type Positions = na::OMatrix<f32, na::Dyn, na::Const<3>>;

/// Mx3 matrix of triangle vertex indices.
pub type Faces = na::OMatrix<u32, na::Dyn, na::Const<3>>;

#[inline]
fn now_ms() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

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

pub struct EdgeSpatialHash {
    pub cell_size: f32,
    inv_cell_size: f32,
    pub cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl EdgeSpatialHash {
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

    pub fn rebuild(
        &mut self,
        q: &Positions,
        q_prev: &Positions,
        edges: &[[u32; 2]],
    ) {
        self.cells.clear();

        for (ei, &[v0, v1]) in edges.iter().enumerate() {
            let _ = ei; // if unused later
            let v0 = v0 as usize;
            let v1 = v1 as usize;

            let min_x = q[(v0, 0)].min(q[(v1, 0)])
                .min(q_prev[(v0, 0)]).min(q_prev[(v1, 0)]);
            let min_y = q[(v0, 1)].min(q[(v1, 1)])
                .min(q_prev[(v0, 1)]).min(q_prev[(v1, 1)]);
            let min_z = q[(v0, 2)].min(q[(v1, 2)])
                .min(q_prev[(v0, 2)]).min(q_prev[(v1, 2)]);

            let max_x = q[(v0, 0)].max(q[(v1, 0)])
                .max(q_prev[(v0, 0)]).max(q_prev[(v1, 0)]);
            let max_y = q[(v0, 1)].max(q[(v1, 1)])
                .max(q_prev[(v0, 1)]).max(q_prev[(v1, 1)]);
            let max_z = q[(v0, 2)].max(q[(v1, 2)])
                .max(q_prev[(v0, 2)]).max(q_prev[(v1, 2)]);

            let (cx0, cy0, cz0) = self.cell_of(min_x, min_y, min_z);
            let (cx1, cy1, cz1) = self.cell_of(max_x, max_y, max_z);

            for cx in cx0..=cx1 {
                for cy in cy0..=cy1 {
                    for cz in cz0..=cz1 {
                        self.cells.entry((cx, cy, cz)).or_default().push(ei as u32);
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
                    if let Some(edges) = self.cells.get(&(cx,cy,cz)) {
                        result.extend(edges);
                    }
                }
            }
        }
        result
    }

    pub fn candidate_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::<(u32, u32)>::new();

        // Optional: rough reserve to reduce reallocations.
        // This estimates the total number of emitted pairs before dedup.
        let mut estimated = 0usize;
        for edges in self.cells.values() {
            let k = edges.len();
            if k >= 2 {
                estimated += k * (k - 1) / 2;
            }
        }
        pairs.reserve(estimated);

        for edges in self.cells.values() {
            for i in 0..edges.len() {
                for j in (i + 1)..edges.len() {
                    let a = edges[i];
                    let b = edges[j];
                    pairs.push(if a < b { (a, b) } else { (b, a) });
                }
            }
        }

        pairs.sort_unstable();
        pairs.dedup();

        pairs
            .into_iter()
            .map(|(a, b)| (a as usize, b as usize))
            .collect()
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

pub(super) fn build_vertex_to_faces(faces: &Faces, num_verts: usize) -> Vec<Vec<u32>> {
    let mut vtf = vec![Vec::new(); num_verts];
    for fi in 0..faces.nrows() {
        for k in 0..3 {
            vtf[faces[(fi, k)] as usize].push(fi as u32);
        }
    }
    vtf
}

pub(super) fn build_vertex_to_edges(edges: &[[u32; 2]], num_verts: usize) -> Vec<Vec<u32>> {
    let mut vte = vec![Vec::new(); num_verts];
    for (ei, &[a, b]) in edges.iter().enumerate() {
        vte[a as usize].push(ei as u32);
        vte[b as usize].push(ei as u32);
    }
    vte
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

/// Find the earliest root of a cubic f(t) in [t0, t1].
///
/// Uses df (the derivative, a quadratic) to find the two critical points,
/// subdividing [t0,t1] into at most 3 guaranteed monotone intervals.
/// Bisects each interval where a sign change exists.
///
/// `thickness` is used to accept roots where f(t) is within a signed volume
/// tolerance rather than requiring exact zero — this prevents disappearing
/// roots due to float imprecision on near-grazing contacts.
fn find_earliest_root(
    f: impl Fn(f32) -> f32,
    df: impl Fn(f32) -> f32,
    t0: f32,
    t1: f32,
    tol: f32,
    thickness: f32,
) -> Option<f32> {

    let tm = (t0 + t1) * 0.5;
    let df0 = df(t0);
    let dfm = df(tm);
    let df1 = df(t1);

    let h = tm - t0;
    let a_coeff = (df1 - 2.0 * dfm + df0) / (2.0 * h * h);
    let b_coeff = (dfm - df0 - a_coeff * (tm * tm - t0 * t0)) / (tm - t0);
    let c_coeff = df0 - a_coeff * t0 * t0 - b_coeff * t0;

    let mut critical: Vec<f32> = Vec::with_capacity(2);
    if a_coeff.abs() < 1e-10 {
        // Degenerate to linear df — one critical point.
        if b_coeff.abs() > 1e-10 {
            let r = -c_coeff / b_coeff;
            if r > t0 && r < t1 { critical.push(r); }
        }
    } else {
        let disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            let r1 = (-b_coeff - sq) / (2.0 * a_coeff);
            let r2 = (-b_coeff + sq) / (2.0 * a_coeff);
            if r1 > t0 && r1 < t1 { critical.push(r1); }
            if r2 > t0 && r2 < t1 { critical.push(r2); }
        }
    }


    let extras = [t0 + (t1-t0)*0.25, t0 + (t1-t0)*0.75];
    for &e in &extras {
        if !critical.contains(&e) {
            critical.push(e);
        }
    }

    critical.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut boundaries = Vec::with_capacity(4);
    boundaries.push(t0);
    boundaries.extend_from_slice(&critical);
    boundaries.push(t1);

    let edge_len = 1.0 / 31.0; //todo change to actual tri edge length or edge length
    let volume_eps = thickness * edge_len * edge_len;
    let f0 = f(t0);
    let f1 = f(t1);

    let is_collision_side = |v: f32| -> bool {
        if f0 > volume_eps {
            v <= volume_eps
        } else if f0 < -volume_eps {
            v >= -volume_eps
        } else {
            true
        }
    };

    let mut earliest: Option<f32> = None;

    for w in boundaries.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if hi - lo < tol { continue; }

        let flo = f(lo);
        let fhi = f(hi);

        let lo_hit = is_collision_side(flo);
        let hi_hit = is_collision_side(fhi);

        if lo_hit {
            if earliest.map_or(true, |e| lo < e) {
                earliest = Some(lo);
            }
            break;
        }

        if !hi_hit {
            continue;
        }

        // Sign change exists in (lo, hi) — bisect.
        let mut a = lo;
        let mut b = hi;
        let mut fa = flo;
        for _ in 0..48 {
            if b - a < tol { break; }
            let mid = (a + b) * 0.5;
            let fmid = f(mid);
            if is_collision_side(fmid) {
                b = mid;
            } else {
                a = mid;
                fa = fmid;
            }
        }
        let root = (a + b) * 0.5;
        if earliest.map_or(true, |e| root < e) {
            earliest = Some(root);
        }
        // Intervals are ordered — first sign change found is the earliest.
        break;
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
        let t_hi = (t + eps).min(1.0);
        let t_lo = (t - eps).max(0.0);
        (f(t_hi) - f(t_lo)) / (t_hi - t_lo)
    };

    let t_coplanar = find_earliest_root(f, df, 0.0, 1.0, 1e-6, thickness)?;

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

/// CCD edge-edge test.
/// Returns Some((t, s, t_edge)) where:
///   t       = collision time in [0,1]
///   s       = parameter along edge (p,q) at contact, in [0,1]
///   t_edge  = parameter along edge (a,b) at contact, in [0,1]
pub fn ccd_edge_edge(
    p0: na::Vector3<f32>, p1: na::Vector3<f32>, // edge 0 start, at t=0 and t=1
    q0: na::Vector3<f32>, q1: na::Vector3<f32>, // edge 0 end
    a0: na::Vector3<f32>, a1: na::Vector3<f32>, // edge 1 start
    b0: na::Vector3<f32>, b1: na::Vector3<f32>, // edge 1 end
    thickness: f32,
) -> Option<(f32, f32, f32)> {
    // Same coplanarity condition as VT: four points coplanar when scalar triple product = 0.
    // Here the four points are the two edge endpoints: p, q, a, b.
    let f = |t: f32| {
        let p = p0 + t * (p1 - p0);
        let q = q0 + t * (q1 - q0);
        let a = a0 + t * (a1 - a0);
        let b = b0 + t * (b1 - b0);
        (q - p).cross(&(b - a)).dot(&(a - p))
    };
    let df = |t: f32| {
        let eps = 1e-5;
        let t_hi = (t + eps).min(1.0);
        let t_lo = (t - eps).max(0.0);
        (f(t_hi) - f(t_lo)) / (t_hi - t_lo)
    };

    let t_coplanar = find_earliest_root(f, df, 0.0, 1.0, 1e-6, thickness)?;

    // At t*, reconstruct positions and check that the crossing point
    // actually lies within both edge extents [0,1].
    let p = p0 + t_coplanar * (p1 - p0);
    let q = q0 + t_coplanar * (q1 - q0);
    let a = a0 + t_coplanar * (a1 - a0);
    let b = b0 + t_coplanar * (b1 - b0);

    let pq = q - p;
    let ab = b - a;

    // Find closest point parameters s (on pq) and u (on ab).
    // Solve the linear system from minimizing |(p + s*pq) - (a + u*ab)|².
    let d = p - a;
    let pq_pq = pq.dot(&pq);
    let ab_ab = ab.dot(&ab);
    let pq_ab = pq.dot(&ab);
    let denom = pq_pq * ab_ab - pq_ab * pq_ab;

    if denom.abs() < 1e-12 {
        // Edges are parallel at t* — no clean crossing.
        return None;
    }

    let s = (pq_ab * ab.dot(&d) - ab_ab * pq.dot(&d)) / denom;
    let u = (pq_pq * ab.dot(&d) - pq_ab * pq.dot(&d)) / denom;

    // Both parameters must be within the edge extents.
    if s < -1e-4 || s > 1.0 + 1e-4 || u < -1e-4 || u > 1.0 + 1e-4 {
        return None;
    }

    // Thickness filter: closest points must actually be within threshold.
    let closest_pq = p + s * pq;
    let closest_ab = a + u * ab;
    if (closest_pq - closest_ab).norm() > thickness * 2.0 {
        return None;
    }

    Some((t_coplanar, s.clamp(0.0, 1.0), u.clamp(0.0, 1.0)))
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

pub fn closest_points_on_segments(
    p0: na::Vector3<f32>,
    p1: na::Vector3<f32>,
    q0: na::Vector3<f32>,
    q1: na::Vector3<f32>,
) -> (f32, f32, na::Vector3<f32>, na::Vector3<f32>) {
    let d1 = p1 - p0; // direction of segment P
    let d2 = q1 - q0; // direction of segment Q
    let r = p0 - q0;

    let a = d1.dot(&d1); // squared length of segment P
    let e = d2.dot(&d2); // squared length of segment Q
    let f = d2.dot(&r);

    let eps = 1e-8f32;

    let mut s;
    let t : f64;

    // Both segments degenerate to points.
    if a <= eps && e <= eps {
        return (0.0, 0.0, p0, q0);
    }

    // First segment degenerates to a point.
    if a <= eps {
        s = 0.0;
        let t = (f / e).clamp(0.0, 1.0);
        let cp = p0;
        let cq = q0 + d2 * t;
        return (s, t, cp, cq);
    }

    let c = d1.dot(&r);

    // Second segment degenerates to a point.
    if e <= eps {
        let t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
        let cp = p0 + d1 * s;
        let cq = q0;
        return (s, t, cp, cq);
    }

    let b = d1.dot(&d2);
    let denom = a * e - b * b;

    // Segments not parallel.
    if denom.abs() > eps {
        s = ((b * f - c * e) / denom).clamp(0.0, 1.0);
    } else {
        // Parallel: choose an arbitrary s first, then clamp t and recompute s.
        s = 0.0;
    }

    let tnom = b * s + f;

    let t = if tnom <= 0.0 {
        s = (-c / a).clamp(0.0, 1.0);
        0.0
    } else if tnom >= e {
        s = ((b - c) / a).clamp(0.0, 1.0);
        1.0
    } else {
        tnom / e
    };

    let cp = p0 + d1 * s;
    let cq = q0 + d2 * t;
    (s, t, cp, cq)
}

// ── DragInfluence ─────────────────────────────────────────────────────────────

pub struct DragInfluence {
    pub vi: usize,
    pub alpha: f32,
    pub offset: na::Vector3<f32>,
}

// ── ClothSimCore ──────────────────────────────────────────────────────────────

/// Shared XPBD state for all fabric-type simulations.
pub struct ClothSimCore {
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
    pub edge_hash:          EdgeSpatialHash,
    pub vertex_to_faces:    Vec<Vec<u32>>,
    pub vertex_to_edges:    Vec<Vec<u32>>,
}

impl ClothSimCore {
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
        let vertex_to_faces = build_vertex_to_faces(&faces, num_verts);
        let vertex_to_edges = build_vertex_to_edges(&edges, num_verts);

        let tri_cell_size = 3.6 / (n as f32 - 1.0);
        let mut triangle_hash = TriangleSpatialHash::new(tri_cell_size);
        triangle_hash.rebuild(&q, &q_prev, &faces);

        let rest_edge = edge_rest_lengths[0]; // or average
        let edge_cell_size = (2.0 * rest_edge).max(0.02);

        let mut edge_hash = EdgeSpatialHash::new(tri_cell_size);
        edge_hash.rebuild(&q, &q_prev, &edges);

        let stretch_lambdas = vec![0.0f32; edges.len()];
        let bend_lambdas = vec![0.0f32; diamonds.len()];

        Self {
            q, q_rest, q_prev, v, w, faces, diamonds, vertex_neighbors,
            edges, edge_rest_lengths, diamond_rest_diag,
            stretch_lambdas, bend_lambdas,
            clicked_vertex: None, dragging_vertices: None, mouse_pos: [0.0; 3],
            triangle_hash,
            edge_hash,
            vertex_to_faces,
            vertex_to_edges,
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
        let vertex_to_faces = build_vertex_to_faces(&faces, num_verts);
        let vertex_to_edges = build_vertex_to_edges(&edges, num_verts);

        let stretch_lambdas = vec![0.0f32; edges.len()];
        let bend_lambdas = vec![0.0f32; diamonds.len()];

        let avg_edge_len = if edge_rest_lengths.is_empty() {
            0.1
        } else {
            edge_rest_lengths.iter().sum::<f32>() / edge_rest_lengths.len() as f32
        };
        let cell_size = avg_edge_len * 2.0;
        let mut triangle_hash = TriangleSpatialHash::new(cell_size);
        triangle_hash.rebuild(&q, &q_prev, &faces);

        let mut edge_hash = EdgeSpatialHash::new(cell_size);
        edge_hash.rebuild(&q, &q_prev, &edges);

        Self {
            q, q_rest, q_prev, v, w, faces, diamonds, vertex_neighbors,
            edges, edge_rest_lengths, diamond_rest_diag,
            stretch_lambdas, bend_lambdas,
            clicked_vertex: None, dragging_vertices: None, mouse_pos: [0.0; 3],
            triangle_hash,
            edge_hash,
            vertex_to_faces,
            vertex_to_edges,
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
                if self.w[inf.vi] == 0.0 { continue; }
                let target = mp + inf.offset;
                let qi = self.q.row(inf.vi).transpose();
                let wt = base * inf.alpha;
                let updated = (1.0 - wt) * qi + wt * target;
                self.q.row_mut(inf.vi).copy_from(&updated.transpose());
            }
        }
    }

    /// CCD-based self-collision resolution (vertex-triangle and edge-edge).
    /// Rebuilds hashes and runs a single full pass. Used by external callers
    /// (e.g. paper_sim) that manage their own iteration count.
    pub fn solve_self_collision(&mut self, params: &SimParams) {
        if !params.self_collision_enabled { return; }
        let threshold = params.self_collision_threshold as f32;
        self.triangle_hash.rebuild(&self.q, &self.q_prev, &self.faces);
        self.edge_hash.rebuild(&self.q, &self.q_prev, &self.edges);
        self.solve_self_collision_pass(threshold, None);
    }

    /// One pass of CCD self-collision resolution. When `dirty` is `Some`, only
    /// candidate pairs involving a dirty feature are generated; the spatial hashes
    /// must already be built before calling with `dirty = Some(...)`. Returns the
    /// set of vertices whose positions were modified.
    fn solve_self_collision_pass(&mut self, threshold: f32, dirty: Option<&HashSet<usize>>) -> HashSet<usize> {
        let vt_pairs = match dirty {
            None => self.close_vertex_triangle_pairs(threshold),
            Some(d) => self.close_vertex_triangle_pairs_dirty(threshold, d),
        };
        let ee_pairs = match dirty {
            None => self.close_edge_edge_pairs(threshold),
            Some(d) => self.close_edge_edge_pairs_dirty(threshold, d),
        };

        let mut vt_events: Vec<(f32, usize, u32, f32, f32, f32)> = Vec::new();

        for &(vi, fi) in &vt_pairs {
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
                vt_events.push((t, vi, fi, wa, wb, wc));
            }
        }

        let mut ee_events: Vec<(f32, usize, usize, f32, f32)> = Vec::new();

        for &(ei, ej) in &ee_pairs {
            let [pa, pb] = self.edges[ei];
            let [qa, qb] = self.edges[ej];

            let pa = pa as usize; let pb = pb as usize;
            let qa = qa as usize; let qb = qb as usize;

            let p0 = na::Vector3::new(self.q_prev[(pa,0)], self.q_prev[(pa,1)], self.q_prev[(pa,2)]);
            let p1 = na::Vector3::new(self.q[(pa,0)],      self.q[(pa,1)],      self.q[(pa,2)]);
            let q0 = na::Vector3::new(self.q_prev[(pb,0)], self.q_prev[(pb,1)], self.q_prev[(pb,2)]);
            let q1 = na::Vector3::new(self.q[(pb,0)],      self.q[(pb,1)],      self.q[(pb,2)]);
            let a0 = na::Vector3::new(self.q_prev[(qa,0)], self.q_prev[(qa,1)], self.q_prev[(qa,2)]);
            let a1 = na::Vector3::new(self.q[(qa,0)],      self.q[(qa,1)],      self.q[(qa,2)]);
            let b0 = na::Vector3::new(self.q_prev[(qb,0)], self.q_prev[(qb,1)], self.q_prev[(qb,2)]);
            let b1 = na::Vector3::new(self.q[(qb,0)],      self.q[(qb,1)],      self.q[(qb,2)]);

            if let Some((t, s, u)) = ccd_edge_edge(p0, p1, q0, q1, a0, a1, b0, b1, threshold) {
                ee_events.push((t, ei, ej, s, u));
            }
        }

        enum CollisionEvent {
            VT { t: f32, vi: usize, fi: u32, wa: f32, wb: f32, wc: f32 },
            EE { t: f32, ei: usize, ej: usize, s: f32, u: f32 },
        }

        impl CollisionEvent {
            fn t(&self) -> f32 {
                match self { Self::VT { t, .. } | Self::EE { t, .. } => *t }
            }
        }

        let mut all_events: Vec<CollisionEvent> = Vec::new();
        for (t, vi, fi, wa, wb, wc) in vt_events {
            all_events.push(CollisionEvent::VT { t, vi, fi, wa, wb, wc });
        }
        for (t, ei, ej, s, u) in ee_events {
            all_events.push(CollisionEvent::EE { t, ei, ej, s, u });
        }

        all_events.sort_unstable_by(|a, b| a.t().partial_cmp(&b.t()).unwrap());

        let mut moved: HashSet<usize> = HashSet::new();

        for event in all_events {
            match event {
                CollisionEvent::VT { t: t_star, vi, fi, wa, wb, wc } => {
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

                    // Fresh t_star against current q
                    let Some((t_star, wa, wb, wc)) = ccd_vertex_triangle(
                        p0, p1, a0, a1, b0, b1, c0, c1, threshold
                    ) else { continue; }; // prior events already resolved this one

                    let lerp = |prev: f32, curr: f32| prev + t_star * (curr - prev);

                    let p_t = na::Vector3::new(
                        lerp(self.q_prev[(vi,0)], self.q[(vi,0)]),
                        lerp(self.q_prev[(vi,1)], self.q[(vi,1)]),
                        lerp(self.q_prev[(vi,2)], self.q[(vi,2)]),
                    );
                    let a_t = na::Vector3::new(
                        lerp(self.q_prev[(i0,0)], self.q[(i0,0)]),
                        lerp(self.q_prev[(i0,1)], self.q[(i0,1)]),
                        lerp(self.q_prev[(i0,2)], self.q[(i0,2)]),
                    );
                    let b_t = na::Vector3::new(
                        lerp(self.q_prev[(i1,0)], self.q[(i1,0)]),
                        lerp(self.q_prev[(i1,1)], self.q[(i1,1)]),
                        lerp(self.q_prev[(i1,2)], self.q[(i1,2)]),
                    );
                    let c_t = na::Vector3::new(
                        lerp(self.q_prev[(i2,0)], self.q[(i2,0)]),
                        lerp(self.q_prev[(i2,1)], self.q[(i2,1)]),
                        lerp(self.q_prev[(i2,2)], self.q[(i2,2)]),
                    );

                    let n_raw = (b_t - a_t).cross(&(c_t - a_t));
                    let n_len = n_raw.norm();
                    if n_len < 1e-12 { continue; }
                    let n_hat = n_raw / n_len;

                    let n_hat = if n_hat.dot(&(p_t - a_t)) < 0.0 { -n_hat } else { n_hat };

                    let signed_dist = n_hat.dot(&(p_t - a_t));
                    let pen = threshold - signed_dist;
                    if pen <= 0.0 { continue; }

                    let wv = self.w[vi];
                    let w0 = self.w[i0] * wa;
                    let w1 = self.w[i1] * wb;
                    let w2 = self.w[i2] * wc;
                    let total_w = wv + w0 * wa + w1 * wb + w2 * wc;
                    if total_w < 1e-12 { continue; }

                    if wv > 0.0 {
                        let delta = (wv / total_w) * pen;
                        self.q[(vi, 0)] += n_hat[0] * delta;
                        self.q[(vi, 1)] += n_hat[1] * delta;
                        self.q[(vi, 2)] += n_hat[2] * delta;
                        moved.insert(vi);
                    }
                    for (ti, wi) in [(i0, w0), (i1, w1), (i2, w2)] {
                        if self.w[ti] > 0.0 {
                            let delta = (wi / total_w) * pen;
                            self.q[(ti, 0)] -= n_hat[0] * delta;
                            self.q[(ti, 1)] -= n_hat[1] * delta;
                            self.q[(ti, 2)] -= n_hat[2] * delta;
                            moved.insert(ti);
                        }
                    }
                }

                CollisionEvent::EE { t: t_star, ei, ej, s, u } => {
                    let [pa, pb] = self.edges[ei];
                    let [qa, qb] = self.edges[ej];
                    let pa = pa as usize; let pb = pb as usize;
                    let qa = qa as usize; let qb = qb as usize;

                    let p0 = na::Vector3::new(self.q_prev[(pa,0)], self.q_prev[(pa,1)], self.q_prev[(pa,2)]);
                    let p1 = na::Vector3::new(self.q[(pa,0)],      self.q[(pa,1)],      self.q[(pa,2)]);
                    let q0 = na::Vector3::new(self.q_prev[(pb,0)], self.q_prev[(pb,1)], self.q_prev[(pb,2)]);
                    let q1 = na::Vector3::new(self.q[(pb,0)],      self.q[(pb,1)],      self.q[(pb,2)]);
                    let a0 = na::Vector3::new(self.q_prev[(qa,0)], self.q_prev[(qa,1)], self.q_prev[(qa,2)]);
                    let a1 = na::Vector3::new(self.q[(qa,0)],      self.q[(qa,1)],      self.q[(qa,2)]);
                    let b0 = na::Vector3::new(self.q_prev[(qb,0)], self.q_prev[(qb,1)], self.q_prev[(qb,2)]);
                    let b1 = na::Vector3::new(self.q[(qb,0)],      self.q[(qb,1)],      self.q[(qb,2)]);

                    // Fresh t_star against current q
                    let Some((t_star, s, u)) = ccd_edge_edge(
                        p0, p1, q0, q1, a0, a1, b0, b1, threshold
                    ) else { continue; };

                    let lerp = |prev: f32, curr: f32| prev + t_star * (curr - prev);

                    let p_t = na::Vector3::new(
                        lerp(self.q_prev[(pa,0)], self.q[(pa,0)]),
                        lerp(self.q_prev[(pa,1)], self.q[(pa,1)]),
                        lerp(self.q_prev[(pa,2)], self.q[(pa,2)]),
                    );
                    let q_t = na::Vector3::new(
                        lerp(self.q_prev[(pb,0)], self.q[(pb,0)]),
                        lerp(self.q_prev[(pb,1)], self.q[(pb,1)]),
                        lerp(self.q_prev[(pb,2)], self.q[(pb,2)]),
                    );
                    let a_t = na::Vector3::new(
                        lerp(self.q_prev[(qa,0)], self.q[(qa,0)]),
                        lerp(self.q_prev[(qa,1)], self.q[(qa,1)]),
                        lerp(self.q_prev[(qa,2)], self.q[(qa,2)]),
                    );
                    let b_t = na::Vector3::new(
                        lerp(self.q_prev[(qb,0)], self.q[(qb,0)]),
                        lerp(self.q_prev[(qb,1)], self.q[(qb,1)]),
                        lerp(self.q_prev[(qb,2)], self.q[(qb,2)]),
                    );

                    let pq_t = q_t - p_t;
                    let ab_t = b_t - a_t;
                    let n_raw = pq_t.cross(&ab_t);
                    let n_len = n_raw.norm();
                    if n_len < 1e-12 { continue; }
                    let n_hat = n_raw / n_len;

                    let contact_pq = p_t + s * pq_t;
                    let contact_ab = a_t + u * ab_t;
                    let sep = contact_pq - contact_ab;

                    let n_hat = if n_hat.dot(&sep) < 0.0 { -n_hat } else { n_hat };

                    let signed_dist = n_hat.dot(&sep);
                    let pen = threshold - signed_dist;
                    if pen <= 0.0 { continue; }

                    let wpa = self.w[pa] * (1.0 - s);
                    let wpb = self.w[pb] * s;
                    let wqa = self.w[qa] * (1.0 - u);
                    let wqb = self.w[qb] * u;
                    let total_w = wpa * (1.0 - s) + wpb * s + wqa * (1.0 - u) + wqb * u;
                    if total_w < 1e-12 { continue; }

                    for (vi, wi, sign) in [
                        (pa, wpa,  1.0f32),
                        (pb, wpb,  1.0f32),
                        (qa, wqa, -1.0f32),
                        (qb, wqb, -1.0f32),
                    ] {
                        if self.w[vi] > 0.0 {
                            let delta = sign * (wi / total_w) * pen;
                            self.q[(vi, 0)] += n_hat[0] * delta;
                            self.q[(vi, 1)] += n_hat[1] * delta;
                            self.q[(vi, 2)] += n_hat[2] * delta;
                            moved.insert(vi);
                        }
                    }
                }
            }
        }

        moved
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
        self.reset_lambdas();
        for _ in 0..params.constraint_iters {
            self.solve_stretch(params);
            self.solve_bend(params, skip_bending);
            self.solve_pins(params);
            self.solve_pulling(params);
        }
        if params.self_collision_enabled {
            let threshold = params.self_collision_threshold as f32;
            self.triangle_hash.rebuild(&self.q, &self.q_prev, &self.faces);
            self.edge_hash.rebuild(&self.q, &self.q_prev, &self.edges);
            let mut dirty: Option<HashSet<usize>> = None;
            for _ in 0..params.self_collision_recompute_iters {
                let moved = self.solve_self_collision_pass(threshold, dirty.as_ref());
                if moved.is_empty() { break; }
                dirty = Some(moved);
            }
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

    pub fn close_edge_edge_pairs(&self, threshold: f32) -> Vec<(usize, usize)> {
        let candidates = self.edge_hash.candidate_pairs();
        let mut out = Vec::with_capacity(candidates.len());
        for (ei, ej) in candidates {
            let [a0, a1] = self.edges[ei];
            let [b0, b1] = self.edges[ej];
            if a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 { continue; }
            out.push((ei, ej));
        }
        out
    }

    /// Filtered VT candidates: only pairs where vi is dirty or fi contains a dirty vertex.
    fn close_vertex_triangle_pairs_dirty(&self, threshold: f32, dirty: &HashSet<usize>) -> Vec<(usize, u32)> {
        let mut pairs = Vec::new();

        // Pass 1: dirty vertices query all triangles via hash.
        for &vi in dirty {
            let px0=self.q_prev[(vi,0)]; let py0=self.q_prev[(vi,1)]; let pz0=self.q_prev[(vi,2)];
            let px1=self.q[(vi,0)];     let py1=self.q[(vi,1)];     let pz1=self.q[(vi,2)];
            let min = [px0.min(px1)-threshold, py0.min(py1)-threshold, pz0.min(pz1)-threshold];
            let max = [px0.max(px1)+threshold, py0.max(py1)+threshold, pz0.max(pz1)+threshold];
            for fi in self.triangle_hash.query_aabb(min, max) {
                let i0=self.faces[(fi as usize,0)] as usize;
                let i1=self.faces[(fi as usize,1)] as usize;
                let i2=self.faces[(fi as usize,2)] as usize;
                if vi==i0 || vi==i1 || vi==i2 { continue; }
                let nb = &self.vertex_neighbors[vi];
                if nb.contains(&(i0 as u32)) || nb.contains(&(i1 as u32)) || nb.contains(&(i2 as u32)) { continue; }
                pairs.push((vi, fi));
            }
        }

        // Pass 2: non-dirty vertices against dirty faces (face moved into a static vertex).
        let dirty_face_aabbs: Vec<(u32, [f32; 3], [f32; 3])> = dirty.iter()
            .flat_map(|&vi| self.vertex_to_faces[vi].iter().copied())
            .collect::<HashSet<u32>>()
            .into_iter()
            .map(|fi| {
                let v0=self.faces[(fi as usize,0)] as usize;
                let v1=self.faces[(fi as usize,1)] as usize;
                let v2=self.faces[(fi as usize,2)] as usize;
                let mut fmin = [f32::MAX; 3];
                let mut fmax = [f32::MIN; 3];
                for &v in &[v0, v1, v2] {
                    for k in 0..3 {
                        fmin[k] = fmin[k].min(self.q[(v,k)]).min(self.q_prev[(v,k)]);
                        fmax[k] = fmax[k].max(self.q[(v,k)]).max(self.q_prev[(v,k)]);
                    }
                }
                (fi, fmin, fmax)
            })
            .collect();

        if !dirty_face_aabbs.is_empty() {
            for vi in 0..self.q.nrows() {
                if dirty.contains(&vi) { continue; }
                let vx=self.q[(vi,0)]; let vy=self.q[(vi,1)]; let vz=self.q[(vi,2)];
                for &(fi, fmin, fmax) in &dirty_face_aabbs {
                    if vx < fmin[0]-threshold || vx > fmax[0]+threshold { continue; }
                    if vy < fmin[1]-threshold || vy > fmax[1]+threshold { continue; }
                    if vz < fmin[2]-threshold || vz > fmax[2]+threshold { continue; }
                    let i0=self.faces[(fi as usize,0)] as usize;
                    let i1=self.faces[(fi as usize,1)] as usize;
                    let i2=self.faces[(fi as usize,2)] as usize;
                    if vi==i0 || vi==i1 || vi==i2 { continue; }
                    let nb = &self.vertex_neighbors[vi];
                    if nb.contains(&(i0 as u32)) || nb.contains(&(i1 as u32)) || nb.contains(&(i2 as u32)) { continue; }
                    pairs.push((vi, fi));
                }
            }
        }

        pairs
    }

    /// Filtered EE candidates: only pairs where at least one edge contains a dirty vertex.
    /// Queries the edge hash per dirty edge to find spatial neighbours.
    fn close_edge_edge_pairs_dirty(&self, threshold: f32, dirty: &HashSet<usize>) -> Vec<(usize, usize)> {
        let dirty_edges: HashSet<u32> = dirty.iter()
            .flat_map(|&vi| self.vertex_to_edges[vi].iter().copied())
            .collect();

        let mut raw = Vec::<(u32, u32)>::new();

        for &ei in &dirty_edges {
            let [v0, v1] = self.edges[ei as usize];
            let v0 = v0 as usize; let v1 = v1 as usize;
            let min_x = self.q[(v0,0)].min(self.q[(v1,0)]).min(self.q_prev[(v0,0)]).min(self.q_prev[(v1,0)]);
            let min_y = self.q[(v0,1)].min(self.q[(v1,1)]).min(self.q_prev[(v0,1)]).min(self.q_prev[(v1,1)]);
            let min_z = self.q[(v0,2)].min(self.q[(v1,2)]).min(self.q_prev[(v0,2)]).min(self.q_prev[(v1,2)]);
            let max_x = self.q[(v0,0)].max(self.q[(v1,0)]).max(self.q_prev[(v0,0)]).max(self.q_prev[(v1,0)]);
            let max_y = self.q[(v0,1)].max(self.q[(v1,1)]).max(self.q_prev[(v0,1)]).max(self.q_prev[(v1,1)]);
            let max_z = self.q[(v0,2)].max(self.q[(v1,2)]).max(self.q_prev[(v0,2)]).max(self.q_prev[(v1,2)]);
            let nearby = self.edge_hash.query_aabb(
                [min_x - threshold, min_y - threshold, min_z - threshold],
                [max_x + threshold, max_y + threshold, max_z + threshold],
            );
            for ej in nearby {
                if ej == ei { continue; }
                let (a, b) = if ei < ej { (ei, ej) } else { (ej, ei) };
                raw.push((a, b));
            }
        }

        raw.sort_unstable();
        raw.dedup();

        let mut out = Vec::new();
        for (a, b) in raw {
            let [a0, a1] = self.edges[a as usize];
            let [b0, b1] = self.edges[b as usize];
            if a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 { continue; }
            out.push((a as usize, b as usize));
        }
        out
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
            if alpha > 0.0 && self.w[v] > 0.0 {
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

// ── SimCore ───────────────────────────────────────────────────────────────────

/// Top-level simulation state containing cloth and rigid body sub-simulations.
pub struct SimCore {
    pub cloth: ClothSimCore,
    pub rigid: super::rigid_sim::RigidSimCore,
}
