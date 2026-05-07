//! CPU bake of a triangle mesh into a sampled signed distance field stored
//! as a flat `Vec<f32>` representing a 3D grid. Sign is computed from the
//! face normal at the closest point — fast and correct for closed manifold
//! meshes outside of deep concavities.

use nalgebra as na;

use crate::sim::obj_loader::ObjMesh;
use crate::sim::shared::closest_point_on_triangle;

pub struct MeshSdfVolume {
    /// Flat row-major: index = z*dy*dx + y*dx + x.
    pub data:       Vec<f32>,
    pub dims:       [u32; 3],
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

#[derive(Clone, Copy)]
struct Aabb { min: [f32; 3], max: [f32; 3] }

impl Aabb {
    fn empty() -> Self { Self { min: [f32::INFINITY; 3], max: [f32::NEG_INFINITY; 3] } }
    fn expand_pt(&mut self, p: [f32; 3]) {
        for k in 0..3 {
            if p[k] < self.min[k] { self.min[k] = p[k]; }
            if p[k] > self.max[k] { self.max[k] = p[k]; }
        }
    }
    fn dist2_to_point(&self, p: [f32; 3]) -> f32 {
        let mut s = 0.0;
        for k in 0..3 {
            let d = if p[k] < self.min[k] { self.min[k] - p[k] }
                    else if p[k] > self.max[k] { p[k] - self.max[k] } else { 0.0 };
            s += d * d;
        }
        s
    }
}

enum Node {
    Leaf  { aabb: Aabb, face: u32 },
    Inner { aabb: Aabb, left: u32, right: u32 },
}

impl Node {
    fn aabb(&self) -> &Aabb {
        match self { Node::Leaf { aabb, .. } | Node::Inner { aabb, .. } => aabb }
    }
}

struct TriBvh {
    nodes: Vec<Node>,
    root:  u32,
}

impl TriBvh {
    fn build(positions: &[[f32; 3]], faces: &[[u32; 3]]) -> Self {
        let m = faces.len();
        let mut leaf_aabbs: Vec<Aabb> = faces.iter().map(|f| {
            let mut a = Aabb::empty();
            for &vi in f { a.expand_pt(positions[vi as usize]); }
            a
        }).collect();
        let mut idx: Vec<u32> = (0..m as u32).collect();
        let mut nodes: Vec<Node> = Vec::with_capacity(2 * m);
        let root = build_rec(&mut nodes, &mut leaf_aabbs, &mut idx, 0, m);
        Self { nodes, root }
    }

    /// Closest-point query: returns (distance, closest_point, face_index).
    fn nearest(&self, p: [f32; 3], positions: &[[f32; 3]], faces: &[[u32; 3]]) -> (f32, [f32; 3], u32) {
        let mut best_d2 = f32::INFINITY;
        let mut best_pt = [0.0; 3];
        let mut best_fi = 0u32;
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if node.aabb().dist2_to_point(p) >= best_d2 { continue; }
            match node {
                Node::Leaf { face, .. } => {
                    let f = faces[*face as usize];
                    let a = na::Vector3::new(
                        positions[f[0] as usize][0],
                        positions[f[0] as usize][1],
                        positions[f[0] as usize][2]);
                    let b = na::Vector3::new(
                        positions[f[1] as usize][0],
                        positions[f[1] as usize][1],
                        positions[f[1] as usize][2]);
                    let c = na::Vector3::new(
                        positions[f[2] as usize][0],
                        positions[f[2] as usize][1],
                        positions[f[2] as usize][2]);
                    let pv = na::Vector3::new(p[0], p[1], p[2]);
                    let cp = closest_point_on_triangle(pv, a, b, c);
                    let d = pv - cp;
                    let d2 = d.dot(&d);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_pt = [cp.x, cp.y, cp.z];
                        best_fi = *face;
                    }
                }
                Node::Inner { left, right, .. } => {
                    // Visit nearer child first to tighten the bound earlier.
                    let l = *left; let r = *right;
                    let dl = self.nodes[l as usize].aabb().dist2_to_point(p);
                    let dr = self.nodes[r as usize].aabb().dist2_to_point(p);
                    if dl < dr { stack.push(r); stack.push(l); }
                    else       { stack.push(l); stack.push(r); }
                }
            }
        }
        (best_d2.sqrt(), best_pt, best_fi)
    }
}

fn build_rec(
    nodes: &mut Vec<Node>,
    leaf_aabbs: &mut [Aabb],
    idx: &mut [u32],
    start: usize,
    end: usize,
) -> u32 {
    let mut bounds = Aabb::empty();
    for &i in &idx[start..end] {
        let a = leaf_aabbs[i as usize];
        bounds.expand_pt(a.min);
        bounds.expand_pt(a.max);
    }

    let count = end - start;
    if count == 1 {
        let id = nodes.len() as u32;
        nodes.push(Node::Leaf { aabb: bounds, face: idx[start] });
        return id;
    }

    let extent = [
        bounds.max[0] - bounds.min[0],
        bounds.max[1] - bounds.min[1],
        bounds.max[2] - bounds.min[2],
    ];
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] { 0 }
               else if extent[1] >= extent[2] { 1 } else { 2 };

    let slice = &mut idx[start..end];
    slice.sort_unstable_by(|&a, &b| {
        let ca = 0.5 * (leaf_aabbs[a as usize].min[axis] + leaf_aabbs[a as usize].max[axis]);
        let cb = 0.5 * (leaf_aabbs[b as usize].min[axis] + leaf_aabbs[b as usize].max[axis]);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = start + count / 2;

    let left  = build_rec(nodes, leaf_aabbs, idx, start, mid);
    let right = build_rec(nodes, leaf_aabbs, idx, mid, end);
    let id = nodes.len() as u32;
    nodes.push(Node::Inner { aabb: bounds, left, right });
    id
}

fn face_normal(f: [u32; 3], positions: &[[f32; 3]]) -> [f32; 3] {
    let a = positions[f[0] as usize];
    let b = positions[f[1] as usize];
    let c = positions[f[2] as usize];
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
    [n[0] / len, n[1] / len, n[2] / len]
}

/// Bake a mesh SDF at `resolution` cubed, padded by `padding` in each axis.
/// Sign comes from the closest-triangle's face normal.
pub fn bake(mesh: &ObjMesh, resolution: u32, padding: f32) -> MeshSdfVolume {
    let (lo, hi) = mesh.bounds();
    let bounds_min = [lo[0] - padding, lo[1] - padding, lo[2] - padding];
    let bounds_max = [hi[0] + padding, hi[1] + padding, hi[2] + padding];

    let bvh = TriBvh::build(&mesh.positions, &mesh.faces);

    let dx = (bounds_max[0] - bounds_min[0]) / (resolution - 1) as f32;
    let dy = (bounds_max[1] - bounds_min[1]) / (resolution - 1) as f32;
    let dz = (bounds_max[2] - bounds_min[2]) / (resolution - 1) as f32;

    let n = resolution as usize;
    let mut data = vec![0.0f32; n * n * n];

    for iz in 0..n {
        let z = bounds_min[2] + iz as f32 * dz;
        for iy in 0..n {
            let y = bounds_min[1] + iy as f32 * dy;
            for ix in 0..n {
                let x = bounds_min[0] + ix as f32 * dx;
                let p = [x, y, z];
                let (dist, cp, fi) = bvh.nearest(p, &mesh.positions, &mesh.faces);
                let nrm = face_normal(mesh.faces[fi as usize], &mesh.positions);
                let dvec = [p[0] - cp[0], p[1] - cp[1], p[2] - cp[2]];
                let s = dvec[0] * nrm[0] + dvec[1] * nrm[1] + dvec[2] * nrm[2];
                let signed = if s >= 0.0 { dist } else { -dist };
                data[iz * n * n + iy * n + ix] = signed;
            }
        }
    }

    MeshSdfVolume {
        data,
        dims: [resolution, resolution, resolution],
        bounds_min,
        bounds_max,
    }
}
