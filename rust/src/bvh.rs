use crate::sim::cloth_sim::{Positions, Faces};

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    #[inline]
    pub fn expand(&mut self, p: [f32; 3]) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(p[i]);
            self.max[i] = self.max[i].max(p[i]);
        }
    }

    #[inline]
    pub fn merge(a: Aabb, b: Aabb) -> Aabb {
        Aabb {
            min: [
                a.min[0].min(b.min[0]),
                a.min[1].min(b.min[1]),
                a.min[2].min(b.min[2]),
            ],
            max: [
                a.max[0].max(b.max[0]),
                a.max[1].max(b.max[1]),
                a.max[2].max(b.max[2]),
            ],
        }
    }

    #[inline]
    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min[0] <= other.max[0] && self.max[0] >= other.min[0] &&
            self.min[1] <= other.max[1] && self.max[1] >= other.min[1] &&
            self.min[2] <= other.max[2] && self.max[2] >= other.min[2]
    }

    #[inline]
    pub fn surface_area(&self) -> f32 {
        let e = [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ];
        2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])
    }

    #[inline]
    pub fn centroid(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    #[inline]
    pub fn padded(&self, amount: f32) -> Aabb {
        Aabb {
            min: [self.min[0] - amount, self.min[1] - amount, self.min[2] - amount],
            max: [self.max[0] + amount, self.max[1] + amount, self.max[2] + amount],
        }
    }
}

/// A node in the BVH. Stored in a flat Vec — no heap pointers.
/// Leaves store one triangle; internal nodes store two child indices.
#[derive(Clone, Debug)]
pub enum BvhNode {
    Leaf {
        aabb: Aabb,
        face_idx: u32,
    },
    Internal {
        aabb: Aabb,
        left: u32,
        right: u32,
    },
}

impl BvhNode {
    #[inline]
    pub fn aabb(&self) -> &Aabb {
        match self {
            BvhNode::Leaf { aabb, .. } => aabb,
            BvhNode::Internal { aabb, .. } => aabb,
        }
    }
}

pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    /// Root is always nodes[0] after build.
    pub root: u32,
}

impl Bvh {
    /// Build a BVH over all triangles in `faces`.
    /// AABBs are swept: union of positions at q_prev (t=0) and q (t=1).
    pub fn build(q: &Positions, q_prev: &Positions, faces: &Faces, padding: f32) -> Self {
        let m = faces.nrows();
        assert!(m > 0);

        // Compute one swept AABB per triangle.
        let mut leaf_aabbs: Vec<Aabb> = (0..m)
            .map(|fi| swept_triangle_aabb(q, q_prev, faces, fi, padding))
            .collect();

        let mut face_indices: Vec<u32> = (0..m as u32).collect();
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * m);

        let root = build_recursive(&mut nodes, &mut leaf_aabbs, &mut face_indices, 0, m);

        Bvh { nodes, root }
    }

    /// Traverse the BVH against itself, collecting all non-adjacent (vi, fi)
    /// pairs whose swept AABBs overlap. Adjacency is checked at leaf level
    /// using `vertex_neighbors`.
    pub fn self_collision_pairs(
        &self,
        faces: &Faces,
        vertex_neighbors: &[std::collections::HashSet<u32>],
    ) -> Vec<(usize, u32)> {
        let mut pairs = Vec::new();
        // Traverse all triangle-triangle pairs
        let mut tri_pairs: Vec<(u32, u32)> = Vec::new();
        traverse_self(&self.nodes, self.root, self.root, true, &mut tri_pairs);

        // Expand triangle pairs into vertex-triangle pairs.
        // For each pair (fa, fb), test all 3 vertices of fa against fb
        // and all 3 vertices of fb against fa, skipping adjacent ones.
        for (fa, fb) in tri_pairs {
            let verts_a = [
                faces[(fa as usize, 0)],
                faces[(fa as usize, 1)],
                faces[(fa as usize, 2)],
            ];
            let verts_b = [
                faces[(fb as usize, 0)],
                faces[(fb as usize, 1)],
                faces[(fb as usize, 2)],
            ];

            // vertices of fa vs triangle fb
            for &vi in &verts_a {
                if is_adjacent(vi, &verts_b, vertex_neighbors) { continue; }
                pairs.push((vi as usize, fb));
            }
            // vertices of fb vs triangle fa
            for &vi in &verts_b {
                if is_adjacent(vi, &verts_a, vertex_neighbors) { continue; }
                pairs.push((vi as usize, fa));
            }
        }

        // Deduplicate — a vertex can appear in multiple faces that all overlap fb.
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Swept AABB for triangle fi: union of its bbox at t=0 and t=1.
fn swept_triangle_aabb(
    q: &Positions,
    q_prev: &Positions,
    faces: &Faces,
    fi: usize,
    padding: f32,
) -> Aabb {
    let v0 = faces[(fi, 0)] as usize;
    let v1 = faces[(fi, 1)] as usize;
    let v2 = faces[(fi, 2)] as usize;
    let mut aabb = Aabb::empty();
    for pos in [q, q_prev] {
        aabb.expand([pos[(v0, 0)], pos[(v0, 1)], pos[(v0, 2)]]);
        aabb.expand([pos[(v1, 0)], pos[(v1, 1)], pos[(v1, 2)]]);
        aabb.expand([pos[(v2, 0)], pos[(v2, 1)], pos[(v2, 2)]]);
    }
    if padding > 0.0 { aabb.padded(padding) } else { aabb }
}

/// Returns true if vertex `vi` belongs to triangle `tri_verts` or shares
/// a neighbor edge with any of its vertices.
#[inline]
fn is_adjacent(vi: u32, tri_verts: &[u32; 3], neighbors: &[std::collections::HashSet<u32>]) -> bool {
    if vi == tri_verts[0] || vi == tri_verts[1] || vi == tri_verts[2] {
        return true;
    }
    let nb = &neighbors[vi as usize];
    nb.contains(&tri_verts[0]) || nb.contains(&tri_verts[1]) || nb.contains(&tri_verts[2])
}

/// Recursive SAH build. Returns the index of the node created.
fn build_recursive(
    nodes: &mut Vec<BvhNode>,
    leaf_aabbs: &mut [Aabb],
    face_indices: &mut [u32],
    start: usize,
    end: usize,
) -> u32 {
    let count = end - start;
    let aabb = face_indices[start..end]
        .iter()
        .zip(leaf_aabbs[start..end].iter())
        .fold(Aabb::empty(), |acc, (_, a)| Aabb::merge(acc, *a));

    if count == 1 {
        let idx = nodes.len() as u32;
        nodes.push(BvhNode::Leaf {
            aabb,
            face_idx: face_indices[start],
        });
        return idx;
    }

    // SAH binned split — find best axis and split point.
    let (axis, split) = sah_split(&leaf_aabbs[start..end], &aabb);

    // Partition face_indices[start..end] by centroid on chosen axis.
    let mid = partition(leaf_aabbs, face_indices, start, end, axis, split);

    // Degenerate partition: fall back to median split.
    let mid = if mid == start || mid == end {
        start + count / 2
    } else {
        mid
    };

    // Reserve a slot for this internal node before recursing.
    let node_idx = nodes.len() as u32;
    nodes.push(BvhNode::Internal { aabb, left: 0, right: 0 }); // placeholders

    let left  = build_recursive(nodes, leaf_aabbs, face_indices, start, mid);
    let right = build_recursive(nodes, leaf_aabbs, face_indices, mid, end);

    nodes[node_idx as usize] = BvhNode::Internal { aabb, left, right };
    node_idx
}

const NUM_BINS: usize = 8;

/// Approximate SAH: bin centroids along each axis, find cheapest split.
/// Returns (best_axis, split_centroid_value).
fn sah_split(aabbs: &[Aabb], parent: &Aabb) -> (usize, f32) {
    let parent_sa = parent.surface_area().max(1e-10);
    let mut best_cost = f32::INFINITY;
    let mut best_axis = 0;
    let mut best_split = 0.0f32;

    for axis in 0..3 {
        let lo = parent.min[axis];
        let hi = parent.max[axis];
        if (hi - lo) < 1e-6 { continue; }

        let inv_range = NUM_BINS as f32 / (hi - lo);
        let mut bin_aabbs = [Aabb::empty(); NUM_BINS];
        let mut bin_counts = [0u32; NUM_BINS];

        for aabb in aabbs {
            let c = aabb.centroid()[axis];
            let bin = ((c - lo) * inv_range) as usize;
            let bin = bin.min(NUM_BINS - 1);
            bin_aabbs[bin] = Aabb::merge(bin_aabbs[bin], *aabb);
            bin_counts[bin] += 1;
        }

        // Sweep left→right accumulating prefix SA and counts.
        let mut left_sa = [0.0f32; NUM_BINS];
        let mut left_count = [0u32; NUM_BINS];
        let mut running = Aabb::empty();
        let mut cnt = 0u32;
        for i in 0..NUM_BINS {
            running = Aabb::merge(running, bin_aabbs[i]);
            cnt += bin_counts[i];
            left_sa[i] = running.surface_area();
            left_count[i] = cnt;
        }

        // Sweep right→left for suffix.
        let mut right_sa = [0.0f32; NUM_BINS];
        let mut right_count = [0u32; NUM_BINS];
        running = Aabb::empty();
        cnt = 0;
        for i in (0..NUM_BINS).rev() {
            running = Aabb::merge(running, bin_aabbs[i]);
            cnt += bin_counts[i];
            right_sa[i] = running.surface_area();
            right_count[i] = cnt;
        }

        // Evaluate SAH cost at each split plane.
        for split_bin in 0..(NUM_BINS - 1) {
            let cost = (left_sa[split_bin]  * left_count[split_bin]  as f32
                + right_sa[split_bin + 1] * right_count[split_bin + 1] as f32)
                / parent_sa;
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_split = lo + (split_bin + 1) as f32 / inv_range;
            }
        }
    }

    (best_axis, best_split)
}

/// Partition face_indices[start..end] so centroids < split on `axis` come first.
/// Returns the partition point (first index of the right partition).
fn partition(
    leaf_aabbs: &mut [Aabb],
    face_indices: &mut [u32],
    start: usize,
    end: usize,
    axis: usize,
    split: f32,
) -> usize {
    let mut lo = start;
    let mut hi = end - 1;
    while lo < hi {
        while lo < hi && leaf_aabbs[lo].centroid()[axis] < split { lo += 1; }
        while lo < hi && leaf_aabbs[hi].centroid()[axis] >= split { hi -= 1; }
        if lo < hi {
            leaf_aabbs.swap(lo, hi);
            face_indices.swap(lo, hi);
            lo += 1;
            hi -= 1;
        }
    }
    // lo is the partition point
    if leaf_aabbs[lo].centroid()[axis] < split { lo + 1 } else { lo }
}

/// Self-collision BVH traversal. Collects overlapping triangle pairs (a, b) with a < b.
/// `same_node` is true on the first call (both sides are the same root).
fn traverse_self(
    nodes: &[BvhNode],
    a: u32,
    b: u32,
    same_node: bool,
    pairs: &mut Vec<(u32, u32)>,
) {
    // Skip if AABBs don't overlap.
    if !same_node && !nodes[a as usize].aabb().overlaps(nodes[b as usize].aabb()) {
        return;
    }

    match (&nodes[a as usize], &nodes[b as usize]) {
        (BvhNode::Leaf { face_idx: fa, .. }, BvhNode::Leaf { face_idx: fb, .. }) => {
            let fa = *fa;
            let fb = *fb;
            if fa < fb {
                pairs.push((fa, fb));
            }
        }

        // Leaf vs internal: descend the internal node.
        (BvhNode::Leaf { .. }, BvhNode::Internal { left, right, .. }) => {
            let (l, r) = (*left, *right);
            traverse_self(nodes, a, l, false, pairs);
            traverse_self(nodes, a, r, false, pairs);
        }

        // Internal vs leaf: same.
        (BvhNode::Internal { left, right, .. }, BvhNode::Leaf { .. }) => {
            let (l, r) = (*left, *right);
            traverse_self(nodes, l, b, false, pairs);
            traverse_self(nodes, r, b, false, pairs);
        }

        // Both internal.
        (BvhNode::Internal { left: al, right: ar, .. },
            BvhNode::Internal { left: bl, right: br, .. }) => {
            let (al, ar, bl, br) = (*al, *ar, *bl, *br);
            if same_node {
                // Self-traversal: avoid duplicate pairs by only visiting
                // (left,left), (left,right), (right,right) — not (right,left).
                traverse_self(nodes, al, al, true,  pairs);
                traverse_self(nodes, al, ar, false, pairs);
                traverse_self(nodes, ar, ar, true,  pairs);
            } else {
                traverse_self(nodes, al, bl, false, pairs);
                traverse_self(nodes, al, br, false, pairs);
                traverse_self(nodes, ar, bl, false, pairs);
                traverse_self(nodes, ar, br, false, pairs);
            }
        }
    }
}