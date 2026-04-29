//! Greedy graph coloring for distance constraints.
//!
//! Two constraints are adjacent iff they share a vertex. Within a single color
//! class, no two constraints touch the same vertex, so a parallel kernel can
//! update both endpoints with plain (non-atomic) stores and remain race-free.

pub struct Coloring {
    /// Constraint indices reordered so that all constraints of color `k` form
    /// the contiguous slice `idx[offsets[k]..offsets[k+1]]`.
    pub idx:     Vec<u32>,
    /// Length `num_colors + 1`. `offsets[0] == 0`, `offsets.last() == idx.len()`.
    pub offsets: Vec<u32>,
}

impl Coloring {
    pub fn num_colors(&self) -> usize { self.offsets.len().saturating_sub(1) }
}

/// Greedy coloring of a constraint set whose dependency is "shares a vertex".
///
/// `endpoints[i]` lists every vertex touched by constraint `i`. For an edge
/// pass this is the two edge vertices; for the bend pass it is the two
/// diamond diagonal vertices `[c, d]`.
pub fn color_constraints(
    endpoints: &[&[u32]],
    num_verts: usize,
) -> Coloring {
    let n_con = endpoints.len();
    if n_con == 0 {
        return Coloring { idx: Vec::new(), offsets: vec![0] };
    }

    // Track up to 64 colors via bitmask per vertex; spill above that is unlikely
    // for cloth grids (4 suffice) and we'd hit a much larger problem first.
    const MAX_COLORS: usize = 64;
    let mut vert_mask: Vec<u64> = vec![0; num_verts];
    let mut color_of:  Vec<u32> = vec![0; n_con];
    let mut max_color: u32 = 0;

    for ci in 0..n_con {
        let mut used: u64 = 0;
        for &v in endpoints[ci] {
            used |= vert_mask[v as usize];
        }
        // Lowest free color = trailing-ones count of `used`.
        let c = (!used).trailing_zeros();
        debug_assert!((c as usize) < MAX_COLORS, "graph-coloring exceeded 64 colors");
        let bit = 1u64 << c;
        for &v in endpoints[ci] {
            vert_mask[v as usize] |= bit;
        }
        color_of[ci] = c;
        if c > max_color { max_color = c; }
    }

    let num_colors = (max_color + 1) as usize;
    let mut counts = vec![0u32; num_colors];
    for &c in &color_of { counts[c as usize] += 1; }

    let mut offsets = vec![0u32; num_colors + 1];
    for k in 0..num_colors { offsets[k + 1] = offsets[k] + counts[k]; }

    let mut cursor = offsets.clone();
    let mut idx = vec![0u32; n_con];
    for (ci, &c) in color_of.iter().enumerate() {
        let slot = cursor[c as usize] as usize;
        idx[slot] = ci as u32;
        cursor[c as usize] += 1;
    }

    Coloring { idx, offsets }
}

/// Debug-only check: every color class is a true independent set.
#[cfg(debug_assertions)]
pub fn validate(coloring: &Coloring, endpoints: &[&[u32]], num_verts: usize) {
    let mut touched = vec![u32::MAX; num_verts];
    for k in 0..coloring.num_colors() {
        let s = coloring.offsets[k] as usize;
        let e = coloring.offsets[k + 1] as usize;
        for &ci in &coloring.idx[s..e] {
            for &v in endpoints[ci as usize] {
                let v = v as usize;
                assert!(
                    touched[v] != k as u32,
                    "color {k} double-touches vertex {v}",
                );
                touched[v] = k as u32;
            }
        }
    }
}

/// Convenience for edges (pairs).
pub fn color_edges(edges: &[[u32; 2]], num_verts: usize) -> Coloring {
    let refs: Vec<&[u32]> = edges.iter().map(|e| &e[..]).collect();
    color_constraints(&refs, num_verts)
}

/// Convenience for diamond bend constraints (only `[c, d]` matter).
pub fn color_bend(diamonds: &[[u32; 4]], num_verts: usize) -> Coloring {
    let pairs: Vec<[u32; 2]> = diamonds.iter().map(|d| [d[2], d[3]]).collect();
    color_edges(&pairs, num_verts)
}
