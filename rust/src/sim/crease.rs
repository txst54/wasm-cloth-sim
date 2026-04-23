use std::collections::{HashMap, HashSet};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};
use web_sys::console;

/// When true, uses higher resolution mesh near crease lines.
/// When false, uses uniform grid resolution everywhere.
pub const USE_ADAPTIVE_MESH: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreaseType {
    Boundary,  // type 1: edge of the square
    Mountain,  // type 2: folds away from camera
    Valley,    // type 3: folds toward camera
}

#[derive(Debug, Clone)]
pub struct CreaseLine {
    pub crease_type: CreaseType,
    pub p1: [f64; 2],
    pub p2: [f64; 2],
}

#[derive(Debug, Clone)]
pub struct CreasePattern {
    pub lines: Vec<CreaseLine>,
    pub bounds: ([f64; 2], [f64; 2]),  // (min, max) of bounding box
}

impl CreasePattern {
    pub fn parse(data: &str) -> Result<Self, String> {
        let mut lines = Vec::new();

        for line in data.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 5 { continue; }

            let type_id: u32 = parts[0].parse().map_err(|_| "invalid type")?;
            let x1: f64 = parts[1].parse().map_err(|_| "invalid x1")?;
            let y1: f64 = parts[2].parse().map_err(|_| "invalid y1")?;
            let x2: f64 = parts[3].parse().map_err(|_| "invalid x2")?;
            let y2: f64 = parts[4].parse().map_err(|_| "invalid y2")?;

            let crease_type = match type_id {
                1 => CreaseType::Boundary,
                2 => CreaseType::Mountain,
                3 => CreaseType::Valley,
                _ => continue,
            };

            lines.push(CreaseLine {
                crease_type,
                p1: [x1, y1],
                p2: [x2, y2],
            });
        }

        if lines.is_empty() {
            return Err("no valid crease lines found".to_string());
        }

        // Compute bounding box from all points
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for line in &lines {
            min_x = min_x.min(line.p1[0]).min(line.p2[0]);
            min_y = min_y.min(line.p1[1]).min(line.p2[1]);
            max_x = max_x.max(line.p1[0]).max(line.p2[0]);
            max_y = max_y.max(line.p1[1]).max(line.p2[1]);
        }

        Ok(Self {
            lines,
            bounds: ([min_x, min_y], [max_x, max_y]),
        })
    }

    /// Normalize coordinates to [-0.9, 0.9] range
    fn normalize_point(&self, p: [f64; 2]) -> [f64; 2] {
        let ([min_x, min_y], [max_x, max_y]) = self.bounds;
        let width = max_x - min_x;
        let height = max_y - min_y;
        let scale = width.max(height);

        // Center and scale to [-0.9, 0.9]
        let cx = (min_x + max_x) / 2.0;
        let cy = (min_y + max_y) / 2.0;

        [
            ((p[0] - cx) / scale) * 1.8,
            ((p[1] - cy) / scale) * 1.8,
        ]
    }

    /// Build a triangulated mesh by overlaying crease pattern on a grid.
    /// `grid_res`: base grid resolution (e.g., 32)
    /// `crease_res`: resolution near creases (e.g., 128) - if 0, uses grid_res
    /// `band_width`: width of refinement band around creases (in grid cells)
    /// Returns (positions, faces, fold_edges, crease_chains) where:
    /// - fold_edges maps edge (min,max) to CreaseType
    /// - crease_chains contains vertex index chains for each non-boundary crease
    pub fn build_mesh(&self, grid_res: usize) -> (Vec<[f32; 3]>, Vec<[u32; 3]>, HashMap<(u32, u32), CreaseType>, Vec<Vec<u32>>) {
        if USE_ADAPTIVE_MESH {
            self.build_mesh_adaptive(grid_res, grid_res * 2, 1)
        } else {
            // Uniform grid - no refinement near creases
            self.build_mesh_adaptive(grid_res, grid_res, 0)
        }
    }

    pub fn build_mesh_adaptive(
        &self,
        grid_res: usize,
        crease_res: usize,
        band_width: usize,
    ) -> (Vec<[f32; 3]>, Vec<[u32; 3]>, HashMap<(u32, u32), CreaseType>, Vec<Vec<u32>>) {
        let grid_spacing = if grid_res > 1 { 1.8 / (grid_res - 1) as f64 } else { 1.8 };
        let crease_spacing = if crease_res > 1 { 1.8 / (crease_res - 1) as f64 } else { 1.8 };
        let snap_eps = 1e-6;
        let band_dist = band_width as f64 * grid_spacing;

        let mut unique_verts: Vec<[f64; 2]> = Vec::new();

        let add_vertex = |verts: &mut Vec<[f64; 2]>, p: [f64; 2]| -> usize {
            for (i, &v) in verts.iter().enumerate() {
                let dx = v[0] - p[0];
                let dy = v[1] - p[1];
                if dx*dx + dy*dy < snap_eps*snap_eps {
                    return i;
                }
            }
            let idx = verts.len();
            verts.push(p);
            idx
        };

        // Normalize ALL crease lines, then split at mutual intersections
        let all_normalized: Vec<([f64; 2], [f64; 2], CreaseType)> = self.lines.iter()
            .map(|l| (self.normalize_point(l.p1), self.normalize_point(l.p2), l.crease_type))
            .collect();
        let split_creases = split_creases_at_intersections(&all_normalized);

        let non_boundary_creases: Vec<&([f64; 2], [f64; 2], CreaseType)> = split_creases.iter()
            .filter(|(_, _, ct)| *ct != CreaseType::Boundary)
            .collect();

        let near_crease = |p: [f64; 2]| -> bool {
            for &&(p1, p2, _) in &non_boundary_creases {
                let dist = point_to_segment_distance(p, p1, p2);
                if dist < band_dist {
                    return true;
                }
            }
            false
        };

        // 1. Add grid vertices, skip interior points near creases
        if grid_res <= 1 {
            // Just add 4 corners
            for &x in &[-0.9f64, 0.9] {
                for &y in &[-0.9f64, 0.9] {
                    add_vertex(&mut unique_verts, [x, y]);
                }
            }
        } else {
            for row in 0..grid_res {
                for col in 0..grid_res {
                    let x = (col as f64 / (grid_res - 1) as f64) * 1.8 - 0.9;
                    let y = (row as f64 / (grid_res - 1) as f64) * 1.8 - 0.9;
                    let is_boundary = row == 0 || row == grid_res - 1 || col == 0 || col == grid_res - 1;
                    if is_boundary || !near_crease([x, y]) {
                        add_vertex(&mut unique_verts, [x, y]);
                    }
                }
            }
        }

        // 1b. Add all crease line endpoint vertices (ensures edge vertices are in the mesh)
        for &(p1, p2, _) in &split_creases {
            if p1[0] >= -0.9 && p1[0] <= 0.9 && p1[1] >= -0.9 && p1[1] <= 0.9 {
                add_vertex(&mut unique_verts, p1);
            }
            if p2[0] >= -0.9 && p2[0] <= 0.9 && p2[1] >= -0.9 && p2[1] <= 0.9 {
                add_vertex(&mut unique_verts, p2);
            }
        }

        // 2. Add dense vertices in band around each non-boundary crease
        for &&(p1, p2, _) in &non_boundary_creases {
            let dx = p2[0] - p1[0];
            let dy = p2[1] - p1[1];
            let len = (dx*dx + dy*dy).sqrt();
            if len < 1e-10 { continue; }

            let nx = -(p2[1] - p1[1]) / len;
            let ny = (p2[0] - p1[0]) / len;

            let num_along = ((len / crease_spacing).ceil() as usize).max(1);
            let num_perp = if band_width > 0 {
                ((band_dist / crease_spacing).ceil() as usize).max(1)
            } else {
                0
            };

            for i in 0..=num_along {
                let t = i as f64 / num_along as f64;
                let cx = p1[0] + t * dx;
                let cy = p1[1] + t * dy;

                if cx >= -0.9 && cx <= 0.9 && cy >= -0.9 && cy <= 0.9 {
                    add_vertex(&mut unique_verts, [cx, cy]);
                }

                for j in 1..=num_perp {
                    let offset = j as f64 * crease_spacing;
                    let px = cx + offset * nx;
                    let py = cy + offset * ny;
                    if px >= -0.9 && px <= 0.9 && py >= -0.9 && py <= 0.9 {
                        add_vertex(&mut unique_verts, [px, py]);
                    }
                    let px = cx - offset * nx;
                    let py = cy - offset * ny;
                    if px >= -0.9 && px <= 0.9 && py >= -0.9 && py <= 0.9 {
                        add_vertex(&mut unique_verts, [px, py]);
                    }
                }
            }
        }

        // 3. Build crease vertex chains from the split segments
        let mut crease_vertex_chains: Vec<(Vec<usize>, CreaseType)> = Vec::new();

        for &(p1, p2, crease_type) in &split_creases {
            let dx = p2[0] - p1[0];
            let dy = p2[1] - p1[1];
            let len = (dx*dx + dy*dy).sqrt();
            if len < 1e-10 { continue; }

            let num_points = ((len / crease_spacing).ceil() as usize).max(1);
            let mut chain_indices: Vec<usize> = Vec::new();

            for i in 0..=num_points {
                let t = i as f64 / num_points as f64;
                let x = p1[0] + t * dx;
                let y = p1[1] + t * dy;
                if x >= -0.9 && x <= 0.9 && y >= -0.9 && y <= 0.9 {
                    let idx = add_vertex(&mut unique_verts, [x, y]);
                    if chain_indices.is_empty() || *chain_indices.last().unwrap() != idx {
                        chain_indices.push(idx);
                    }
                }
            }

            if chain_indices.len() >= 2 {
                crease_vertex_chains.push((chain_indices, crease_type));
            }
        }

        // 4. Build constrained Delaunay triangulation
        let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
            ConstrainedDelaunayTriangulation::new();

        let mut vertex_handles = Vec::new();
        for &[x, y] in &unique_verts {
            let handle = cdt.insert(Point2::new(x, y)).unwrap();
            vertex_handles.push(handle);
        }

        // All constraint edges should succeed since segments were pre-split at intersections
        let mut constraint_edges: Vec<(usize, usize, CreaseType)> = Vec::new();
        let mut failed_constraints = 0usize;
        for (chain, crease_type) in &crease_vertex_chains {
            for i in 0..chain.len()-1 {
                let a = chain[i];
                let b = chain[i+1];
                if a != b {
                    if cdt.can_add_constraint(vertex_handles[a], vertex_handles[b]) {
                        let _ = cdt.add_constraint(vertex_handles[a], vertex_handles[b]);
                        constraint_edges.push((a, b, *crease_type));
                    } else {
                        failed_constraints += 1;
                    }
                }
            }
        }
        if failed_constraints > 0 {
            console::warn_1(&format!(
                "Crease pattern: {} constraint edge(s) could not be added (unexpected)",
                failed_constraints
            ).into());
        }

        // 4. Build vertex index map from handle to our index
        let mut handle_to_idx: HashMap<_, u32> = HashMap::new();
        for (i, &handle) in vertex_handles.iter().enumerate() {
            handle_to_idx.insert(handle, i as u32);
        }

        // 5. Extract faces from triangulation with consistent winding (CCW from +Z)
        let mut faces: Vec<[u32; 3]> = Vec::new();
        for face in cdt.inner_faces() {
            let verts = face.vertices();
            let i0 = handle_to_idx[&verts[0].fix()];
            let i1 = handle_to_idx[&verts[1].fix()];
            let i2 = handle_to_idx[&verts[2].fix()];

            // Check winding by computing cross product (should point +Z for CCW)
            let p0 = unique_verts[i0 as usize];
            let p1 = unique_verts[i1 as usize];
            let p2 = unique_verts[i2 as usize];
            let e1 = [p1[0] - p0[0], p1[1] - p0[1]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1]];
            let cross_z = e1[0] * e2[1] - e1[1] * e2[0];  // z component of cross product

            if cross_z >= 0.0 {
                faces.push([i0, i1, i2]);  // CCW, keep as is
            } else {
                faces.push([i0, i2, i1]);  // CW, flip to make CCW
            }
        }

        // Convert positions to f32 with z=0
        let positions: Vec<[f32; 3]> = unique_verts
            .iter()
            .map(|&[x, y]| [x as f32, y as f32, 0.0])
            .collect();

        // Build fold_edges map (only mountain and valley folds)
        let mut fold_edges: HashMap<(u32, u32), CreaseType> = HashMap::new();
        for &(a, b, crease_type) in &constraint_edges {
            if crease_type != CreaseType::Boundary {
                let (lo, hi) = if a < b { (a as u32, b as u32) } else { (b as u32, a as u32) };
                fold_edges.insert((lo, hi), crease_type);
            }
        }

        // Build crease chains from SUCCESSFUL constraint edges only
        // Group consecutive edges by crease type to form chains
        let mut crease_chains: Vec<Vec<u32>> = Vec::new();
        for (chain, crease_type) in &crease_vertex_chains {
            if *crease_type == CreaseType::Boundary { continue; }

            // Build chain only from edges that were successfully added as constraints
            let mut valid_chain: Vec<u32> = Vec::new();
            for i in 0..chain.len().saturating_sub(1) {
                let a = chain[i];
                let b = chain[i+1];
                let (lo, hi) = if a < b { (a as u32, b as u32) } else { (b as u32, a as u32) };

                if fold_edges.contains_key(&(lo, hi)) {
                    if valid_chain.is_empty() {
                        valid_chain.push(a as u32);
                    }
                    valid_chain.push(b as u32);
                } else if !valid_chain.is_empty() {
                    // Edge was dropped - start a new chain segment
                    if valid_chain.len() >= 2 {
                        crease_chains.push(valid_chain);
                    }
                    valid_chain = Vec::new();
                }
            }
            if valid_chain.len() >= 2 {
                crease_chains.push(valid_chain);
            }
        }

        (positions, faces, fold_edges, crease_chains)
    }
}

/// Intersection of two line segments. Returns (t, u) parameters if they intersect
/// at an interior point (excluding shared endpoints within `eps` tolerance).
fn segment_intersection(a1: [f64; 2], a2: [f64; 2], b1: [f64; 2], b2: [f64; 2], eps: f64) -> Option<(f64, f64)> {
    let d1x = a2[0] - a1[0];
    let d1y = a2[1] - a1[1];
    let d2x = b2[0] - b1[0];
    let d2y = b2[1] - b1[1];

    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-12 { return None; } // parallel or collinear

    let dx = b1[0] - a1[0];
    let dy = b1[1] - a1[1];
    let t = (dx * d2y - dy * d2x) / denom;
    let u = (dx * d1y - dy * d1x) / denom;

    if t > eps && t < 1.0 - eps && u > eps && u < 1.0 - eps {
        Some((t, u))
    } else {
        None
    }
}

/// Split a set of crease segments at all mutual intersection points.
/// Returns new segments where no two cross in their interior.
fn split_creases_at_intersections(creases: &[([f64; 2], [f64; 2], CreaseType)]) -> Vec<([f64; 2], [f64; 2], CreaseType)> {
    let n = creases.len();
    let eps = 1e-9;

    // For each crease, collect t-parameters where it is split
    let mut splits: Vec<Vec<f64>> = vec![Vec::new(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            if let Some((t, u)) = segment_intersection(
                creases[i].0, creases[i].1,
                creases[j].0, creases[j].1,
                eps,
            ) {
                splits[i].push(t);
                splits[j].push(u);
            }
        }
    }

    let mut result = Vec::new();
    for (i, &(p1, p2, ctype)) in creases.iter().enumerate() {
        let mut ts = splits[i].clone();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts.dedup_by(|a, b| (*a - *b).abs() < eps);

        // Build ordered list of t values: 0, splits..., 1
        let mut all_t = Vec::with_capacity(ts.len() + 2);
        all_t.push(0.0);
        for &t in &ts {
            all_t.push(t);
        }
        all_t.push(1.0);

        for w in all_t.windows(2) {
            let t0 = w[0];
            let t1 = w[1];
            let q1 = [
                p1[0] + t0 * (p2[0] - p1[0]),
                p1[1] + t0 * (p2[1] - p1[1]),
            ];
            let q2 = [
                p1[0] + t1 * (p2[0] - p1[0]),
                p1[1] + t1 * (p2[1] - p1[1]),
            ];
            result.push((q1, q2, ctype));
        }
    }

    result
}

/// Distance from point p to line segment (a, b).
fn point_to_segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len_sq = dx*dx + dy*dy;
    if len_sq < 1e-20 {
        // Degenerate segment - just distance to point a
        let px = p[0] - a[0];
        let py = p[1] - a[1];
        return (px*px + py*py).sqrt();
    }
    // Project p onto the line, clamped to [0, 1]
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq).clamp(0.0, 1.0);
    let proj_x = a[0] + t * dx;
    let proj_y = a[1] + t * dy;
    let px = p[0] - proj_x;
    let py = p[1] - proj_y;
    (px*px + py*py).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let data = "1 0 0 100 0\n2 50 0 50 100\n3 25 50 75 50";
        let cp = CreasePattern::parse(data).unwrap();
        assert_eq!(cp.lines.len(), 3);
        assert_eq!(cp.lines[0].crease_type, CreaseType::Boundary);
        assert_eq!(cp.lines[1].crease_type, CreaseType::Mountain);
        assert_eq!(cp.lines[2].crease_type, CreaseType::Valley);
    }
}
