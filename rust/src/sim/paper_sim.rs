use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};

use nalgebra as na;

use crate::params::SimParams;
use crate::{platform_log, platform_warn, platform_log_interval};
use super::shared::{Positions, ClothSimCore};
use super::crease::{CreasePattern, CreaseType, find_edges_on_creases, find_overlapping_edges};

// ── Fold specification ────────────────────────────────────────────────────────

/// Direction of fold: mountain folds one way, valley the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldDirection {
    Mountain,  // folds away from camera (decreasing dihedral angle)
    Valley,    // folds toward camera (increasing dihedral angle)
}

/// Describes the target fold for one hinge edge.
pub struct FoldSpec {
    /// Target fold angle (radians). 0 = flat/unfolded.
    /// Mountain folds use negative values, valley folds use positive.
    pub target_angle: f32,
    /// XPBD compliance α (m²/N equivalent for angular constraints).
    /// Smaller = stiffer / faster fold.  Typical range: 1e-6 … 1e-2.
    /// Default 1e-4 is stable for dt=0.01 and avoids the divergence that
    /// pure PBD produces on large-angle corrections.
    pub compliance: f32,
    /// Direction of fold (mountain vs valley).
    pub direction: FoldDirection,
    /// XPBD constraint damping β (stiffness, not inverse).
    /// Damps velocity along the constraint gradient.  Typical range: 0 … 10.
    /// Default 0 means no additional damping beyond implicit numerical damping.
    pub damping: f32,
}

// ── HingeConstraint ───────────────────────────────────────────────────────────

/// A dihedral-angle (XPBD) constraint bound to one interior edge of the mesh.
///
/// Hinge edges skip the default bending constraint and instead have their
/// dihedral angle driven toward `current_angle` by the XPBD solver.
/// `current_angle` itself is rate-limited to advance toward `target_angle`
/// by at most `PaperSim::fold_speed * dt` radians per step — this caps the
/// per-step constraint violation `C` and prevents the solver from diverging
/// on large angle jumps.
pub struct HingeConstraint {
    pub diamond_idx: usize,
    /// Dihedral angle at construction (flat mesh → ≈ π).
    pub rest_angle: f32,
    /// Desired fold angle (0 = flat, negative = mountain, positive = valley).
    pub target_angle: f32,
    /// Current effective fold goal — smoothly advances toward `target_angle`.
    /// Actual dihedral goal = rest_angle + current_angle.
    pub current_angle: f32,
    pub compliance: f32,
    /// Nominal length of the hinge edge (used to normalise compliance).
    pub rest_edge_len: f32,
    pub direction: FoldDirection,
    /// XPBD constraint damping β (stiffness, not inverse).
    pub damping: f32,
    /// XPBD Lagrange multiplier (reset each substep).
    pub lambda: f32,
}

// ── PaperSim ──────────────────────────────────────────────────────────────────

/// A 3-vertex constraint that keeps interior crease vertices on the line
/// between their neighbors.  Unlike the old collinearity constraint this
/// moves all three vertices (weighted by inverse mass) and accumulates a
/// proper XPBD Lagrange multiplier, making it much stiffer at high fold
/// angles.
pub struct CreaseBendConstraint {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub lambda: f32,
}

pub struct PaperSim {
    pub core: ClothSimCore,
    pub hinges: Vec<HingeConstraint>,
    pub fold_speed: f32,
    pub crease_chains: Vec<Vec<u32>>,
    pub crease_bend_compliance: f32,
    pub crease_bends: Vec<CreaseBendConstraint>,
}

impl Deref for PaperSim {
    type Target = ClothSimCore;
    fn deref(&self) -> &ClothSimCore { &self.core }
}

impl DerefMut for PaperSim {
    fn deref_mut(&mut self) -> &mut ClothSimCore { &mut self.core }
}

impl PaperSim {
    /// Create a PaperSim from an NxN grid (used for simple paper demos).
    pub fn from_grid(resolution: usize) -> Self {
        let n = resolution;
        Self {
            core: ClothSimCore::from_grid(n, &[(n-1)*n + (n-1)]),
            hinges: Vec::new(),
            fold_speed: 5.0,
            crease_chains: Vec::new(),
            crease_bend_compliance: 1e-6,
            crease_bends: Vec::new(),
        }
    }

    /// Create PaperSim from a crease pattern overlaid on a grid.
    /// `resolution`: grid resolution for mesh generation (higher = more triangles)
    /// Returns (PaperSim, positions, faces, colors, edge_colors) for building Cloth.
    pub fn from_crease_pattern(cp: &CreasePattern, resolution: usize) -> (Self, Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<[f32; 3]>, HashMap<(u32, u32), CreaseType>) {
        let (positions, faces, fold_edges, crease_chains, split_creases) = cp.build_mesh(resolution);

        // Find top-right corner vertex to pin (closest to (0.9, 0.9))
        let mut pin_idx = 0usize;
        let mut best_dist = f32::MAX;
        for (i, &[x, y, _]) in positions.iter().enumerate() {
            let d = (x - 0.9).abs() + (y - 0.9).abs();
            if d < best_dist {
                best_dist = d;
                pin_idx = i;
            }
        }

        // Build SimCore from mesh with top corner pinned
        let core = ClothSimCore::from_mesh(&positions, &faces, &[]);

        // Check for duplicate/overlapping edges in the mesh
        let overlapping = find_overlapping_edges(&positions, &core.edges, 1e-6);
        if !overlapping.is_empty() {
            platform_warn!(
                "WARNING: Found {} overlapping edge pairs in mesh!",
                overlapping.len()
            );
            for (e1, e2) in &overlapping {
                platform_warn!(
                    "  Overlap: edge [{}, {}] and edge [{}, {}]",
                    e1[0], e1[1], e2[0], e2[1]
                );
            }
        } else {
            platform_log!("No overlapping edges found in mesh");
        }

        // Check ALL mesh edges against crease lines to find edges that lie on creases
        // This catches edges that the triangulation created that happen to align with creases
        let tolerance = 1e-6; // tolerance for "on crease" check
        let edges_on_creases = find_edges_on_creases(&positions, &core.edges, &split_creases, tolerance);

        // Merge fold_edges (from CDT constraints) with edges_on_creases (geometric check)
        let mut all_fold_edges: HashMap<(u32, u32), CreaseType> = fold_edges.clone();
        let mut extra_edges = 0usize;
        for (edge, crease_type) in &edges_on_creases {
            if !all_fold_edges.contains_key(edge) {
                all_fold_edges.insert(*edge, *crease_type);
                extra_edges += 1;
            }
        }
        if extra_edges > 0 {
            platform_log!(
                "Found {} additional edges lying on crease lines",
                extra_edges
            );
        }

        // Generate colors: white for normal faces, red-ish for mountain, blue-ish for valley
        let mut vertex_colors = vec![[0.85f32, 0.85, 0.85]; positions.len()];

        // Color vertices that are part of fold edges
        // for (&(a, b), &crease_type) in &all_fold_edges {
        //     let color = match crease_type {
        //         CreaseType::Mountain => [1.0, 0.6, 0.6],  // reddish
        //         CreaseType::Valley => [0.6, 0.6, 1.0],    // bluish
        //         CreaseType::Boundary => [0.85, 0.85, 0.85],
        //     };
        //     vertex_colors[a as usize] = color;
        //     vertex_colors[b as usize] = color;
        // }

        // Build crease-bend triples from chains: for each consecutive (A,B,C),
        // B is the interior vertex that should stay on line AC.
        let crease_bends: Vec<CreaseBendConstraint> = crease_chains.iter()
            .flat_map(|chain| {
                chain.windows(3).map(|w| CreaseBendConstraint {
                    a: w[0], b: w[1], c: w[2], lambda: 0.0,
                })
            })
            .collect();

        let mut sim = Self {
            core,
            hinges: Vec::new(),
            fold_speed: 5.0,
            crease_chains,
            crease_bend_compliance: 1e-6,
            crease_bends,
        };

        // Build fold map from all fold edges (CDT constraints + geometric matches)
        let mut fold_map: HashMap<(u32, u32), FoldSpec> = HashMap::new();
        let edge_colors = all_fold_edges.clone();
        for ((a, b), crease_type) in all_fold_edges {
            let direction = match crease_type {
                CreaseType::Mountain => FoldDirection::Mountain,
                CreaseType::Valley => FoldDirection::Valley,
                CreaseType::Boundary => continue,
            };
            fold_map.insert((a, b), FoldSpec {
                target_angle: 0.0,  // start flat (0 = unfolded)
                compliance: 1e-4,
                direction,
                damping: 0.5,
            });
        }
        sim.set_fold_map(fold_map);

        (sim, positions, faces, vertex_colors, edge_colors)
    }

    /// Register hinges from a fold map.
    /// Keys: `(min_v, max_v)` edge pairs.  Edges not in the diamond list are warned.
    ///
    /// Diamond c/d ordering is already consistent from `build_diamonds` (derived
    /// from face winding), so no geometric swap heuristic is needed here.
    pub fn set_fold_map(&mut self, fold_map: HashMap<(u32, u32), FoldSpec>) {
        let edge_to_diamond: HashMap<(u32, u32), usize> = self
            .core.diamonds.iter().enumerate()
            .map(|(di, &[a, b, _, _])| (normalise_edge(a, b), di))
            .collect();

        self.hinges.clear();
        let mut dropped_edges = 0usize;
        for ((ea, eb), spec) in fold_map {
            let key = normalise_edge(ea, eb);
            if let Some(&di) = edge_to_diamond.get(&key) {
                let [a, b, c, d] = self.core.diamonds[di];
                let rest_angle = dihedral_angle(&self.core.q, a, b, c, d);
                let edge_len = (self.core.q_rest.row(b as usize) - self.core.q_rest.row(a as usize)).norm();

                self.hinges.push(HingeConstraint {
                    diamond_idx: di,
                    rest_angle,
                    target_angle: spec.target_angle,
                    current_angle: 0.0,
                    compliance: spec.compliance / edge_len.max(1e-12),
                    rest_edge_len: edge_len,
                    direction: spec.direction,
                    damping: spec.damping,
                    lambda: 0.0,
                });
            } else {
                dropped_edges += 1;
            }
        }
        if dropped_edges > 0 {
            platform_warn!(
                "PaperSim: {} fold edge(s) have no corresponding diamond (boundary edges or triangulation mismatch)",
                dropped_edges
            );
        }

        let mountain_count = self.hinges.iter().filter(|h| h.direction == FoldDirection::Mountain).count();
        let valley_count = self.hinges.iter().filter(|h| h.direction == FoldDirection::Valley).count();
        platform_log!(
            "Hinges: {} total, rest ≈ {:.3}, {} mountain, {} valley",
            self.hinges.len(),
            self.hinges.first().map_or(0.0, |h| h.rest_angle),
            mountain_count, valley_count
        );
    }

    /// Set the fold angle for all hinges.
    /// `angle_degrees` is the fold amount: 0 = flat.
    /// Mountain folds get negative target, valley folds get positive target.
    pub fn set_fold_angle(&mut self, angle_degrees: f32) {
        let fold_amount = angle_degrees.to_radians();
        for h in self.hinges.iter_mut() {
            h.target_angle = match h.direction {
                FoldDirection::Mountain => -fold_amount,
                FoldDirection::Valley => fold_amount,
            };
        }
    }

    pub fn step(&mut self, params: &SimParams) {
        let dt = params.time_step as f32;
        let max_delta = self.fold_speed * dt;

        // Rate-limit each hinge toward its target (per-frame, total advance unchanged)
        for h in self.hinges.iter_mut() {
            let delta = (h.target_angle - h.current_angle).clamp(-max_delta, max_delta);
            h.current_angle += delta;
        }

        let skip: HashSet<usize> = self.hinges.iter().map(|h| h.diamond_idx).collect();

        // Substepping is XPBD-only.
        let n_sub = if params.use_distance_constraints {
            params.num_substeps.max(1)
        } else {
            1
        };
        let sub_params = if n_sub > 1 {
            let mut p = params.clone();
            p.time_step = params.time_step / n_sub as f64;
            p
        } else {
            params.clone()
        };
        let sub_dt = sub_params.time_step as f32;
        let cb_alpha_tilde = self.crease_bend_compliance / (sub_dt * sub_dt);

        for _ in 0..n_sub {
            // 1. Predict
            self.core.predict(&sub_params);

            // 2. Reset all lambdas (per-substep)
            self.core.reset_lambdas();
            for h in &mut self.hinges { h.lambda = 0.0; }
            for cb in &mut self.crease_bends { cb.lambda = 0.0; }

            // 3. Unified XPBD iteration loop
            for _ in 0..sub_params.constraint_iters {
                self.core.solve_stretch(&sub_params);
                self.core.solve_bend(&sub_params, &skip);
                self.core.solve_pins(&sub_params);
                self.core.solve_pulling(&sub_params);

                // Hinge dihedral constraints with XPBD damping
                for h in &mut self.hinges {
                    let [a, b, c, d] = self.core.diamonds[h.diamond_idx];
                    let alpha_tilde = h.compliance / (sub_dt * sub_dt);
                    let gamma = h.compliance * h.damping / sub_dt;
                    let goal_dihedral = h.current_angle;

                    apply_hinge_xpbd(
                        &mut self.core.q, &self.core.q_prev, &self.core.w,
                        a as usize, b as usize, c as usize, d as usize,
                        goal_dihedral, alpha_tilde, gamma, &mut h.lambda,
                    );
                }

                // // Crease-bend constraints (keep crease lines straight)
                // for cb in &mut self.crease_bends {
                //     apply_crease_bend_xpbd(
                //         &mut self.core.q, &self.core.w,
                //         cb.a as usize, cb.b as usize, cb.c as usize,
                //         cb_alpha_tilde, &mut cb.lambda,
                //     );
                // }
            }

            self.core.solve_self_collision(&sub_params);

            // 5. Derive velocity from position change
            self.core.update_velocity(&sub_params);
        }
        let _ = cb_alpha_tilde;
    }

}

// ── XPBD dihedral-angle constraint ───────────────────────────────────────────

/// XPBD dihedral-angle constraint with Rayleigh damping (Macklin et al. 2016 §5).
///
/// Unlike plain PBD this accumulates the Lagrange multiplier `λ` across
/// iterations within the same substep, which prevents the large-angle
/// overcorrection that caused divergence under the old PBD formulation.
///
/// Damped formula (equation 26 from the paper):
/// ```text
/// γ = α̃ β̃ / Δt = compliance · damping / dt
/// Δλ = -(C + α̃ λ + γ ∇C·(x - x^n)) / ((1+γ) Σ wᵢ |∇C|² + α̃)
/// λ  += Δλ
/// Δpᵢ = wᵢ Δλ ∇C_i
/// ```
/// where `α̃ = α / dt²`, `β̃ = dt² β`, and the constraint `C = θ − θ_goal`.
pub(crate) fn apply_hinge_xpbd(
    q:           &mut Positions,
    q_prev:      &Positions,
    w:           &na::DVector<f32>,
    a: usize, b: usize, c: usize, d: usize,
    goal_angle:  f32,
    alpha_tilde: f32,   // compliance / dt²
    gamma:       f32,   // compliance * damping / dt
    lambda:      &mut f32,
) {
    // Doc convention: [a, b] = crease edge (a→b is the "forward" direction),
    // c = flap1 (backward face, CCW = (b, a, c)), d = flap2 (forward face, CCW = (a, b, d)).
    let pa = row3(q, a); let pb = row3(q, b);
    let pc = row3(q, c); let pd = row3(q, d);

    let edge = pb - pa;
    let crease_len = edge.norm();
    if crease_len < 1e-8 { return; }
    let e_hat = edge / crease_len;

    // CCW face normals — both point the same way on a flat mesh, so cos(θ_flat)=+1.
    let n1_raw = (pa - pb).cross(&(pc - pb));   // face1 = (b, a, c)
    let n2_raw = (pb - pa).cross(&(pd - pa));   // face2 = (a, b, d)
    let n1l = n1_raw.norm(); let n2l = n2_raw.norm();
    if n1l < 1e-8 || n2l < 1e-8 { return; }
    let n1 = n1_raw / n1l; let n2 = n2_raw / n2l;

    let cos_t = n1.dot(&n2).clamp(-1.0, 1.0);
    let sin_t = n1.cross(&n2).dot(&e_hat);
    let theta = sin_t.atan2(cos_t);

    let c_val = theta - goal_angle;
    if c_val.abs() < 1e-5 { return; }

    // Moment arms and projection coefficients (matches doc's updateCreaseGeo).
    let v1 = pc - pa;   // flap1 - edge_v1
    let v2 = pd - pa;   // flap2 - edge_v1
    let proj1 = e_hat.dot(&v1);
    let proj2 = e_hat.dot(&v2);
    let h1_sq = (v1.norm_squared() - proj1 * proj1).max(0.0);
    let h2_sq = (v2.norm_squared() - proj2 * proj2).max(0.0);
    if h1_sq < 1e-12 || h2_sq < 1e-12 { return; }
    let h1 = h1_sq.sqrt();
    let h2 = h2_sq.sqrt();
    let coef1 = proj1 / crease_len;
    let coef2 = proj2 / crease_len;

    // ∂θ/∂x_i — flap nodes move along their face normal at rate 1/h;
    // edge nodes are weighted combinations (doc §7).
    let g_c = n1 / h1;                                                    // ∇flap1
    let g_d = n2 / h2;                                                    // ∇flap2
    let g_a = -((1.0 - coef1) / h1) * n1 - ((1.0 - coef2) / h2) * n2;     // ∇edge_v1
    let g_b = -(coef1 / h1) * n1 - (coef2 / h2) * n2;                     // ∇edge_v2

    let wa = w[a]; let wb = w[b]; let wc = w[c]; let wd = w[d];

    // Damping term: γ ∇C·(x - x^n)
    let dx_a = pa - row3(q_prev, a);
    let dx_b = pb - row3(q_prev, b);
    let dx_c = pc - row3(q_prev, c);
    let dx_d = pd - row3(q_prev, d);
    let grad_dot_dx = g_a.dot(&dx_a) + g_b.dot(&dx_b) + g_c.dot(&dx_c) + g_d.dot(&dx_d);

    let weighted_grads = wa * g_a.norm_squared()
                       + wb * g_b.norm_squared()
                       + wc * g_c.norm_squared()
                       + wd * g_d.norm_squared();
    let denom = (1.0 + gamma) * weighted_grads + alpha_tilde;
    if denom < 1e-12 { return; }

    let dl = -(c_val + alpha_tilde * *lambda + gamma * grad_dot_dx) / denom;
    *lambda += dl;

    platform_log_interval!(100, 1,
        "θ={:.3} goal={:.3} c={:.4} dl={:.6} γ∇C·dx={:.4} γ={:.2e}",
        theta, goal_angle, c_val, dl, gamma * grad_dot_dx, gamma
    );

    add_scaled(q, a, wa * dl * g_a);
    add_scaled(q, b, wb * dl * g_b);
    add_scaled(q, c, wc * dl * g_c);
    add_scaled(q, d, wd * dl * g_d);
}


// ── XPBD crease-bend constraint ──────────────────────────────────────────────

/// 3-body XPBD constraint: keeps vertex B on the line through A and C.
///
/// C = perpendicular distance from B to line AC.
///
/// Gradients (exact):
///   ∂C/∂A = -(1-t) n̂,   ∂C/∂B = n̂,   ∂C/∂C = -t n̂
///
/// where t = projection parameter of B onto AC and n̂ = unit perpendicular.
/// All three vertices move, weighted by inverse mass, so the crease stays
/// stiff even when A and C are light.
fn apply_crease_bend_xpbd(
    q:           &mut Positions,
    w:           &na::DVector<f32>,
    a: usize, b: usize, c: usize,
    alpha_tilde: f32,
    lambda:      &mut f32,
) {
    let pa = row3(q, a);
    let pb = row3(q, b);
    let pc = row3(q, c);

    let edge = pc - pa;
    let edge_len_sq = edge.norm_squared();
    if edge_len_sq < 1e-12 { return; }

    let v = pb - pa;
    let t = v.dot(&edge) / edge_len_sq;
    let proj = pa + t * edge;
    let perp = pb - proj;
    let dist = perp.norm();

    if dist < 1e-6 { return; }

    let n_hat = perp / dist;
    let c_val = dist;

    let wa = w[a]; let wb = w[b]; let wc = w[c];

    let ga = -(1.0 - t);
    let gb = 1.0;
    let gc = -t;

    let denom = wa * ga * ga + wb * gb * gb + wc * gc * gc + alpha_tilde;
    if denom < 1e-12 { return; }

    let dl = -(c_val + alpha_tilde * *lambda) / denom;
    *lambda += dl;

    add_scaled(q, a, wa * dl * ga * n_hat);
    add_scaled(q, b, wb * dl * gb * n_hat);
    add_scaled(q, c, wc * dl * gc * n_hat);
}

// ── Tiny inline utilities ─────────────────────────────────────────────────────

#[inline]
pub(crate) fn normalise_edge(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Compute dihedral "fold angle" for a diamond.
/// Returns 0 for a flat mesh, negative for mountain folds, positive for valley folds.
/// Range: [-π, π]
pub(crate) fn dihedral_angle(q: &Positions, a: u32, b: u32, c: u32, d: u32) -> f32 {
    let pa = row3(q, a as usize); let pb = row3(q, b as usize);
    let pc = row3(q, c as usize); let pd = row3(q, d as usize);
    let edge = pb - pa;
    let e_len = edge.norm();
    if e_len < 1e-8 { return 0.0; }
    let e_hat = edge / e_len;
    // Match apply_hinge_xpbd: c = flap1 (backward face), d = flap2 (forward face).
    let n1_raw = (pa - pb).cross(&(pc - pb));
    let n2_raw = (pb - pa).cross(&(pd - pa));
    let n1l = n1_raw.norm(); let n2l = n2_raw.norm();
    if n1l < 1e-8 || n2l < 1e-8 { return 0.0; }
    let n1 = n1_raw / n1l; let n2 = n2_raw / n2l;
    let cos_t = n1.dot(&n2).clamp(-1.0, 1.0);
    n1.cross(&n2).dot(&e_hat).atan2(cos_t)
}

#[inline]
fn row3(q: &Positions, i: usize) -> na::Vector3<f32> {
    na::Vector3::new(q[(i, 0)], q[(i, 1)], q[(i, 2)])
}

#[inline]
fn add_scaled(q: &mut Positions, i: usize, dv: na::Vector3<f32>) {
    q[(i, 0)] += dv[0]; q[(i, 1)] += dv[1]; q[(i, 2)] += dv[2];
}

// ── Tests ────────────────────────────────────────────────────────────────────
//
// Coverage for the (non-particle) paper simulation:
//   * `normalise_edge`, `dihedral_angle`      — geometry primitives
//   * `apply_hinge_xpbd`                       — the XPBD dihedral kernel
//   * `PaperSim::{from_grid, set_fold_map, set_fold_angle, step}`
//   * `PaperSim::from_crease_pattern`          — crease-pattern-driven meshes
//
// The crease-bend path (`apply_crease_bend_xpbd`) is intentionally left
// untested: it is dead code (commented out in `PaperSim::step`).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SimParams;
    use crate::sim::{CreasePattern, CreaseType};

    const PI: f32 = std::f32::consts::PI;

    // ── helpers ─────────────────────────────────────────────────────────────

    /// Flat 2-triangle diamond in the `z = 0` plane, indices `[0, 1, 2, 3]`:
    /// `a=(0,0,0)  b=(1,0,0)  c=(0.5,-1,0)  d=(0.5,1,0)`.
    ///
    /// Winding matches the `[edge_v1, edge_v2, flap1, flap2]` convention, so the
    /// flat dihedral angle is ≈ 0 and folding flap `d` toward +z is a valley
    /// (positive) fold.
    fn flat_diamond() -> (Positions, na::DVector<f32>) {
        let mut q = Positions::zeros(4);
        let pts = [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        for (i, p) in pts.iter().enumerate() {
            q[(i, 0)] = p[0];
            q[(i, 1)] = p[1];
            q[(i, 2)] = p[2];
        }
        (q, na::DVector::from_element(4, 1.0f32))
    }

    /// Rotate flap `d` out of the plane by `phi` radians about the crease
    /// (the x-axis). `d` starts at unit distance from the crease.
    fn fold_flap_d(q: &mut Positions, phi: f32) {
        q[(3, 1)] = phi.cos();
        q[(3, 2)] = phi.sin();
    }

    fn all_finite(q: &Positions) -> bool {
        (0..q.nrows()).all(|i| (0..3).all(|k| q[(i, k)].is_finite()))
    }

    fn vertex_move(q: &Positions, q0: &Positions, i: usize) -> f32 {
        let dx = q[(i, 0)] - q0[(i, 0)];
        let dy = q[(i, 1)] - q0[(i, 1)];
        let dz = q[(i, 2)] - q0[(i, 2)];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn max_vertex_move(q: &Positions, q0: &Positions) -> f32 {
        (0..q.nrows()).fold(0.0f32, |m, i| m.max(vertex_move(q, q0, i)))
    }

    fn center_of_mass(q: &Positions) -> [f32; 3] {
        let n = q.nrows() as f32;
        let mut c = [0.0f32; 3];
        for i in 0..q.nrows() {
            for k in 0..3 {
                c[k] += q[(i, k)];
            }
        }
        [c[0] / n, c[1] / n, c[2] / n]
    }

    /// Central vertical fold map for an `NxN` grid — mirrors the wasm harness's
    /// `central_vertical_fold`. Every edge here is interior, so it maps 1:1 to a
    /// diamond.
    fn central_vertical_fold(n: usize, dir: FoldDirection) -> HashMap<(u32, u32), FoldSpec> {
        let mut map = HashMap::new();
        let col = n / 2;
        for row in 0..(n - 1) {
            let a = (row * n + col) as u32;
            let b = ((row + 1) * n + col) as u32;
            map.insert(normalise_edge(a, b), FoldSpec {
                target_angle: PI,
                compliance: 1e-4,
                direction: dir,
                damping: 0.5,
            });
        }
        map
    }

    /// Mean dihedral angle across every hinge diamond.
    fn mean_hinge_dihedral(sim: &PaperSim) -> f32 {
        if sim.hinges.is_empty() {
            return 0.0;
        }
        let sum: f32 = sim.hinges.iter().map(|h| {
            let [a, b, c, d] = sim.core.diamonds[h.diamond_idx];
            dihedral_angle(&sim.core.q, a, b, c, d)
        }).sum();
        sum / sim.hinges.len() as f32
    }

    fn max_edge_strain(core: &ClothSimCore) -> f32 {
        core.edges.iter().enumerate().map(|(ei, &[a, b])| {
            let (a, b) = (a as usize, b as usize);
            let dx = core.q[(a, 0)] - core.q[(b, 0)];
            let dy = core.q[(a, 1)] - core.q[(b, 1)];
            let dz = core.q[(a, 2)] - core.q[(b, 2)];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            let rest = core.edge_rest_lengths[ei];
            ((len - rest) / rest).abs()
        }).fold(0.0f32, f32::max)
    }

    fn step_n(sim: &mut PaperSim, params: &SimParams, k: usize) {
        for _ in 0..k {
            sim.step(params);
        }
    }

    fn no_gravity() -> SimParams {
        SimParams { gravity_enabled: false, ..SimParams::default() }
    }

    /// Unit-square boundary plus one horizontal mountain and one vertical valley
    /// crossing at the centre.
    fn sample_crease_pattern() -> CreasePattern {
        let data = "\
1 0 0 100 0
1 100 0 100 100
1 100 100 0 100
1 0 100 0 0
2 0 50 100 50
3 50 0 50 100";
        CreasePattern::parse(data).unwrap()
    }

    // ── normalise_edge ─────────────────────────────────────────────────────

    #[test]
    fn normalise_edge_orders_the_pair() {
        assert_eq!(normalise_edge(3, 7), (3, 7));
        assert_eq!(normalise_edge(7, 3), (3, 7));
        assert_eq!(normalise_edge(5, 5), (5, 5));
    }

    // ── dihedral_angle ─────────────────────────────────────────────────────

    #[test]
    fn dihedral_of_flat_diamond_is_zero() {
        let (q, _) = flat_diamond();
        assert!(dihedral_angle(&q, 0, 1, 2, 3).abs() < 1e-5);
    }

    #[test]
    fn dihedral_measures_a_known_ninety_degree_valley_fold() {
        let (mut q, _) = flat_diamond();
        fold_flap_d(&mut q, PI / 2.0);
        let theta = dihedral_angle(&q, 0, 1, 2, 3);
        assert!((theta - PI / 2.0).abs() < 1e-4, "got {theta}");
    }

    #[test]
    fn dihedral_sign_flips_when_the_flaps_are_swapped() {
        let (mut q, _) = flat_diamond();
        fold_flap_d(&mut q, 0.7);
        let t1 = dihedral_angle(&q, 0, 1, 2, 3);
        let t2 = dihedral_angle(&q, 0, 1, 3, 2);
        assert!((t1 + t2).abs() < 1e-4, "t1={t1} t2={t2}");
    }

    #[test]
    fn dihedral_stays_within_pi_range() {
        let (mut q, _) = flat_diamond();
        for deg in [-179.0f32, -90.0, -1.0, 1.0, 90.0, 179.0] {
            fold_flap_d(&mut q, deg.to_radians());
            let t = dihedral_angle(&q, 0, 1, 2, 3);
            assert!((-PI..=PI).contains(&t), "deg={deg} theta={t}");
        }
    }

    #[test]
    fn dihedral_of_degenerate_crease_edge_is_zero() {
        let (mut q, _) = flat_diamond();
        q[(1, 0)] = 0.0; // b coincides with a
        assert_eq!(dihedral_angle(&q, 0, 1, 2, 3), 0.0);
    }

    #[test]
    fn dihedral_with_collinear_flap_is_zero() {
        let (mut q, _) = flat_diamond();
        q[(2, 1)] = 0.0; // c lands on the crease line -> degenerate face normal
        assert_eq!(dihedral_angle(&q, 0, 1, 2, 3), 0.0);
    }

    // ── apply_hinge_xpbd ───────────────────────────────────────────────────

    #[test]
    fn hinge_call_reduces_constraint_residual() {
        let (mut q, w) = flat_diamond();
        let q_prev = q.clone();
        let goal = 0.5f32;
        let before = dihedral_angle(&q, 0, 1, 2, 3) - goal;
        let mut lambda = 0.0;
        apply_hinge_xpbd(&mut q, &q_prev, &w, 0, 1, 2, 3, goal, 0.0, 0.0, &mut lambda);
        let after = dihedral_angle(&q, 0, 1, 2, 3) - goal;
        assert!(after.abs() < before.abs(), "before={before} after={after}");
        assert!(all_finite(&q));
    }

    #[test]
    fn hinge_iteration_converges_to_the_goal_angle() {
        let (mut q, w) = flat_diamond();
        let q_prev = q.clone();
        let goal = 0.6f32;
        let mut lambda = 0.0;
        for _ in 0..40 {
            apply_hinge_xpbd(&mut q, &q_prev, &w, 0, 1, 2, 3, goal, 0.0, 0.0, &mut lambda);
        }
        let theta = dihedral_angle(&q, 0, 1, 2, 3);
        assert!((theta - goal).abs() < 1e-2, "theta={theta}");
    }

    #[test]
    fn hinge_leaves_pinned_edge_vertices_fixed() {
        let (mut q, mut w) = flat_diamond();
        w[0] = 0.0;
        w[1] = 0.0;
        let q_prev = q.clone();
        let start = q.clone();
        let mut lambda = 0.0;
        for _ in 0..40 {
            apply_hinge_xpbd(&mut q, &q_prev, &w, 0, 1, 2, 3, 0.5, 0.0, 0.0, &mut lambda);
        }
        assert!(vertex_move(&q, &start, 0) < 1e-6);
        assert!(vertex_move(&q, &start, 1) < 1e-6);
        // the free flaps still bend the diamond toward the goal
        assert!(dihedral_angle(&q, 0, 1, 2, 3).abs() > 0.3);
    }

    #[test]
    fn hinge_with_huge_compliance_barely_moves_anything() {
        let (mut q, w) = flat_diamond();
        let q_prev = q.clone();
        let mut lambda = 0.0;
        apply_hinge_xpbd(&mut q, &q_prev, &w, 0, 1, 2, 3, 1.0, 1e12, 0.0, &mut lambda);
        assert!(max_vertex_move(&q, &q_prev) < 1e-6);
    }

    #[test]
    fn hinge_early_out_paths_are_noops() {
        // (a) crease edge shorter than 1e-8
        {
            let (mut q, w) = flat_diamond();
            q[(1, 0)] = 0.0;
            let q_prev = q.clone();
            let mut lambda = 0.0;
            apply_hinge_xpbd(&mut q, &q_prev, &w, 0, 1, 2, 3, 0.5, 0.0, 0.0, &mut lambda);
            assert_eq!(lambda, 0.0);
            assert!(max_vertex_move(&q, &q_prev) == 0.0);
        }
        // (b) constraint already satisfied (|C| < 1e-5)
        {
            let (mut q, w) = flat_diamond();
            let q_prev = q.clone();
            let mut lambda = 0.0;
            apply_hinge_xpbd(&mut q, &q_prev, &w, 0, 1, 2, 3, 0.0, 0.0, 0.0, &mut lambda);
            assert_eq!(lambda, 0.0);
            assert!(max_vertex_move(&q, &q_prev) == 0.0);
        }
        // (c) degenerate face normal (collinear flap)
        {
            let (mut q, w) = flat_diamond();
            q[(2, 1)] = 0.0;
            let q_prev = q.clone();
            let mut lambda = 0.0;
            apply_hinge_xpbd(&mut q, &q_prev, &w, 0, 1, 2, 3, 0.5, 0.0, 0.0, &mut lambda);
            assert_eq!(lambda, 0.0);
            assert!(all_finite(&q));
        }
    }

    #[test]
    fn hinge_damping_term_shrinks_the_correction() {
        // q_prev == q, so the damping dot-product is exactly zero and only the
        // `(1 + gamma)` denominator term is exercised.
        let (q0, w) = flat_diamond();
        let q_prev = q0.clone();

        let mut q_a = q0.clone();
        let mut la = 0.0;
        apply_hinge_xpbd(&mut q_a, &q_prev, &w, 0, 1, 2, 3, 0.5, 0.0, 0.0, &mut la);

        let mut q_b = q0.clone();
        let mut lb = 0.0;
        apply_hinge_xpbd(&mut q_b, &q_prev, &w, 0, 1, 2, 3, 0.5, 0.0, 100.0, &mut lb);

        assert!(lb.abs() < la.abs(), "undamped |dl|={} damped |dl|={}", la.abs(), lb.abs());
        assert!(all_finite(&q_b));
    }

    #[test]
    fn hinge_correction_conserves_center_of_mass_for_equal_masses() {
        let (mut q, w) = flat_diamond();
        let q_prev = q.clone();
        let com0 = center_of_mass(&q);
        let mut lambda = 0.0;
        for _ in 0..10 {
            apply_hinge_xpbd(&mut q, &q_prev, &w, 0, 1, 2, 3, 0.5, 0.0, 0.0, &mut lambda);
        }
        let com1 = center_of_mass(&q);
        let drift = ((com1[0] - com0[0]).powi(2)
            + (com1[1] - com0[1]).powi(2)
            + (com1[2] - com0[2]).powi(2)).sqrt();
        assert!(drift < 1e-4, "COM drift {drift}");
    }

    // ── PaperSim::from_grid ────────────────────────────────────────────────

    #[test]
    fn from_grid_has_expected_topology_and_defaults() {
        let n = 6;
        let sim = PaperSim::from_grid(n);
        assert_eq!(sim.core.q.nrows(), n * n);
        assert_eq!(sim.core.faces.nrows(), 2 * (n - 1) * (n - 1));
        assert_eq!(sim.core.edges.len(), (n - 1) * (3 * n - 1));
        // every non-perimeter edge is shared by two triangles -> one diamond
        assert_eq!(sim.core.diamonds.len(), sim.core.edges.len() - 4 * (n - 1));
        assert_eq!(sim.fold_speed, 5.0);
        assert!(sim.hinges.is_empty());
        assert!(sim.crease_bends.is_empty());
        assert!(sim.crease_chains.is_empty());
    }

    #[test]
    fn from_grid_pins_only_the_far_corner() {
        let n = 6;
        let sim = PaperSim::from_grid(n);
        let pin = (n - 1) * n + (n - 1);
        assert_eq!(sim.core.w[pin], 0.0);
        for i in 0..n * n {
            if i != pin {
                assert_eq!(sim.core.w[i], 1.0);
            }
        }
    }

    // ── PaperSim::set_fold_map ─────────────────────────────────────────────

    #[test]
    fn set_fold_map_builds_one_hinge_per_interior_fold_edge() {
        let n = 8;
        let mut sim = PaperSim::from_grid(n);
        sim.set_fold_map(central_vertical_fold(n, FoldDirection::Mountain));
        assert_eq!(sim.hinges.len(), n - 1);
        for h in &sim.hinges {
            assert_eq!(h.direction, FoldDirection::Mountain);
            assert_eq!(h.current_angle, 0.0);
            assert!(h.rest_angle.abs() < 1e-4, "rest_angle {}", h.rest_angle);
            assert!(h.rest_edge_len > 0.0);
            assert!((h.compliance - 1e-4 / h.rest_edge_len).abs() < 1e-9);
            assert!(h.diamond_idx < sim.core.diamonds.len());
        }
    }

    #[test]
    fn set_fold_map_drops_boundary_edges_without_a_diamond() {
        let n = 8;
        let mut sim = PaperSim::from_grid(n);
        let mut map = central_vertical_fold(n, FoldDirection::Mountain);
        map.insert(normalise_edge(0, 1), FoldSpec {
            target_angle: PI,
            compliance: 1e-4,
            direction: FoldDirection::Valley,
            damping: 0.5,
        });
        sim.set_fold_map(map);
        assert_eq!(sim.hinges.len(), n - 1);
    }

    #[test]
    fn set_fold_map_clears_previous_hinges_on_repeat() {
        let n = 6;
        let mut sim = PaperSim::from_grid(n);
        sim.set_fold_map(central_vertical_fold(n, FoldDirection::Mountain));
        let first = sim.hinges.len();
        sim.set_fold_map(central_vertical_fold(n, FoldDirection::Mountain));
        assert_eq!(sim.hinges.len(), first);
    }

    #[test]
    fn set_fold_map_partitions_mountain_and_valley_hinges() {
        let n = 6;
        let mut sim = PaperSim::from_grid(n);
        let mut map = central_vertical_fold(n, FoldDirection::Mountain);
        let col = n / 2;
        let a = col as u32;
        let b = (n + col) as u32;
        map.get_mut(&normalise_edge(a, b)).unwrap().direction = FoldDirection::Valley;
        sim.set_fold_map(map);
        let mountains = sim.hinges.iter().filter(|h| h.direction == FoldDirection::Mountain).count();
        let valleys = sim.hinges.iter().filter(|h| h.direction == FoldDirection::Valley).count();
        assert_eq!(valleys, 1);
        assert_eq!(mountains + valleys, sim.hinges.len());
    }

    #[test]
    fn set_fold_map_ignores_out_of_range_edges() {
        let n = 5;
        let mut sim = PaperSim::from_grid(n);
        let mut map = HashMap::new();
        map.insert((999u32, 1000u32), FoldSpec {
            target_angle: 1.0,
            compliance: 1e-4,
            direction: FoldDirection::Mountain,
            damping: 0.0,
        });
        sim.set_fold_map(map); // must not panic
        assert!(sim.hinges.is_empty());
    }

    // ── PaperSim::set_fold_angle ───────────────────────────────────────────

    #[test]
    fn set_fold_angle_signs_targets_by_direction() {
        let n = 6;
        let mut sim = PaperSim::from_grid(n);
        let mut map = central_vertical_fold(n, FoldDirection::Mountain);
        let col = n / 2;
        map.get_mut(&normalise_edge(col as u32, (n + col) as u32))
            .unwrap()
            .direction = FoldDirection::Valley;
        sim.set_fold_map(map);

        sim.set_fold_angle(90.0);
        let want = 90.0f32.to_radians();
        for h in &sim.hinges {
            match h.direction {
                FoldDirection::Mountain => assert!((h.target_angle + want).abs() < 1e-5),
                FoldDirection::Valley => assert!((h.target_angle - want).abs() < 1e-5),
            }
            assert_eq!(h.current_angle, 0.0, "set_fold_angle must not touch current_angle");
        }
    }

    #[test]
    fn set_fold_angle_zero_clears_all_targets() {
        let n = 6;
        let mut sim = PaperSim::from_grid(n);
        sim.set_fold_map(central_vertical_fold(n, FoldDirection::Mountain));
        sim.set_fold_angle(120.0);
        sim.set_fold_angle(0.0);
        for h in &sim.hinges {
            assert_eq!(h.target_angle, 0.0);
        }
    }

    // ── PaperSim::step ─────────────────────────────────────────────────────

    #[test]
    fn step_flat_paper_with_no_forces_stays_flat() {
        let mut sim = PaperSim::from_grid(6);
        let q0 = sim.core.q.clone();
        step_n(&mut sim, &no_gravity(), 30);
        assert!(all_finite(&sim.core.q));
        assert!(max_vertex_move(&sim.core.q, &q0) < 1e-3);
    }

    #[test]
    fn step_rate_limits_current_angle_then_saturates() {
        let n = 8;
        let mut sim = PaperSim::from_grid(n);
        sim.set_fold_map(central_vertical_fold(n, FoldDirection::Mountain));
        sim.set_fold_angle(180.0);
        let params = no_gravity();
        let max_delta = sim.fold_speed * params.time_step as f32;

        sim.step(&params);
        for h in &sim.hinges {
            assert!((h.current_angle + max_delta).abs() < 1e-6, "after 1 step: {}", h.current_angle);
        }

        // PI / max_delta ~= 63 steps to reach the target; 150 leaves margin.
        step_n(&mut sim, &params, 150);
        for h in &sim.hinges {
            assert!((h.current_angle + PI).abs() < 1e-4, "not saturated: {}", h.current_angle);
        }

        let before: Vec<f32> = sim.hinges.iter().map(|h| h.current_angle).collect();
        sim.step(&params);
        for (h, b) in sim.hinges.iter().zip(before) {
            assert!((h.current_angle - b).abs() < 1e-7, "moved past target");
        }
    }

    #[test]
    fn step_drives_the_fold_toward_its_target() {
        let n = 8;
        let mut sim = PaperSim::from_grid(n);
        sim.set_fold_map(central_vertical_fold(n, FoldDirection::Valley));
        sim.set_fold_angle(70.0);
        let params = no_gravity();

        let d0 = mean_hinge_dihedral(&sim);
        step_n(&mut sim, &params, 40);
        let d1 = mean_hinge_dihedral(&sim);
        step_n(&mut sim, &params, 140);
        let d2 = mean_hinge_dihedral(&sim);

        assert!(d1 > d0 + 0.1, "fold did not progress: d0={d0} d1={d1}");
        assert!(d2 >= d1 - 0.05, "fold regressed: d1={d1} d2={d2}");
        let target = 70.0f32.to_radians();
        assert!((d2 - target).abs() < 0.3, "d2={d2} target={target}");
        assert!(all_finite(&sim.core.q));
    }

    #[test]
    fn step_never_moves_the_pinned_corner() {
        let n = 8;
        let mut sim = PaperSim::from_grid(n);
        sim.set_fold_map(central_vertical_fold(n, FoldDirection::Mountain));
        sim.set_fold_angle(120.0);
        let pin = (n - 1) * n + (n - 1);
        step_n(&mut sim, &SimParams::default(), 80); // gravity on
        for k in 0..3 {
            assert!((sim.core.q[(pin, k)] - sim.core.q_rest[(pin, k)]).abs() < 1e-9);
        }
        assert!(all_finite(&sim.core.q));
    }

    #[test]
    fn step_substep_path_is_stable_and_folds() {
        let n = 8;
        let mut sim = PaperSim::from_grid(n);
        sim.set_fold_map(central_vertical_fold(n, FoldDirection::Valley));
        sim.set_fold_angle(60.0);
        let params = SimParams {
            gravity_enabled: false,
            use_distance_constraints: true,
            num_substeps: 4,
            ..SimParams::default()
        };
        step_n(&mut sim, &params, 60);
        assert!(all_finite(&sim.core.q));
        assert!(mean_hinge_dihedral(&sim).abs() > 0.2, "fold should have advanced");
        assert!(max_edge_strain(&sim.core) < 0.3, "strain {}", max_edge_strain(&sim.core));
    }

    #[test]
    fn step_excludes_hinge_diamonds_from_bending() {
        let n = 6;
        let mut sim = PaperSim::from_grid(n);
        sim.set_fold_map(central_vertical_fold(n, FoldDirection::Mountain));
        let skip: HashSet<usize> = sim.hinges.iter().map(|h| h.diamond_idx).collect();
        assert_eq!(skip.len(), sim.hinges.len(), "hinge diamonds must be distinct");
        for h in &sim.hinges {
            assert!(skip.contains(&h.diamond_idx));
        }
    }

    /// Two independently constructed sims are *not* bitwise-identical:
    /// `build_diamonds`, the `set_fold_map` fold-map loop, and the
    /// self-collision spatial hashes iterate `std::collections::HashMap`s, so
    /// the Gauss-Seidel constraint order differs and per-vertex positions drift
    /// by an FP-noise amount that varies run to run. The *macroscopic* fold
    /// outcome is still reproducible, which is what this asserts.
    #[test]
    fn step_fold_outcome_is_reproducible() {
        let n = 6;
        let build = || {
            let mut s = PaperSim::from_grid(n);
            s.set_fold_map(central_vertical_fold(n, FoldDirection::Mountain));
            s.set_fold_angle(90.0);
            s
        };
        let mut a = build();
        let mut b = build();
        let params = SimParams::default();
        for _ in 0..30 {
            a.step(&params);
            b.step(&params);
        }
        assert!(all_finite(&a.core.q) && all_finite(&b.core.q));

        let da = mean_hinge_dihedral(&a);
        let db = mean_hinge_dihedral(&b);
        assert!((da - db).abs() < 0.05, "fold angle not reproducible: {da} vs {db}");

        let ca = center_of_mass(&a.core.q);
        let cb = center_of_mass(&b.core.q);
        let com_drift = ((ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2) + (ca[2] - cb[2]).powi(2)).sqrt();
        assert!(com_drift < 0.02, "COM not reproducible: drift {com_drift}");
    }

    #[test]
    fn step_large_fold_stays_bounded() {
        let n = 8;
        let mut sim = PaperSim::from_grid(n);
        sim.set_fold_map(central_vertical_fold(n, FoldDirection::Valley));
        sim.set_fold_angle(150.0);
        let params = no_gravity();
        step_n(&mut sim, &params, 200);
        assert!(all_finite(&sim.core.q));
        for i in 0..sim.core.q.nrows() {
            for k in 0..3 {
                assert!(sim.core.q[(i, k)].abs() < 10.0, "vertex {i} escaped: {}", sim.core.q[(i, k)]);
            }
        }
        assert!(max_edge_strain(&sim.core) < 0.5, "strain {}", max_edge_strain(&sim.core));
    }

    #[test]
    fn step_unpinned_paper_falls_under_gravity() {
        let mut sim = PaperSim::from_grid(6);
        for i in 0..sim.core.w.len() {
            sim.core.w[i] = 1.0; // release the pin
        }
        let y0 = center_of_mass(&sim.core.q)[1];
        step_n(&mut sim, &SimParams::default(), 20);
        let y1 = center_of_mass(&sim.core.q)[1];
        assert!(y1 < y0 - 0.05, "paper did not fall: y0={y0} y1={y1}");
        assert!(all_finite(&sim.core.q));
    }

    #[test]
    fn step_pinned_paper_sags_but_holds_the_pin() {
        let n = 6;
        let mut sim = PaperSim::from_grid(n);
        let pin = (n - 1) * n + (n - 1);
        step_n(&mut sim, &SimParams::default(), 60);
        for k in 0..3 {
            assert!((sim.core.q[(pin, k)] - sim.core.q_rest[(pin, k)]).abs() < 1e-9);
        }
        // the diagonally opposite corner drops under gravity
        assert!(sim.core.q[(0, 1)] < sim.core.q_rest[(0, 1)] - 0.01);
        assert!(all_finite(&sim.core.q));
    }

    // ── PaperSim::from_crease_pattern ──────────────────────────────────────

    #[test]
    fn from_crease_pattern_builds_a_consistent_mesh() {
        let cp = sample_crease_pattern();
        let (sim, positions, faces, colors, edge_colors) =
            PaperSim::from_crease_pattern(&cp, 16);

        assert_eq!(positions.len(), colors.len());
        assert_eq!(positions.len(), sim.core.q.nrows());
        assert!(!faces.is_empty());
        for f in &faces {
            for &v in f {
                assert!((v as usize) < positions.len());
            }
        }

        assert!(!sim.crease_chains.is_empty());
        let expected_bends: usize = sim.crease_chains.iter()
            .map(|c| c.len().saturating_sub(2))
            .sum();
        assert_eq!(sim.crease_bends.len(), expected_bends);

        // `edge_colors` (fold edges from the CDT constraints + geometric
        // crease matches) index into `positions`. Not every entry survives as a
        // literal triangle edge — `fix_collinear_triangles` can re-split a
        // constrained span — so only the index validity is guaranteed here.
        for &(a, b) in edge_colors.keys() {
            assert!((a as usize) < positions.len() && (b as usize) < positions.len());
            assert_ne!(a, b);
        }

        assert!(!sim.hinges.is_empty(), "crossing creases should produce hinges");
        for h in &sim.hinges {
            assert!(h.diamond_idx < sim.core.diamonds.len());
        }
    }

    /// The crease-pattern path deliberately pins nothing (unlike `from_grid`) so
    /// the sheet floats freely while folding. `from_crease_pattern` still
    /// computes a `pin_idx` but passes an empty pin list to `from_mesh`.
    #[test]
    fn from_crease_pattern_leaves_every_vertex_free() {
        let cp = sample_crease_pattern();
        let (sim, ..) = PaperSim::from_crease_pattern(&cp, 16);
        assert!((0..sim.core.w.len()).all(|i| sim.core.w[i] == 1.0));
    }

    #[test]
    fn from_crease_pattern_directions_match_crease_types() {
        let cp = sample_crease_pattern();
        let (sim, ..) = PaperSim::from_crease_pattern(&cp, 16);
        let has_mountain = sim.hinges.iter().any(|h| h.direction == FoldDirection::Mountain);
        let has_valley = sim.hinges.iter().any(|h| h.direction == FoldDirection::Valley);
        assert!(has_mountain, "horizontal mountain crease should yield a mountain hinge");
        assert!(has_valley, "vertical valley crease should yield a valley hinge");
    }

    #[test]
    fn from_crease_pattern_is_deterministic() {
        let cp = sample_crease_pattern();
        let (s1, p1, ..) = PaperSim::from_crease_pattern(&cp, 16);
        let (s2, p2, ..) = PaperSim::from_crease_pattern(&cp, 16);
        assert_eq!(p1.len(), p2.len());
        assert_eq!(s1.hinges.len(), s2.hinges.len());
        assert_eq!(s1.core.diamonds.len(), s2.core.diamonds.len());
    }

    #[test]
    fn from_crease_pattern_low_resolution_does_not_panic() {
        let cp = sample_crease_pattern();
        let (sim, positions, ..) = PaperSim::from_crease_pattern(&cp, 4);
        assert_eq!(positions.len(), sim.core.q.nrows());
    }

    #[test]
    fn from_crease_pattern_mesh_steps_without_blowing_up() {
        let cp = sample_crease_pattern();
        let (mut sim, ..) = PaperSim::from_crease_pattern(&cp, 12);
        sim.set_fold_angle(45.0);
        step_n(&mut sim, &no_gravity(), 60);
        assert!(all_finite(&sim.core.q));
    }
}
