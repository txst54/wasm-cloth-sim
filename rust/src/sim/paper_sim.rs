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
        for (&(a, b), &crease_type) in &all_fold_edges {
            let color = match crease_type {
                CreaseType::Mountain => [1.0, 0.6, 0.6],  // reddish
                CreaseType::Valley => [0.6, 0.6, 1.0],    // bluish
                CreaseType::Boundary => [0.85, 0.85, 0.85],
            };
            vertex_colors[a as usize] = color;
            vertex_colors[b as usize] = color;
        }

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
fn apply_hinge_xpbd(
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
fn normalise_edge(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Compute dihedral "fold angle" for a diamond.
/// Returns 0 for a flat mesh, negative for mountain folds, positive for valley folds.
/// Range: [-π, π]
fn dihedral_angle(q: &Positions, a: u32, b: u32, c: u32, d: u32) -> f32 {
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
