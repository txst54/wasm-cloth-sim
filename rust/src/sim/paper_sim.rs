use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};

use nalgebra as na;
use web_sys::console;

use crate::{cloth::Cloth, gpu::GpuContext, params::SimParams};
use super::shared::{Positions, SimCore};
use super::traits::MeshSim;
use super::crease::{CreasePattern, CreaseType};

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
    pub direction: FoldDirection,
    /// XPBD Lagrange multiplier (reset each substep).
    pub lambda: f32,
}

// ── PaperSim ──────────────────────────────────────────────────────────────────

pub struct PaperSim {
    pub core: SimCore,
    pub hinges: Vec<HingeConstraint>,
    pub fold_speed: f32,
    pub crease_chains: Vec<Vec<u32>>,
    pub collinearity_compliance: f32,
    pub collinearity_lambdas: Vec<f32>,
}

impl Deref for PaperSim {
    type Target = SimCore;
    fn deref(&self) -> &SimCore { &self.core }
}

impl DerefMut for PaperSim {
    fn deref_mut(&mut self) -> &mut SimCore { &mut self.core }
}

impl PaperSim {
    pub fn from_cloth(cloth: &Cloth) -> Self {
        Self {
            core: {
                let n = cloth.resolution as usize;
                SimCore::from_cloth(cloth, &[(n-1)*n + (n-1)]) // upper-right only
            },
            hinges: Vec::new(),
            fold_speed: 5.0,
            crease_chains: Vec::new(),
            collinearity_compliance: 1e-4,
            collinearity_lambdas: Vec::new(),
        }
    }

    /// Create PaperSim from a crease pattern overlaid on a 64x64 grid.
    /// Returns (PaperSim, positions, faces, colors) for building Cloth.
    pub fn from_crease_pattern(cp: &CreasePattern) -> (Self, Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<[f32; 3]>, HashMap<(u32, u32), CreaseType>) {
        const GRID_RES: usize = 32;
        let (positions, faces, fold_edges, crease_chains) = cp.build_mesh(GRID_RES);

        // Generate colors: white for normal faces, red-ish for mountain, blue-ish for valley
        let mut vertex_colors = vec![[0.85f32, 0.85, 0.85]; positions.len()];

        // Color vertices that are part of fold edges
        for (&(a, b), &crease_type) in &fold_edges {
            let color = match crease_type {
                CreaseType::Mountain => [1.0, 0.6, 0.6],  // reddish
                CreaseType::Valley => [0.6, 0.6, 1.0],    // bluish
                CreaseType::Boundary => [0.85, 0.85, 0.85],
            };
            vertex_colors[a as usize] = color;
            vertex_colors[b as usize] = color;
        }

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
        let core = SimCore::from_mesh(&positions, &faces, &[pin_idx]);

        // Count collinearity interior vertices for lambda storage
        let num_collinearity = crease_chains.iter()
            .map(|c| c.len().saturating_sub(2))
            .sum();

        let mut sim = Self {
            core,
            hinges: Vec::new(),
            fold_speed: 5.0,
            crease_chains,
            collinearity_compliance: 1e-4,
            collinearity_lambdas: vec![0.0; num_collinearity],
        };

        // Build fold map from crease pattern edges
        let mut fold_map: HashMap<(u32, u32), FoldSpec> = HashMap::new();
        let edge_colors = fold_edges.clone();
        for ((a, b), crease_type) in fold_edges {
            let direction = match crease_type {
                CreaseType::Mountain => FoldDirection::Mountain,
                CreaseType::Valley => FoldDirection::Valley,
                CreaseType::Boundary => continue,
            };
            fold_map.insert((a, b), FoldSpec {
                target_angle: 0.0,  // start flat (0 = unfolded)
                compliance: 1e-4,
                direction,
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

                self.hinges.push(HingeConstraint {
                    diamond_idx: di,
                    rest_angle,
                    target_angle: spec.target_angle,
                    current_angle: 0.0,
                    compliance: spec.compliance,
                    direction: spec.direction,
                    lambda: 0.0,
                });
            } else {
                dropped_edges += 1;
            }
        }
        if dropped_edges > 0 {
            console::warn_1(&format!(
                "PaperSim: {} fold edge(s) have no corresponding diamond (boundary edges or triangulation mismatch)",
                dropped_edges
            ).into());
        }

        let mountain_count = self.hinges.iter().filter(|h| h.direction == FoldDirection::Mountain).count();
        let valley_count = self.hinges.iter().filter(|h| h.direction == FoldDirection::Valley).count();
        console::log_1(&format!(
            "Hinges: {} total, rest ≈ {:.3}, {} mountain, {} valley",
            self.hinges.len(),
            self.hinges.first().map_or(0.0, |h| h.rest_angle),
            mountain_count, valley_count
        ).into());
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

        // Rate-limit each hinge toward its target
        for h in self.hinges.iter_mut() {
            let delta = (h.target_angle - h.current_angle).clamp(-max_delta, max_delta);
            h.current_angle += delta;
        }

        let skip: HashSet<usize> = self.hinges.iter().map(|h| h.diamond_idx).collect();

        // 1. Predict
        self.core.predict(params);
        let sc_pairs = self.core.precompute_collision_pairs(params);

        // 2. Reset all lambdas
        self.core.reset_lambdas();
        for h in &mut self.hinges { h.lambda = 0.0; }
        for cl in &mut self.collinearity_lambdas { *cl = 0.0; }

        // 3. Unified XPBD iteration loop
        let col_alpha_tilde = self.collinearity_compliance / (dt * dt);
        for _ in 0..params.constraint_iters {
            self.core.solve_stretch(params);
            self.core.solve_bend(params, &skip);
            self.core.solve_pins(params);
            self.core.solve_pulling(params);

            // Hinge dihedral constraints
            for h in &mut self.hinges {
                let [a, b, c, d] = self.core.diamonds[h.diamond_idx];
                let alpha_tilde = h.compliance / (dt * dt);
                let goal_dihedral = h.rest_angle + h.current_angle;

                apply_hinge_xpbd(
                    &mut self.core.q, &self.core.w,
                    a as usize, b as usize, c as usize, d as usize,
                    goal_dihedral, alpha_tilde, &mut h.lambda,
                );
            }

            // Collinearity constraints
            let mut cl_idx = 0;
            for chain in &self.crease_chains {
                if chain.len() < 3 {
                    cl_idx += chain.len().saturating_sub(2);
                    continue;
                }
                let a = chain[0];
                let b = chain[chain.len() - 1];
                for &vi in &chain[1..chain.len()-1] {
                    apply_collinearity_xpbd(
                        &mut self.core.q, &self.core.w,
                        vi as usize, a as usize, b as usize,
                        col_alpha_tilde, &mut self.collinearity_lambdas[cl_idx],
                    );
                    cl_idx += 1;
                }
            }

            self.core.solve_self_collision(params, &sc_pairs);
        }

        // 4. Remove rigid-body rotation from constraint corrections
        self.core.remove_rigid_rotation();

        // 5. Derive velocity from position change
        self.core.update_velocity(params);
    }

    pub fn write_to_cloth(&self, cloth: &mut Cloth, ctx: &GpuContext) {
        self.core.write_to_cloth(cloth, ctx);
    }
}

impl MeshSim for PaperSim {
    fn step(&mut self, params: &SimParams)                        { self.step(params); }
    fn write_to_cloth(&self, cloth: &mut Cloth, ctx: &GpuContext) { self.core.write_to_cloth(cloth, ctx); }
    fn positions(&self) -> &Positions                             { &self.core.q }
    fn set_clicked_vertex(&mut self, vi: Option<usize>)           { self.core.clicked_vertex = vi; }
    fn set_mouse_pos(&mut self, pos: [f32; 3])                    { self.core.mouse_pos = pos; }
}

// ── XPBD dihedral-angle constraint ───────────────────────────────────────────

/// XPBD dihedral-angle constraint (Müller et al. 2020).
///
/// Unlike plain PBD this accumulates the Lagrange multiplier `λ` across
/// iterations within the same substep, which prevents the large-angle
/// overcorrection that caused divergence under the old PBD formulation.
///
/// Formula:
/// ```text
/// Δλ = -(C + α̃ λ) / (Σ wᵢ |∇C|² + α̃)
/// λ  += Δλ
/// Δpᵢ = wᵢ Δλ ∇C_i
/// ```
/// where `α̃ = α / dt²` and the constraint `C = θ − θ_goal`.
fn apply_hinge_xpbd(
    q:           &mut Positions,
    w:           &na::DVector<f32>,
    a: usize, b: usize, c: usize, d: usize,
    goal_angle:  f32,
    alpha_tilde: f32,   // compliance / dt²
    lambda:      &mut f32,
) {
    let p1 = row3(q, a); let p2 = row3(q, b);
    let p3 = row3(q, c); let p4 = row3(q, d);

    let e = p2 - p1;
    let e_len = e.norm();
    if e_len < 1e-8 { return; }
    let e_hat = e / e_len;

    let n1 = (p2 - p1).cross(&(p3 - p1));
    let n2 = (p2 - p1).cross(&(p4 - p1));
    let n1l = n1.norm(); let n2l = n2.norm();
    if n1l < 1e-8 || n2l < 1e-8 { return; }
    let n1h = n1 / n1l; let n2h = n2 / n2l;

    let cos_t = n1h.dot(&n2h).clamp(-1.0, 1.0);
    let sin_t = n1h.cross(&n2h).dot(&e_hat);
    let theta = sin_t.atan2(cos_t);

    let mut c_val = theta - goal_angle;
    // Wrap to [-π, π] for shortest angular path (handles ±π equivalence)
    let mut c_val = theta - goal_angle;
    c_val = (c_val + std::f32::consts::PI)
        .rem_euclid(2.0 * std::f32::consts::PI)
        - std::f32::consts::PI;
    if c_val.abs() < 1e-4 { return; }

    // Gradients ∂θ/∂pᵢ  (same derivation as PBD bending, Müller 2006 §4)
    let g_c = (e_len / n1l) * n1h;
    let g_d = -(e_len / n2l) * n2h;
    let g_a = -((p3 - p2).dot(&e_hat) / n1l) * n1h
            + ((p4 - p2).dot(&e_hat) / n2l) * n2h;
    let g_b =  ((p3 - p1).dot(&e_hat) / n1l) * n1h
            - ((p4 - p1).dot(&e_hat) / n2l) * n2h;

    let wa = w[a]; let wb = w[b]; let wc = w[c]; let wd = w[d];

    // XPBD denominator includes compliance term, bounding per-iteration correction.
    let weighted_grads = wa * g_a.norm_squared()
                       + wb * g_b.norm_squared()
                       + wc * g_c.norm_squared()
                       + wd * g_d.norm_squared();
    let denom = weighted_grads + alpha_tilde;
    if denom < 1e-12 { return; }

    let dl = -(c_val + alpha_tilde * *lambda) / denom;
    *lambda += dl;

    add_scaled(q, a, wa * dl * g_a);
    add_scaled(q, b, wb * dl * g_b);
    add_scaled(q, c, wc * dl * g_c);
    add_scaled(q, d, wd * dl * g_d);
}


// ── XPBD collinearity constraint ──────────────────────────────────────────────

/// XPBD constraint that pushes vertex `p` toward the line defined by endpoints `a` and `b`.
/// C = distance from p to line ab.
/// Only moves vertex p (endpoints are assumed fixed by hinge constraints).
fn apply_collinearity_xpbd(
    q:           &mut Positions,
    w:           &na::DVector<f32>,
    p: usize, a: usize, b: usize,
    alpha_tilde: f32,
    lambda:      &mut f32,
) {
    let wp = w[p];
    if wp < 1e-12 { return; }

    let pa = row3(q, a);
    let pb = row3(q, b);
    let pp = row3(q, p);

    let edge = pb - pa;
    let edge_len_sq = edge.norm_squared();
    if edge_len_sq < 1e-12 { return; }

    // Vector from a to p
    let v = pp - pa;

    // Project p onto line ab: proj = pa + t * edge
    let t = v.dot(&edge) / edge_len_sq;
    let proj = pa + t * edge;

    // Perpendicular vector from line to p
    let perp = pp - proj;
    let dist = perp.norm();

    if dist < 1e-6 { return; }

    // Constraint value C = distance to line
    let c_val = dist;

    // Gradient: unit vector pointing from line toward p
    let grad = perp / dist;

    // XPBD: Δλ = -(C + α̃λ) / (w|∇C|² + α̃)
    // Since |∇C| = 1: denom = w + α̃
    let denom = wp + alpha_tilde;
    if denom < 1e-12 { return; }

    let dl = -(c_val + alpha_tilde * *lambda) / denom;
    *lambda += dl;

    // Move p toward the line
    add_scaled(q, p, wp * dl * grad);
}

// ── Tiny inline utilities ─────────────────────────────────────────────────────

#[inline]
fn normalise_edge(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

fn dihedral_angle(q: &Positions, a: u32, b: u32, c: u32, d: u32) -> f32 {
    let p1 = row3(q, a as usize); let p2 = row3(q, b as usize);
    let p3 = row3(q, c as usize); let p4 = row3(q, d as usize);
    let e = p2 - p1;
    let e_len = e.norm();
    if e_len < 1e-8 { return std::f32::consts::PI; }
    let e_hat = e / e_len;
    let n1 = (p2 - p1).cross(&(p3 - p1));
    let n2 = (p2 - p1).cross(&(p4 - p1));
    let n1l = n1.norm(); let n2l = n2.norm();
    if n1l < 1e-8 || n2l < 1e-8 { return std::f32::consts::PI; }
    let n1h = n1 / n1l; let n2h = n2 / n2l;
    let cos_t = n1h.dot(&n2h).clamp(-1.0, 1.0);
    n1h.cross(&n2h).dot(&e_hat).atan2(cos_t)
}

#[inline]
fn row3(q: &Positions, i: usize) -> na::Vector3<f32> {
    na::Vector3::new(q[(i, 0)], q[(i, 1)], q[(i, 2)])
}

#[inline]
fn add_scaled(q: &mut Positions, i: usize, dv: na::Vector3<f32>) {
    q[(i, 0)] += dv[0]; q[(i, 1)] += dv[1]; q[(i, 2)] += dv[2];
}
