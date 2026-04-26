use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};

use nalgebra as na;
use web_sys::console;

use crate::{cloth::Cloth, gpu::GpuContext, params::SimParams};
use super::shared::{Positions, ClothSimCore};
use super::traits::MeshSim;
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
    pub fn from_cloth(cloth: &Cloth) -> Self {
        Self {
            core: {
                let n = cloth.resolution as usize;
                ClothSimCore::from_cloth(cloth, &[(n-1)*n + (n-1)]) // upper-right only
            },
            hinges: Vec::new(),
            fold_speed: 5.0,
            crease_chains: Vec::new(),
            crease_bend_compliance: 1e-6,
            crease_bends: Vec::new(),
        }
    }

    /// Create PaperSim from a crease pattern overlaid on a 64x64 grid.
    /// Returns (PaperSim, positions, faces, colors) for building Cloth.
    pub fn from_crease_pattern(cp: &CreasePattern) -> (Self, Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<[f32; 3]>, HashMap<(u32, u32), CreaseType>) {
        const GRID_RES: usize = 1;
        let (positions, faces, fold_edges, crease_chains, split_creases) = cp.build_mesh(GRID_RES);

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
            console::warn_1(&format!(
                "WARNING: Found {} overlapping edge pairs in mesh!",
                overlapping.len()
            ).into());
            for (e1, e2) in &overlapping {
                console::warn_1(&format!(
                    "  Overlap: edge [{}, {}] and edge [{}, {}]",
                    e1[0], e1[1], e2[0], e2[1]
                ).into());
            }
        } else {
            console::log_1(&"No overlapping edges found in mesh".into());
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
            console::log_1(&format!(
                "Found {} additional edges lying on crease lines",
                extra_edges
            ).into());
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

                self.hinges.push(HingeConstraint {
                    diamond_idx: di,
                    rest_angle,
                    target_angle: spec.target_angle,
                    current_angle: 0.0,
                    compliance: spec.compliance,
                    direction: spec.direction,
                    damping: spec.damping,
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

        // 2. Reset all lambdas
        self.core.reset_lambdas();
        for h in &mut self.hinges { h.lambda = 0.0; }
        for cb in &mut self.crease_bends { cb.lambda = 0.0; }

        // 3. Unified XPBD iteration loop
        let cb_alpha_tilde = self.crease_bend_compliance / (dt * dt);
        for _ in 0..params.constraint_iters {
            self.core.solve_stretch(params);
            self.core.solve_bend(params, &skip);
            self.core.solve_pins(params);
            self.core.solve_pulling(params);

            // Hinge dihedral constraints with XPBD damping
            for h in &mut self.hinges {
                let [a, b, c, d] = self.core.diamonds[h.diamond_idx];
                let alpha_tilde = h.compliance / (dt * dt);
                // γ = α̃ β̃ / Δt = (compliance/dt²)(dt²·damping)/dt = compliance·damping/dt
                let gamma = h.compliance * h.damping / dt;
                let goal_dihedral = h.rest_angle + h.current_angle;

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

        self.core.solve_self_collision(params);

        // 4. Remove rigid-body rotation from constraint corrections
        // self.core.remove_rigid_rotation();

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
    let raw_theta = sin_t.atan2(cos_t);
    // Shift by π so flat = 0 (matching dihedral_angle convention)
    let theta = {
        let shifted = raw_theta + std::f32::consts::PI;
        (shifted + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI) - std::f32::consts::PI
    };

    // C = θ - goal, already in [-π, π] range
    let c_val = theta - goal_angle;
    if c_val.abs() < 1e-4 { return; }

    // Gradients ∂θ/∂pᵢ  (same derivation as PBD bending, Müller 2006 §4)
    let g_c = (e_len / n1l) * n1h;
    let g_d = -(e_len / n2l) * n2h;
    let g_a = -((p3 - p2).dot(&e_hat) / n1l) * n1h
            + ((p4 - p2).dot(&e_hat) / n2l) * n2h;
    let g_b =  ((p3 - p1).dot(&e_hat) / n1l) * n1h
            - ((p4 - p1).dot(&e_hat) / n2l) * n2h;

    let wa = w[a]; let wb = w[b]; let wc = w[c]; let wd = w[d];

    // Damping term: γ ∇C·(x - x^n)
    let dx_a = p1 - row3(q_prev, a);
    let dx_b = p2 - row3(q_prev, b);
    let dx_c = p3 - row3(q_prev, c);
    let dx_d = p4 - row3(q_prev, d);
    let grad_dot_dx = g_a.dot(&dx_a) + g_b.dot(&dx_b) + g_c.dot(&dx_c) + g_d.dot(&dx_d);

    // XPBD denominator includes compliance and damping terms
    let weighted_grads = wa * g_a.norm_squared()
                       + wb * g_b.norm_squared()
                       + wc * g_c.norm_squared()
                       + wd * g_d.norm_squared();
    let denom = (1.0 + gamma) * weighted_grads + alpha_tilde;
    if denom < 1e-12 { return; }

    let dl = -(c_val + alpha_tilde * *lambda + gamma * grad_dot_dx) / denom;
    *lambda += dl;

    // Debug: log every ~3000 calls
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if count % 3000 == 0 {
        console::log_1(&format!(
            "θ={:.3} goal={:.3} c={:.4} dl={:.6} γ∇C·dx={:.4} γ={:.2e}",
            theta, goal_angle, c_val, dl, gamma * grad_dot_dx, gamma
        ).into());
    }

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
    let p1 = row3(q, a as usize); let p2 = row3(q, b as usize);
    let p3 = row3(q, c as usize); let p4 = row3(q, d as usize);
    let e = p2 - p1;
    let e_len = e.norm();
    if e_len < 1e-8 { return 0.0; }
    let e_hat = e / e_len;
    let n1 = (p2 - p1).cross(&(p3 - p1));
    let n2 = (p2 - p1).cross(&(p4 - p1));
    let n1l = n1.norm(); let n2l = n2.norm();
    if n1l < 1e-8 || n2l < 1e-8 { return 0.0; }
    let n1h = n1 / n1l; let n2h = n2 / n2l;
    let cos_t = n1h.dot(&n2h).clamp(-1.0, 1.0);
    let raw = n1h.cross(&n2h).dot(&e_hat).atan2(cos_t);
    // Shift by π so flat (raw = ±π) becomes 0, and normalize to [-π, π]
    let shifted = raw + std::f32::consts::PI;
    (shifted + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI) - std::f32::consts::PI
}

#[inline]
fn row3(q: &Positions, i: usize) -> na::Vector3<f32> {
    na::Vector3::new(q[(i, 0)], q[(i, 1)], q[(i, 2)])
}

#[inline]
fn add_scaled(q: &mut Positions, i: usize, dv: na::Vector3<f32>) {
    q[(i, 0)] += dv[0]; q[(i, 1)] += dv[1]; q[(i, 2)] += dv[2];
}
