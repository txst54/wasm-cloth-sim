use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};

use nalgebra as na;

use crate::{cloth::Cloth, gpu::GpuContext, params::SimParams};
use super::shared::{Positions, SimCore};
use super::traits::MeshSim;

// ── Fold specification ────────────────────────────────────────────────────────

/// Describes the target fold for one hinge edge.
pub struct FoldSpec {
    /// Target dihedral angle (radians) at `fold_progress = 1`.
    /// A flat mesh starts near π; folding inward decreases it toward 0.
    pub target_angle: f32,
    /// XPBD compliance α (m²/N equivalent for angular constraints).
    /// Smaller = stiffer / faster fold.  Typical range: 1e-6 … 1e-2.
    /// Default 1e-4 is stable for dt=0.01 and avoids the divergence that
    /// pure PBD produces on large-angle corrections.
    pub compliance: f32,
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
    /// Index into `SimCore::diamonds`.
    pub diamond_idx: usize,
    /// Dihedral angle at construction (flat mesh → ≈ π).
    pub rest_angle: f32,
    /// Desired dihedral angle set by the user.
    pub target_angle: f32,
    /// Current effective goal — smoothly advances toward `target_angle` each
    /// step.  The constraint drives the mesh toward this value.
    pub current_angle: f32,
    /// XPBD compliance α.
    pub compliance: f32,
}

// ── PaperSim ──────────────────────────────────────────────────────────────────

pub struct PaperSim {
    pub core: SimCore,
    pub hinges: Vec<HingeConstraint>,
    /// Maximum rate at which `current_angle` advances toward `target_angle`,
    /// in radians per second.  Keeping this moderate (e.g. 3–10 rad/s) caps
    /// the per-step constraint violation and prevents divergence on fast slider
    /// moves.  Default: 5.0 rad/s (≈ 286 °/s — a 90° fold takes ~0.3 s).
    pub fold_speed: f32,
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
        }
    }

    /// Register hinges from a fold map.
    /// Keys: `(min_v, max_v)` edge pairs.  Edges not in the diamond list are ignored.
    pub fn set_fold_map(&mut self, fold_map: HashMap<(u32, u32), FoldSpec>) {
        let edge_to_diamond: HashMap<(u32, u32), usize> = self
            .core.diamonds.iter().enumerate()
            .map(|(di, &[a, b, _, _])| (normalise_edge(a, b), di))
            .collect();

        self.hinges.clear();
        for ((ea, eb), spec) in fold_map {
            let key = normalise_edge(ea, eb);
            if let Some(&di) = edge_to_diamond.get(&key) {
                let [a, b, c, d] = self.core.diamonds[di];
                let rest_angle = dihedral_angle(&self.core.q, a, b, c, d);
                self.hinges.push(HingeConstraint {
                    diamond_idx: di,
                    rest_angle,
                    target_angle: spec.target_angle,
                    current_angle: rest_angle,
                    compliance: spec.compliance,
                });
            }
        }
    }

    pub fn step(&mut self, params: &SimParams) {
        let dt = params.time_step as f32;
        let max_delta = self.fold_speed * dt;

        // Advance each hinge's effective goal toward its user-set target by at
        // most `fold_speed * dt` radians.  This caps |C| per step.
        for h in self.hinges.iter_mut() {
            let delta = (h.target_angle - h.current_angle).clamp(-max_delta, max_delta);
            h.current_angle += delta;
        }

        let skip: HashSet<usize> = self.hinges.iter().map(|h| h.diamond_idx).collect();

        // Run the core PBD step with NO hinge constraints in the inner loop.
        //
        // Applying hinges inside the loop interleaves them with stretch/bending
        // iterations: every stretch correction "undoes" part of the hinge
        // correction, and the hinge pushes back next iteration.  This
        // Gauss-Seidel ping-pong injects energy into the mesh and produces the
        // travelling-wave / bump instability the user observed.
        self.core.step(params, &skip, |_q, _w| {});

        // Apply hinge corrections ONCE, after all PBD iterations have settled.
        // At this point stretch/bend are satisfied; the hinge nudges the geometry
        // by at most max_delta radians without triggering further constraint fights.
        for h in &self.hinges {
            let [a, b, c, d] = self.core.diamonds[h.diamond_idx];
            let alpha_tilde  = h.compliance / (dt * dt);
            let mut lambda   = 0.0_f32;
            apply_hinge_xpbd(
                &mut self.core.q, &self.core.w,
                a as usize, b as usize, c as usize, d as usize,
                h.current_angle, alpha_tilde, &mut lambda,
            );
        }

        // Zero velocity on fold-line vertices (the 'a' and 'b' edge endpoints
        // of each hinge diamond).  The post-loop hinge correction shifts these
        // positions slightly; without this zero-out the shift would become a
        // velocity in the next step and propagate as a shear/stretch wave through
        // the mesh.  Making fold-line vertices quasi-static (position-driven,
        // no inertia) eliminates the wave source while leaving panel vertices
        // (c/d) free to carry legitimate dynamics.
        let mut zeroed: HashSet<usize> = HashSet::new();
        for h in &self.hinges {
            let [a, b, _, _] = self.core.diamonds[h.diamond_idx];
            for &vi in &[a as usize, b as usize] {
                if zeroed.insert(vi) && self.core.w[vi] > 0.0 {
                    self.core.v[(vi, 0)] = 0.0;
                    self.core.v[(vi, 1)] = 0.0;
                    self.core.v[(vi, 2)] = 0.0;
                }
            }
        }
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

    let c_val = theta - goal_angle;
    if c_val.abs() < 1e-8 { return; }

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
