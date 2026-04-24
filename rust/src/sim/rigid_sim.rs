use nalgebra as na;

use crate::rigid_body::RigidBodyInstance;

// ── Params ────────────────────────────────────────────────────────────────────

pub struct RigidSimParams {
    pub time_step:         f32,
    pub gravity_enabled:   bool,
    /// Gravitational constant G for N-body attraction (not a constant downward g).
    pub gravity_g:         f32,
    pub newton_max_iters:  u32,
    pub newton_tol:        f32,
}

impl Default for RigidSimParams {
    fn default() -> Self {
        Self {
            time_step:        1e-2,
            gravity_enabled:  true,
            gravity_g:        6.674e-11,
            newton_max_iters: 32,
            newton_tol:       1e-6,
        }
    }
}

// ── RigidSimCore ──────────────────────────────────────────────────────────────

pub struct RigidSimCore {
    pub bodies: Vec<RigidBodyInstance>,
    pub time:   f32,
}

impl RigidSimCore {
    pub fn new(bodies: Vec<RigidBodyInstance>) -> Self {
        Self { bodies, time: 0.0 }
    }

    pub fn step(&mut self, params: &RigidSimParams) {
        self.time += params.time_step;
        self.integrate_translational(params);
        self.integrate_rotational(params);
    }

    /// Symplectic-Euler translational step with pairwise N-body gravity.
    ///
    /// Mirrors the C++ translational block: advance positions first with the
    /// current velocity, compute forces at the new positions, then update
    /// velocities.
    fn integrate_translational(&mut self, params: &RigidSimParams) {
        let h = params.time_step;
        let n = self.bodies.len();

        // q_{k+1} = q_k + h v_k
        let c_next: Vec<na::Vector3<f32>> = self.bodies.iter()
            .map(|b| b.c + h * b.cvel)
            .collect();

        // Gravitational forces evaluated at the new positions
        let mut forces = vec![na::Vector3::<f32>::zeros(); n];
        if params.gravity_enabled {
            for i in 0..n {
                for j in (i + 1)..n {
                    let mi = self.bodies[i].density * self.bodies[i].get_template().volume;
                    let mj = self.bodies[j].density * self.bodies[j].get_template().volume;
                    let mu = params.gravity_g * mi * mj;
                    let r  = c_next[i] - c_next[j];
                    let d  = r.norm();
                    if d < 1e-10 { continue; }
                    let d3 = d * d * d;
                    let f  = -mu * r / d3;
                    forces[i] += f;
                    forces[j] -= f;
                }
            }
        }

        // v_{k+1} = v_k + h M^{-1} F
        for i in 0..n {
            let mass = self.bodies[i].density * self.bodies[i].get_template().volume;
            self.bodies[i].cvel += h * forces[i] / mass;
            self.bodies[i].c    = c_next[i];
        }
    }

    /// Lie-group variational integrator for the rotational DOFs.
    ///
    /// 1. Advance theta explicitly: θ_{k+1} = log(R(h ω_k) R(θ_k))
    /// 2. Newton solve for ω_{k+1} via the discrete Euler-Lagrange equation.
    fn integrate_rotational(&mut self, params: &RigidSimParams) {
        let h = params.time_step;
        let n = self.bodies.len();

        // Save current angular velocities ω_k
        let omega_k: Vec<na::Vector3<f32>> = self.bodies.iter().map(|b| b.w).collect();

        // Advance rotation vectors: θ_{k+1} = log(R(h ω_k) R(θ_k))
        let theta_next: Vec<na::Vector3<f32>> = self.bodies.iter()
            .zip(&omega_k)
            .map(|(b, &w)| advance_theta(b.theta, w, h))
            .collect();

        // Newton solve for ω_{k+1}
        let mut omega_next = omega_k.clone();

        for _ in 0..params.newton_max_iters {
            let residuals: Vec<na::Vector3<f32>> = (0..n)
                .map(|i| {
                    let inertia = self.bodies[i].get_template()
                        .inertia_body_scaled(self.bodies[i].density);
                    residual_omega_i(theta_next[i], omega_k[i], omega_next[i], &inertia, h)
                })
                .collect();

            let res_norm: f32 = residuals.iter()
                .map(|r| r.norm_squared())
                .sum::<f32>()
                .sqrt();
            if res_norm <= params.newton_tol { break; }

            for i in 0..n {
                let inertia = self.bodies[i].get_template()
                    .inertia_body_scaled(self.bodies[i].density);
                let j = approx_jacobian_omega_i(theta_next[i], omega_next[i], &inertia, h);
                if let Some(j_inv) = j.try_inverse() {
                    omega_next[i] -= j_inv * residuals[i];
                }
            }
        }

        for i in 0..n {
            self.bodies[i].theta = theta_next[i];
            self.bodies[i].w     = omega_next[i];
        }
    }
}

// ── Rotation utilities ────────────────────────────────────────────────────────

/// Skew-symmetric matrix (hat map) of a 3-vector.
#[inline]
fn hat(v: na::Vector3<f32>) -> na::Matrix3<f32> {
    na::Matrix3::new(
         0.0, -v[2],  v[1],
        v[2],   0.0, -v[0],
       -v[1],  v[0],   0.0,
    )
}

/// Rodrigues rotation matrix from an axis-angle vector θ.
/// R(θ) = exp(hat(θ))
fn rotation_matrix(theta: na::Vector3<f32>) -> na::Matrix3<f32> {
    let phi = theta.norm();
    if phi < 1e-10 {
        return na::Matrix3::identity();
    }
    let axis = theta / phi;
    let s = phi.sin();
    let c = phi.cos();
    c * na::Matrix3::identity() + (1.0 - c) * axis * axis.transpose() + s * hat(axis)
}

/// Matrix logarithm of a rotation matrix — returns the axis-angle vector θ
/// such that R = exp(hat(θ)).
fn rotation_log(r: na::Matrix3<f32>) -> na::Vector3<f32> {
    let cos_angle = ((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();
    if angle.abs() < 1e-8 {
        return na::Vector3::zeros();
    }
    let factor = angle / (2.0 * angle.sin());
    na::Vector3::new(
        r[(2, 1)] - r[(1, 2)],
        r[(0, 2)] - r[(2, 0)],
        r[(1, 0)] - r[(0, 1)],
    ) * factor
}

/// Explicit rotation-vector update:
///   θ_{k+1} = log(R(h ω) R(θ_k))
fn advance_theta(theta: na::Vector3<f32>, omega: na::Vector3<f32>, h: f32) -> na::Vector3<f32> {
    rotation_log(rotation_matrix(h * omega) * rotation_matrix(theta))
}

/// Right-trivialized derivative of exp at θ (the "T matrix"):
///   T(θ) = I − (1−cos φ)/φ² hat(θ) + (φ−sin φ)/φ³ hat(θ)²
///
/// Satisfies  ω_body = T(θ) θ̇  where  Ṙ = R hat(ω_body).
fn t_matrix(theta: na::Vector3<f32>) -> na::Matrix3<f32> {
    let phi = theta.norm();
    if phi < 1e-8 {
        return na::Matrix3::identity();
    }
    let hat_t = hat(theta);
    let phi2  = phi * phi;
    na::Matrix3::identity()
        - (1.0 - phi.cos()) / phi2        * hat_t
        + (phi - phi.sin()) / (phi2 * phi) * (hat_t * hat_t)
}

/// Transpose of the T-matrix:
///   T(θ)ᵀ = I + (1−cos φ)/φ² hat(θ) + (φ−sin φ)/φ³ hat(θ)²
///
/// (hat(θ)ᵀ = −hat(θ) flips the sign on the middle term.)
fn t_matrix_transpose(theta: na::Vector3<f32>) -> na::Matrix3<f32> {
    let phi = theta.norm();
    if phi < 1e-8 {
        return na::Matrix3::identity();
    }
    let hat_t = hat(theta);
    let phi2  = phi * phi;
    na::Matrix3::identity()
        + (1.0 - phi.cos()) / phi2        * hat_t
        + (phi - phi.sin()) / (phi2 * phi) * (hat_t * hat_t)
}

/// Inverse-transpose of the T-matrix:
///   T(θ)⁻ᵀ = I − ½ hat(θ) + c hat(θ)²
///   where c = 1/φ² − (1+cos φ)/(2φ sin φ)
///
/// Limit at φ→0: c → 1/12.
fn t_matrix_inv_transpose(theta: na::Vector3<f32>) -> na::Matrix3<f32> {
    let phi = theta.norm();
    let hat_t = hat(theta);
    let hat2 = hat_t * hat_t;
    if phi < 1e-8 {
        return na::Matrix3::identity() - 0.5 * hat_t + (1.0 / 12.0) * hat2;
    }
    let phi2 = phi * phi;
    let c = 1.0 / phi2 - (1.0 + phi.cos()) / (2.0 * phi * phi.sin());
    na::Matrix3::identity() - 0.5 * hat_t + c * hat2
}

// ── Newton iteration ──────────────────────────────────────────────────────────

/// Per-body rotational residual — the discrete Euler-Lagrange equation for
/// the SO(3) variational integrator:
///
///   R_i = (1/h) T(θ_{k+1})ᵀ [ T⁻ᵀ(h ω_k) I ω_k
///                             − R(h ω_{k+1}) T⁻ᵀ(h ω_{k+1}) I ω_{k+1} ]
///
/// `inertia` is the density-scaled body-frame inertia tensor (I = ρ · I_body).
fn residual_omega_i(
    theta_next: na::Vector3<f32>,
    omega_k:    na::Vector3<f32>,
    omega_next: na::Vector3<f32>,
    inertia:    &na::Matrix3<f32>,
    h:          f32,
) -> na::Vector3<f32> {
    let prev = t_matrix_inv_transpose(h * omega_k) * inertia * omega_k;
    let next = rotation_matrix(h * omega_next)
        * t_matrix_inv_transpose(h * omega_next)
        * inertia
        * omega_next;
    (1.0 / h) * t_matrix_transpose(theta_next) * (prev - next)
}

/// Approximate Jacobian ∂R_i/∂ω_{k+1} used in the Newton solve:
///
///   J_i ≈ −(1/h) T(θ_{k+1})ᵀ R(h ω_{k+1}) T⁻ᵀ(h ω_{k+1}) I
fn approx_jacobian_omega_i(
    theta_next: na::Vector3<f32>,
    omega_next: na::Vector3<f32>,
    inertia:    &na::Matrix3<f32>,
    h:          f32,
) -> na::Matrix3<f32> {
    -(1.0 / h)
        * t_matrix_transpose(theta_next)
        * rotation_matrix(h * omega_next)
        * t_matrix_inv_transpose(h * omega_next)
        * inertia
}
