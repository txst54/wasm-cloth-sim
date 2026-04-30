#[derive(Debug, Clone)]
#[cfg_attr(feature = "native", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "native", serde(default))]
pub struct SimParams {
    pub time_step:         f64,
    pub constraint_iters:  u32,
    /// Number of XPBD substeps per frame.
    /// Only applied when `use_distance_constraints = true`.
    /// Each substep advances the simulation by `time_step / num_substeps`,
    /// runs `predict → reset λ → constraint_iters → update_velocity`.
    pub num_substeps:      u32,

    pub gravity_enabled:   bool,
    pub gravity_g:         f64,

    pub pin_enabled:       bool,
    pub pin_weight:        f64,

    pub stretch_enabled:   bool,
    pub stretch_weight:    f64,
    /// XPBD compliance α (m²/N) for distance-constraint stretch.
    /// Used only when `use_distance_constraints = true`.
    /// Smaller = stiffer. Typical 1e-8 (rigid) … 1e-4 (soft).
    pub stretch_compliance: f64,

    pub bending_enabled:   bool,
    pub bending_weight:    f64,
    /// XPBD compliance α for distance-constraint bending (diamond diagonals).
    /// Used only when `use_distance_constraints = true`.
    pub bend_compliance:    f64,

    pub pulling_enabled:   bool,
    pub pulling_weight:    f64,
    pub pulling_area:     u32,

    pub self_collision_enabled:   bool,
    /// Distance (world units) below which a vertex is considered in contact with a triangle.
    /// Default ≈ 1.5× the rest edge length for a 64-resolution cloth on [−1, 1].
    pub self_collision_threshold: f64,
    pub self_collision_recompute_iters: u16,
    /// When true, stretch and bending use per-edge distance constraints (fast, O(E)).
    /// When false, they use shape-matching with SVD polar decomposition (slower, higher quality).
    pub use_distance_constraints: bool,
    /// Velocity damping applied after each step: v *= (1 - damping).
    /// 0.0 = no damping (default); 0.05 = 5% reduction per step.
    pub damping: f64,
    /// Coulomb friction on contacts (self + SDF) for particle cloth.
    pub friction_enabled: bool,
    /// Friction coefficient μ. Tangential Lagrange multiplier is clamped to μ·Δλ_n.
    pub friction_mu: f64,
    /// Velocity-averaging damping for cloth-cloth contacts (Stable Cloth-Cloth Friction).
    pub cloth_friction_enabled: bool,
    /// Physical damping coefficient d_physical (1/s). Effective d = clamp(h·d_physical, 0, 1).
    pub cloth_friction_d: f64,
    /// Grid resolution for mesh generation (default 32).
    pub resolution: u32,
    /// Number of warmup steps to run at target_angle=0 before applying the target fold angle.
    pub warmup_steps: u32,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            time_step:        1e-2,
            constraint_iters: 5,
            num_substeps:     10,

            gravity_enabled:  true,
            gravity_g:        -9.8,

            pin_enabled:      true,
            pin_weight:       1.0,

            stretch_enabled:  true,
            stretch_weight:   0.5,
            stretch_compliance: 1e-7,

            bending_enabled:  true,
            bending_weight:   0.5,
            bend_compliance:    1e-6,

            pulling_enabled:  true,
            pulling_weight:   0.1,
            pulling_area:     5,

            self_collision_enabled:   true,
            self_collision_threshold: 0.02,
            self_collision_recompute_iters: 1,
            use_distance_constraints: false,
            damping: 0.0,
            friction_enabled: true,
            friction_mu: 0.3,
            cloth_friction_enabled: true,
            cloth_friction_d: 30.0,
            resolution: 32,
            warmup_steps: 0,
        }
    }
}
