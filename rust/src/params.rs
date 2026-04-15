pub struct SimParams {
    pub time_step:         f64,
    pub constraint_iters:  u32,

    pub gravity_enabled:   bool,
    pub gravity_g:         f64,

    pub pin_enabled:       bool,
    pub pin_weight:        f64,

    pub stretch_enabled:   bool,
    pub stretch_weight:    f64,

    pub bending_enabled:   bool,
    pub bending_weight:    f64,

    pub pulling_enabled:   bool,
    pub pulling_weight:    f64,
    pub pulling_area:     u32,

    pub self_collision_enabled:   bool,
    /// Distance (world units) below which a vertex is considered in contact with a triangle.
    /// Default ≈ 1.5× the rest edge length for a 64-resolution cloth on [−1, 1].
    pub self_collision_threshold: f64,
    /// When true, collision pairs are rebuilt every constraint iteration (more accurate,
    /// catches collisions formed during projection, but slower).
    /// When false, pairs are built once from predicted positions before the constraint loop.
    pub self_collision_recompute_pairs: bool,
    /// When true, stretch and bending use per-edge distance constraints (fast, O(E)).
    /// When false, they use shape-matching with SVD polar decomposition (slower, higher quality).
    pub use_distance_constraints: bool,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            time_step:        1e-2,
            constraint_iters: 5,

            gravity_enabled:  true,
            gravity_g:        -9.8,

            pin_enabled:      true,
            pin_weight:       1.0,

            stretch_enabled:  true,
            stretch_weight:   0.5,

            bending_enabled:  true,
            bending_weight:   0.5,

            pulling_enabled:  true,
            pulling_weight:   0.1,
            pulling_area:     5,

            self_collision_enabled:   true,
            self_collision_threshold: 0.02,
            self_collision_recompute_pairs: false,
            use_distance_constraints: false,
        }
    }
}
