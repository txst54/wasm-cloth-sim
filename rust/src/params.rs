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
            pulling_weight:   0.5,
        }
    }
}
