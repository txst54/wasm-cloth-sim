use std::rc::Rc;
use nalgebra as na;

use super::template::RigidBodyTemplate;

/// One simulated rigid body: state (position, orientation, velocities) + a
/// reference to its shared shape template.
pub struct RigidBodyInstance {
    /// Center-of-mass position in world space.
    pub c: na::Vector3<f32>,
    /// Rotation encoded as an axis-angle vector.
    pub theta: na::Vector3<f32>,
    /// Linear velocity of the center of mass.
    pub cvel: na::Vector3<f32>,
    /// Angular velocity in world space (radians / second).
    pub w: na::Vector3<f32>,
    pub density: f32,
    /// State at the start of the previous step — used for CCD against cloth.
    pub prev_c:     na::Vector3<f32>,
    pub prev_theta: na::Vector3<f32>,
    template: Rc<RigidBodyTemplate>,
}

impl RigidBodyInstance {
    pub fn new(
        template: Rc<RigidBodyTemplate>,
        c: na::Vector3<f32>,
        theta: na::Vector3<f32>,
        cvel: na::Vector3<f32>,
        w: na::Vector3<f32>,
        density: f32,
    ) -> Self {
        Self { c, theta, cvel, w, density, prev_c: c, prev_theta: theta, template }
    }

    pub fn get_template(&self) -> &RigidBodyTemplate {
        &self.template
    }

    pub fn mass(&self) -> f32 {
        self.template.mass(self.density)
    }

    /// Rotation matrix mapping body frame → world frame, derived from `theta`.
    pub fn rotation(&self) -> na::Matrix3<f32> {
        let angle = self.theta.norm();
        if angle < 1e-8 {
            return na::Matrix3::identity();
        }
        let axis = na::Unit::new_normalize(self.theta);
        *na::Rotation3::from_axis_angle(&axis, angle).matrix()
    }

    /// Inertia tensor in world frame: R * I_body * R^T, scaled by density.
    pub fn inertia_world(&self) -> na::Matrix3<f32> {
        let r = self.rotation();
        let i_body = self.template.inertia_body_scaled(self.density);
        r * i_body * r.transpose()
    }

    pub fn inertia_world_inv(&self) -> na::Matrix3<f32> {
        self.inertia_world()
            .try_inverse()
            .unwrap_or(na::Matrix3::zeros())
    }

    /// Transform a point from body frame to world frame.
    pub fn body_to_world(&self, p_body: na::Vector3<f32>) -> na::Vector3<f32> {
        self.rotation() * p_body + self.c
    }

    /// Velocity of a world-space point fixed to this body:
    ///   v = cvel + w × (p_world - c)
    pub fn velocity_at(&self, p_world: na::Vector3<f32>) -> na::Vector3<f32> {
        self.cvel + self.w.cross(&(p_world - self.c))
    }

    /// Current world-space positions of all template vertices.
    pub fn world_vertices(&self) -> Vec<na::Vector3<f32>> {
        let r = self.rotation();
        self.template.vertices.iter().map(|&v| r * v + self.c).collect()
    }

    /// World-space vertex positions at the start of the previous step (for CCD).
    pub fn prev_world_vertices(&self) -> Vec<na::Vector3<f32>> {
        let angle = self.prev_theta.norm();
        let r = if angle < 1e-8 {
            na::Matrix3::identity()
        } else {
            let axis = na::Unit::new_normalize(self.prev_theta);
            *na::Rotation3::from_axis_angle(&axis, angle).matrix()
        };
        self.template.vertices.iter().map(|&v| r * v + self.prev_c).collect()
    }
}
