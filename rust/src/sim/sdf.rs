//! Analytic signed distance fields for rigid obstacles.
//!
//! Each SDF returns `(distance, outward_normal)` for a query point in body
//! frame. The wrapping `SdfObstacle` carries a rigid-body transform so the
//! particle cloth can collide against rotated / translated shapes.

use nalgebra as na;

#[derive(Clone)]
pub enum SignedDistanceField {
    Sphere  { radius:   f32 },
    Plane   { n: na::Vector3<f32>, d: f32 },
    Box     { half_ext: na::Vector3<f32> },
    Capsule { half_h:   f32, radius: f32 },
}

impl SignedDistanceField {
    /// Returns `(signed_distance, outward_unit_normal)` at body-frame point `p`.
    pub fn sample(&self, p: na::Vector3<f32>) -> (f32, na::Vector3<f32>) {
        match *self {
            SignedDistanceField::Sphere { radius } => {
                let len = p.norm();
                if len < 1e-8 {
                    (-radius, na::Vector3::y())
                } else {
                    (len - radius, p / len)
                }
            }
            SignedDistanceField::Plane { n, d } => {
                let nn = n.normalize();
                (p.dot(&nn) - d, nn)
            }
            SignedDistanceField::Box { half_ext } => {
                let q = p.abs() - half_ext;
                let outside = na::Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
                let outside_len = outside.norm();
                let inside_d = q.x.max(q.y.max(q.z)).min(0.0);
                let dist = outside_len + inside_d;
                let n = if outside_len > 1e-8 {
                    // Outside: gradient is the sign-restored direction of `outside`.
                    let signed = na::Vector3::new(
                        outside.x * p.x.signum(),
                        outside.y * p.y.signum(),
                        outside.z * p.z.signum(),
                    );
                    signed.normalize()
                } else {
                    // Inside: pick the axis with the smallest penetration.
                    let (mut nx, mut ny, mut nz) = (0.0, 0.0, 0.0);
                    if q.x >= q.y && q.x >= q.z      { nx = p.x.signum(); }
                    else if q.y >= q.x && q.y >= q.z { ny = p.y.signum(); }
                    else                              { nz = p.z.signum(); }
                    na::Vector3::new(nx, ny, nz)
                };
                (dist, n)
            }
            SignedDistanceField::Capsule { half_h, radius } => {
                // Segment along Y axis from -half_h to +half_h.
                let py = p.y.clamp(-half_h, half_h);
                let closest = na::Vector3::new(0.0, py, 0.0);
                let d = p - closest;
                let len = d.norm();
                if len < 1e-8 {
                    (-radius, na::Vector3::y())
                } else {
                    (len - radius, d / len)
                }
            }
        }
    }
}

/// Kinematic obstacle: a transform plus an SDF in body frame.
#[derive(Clone)]
pub struct SdfObstacle {
    pub center:   na::Vector3<f32>,
    pub rotation: na::Matrix3<f32>,        // body→world
    pub vel:      na::Vector3<f32>,        // for friction reference
    pub sdf:      SignedDistanceField,
}

impl SdfObstacle {
    pub fn sphere(center: na::Vector3<f32>, radius: f32) -> Self {
        Self {
            center, rotation: na::Matrix3::identity(), vel: na::Vector3::zeros(),
            sdf: SignedDistanceField::Sphere { radius },
        }
    }
    pub fn plane(n: na::Vector3<f32>, d: f32) -> Self {
        Self {
            center: na::Vector3::zeros(), rotation: na::Matrix3::identity(),
            vel: na::Vector3::zeros(),
            sdf: SignedDistanceField::Plane { n, d },
        }
    }
    pub fn boxx(center: na::Vector3<f32>, half_ext: na::Vector3<f32>) -> Self {
        Self {
            center, rotation: na::Matrix3::identity(), vel: na::Vector3::zeros(),
            sdf: SignedDistanceField::Box { half_ext },
        }
    }

    /// `(signed_distance, world_outward_normal)` at world-space point `p`.
    pub fn query(&self, p_world: na::Vector3<f32>) -> (f32, na::Vector3<f32>) {
        let p_body = self.rotation.transpose() * (p_world - self.center);
        let (d, n_body) = self.sdf.sample(p_body);
        (d, self.rotation * n_body)
    }
}
