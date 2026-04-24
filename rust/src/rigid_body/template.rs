use std::rc::Rc;
use nalgebra as na;

/// Precomputed shape data for a rigid body mesh. Shared across instances.
///
/// All vertex positions are stored in body frame (COM at origin).
/// Inertia is computed for unit density; scale by `density` per instance.
pub struct RigidBodyTemplate {
    /// Vertex positions in body frame (COM is at origin).
    pub vertices: Vec<na::Vector3<f32>>,
    pub faces: Vec<[u32; 3]>,
    /// Signed volume (positive when faces are outward-CCW).
    pub volume: f32,
    /// Inertia tensor at COM in body frame, for unit density.
    /// Multiply by actual density to get the real I_body.
    pub inertia_body: na::Matrix3<f32>,
}

impl RigidBodyTemplate {
    /// Build a template from a watertight triangle mesh.
    /// Faces must be oriented outward (CCW winding when viewed from outside).
    pub fn new(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Rc<Self> {
        let verts: Vec<na::Vector3<f32>> = vertices
            .iter()
            .map(|&[x, y, z]| na::Vector3::new(x, y, z))
            .collect();

        let (volume, com) = compute_volume_and_com(&verts, faces);

        let verts_body: Vec<na::Vector3<f32>> = verts.iter().map(|v| v - com).collect();

        let inertia_body = compute_inertia_body(&verts_body, faces);

        Rc::new(Self {
            vertices: verts_body,
            faces: faces.to_vec(),
            volume,
            inertia_body,
        })
    }

    pub fn mass(&self, density: f32) -> f32 {
        density * self.volume
    }

    pub fn inertia_body_scaled(&self, density: f32) -> na::Matrix3<f32> {
        density * self.inertia_body
    }
}

/// Returns (volume, center_of_mass) for a closed mesh using signed tetrahedra
/// (divergence theorem). Apex of each tetrahedron is the origin.
fn compute_volume_and_com(
    verts: &[na::Vector3<f32>],
    faces: &[[u32; 3]],
) -> (f32, na::Vector3<f32>) {
    let mut vol_sum = 0.0f32;
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut cz = 0.0f32;

    for &[i0, i1, i2] in faces {
        let a = verts[i0 as usize];
        let b = verts[i1 as usize];
        let c = verts[i2 as usize];
        // d = a · (b × c) = 6 × signed tetrahedral volume
        let d = a.dot(&b.cross(&c));
        vol_sum += d;
        // Centroid of tet (origin, a, b, c) = (0 + a + b + c) / 4
        cx += d * (a.x + b.x + c.x);
        cy += d * (a.y + b.y + c.y);
        cz += d * (a.z + b.z + c.z);
    }

    let volume = (vol_sum / 6.0).abs();
    let com = if vol_sum.abs() > 1e-12 {
        na::Vector3::new(cx, cy, cz) / (4.0 * vol_sum)
    } else {
        na::Vector3::zeros()
    };

    (volume, com)
}

/// Computes the inertia tensor at the origin (which should be the COM) for unit
/// density, using the Mirtich 1996 polyhedral formulas.
fn compute_inertia_body(
    verts: &[na::Vector3<f32>], // already in COM frame
    faces: &[[u32; 3]],
) -> na::Matrix3<f32> {
    let mut ix2 = 0.0f32;
    let mut iy2 = 0.0f32;
    let mut iz2 = 0.0f32;
    let mut ixy = 0.0f32;
    let mut ixz = 0.0f32;
    let mut iyz = 0.0f32;

    for &[i0, i1, i2] in faces {
        let a = verts[i0 as usize];
        let b = verts[i1 as usize];
        let c = verts[i2 as usize];
        let d = a.dot(&b.cross(&c));

        // Second-moment integrals (Mirtich 1996, Table 1).
        ix2 += d * (a.x * a.x + a.x * b.x + b.x * b.x + a.x * c.x + b.x * c.x + c.x * c.x);
        iy2 += d * (a.y * a.y + a.y * b.y + b.y * b.y + a.y * c.y + b.y * c.y + c.y * c.y);
        iz2 += d * (a.z * a.z + a.z * b.z + b.z * b.z + a.z * c.z + b.z * c.z + c.z * c.z);

        ixy += d
            * (2.0 * a.x * a.y
                + a.x * b.y
                + a.x * c.y
                + b.x * a.y
                + 2.0 * b.x * b.y
                + b.x * c.y
                + c.x * a.y
                + c.x * b.y
                + 2.0 * c.x * c.y);
        ixz += d
            * (2.0 * a.x * a.z
                + a.x * b.z
                + a.x * c.z
                + b.x * a.z
                + 2.0 * b.x * b.z
                + b.x * c.z
                + c.x * a.z
                + c.x * b.z
                + 2.0 * c.x * c.z);
        iyz += d
            * (2.0 * a.y * a.z
                + a.y * b.z
                + a.y * c.z
                + b.y * a.z
                + 2.0 * b.y * b.z
                + b.y * c.z
                + c.y * a.z
                + c.y * b.z
                + 2.0 * c.y * c.z);
    }

    ix2 /= 60.0;
    iy2 /= 60.0;
    iz2 /= 60.0;
    ixy /= 120.0;
    ixz /= 120.0;
    iyz /= 120.0;

    // Standard inertia tensor: diagonal = sum of OTHER two second moments,
    // off-diagonal = negative products of inertia.
    na::Matrix3::new(
        iy2 + iz2, -ixy,      -ixz,
        -ixy,       ix2 + iz2, -iyz,
        -ixz,      -iyz,        ix2 + iy2,
    )
}
