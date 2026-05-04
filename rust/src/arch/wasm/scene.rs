//! Scene constants and procedural mesh builders shared across WASM entry
//! points: light position, palette, cube/sphere/ground meshes.

use std::collections::HashMap;

use super::cloth::Cloth;
use super::gpu::GpuContext;
use super::light::Light;

// ── Light ─────────────────────────────────────────────────────────────────────

/// Default key-light world-space position used by every WASM scene.
pub const LIGHT_POS: [f32; 3] = [2.0, 0.0, 0.5];

pub fn make_light(ctx: &GpuContext) -> Light {
    Light::new(ctx, LIGHT_POS)
}

// ── Palette ───────────────────────────────────────────────────────────────────

pub mod color {
    pub const WOOD:   [f32; 3] = [0.72, 0.53, 0.30];
    pub const SPHERE: [f32; 3] = [0.85, 0.85, 0.85];
    pub const GROUND: [f32; 3] = [0.50, 0.50, 0.50];
}

// ── Cube ──────────────────────────────────────────────────────────────────────

/// Half-side length used by the demo cubes (0.25 × 0.25 × 0.25 cube).
const CUBE_HALF: f32 = 0.125;

/// Vertices of a unit-axis-aligned cube centered at the origin (8 verts).
pub const CUBE_VERTS: [[f32; 3]; 8] = {
    let h = CUBE_HALF;
    [
        [-h, -h, -h], [ h, -h, -h], [ h,  h, -h], [-h,  h, -h], // -Z face
        [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h], // +Z face
    ]
};

/// 12 outward-CCW triangles for the cube above.
pub const CUBE_FACES: [[u32; 3]; 12] = [
    [4, 5, 6], [4, 6, 7], // +Z
    [1, 0, 3], [1, 3, 2], // -Z
    [0, 4, 7], [0, 7, 3], // -X
    [5, 1, 2], [5, 2, 6], // +X
    [7, 6, 2], [7, 2, 3], // +Y
    [0, 1, 5], [0, 5, 4], // -Y
];

/// Build a render-only cloth mesh for a wood-coloured cube.
pub fn cube_cloth(ctx: &GpuContext, light: &Light) -> Cloth {
    Cloth::from_mesh(
        ctx,
        CUBE_VERTS.to_vec(),
        CUBE_FACES.to_vec(),
        vec![color::SPHERE; CUBE_VERTS.len()],
        HashMap::new(),
        light,
    )
}

// ── Sphere (subdivided octahedron) ────────────────────────────────────────────

/// Build an approximate sphere by subdividing a unit octahedron 3 times
/// (8 · 4³ = 512 faces) and projecting each new vertex onto the unit sphere.
/// The result is then scaled by `radius` and translated to `center`.
pub fn octa_sphere_mesh(center: [f32; 3], radius: f32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut verts: Vec<[f32; 3]> = vec![
        [ 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0,  1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0,  1.0], [0.0, 0.0, -1.0],
    ];
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ];

    for _ in 0..3 {
        let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midpoint = |a: u32, b: u32, verts: &mut Vec<[f32; 3]>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&i) = cache.get(&key) { return i; }
            let pa = verts[a as usize]; let pb = verts[b as usize];
            let mx = (pa[0] + pb[0]) * 0.5;
            let my = (pa[1] + pb[1]) * 0.5;
            let mz = (pa[2] + pb[2]) * 0.5;
            let inv = 1.0 / (mx*mx + my*my + mz*mz).sqrt();
            verts.push([mx * inv, my * inv, mz * inv]);
            let idx = (verts.len() - 1) as u32;
            cache.insert(key, idx);
            idx
        };
        for &[a, b, c] in &faces {
            let ab = midpoint(a, b, &mut verts);
            let bc = midpoint(b, c, &mut verts);
            let ca = midpoint(c, a, &mut verts);
            new_faces.push([a, ab, ca]);
            new_faces.push([b, bc, ab]);
            new_faces.push([c, ca, bc]);
            new_faces.push([ab, bc, ca]);
        }
        faces = new_faces;
    }

    let world_verts = verts.iter().map(|v| [
        center[0] + v[0] * radius,
        center[1] + v[1] * radius,
        center[2] + v[2] * radius,
    ]).collect();
    (world_verts, faces)
}

pub fn sphere_cloth(ctx: &GpuContext, light: &Light, center: [f32; 3], radius: f32) -> Cloth {
    let (verts, faces) = octa_sphere_mesh(center, radius);
    let colors = vec![color::SPHERE; faces.len()];
    Cloth::from_mesh(ctx, verts, faces, colors, HashMap::new(), light)
}

// ── Ground plane ──────────────────────────────────────────────────────────────

/// Large axis-aligned ground quad at `y`, `2*half_extent` wide.
pub fn ground_cloth(ctx: &GpuContext, light: &Light, y: f32, half_extent: f32) -> Cloth {
    let s = half_extent;
    let verts = vec![
        [-s, y, -s], [ s, y, -s],
        [ s, y,  s], [-s, y,  s],
    ];
    let faces = vec![[0u32, 2, 1], [0, 3, 2]];
    let colors = vec![color::SPHERE; 2];
    Cloth::from_mesh(ctx, verts, faces, colors, HashMap::new(), light)
}
