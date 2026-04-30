use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::gpu::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    position:  [f32; 3],
    _pad:      f32,
}

pub struct Camera {
    pub yaw:   f32,
    pub pitch: f32,
    pub dist:   f32,
    pub aspect: f32,
    pub target: [f32; 3],
    pub eye: [f32; 3],
    pub inv_view_proj: [[f32; 4]; 4],
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group:        wgpu::BindGroup,
    uniform_buffer:        wgpu::Buffer,
}

impl Camera {
    pub fn new(ctx: &GpuContext) -> Self {
        let aspect = ctx.config.width as f32 / ctx.config.height as f32;
        let yaw   = 0.3f32;
        let pitch = 0.3f32;
        let dist  = 2.0f32;
        let target = [0.0, 0.0, 0.0];

        let built = Self::build(yaw, pitch, dist, target, aspect);

        let uniform_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera UB"),
            contents: bytemuck::bytes_of(&built.uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            },
        );

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BG"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            yaw, pitch, dist, aspect, target,
            eye: built.eye,
            inv_view_proj: built.inv_view_proj,
            bind_group_layout, bind_group, uniform_buffer,
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue) {
        let built = Self::build(self.yaw, self.pitch, self.dist, self.target, self.aspect);
        self.eye           = built.eye;
        self.inv_view_proj = built.inv_view_proj;
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&built.uniform));
    }

    pub fn forward(&self) -> [f32; 3] {
        normalize3([
            -self.pitch.cos() * self.yaw.sin(),
            -self.pitch.sin(),
            -self.pitch.cos() * self.yaw.cos(),
        ])
    }

    pub fn right(&self) -> [f32; 3] {
        normalize3(cross3(self.forward(), [0.0, 1.0, 0.0]))
    }

    fn build(yaw: f32, pitch: f32, dist: f32, target: [f32; 3], aspect: f32) -> Built {
        let eye = [
            target[0] + pitch.cos() * yaw.sin() * dist,
            target[1] + pitch.sin() * dist,
            target[2] + pitch.cos() * yaw.cos() * dist,
        ];

        let view_cm = look_at_col(eye, target, [0.0, 1.0, 0.0]);
        let proj_cm = perspective_col(std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0);
        let view_proj_cm = mat4_mul_col(proj_cm, view_cm);

        let inv_view_rm = inv_look_at_row(eye, target, [0.0, 1.0, 0.0]);
        let inv_proj_rm = inv_perspective_row(std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0);
        let inv_vp_rm   = mat4_mul_row(inv_view_rm, inv_proj_rm);

        Built {
            uniform: CameraUniform { view_proj: view_proj_cm, position: eye, _pad: 0.0 },
            eye,
            inv_view_proj: inv_vp_rm,
        }
    }
}

struct Built {
    uniform:       CameraUniform,
    eye:           [f32; 3],
    inv_view_proj: [[f32; 4]; 4],
}

fn look_at_col(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3(sub3(center, eye));
    let r = normalize3(cross3(f, up));
    let u = cross3(r, f);
    transpose4([
        [ r[0],  r[1],  r[2], -dot3(r, eye)],
        [ u[0],  u[1],  u[2], -dot3(u, eye)],
        [-f[0], -f[1], -f[2],  dot3(f, eye)],
        [ 0.0,   0.0,   0.0,   1.0         ],
    ])
}

fn perspective_col(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y * 0.5).tan();
    transpose4([
        [f / aspect, 0.0, 0.0,                        0.0                      ],
        [0.0,        f,   0.0,                        0.0                      ],
        [0.0,        0.0, far / (near - far),         near * far / (near - far)],
        [0.0,        0.0, -1.0,                       0.0                      ],
    ])
}

fn mat4_mul_col(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut c = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            for k in 0..4 {
                c[col][row] += a[k][row] * b[col][k];
            }
        }
    }
    c
}

fn transpose4(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

fn inv_look_at_row(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3(sub3(center, eye));
    let r = normalize3(cross3(f, up));
    let u = cross3(r, f);
    [
        [ r[0],  u[0], -f[0], eye[0]],
        [ r[1],  u[1], -f[1], eye[1]],
        [ r[2],  u[2], -f[2], eye[2]],
        [ 0.0,   0.0,   0.0,  1.0   ],
    ]
}

fn inv_perspective_row(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let inv_b = (near - far) / (near * far);
    let a_over_b = 1.0 / near;
    [
        [aspect / f, 0.0,   0.0,       0.0      ],
        [0.0,        1.0/f, 0.0,       0.0      ],
        [0.0,        0.0,   0.0,      -1.0      ],
        [0.0,        0.0,   inv_b,     a_over_b ],
    ]
}

pub fn mat4_mul_row(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut c = [[0.0f32; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            for k in 0..4 {
                c[row][col] += a[row][k] * b[k][col];
            }
        }
    }
    c
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if len > 1e-8 { [v[0]/len, v[1]/len, v[2]/len] } else { [0.0, 0.0, 1.0] }
}
