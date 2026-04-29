// Predict pass: gravity, damping, integrate v → q. One thread per particle.
//
// Pinned particles (w_inv == 0) are zeroed and not advanced.

struct Params {
  h:        f32,
  g:        f32,
  damping:  f32,
  n:        u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> q:      array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> q_prev: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> v:      array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       w_inv:  array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }

  q_prev[i] = q[i];

  let wi = w_inv[i];
  if (wi == 0.0) {
    v[i] = vec4<f32>(0.0);
    return;
  }

  var vi = v[i].xyz;
  vi.y = vi.y + params.h * params.g;
  let d = max(1.0 - params.damping, 0.0);
  vi = vi * d;
  v[i] = vec4<f32>(vi, 0.0);

  let qi = q[i].xyz + params.h * vi;
  q[i] = vec4<f32>(qi, 0.0);
}
