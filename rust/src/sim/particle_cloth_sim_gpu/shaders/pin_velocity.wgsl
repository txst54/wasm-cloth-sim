// Combined pin + velocity update.
//
// Pin: pinned (w_inv == 0) particles are snapped back to q_prev.
//      A single mouse-dragged vertex `mouse_idx` (sentinel u32::MAX = none) is
//      snapped to `mouse_pos` if `pin_enabled != 0`.
// Velocity: v ← (q - q_prev) / h.

struct Params {
  inv_h:       f32,
  pin_enabled: u32,
  mouse_idx:   u32,
  n:           u32,
  mouse_pos:   vec4<f32>,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> q:      array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       q_prev: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> v:      array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       w_inv:  array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }

  if (params.pin_enabled != 0u && i == params.mouse_idx) {
    q[i] = vec4<f32>(params.mouse_pos.xyz, 0.0);
  }
  let wi = w_inv[i];
  if (wi == 0.0) {
    q[i] = q_prev[i];
    v[i] = vec4<f32>(0.0);
    return;
  }
  let dq = q[i].xyz - q_prev[i].xyz;
  v[i] = vec4<f32>(dq * params.inv_h, 0.0);
}
