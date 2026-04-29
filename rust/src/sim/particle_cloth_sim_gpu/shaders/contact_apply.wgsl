// Apply accumulated dq → q and zero dq for next substep.

struct Params {
  n:        u32,
  inv_scale: f32,
  _pad0:    u32,
  _pad1:    u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> q:      array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dq:     array<atomic<i32>>;
@group(0) @binding(3) var<storage, read>       w_inv:  array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  if (w_inv[i] == 0.0) { return; }
  let dx = f32(atomicExchange(&dq[i * 3u + 0u], 0)) * params.inv_scale;
  let dy = f32(atomicExchange(&dq[i * 3u + 1u], 0)) * params.inv_scale;
  let dz = f32(atomicExchange(&dq[i * 3u + 2u], 0)) * params.inv_scale;
  let p = q[i].xyz + vec3<f32>(dx, dy, dz);
  q[i] = vec4<f32>(p, 0.0);
}
