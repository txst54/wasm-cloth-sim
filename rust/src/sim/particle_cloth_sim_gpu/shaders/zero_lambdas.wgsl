// Zero a 1D float buffer. Used to reset stretch_lambda and bend_lambda each substep.

struct Meta { n: u32 };

@group(0) @binding(0) var<uniform>             mta: Meta;
@group(0) @binding(1) var<storage, read_write> buf:  array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= mta.n) { return; }
  buf[i] = 0.0;
}
