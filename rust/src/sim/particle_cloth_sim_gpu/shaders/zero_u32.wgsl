// Zero a u32 buffer. Used between substeps for cell_count and dq.

struct Meta { n: u32 };

@group(0) @binding(0) var<uniform>             mta: Meta;
@group(0) @binding(1) var<storage, read_write> buf:  array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= mta.n) { return; }
  buf[i] = 0u;
}
