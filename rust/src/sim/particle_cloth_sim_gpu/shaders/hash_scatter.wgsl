// Build pass 3: each particle atomically claims its slot in the per-cell run
// and writes itself into particle_id[slot].

struct Meta { n: u32 };

@group(0) @binding(0) var<uniform>             mta:          Meta;
@group(0) @binding(1) var<storage, read>       particle_cell: array<u32>;
@group(0) @binding(2) var<storage, read_write> cell_cursor:   array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> particle_id:   array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= mta.n) { return; }
  let key = particle_cell[i];
  let slot = atomicAdd(&cell_cursor[key], 1u);
  particle_id[slot] = i;
}
