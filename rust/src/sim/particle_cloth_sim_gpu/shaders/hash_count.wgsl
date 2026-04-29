// Build pass 1 of the uniform spatial hash:
//   - clear cell_count (separate dispatch zeroes it via zero_u32 pipeline),
//   - for each particle, hash its cell key, atomicAdd cell_count[key],
//   - record the particle's cell key into particle_cell so pass 3 can scatter
//     without recomputing.
//
// The hash function matches the CPU `ParticleHash` for portable behavior.

struct HashParams {
  inv_h:      f32,
  table_size: u32,
  n:          u32,
  _pad:       u32,
};

@group(0) @binding(0) var<uniform>             hp:           HashParams;
@group(0) @binding(1) var<storage, read>       q:            array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cell_count:   array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> particle_cell: array<u32>;

fn hash_key(ix: i32, iy: i32, iz: i32, ts: u32) -> u32 {
  // Match CPU: i64 xor of multiplied components, mod table_size.
  // 32-bit wrapping XOR is the same up to high-bit truncation; we then mod.
  let a = u32(ix) * 73856093u;
  let b = u32(iy) * 19349663u;
  let c = u32(iz) * 83492791u;
  let h = a ^ b ^ c;
  return h % ts;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= hp.n) { return; }
  let p = q[i].xyz;
  let ix = i32(floor(p.x * hp.inv_h));
  let iy = i32(floor(p.y * hp.inv_h));
  let iz = i32(floor(p.z * hp.inv_h));
  let key = hash_key(ix, iy, iz, hp.table_size);
  particle_cell[i] = key;
  atomicAdd(&cell_count[key], 1u);
}
