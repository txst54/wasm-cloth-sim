// Self-collision (particle-vs-particle) — Jacobi pass.
//
// One thread per particle i. We iterate the 27 hash cells around q[i], visit
// every candidate j, and *accumulate* position corrections into a global delta
// buffer `dq` via atomic fixed-point integers (WGSL has no atomic<f32>). A
// follow-up `contact_apply` pass converts `dq` back to floats and adds them
// to `q`, then zeroes `dq` for the next substep.
//
// Conventions:
//   - Only pairs with j > i contribute. Both i and j receive their share of
//     the correction; the symmetric thread (the j-thread) skips this pair, so
//     each pair is solved exactly once.
//   - one_ring_csr_row[i]..one_ring_csr_row[i+1] indexes the topological 1-ring
//     of i; those j are skipped (structural neighbors must not contact).
//   - d_coll = min(r_i + r_j, |q_rest_j - q_rest_i|), matching the CPU code.
//
// Friction (Macklin §3.5): tangential delta capped by μ * Δλ_n. Added to the
// same dq accumulator.

struct Params {
  h:       f32,
  alpha:   f32,
  mu:      f32,
  n:       u32,
  inv_h:   f32,    // hash inv_cell_size (cell = 2 * r_max)
  ts:      u32,    // hash table_size
  scale:   f32,    // fixed-point scale, e.g. 2^20
  _pad:    u32,
};

@group(0) @binding(0) var<uniform>             params:        Params;
@group(0) @binding(1) var<storage, read>       q:             array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       q_pred:        array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       q_rest:        array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       w_inv:         array<f32>;
@group(0) @binding(5) var<storage, read>       radius:        array<f32>;

@group(1) @binding(0) var<storage, read>       cell_start:    array<u32>;
@group(1) @binding(1) var<storage, read>       particle_id:   array<u32>;
@group(1) @binding(2) var<storage, read>       one_ring_row:  array<u32>;
@group(1) @binding(3) var<storage, read>       one_ring_col:  array<u32>;

@group(2) @binding(0) var<storage, read_write> dq:            array<atomic<i32>>;

fn hash_key(ix: i32, iy: i32, iz: i32, ts: u32) -> u32 {
  let a = u32(ix) * 73856093u;
  let b = u32(iy) * 19349663u;
  let c = u32(iz) * 83492791u;
  return (a ^ b ^ c) % ts;
}

fn add_dq(i: u32, c: u32, v: f32) {
  // Saturate to i32 range to avoid wrap on extreme values.
  let scaled = clamp(v * params.scale, -2.14e9, 2.14e9);
  atomicAdd(&dq[i * 3u + c], i32(scaled));
}

fn in_one_ring(i: u32, j: u32) -> bool {
  let s = one_ring_row[i];
  let e = one_ring_row[i + 1u];
  for (var k: u32 = s; k < e; k = k + 1u) {
    if (one_ring_col[k] == j) { return true; }
  }
  return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }

  let pi   = q[i].xyz;
  let ri   = radius[i];
  let wi   = w_inv[i];
  let pri  = q_rest[i].xyz;
  let ppi  = q_pred[i].xyz;

  let ix = i32(floor(pi.x * params.inv_h));
  let iy = i32(floor(pi.y * params.inv_h));
  let iz = i32(floor(pi.z * params.inv_h));

  for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let key = hash_key(ix + dx, iy + dy, iz + dz, params.ts);
        let s = cell_start[key];
        let e = cell_start[key + 1u];
        for (var k: u32 = s; k < e; k = k + 1u) {
          let j = particle_id[k];
          if (j <= i) { continue; }
          if (in_one_ring(i, j)) { continue; }

          let pj  = q[j].xyz;
          let d   = pj - pi;
          let r_sum = ri + radius[j];
          let prj  = q_rest[j].xyz;
          let d_rest = length(prj - pri);
          let d_coll = min(r_sum, d_rest);
          let d2 = dot(d, d);
          if (d2 >= d_coll * d_coll || d2 < 1e-12) { continue; }
          let dist = sqrt(d2);
          let c = dist - d_coll;       // < 0
          let wj = w_inv[j];
          let denom = wi + wj + params.alpha;
          if (denom < 1e-12) { continue; }
          let dlam = -c / denom;       // > 0
          let n = d / dist;

          // Normal correction.
          let di_n = -wi * dlam * n;
          let dj_n =  wj * dlam * n;
          if (wi > 0.0) {
            add_dq(i, 0u, di_n.x);
            add_dq(i, 1u, di_n.y);
            add_dq(i, 2u, di_n.z);
          }
          if (wj > 0.0) {
            add_dq(j, 0u, dj_n.x);
            add_dq(j, 1u, dj_n.y);
            add_dq(j, 2u, dj_n.z);
          }

          // Tangential friction.
          if (params.mu > 0.0) {
            let ppj = q_pred[j].xyz;
            let da  = (q[i].xyz - ppi) - (q[j].xyz - ppj);
            let dn  = dot(da, n);
            let dt  = da - dn * n;
            let tlen = length(dt);
            if (tlen > 1e-8) {
              let denom_f = wi + wj;
              if (denom_f > 1e-12) {
                var dlf = tlen / denom_f;
                let cap = params.mu * dlam;
                if (dlf > cap) { dlf = cap; }
                let th = dt / tlen;
                let di_f = -wi * dlf * th;
                let dj_f =  wj * dlf * th;
                if (wi > 0.0) {
                  add_dq(i, 0u, di_f.x);
                  add_dq(i, 1u, di_f.y);
                  add_dq(i, 2u, di_f.z);
                }
                if (wj > 0.0) {
                  add_dq(j, 0u, dj_f.x);
                  add_dq(j, 1u, dj_f.y);
                  add_dq(j, 2u, dj_f.z);
                }
              }
            }
          }
        }
      }
    }
  }
}
