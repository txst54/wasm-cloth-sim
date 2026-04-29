// Parallel XPBD distance-constraint kernel. One thread per *active* constraint
// in the current color class. Within a color, no two constraints share a
// vertex, so the two endpoint stores are race-free without atomics.
//
// The same pipeline solves both stretch (edges) and bend (diamond diagonals);
// the caller swaps the bind group to point at the appropriate pair / rest /
// lambda / color buffers.

struct Params {
  h:        f32,
  alpha:    f32,    // compliance / h^2 for this constraint family
  n_color:  u32,    // size of this color class (count of constraints to solve)
  base:     u32,    // offset into color_idx where this color class starts
};

@group(0) @binding(0) var<uniform>             params:   Params;
@group(0) @binding(1) var<storage, read_write> q:        array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       w_inv:    array<f32>;

@group(1) @binding(0) var<storage, read>       pairs:    array<vec2<u32>>;
@group(1) @binding(1) var<storage, read>       rest:     array<f32>;
@group(1) @binding(2) var<storage, read_write> lambda:   array<f32>;
@group(1) @binding(3) var<storage, read>       color_idx: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= params.n_color) { return; }
  let ci  = color_idx[params.base + t];
  let p   = pairs[ci];
  let a   = p.x;
  let b   = p.y;

  let pa  = q[a].xyz;
  let pb  = q[b].xyz;
  let wa  = w_inv[a];
  let wb  = w_inv[b];

  let d   = pb - pa;
  let len = length(d);
  if (len < 1e-12) { return; }
  let n   = d / len;

  let c     = len - rest[ci];
  let lam   = lambda[ci];
  let denom = wa + wb + params.alpha;
  if (denom < 1e-12) { return; }
  let dlam  = (-c - params.alpha * lam) / denom;

  lambda[ci] = lam + dlam;

  // Color class is an independent set ⇒ plain stores are race-free.
  if (wa > 0.0) { q[a] = vec4<f32>(pa - wa * dlam * n, 0.0); }
  if (wb > 0.0) { q[b] = vec4<f32>(pb + wb * dlam * n, 0.0); }
}
