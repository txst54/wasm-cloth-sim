// Rigid SDF contact. One thread per particle, each obstacle queried in turn.
// Each thread only writes its own q[i] → no race.
//
// Supported SDF kinds (must match the encoding below):
//   0 = sphere   { radius = a.x }
//   1 = plane    { n = a.xyz (unit), d = b.x }
//   2 = box      { half_ext = a.xyz }
//   3 = capsule  { half_h = a.x, radius = a.y }
//
// Each obstacle has a body→world rotation `rot` (3x vec4<f32>) and a center.

struct Params {
  alpha:        f32,
  mu:           f32,
  n:            u32,
  num_obs:      u32,
};

struct Obstacle {
  center:  vec4<f32>,         // .xyz used
  rot0:    vec4<f32>,         // rotation rows
  rot1:    vec4<f32>,
  rot2:    vec4<f32>,
  a:       vec4<f32>,         // shape params A
  b:       vec4<f32>,         // shape params B
  kind:    u32,
  _p0:     u32,
  _p1:     u32,
  _p2:     u32,
};

@group(0) @binding(0) var<uniform>             params:    Params;
@group(0) @binding(1) var<storage, read_write> q:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       q_pred:    array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       w_inv:     array<f32>;
@group(0) @binding(4) var<storage, read>       radius:    array<f32>;
@group(0) @binding(5) var<storage, read>       obstacles: array<Obstacle>;

fn rot_to_body(o: Obstacle, v: vec3<f32>) -> vec3<f32> {
  // Body = rotᵀ * v. rot rows are world-space axes; columns are body axes.
  return vec3<f32>(
    o.rot0.x * v.x + o.rot1.x * v.y + o.rot2.x * v.z,
    o.rot0.y * v.x + o.rot1.y * v.y + o.rot2.y * v.z,
    o.rot0.z * v.x + o.rot1.z * v.y + o.rot2.z * v.z,
  );
}

fn rot_to_world(o: Obstacle, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    o.rot0.x * v.x + o.rot0.y * v.y + o.rot0.z * v.z,
    o.rot1.x * v.x + o.rot1.y * v.y + o.rot1.z * v.z,
    o.rot2.x * v.x + o.rot2.y * v.y + o.rot2.z * v.z,
  );
}

struct DN { d: f32, n: vec3<f32> };

fn sdf_sample(o: Obstacle, p: vec3<f32>) -> DN {
  if (o.kind == 0u) {
    let len = length(p);
    if (len < 1e-8) { return DN(-o.a.x, vec3<f32>(0.0, 1.0, 0.0)); }
    return DN(len - o.a.x, p / len);
  }
  if (o.kind == 1u) {
    let nrm = normalize(o.a.xyz);
    return DN(dot(p, nrm) - o.b.x, nrm);
  }
  if (o.kind == 2u) {
    let qv = abs(p) - o.a.xyz;
    let outside = max(qv, vec3<f32>(0.0));
    let outside_len = length(outside);
    let inside_d = min(max(qv.x, max(qv.y, qv.z)), 0.0);
    let dist = outside_len + inside_d;
    var nrm: vec3<f32>;
    if (outside_len > 1e-8) {
      let signed_v = vec3<f32>(
        outside.x * sign(p.x),
        outside.y * sign(p.y),
        outside.z * sign(p.z),
      );
      nrm = normalize(signed_v);
    } else {
      var nx = 0.0; var ny = 0.0; var nz = 0.0;
      if (qv.x >= qv.y && qv.x >= qv.z)      { nx = sign(p.x); }
      else if (qv.y >= qv.x && qv.y >= qv.z) { ny = sign(p.y); }
      else                                    { nz = sign(p.z); }
      nrm = vec3<f32>(nx, ny, nz);
    }
    return DN(dist, nrm);
  }
  // capsule (kind == 3)
  let py = clamp(p.y, -o.a.x, o.a.x);
  let closest = vec3<f32>(0.0, py, 0.0);
  let d = p - closest;
  let len = length(d);
  if (len < 1e-8) { return DN(-o.a.y, vec3<f32>(0.0, 1.0, 0.0)); }
  return DN(len - o.a.y, d / len);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let wi = w_inv[i];
  if (wi == 0.0) { return; }

  let ri = radius[i];
  for (var oi: u32 = 0u; oi < params.num_obs; oi = oi + 1u) {
    let o = obstacles[oi];
    let p_world = q[i].xyz;
    let p_body  = rot_to_body(o, p_world - o.center.xyz);
    let s = sdf_sample(o, p_body);
    let phi = s.d - ri;
    if (phi >= 0.0) { continue; }
    let denom = wi + params.alpha;
    if (denom < 1e-12) { continue; }
    let dlam = -phi / denom;
    let n_world = rot_to_world(o, s.n);
    var pos = p_world + wi * dlam * n_world;

    if (params.mu > 0.0) {
      let pp = q_pred[i].xyz;
      let da = pos - pp;
      let dn = dot(da, n_world);
      let dt = da - dn * n_world;
      let tlen = length(dt);
      if (tlen > 1e-8) {
        var dlf = tlen / wi;
        let cap = params.mu * dlam;
        if (dlf > cap) { dlf = cap; }
        let th = dt / tlen;
        pos = pos - wi * dlf * th;
      }
    }
    q[i] = vec4<f32>(pos, 0.0);
  }
}
