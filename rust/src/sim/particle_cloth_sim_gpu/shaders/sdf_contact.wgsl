// Rigid SDF contact. One thread per particle, each obstacle queried in turn.
// Each thread only writes its own q[i] → no race.
//
// Supported SDF kinds (must match the encoding below):
//   0 = sphere   { radius = a.x }
//   1 = plane    { n = a.xyz (unit), d = b.x }
//   2 = box      { half_ext = a.xyz }
//   3 = capsule  { half_h = a.x, radius = a.y }
//   4 = mesh     { bounds_min = a.xyz, bounds_max = b.xyz }  → samples mesh_sdf
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
@group(0) @binding(6) var                      mesh_sdf:     texture_3d<f32>;

fn rot_to_body(o: Obstacle, v: vec3<f32>) -> vec3<f32> {
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

// Manual trilinear texture lookup (R32Float is non-filterable on the
// default WebGPU feature set, so we do the eight-corner blend ourselves).
fn fetch_sdf(uvw: vec3<f32>) -> f32 {
  let dims_u = textureDimensions(mesh_sdf);
  let dims = vec3<f32>(dims_u);
  let coord = clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0)) * (dims - vec3<f32>(1.0));
  let i0 = vec3<i32>(floor(coord));
  let i1 = i0 + vec3<i32>(1);
  let max_i = vec3<i32>(dims_u) - vec3<i32>(1);
  let i0c = clamp(i0, vec3<i32>(0), max_i);
  let i1c = clamp(i1, vec3<i32>(0), max_i);
  let f = coord - vec3<f32>(i0);
  let c000 = textureLoad(mesh_sdf, vec3<i32>(i0c.x, i0c.y, i0c.z), 0).r;
  let c100 = textureLoad(mesh_sdf, vec3<i32>(i1c.x, i0c.y, i0c.z), 0).r;
  let c010 = textureLoad(mesh_sdf, vec3<i32>(i0c.x, i1c.y, i0c.z), 0).r;
  let c110 = textureLoad(mesh_sdf, vec3<i32>(i1c.x, i1c.y, i0c.z), 0).r;
  let c001 = textureLoad(mesh_sdf, vec3<i32>(i0c.x, i0c.y, i1c.z), 0).r;
  let c101 = textureLoad(mesh_sdf, vec3<i32>(i1c.x, i0c.y, i1c.z), 0).r;
  let c011 = textureLoad(mesh_sdf, vec3<i32>(i0c.x, i1c.y, i1c.z), 0).r;
  let c111 = textureLoad(mesh_sdf, vec3<i32>(i1c.x, i1c.y, i1c.z), 0).r;
  let cx00 = mix(c000, c100, f.x);
  let cx10 = mix(c010, c110, f.x);
  let cx01 = mix(c001, c101, f.x);
  let cx11 = mix(c011, c111, f.x);
  let cxy0 = mix(cx00, cx10, f.y);
  let cxy1 = mix(cx01, cx11, f.y);
  return mix(cxy0, cxy1, f.z);
}

fn sample_mesh(o: Obstacle, p: vec3<f32>) -> DN {
  let bmin = o.a.xyz;
  let bmax = o.b.xyz;
  let extent = max(bmax - bmin, vec3<f32>(1e-6));

  // Always go through the texture. For points outside the baked volume we
  // sample at the clamped uvw (the nearest bbox-boundary texel) and add
  // the distance from p to that boundary point — i.e. dist_to_mesh ≈
  // boundary_sdf + ||p - clamp(p)||. This smoothly extends the SDF past
  // the bbox without turning the bbox itself into a solid wall, which is
  // what the previous early-out did.
  let cp_bbox    = clamp(p, bmin, bmax);
  let outside    = p - cp_bbox;
  let outside_len = length(outside);
  let uvw = (cp_bbox - bmin) / extent;

  let dims     = vec3<f32>(textureDimensions(mesh_sdf));
  let step_uvw = vec3<f32>(1.0) / dims;

  let d_tex = fetch_sdf(uvw);
  let d     = d_tex + outside_len;

  // Per-uvw central differences, scaled by 1/extent to convert into a
  // proper world-space gradient (otherwise anisotropic bboxes bias the
  // normal toward the longest axis).
  let dx = fetch_sdf(uvw + vec3<f32>(step_uvw.x, 0.0, 0.0))
         - fetch_sdf(uvw - vec3<f32>(step_uvw.x, 0.0, 0.0));
  let dy = fetch_sdf(uvw + vec3<f32>(0.0, step_uvw.y, 0.0))
         - fetch_sdf(uvw - vec3<f32>(0.0, step_uvw.y, 0.0));
  let dz = fetch_sdf(uvw + vec3<f32>(0.0, 0.0, step_uvw.z))
         - fetch_sdf(uvw - vec3<f32>(0.0, 0.0, step_uvw.z));
  let g_tex = vec3<f32>(dx / extent.x, dy / extent.y, dz / extent.z);

  // Outside the bbox the distance term is dominated by the away-from-bbox
  // displacement, so the gradient direction is just `outside`.
  var g: vec3<f32>;
  if (outside_len > 1e-6) { g = outside; } else { g = g_tex; }
  let glen = length(g);
  var nrm = vec3<f32>(0.0, 1.0, 0.0);
  if (glen > 1e-6) { nrm = g / glen; }
  return DN(d, nrm);
}

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
  if (o.kind == 3u) {
    let py = clamp(p.y, -o.a.x, o.a.x);
    let closest = vec3<f32>(0.0, py, 0.0);
    let d = p - closest;
    let len = length(d);
    if (len < 1e-8) { return DN(-o.a.y, vec3<f32>(0.0, 1.0, 0.0)); }
    return DN(len - o.a.y, d / len);
  }
  return sample_mesh(o, p);
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
