// XPBD dihedral-angle constraint kernel.
//
// One thread per active hinge in the current color class. Within a color
// class no two hinges share any of their 4 vertices, so the four endpoint
// stores are race-free without atomics.
//
// Mirrors `paper_sim::apply_hinge_xpbd` exactly: computes signed dihedral θ
// via atan2(sin, cos), the four ∂θ/∂x_i gradients, then a damped XPBD update
//
//   γ      = compliance · damping / dt
//   denom  = (1+γ) Σ wᵢ |∇θᵢ|² + α̃
//   Δλ     = -(C + α̃ λ + γ ∇C·(x - x^prev)) / denom
//   λ     += Δλ
//   q_i   += w_i · Δλ · ∇θ_i
//
// where C = θ - goal_angle.

struct Params {
  h_dt:        f32,
  n_color:     u32,
  base:        u32,
  _pad:        u32,
};

struct Hinge {
  abcd:  vec4<u32>, // a, b, c, d
};

struct Meta {
  // compliance is already divided by rest_edge_len at construction.
  compliance:   f32,
  damping:      f32,
  goal_angle:   f32,
  _pad:         f32,
};

@group(0) @binding(0) var<uniform>             params:    Params;
@group(0) @binding(1) var<storage, read_write> q:         array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       q_prev:    array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       w_inv:     array<f32>;

@group(1) @binding(0) var<storage, read>       hinges:    array<Hinge>;
@group(1) @binding(1) var<storage, read>       hmeta:     array<Meta>;
@group(1) @binding(2) var<storage, read_write> lambda:    array<f32>;
@group(1) @binding(3) var<storage, read>       color_idx: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= params.n_color) { return; }
  let hi = color_idx[params.base + t];

  let h = hinges[hi];
  let a = h.abcd.x;
  let b = h.abcd.y;
  let c = h.abcd.z;
  let d = h.abcd.w;

  let pa = q[a].xyz;
  let pb = q[b].xyz;
  let pc = q[c].xyz;
  let pd = q[d].xyz;

  let edge = pb - pa;
  let crease_len = length(edge);
  if (crease_len < 1e-8) { return; }
  let e_hat = edge / crease_len;

  // CCW face normals: face1 = (b,a,c), face2 = (a,b,d).
  let n1_raw = cross(pa - pb, pc - pb);
  let n2_raw = cross(pb - pa, pd - pa);
  let n1l = length(n1_raw);
  let n2l = length(n2_raw);
  if (n1l < 1e-8 || n2l < 1e-8) { return; }
  let n1 = n1_raw / n1l;
  let n2 = n2_raw / n2l;

  let cos_t = clamp(dot(n1, n2), -1.0, 1.0);
  let sin_t = dot(cross(n1, n2), e_hat);
  let theta = atan2(sin_t, cos_t);

  let m = hmeta[hi];
  let c_val = theta - m.goal_angle;
  if (abs(c_val) < 1e-5) { return; }

  // Moment arms.
  let v1 = pc - pa;
  let v2 = pd - pa;
  let proj1 = dot(e_hat, v1);
  let proj2 = dot(e_hat, v2);
  let h1_sq = max(dot(v1, v1) - proj1 * proj1, 0.0);
  let h2_sq = max(dot(v2, v2) - proj2 * proj2, 0.0);
  if (h1_sq < 1e-12 || h2_sq < 1e-12) { return; }
  let h1 = sqrt(h1_sq);
  let h2 = sqrt(h2_sq);
  let coef1 = proj1 / crease_len;
  let coef2 = proj2 / crease_len;

  let g_c = n1 / h1;
  let g_d = n2 / h2;
  let g_a = -((1.0 - coef1) / h1) * n1 - ((1.0 - coef2) / h2) * n2;
  let g_b = -(coef1 / h1) * n1 - (coef2 / h2) * n2;

  let wa = w_inv[a]; let wb = w_inv[b];
  let wc = w_inv[c]; let wd = w_inv[d];

  let dx_a = pa - q_prev[a].xyz;
  let dx_b = pb - q_prev[b].xyz;
  let dx_c = pc - q_prev[c].xyz;
  let dx_d = pd - q_prev[d].xyz;
  let grad_dot_dx = dot(g_a, dx_a) + dot(g_b, dx_b) + dot(g_c, dx_c) + dot(g_d, dx_d);

  let weighted = wa * dot(g_a, g_a)
               + wb * dot(g_b, g_b)
               + wc * dot(g_c, g_c)
               + wd * dot(g_d, g_d);

  let dt = params.h_dt;
  let alpha_tilde = m.compliance / (dt * dt);
  let gamma       = m.compliance * m.damping / dt;
  let denom = (1.0 + gamma) * weighted + alpha_tilde;
  if (denom < 1e-12) { return; }

  let lam = lambda[hi];
  let dl  = -(c_val + alpha_tilde * lam + gamma * grad_dot_dx) / denom;
  lambda[hi] = lam + dl;

  if (wa > 0.0) { q[a] = vec4<f32>(pa + wa * dl * g_a, 0.0); }
  if (wb > 0.0) { q[b] = vec4<f32>(pb + wb * dl * g_b, 0.0); }
  if (wc > 0.0) { q[c] = vec4<f32>(pc + wc * dl * g_c, 0.0); }
  if (wd > 0.0) { q[d] = vec4<f32>(pd + wd * dl * g_d, 0.0); }
}
