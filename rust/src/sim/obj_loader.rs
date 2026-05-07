//! Minimal OBJ parser. Reads `v` (positions) and `f` (faces, triangulated as
//! a fan). Ignores `vn`, `vt`, smoothing groups, materials, comments.

use nalgebra as na;

pub struct ObjMesh {
    pub positions: Vec<[f32; 3]>,
    pub faces:     Vec<[u32; 3]>,
}

impl ObjMesh {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut faces:     Vec<[u32; 3]> = Vec::new();

        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let mut it = line.split_whitespace();
            let tag = match it.next() { Some(t) => t, None => continue };
            match tag {
                "v" => {
                    let parse = |s: Option<&str>| -> Result<f32, String> {
                        s.ok_or_else(|| format!("line {}: vertex missing component", line_no + 1))?
                            .parse::<f32>().map_err(|e| format!("line {}: {}", line_no + 1, e))
                    };
                    let x = parse(it.next())?;
                    let y = parse(it.next())?;
                    let z = parse(it.next())?;
                    positions.push([x, y, z]);
                }
                "f" => {
                    let toks: Vec<&str> = it.collect();
                    if toks.len() < 3 { continue; }
                    let parse_idx = |t: &str| -> Result<u32, String> {
                        let s = t.split('/').next().unwrap_or(t);
                        let i: i64 = s.parse().map_err(|e| format!("line {}: {}", line_no + 1, e))?;
                        let n = positions.len() as i64;
                        let r = if i > 0 { i - 1 } else { n + i };
                        if r < 0 || r >= n {
                            Err(format!("line {}: face index out of range", line_no + 1))
                        } else { Ok(r as u32) }
                    };
                    let i0 = parse_idx(toks[0])?;
                    for k in 1..(toks.len() - 1) {
                        let i1 = parse_idx(toks[k])?;
                        let i2 = parse_idx(toks[k + 1])?;
                        faces.push([i0, i1, i2]);
                    }
                }
                _ => {}
            }
        }

        if positions.is_empty() { return Err("OBJ has no vertices".into()); }
        if faces.is_empty()     { return Err("OBJ has no faces".into()); }
        Ok(Self { positions, faces })
    }

    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for k in 0..3 {
                if p[k] < lo[k] { lo[k] = p[k]; }
                if p[k] > hi[k] { hi[k] = p[k]; }
            }
        }
        (lo, hi)
    }

    /// Translates so the bbox center is at the origin and scales so the
    /// largest half-extent is 1. Returns `(offset, scale)` describing the
    /// transform applied: `p_new = (p_old - offset) * scale`.
    pub fn recenter_unit(&mut self) -> (na::Vector3<f32>, f32) {
        let (lo, hi) = self.bounds();
        let center = na::Vector3::new(
            0.5 * (lo[0] + hi[0]),
            0.5 * (lo[1] + hi[1]),
            0.5 * (lo[2] + hi[2]),
        );
        let half = [
            0.5 * (hi[0] - lo[0]),
            0.5 * (hi[1] - lo[1]),
            0.5 * (hi[2] - lo[2]),
        ];
        let max_half = half[0].max(half[1]).max(half[2]).max(1e-6);
        let scale = 1.0 / max_half;
        for p in self.positions.iter_mut() {
            p[0] = (p[0] - center.x) * scale;
            p[1] = (p[1] - center.y) * scale;
            p[2] = (p[2] - center.z) * scale;
        }
        (center, scale)
    }
}
