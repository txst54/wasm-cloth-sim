//! Flat-array uniform spatial hash for particles.
//!
//! O(N) build via two linear passes (counting sort). No `HashMap`,
//! no pointer chasing — all data is contiguous `Vec<u32>` for WASM-friendliness.

use super::shared::Positions;

#[inline]
fn hash_cell(ix: i32, iy: i32, iz: i32, table_size: u32) -> u32 {
    let h = (ix.wrapping_mul(73856093) as i64
        ^ iy.wrapping_mul(19349663) as i64
        ^ iz.wrapping_mul(83492791) as i64) as u64;
    (h % table_size as u64) as u32
}

pub struct ParticleHash {
    pub h:           f32,
    inv_h:           f32,
    table_size:      u32,
    /// Length `table_size + 1`. `cell_start[k]..cell_start[k+1]` indexes into `particle_id`.
    pub cell_start:  Vec<u32>,
    /// Particles re-ordered so each cell is a contiguous run.
    pub particle_id: Vec<u32>,
}

impl ParticleHash {
    pub fn new(num_particles: usize, cell_size: f32) -> Self {
        let table_size = next_prime((num_particles * 2).max(31) as u32);
        Self {
            h: cell_size,
            inv_h: 1.0 / cell_size,
            table_size,
            cell_start: vec![0; table_size as usize + 1],
            particle_id: vec![0; num_particles],
        }
    }

    pub fn set_cell_size(&mut self, h: f32) {
        self.h = h;
        self.inv_h = 1.0 / h;
    }

    #[inline]
    fn cell_of(&self, x: f32, y: f32, z: f32) -> (i32, i32, i32) {
        (
            (x * self.inv_h).floor() as i32,
            (y * self.inv_h).floor() as i32,
            (z * self.inv_h).floor() as i32,
        )
    }

    /// O(N) two-pass counting sort.
    pub fn rebuild(&mut self, q: &Positions) {
        let n = q.nrows();
        if self.particle_id.len() != n {
            self.particle_id.resize(n, 0);
        }
        let ts = self.table_size as usize;

        // Pass 1: count
        for c in self.cell_start.iter_mut() { *c = 0; }
        for i in 0..n {
            let (ix, iy, iz) = self.cell_of(q[(i, 0)], q[(i, 1)], q[(i, 2)]);
            let key = hash_cell(ix, iy, iz, self.table_size) as usize;
            self.cell_start[key] += 1;
        }

        // Prefix sum
        let mut acc: u32 = 0;
        for k in 0..=ts {
            let v = self.cell_start[k];
            self.cell_start[k] = acc;
            acc += v;
        }

        // Pass 2: scatter (use cell_start as a write cursor, then fix it back)
        let mut cursor = self.cell_start.clone();
        for i in 0..n {
            let (ix, iy, iz) = self.cell_of(q[(i, 0)], q[(i, 1)], q[(i, 2)]);
            let key = hash_cell(ix, iy, iz, self.table_size) as usize;
            let slot = cursor[key] as usize;
            self.particle_id[slot] = i as u32;
            cursor[key] += 1;
        }
    }

    /// Visit every particle in the 27 cells around `p`. Calls `f(j)` for each.
    /// **NOTE**: cells alias under hashing, so `f` may receive particles outside
    /// the 3³ neighborhood; the caller still does an exact distance test.
    #[inline]
    pub fn for_each_neighbor<F: FnMut(u32)>(&self, p: [f32; 3], mut f: F) {
        let (ix, iy, iz) = self.cell_of(p[0], p[1], p[2]);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let key = hash_cell(ix + dx, iy + dy, iz + dz, self.table_size) as usize;
                    let s = self.cell_start[key] as usize;
                    let e = self.cell_start[key + 1] as usize;
                    for k in s..e {
                        f(self.particle_id[k]);
                    }
                }
            }
        }
    }
}

fn next_prime(mut n: u32) -> u32 {
    if n < 2 { return 2; }
    if n % 2 == 0 { n += 1; }
    loop {
        if is_prime(n) { return n; }
        n += 2;
    }
}

fn is_prime(n: u32) -> bool {
    if n < 2 { return false; }
    if n % 2 == 0 { return n == 2; }
    let mut d = 3u32;
    while d * d <= n {
        if n % d == 0 { return false; }
        d += 2;
    }
    true
}
