// Exclusive prefix sum of cell_count → cell_start. Single-workgroup serial
// scan; correct up to ~ unlimited table size but slow for huge tables. For a
// cloth with ~1k particles, table_size ≈ next_prime(2*N) is small, so this
// is fine.
//
// cell_start has length table_size + 1; cell_start[table_size] is the total.
// We also produce cell_cursor (clone of cell_start) for the scatter pass to
// use as a write cursor.

struct Meta { table_size: u32 };

@group(0) @binding(0) var<uniform>             mta:        Meta;
@group(0) @binding(1) var<storage, read>       cell_count:  array<u32>;
@group(0) @binding(2) var<storage, read_write> cell_start:  array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_cursor: array<u32>;

@compute @workgroup_size(1)
fn main() {
  var acc: u32 = 0u;
  let ts = mta.table_size;
  for (var k: u32 = 0u; k < ts; k = k + 1u) {
    cell_start[k]  = acc;
    cell_cursor[k] = acc;
    acc = acc + cell_count[k];
  }
  cell_start[ts]  = acc;
  cell_cursor[ts] = acc;
}
