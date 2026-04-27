//! Native debugging harness - headless simulation.

use std::collections::HashSet;

use super::platform::init_platform;
use crate::sim::{ClothSimCore, PaperSim, CreasePattern, Positions};
use crate::params::SimParams;
use crate::platform_log;

/// Run a headless cloth simulation for debugging.
pub fn run_cloth_headless(steps: usize, resolution: usize, params: &SimParams) {
    init_platform();

    platform_log!("Starting headless cloth simulation: {} steps, {}x{} grid",
                  steps, resolution, resolution);

    let positions = create_grid_positions(resolution);
    let faces = create_grid_faces(resolution);

    let mut sim = ClothSimCore::from_mesh(&positions, &faces, &[]);

    let start = std::time::Instant::now();

    for step in 0..steps {
        sim.step(params, &HashSet::new());

        if step % 100 == 0 {
            let avg = compute_average_position(&sim.q);
            platform_log!("Step {}: avg position = ({:.4}, {:.4}, {:.4})",
                          step, avg[0], avg[1], avg[2]);
        }
    }

    let elapsed = start.elapsed();
    platform_log!("Completed {} steps in {:?} ({:.2} steps/sec)",
                  steps, elapsed, steps as f64 / elapsed.as_secs_f64());
}

/// Run paper simulation from a .cp file (headless).
pub fn run_paper_headless(cp_data: &str, steps: usize, params: &SimParams) {
    init_platform();

    let cp = CreasePattern::parse(cp_data).expect("Failed to parse crease pattern");
    let (mut sim, positions, faces, _, _) = PaperSim::from_crease_pattern(&cp);

    platform_log!("Loaded crease pattern: {} vertices, {} faces",
                  positions.len(), faces.len());

    let start = std::time::Instant::now();

    for step in 0..steps {
        sim.step(params);

        if step % 100 == 0 {
            let avg = compute_average_position(&sim.q);
            platform_log!("Step {}: avg position = ({:.4}, {:.4}, {:.4})",
                          step, avg[0], avg[1], avg[2]);
        }
    }

    let elapsed = start.elapsed();
    platform_log!("Completed {} steps in {:?} ({:.2} steps/sec)",
                  steps, elapsed, steps as f64 / elapsed.as_secs_f64());
}

fn create_grid_positions(n: usize) -> Vec<[f32; 3]> {
    let mut positions = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            let x = (col as f32 / (n - 1) as f32) * 2.0 - 1.0;
            let y = (row as f32 / (n - 1) as f32) * 2.0 - 1.0;
            positions.push([x * 0.9, y * 0.9, 0.0]);
        }
    }
    positions
}

fn create_grid_faces(n: usize) -> Vec<[u32; 3]> {
    let mut faces = Vec::new();
    for row in 0..(n - 1) {
        for col in 0..(n - 1) {
            let tl = (row * n + col) as u32;
            let tr = tl + 1;
            let bl = ((row + 1) * n + col) as u32;
            let br = bl + 1;
            faces.push([tl, tr, br]);
            faces.push([tl, br, bl]);
        }
    }
    faces
}

fn compute_average_position(q: &Positions) -> [f32; 3] {
    let n = q.nrows();
    let mut sum = [0.0f32; 3];
    for i in 0..n {
        sum[0] += q[(i, 0)];
        sum[1] += q[(i, 1)];
        sum[2] += q[(i, 2)];
    }
    [sum[0] / n as f32, sum[1] / n as f32, sum[2] / n as f32]
}
