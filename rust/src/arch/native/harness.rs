//! Native debugging harness - headless simulation.

use std::collections::HashSet;

use super::platform::init_platform;
use crate::sim::{ClothSimCore, PaperSim, CreasePattern, Positions};
use crate::params::SimParams;
use crate::platform_log;
use crate::platform_context::PlatformContext;

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
        PlatformContext::set_step(step);
        sim.step(params, &HashSet::new());

        if step % 100 == 0 {
            let avg = compute_average_position(&sim.q);
            platform_log!("avg position = ({:.4}, {:.4}, {:.4})",
                         avg[0], avg[1], avg[2]);
        }
    }

    let elapsed = start.elapsed();
    platform_log!("Completed {} steps in {:?} ({:.2} steps/sec)",
                  steps, elapsed, steps as f64 / elapsed.as_secs_f64());
}

/// Run paper simulation from a .cp file (headless).
/// `target_angle`: fold angle in degrees (0-90). Simulation warms up at 0 degrees
/// for `params.warmup_steps`, then applies the target angle.
pub fn run_paper_headless(cp_data: &str, steps: usize, target_angle: f32, params: &SimParams) {
    init_platform();

    let resolution = params.resolution as usize;
    let warmup_steps = params.warmup_steps as usize;
    let cp = CreasePattern::parse(cp_data).expect("Failed to parse crease pattern");
    let (mut sim, positions, faces, _, _) = PaperSim::from_crease_pattern(&cp, resolution);

    platform_log!("Loaded crease pattern: {} vertices, {} faces (resolution: {})",
                  positions.len(), faces.len(), resolution);
    platform_log!("Warmup: {} steps at 0°, then {} steps at {}°",
                  warmup_steps, steps, target_angle);

    let start = std::time::Instant::now();

    // Warmup phase: run at target_angle = 0
    sim.set_fold_angle(0.0);
    for step in 0..warmup_steps {
        PlatformContext::set_step(step);
        sim.step(params);

        if step % 100 == 0 {
            let avg = compute_average_position(&sim.q);
            platform_log!("Warmup: avg position = ({:.4}, {:.4}, {:.4})",
                          avg[0], avg[1], avg[2]);
        }
    }

    if warmup_steps > 0 {
        let warmup_elapsed = start.elapsed();
        platform_log!("Warmup complete in {:?}", warmup_elapsed);
    }

    // Main phase: apply target angle
    sim.set_fold_angle(target_angle);
    let main_start = std::time::Instant::now();

    for step in 0..steps {
        PlatformContext::set_step(warmup_steps + step);
        sim.step(params);

        // if step % 100 == 0 {
        //     let avg = compute_average_position(&sim.q);
        //     platform_log!("avg position = ({:.4}, {:.4}, {:.4})",
        //                   avg[0], avg[1], avg[2]);
        // }
    }

    let main_elapsed = main_start.elapsed();
    let total_elapsed = start.elapsed();
    platform_log!("Completed {} steps in {:?} ({:.2} steps/sec)",
                  steps, main_elapsed, steps as f64 / main_elapsed.as_secs_f64());
    platform_log!("Total time (including warmup): {:?}", total_elapsed);
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
