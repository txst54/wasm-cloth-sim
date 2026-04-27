//! Native CLI for headless simulation debugging.

use my_webgpu_app::arch::native::{run_cloth_headless, run_paper_headless};
use my_webgpu_app::params::SimParams;

fn load_params(path: Option<&str>) -> SimParams {
    match path {
        Some(p) => {
            let contents = std::fs::read_to_string(p)
                .unwrap_or_else(|e| panic!("Failed to read settings file '{}': {}", p, e));
            serde_json::from_str(&contents)
                .unwrap_or_else(|e| panic!("Failed to parse settings file '{}': {}", p, e))
        }
        None => SimParams::default(),
    }
}

fn print_usage() {
    eprintln!("Usage: native <command> [options]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  cloth [options]              Run cloth simulation");
    eprintln!("  paper <cp_file> [options]    Run paper simulation with crease pattern");
    eprintln!("  dump-settings [file]         Dump default settings to JSON file (or stdout)");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --steps <n>        Number of simulation steps (default: 1000)");
    eprintln!("  --resolution <n>   Grid resolution for cloth (default: 32)");
    eprintln!("  --settings <file>  Load simulation parameters from JSON file");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  native cloth --steps 500 --resolution 16");
    eprintln!("  native cloth --settings params.json");
    eprintln!("  native paper crane.cp --steps 2000");
    eprintln!("  native dump-settings params.json");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        "cloth" => {
            let mut steps = 1000usize;
            let mut resolution = 32usize;
            let mut settings_file: Option<&str> = None;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--steps" => {
                        i += 1;
                        steps = args.get(i).and_then(|s| s.parse().ok())
                            .expect("--steps requires a number");
                    }
                    "--resolution" => {
                        i += 1;
                        resolution = args.get(i).and_then(|s| s.parse().ok())
                            .expect("--resolution requires a number");
                    }
                    "--settings" => {
                        i += 1;
                        settings_file = args.get(i).map(|s| s.as_str());
                        if settings_file.is_none() {
                            eprintln!("--settings requires a file path");
                            std::process::exit(1);
                        }
                    }
                    other => {
                        eprintln!("Unknown option: {}", other);
                        print_usage();
                        std::process::exit(1);
                    }
                }
                i += 1;
            }

            let params = load_params(settings_file);
            run_cloth_headless(steps, resolution, &params);
        }

        "paper" => {
            let cp_file = args.get(2).expect("Missing crease pattern file");
            let mut steps = 1000usize;
            let mut settings_file: Option<&str> = None;

            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--steps" => {
                        i += 1;
                        steps = args.get(i).and_then(|s| s.parse().ok())
                            .expect("--steps requires a number");
                    }
                    "--settings" => {
                        i += 1;
                        settings_file = args.get(i).map(|s| s.as_str());
                        if settings_file.is_none() {
                            eprintln!("--settings requires a file path");
                            std::process::exit(1);
                        }
                    }
                    other => {
                        eprintln!("Unknown option: {}", other);
                        print_usage();
                        std::process::exit(1);
                    }
                }
                i += 1;
            }

            let cp_data = std::fs::read_to_string(cp_file)
                .expect("Failed to read crease pattern file");
            let params = load_params(settings_file);
            run_paper_headless(&cp_data, steps, &params);
        }

        "dump-settings" => {
            let params = SimParams::default();
            let json = serde_json::to_string_pretty(&params)
                .expect("Failed to serialize settings");

            match args.get(2) {
                Some(path) => {
                    std::fs::write(path, &json)
                        .unwrap_or_else(|e| panic!("Failed to write '{}': {}", path, e));
                    eprintln!("Default settings written to: {}", path);
                }
                None => {
                    println!("{}", json);
                }
            }
        }

        "--help" | "-h" | "help" => {
            print_usage();
        }

        other => {
            eprintln!("Unknown command: {}", other);
            print_usage();
            std::process::exit(1);
        }
    }
}
