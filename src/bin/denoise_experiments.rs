use denoise::experiments::{
    ablation, backend_compare, conv_scaling, correctness, fused_update, large_scale, mlp_scaling,
    range, scaling_sweep, steps_scaling, write_named_experiment_reports_to_dir, ExperimentResult,
};

struct ExperimentGroup {
    name: &'static str,
    title: &'static str,
    run: fn() -> Vec<ExperimentResult>,
}

fn groups() -> Vec<ExperimentGroup> {
    vec![
        ExperimentGroup {
            name: correctness::NAME,
            title: correctness::TITLE,
            run: correctness::run,
        },
        ExperimentGroup {
            name: range::NAME,
            title: range::TITLE,
            run: range::run,
        },
        ExperimentGroup {
            name: fused_update::NAME,
            title: fused_update::TITLE,
            run: fused_update::run,
        },
        ExperimentGroup {
            name: ablation::NAME,
            title: ablation::TITLE,
            run: ablation::run,
        },
        ExperimentGroup {
            name: mlp_scaling::NAME,
            title: mlp_scaling::TITLE,
            run: mlp_scaling::run,
        },
        ExperimentGroup {
            name: conv_scaling::NAME,
            title: conv_scaling::TITLE,
            run: conv_scaling::run,
        },
        ExperimentGroup {
            name: backend_compare::NAME,
            title: backend_compare::TITLE,
            run: backend_compare::run,
        },
        ExperimentGroup {
            name: scaling_sweep::NAME,
            title: scaling_sweep::TITLE,
            run: scaling_sweep::run,
        },
        ExperimentGroup {
            name: steps_scaling::NAME,
            title: steps_scaling::TITLE,
            run: steps_scaling::run,
        },
        ExperimentGroup {
            name: large_scale::NAME,
            title: large_scale::TITLE,
            run: large_scale::run,
        },
    ]
}

fn usage() {
    eprintln!("Usage:");
    eprintln!("  cargo run --release --bin denoise_experiments -- all");
    eprintln!("  cargo run --release --bin denoise_experiments -- <group>");
    eprintln!();
    eprintln!("Groups:");
    for group in groups() {
        eprintln!("  {:<16} {}", group.name, group.title);
    }
}

fn write_group(name: &str, results: &[ExperimentResult]) -> std::io::Result<()> {
    let dir = format!("outputs/experiments/{name}");
    let stem = format!("denoise_experiments_{name}");
    write_named_experiment_reports_to_dir(results, dir, &stem)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let selected = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let groups = groups();

    if selected == "--help" || selected == "-h" {
        usage();
        return Ok(());
    }

    if selected == "all" {
        let mut all_results = Vec::new();
        for group in groups {
            println!("running {} ({})", group.name, group.title);
            let results = (group.run)();
            write_group(group.name, &results)?;
            all_results.extend(results);
        }
        write_named_experiment_reports_to_dir(
            &all_results,
            "outputs/experiments/all",
            "denoise_experiments_all",
        )?;
        println!("wrote outputs/experiments/all/denoise_experiments_all.md");
        println!("wrote per-group reports under outputs/experiments/<group>/");
        return Ok(());
    }

    if let Some(group) = groups.into_iter().find(|group| group.name == selected) {
        println!("running {} ({})", group.name, group.title);
        let results = (group.run)();
        write_group(group.name, &results)?;
        println!(
            "wrote outputs/experiments/{}/denoise_experiments_{}.md",
            group.name, group.name
        );
        return Ok(());
    }

    eprintln!("unknown experiment group: {selected}");
    usage();
    std::process::exit(2);
}
