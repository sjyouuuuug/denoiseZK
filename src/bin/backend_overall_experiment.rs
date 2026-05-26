use denoise::experiments::{
    backend_overall, write_named_experiment_reports_to_dir, ExperimentResult,
};

fn print_summary(results: &[ExperimentResult]) {
    println!(
        "{:<28} {:<5} {:>5} {:<10} {:<10} {:>12} {:>12} {:>10}",
        "case", "be", "N", "run", "status", "constraints", "variables", "proof"
    );
    for r in results {
        println!(
            "{:<28} {:<5} {:>5} {:<10} {:<10} {:>12} {:>12} {:>10}",
            r.case,
            r.backend,
            r.n,
            r.run_mode,
            r.status,
            r.primary_constraints
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            r.primary_variables
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            r.proof_size_bytes
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "running {} ({})",
        backend_overall::NAME,
        backend_overall::TITLE
    );
    println!("config: scale=16, range=Bits, update=FusedFloor, total_iters=4, nova_steps=2x2");

    let results = backend_overall::run();
    write_named_experiment_reports_to_dir(
        &results,
        "outputs/experiments/backend_overall",
        "backend_overall_experiment",
    )?;

    print_summary(&results);
    println!("wrote outputs/experiments/backend_overall/backend_overall_experiment.md");
    println!("wrote outputs/experiments/backend_overall/backend_overall_experiment.json");
    println!("wrote outputs/experiments/backend_overall/backend_overall_experiment.csv");
    Ok(())
}
