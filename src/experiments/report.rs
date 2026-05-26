use std::{collections::BTreeMap, fs, path::Path, process::Command};

use super::schema::ExperimentResult;

pub fn write_experiment_reports(results: &[ExperimentResult]) -> std::io::Result<()> {
    write_experiment_reports_to_dir(results, "outputs/experiments")
}

pub fn write_named_experiment_reports(
    results: &[ExperimentResult],
    name: &str,
) -> std::io::Result<()> {
    write_named_experiment_reports_to_dir(results, "outputs/experiments", name)
}

pub fn write_experiment_reports_to_dir<P: AsRef<Path>>(
    results: &[ExperimentResult],
    dir: P,
) -> std::io::Result<()> {
    write_named_experiment_reports_to_dir(results, dir, "denoise_experiments")
}

pub fn write_named_experiment_reports_to_dir<P: AsRef<Path>>(
    results: &[ExperimentResult],
    dir: P,
    name: &str,
) -> std::io::Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    fs::write(
        dir.join(format!("{name}.json")),
        serde_json::to_string_pretty(results).expect("serialize experiment json"),
    )?;
    fs::write(dir.join(format!("{name}.csv")), csv_report(results))?;
    fs::write(dir.join(format!("{name}.md")), markdown_report(results))?;
    Ok(())
}

fn rust_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn opt_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn opt_f64(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.3}")).unwrap_or_default()
}

fn csv_report(results: &[ExperimentResult]) -> String {
    let headers = [
        "case",
        "group",
        "backend",
        "update_mode",
        "n",
        "hidden",
        "image_h",
        "image_w",
        "kernel_h",
        "kernel_w",
        "total_iters",
        "num_steps",
        "num_iters_per_step",
        "scale",
        "range_mode",
        "run_mode",
        "status",
        "primary_constraints",
        "secondary_constraints",
        "primary_variables",
        "secondary_variables",
        "proof_size_bytes",
        "recursive_prove_ms",
        "compressed_prove_ms",
        "compressed_verify_ms",
        "witness_gen_ms",
        "setup_ms",
        "error",
    ];
    let mut out = String::new();
    out.push_str(&headers.join(","));
    out.push('\n');
    for r in results {
        let fields = vec![
            r.case.clone(),
            r.group.clone(),
            r.backend.clone(),
            r.update_mode.clone(),
            r.n.to_string(),
            opt_usize(r.hidden),
            opt_usize(r.image_h),
            opt_usize(r.image_w),
            opt_usize(r.kernel_h),
            opt_usize(r.kernel_w),
            r.total_iters.to_string(),
            r.num_steps.to_string(),
            r.num_iters_per_step.to_string(),
            r.scale.to_string(),
            r.range_mode.clone(),
            r.run_mode.clone(),
            r.status.clone(),
            opt_usize(r.primary_constraints),
            opt_usize(r.secondary_constraints),
            opt_usize(r.primary_variables),
            opt_usize(r.secondary_variables),
            opt_usize(r.proof_size_bytes),
            opt_f64(r.recursive_prove_ms),
            opt_f64(r.compressed_prove_ms),
            opt_f64(r.compressed_verify_ms),
            opt_f64(r.witness_gen_ms),
            opt_f64(r.setup_ms),
            r.error.clone().unwrap_or_default(),
        ];
        out.push_str(
            &fields
                .iter()
                .map(|field| csv_escape(field))
                .collect::<Vec<_>>()
                .join(","),
        );
        out.push('\n');
    }
    out
}

fn markdown_report(results: &[ExperimentResult]) -> String {
    let mut by_group: BTreeMap<&str, Vec<&ExperimentResult>> = BTreeMap::new();
    for result in results {
        by_group.entry(&result.group).or_default().push(result);
    }

    let mut out = String::new();
    out.push_str("# Denoise zkML Experiments\n\n");
    out.push_str("## Environment\n\n");
    out.push_str(&format!("- rust: {}\n", rust_version()));
    out.push_str("- scale: 16\n");
    out.push_str("- curve/backend: Nova local test setup\n");
    out.push_str("- official cases: N >= 16\n\n");

    let order = [
        ("correctness", "Experiment 1: End-to-end correctness"),
        ("range", "Experiment 2: Range check optimization"),
        ("fused_update", "Experiment 3: DoubleFloor vs FusedFloor"),
        ("ablation", "Experiment 3b: Range check and fused update ablation"),
        ("mlp_scaling", "Experiment 4: MLP scaling"),
        ("conv_scaling", "Experiment 5: Conv scaling"),
        ("backend_compare", "Experiment 6: MLP vs Conv"),
        ("steps_scaling", "Experiment 7: Recursive step scaling"),
        ("large_scale", "Experiment 8: Large dimension cases"),
    ];

    for (key, title) in order {
        let Some(rows) = by_group.get(key) else {
            continue;
        };
        out.push_str(&format!("## {title}\n\n"));
        out.push_str("| case | backend | N | structure | iters | mode | range | run | constraints | variables | recursive ms | compressed ms | verify ms | proof bytes | status |\n");
        out.push_str("|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|\n");
        for r in rows {
            let structure = match (r.hidden, r.image_h, r.image_w, r.kernel_h, r.kernel_w) {
                (Some(h), _, _, _, _) => format!("H={h}"),
                (_, Some(ih), Some(iw), Some(kh), Some(kw)) => {
                    format!("{ih}x{iw}, K={kh}x{kw}")
                }
                _ => "-".to_string(),
            };
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                r.case,
                r.backend,
                r.n,
                structure,
                r.total_iters,
                r.update_mode,
                r.range_mode,
                r.run_mode,
                opt_usize(r.primary_constraints),
                opt_usize(r.primary_variables),
                opt_f64(r.recursive_prove_ms),
                opt_f64(r.compressed_prove_ms),
                opt_f64(r.compressed_verify_ms),
                opt_usize(r.proof_size_bytes),
                r.status,
            ));
        }
        out.push('\n');
    }

    out.push_str("## Notes\n\n");
    out.push_str("- No affine experiments are included.\n");
    out.push_str("- All official cases use N >= 16.\n");
    out.push_str("- FusedFloor has different fixed-point semantics from DoubleFloor.\n");
    out.push_str("- BuildOnly cases are used for large-scale circuit-size analysis.\n");
    out.push_str("- Commitment modules remain in the codebase but are not reported as a separate overhead experiment here.\n");
    out
}
