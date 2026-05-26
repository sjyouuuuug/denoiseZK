use super::schema::NovaStepView;

pub fn reshape_flat_to_matrix(flat: &[i64], height: usize, width: usize) -> Vec<Vec<i64>> {
    assert_eq!(flat.len(), height * width, "flat length must equal H*W");
    (0..height)
        .map(|row| {
            (0..width)
                .map(|col| flat[row * width + col])
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn build_nova_step_views(num_steps: usize, num_iters_per_step: usize) -> Vec<NovaStepView> {
    (0..num_steps)
        .map(|nova_step_index| {
            let iter_start = nova_step_index * num_iters_per_step;
            let iter_end = iter_start + num_iters_per_step - 1;
            NovaStepView {
                nova_step_index,
                iter_start,
                iter_end,
                label: format!(
                    "Nova step {nova_step_index}: x_{iter_start} -> x_{}",
                    iter_end + 1
                ),
            }
        })
        .collect()
}
