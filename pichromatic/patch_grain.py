import re

with open("src/film/development/grain.rs", "r") as f:
    content = f.read()

old = """pub fn apply_particle_grain_overwrite(
    dyes: &mut DyePlanes,
    d_max_per_layer: &[f32],
    kappa_ref_per_layer: &[f32],
    gamma_contrast_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
    crystal_sizes: &[Option<crate::film::stock::LogNormalDist>],
) {
    for (layer_i, plane) in dyes.image_dye.iter_mut().enumerate() {
        let d_max = d_max_per_layer[layer_i];
        let kappa_ref = kappa_ref_per_layer[layer_i];
        let gamma = gamma_contrast_per_layer[layer_i].max(1e-6);
        if kappa_ref <= 0.0 || d_max <= 0.0 {
            continue;
        }
        let probabilities: Vec<f32> = plane
            .iter()
            .map(|&density| {
                let p_hd = (density / d_max).clamp(0.0, 1.0);
                p_hd.powf(gamma)
            })
            .collect();
        let particles = particle_field(
            &probabilities,
            dyes.width,
            dyes.height,
            kappa_ref,
            pixel_pitch_um,
            crystal_sizes.get(layer_i).and_then(|value| value.as_ref()),
            seed,
            layer_i,
            true,
        );
        plane
            .par_iter_mut()
            .zip(particles.par_iter())
            .for_each(|(density, &fraction)| {
                *density = (fraction * d_max).clamp(0.0, d_max * 1.05);
            });
    }
}"""

new = """pub fn apply_particle_grain_overwrite(
    dyes: &mut DyePlanes,
    d_max_per_layer: &[f32],
    kappa_ref_per_layer: &[f32],
    gamma_contrast_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
    crystal_sizes: &[Option<crate::film::stock::LogNormalDist>],
    sigma_dir_px: f32,
    dir_inhibition_matrix: &[Vec<f32>],
    sigma_px: f32,
    adjacency_beta: f32,
) {
    let mut f_dev = vec![vec![0.0f32; dyes.width * dyes.height]; dyes.image_dye.len()];
    for (layer_i, plane) in dyes.image_dye.iter().enumerate() {
        let d_max = d_max_per_layer[layer_i];
        let kappa_ref = kappa_ref_per_layer[layer_i];
        let gamma = gamma_contrast_per_layer[layer_i].max(1e-6);
        if kappa_ref <= 0.0 || d_max <= 0.0 {
            continue;
        }
        let probabilities: Vec<f32> = plane
            .iter()
            .map(|&density| {
                let p_hd = (density / d_max).clamp(0.0, 1.0);
                p_hd.powf(gamma)
            })
            .collect();
        let particles = particle_field(
            &probabilities,
            dyes.width,
            dyes.height,
            kappa_ref,
            pixel_pitch_um,
            crystal_sizes.get(layer_i).and_then(|value| value.as_ref()),
            seed,
            layer_i,
            true,
        );
        f_dev[layer_i] = particles;
    }

    dyes.image_dye = f_dev;
    crate::film::development::diffusion::apply_dir_inhibition(dyes, sigma_dir_px, dir_inhibition_matrix);
    crate::film::development::diffusion::apply_adjacency(dyes, sigma_px, adjacency_beta);

    for (layer_i, plane) in dyes.image_dye.iter_mut().enumerate() {
        let d_max = d_max_per_layer[layer_i];
        plane
            .par_iter_mut()
            .for_each(|fraction| {
                *fraction = (*fraction * d_max).clamp(0.0, d_max * 1.05);
            });
    }
}"""

if old in content:
    content = content.replace(old, new)
    with open("src/film/development/grain.rs", "w") as f:
        f.write(content)
else:
    print("Could not find old function")
