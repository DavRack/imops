import re

with open("src/film/development/mod.rs", "r") as f:
    content = f.read()

content = content.replace(
"""        apply_particle_grain_overwrite(
            &mut dyes,
            &d_max,
            &kappas,
            &gammas,
            pixel_pitch_um,
            seed,
            &crystal_sizes, 0.0, &[], 0.0, 0.0);
    } else {""",
"""        apply_particle_grain_overwrite(
            &mut dyes,
            &d_max,
            &kappas,
            &gammas,
            pixel_pitch_um,
            seed,
            &crystal_sizes,
            sigma_dir_px,
            &stock.dir_inhibition_matrix,
            sigma_px,
            stock.adjacency_beta,
        );
    } else {"""
)

with open("src/film/development/mod.rs", "w") as f:
    f.write(content)
