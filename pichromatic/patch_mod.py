import re

with open("src/film/development/mod.rs", "r") as f:
    content = f.read()

old = """    if resolves_particles {
        apply_particle_grain_overwrite(
            &mut dyes,
            &d_max,
            &kappas,
            &gammas,
            pixel_pitch_um,
            seed,
            &crystal_sizes,
        );
    } else {"""

new = """    if resolves_particles {
        apply_particle_grain_overwrite(
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

if old in content:
    content = content.replace(old, new)
    with open("src/film/development/mod.rs", "w") as f:
        f.write(content)
else:
    print("Could not find old if block")
