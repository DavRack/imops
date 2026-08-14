import re

with open("src/film/development/grain.rs", "r") as f:
    content = f.read()

# For single line
content = re.sub(
    r'apply_particle_grain_overwrite\((.*?)\&crystal_sizes\s*\)',
    r'apply_particle_grain_overwrite(\1&crystal_sizes, 0.0, &[], 0.0, 0.0)',
    content,
    flags=re.DOTALL
)

# For &[None] cases
content = re.sub(
    r'apply_particle_grain_overwrite\((.*?)\&\[None\]\s*\)',
    r'apply_particle_grain_overwrite(\1&[None], 0.0, &[], 0.0, 0.0)',
    content,
    flags=re.DOTALL
)

with open("src/film/development/grain.rs", "w") as f:
    f.write(content)
