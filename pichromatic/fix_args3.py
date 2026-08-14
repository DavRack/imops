import re

files = ["src/film/development/grain.rs", "src/film/development/mod.rs"]

for file in files:
    with open(file, "r") as f:
        content = f.read()

    # only replace apply_particle_grain_overwrite arguments
    def repl(m):
        args = m.group(1)
        # remove trailing spaces and commas
        args = args.rstrip(" ,\n")
        return f"apply_particle_grain_overwrite({args}, 0.0, &[], 0.0, 0.0)"
    
    content = re.sub(
        r'apply_particle_grain_overwrite\((.*?\&(?:crystal_sizes|\[None\]))\s*,?\s*\)',
        repl,
        content,
        flags=re.DOTALL
    )
    
    with open(file, "w") as f:
        f.write(content)
