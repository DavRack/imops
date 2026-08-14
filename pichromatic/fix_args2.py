import re

files = ["src/film/development/grain.rs", "src/film/development/mod.rs"]

for file in files:
    with open(file, "r") as f:
        content = f.read()

    # match `&crystal_sizes,` or `&[None],` at the end
    content = re.sub(
        r'(\&crystal_sizes|\&\[None\])\s*,?\s*\)',
        r'\1, 0.0, &[], 0.0, 0.0)',
        content
    )
    
    with open(file, "w") as f:
        f.write(content)
