import re
import glob

files = ["src/film/development/grain.rs", "src/film/development/mod.rs"]

for file in files:
    with open(file, "r") as f:
        content = f.read()

    # The issue is that &crystal_sizes is the last argument.
    # So we want to replace `&crystal_sizes,\n        )` or `&[None],\n        )`
    # with `&crystal_sizes, 0.0, &[], 0.0, 0.0)`
    
    content = re.sub(
        r'(\&crystal_sizes|\&\[None\])(\s*)\)',
        r'\1, 0.0, &[], 0.0, 0.0\2)',
        content
    )
    
    with open(file, "w") as f:
        f.write(content)
