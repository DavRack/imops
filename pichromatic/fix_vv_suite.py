import re

with open("tests/vv_suite.rs", "r") as f:
    content = f.read()

content = re.sub(
    r'(\&\[None\])(\s*)\)',
    r'\1, 0.0, &[], 0.0, 0.0\2)',
    content
)

with open("tests/vv_suite.rs", "w") as f:
    f.write(content)
