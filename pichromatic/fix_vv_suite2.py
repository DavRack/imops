with open("tests/vv_suite.rs", "r") as f:
    content = f.read()

content = content.replace("&[None][None], 0.0, &[], 0.0, 0.0", "&[None], 0.0, &[], 0.0, 0.0")

with open("tests/vv_suite.rs", "w") as f:
    f.write(content)
