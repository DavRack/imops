class pix:
    def __init__(self, r, g, b) -> None:
        self.r = r
        self.g = g
        self.b = b

    def __str__(self) -> str:
        return f"{self.r} {self.g} {self.b}"

    def luminance(self):
        return 0.2126*self.r + 0.7152*self.g + 0.0722*self.b

    def saturation(self):
        return 1-((3*(min(self.r, self.g, self.b)))/(self.r+self.g+self.b))

    def mul(self, v):
        return pix(
        self.r * v,
        self.g * v,
        self.b * v,
        )


p = pix(
0.10467708,
0.12962304,
0.011212737,
        )

def luminance(r, g, b):
    return 0.2126*r + 0.7152*g + 0.0722*b

def saturation(r, g, b):
    return 1-((3*(min(r, g, b)))/(r+g+b))

print(p)
print("luminance", p.luminance())
print("saturation", p.saturation())

p = p.mul(2)
print("new p", p)
print("luminance", p.luminance())
print("saturation", p.saturation())
