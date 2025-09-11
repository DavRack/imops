pub type SubPixel = f32;
pub type Pixel = [SubPixel; CHANNELS_PER_PIXEL];
pub type ImageBuffer = Vec<Pixel>;

pub const CHANNELS_PER_PIXEL: usize = 3;
pub const R_RELATIVE_LUMINANCE: SubPixel = 0.2126;
pub const G_RELATIVE_LUMINANCE: SubPixel = 0.7152;
pub const B_RELATIVE_LUMINANCE: SubPixel = 0.0722;
pub const MIDDLE_GRAY: SubPixel = 0.185;

pub trait PixelOps {
    fn luminance(self) -> SubPixel;
    fn saturation(self) -> SubPixel;
}
impl PixelOps for Pixel{
    fn luminance(self) -> SubPixel{
        let [r, g, b] = self;
        let y = R_RELATIVE_LUMINANCE*r + G_RELATIVE_LUMINANCE*g + B_RELATIVE_LUMINANCE*b;
        return y
    }
    fn saturation(self) -> SubPixel {
        let [r, g, b] = self;
        let s = 1.0-((3.0*r.min(g).min(b))/(r+g+b));
        return s
    }
}

pub trait BufferOps {
    fn max_luminance(self) -> SubPixel;
}

impl BufferOps for ImageBuffer {
    fn max_luminance(self) -> SubPixel{
        self.iter().fold(0.0, |current_max, pixel| pixel.luminance().max(current_max))
    }
}
