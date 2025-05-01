pub const CHANNELS_PER_PIXEL: usize = 3;
pub type SubPixel = f32;
pub type Pixel = [SubPixel; CHANNELS_PER_PIXEL];
pub type ImageBuffer = Vec<Pixel>;

pub trait PixelOps {
    fn luminance(self) -> SubPixel;
    fn saturation(self) -> SubPixel;
}
impl PixelOps for Pixel{
    fn luminance(self) -> SubPixel{
        let [r, g, b] = self;
        let y = 0.2126*r + 0.7152*g + 0.0722*b;
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
