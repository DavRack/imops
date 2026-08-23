use crate::color::ColorSpaceTag;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::{
    cfa_coeffs::cfa_coeffs,
    contrast::contrast,
    cst::{camera_cst, cst},
    demosaic::{self, DemosaicAlgorithm},
    exp::exp,
    highlight_reconstruction::highlight_reconstruction,
    image::ImageMetadata,
    lch::lch,
    tone_map::{gamma_encode, inverse_hd_tone_map, inverse_hd_tone_map_with_gain, sigmoid},
};

pub const CHANNELS_PER_PIXEL: usize = 3;

pub type SubPixel = f32;
pub type Pixel = [SubPixel; CHANNELS_PER_PIXEL];

pub type ImageBuffer = Vec<Pixel>;

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub rgb_data: ImageBuffer,
    /// Bayer / mosaic samples. Arc so pipeline runs can share without cloning.
    pub raw_data: Arc<[SubPixel]>,
    pub metadata: ImageMetadata,
}

impl Hash for Image {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.rgb_data.len().hash(hasher);
        for pixel in &self.rgb_data {
            for channel in pixel {
                channel.to_bits().hash(hasher);
            }
        }

        self.raw_data.len().hash(hasher);
        for sample in self.raw_data.iter() {
            sample.to_bits().hash(hasher);
        }

        Hash::hash(&self.metadata, hasher);
    }
}

impl Image {
    pub fn bitwise_eq(&self, other: &Self) -> bool {
        self.rgb_data.len() == other.rgb_data.len()
            && self
                .rgb_data
                .iter()
                .zip(&other.rgb_data)
                .all(|(left, right)| {
                    left.iter()
                        .zip(right)
                        .all(|(left, right)| left.to_bits() == right.to_bits())
                })
            && self.raw_data.len() == other.raw_data.len()
            && self
                .raw_data
                .iter()
                .zip(other.raw_data.iter())
                .all(|(left, right)| left.to_bits() == right.to_bits())
            && self.metadata.bitwise_eq(&other.metadata)
    }
}

impl Default for Image {
    fn default() -> Self {
        Self {
            rgb_data: Vec::new(),
            raw_data: Arc::from([]),
            metadata: ImageMetadata::default(),
        }
    }
}

// pub const R_RELATIVE_LUMINANCE: SubPixel = 0.2126;
// pub const G_RELATIVE_LUMINANCE: SubPixel = 0.7152;
// pub const B_RELATIVE_LUMINANCE: SubPixel = 0.0722;
// for acescg
pub const R_RELATIVE_LUMINANCE: SubPixel = 0.2722287168;
pub const G_RELATIVE_LUMINANCE: SubPixel = 0.6740817658;
pub const B_RELATIVE_LUMINANCE: SubPixel = 0.0536895174;
pub const MIDDLE_GRAY: SubPixel = 0.185;

pub trait PixelOps {
    fn luminance(self) -> SubPixel;
    fn norm(self) -> SubPixel;
}
impl PixelOps for Pixel{
    fn luminance(self) -> SubPixel{
        let [r, g, b] = self;
        let y = R_RELATIVE_LUMINANCE*r + G_RELATIVE_LUMINANCE*g + B_RELATIVE_LUMINANCE*b;
        return y
    }
    fn norm(self) -> SubPixel{
        let [r, g, b] = self;
        let y = r.max(g).max(b);
        return y
    }
}

impl Image {
    pub fn hash(&self) -> u64{
        return self.metadata.hash()
    }

    pub fn lch(&mut self, lch_coefs: [SubPixel; 3]) -> &mut Image{
        let [l_coef, c_coef, h_coef] = lch_coefs;
        lch(
            &mut self.rgb_data,
            self.metadata.color_space.unwrap(),
            l_coef, c_coef, h_coef
        );
        return self
    }

    pub fn exp(&mut self, ev: SubPixel) -> &mut Image{
        exp( &mut self.rgb_data, ev);
        return self
    }

    pub fn cst(&mut self, target_cs: ColorSpaceTag) -> &mut Image{
        cst(
            &mut self.rgb_data,
            self.metadata.color_space.expect("Image needs to have a color space to perform a CST"),
            target_cs,
        );
        self.metadata.color_space = Some(target_cs);
        return self
    }

    pub fn camera_cst(&mut self, target_cs: ColorSpaceTag, camera_color_matrix: &[f32]) -> &mut Image{
        camera_cst(&mut self.rgb_data, target_cs, camera_color_matrix);
        self.metadata.color_space = Some(target_cs);
        return self
    }

    pub fn sigmoid_tone_map(&mut self) -> &mut Image {
        sigmoid(&mut self.rgb_data);
        self
    }

    pub fn sigmoid_tone_map_with_gain(&mut self, gain: f32) -> &mut Image {
        crate::tone_map::sigmoid_with_gain(&mut self.rgb_data, gain);
        self
    }

    pub fn inverse_hd_tone_map(&mut self, s_curve: f32) -> &mut Image {
        inverse_hd_tone_map(&mut self.rgb_data, s_curve);
        self
    }

    pub fn inverse_hd_tone_map_with_gain(
        &mut self,
        s_curve: f32,
        gain: f32,
        ceiling: f32,
    ) -> &mut Image {
        inverse_hd_tone_map_with_gain(&mut self.rgb_data, s_curve, gain, ceiling);
        self
    }

    /// Power-law display encode (`γ` = 2.2 ≈ sRGB, 2.4 = BT.1886).
    pub fn gamma(&mut self, gamma: SubPixel) -> &mut Image {
        gamma_encode(&mut self.rgb_data, gamma);
        self
    }
    
    pub fn cfa_coeffs(&mut self, wb_coeffs: [SubPixel; 4]) -> &mut Image{
        cfa_coeffs(self, wb_coeffs);
        return self
    }

    pub fn contrast(&mut self, value: SubPixel) -> &mut Image{
        contrast(&mut self.rgb_data, value);
        return self
    }
    pub fn highlight_reconstruction(&mut self, wb_coeffs: [SubPixel; 4]) -> &mut Image{
        highlight_reconstruction(&mut self.rgb_data, wb_coeffs);
        return self
    }

    // ponytail: BM3D denoising implementation method
    pub fn bm3d(&mut self, intensity: f32) -> &mut Image{
        crate::bm3d::bm3d(&mut self.rgb_data, self.metadata.width, self.metadata.height, intensity);
        return self
    }

    // ponytail: Chroma-only BM3D denoising
    pub fn chroma_bm3d(&mut self, intensity: f32) -> &mut Image{
        crate::bm3d::chroma_bm3d(&mut self.rgb_data, self.metadata.width, self.metadata.height, intensity);
        return self
    }

    pub fn chroma_denoise(&mut self, radius: usize, epsilon: f32) -> &mut Image{
        crate::chroma_denoise::chroma_denoise(
            &mut self.rgb_data,
            self.metadata.width,
            self.metadata.height,
            radius,
            epsilon,
        );
        self
    }

    pub fn demosaic(
        self,
        demosaic_algorithm: impl DemosaicAlgorithm
    ) -> Image{
        return demosaic::demosaic(
            self,
            demosaic_algorithm
        )
    }

    pub fn vignette(&mut self, strength: f32) -> &mut Image {
        if let Some(ref opcode_list3) = self.metadata.opcode_list3 {
            crate::vignette::apply_vignette_radial_correction(
                &mut self.rgb_data,
                self.metadata.width,
                self.metadata.height,
                opcode_list3,
                strength,
            );
        }
        self
    }

    /// Rotate RGB clockwise by 90/180/270°. Updates `metadata.width` / `height`.
    pub fn rotate(&mut self, turn: crate::rotation::QuarterTurn) -> &mut Image {
        if self.rgb_data.is_empty() {
            let (nw, nh) = turn.output_size(self.metadata.width, self.metadata.height);
            self.metadata.width = nw;
            self.metadata.height = nh;
            return self;
        }
        let (nw, nh, out) = crate::rotation::rotate_rgb(
            &self.rgb_data,
            self.metadata.width,
            self.metadata.height,
            turn,
        );
        self.rgb_data = out;
        self.metadata.width = nw;
        self.metadata.height = nh;
        self
    }

    /// Physically-based analog film simulation. Requires absolute-luminance ACEScg input
    /// (see BaselineExposureCompensation in the pipeline).
    pub fn film(
        &mut self,
        params: &crate::film::FilmParams,
    ) -> Result<&mut Image, crate::film::FilmError> {
        crate::film::process(self, params)?;
        Ok(self)
    }
}
