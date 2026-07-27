//! Discrete image rotation (90° / 180° / 270° clockwise).

use crate::gpu::{GpuContext, GpuImageBuffer};
use crate::image::ImageOrientation;
use crate::pixel::{ImageBuffer, Pixel};
use rayon::prelude::*;

/// Clockwise quarter-turn applied to RGB (and metadata width/height).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarterTurn {
    Deg90,
    Deg180,
    Deg270,
}

impl QuarterTurn {
    pub fn from_orientation(orientation: ImageOrientation) -> Option<Self> {
        match orientation {
            ImageOrientation::Normal => None,
            ImageOrientation::Rotate90 => Some(Self::Deg90),
            ImageOrientation::Rotate180 => Some(Self::Deg180),
            ImageOrientation::Rotate270 => Some(Self::Deg270),
        }
    }

    pub fn output_size(self, width: usize, height: usize) -> (usize, usize) {
        match self {
            Self::Deg90 | Self::Deg270 => (height, width),
            Self::Deg180 => (width, height),
        }
    }
}

/// Rotate an interleaved RGB buffer clockwise. Returns `(new_width, new_height, pixels)`.
pub fn rotate_rgb(
    rgb: &[Pixel],
    width: usize,
    height: usize,
    turn: QuarterTurn,
) -> (usize, usize, ImageBuffer) {
    assert_eq!(rgb.len(), width * height, "rgb length must equal width*height");

    match turn {
        QuarterTurn::Deg90 => {
            let (nw, nh) = (height, width);
            let mut out = vec![[0.0; 3]; nw * nh];
            out.par_iter_mut().enumerate().for_each(|(i, dest)| {
                let dx = i % nw;
                let dy = i / nw;
                // (x,y) → (h-1-y, x)  ⇒ inverse: x=dy, y=h-1-dx
                let sx = dy;
                let sy = height - 1 - dx;
                *dest = rgb[sy * width + sx];
            });
            (nw, nh, out)
        }
        QuarterTurn::Deg180 => {
            let mut out = vec![[0.0; 3]; width * height];
            out.par_iter_mut().enumerate().for_each(|(i, dest)| {
                let dx = i % width;
                let dy = i / width;
                let sx = width - 1 - dx;
                let sy = height - 1 - dy;
                *dest = rgb[sy * width + sx];
            });
            (width, height, out)
        }
        QuarterTurn::Deg270 => {
            let (nw, nh) = (height, width);
            let mut out = vec![[0.0; 3]; nw * nh];
            out.par_iter_mut().enumerate().for_each(|(i, dest)| {
                let dx = i % nw;
                let dy = i / nw;
                // (x,y) → (y, w-1-x)  ⇒ inverse: x=w-1-dy, y=dx
                let sx = width - 1 - dy;
                let sy = dx;
                *dest = rgb[sy * width + sx];
            });
            (nw, nh, out)
        }
    }
}

/// GPU rotate: reads `src`, writes a new buffer sized for `turn`. Caller recycles `src`.
pub fn rotate_gpu(
    ctx: &GpuContext,
    src: &GpuImageBuffer,
    turn: QuarterTurn,
) -> GpuImageBuffer {
    let (out_w, out_h) = turn.output_size(src.width, src.height);
    let dst = ctx.acquire_rgba_buffer(out_w, out_h);

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        mode: u32,
        _pad: [u32; 3],
    }

    let mode = match turn {
        QuarterTurn::Deg90 => 1u32,
        QuarterTurn::Deg180 => 2u32,
        QuarterTurn::Deg270 => 3u32,
    };

    let params = Params {
        src_w: src.width as u32,
        src_h: src.height as u32,
        dst_w: out_w as u32,
        dst_h: out_h as u32,
        mode,
        _pad: [0; 3],
    };

    let shader = r#"
        struct Params {
            src_w: u32,
            src_h: u32,
            dst_w: u32,
            dst_h: u32,
            mode: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        };

        @group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
        @group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
        @group(0) @binding(2) var<uniform> params: Params;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let dx = global_id.x;
            let dy = global_id.y;
            if (dx >= params.dst_w || dy >= params.dst_h) {
                return;
            }

            var sx: u32;
            var sy: u32;
            if (params.mode == 1u) {
                // 90° CW: (x,y) → (h-1-y, x)  ⇒ src from dst
                sx = dy;
                sy = params.src_h - 1u - dx;
            } else if (params.mode == 2u) {
                sx = params.src_w - 1u - dx;
                sy = params.src_h - 1u - dy;
            } else {
                // 270° CW
                sx = params.src_w - 1u - dy;
                sy = dx;
            }

            let sidx = sy * params.src_w + sx;
            let didx = dy * params.dst_w + dx;
            dst[didx] = src[sidx];
        }
    "#;

    let gx = (out_w as u32 + 15) / 16;
    let gy = (out_h as u32 + 15) / 16;
    ctx.dispatch_compute_shader_2d_multi(
        "rotation",
        shader,
        &[&src.buffer, &dst.buffer],
        bytemuck::bytes_of(&params),
        gx,
        gy,
    );

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_90_swaps_dims_and_maps_corners() {
        // 2x3:
        // A B
        // C D
        // E F
        let rgb: ImageBuffer = vec![
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ];
        let (nw, nh, out) = rotate_rgb(&rgb, 2, 3, QuarterTurn::Deg90);
        assert_eq!((nw, nh), (3, 2));
        // Top-left of output should be E (bottom-left of input)
        assert_eq!(out[0][0], 5.0);
        // Top-right should be A
        assert_eq!(out[2][0], 1.0);
        // Bottom-left should be F
        assert_eq!(out[3][0], 6.0);
        // Bottom-right should be B
        assert_eq!(out[5][0], 2.0);
    }

    #[test]
    fn rotate_180_keeps_dims() {
        let rgb: ImageBuffer = vec![
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        let (nw, nh, out) = rotate_rgb(&rgb, 2, 2, QuarterTurn::Deg180);
        assert_eq!((nw, nh), (2, 2));
        assert_eq!(out[0][0], 4.0);
        assert_eq!(out[3][0], 1.0);
    }
}
