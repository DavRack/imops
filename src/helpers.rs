use std::usize;

use rawler::pixarray::PixF32;
use rawler::pixarray::RgbF32;

pub fn index2d(height: usize, width: usize) -> impl Iterator<Item = (usize, usize, usize)>{
    (0..(width*height)).into_iter().map(move |idx|{
        let x = idx % width;
        let y = (idx - x) / width;
        (idx, y, x)
    })
}

pub trait Mask {
    fn mask(&self, grayscale_mask: Vec<f32>) -> impl Iterator<Item = f32> where
        Self: Iterator + Sized + Copy + Iterator<Item = f32>,

    {
        self.zip(grayscale_mask).map(|(v, m)| v*m)
    }
}

impl Mask for rayon::slice::Iter<'_, f32> {}
impl Mask for std::slice::Iter<'_, f32> {}


pub trait PixelTail<T> {
   fn get_tail(&self, tail_size: usize, center: usize) -> Vec<(usize, T)>;
   fn get_px_tail(&self, tail_size: usize, center: usize) -> Vec<T>;
}


impl PixelTail<f32> for PixF32 {
    fn get_tail(&self, tail_radious: usize, center: usize) -> Vec<(usize, f32)>{
        let tail_radious = tail_radious as i32;
        let col: i32 = (center % self.width) as i32;
        let row: i32 = (center as i32 - col)/self.width as i32;

        let tail_side = (2*tail_radious as usize)+1;
        let mut tail = Vec::with_capacity(tail_side*tail_side);

        for i in -(tail_radious)..(tail_radious+1) {
            for j in -(tail_radious)..(tail_radious+1) {
                let image_index = (row+i).clamp(0, (self.height-1) as i32) as usize * self.width + (col+j).clamp(0, (self.width-1) as i32) as usize;
                tail.push(
                    (
                        image_index,
                        self.data[image_index]
                    )
                );
            }
        }
        return tail
    }
    fn get_px_tail(&self, tail_radious: usize, center: usize) -> Vec<f32>{
        self.get_tail(tail_radious, center).into_iter().map(|(_, tail)| tail).collect()
    }
}

impl PixelTail<[f32; 3]> for RgbF32 {
    #[inline]
    fn get_tail(&self, tail_radious: usize, center: usize) -> Vec<(usize, [f32; 3])>{
        let tail_radious = tail_radious as i32;
        let col: i32 = (center % self.width) as i32;
        let row: i32 = (center as i32 - col)/self.width as i32;

        let tail_side = (2*tail_radious as usize)+1;
        let mut tail = Vec::with_capacity(tail_side*tail_side);

        for i in -(tail_radious)..(tail_radious+1) {
            for j in -(tail_radious)..(tail_radious+1) {
                let image_index = (row+i).clamp(0, (self.height-1) as i32) as usize * self.width + (col+j).clamp(0, (self.width-1) as i32) as usize;
                tail.push(
                    (
                        image_index,
                        self.data[image_index]
                    )
                );
            }
        }
        return tail
    }

    #[inline]
    fn get_px_tail(&self, tail_radious: usize, center: usize) -> Vec<[f32; 3]>{
        let tail_radious = tail_radious as i32;
        let col: i32 = (center % self.width) as i32;
        let row: i32 = (center as i32 - col)/self.width as i32;

        let tail = (-(tail_radious)..(tail_radious+1)).into_iter().zip(-(tail_radious)..(tail_radious+1)).map(|(i,j)|{
            let image_index = (row+i).clamp(0, (self.height-1) as i32) as usize * self.width + (col+j).clamp(0, (self.width-1) as i32) as usize;
            self.data[image_index]
        }).collect();
        return tail
    }
}

pub trait Stats {
    fn mean(self) -> f32;
    fn median(self) -> f32;
    fn sd(self) -> f32;
    fn variance(self) -> f32;
    fn max(self) -> f32;
    fn min(self) -> f32;
}

impl Stats for std::slice::Iter<'_, f32>{
    fn mean(self) -> f32{
        let lenght = self.len();
        if lenght == 0{
            return 0.0
        }
        let sum: f32 = self.sum();
        return sum/lenght as f32
    }

    fn variance(self) -> f32{
        let len = self.len();
        let mean = self.clone().mean();
        let variance = self.map(|v| (mean-v).powi(2)).sum::<f32>()/(len as f32 - 1.0);
        variance
    }

    fn sd(self) -> f32{
        self.variance().powf(0.5)
    }

    fn max(self) -> f32{
        let r = self.cloned().reduce(f32::max).unwrap();
        r
    }

    fn min(self) -> f32{
        let r = self.cloned().reduce(f32::min).unwrap();
        r
    }

    fn median(self) -> f32{
        let mut sorted_values: Vec<f32> = self.map(|x|*x).collect();
        sorted_values.sort_by(f32::total_cmp);
        let middle_idx = sorted_values.len()/2;
        sorted_values[middle_idx]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_f32(){
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        assert_eq!(data.iter().mean(), 3.5)
    }

    #[test]
    fn test_sd_f32(){
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        assert_eq!(data.iter().sd(), 1.8708287)
    }

    #[test]
    fn test_variance_f32(){
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        assert_eq!(data.iter().variance(), 3.5)
    }

    #[test]
    fn test_median_f32(){
        let data = vec![7.0, 4.0, 3.0, 2.0, 5.0, 6.0, 1.0];

        assert_eq!(data.iter().median(), 4.0)
    }

    #[test]
    fn test_make_tiles_f32(){
        let width = 9;
        let height = 9;
        let og_image = PixF32::new_with(
            vec![
                0.0, 1.0, 2.0,   1.0, 1.0, 1.0,  2.0, 2.0, 2.0,
                3.0, 4.0, 5.0,   1.0, 1.0, 1.0,  2.0, 2.0, 2.0,
                6.0, 7.0, 8.0,   1.0, 1.0, 1.0,  2.0, 2.0, 2.0,
                                                
                3.0, 3.0, 3.0,   4.0, 4.0, 4.0,  5.0, 5.0, 5.0,
                3.0, 3.0, 3.0,   4.0, 4.0, 4.0,  5.0, 5.0, 5.0,
                3.0, 3.0, 3.0,   4.0, 4.0, 4.0,  5.0, 5.0, 5.0,
                                                
                6.0, 6.0, 6.0,   7.0, 7.0, 7.0,  8.0, 8.0, 8.0,
                6.0, 6.0, 6.0,   7.0, 7.0, 7.0,  8.0, 8.0, 8.0,
                6.0, 6.0, 6.0,   7.0, 7.0, 7.0,  8.0, 8.0, 8.0,
            ],
            width,
            height
        );

        // expected tile for center pixel "4.0"
        let expected_tile = vec![
            (30, 4.0), (31, 4.0), (32, 4.0),
            (39, 4.0), (40, 4.0), (41, 4.0),
            (48, 4.0), (49, 4.0), (50, 4.0),
        ];

        let calculated_tile = og_image.get_tail(1, 40);

        assert_eq!(expected_tile, calculated_tile);

        let expected_tile = vec![
            (0, 0.0), (0, 0.0), (1, 1.0),
            (0, 0.0), (0, 0.0), (1, 1.0),
            (9, 3.0), (9, 3.0), (10,4.0),
        ];

        let calculated_tile = og_image.get_tail(1, 0);
        assert_eq!(expected_tile, calculated_tile);

        let expected_tile = vec![
            (0, 0.0), (1, 1.0), (2, 2.0),
            (0, 0.0), (1, 1.0), (2, 2.0),
            (9, 3.0), (10,4.0), (11,5.0),
        ];

        let calculated_tile = og_image.get_tail(1, 1);
        assert_eq!(expected_tile, calculated_tile);
    }

    #[test]
    fn test_make_tiles_rgb(){
        let width = 9;
        let height = 9;
        let og_image = RgbF32::new_with(
            vec![
                [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0],   [1.0,1.0,1.0], [1.0,1.0,1.0], [1.0,1.0,1.0],  [2.0,2.0,2.0], [2.0,2.0,2.0], [2.0,2.0,2.0],
                [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0],   [1.0,1.0,1.0], [1.0,1.0,1.0], [1.0,1.0,1.0],  [2.0,2.0,2.0], [2.0,2.0,2.0], [2.0,2.0,2.0],
                [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0],   [1.0,1.0,1.0], [1.0,1.0,1.0], [1.0,1.0,1.0],  [2.0,2.0,2.0], [2.0,2.0,2.0], [2.0,2.0,2.0],
                                                
                [3.0,3.0,3.0], [3.0,3.0,3.0], [3.0,3.0,3.0],   [4.0,4.0,4.0], [4.0,4.0,4.0], [4.0,4.0,4.0],  [5.0,5.0,5.0], [5.0,5.0,5.0], [5.0,5.0,5.0],
                [3.0,3.0,3.0], [3.0,3.0,3.0], [3.0,3.0,3.0],   [4.0,4.0,4.0], [4.0,4.0,4.0], [4.0,4.0,4.0],  [5.0,5.0,5.0], [5.0,5.0,5.0], [5.0,5.0,5.0],
                [3.0,3.0,3.0], [3.0,3.0,3.0], [3.0,3.0,3.0],   [4.0,4.0,4.0], [4.0,4.0,4.0], [4.0,4.0,4.0],  [5.0,5.0,5.0], [5.0,5.0,5.0], [5.0,5.0,5.0],
                                                
                [6.0,6.0,6.0], [6.0,6.0,6.0], [6.0,6.0,6.0],   [7.0,7.0,7.0], [7.0,7.0,7.0], [7.0,7.0,7.0],  [8.0,8.0,8.0], [8.0,8.0,8.0], [8.0,8.0,8.0],
                [6.0,6.0,6.0], [6.0,6.0,6.0], [6.0,6.0,6.0],   [7.0,7.0,7.0], [7.0,7.0,7.0], [7.0,7.0,7.0],  [8.0,8.0,8.0], [8.0,8.0,8.0], [8.0,8.0,8.0],
                [6.0,6.0,6.0], [6.0,6.0,6.0], [6.0,6.0,6.0],   [7.0,7.0,7.0], [7.0,7.0,7.0], [7.0,7.0,7.0],  [8.0,8.0,8.0], [8.0,8.0,8.0], [8.0,8.0,8.0],
            ],
            width,
            height
        );

        // expected tile for center pixel "4.0"
        let expected_tile = vec![
            (30, [4.0, 4.0, 4.0]), (31, [4.0, 4.0, 4.0]), (32, [4.0, 4.0, 4.0] ),
            (39, [4.0, 4.0, 4.0]), (40, [4.0, 4.0, 4.0]), (41, [4.0, 4.0, 4.0] ),
            (48, [4.0, 4.0, 4.0]), (49, [4.0, 4.0, 4.0]), (50, [4.0, 4.0, 4.0] ),
        ];

        let calculated_tile = og_image.get_tail(1, 40);

        assert_eq!(expected_tile, calculated_tile)
    }
}
