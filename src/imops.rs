use std::usize;

use ndarray::{Array3, s};

#[derive(Clone)]
pub struct FormedImage {
    pub height: usize,
    pub width: usize,
    pub black: f32,
    pub white: f32,
    pub data: Array3<f32>,
}

pub trait PipelineModule {
    fn process(&self, image: &mut FormedImage) -> FormedImage;
}

pub struct Exp {
    pub ev: f32
}

pub struct Sigmoid {
    pub c: f32
}

pub struct Contrast {
    pub c: f32
}

pub struct Wb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl PipelineModule for Exp {
    fn process(&self, image: &mut FormedImage) -> FormedImage {
        let value = (2.0 as f32).powf(self.ev);
        image.data.par_mapv_inplace(|v| v*value);
        return image.clone();
    }
}

impl PipelineModule for Sigmoid {
    fn process(&self, image: &mut FormedImage) -> FormedImage {
        image.data.par_mapv_inplace(|v| (1.0 / (1.0 + (1.0/(self.c*v)))).powf(2.0));
        return image.clone();
    }
}

impl PipelineModule for Contrast {
    fn process(&self, image: &mut FormedImage) -> FormedImage {
        const MIDDLE_GRAY: f32 = 0.18;
        image.data.par_mapv_inplace(|v| (MIDDLE_GRAY*(v/MIDDLE_GRAY)).powf(self.c));
        return image.clone();
    }
}

impl PipelineModule for Wb {
    fn process(&self, image: &mut FormedImage) -> FormedImage {
        let rv = self.r;
        let gv = self.g;
        let bv = self.b;

        let r = image.data.clone();
        let r1 = r.slice(s![.., ..,0]);

        let g = image.data.clone();
        let g1 = g.slice(s![.., ..,1]);

        let b = image.data.clone();
        let b1 = b.slice(s![.., ..,2]);

        image.data.slice_mut(s![.., ..,0]).assign(&r1.map(|r|r*rv).clone());
        image.data.slice_mut(s![.., ..,1]).assign(&g1.map(|g|g*gv).clone());
        image.data.slice_mut(s![.., ..,2]).assign(&b1.map(|b|b*bv).clone());
        return image.clone()
    }
}

pub fn get_channel(c: usize, data: &mut Array3<f32>) -> Array3<f32>{
    let shape = data.shape();

    let mut final_image = Array3::<f32>::zeros((shape[0], shape[1], 3));
    final_image.slice_mut(s![.., ..,c]).assign(&data.slice(s![.., .., c]));
    return final_image
}

pub fn film_curve(p: f64, d: f64, a: f64, b: f64, p2: f64, data: &mut Array3<f32>) -> Array3<f32> {
    let f1 = |x: f64| d*(x-p)+p;

    let c2 = (d*a)/f1(a);
    let c1 = (d*a)/(c2*(a.powf(c2)));
    let c4 = (-d*(p2-b))/(f1(b)-1.0);
    let c3 = (-d)/(c4*(p2-b).powf(c4-1.0));
    
    let f2 = |x: f64| c1*(x.powf(c2));
    let f3 = |x: f64| c3*(-x+p2).powf(c4)+1.0;

    let f = |x: f64| 
        if x > p2 { 1.0 }
        else if b < x && x <= p2 { f3(x) }
        else if a < x && x <= b { f1(x) }
        else if 0.0 <= x && x <= a { f2(x) }
        else {0.0}
    ;

    data.par_mapv_inplace(|x| f(x as f64) as f32);
    return data.clone();
}
