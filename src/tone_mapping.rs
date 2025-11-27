pub fn sigmoid(x: f32)-> f32{
    // let scaled_one = (1.0/raw_max_value)*max_image_value;
    // let sigmoid_normalization_constant = 1.0 + (1.0/(scaled_one*c)).powi(2);
    let c: f32 = 2.0;
    let snc = 1.0+(1.0/(c*(16.0_f32-1.0).exp2()).powi(2));
    // let sigmoid = |x:f32| (sigmoid_normalization_constant / (1.0 + (1.0/(c*x)))).powi(2);

    return ((1.0 / (1.0 + (1.0/(c*x)))).powi(2))
}

// fn camera_max_value(max_image_value: f32, max_raw_value: f32) -> f32 {
    // max_raw_value should be from 0 - 1, the max photosite value
    // max_image_value should be from 0 - inf, the max subpixel value of the proceced image
    // 
    // why is this function needed? 
    // lets say we take a photo and no part of that photo is pure white, lets say max 0.8
    // then if we add 
// }

#[cfg(test)]
mod test{
    use crate::sigmoid::sigmoid;

    #[test]
    fn edge_cases_0() {
        let result = sigmoid(0.0, 1.0, 1.0, 1.0);
        let expected = 0.0;
        assert_eq!(expected, result);
    }
    #[test]
    fn edge_cases_1() {
        let result = sigmoid(1.0, 1.0, 1.0, 1.0);
        let expected = 1.0;
        assert_eq!(expected, result);
    }
    #[test]
    fn scaled_input() {
        let result = sigmoid(16.0, 1.0, 1.0, 1.0);
        let expected = 1.0;
        assert_eq!(expected, result);
    }
}
