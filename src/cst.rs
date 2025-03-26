const OKLAB_LAB_TO_LMS: [[f32; 3]; 3] = [
    [1.0, 0.396_337_78, 0.215_803_76],
    [1.0, -0.105_561_346, -0.063_854_17],
    [1.0, -0.089_484_18, -1.291_485_5],
];

const OKLAB_LMS_TO_LAB: [[f32; 3]; 3] = [
    [0.210_454_26, 0.793_617_8, -0.004_072_047],
    [1.977_998_5, -2.428_592_2, 0.450_593_7],
    [0.025_904_037, 0.782_771_77, -0.808_675_77],
];

const OKLAB_XYZ_TO_LMS: [[f32; 3]; 3] = [
    [ 0.96619195,  0.55127368, -0.31248951,],
    [ 0.00165046,  0.76884006,  0.17778831,],
    [ 0.07304166,  0.188758  ,  0.64682497,],
];

const OKLAB_LMS_TO_XYZ: [[f32; 3]; 3] = [
    [ 0.98232712, -0.88026228,  0.71652658,],
    [ 0.02524606,  1.37215964, -0.3649593 ,],
    [-0.11829507, -0.30102463,  1.57160392,],
];
const ONE_THIRD: f32 = 1.0/3.0;

#[inline]
fn matmul(m: &[[f32; 3]; 3], x: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * x[0] + m[0][1] * x[1] + m[0][2] * x[2],
        m[1][0] * x[0] + m[1][1] * x[1] + m[1][2] * x[2],
        m[2][0] * x[0] + m[2][1] * x[1] + m[2][2] * x[2],
    ]
}

#[inline]
pub fn xyz_to_oklab(src: [f32; 3]) -> [f32; 3]{
        let lms = matmul(&OKLAB_XYZ_TO_LMS, src).map(|v| v.powf(ONE_THIRD));
        matmul(&OKLAB_LMS_TO_LAB, lms)
    // xyz -> srgb -> oklab
}

pub fn xyz_to_oklab_l(src: [f32; 3]) -> f32{
        let lms = matmul(&OKLAB_XYZ_TO_LMS, src).map(|v| v.powf(ONE_THIRD));
        OKLAB_LMS_TO_LAB[0][0] * lms[0] + OKLAB_LMS_TO_LAB[0][1] * lms[1] + OKLAB_LMS_TO_LAB[0][2] * lms[2]
    // xyz -> srgb -> oklab
}

#[inline]
pub fn oklab_to_xyz(src: [f32; 3]) -> [f32; 3]{
        let lms = matmul(&OKLAB_LAB_TO_LMS, src).map(|x| x.powi(3));
        matmul(&OKLAB_LMS_TO_XYZ, lms)
    // oklab -> srgb -> oklab
}
