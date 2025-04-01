// pub fn applyParametricMask(image: Vec<[f32; 3]>) -> Vec<[f32; 3]>{
//     let mask_fn = |x: [f32; 3]|{
//         let [r, g, b] = x;
//         let luminance = r.max(g).max(b);
//         if luminance > 0.18{
//             0
//         }else{
//             1
//         }
//     };
//     image.iter_mut().for_each(|p|{
//         let mask = mask_fn(p);
//         *p = p.map(|v| v*p)
//     });
//     image
// }
