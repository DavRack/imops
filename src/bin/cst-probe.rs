use color::ColorSpaceTag;

fn main() {
    let samples = [
        [0.0f32, 0.0, 0.0],
        [0.02, 0.02, 0.02],
        [0.05, 0.04, 0.03],
        [0.185, 0.185, 0.185],
        [0.1, 0.15, 0.05],
        [0.3, 0.2, 0.1],
        [1.0, 0.5, 0.2],
        [0.01, 0.02, 0.0],
    ];
    println!("ACEScg → Rec2020:");
    for s in samples {
        let o = ColorSpaceTag::AcesCg.convert(ColorSpaceTag::Rec2020, s);
        println!("  {s:?} → {o:?}");
    }
}
