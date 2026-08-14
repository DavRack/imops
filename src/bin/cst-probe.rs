use pichromatic::color::ColorSpaceTag;

fn print_matrix(name: &str, from: ColorSpaceTag, to: ColorSpaceTag) {
    let c0 = from.convert(to, [1.0, 0.0, 0.0]);
    let c1 = from.convert(to, [0.0, 1.0, 0.0]);
    let c2 = from.convert(to, [0.0, 0.0, 1.0]);
    println!("pub const {}: [[f32; 3]; 3] = [", name);
    println!("    [{:.9}, {:.9}, {:.9}],", c0[0], c1[0], c2[0]);
    println!("    [{:.9}, {:.9}, {:.9}],", c0[1], c1[1], c2[1]);
    println!("    [{:.9}, {:.9}, {:.9}],", c0[2], c1[2], c2[2]);
    println!("];");
}

fn main() {
    print_matrix("LINEAR_SRGB_TO_XYZ_D65", ColorSpaceTag::LinearSrgb, ColorSpaceTag::XyzD65);
    print_matrix("XYZ_D65_TO_LINEAR_SRGB", ColorSpaceTag::XyzD65, ColorSpaceTag::LinearSrgb);
    print_matrix("ACESCG_TO_XYZ_D65", ColorSpaceTag::AcesCg, ColorSpaceTag::XyzD65);
    print_matrix("XYZ_D65_TO_ACESCG", ColorSpaceTag::XyzD65, ColorSpaceTag::AcesCg);
    print_matrix("ACESCG_TO_LINEAR_SRGB", ColorSpaceTag::AcesCg, ColorSpaceTag::LinearSrgb);
    print_matrix("LINEAR_SRGB_TO_ACESCG", ColorSpaceTag::LinearSrgb, ColorSpaceTag::AcesCg);
    print_matrix("DISPLAY_P3_TO_XYZ_D65", ColorSpaceTag::DisplayP3, ColorSpaceTag::XyzD65);
    print_matrix("XYZ_D65_TO_DISPLAY_P3", ColorSpaceTag::XyzD65, ColorSpaceTag::DisplayP3);
    print_matrix("REC2020_TO_XYZ_D65", ColorSpaceTag::Rec2020, ColorSpaceTag::XyzD65);
    print_matrix("XYZ_D65_TO_REC2020", ColorSpaceTag::XyzD65, ColorSpaceTag::Rec2020);

    println!("Oklab conversions:");
    let lab = ColorSpaceTag::LinearSrgb.convert(ColorSpaceTag::Oklab, [0.5, 0.3, 0.8]);
    println!("LinearSrgb [0.5, 0.3, 0.8] -> Oklab: {:?}", lab);
    let srgb = ColorSpaceTag::Oklab.convert(ColorSpaceTag::LinearSrgb, lab);
    println!("Oklab -> LinearSrgb: {:?}", srgb);

    let lch = ColorSpaceTag::Oklab.convert(ColorSpaceTag::Oklch, lab);
    println!("Oklab -> Oklch: {:?}", lch);
    let lab2 = ColorSpaceTag::Oklch.convert(ColorSpaceTag::Oklab, lch);
    println!("Oklch -> Oklab: {:?}", lab2);
}

