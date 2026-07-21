use exr::prelude::*;
fn main() {
    let path = std::env::args().nth(1).expect("exr-hist <file>");
    let image = read_first_rgba_layer_from_file(
        &path,
        |resolution, _| vec![vec![[0.0f32; 3]; resolution.width()]; resolution.height()],
        |rows, pos, (r, g, b, _a): (f32, f32, f32, f32)| {
            rows[pos.y()][pos.x()] = [r, g, b];
        },
    )
    .expect("read");
    let mut hist = [0u64; 20];
    let mut n = 0u64;
    let mut y_sum = 0.0f64;
    let mut shadow_mean = [0.0f64; 3];
    let mut shadow_n = 0u64;
    let mut deep_mean = [0.0f64; 3];
    let mut deep_n = 0u64;
    for row in &image.layer_data.channel_data.pixels {
        for rgb in row {
            let y = 0.2627 * rgb[0] + 0.6780 * rgb[1] + 0.0593 * rgb[2];
            y_sum += y as f64;
            let bin = ((y / 1.2) * 20.0).floor().clamp(0.0, 19.0) as usize;
            hist[bin] += 1;
            n += 1;
            if y < 0.08 {
                shadow_n += 1;
                for c in 0..3 { shadow_mean[c] += rgb[c] as f64; }
            }
            if y < 0.02 {
                deep_n += 1;
                for c in 0..3 { deep_mean[c] += rgb[c] as f64; }
            }
        }
    }
    println!("n={n} meanY={:.4}", y_sum / n as f64);
    println!("frac Y<0.02={:.4} frac Y<0.08={:.4}", deep_n as f64/n as f64, shadow_n as f64/n as f64);
    if deep_n > 0 {
        println!("deep Y<0.02 meanRGB=[{:.4},{:.4},{:.4}]",
            deep_mean[0]/deep_n as f64, deep_mean[1]/deep_n as f64, deep_mean[2]/deep_n as f64);
    }
    if shadow_n > 0 {
        println!("shadow Y<0.08 meanRGB=[{:.4},{:.4},{:.4}]",
            shadow_mean[0]/shadow_n as f64, shadow_mean[1]/shadow_n as f64, shadow_mean[2]/shadow_n as f64);
    }
    println!("Y hist over [0,1.2):");
    for (i, &c) in hist.iter().enumerate() {
        let lo = i as f32 * 0.06;
        let pct = 100.0 * c as f64 / n as f64;
        let bar = "#".repeat((pct * 1.5) as usize);
        println!("  [{lo:.2},{:.2}) {pct:5.2}% {bar}", lo + 0.06);
    }
}
