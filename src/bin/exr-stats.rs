use exr::prelude::*;

fn main() {
    let path = std::env::args().nth(1).expect("usage: exr-stats <file.exr>");

    let meta = MetaData::read_from_file(&path, false).expect("meta");
    for h in &meta.headers {
        println!(
            "layer={:?} compression={:?} size={:?}",
            h.own_attributes.layer_name, h.compression, h.layer_size
        );
        println!("  chromaticities={:?}", h.shared_attributes.chromaticities);
        println!(
            "  software={:?} comments={:?}",
            h.own_attributes.software_name, h.own_attributes.comments
        );
        for (k, v) in &h.own_attributes.other {
            println!("  attr {k} = {v:?}");
        }
        println!(
            "  exposure={:?} aperture={:?} iso={:?}",
            h.own_attributes.exposure, h.own_attributes.aperture, h.own_attributes.iso_speed
        );
    }

    let image = read_first_rgba_layer_from_file(
        &path,
        |resolution, _| {
            vec![vec![[0.0f32; 3]; resolution.width()]; resolution.height()]
        },
        |rows, pos, (r, g, b, _a): (f32, f32, f32, f32)| {
            rows[pos.y()][pos.x()] = [r, g, b];
        },
    )
    .expect("read pixels");

    let mut minv = [f32::INFINITY; 3];
    let mut maxv = [f32::NEG_INFINITY; 3];
    let mut sum = [0.0f64; 3];
    let mut n = 0u64;
    let mut near_gray = 0u64;
    let mut le0 = [0u64; 3];
    let mut ge099 = [0u64; 3];
    let mut nonfinite = 0u64;

    for row in &image.layer_data.channel_data.pixels {
        for rgb in row {
            if !rgb.iter().all(|c| c.is_finite()) {
                nonfinite += 1;
                continue;
            }
            for c in 0..3 {
                minv[c] = minv[c].min(rgb[c]);
                maxv[c] = maxv[c].max(rgb[c]);
                sum[c] += rgb[c] as f64;
                if rgb[c] <= 0.0 {
                    le0[c] += 1;
                }
                if rgb[c] >= 0.99 {
                    ge099[c] += 1;
                }
            }
            let mx = rgb[0].max(rgb[1]).max(rgb[2]);
            let mn = rgb[0].min(rgb[1]).min(rgb[2]);
            if mx - mn < 1e-3 * (1.0 + mx.abs()) {
                near_gray += 1;
            }
            n += 1;
        }
    }

    let mean = [sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64];
    println!("n={n} nonfinite={nonfinite}");
    println!("min={minv:?}");
    println!("max={maxv:?}");
    println!("mean={mean:?}");
    println!("near_gray_frac={:.4}", near_gray as f64 / n as f64);
    println!(
        "frac<=0=[{:.4},{:.4},{:.4}] frac>=0.99=[{:.4},{:.4},{:.4}]",
        le0[0] as f64 / n as f64,
        le0[1] as f64 / n as f64,
        le0[2] as f64 / n as f64,
        ge099[0] as f64 / n as f64,
        ge099[1] as f64 / n as f64,
        ge099[2] as f64 / n as f64,
    );
}
