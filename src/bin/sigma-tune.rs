use pichromatic::film::constants::ABSORPTION_SIGMA_SCALE_PER_UM;
use pichromatic::film::exposure::absorption::{absorb_stack, integrated_absorbed};
use pichromatic::film::exposure::radiance::{relative_to_absolute_luminance, sunny16_exposure};
use pichromatic::film::exposure::upsample::upsample_acescg;
use pichromatic::film::spectrum::WavelengthGrid;
use pichromatic::film::stock::StockId;
use pichromatic::film::units::IsoSpeed;
use pichromatic::pixel::MIDDLE_GRAY;

fn fluence_from_upsample(px: [f32; 3]) -> [f64; 16] {
    let grid = WavelengthGrid::mvp();
    let spectrum = upsample_acescg(px);
    let mut out = [0.0f64; 16];
    for (i, &lambda) in grid.wavelengths_nm.iter().enumerate() {
        out[i] = spectrum[i] / (550.0 / lambda);
    }
    out
}

fn main() {
    let e = sunny16_exposure(IsoSpeed(200.0));
    let g = relative_to_absolute_luminance(
        MIDDLE_GRAY as f64,
        e.shutter_seconds as f64,
        e.f_number as f64,
        e.iso as f64,
    ) as f32;
    let stock = StockId::ColorNeg200.load().unwrap();
    let spectrum = fluence_from_upsample([g, g, g]);
    let mean_s: f64 = upsample_acescg([g, g, g]).iter().sum::<f64>() / 16.0;
    println!("mid abs={g} upsample_mean={mean_s} current_sigma={}", ABSORPTION_SIGMA_SCALE_PER_UM);

    for &sig in &[0.018f64, 0.1, 0.5, 1.0, 1.2, 2.0, 3.0, 5.0, 8.0, 12.0] {
        let (layers, _) = absorb_stack(&stock.layers, &spectrum, sig);
        let mut fracs = Vec::new();
        let mut ei = 0usize;
        for (li, layer) in stock.layers.iter().enumerate() {
            if layer.kind != pichromatic::film::stock::LayerKind::Emulsion {
                continue;
            }
            let phi = integrated_absorbed(&layers.iter().filter(|l| l.produces_latent).nth(ei).unwrap().absorbed);
            // re-find emulsion layers properly
            let _ = (li, phi);
            ei += 1;
        }
        // simpler: absorb and LUT sample
        let (layers, _) = absorb_stack(&stock.layers, &spectrum, sig);
        ei = 0;
        for (li, layer) in stock.layers.iter().enumerate() {
            if layer.kind != pichromatic::film::stock::LayerKind::Emulsion {
                continue;
            }
            let la = layers.iter().filter(|l| l.produces_latent).nth(ei).unwrap();
            let phi = integrated_absorbed(&la.absorbed);
            let f = stock.capture_luts[li].as_ref().unwrap().sample(phi);
            fracs.push(f);
            ei += 1;
        }
        println!("sigma={sig:.2} fracs={:?}", fracs);
    }
}
