use std::path::Path;

use pichromatic::film::development::reduction::reduce;
use pichromatic::film::stock::StockId;
use pichromatic::film::stock::{FilmStock, LayerKind};
use pichromatic::film::types::{DyePlanes, LatentPlanes};
use serde_json::Value;

fn collect_numbers(value: &Value, out: &mut Vec<f64>) {
    match value {
        Value::Number(number) => out.push(number.as_f64().unwrap()),
        Value::Array(values) => values.iter().for_each(|value| collect_numbers(value, out)),
        _ => {}
    }
}

fn finite_numbers(value: &Value) -> Vec<f64> {
    let mut out = Vec::new();
    collect_numbers(value, &mut out);
    out
}

fn peak_nm(values: &Value, wavelengths: &[f64], column: usize) -> f64 {
    let rows = values.as_array().expect("curve rows");
    let (index, _) = rows
        .iter()
        .enumerate()
        .filter_map(|(i, row)| row.as_array()?.get(column)?.as_f64().map(|v| (i, v)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("finite curve peak");
    wavelengths[index]
}

fn column_samples(values: &Value, column: usize) -> Vec<Option<f64>> {
    values
        .as_array()
        .expect("curve rows")
        .iter()
        .map(|row| {
            row.as_array()
                .and_then(|row| row.get(column))
                .and_then(Value::as_f64)
        })
        .collect()
}

fn scalar_samples(values: &Value) -> Vec<Option<f64>> {
    values
        .as_array()
        .expect("scalar samples")
        .iter()
        .map(Value::as_f64)
        .collect()
}

fn interpolate(xs: &[f64], ys: &[Option<f64>], target: f64) -> Option<f64> {
    xs.windows(2)
        .zip(ys.windows(2))
        .find_map(|(x_pair, y_pair)| {
            let [x0, x1] = x_pair else { unreachable!() };
            let [Some(y0), Some(y1)] = y_pair else {
                return None;
            };
            if *x0 <= target && target <= *x1 {
                let fraction = (target - *x0) / (*x1 - *x0);
                Some(y0 + fraction * (y1 - y0))
            } else {
                None
            }
        })
        .or_else(|| {
            xs.last()
                .zip(ys.last())
                .filter(|(x, y)| **x == target && y.is_some())
                .and_then(|(_, y)| *y)
        })
}

fn reconstruction_error(values: &Value, wavelengths: &[f64], column: usize) -> (f64, f64) {
    reconstruction_error_for_samples(&column_samples(values, column), wavelengths)
}

fn reconstruction_error_for_samples(samples: &[Option<f64>], wavelengths: &[f64]) -> (f64, f64) {
    let grid: Vec<f64> = (400..=700).step_by(20).map(f64::from).collect();
    let grid_samples: Vec<Option<f64>> = grid
        .iter()
        .map(|&wavelength| interpolate(wavelengths, samples, wavelength))
        .collect();
    let errors: Vec<f64> = wavelengths
        .iter()
        .zip(samples.iter().copied())
        .filter_map(|(&wavelength, sample)| {
            if !(400.0..=700.0).contains(&wavelength) {
                return None;
            }
            let sample = sample?;
            let reconstructed = interpolate(&grid, &grid_samples, wavelength)?;
            Some(sample - reconstructed)
        })
        .collect();
    assert!(!errors.is_empty(), "reference curve has no shared samples");
    let max = errors.iter().map(|error| error.abs()).fold(0.0, f64::max);
    let rms = (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    (max, rms)
}

fn runtime_grid_samples(samples: &[Option<f64>], wavelengths: &[f64]) -> Vec<f64> {
    (400..=700)
        .step_by(20)
        .map(|wavelength| {
            interpolate(wavelengths, samples, f64::from(wavelength))
                .unwrap_or_else(|| panic!("reference has no sample at {wavelength} nm"))
        })
        .collect()
}

fn range(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .copied()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        })
}

fn compare_spectral_series(
    label: &str,
    reference: &[Option<f64>],
    wavelengths: &[f64],
    model: &[f64],
) {
    let reference_grid = runtime_grid_samples(reference, wavelengths);
    let errors: Vec<f64> = reference_grid
        .iter()
        .zip(model)
        .map(|(reference, model)| model - reference)
        .collect();
    let max_error = errors.iter().map(|error| error.abs()).fold(0.0, f64::max);
    let rms_error =
        (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    println!(
        "{label}: reference range {:?}, imops range {:?}, max/RMS at 20 nm = {max_error:.6}/{rms_error:.6}, reference reconstruction max/RMS = {:?}",
        range(&reference_grid),
        range(model),
        reconstruction_error_for_samples(reference, wavelengths),
    );
}

fn runtime_density_spectrum(stock: &FilmStock, dyes: &DyePlanes) -> Vec<f64> {
    let mut density = vec![0.0; 16];
    let mut emulsion_index = 0;
    for layer in &stock.layers {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        let coupler = layer.coupler.as_ref().expect("emulsion coupler");
        let image_density = f64::from(dyes.image_dye[emulsion_index][0]);
        let mask_density = f64::from(dyes.mask_dye[emulsion_index][0]);
        for (index, value) in density.iter_mut().enumerate() {
            *value += image_density * coupler.epsilon.samples[index];
            if let Some(mask) = &coupler.mask_epsilon {
                *value += mask_density * mask.samples[index];
            }
        }
        emulsion_index += 1;
    }
    density
}

fn runtime_dyes_at(stock: &FilmStock, fraction: f32) -> DyePlanes {
    let layer_count = stock.emulsion_layers().count();
    let latent = LatentPlanes {
        width: 1,
        height: 1,
        layers: vec![vec![fraction]; layer_count],
    };
    reduce(stock, &latent)
}

fn runtime_full_dye_spectrum(stock: &FilmStock, reference_channel: usize) -> Vec<f64> {
    let emulsions: Vec<_> = stock.emulsion_layers().collect();
    let pair = 2 - reference_channel; // imops stores B, G, R; profile stores R, G, B.
    let first = pair * 2;
    let d_max = emulsions[first].1.coupler.as_ref().unwrap().d_max
        + emulsions[first + 1].1.coupler.as_ref().unwrap().d_max;
    emulsions[first]
        .1
        .coupler
        .as_ref()
        .unwrap()
        .epsilon
        .samples
        .iter()
        .map(|&epsilon| f64::from(d_max) * epsilon)
        .collect()
}

fn runtime_layer_curve(stock: &FilmStock, pair: usize, layer_in_pair: usize) -> Vec<f64> {
    let emulsions: Vec<_> = stock.emulsion_layers().collect();
    let (layer_index, layer) = emulsions[pair * 2 + layer_in_pair];
    let lut = stock.capture_luts[layer_index].as_ref().unwrap();
    let d_max = f64::from(layer.coupler.as_ref().unwrap().d_max);
    let inverse_gamma = 1.0 / f64::from(layer.gamma_contrast);
    lut.fraction
        .iter()
        .map(|&fraction| d_max * f64::from(fraction).powf(inverse_gamma))
        .collect()
}

fn runtime_hd_curve(stock: &FilmStock, pair: usize) -> Vec<f64> {
    let fast = runtime_layer_curve(stock, pair, 0);
    let slow = runtime_layer_curve(stock, pair, 1);
    fast.into_iter()
        .zip(slow)
        .map(|(fast, slow)| fast + slow)
        .collect()
}

fn required_column_samples(values: &Value, column: usize) -> Vec<f64> {
    column_samples(values, column)
        .into_iter()
        .enumerate()
        .map(|(index, value)| value.unwrap_or_else(|| panic!("missing sample {index}")))
        .collect()
}

fn layered_samples(values: &Value, channel: usize, layer: usize) -> Vec<f64> {
    values
        .as_array()
        .expect("layered density rows")
        .iter()
        .enumerate()
        .map(|(index, row)| {
            row.as_array()
                .and_then(|channels| channels.get(channel))
                .and_then(Value::as_array)
                .and_then(|layers| layers.get(layer))
                .and_then(Value::as_f64)
                .unwrap_or_else(|| panic!("missing layered sample {index}"))
        })
        .collect()
}

fn relative_curve_error(
    reference: &[f64],
    ref_x: &[f64],
    model: &[f64],
    model_x: &[f64],
) -> (f64, f64) {
    let mut errors = Vec::new();
    for (i, &x) in ref_x.iter().enumerate() {
        if x >= model_x[0] && x <= *model_x.last().unwrap() {
            if let Some(model_val) = interpolate(
                model_x,
                &model.iter().copied().map(Some).collect::<Vec<_>>(),
                x,
            ) {
                errors.push(model_val - reference[i]);
            }
        }
    }
    if errors.is_empty() {
        return (0.0, 0.0);
    }
    let max = errors.iter().map(|error| error.abs()).fold(0.0, f64::max);
    let rms = (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    (max, rms)
}

fn three_positions(values: &[f64]) -> [f64; 3] {
    [
        values[0],
        values[values.len() / 2],
        values[values.len() - 1],
    ]
}

#[test]
#[ignore = "requires the user's local spektrafilm profile"]
fn portra_reference_comparison() {
    let path = std::env::var_os("SPEKTRAFILM_PORTRA_PROFILE")
        .expect("set SPEKTRAFILM_PORTRA_PROFILE to the local Portra profile");
    assert!(
        Path::new(&path).is_file(),
        "profile does not exist: {:?}",
        path
    );
    let profile: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read local Portra profile"))
            .expect("parse local Portra profile");
    let data = &profile["data"];
    let info = &profile["info"];
    let wavelengths: Vec<f64> = data["wavelengths"]
        .as_array()
        .expect("reference wavelengths")
        .iter()
        .map(|v| v.as_f64().expect("finite wavelength"))
        .collect();
    assert_eq!(wavelengths.len(), 81);
    assert_eq!(wavelengths.first(), Some(&380.0));
    assert_eq!(wavelengths.last(), Some(&780.0));
    assert_eq!(info["reference_illuminant"], "D55");
    assert_eq!(info["viewing_illuminant"], "D50");

    for key in [
        "log_sensitivity",
        "channel_density",
        "base_density",
        "midscale_neutral_density",
    ] {
        let numbers = finite_numbers(&data[key]);
        assert!(!numbers.is_empty(), "empty {key}");
        assert!(
            numbers.into_iter().all(|v| v.is_finite()),
            "non-finite {key}"
        );
    }

    let log_exposure = finite_numbers(&data["log_exposure"]);
    let density_curves = data["density_curves"].as_array().expect("H&D rows");
    assert_eq!(log_exposure.len(), density_curves.len());
    assert!(density_curves.iter().all(|row| {
        row.as_array().is_some_and(|row| {
            row.len() == 3
                && row
                    .iter()
                    .all(|value| value.as_f64().is_some_and(f64::is_finite))
        })
    }));

    let layered_curves = data["density_curves_layers"]
        .as_array()
        .expect("layered H&D rows");
    assert_eq!(log_exposure.len(), layered_curves.len());
    assert!(layered_curves.iter().all(|row| {
        row.as_array().is_some_and(|channels| {
            channels.len() == 3
                && channels.iter().all(|channel| {
                    channel.as_array().is_some_and(|layers| {
                        layers.len() == 3
                            && layers
                                .iter()
                                .all(|value| value.as_f64().is_some_and(f64::is_finite))
                    })
                })
        })
    }));
    // These are measured curves; shape validation deliberately does not assert monotonicity.

    for key in ["density_curves", "density_curves_layers"] {
        let numbers = finite_numbers(&data[key]);
        assert!(!numbers.is_empty(), "empty {key}");
        assert!(
            numbers.into_iter().all(|v| v.is_finite()),
            "non-finite {key}"
        );
    }

    let stock = StockId::Portra400.load().expect("Portra runtime stock");
    let model_peaks: Vec<f64> = stock
        .emulsion_layers()
        .map(|(_, layer)| {
            layer
                .spectral_sensitivity
                .as_ref()
                .unwrap()
                .peak_wavelength()
        })
        .collect();
    let reference_peaks = [
        peak_nm(&data["log_sensitivity"], &wavelengths, 2),
        peak_nm(&data["log_sensitivity"], &wavelengths, 1),
        peak_nm(&data["log_sensitivity"], &wavelengths, 0),
    ];
    for (model, reference) in model_peaks
        .chunks(2)
        .map(|pair| pair[0])
        .zip(reference_peaks)
    {
        assert!(
            (model - reference).abs() <= 10.0,
            "model peak {model} vs reference {reference}"
        );
    }

    for (key, columns) in [
        ("log_sensitivity", [0, 1, 2]),
        ("channel_density", [0, 1, 2]),
    ] {
        let errors: Vec<(f64, f64)> = columns
            .into_iter()
            .map(|column| reconstruction_error(&data[key], &wavelengths, column))
            .collect();
        println!("{key} reconstruction max/RMS R/G/B: {errors:?}");
    }

    for (channel, label) in [(0, "R"), (1, "G"), (2, "B")] {
        compare_spectral_series(
            &format!("dye-density spectrum {label}"),
            &column_samples(&data["channel_density"], channel),
            &wavelengths,
            &runtime_full_dye_spectrum(&stock, channel),
        );
    }

    let base_reference = scalar_samples(&data["base_density"]);
    let midscale_reference = scalar_samples(&data["midscale_neutral_density"]);
    let base_model = runtime_density_spectrum(&stock, &runtime_dyes_at(&stock, 0.0));
    let midscale_model = runtime_density_spectrum(&stock, &runtime_dyes_at(&stock, 0.5));
    compare_spectral_series("base density", &base_reference, &wavelengths, &base_model);
    compare_spectral_series(
        "midscale-neutral density",
        &midscale_reference,
        &wavelengths,
        &midscale_model,
    );

    let reference_exposure_range = range(&log_exposure);
    let model_log10_fluence = {
        let emulsions: Vec<_> = stock.emulsion_layers().collect();
        let lut = stock.capture_luts[emulsions[0].0].as_ref().unwrap();
        lut.log10_fluence
            .iter()
            .map(|&x| f64::from(x))
            .collect::<Vec<f64>>()
    };
    let runtime_exposure_range = (model_log10_fluence[0], *model_log10_fluence.last().unwrap());
    for (channel, label) in [(0, "R"), (1, "G"), (2, "B")] {
        let reference = required_column_samples(&data["density_curves"], channel);
        let model = runtime_hd_curve(&stock, 2 - channel);
        println!(
            "H&D density {label}: reference log-exposure {:?}, range {:?}; imops log10-fluence {:?}, range {:?}; normalized-position max/RMS {:?}; low/mid/high reference={:?}, imops={:?}",
            reference_exposure_range,
            range(&reference),
            runtime_exposure_range,
            range(&model),
            relative_curve_error(&reference, &log_exposure, &model, &model_log10_fluence),
            three_positions(&reference),
            three_positions(&model),
        );

        for layer in 0..3 {
            let reference = layered_samples(&data["density_curves_layers"], channel, layer);
            let model = if layer < 2 {
                runtime_layer_curve(&stock, 2 - channel, layer)
            } else {
                Vec::new()
            };
            if model.is_empty() {
                println!(
                    "layered density {label}/{layer}: reference range {:?}, low/mid/high={:?}; imops has two fast/slow layers",
                    range(&reference),
                    three_positions(&reference),
                );
            } else {
                println!(
                    "layered density {label}/{layer}: reference range {:?}, imops range {:?}, normalized-position max/RMS {:?}; low/mid/high reference={:?}, imops={:?}",
                    range(&reference),
                    range(&model),
                    relative_curve_error(&reference, &log_exposure, &model, &model_log10_fluence),
                    three_positions(&reference),
                    three_positions(&model),
                );
            }
        }
    }
    println!("reference grid: 380..780 nm / 5 nm; runtime: 400..700 nm / 20 nm");
    println!("reference illuminants: D55 capture, D50 viewing");
    println!("runtime sensitivity peaks B/G/R: {:?}", model_peaks);
}
