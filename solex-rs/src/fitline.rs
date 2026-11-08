//! Mean/max image computation and spectral-line polynomial fit
//! (solex_util.compute_mean_max / compute_mean_return_fit).

use crate::num::*;
use crate::ser::{self, SerInfo};
use std::io;

pub struct LineFit {
    /// per-row: (floor index, fractional weight, curve value)
    pub fit: Vec<(i64, f64, f64)>,
    pub y1: usize,
    pub y2: usize,
    pub poly: [f64; 4], // ascending coefficients of final fit
    /// sharp (unblurred) line minimum per row, for the diagnostic plot
    pub sharp: Vec<usize>,
    /// rows whose minimum agrees with the fit within 5 px
    pub mask: Vec<bool>,
    /// polynomial curve sampled over all rows
    pub curve: Vec<f64>,
}

/// Pass 1: mean and max images over all frames, parallel over frame chunks.
pub fn compute_mean_max(path: &str, info: &SerInfo, nthreads: usize) -> io::Result<(Img, Img)> {
    let npix = info.ih * info.iw;
    let nf = info.frame_count;
    let nchunk = nthreads.min(nf).max(1);
    let chunk = nf.div_ceil(nchunk);
    let mut partials: io::Result<Vec<(Vec<u64>, Vec<u16>)>> = Ok(Vec::new());
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for ci in 0..nchunk {
            let f0 = ci * chunk;
            let f1 = (f0 + chunk).min(nf);
            if f0 >= f1 { continue; }
            let info = info;
            handles.push(s.spawn(move || -> io::Result<(Vec<u64>, Vec<u16>)> {
                let mut sum = vec![0u64; npix];
                let mut mx = vec![0u16; npix];
                let mut file = ser::open(path)?;
                for f in f0..f1 {
                    let frame = ser::read_frame(&mut file, info, f)?;
                    for i in 0..npix {
                        sum[i] += frame[i] as u64;
                        if frame[i] > mx[i] { mx[i] = frame[i]; }
                    }
                }
                Ok((sum, mx))
            }));
        }
        let mut out = Vec::new();
        for h in handles {
            match h.join() {
                Ok(Ok(v)) => out.push(v),
                Ok(Err(e)) => {
                    partials = Err(e);
                    return;
                }
                Err(_) => {
                    partials = Err(io::Error::new(io::ErrorKind::Other, "frame reader thread panicked"));
                    return;
                }
            }
        }
        partials = Ok(out);
    });
    let partials = partials?;
    let mut sum = vec![0u64; npix];
    let mut mx = vec![0u16; npix];
    for (ps, pm) in &partials {
        for i in 0..npix {
            sum[i] += ps[i];
            if pm[i] > mx[i] { mx[i] = pm[i]; }
        }
    }
    let mean = Img { h: info.ih, w: info.iw, d: sum.iter().map(|&v| (v as f64 / nf as f64) as u16).collect() };
    let max_img = Img { h: info.ih, w: info.iw, d: mx };
    Ok((mean, max_img))
}

/// detect_bord(img, axis=1): row means of blurred image, threshold median/5.
pub fn detect_bord(img: &Img) -> (usize, usize) {
    let blur = box_blur_u16(img, 5, 5);
    let mut ymean: Vec<f64> = Vec::with_capacity(blur.h);
    for r in 0..blur.h {
        let row = blur.row(r);
        ymean.push(row.iter().map(|&v| v as f64).sum::<f64>() / blur.w as f64);
    }
    let thresh = median(&ymean) / 5.0;
    let where_sun: Vec<bool> = ymean.iter().map(|&v| v > thresh).collect();
    let lb = where_sun.iter().position(|&b| b).unwrap_or(0);
    let ub = blur.h - 1 - where_sun.iter().rev().position(|&b| b).unwrap_or(0);
    (lb, ub)
}

pub fn compute_mean_return_fit(
    _path: &str,
    _info: &SerInfo,
    mean_img: &Img,
    max_img: &Img,
    _nthreads: usize,
) -> LineFit {
    let ih = mean_img.h;
    let iw = mean_img.w;
    let (mut y1, mut y2) = detect_bord(max_img);
    let clip = ((y2 - y1) as f64 * 0.05) as usize;
    y1 = (y1 + clip).min(ih - 1);
    y2 = y2.saturating_sub(clip);
    if y2 <= y1 { y2 = ih - 1; }

    let blur_width_x = 25usize;
    let blur_width_y = (((y2 - y1) as f64 * 0.01) as usize).max(1);
    let blur = box_blur_u16(mean_img, blur_width_x, blur_width_y);
    // min_intensity = 12 + argmin(blur[:, 12:-12], axis=1)
    let mut min_intensity = vec![0usize; ih];
    let half = blur_width_x / 2;
    for r in 0..ih {
        let row = blur.row(r);
        let seg = &row[half..iw - half];
        let mut b = 0usize;
        for i in 1..seg.len() { if seg[i] < seg[b] { b = i; } }
        min_intensity[r] = half + b;
    }
    let xs: Vec<f64> = (y1..y2).map(|i| i as f64).collect();
    let ys: Vec<f64> = (y1..y2).map(|i| min_intensity[i] as f64).collect();
    let mut p = polyfit3(&xs, &ys); // ascending
    // remove outliers
    let delta: Vec<f64> = (0..xs.len()).map(|i| polyval(&p, xs[i]) - ys[i]).collect();
    let stdv = std_pop(&delta);
    let keep: Vec<bool> = if stdv > 0.0 {
        delta.iter().map(|d| (d / stdv).abs() < 3.0).collect()
    } else {
        delta.iter().map(|_| true).collect()
    };
    let xs2: Vec<f64> = xs.iter().zip(&keep).filter(|(_, k)| **k).map(|(x, _)| *x).collect();
    let ys2: Vec<f64> = ys.iter().zip(&keep).filter(|(_, k)| **k).map(|(y, _)| *y).collect();
    if xs2.len() >= 4 { p = polyfit3(&xs2, &ys2); }

    // find shift to non-blurred minimum
    let mut sharp = vec![0usize; ih];
    for r in 0..ih {
        let row = mean_img.row(r);
        let mut b = 0usize;
        for i in 1..row.len() { if row[i] < row[b] { b = i; } }
        sharp[r] = b;
    }
    let delta_sharp: Vec<f64> = (y1..y2).map(|i| polyval(&p, i as f64) - sharp[i] as f64).collect();
    // mode of round(delta_sharp, 1)
    let mut vals: Vec<i64> = delta_sharp.iter().map(|&d| (round_half_even(d * 10.0)) as i64).collect();
    vals.sort();
    let mut best_val = 0i64;
    let mut best_cnt = 0usize;
    let mut i = 0;
    while i < vals.len() {
        let mut j = i;
        while j < vals.len() && vals[j] == vals[i] { j += 1; }
        if j - i > best_cnt { best_cnt = j - i; best_val = vals[i]; }
        i = j;
    }
    let shift = best_val as f64 / 10.0;
    let tol = 5.0;
    let mask: Vec<bool> = delta_sharp.iter().map(|d| (d - shift).abs() < tol).collect();
    let xs3: Vec<f64> = (y1..y2).zip(&mask).filter(|(_, m)| **m).map(|(i, _)| i as f64).collect();
    let ys3: Vec<f64> = (y1..y2).zip(&mask).filter(|(_, m)| **m).map(|(i, _)| sharp[i] as f64).collect();
    if xs3.len() >= 4 { p = polyfit3(&xs3, &ys3); }

    let curve: Vec<f64> = (0..ih).map(|i| polyval(&p, i as f64)).collect();
    let fit = curve.iter().map(|&c| (c.floor() as i64, c - c.floor(), c)).collect();
    let mut mask_full = vec![false; ih];
    for (j, &m) in mask.iter().enumerate() { mask_full[y1 + j] = m; }
    LineFit { fit, y1, y2, poly: p, sharp, mask: mask_full, curve }
}
