//! Transversalium (horizontal banding) correction (solex_util.correct_transversalium2).

use crate::num::*;
use std::f64::consts::PI;

fn tukey_taper(n: usize, a: f64) -> Vec<f64> {
    if n == 0 { return Vec::new(); }
    let rise_end = a * n as f64 / 2.0;
    (0..n).map(|i| {
        let x = i as f64;
        if x < rise_end {
            0.5 * (1.0 - (2.0 * PI * x / (a * n as f64)).cos())
        } else if x > (n as f64 - rise_end) {
            let y = n as f64 - x;
            0.5 * (1.0 - (2.0 * PI * y / (a * n as f64)).cos())
        } else { 1.0 }
    }).collect()
}

/// img: f64 (values 0..65535). circle=(cx,cy,r), borders=[minX,minY,maxX,maxY].
pub fn correct_transversalium2(img: &ImgF, circle: (f64, f64, f64), borders: &[f64; 4], trans_strength: i64, stubborn: bool, plot: Option<String>) -> Img {
    let (cx, cy, r) = circle;
    let h = img.h;
    let w = img.w;
    let y1 = (cy - r).max(borders[1]).ceil() as i64;
    let y2 = (cy + r).min(borders[3]).floor() as i64;
    let y1 = y1.max(0) as usize;
    let y2 = (y2 as usize).min(h - 1);
    let n = (y2 as i64 - (y1 as i64 + 1)).max(0) as usize;

    let mut y_ratios_r: Vec<f64> = vec![0.0];
    if n > 0 {
        let ys: Vec<i64> = ((y1 + 1)..y2).map(|i| i as i64).collect();
        let dxs: Vec<i64> = ys.iter().map(|&y| {
            (r * r - (y as f64 - cy) * (y as f64 - cy)).max(0.0).sqrt().floor() as i64
        }).collect();
        let mut x1s: Vec<i64> = ys.iter().zip(&dxs).map(|(&y, &dx)| {
            let _ = y;
            ((cx - dx as f64).max(borders[0])).trunc() as i64
        }).collect();
        let mut x2s: Vec<i64> = ys.iter().zip(&dxs).map(|(&y, &dx)| {
            let _ = y;
            ((cx + dx as f64).min(borders[2])).trunc() as i64
        }).collect();
        for v in x1s.iter_mut() { *v = (*v).max(0); }
        for v in x2s.iter_mut() { *v = (*v).min(w as i64 - 1); }
        let lengths: Vec<i64> = x1s.iter().zip(&x2s).map(|(&a, &b)| (b - a).max(0)).collect();
        let max_len = lengths.iter().cloned().max().unwrap_or(0) as usize;
        if max_len > 1 {
            // padded log ratios
            let mut padded: Vec<Vec<f64>> = Vec::with_capacity(n);
            for i in 0..n {
                let y = ys[i] as usize;
                let x1 = x1s[i] as usize;
                let len = lengths[i] as usize;
                let mut row = vec![f64::NAN; len];
                for k in 0..len {
                    let s0 = img.g(y - 1, x1 + k);
                    let s1 = img.g(y, x1 + k);
                    let v = (s1 / s0).ln();
                    if v.is_finite() { row[k] = v; }
                }
                padded.push(row);
            }
            y_ratios_r = vec![0.0];
            for row in &padded {
                let vals: Vec<f64> = row.iter().cloned().filter(|v| v.is_finite()).collect();
                let m = if vals.is_empty() {
                    0.0
                } else {
                    let med = median(&vals);
                    let devs: Vec<f64> = vals.iter().map(|v| (v - med).abs()).collect();
                    let mad = median(&devs);
                    if mad > 0.0 {
                        let kept: Vec<f64> = vals.iter().cloned().filter(|v| (v - med).abs() / mad < 2.0).collect();
                        if kept.is_empty() { 0.0 } else { kept.iter().sum::<f64>() / kept.len() as f64 }
                    } else {
                        vals.iter().sum::<f64>() / vals.len() as f64
                    }
                };
                y_ratios_r.push(m);
            }
        } else {
            y_ratios_r = vec![0.0; n + 1];
        }
    }

    let len = y_ratios_r.len();
    let window_i = (trans_strength).min(len as i64 / 2 * 2 - 1);
    let trend = if window_i >= 5 {
        savgol(&y_ratios_r, window_i as usize, 3)
    } else {
        y_ratios_r.clone()
    };
    let mut detrended: Vec<f64> = y_ratios_r.iter().zip(&trend).map(|(a, b)| a - b).collect();
    let mean = detrended.iter().sum::<f64>() / detrended.len() as f64;
    for v in detrended.iter_mut() { *v -= mean; }
    let mut correction = vec![0.0; len];
    let mut acc = 0.0;
    for i in 0..len {
        acc += -detrended[i];
        correction[i] = acc.exp();
    }

    if stubborn {
        // spurious line removal path
        let mut c = vec![0.0f64; h];
        for i in y1..y2.min(h) {
            if i - y1 < correction.len() { c[i] = correction[i - y1].ln(); }
        }
        let thr = std_pop(&correction.iter().map(|v| v.ln()).collect::<Vec<f64>>()) * 2.5;
        let mut spurious = vec![false; h];
        for i in 0..h { spurious[i] = c[i].abs() > thr; }
        let rolled: Vec<bool> = (0..h).map(|i| spurious[(i + 1) % h] || spurious[(i + h - 1) % h]).collect();
        for i in 0..h { spurious[i] = spurious[i] || rolled[i]; }
        let filtered = apply_lin_filter(img, 101, 5, &spurious, y1, y2, circle);
        let mut out = Img::new(h, w);
        for i in 0..filtered.d.len() {
            let v = filtered.d[i];
            out.d[i] = if v.is_nan() { 0 } else { v.min(65535.0) as u16 };
        }
        return out;
    }

    let nn = correction.len();
    let taper = tukey_taper(nn, 0.05);
    let correction_t: Vec<f64> = correction.iter().zip(&taper).map(|(&c, &t)| 1.0 + (c - 1.0) * t).collect();

    let mut c = vec![1.0f64; h];
    for i in y1..y2.min(h) {
        if i - y1 < correction_t.len() { c[i] = correction_t[i - y1]; }
    }
    if let Some(p) = plot.as_deref() {
        let _ = crate::plot::transversalium(p, &c);
    }

    let mut out = Img::new(h, w);
    for r in 0..h {
        for col in 0..w {
            let v = img.g(r, col) * c[r];
            out.s(r, col, if v > 65535.0 { 65535 } else { v as u16 });
        }
    }
    out
}

fn apply_lin_filter(img: &ImgF, linlen: usize, half_width: usize, spurious: &[bool], y1: usize, y2: usize, circle: (f64, f64, f64)) -> ImgF {
    let h = img.h;
    let w = img.w;
    // rolling sums of spurious flag
    let mut cumsum = vec![0i64; h];
    let mut acc = 0i64;
    for i in 0..h { acc += if spurious[i] { 1 } else { 0 }; cumsum[i] = acc; }
    let hw = half_width;
    let mut delayed = vec![0i64; h];
    for i in 0..h { delayed[i] = if i >= hw { cumsum[i - hw] } else { 0 }; }
    for i in 0..hw { delayed[i] = 0; }
    let mut advanced = vec![0i64; h];
    for i in 0..h { advanced[i] = if i + hw < h { cumsum[i + hw] } else { cumsum[h - 1] }; }
    if hw > 0 && h > hw {
        let src = advanced[h - hw - 1];
        for i in (h - hw)..h { advanced[i] = src; }
    }
    let rolling: Vec<f64> = (0..h).map(|i| {
        (2 * hw) as f64 - (advanced[i] - delayed[i] - if spurious[i] { 1 } else { 0 }) as f64
    }).collect();

    let logimg: ImgF = ImgF { h, w, d: img.d.iter().map(|&v| v.ln()).collect() };

    // filtered: log(img) with spurious rows zeroed
    let mut filtered = logimg.clone();
    for i in 0..h {
        if spurious[i] {
            for col in 0..w { filtered.s(i, col, 0.0); }
        }
    }
    // kernel: ones((2hw+1, linlen)) with center row zero
    let mut kernel = vec![vec![1.0f64; linlen]; 2 * hw + 1];
    for col in 0..linlen { kernel[hw][col] = 0.0; }
    let result = filter2d(&filtered, &kernel);
    let mut result2 = result.clone();
    for i in 0..h {
        let denom = (linlen as f64) * rolling[i];
        for col in 0..w {
            result2.s(i, col, if denom == 0.0 { f64::NAN } else { result.g(i, col) / denom });
        }
    }
    let _ = result2;

    // filt2: forward/backward fill of spurious rows
    let mut filt2 = logimg.clone();
    let valid_exists = (0..h).any(|i| !spurious[i]);
    if valid_exists {
        for i in 0..h {
            if spurious[i] {
                // nearest valid at or before i
                let mut prev: i64 = -1;
                for j in (0..=i).rev() { if !spurious[j] { prev = j as i64; break; } }
                if prev >= 0 {
                    for col in 0..w { filt2.s(i, col, logimg.g(prev as usize, col) / 2.0); }
                } else {
                    for col in 0..w { filt2.s(i, col, 0.0); }
                }
            }
        }
        for i in 0..h {
            if spurious[i] {
                let mut next: i64 = -1;
                for j in i..h { if !spurious[j] { next = j as i64; break; } }
                if next >= 0 {
                    for col in 0..w { filt2.s(i, col, filt2.g(i, col) + logimg.g(next as usize, col) / 2.0); }
                }
            }
        }
    } else {
        for v in filt2.d.iter_mut() { *v = 0.0; }
    }
    let ksum = (2 * hw * linlen) as f64;
    let knorm: Vec<Vec<f64>> = kernel.iter().map(|row| row.iter().map(|&v| v / ksum).collect()).collect();
    let result3 = filter2d(&filt2, &knorm);

    let klin: Vec<Vec<f64>> = vec![vec![1.0 / linlen as f64; linlen]];
    let result4 = filter2d(&logimg, &klin);

    let mut delta = ImgF::new(h, w);
    for i in 0..(h * w) { delta.d[i] = result4.d[i] - result3.d[i]; }

    let taper = tukey_taper(y2.saturating_sub(y1), 0.05);
    let mut cvec = vec![0.0f64; h];
    for i in y1..y2.min(h) {
        if i - y1 < taper.len() { cvec[i] = taper[i - y1]; }
    }
    let delta = fix_edge_effect(delta, circle, 101 + 20);

    let mut out = ImgF::new(h, w);
    for r in 0..h {
        for col in 0..w {
            out.s(r, col, img.g(r, col) * (-delta.g(r, col) * cvec[r]).exp());
        }
    }
    out
}

fn fix_edge_effect(mut mult: ImgF, circle: (f64, f64, f64), linlen: usize) -> ImgF {
    let (cx, cy, r) = circle;
    let h = mult.h;
    let w = mult.w;
    let y1 = (cy - r).max(0.0).ceil() as usize;
    let y2 = (cy + r).min(h as f64 - 1.0).floor() as usize;
    let halflen = linlen / 2;
    if y2 <= y1 {
        for i in 0..y1.min(h) { for col in 0..w { mult.s(i, col, 0.0); } }
        for i in (y2 + 1)..h { for col in 0..w { mult.s(i, col, 0.0); } }
        return mult;
    }
    for i in 0..y1 { for col in 0..w { mult.s(i, col, 0.0); } }
    for i in (y2 + 1)..h { for col in 0..w { mult.s(i, col, 0.0); } }
    for y in y1..=y2 {
        let dx = (r * r - (y as f64 - cy) * (y as f64 - cy)).max(0.0).sqrt();
        let x1 = (cx - dx).max(0.0) as usize;
        let x2 = (cx + dx).min(w as f64 - 1.0) as usize;
        let valid = (x2 as i64 - x1 as i64) >= linlen as i64;
        for col in 0..x1 { mult.s(y, col, 0.0); }
        for col in x2..w { mult.s(y, col, 0.0); }
        if valid {
            if x1 > 0 && x1 + halflen < w {
                let v = mult.g(y, x1 + halflen);
                for col in x1..(x1 + halflen).min(w) { mult.s(y, col, v); }
            }
            if x2 < w - 1 && x2 > halflen {
                let v = mult.g(y, x2 - halflen - 1);
                for col in (x2 - halflen).max(0)..x2 { mult.s(y, col, v); }
            }
        }
    }
    mult
}
