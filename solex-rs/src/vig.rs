//! De-vignetting (solex_util.removeVignette).

use crate::num::*;

pub fn remove_vignette(frame: &Img, circle: (f64, f64, f64)) -> ImgF {
    let (cx, cy, r) = circle;
    let h = frame.h;
    let w = frame.w;
    // y_arr = percentile 85 per column; y_arr2 = per row
    let y_arr: Vec<f64> = (0..w).map(|c| {
        let col: Vec<f64> = (0..h).map(|rr| frame.g(rr, c) as f64).collect();
        percentile(&col, 85.0)
    }).collect();
    let y_arr2: Vec<f64> = (0..h).map(|rr| {
        let row: Vec<f64> = frame.row(rr).iter().map(|&v| v as f64).collect();
        percentile(&row, 85.0)
    }).collect();

    let shrink = 65i64;
    let cxi = cx.trunc() as i64;
    let cyi = cy.trunc() as i64;
    let start1 = 0i64.max((cx - r + shrink as f64).trunc() as i64) as usize;
    let end1 = (y_arr.len() as i64).min((cx + r + 1.0 - shrink as f64).trunc() as i64).max(0) as usize;
    let start2 = 0i64.max((cy - r + shrink as f64).trunc() as i64) as usize;
    let end2 = (y_arr2.len() as i64).min((cy + r + 1.0 - shrink as f64).trunc() as i64).max(0) as usize;
    if end1 <= start1 || end2 <= start2 {
        return frame.to_f();
    }
    let y1 = &y_arr[start1..end1];
    let y2 = &y_arr2[start2..end2];
    let len1 = y1.len() as i64;
    let len2 = y2.len() as i64;
    let x1: Vec<f64> = (0..len1).map(|i| (i + start1 as i64 - cxi) as f64).collect();
    let x2: Vec<f64> = (0..len2).map(|i| (i + start2 as i64 - cyi) as f64).collect();
    if len1 < 20 || len2 < 20 {
        return frame.to_f();
    }
    let scale_pix = ((len1 as f64 / 2.75).min(len2 as f64 / 2.75) as i64) / 2 * 2 - 1;
    let win = (801usize).min(scale_pix.max(1) as usize);
    let trend1 = savgol(y1, win, 3);
    let trend2 = savgol(y2, win, 3);

    let mm = x1[0].min(x2[0]) as i64;
    let maxx = x1[x1.len() - 1].max(x2[x2.len() - 1]) as i64;
    let dw = (maxx - mm + 1) as usize;
    let mut dest1 = vec![f64::NAN; dw];
    let mut dest2 = vec![f64::NAN; dw];
    let o1 = (x1[0] as i64 - mm) as usize;
    let e1 = (x1[x1.len() - 1] as i64 - mm + 1) as usize;
    for i in o1..e1.min(dw) { dest1[i] = trend1[i - o1]; }
    let o2 = (x2[0] as i64 - mm) as usize;
    let e2 = (x2[x2.len() - 1] as i64 - mm + 1) as usize;
    for i in o2..e2.min(dw) { dest2[i] = trend2[i - o2]; }

    let ratio: Vec<f64> = (0..dw).map(|i| {
        if dest1[i] == 0.0 || dest2[i] == 0.0 || !dest1[i].is_finite() || !dest2[i].is_finite() {
            f64::NAN
        } else { dest1[i] / dest2[i] }
    }).collect();

    let mut corr = vec![f64::NAN; h];
    for i in 0..dw {
        let idx = (mm + i as i64 + cyi) as i64;
        // Python negative-index wrap
        let idx = if idx < 0 { idx + h as i64 } else { idx };
        if idx >= 0 && idx < h as i64 { corr[idx as usize] = ratio[i]; }
    }
    let mask: Vec<bool> = corr.iter().map(|v| v.is_nan()).collect();
    let any_nan = mask.iter().any(|&b| b);
    let all_nan = mask.iter().all(|&b| b);
    if any_nan && !all_nan {
        // last valid index at or before i (-1 if none)
        let mut idx_ff = vec![-1i64; h];
        let mut last = -1i64;
        for i in 0..h {
            if !mask[i] { last = i as i64; }
            idx_ff[i] = last;
        }
        let mut idx_bf = vec![h as i64; h];
        let mut next = h as i64;
        for i in (0..h).rev() {
            if !mask[i] { next = i as i64; }
            idx_bf[i] = next;
        }
        for i in 0..h {
            let src = if idx_ff[i] >= 0 { idx_ff[i] } else { idx_bf[i].min(h as i64 - 1) };
            corr[i] = corr[src as usize];
        }
    }
    let sigma = (2i64.max((150i64).min(scale_pix / 4))) as f64;
    let corr = gauss1d_reflect(&corr, sigma);

    let mut out = ImgF::new(h, w);
    for rr in 0..h {
        for c in 0..w {
            out.s(rr, c, frame.g(rr, c) as f64 * corr[rr]);
        }
    }
    out
}
