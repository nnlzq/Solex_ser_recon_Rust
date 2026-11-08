//! Edge detection, ellipse fitting and circularisation
//! (ellipse_to_circle.py).

use crate::geom::{self, get_correction_matrix};
use crate::num::*;
use std::f64::consts::PI;

pub struct EllipseResult {
    pub image: Img,
    pub circle: (f64, f64, f64),
    pub ratio: f64,
    pub phi: f64,
    pub theta: f64,
    pub borders: [f64; 4],
    pub ok: bool,
    /// all canny edge pixels (row, col), for the diagnostic plot
    pub raw_x: Vec<(f64, f64)>,
    /// edge points kept for the final fit (row, col)
    pub pts_t: Vec<(f64, f64)>,
    /// ellipse centre in fit space (row, col)
    pub fit_center: (f64, f64),
    pub height: f64,
}

const NUM_REG: usize = 2;

fn argwhere(mask: &[bool], w: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for i in 0..mask.len() {
        if mask[i] { v.push((i / w, i % w)); }
    }
    v
}

pub fn downscale_local_mean(img: &ImgF, f: usize) -> ImgF {
    let h = img.h / f;
    let w = img.w / f;
    let mut o = ImgF::new(h, w);
    for r in 0..h {
        for c in 0..w {
            let mut s = 0.0;
            for dr in 0..f {
                for dc in 0..f {
                    s += img.g(r * f + dr, c * f + dc);
                }
            }
            o.s(r, c, s / (f * f) as f64);
        }
    }
    o
}

pub(crate) fn dbg_on() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("SOLEX_DEBUG").is_ok())
}

fn get_flood_image(img: &ImgF) -> ImgF {
    let h = img.h;
    let w = img.w;
    let thresh = 0.9 * img.d.iter().sum::<f64>() / (h * w) as f64;
    println!("thresh= {}", thresh);
    println!("{}", w);
    let blur_width = (h as f64 * 0.01) as usize;
    let blurred = box_blur_f(img, blur_width, blur_width);
    if dbg_on() {
        if let Ok(fp) = std::env::var("SOLEX_DUMP_FLOOD") {
            let mut s = String::with_capacity(blurred.d.len() * 24);
            for r in 0..h {
                for c in 0..w {
                    s.push_str(&format!("{:.20} ", blurred.g(r, c)));
                }
                s.push('\n');
            }
            use std::io::Write;
            if let Ok(mut fl) = std::fs::OpenOptions::new().create(true).append(true).open(fp) {
                let _ = writeln!(fl, "###CALL");
                let _ = fl.write_all(s.as_bytes());
            }
        }
    }
    let very_bright = percentile(&blurred.d, 99.0);
    println!("very bright: {}", very_bright);
    let data: Vec<f64> = blurred.d.iter().cloned().filter(|&v| v < very_bright).collect();
    if data.is_empty() {
        let mut o = blurred.clone();
        for v in o.d.iter_mut() { *v = if *v < thresh { 0.0 } else { 65000.0 }; }
        return o;
    }
    let lo = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // np.histogram(data, bins=20): edges linspace(lo,hi,21)
    let edges: Vec<f64> = (0..=20).map(|i| lo + (hi - lo) * i as f64 / 20.0).collect();
    let mut counts = vec![0.0f64; 20];
    for &v in &data {
        let idx = if hi > lo {
            (((v - lo) / (hi - lo)) * 20.0).floor() as i64
        } else { 0 };
        let idx = idx.clamp(0, 19) as usize;
        counts[idx] += 1.0;
    }
    // Polynomial.fit(bins[1:], counts, 3): domain [x0, x19] mapped to [-1,1]
    let x: Vec<f64> = edges[1..].to_vec();
    let mid = (x[0] + x[19]) / 2.0;
    let half = (x[19] - x[0]) / 2.0;
    let xs: Vec<f64> = if half > 0.0 { x.iter().map(|&v| (v - mid) / half).collect() } else { vec![0.0; 20] };
    let c = polyfit3(&xs, &counts); // ascending in x'
    // g(x) = c0 + c1*t + c2*t^2 + c3*t^3, t=(x-mid)/half. Expand into standard
    // powers of x (Python: Polynomial.fit(...).convert().coef = [d, c, b, a]):
    //   a = c3/s^3
    //   b = c2/s^2 - 3*c3*m/s^3
    //   c = c1/s - 2*c2*m/s^2 + 3*c3*m^2/s^3
    //   d = c0 - c1*m/s + c2*m^2/s^2 - c3*m^3/s^3
    let (a, b, cc) = if half > 0.0 {
        (
            c[3] / half.powi(3),
            c[2] / half.powi(2) - 3.0 * c[3] * mid / half.powi(3),
            c[1] / half - 2.0 * c[2] * mid / half.powi(2) + 3.0 * c[3] * mid * mid / half.powi(3),
        )
    } else { (0.0, 0.0, 0.0) };
    let dd = if half > 0.0 {
        c[0] - c[1] * mid / half + c[2] * mid * mid / (half * half)
            - c[3] * mid * mid * mid / (half * half * half)
    } else { c[0] };
    println!("cubic fit coeffs. : [{:.8} {:.8} {:.8} {:.8}]", dd, cc, b, a);
    let disc = 4.0 * b * b - 12.0 * a * cc;
    // Python: division by 6*a yields inf/nan when a==0; comparisons then fail and
    // thresh3 falls back to thresh. Reproduce by keeping the raw value.
    let thresh2 = if disc >= 0.0 {
        (-2.0 * b + disc.sqrt()) / (6.0 * a)
    } else {
        thresh
    };
    let mut start_i: i64 = -1;
    for i in 0..20 {
        if edges[i] <= thresh2 && thresh2 < edges[i + 1] { start_i = i as i64; }
    }
    let thresh3 = if start_i < 0 {
        thresh
    } else {
        let mut i = start_i as usize;
        while i > 0 && i < 19 {
            if counts[i - 1] < counts[i] { i -= 1; }
            else if counts[i + 1] < counts[i] { i += 1; }
            else { break; }
        }
        if i >= 1 { i -= 1; }
        edges[i]
    };
    println!("thresh2= {}", thresh2);
    println!("thresh3 =  {}", thresh3);
    if dbg_on() {
        eprintln!("DBG flood: thresh={} very_bright={} coeffs(d,c,b,a)=[{} {} {} {}] thresh2={} thresh3={}",
            thresh, very_bright, dd, cc, b, a, thresh2, thresh3);
    }
    let mut o = blurred.clone();
    for v in o.d.iter_mut() {
        *v = if *v < thresh3 { 0.0 } else { 65000.0 };
    }
    o
}

/// skimage.feature.canny (0.26), returns boolean edge mask
fn canny(img: &ImgF, sigma: f64, low: f64, high: f64) -> Vec<bool> {
    let (h, w) = (img.h, img.w);
    // smoothed = gaussian(img, mode constant) / (gaussian(ones) + eps)
    let s1 = gauss1d_const_axis(img, sigma, 0);
    let mut smoothed = gauss1d_const_axis(&s1, sigma, 1);
    let o1 = gauss1d_const_axis(&ImgF { h, w, d: vec![1.0; h * w] }, sigma, 0);
    let o2 = gauss1d_const_axis(&o1, sigma, 1);
    let eps = f64::EPSILON;
    for i in 0..smoothed.d.len() {
        smoothed.d[i] /= o2.d[i] + eps;
    }
    let isobel = sobel(&smoothed, 0);
    let jsobel = sobel(&smoothed, 1);
    // skimage: magnitude = isobel*isobel; magnitude += jsobel*jsobel; sqrt(out=magnitude)
    let mag: Vec<f64> = (0..h * w).map(|i| {
        let x = isobel.d[i];
        let mut t = x * x;
        t += jsobel.d[i] * jsobel.d[i];
        t.sqrt()
    }).collect();
    let mut out = vec![0.0f64; h * w];
    let low_t = if low == 0.0 { 1e-14 } else { low };
    for x in 1..h.saturating_sub(1) {
        for y in 1..w.saturating_sub(1) {
            let grad = mag[x * w + y];
            if grad < low_t { continue; }
            let gy = isobel.d[x * w + y];
            let gx = jsobel.d[x * w + y];
            let is_up = gy >= 0.0;
            let is_down = gy <= 0.0;
            let is_right = gx >= 0.0;
            let is_left = gx <= 0.0;
            let cond1 = (is_up && is_right) || (is_down && is_left);
            let cond2 = (is_down && is_right) || (is_up && is_left);
            let mut keep = false;
            if cond1 {
                let (wt, n11, n12, n21, n22) = if gy.abs() > gx.abs() {
                    (gx.abs() / gy.abs(),
                     mag[(x + 1) * w + y], mag[(x + 1) * w + y + 1],
                     mag[(x - 1) * w + y], mag[(x - 1) * w + y - 1])
                } else {
                    (gy.abs() / gx.abs(),
                     mag[x * w + y + 1], mag[(x + 1) * w + y + 1],
                     mag[x * w + y - 1], mag[(x - 1) * w + y - 1])
                };
                if n12 * wt + n11 * (1.0 - wt) <= grad && n22 * wt + n21 * (1.0 - wt) <= grad { keep = true; }
            }
            if cond2 {
                let (wt, n11, n12, n21, n22) = if gy.abs() < gx.abs() {
                    (gy.abs() / gx.abs(),
                     mag[x * w + y + 1], mag[(x - 1) * w + y + 1],
                     mag[x * w + y - 1], mag[(x + 1) * w + y - 1])
                } else {
                    (gx.abs() / gy.abs(),
                     mag[(x - 1) * w + y], mag[(x - 1) * w + y + 1],
                     mag[(x + 1) * w + y], mag[(x + 1) * w + y - 1])
                };
                if n12 * wt + n11 * (1.0 - wt) <= grad && n22 * wt + n21 * (1.0 - wt) <= grad { keep = true; }
            }
            if keep { out[x * w + y] = grad; }
        }
    }
    if dbg_on() {
        if let Ok(fp) = std::env::var("SOLEX_DUMP_CANNY") {
            use std::io::Write;
            if let Ok(mut fl) = std::fs::OpenOptions::new().create(true).append(true).open(fp) {
                let _ = writeln!(fl, "###CALL");
                for (name, arr) in [("smoothed", &smoothed.d), ("isobel", &isobel.d),
                                    ("jsobel", &jsobel.d), ("mag", &mag), ("nms", &out)] {
                    let _ = writeln!(fl, "@{}", name);
                    for r in 0..h {
                        for c in 0..w {
                            let _ = write!(fl, "{:.20} ", arr[r * w + c]);
                        }
                        let _ = writeln!(fl);
                    }
                }
            }
        }
    }
    // hysteresis
    let low_masked: Vec<f64> = out.iter().map(|&m| if m > low_t { m } else { 0.0 }).collect();
    let low_mask: Vec<bool> = low_masked.iter().map(|&m| m > 0.0).collect();
    let (lab, n) = label8(&low_mask, h, w);
    if n == 0 { return vec![false; h * w]; }
    let mut keep_lab = vec![false; n + 1];
    for i in 0..low_masked.len() {
        if low_mask[i] && low_masked[i] >= high { keep_lab[lab[i] as usize] = true; }
    }
    (0..h * w).map(|i| lab[i] > 0 && keep_lab[lab[i] as usize]).collect()
}

/// returns Some((filtered X points, all edge pixels), in downscaled coords),
/// or None if no edges
fn get_edge_list(img: &ImgF, sigma: f64) -> Option<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
    if sigma <= 0.0 { return None; }
    let (h, w) = (img.h, img.w);
    let low = median(&box_blur_f(img, 5, 5).d) / 10.0;
    let high = low * 1.5;
    println!("using thresholds: {} {}", low, high);
    let flooded = get_flood_image(img);
    let edges = canny(&flooded, sigma, low, high);
    let raw_x = argwhere(&edges, w);
    let (lab, nf) = label8(&edges, h, w);
    if nf == 0 { return get_edge_list(img, sigma - 0.5); }
    let mut region_sizes: Vec<f64> = vec![-1.0];
    for i in 1..=nf {
        region_sizes.push(lab.iter().filter(|&&l| l == i as i32).count() as f64);
    }
    let mut sorted = region_sizes.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let top: Vec<f64> = sorted.iter().cloned().take(nf.min(NUM_REG)).collect();
    let mut filt = vec![false; h * w];
    for &label in &top {
        let idx = region_sizes.iter().position(|&v| v == label).unwrap();
        if idx == 0 { continue; }
        for i in 0..lab.len() { if lab[i] == idx as i32 { filt[i] = true; } }
    }
    let x = argwhere(&filt, w);
    if x.is_empty() { return get_edge_list(img, sigma - 0.5); }
    let xpts: Vec<(f64, f64)> = x.iter().map(|&(r, c)| (r as f64, c as f64)).collect();
    let hull = hull_indices(&xpts);
    let mut xd = vec![false; h * w];
    for &vi in &hull { let (r, c) = x[vi]; xd[r * w + c] = true; }
    let mut touched = vec![false; nf + 1];
    for i in 0..lab.len() { if xd[i] && lab[i] > 0 { touched[lab[i] as usize] = true; } }
    let mut filt2 = vec![false; h * w];
    for &label in &top {
        let idx = region_sizes.iter().position(|&v| v == label).unwrap();
        if idx > 0 && touched[idx] {
            for i in 0..lab.len() { if lab[i] == idx as i32 { filt2[i] = true; } }
        }
    }
    // crop bounds from x (pre-refilter)
    let x_min = x.iter().map(|p| p.0).min().unwrap() as f64;
    let x_max = x.iter().map(|p| p.0).max().unwrap() as f64;
    let dx = x_max - x_min;
    let crop = 0.017;
    let r0 = (x_min + dx * crop) as usize;
    let r1 = (x_max - dx * crop) as usize;
    for i in 0..(h * w) {
        if filt2[i] {
            let r = i / w;
            if r < r0 || r >= r1 { filt2[i] = false; }
        }
    }
    let x2 = argwhere(&filt2, w);
    if dbg_on() {
        eprintln!("DBG edges: low={} high={} rawX={} nf={} sizes={:?} X={} hull={} X2={}",
            low, high, raw_x.len(), nf, &region_sizes[1..], x.len(), hull.len(), x2.len());
        if let Ok(fp) = std::env::var("SOLEX_DUMP_RAWX") {
            let s: String = raw_x.iter().map(|p| format!("{} {}\n", p.0, p.1)).collect();
            let _ = std::fs::write(fp, s);
        }
    }
    if x2.is_empty() { return None; }
    let raw_pts: Vec<(f64, f64)> = raw_x.iter().map(|&(r, c)| (r as f64, c as f64)).collect();
    Some((x2.iter().map(|&(r, c)| (r as f64, c as f64)).collect(), raw_pts))
}

/// Halir-Flatter least-squares ellipse fit. points are (x,y) pairs.
/// Returns conic coefficients [a,b,c,d,e,f] for ax²+bxy+cy²+dx+ey+f=0.
pub fn lsq_ellipse_fit(points: &[(f64, f64)]) -> Option<[f64; 6]> {
    if points.len() < 5 { return None; }
    let mut s1 = vec![vec![0.0f64; 3]; 3];
    let mut s2 = vec![vec![0.0f64; 3]; 3];
    let mut s3 = vec![vec![0.0f64; 3]; 3];
    for &(x, y) in points {
        let d1 = [x * x, x * y, y * y];
        let d2 = [x, y, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                s1[i][j] += d1[i] * d1[j];
                s2[i][j] += d1[i] * d2[j];
                s3[i][j] += d2[i] * d2[j];
            }
        }
    }
    let s3i = inv_n(&s3)?;
    // t = S1 - S2 @ S3i @ S2^T
    let s2t = vec![
        vec![s2[0][0], s2[1][0], s2[2][0]],
        vec![s2[0][1], s2[1][1], s2[2][1]],
        vec![s2[0][2], s2[1][2], s2[2][2]],
    ];
    let a = matmul(&s2, &s3i);
    let b = matmul(&a, &s2t);
    let t: Vec<Vec<f64>> = (0..3).map(|i| (0..3).map(|j| s1[i][j] - b[i][j]).collect()).collect();
    // C1^-1 = [[0,0,0.5],[0,-1,0],[0.5,0,0]]
    let c1i = vec![vec![0.0, 0.0, 0.5], vec![0.0, -1.0, 0.0], vec![0.5, 0.0, 0.0]];
    let m = matmul(&c1i, &t);
    let mm = [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ];
    let vecs = eig3_vecs(mm);
    if dbg_on() {
        eprintln!("DBG M = [[{:.6} {:.6} {:.6}][{:.6} {:.6} {:.6}][{:.6} {:.6} {:.6}]]",
            mm[0][0], mm[0][1], mm[0][2], mm[1][0], mm[1][1], mm[1][2], mm[2][0], mm[2][1], mm[2][2]);
        for v in &vecs {
            eprintln!("DBG eigvec = [{:.17} {:.17} {:.17}] 4ac-b2={:.6e}", v[0], v[1], v[2], 4.0 * v[0] * v[2] - v[1] * v[1]);
        }
    }
    for v in vecs {
        let (a, b, c) = (v[0], v[1], v[2]);
        if 4.0 * a * c - b * b > 0.0 {
            // [d,e,f] = -S3^-1 @ S2^T @ [a b c]
            let sv = matvec(&s2t, &[a, b, c]);
            let neg = [-sv[0], -sv[1], -sv[2]];
            let def = matvec(&s3i, &neg);
            return Some([a, b, c, def[0], def[1], def[2]]);
        }
    }
    None
}

/// center, width, height, phi
pub fn as_parameters(coef: &[f64; 6]) -> ((f64, f64), f64, f64, f64) {
    let a = coef[0];
    let b = coef[1] / 2.0;
    let c = coef[2];
    let d = coef[3] / 2.0;
    let f = coef[4] / 2.0;
    let g = coef[5];
    let den = b * b - a * c;
    let x0 = (c * d - b * f) / den;
    let y0 = (a * f - b * d) / den;
    let numerator = 2.0 * (a * f * f + c * d * d + g * b * b - 2.0 * b * d * f - a * c * g);
    let sq = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let denominator1 = den * (sq - (c + a));
    let denominator2 = den * (-sq - (c + a));
    let height = (numerator / denominator1).sqrt();
    let width = (numerator / denominator2).sqrt();
    let phi = if b == 0.0 && a > c { 0.0 }
        else if b == 0.0 && a < c { PI / 2.0 }
        else if b != 0.0 && a > c { 0.5 * (2.0 * b / (a - c)).atan() }
        else if b != 0.0 && a < c { 0.5 * (PI + (2.0 * b / (a - c)).atan()) }
        else { 0.0 };
    ((x0, y0), width, height, phi)
}

fn dofit(points: &[(f64, f64)]) -> Option<((f64, f64), f64, f64, f64)> {
    let coef = lsq_ellipse_fit(points)?;
    Some(as_parameters(&coef))
}

pub struct TwoStep {
    pub center: (f64, f64),
    pub height: f64,
    pub phi: f64,
    pub ratio: f64,
    pub points_t: Vec<(f64, f64)>,
}

fn threshold_points(points: &[(f64, f64)], center: (f64, f64), height: f64, phi: f64, ratio_hw: f64) -> Vec<(f64, f64)> {
    let (mat, _) = get_correction_matrix(phi, ratio_hw);
    let values: Vec<f64> = points.iter().map(|p| {
        let (x, y) = mat.apply(p.0 - center.0, p.1 - center.1);
        (x * height).hypot(y * height) - 1.0
    }).collect();
    let maxv = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    points.iter().zip(&values).filter(|(_, &v)| v > -maxv).map(|(p, _)| *p).collect()
}

pub fn two_step(points: &[(f64, f64)]) -> Option<TwoStep> {
    let (center, width, height, phi) = dofit(points)?;
    let pts_t = threshold_points(points, center, height, phi, height / width);
    let pts_t = if pts_t.len() >= 5 { pts_t } else { points.to_vec() };
    let (center, width, height, phi) = dofit(&pts_t)?;
    let (_mat, _) = get_correction_matrix(phi, height / width);
    let _ = threshold_points(&pts_t, center, height, phi, height / width);
    let mut ratio = width / height;
    let mut phi = phi;
    let mut height = height;
    for _ in 0..2 {
        if phi > PI / 4.0 {
            phi -= PI / 2.0;
            ratio = 1.0 / ratio;
            height /= ratio;
        }
        if phi < -PI / 4.0 {
            phi += PI / 2.0;
            ratio = 1.0 / ratio;
            height /= ratio;
        }
    }
    Some(TwoStep { center, height, phi, ratio, points_t: pts_t })
}

/// Main entry: circularise a disk image (u16).
pub fn ellipse_to_circle(disk: &Img) -> EllipseResult {
    let image = ImgF { h: disk.h, w: disk.w, d: disk.d.iter().map(|&v| v as f64 / 65536.0).collect() };
    let factor = 4usize;
    let down = downscale_local_mean(&image, factor);
    match get_edge_list(&down, 2.0) {
        Some((x, raw_x)) => {
            let pts: Vec<(f64, f64)> = x.iter().map(|(r, c)| (r * factor as f64, c * factor as f64)).collect();
            let raw_pts: Vec<(f64, f64)> = raw_x.iter().map(|(r, c)| (r * factor as f64, c * factor as f64)).collect();
            if dbg_on() {
                if let Ok(fp) = std::env::var("SOLEX_DUMP_PTS") {
                    let s: String = pts.iter().map(|p| format!("{:.17} {:.17}\n", p.0, p.1)).collect();
                    let _ = std::fs::write(fp, s);
                }
            }
            match two_step(&pts) {
                Some(ts) => {
                    let center = (ts.center.1, ts.center.0); // swap
                    let (fix_img, new_circle, mat3) = geom::correct_image(&image, ts.phi, ts.ratio, center, ts.height);
                    let (_m, theta) = get_correction_matrix(ts.phi, ts.ratio);
                    let borders = geom::compute_borders(&ts.points_t, &mat3);
                    println!("unrotation angle theta = {:.3} degrees", theta.to_degrees());
                    println!("Y/X ratio : {:.3}", ts.ratio);
                    println!("sun borders found:[{} {} {} {}]", borders[0], borders[1], borders[2], borders[3]);
                    EllipseResult { image: fix_img, circle: new_circle, ratio: ts.ratio, phi: ts.phi, theta, borders, ok: true,
                        raw_x: raw_pts, pts_t: ts.points_t.clone(), fit_center: ts.center, height: ts.height }
                }
                None => failed_result(&image),
            }
        }
        None => failed_result(&image),
    }
}

fn failed_result(image: &ImgF) -> EllipseResult {
    let (fix_img, _nc, _m3) = geom::correct_image(image, 0.0, 1.0, (-1.0, -1.0), -1.0);
    EllipseResult { image: fix_img, circle: (-1.0, -1.0, -1.0), ratio: 1.0, phi: 0.0, theta: 0.0, borders: [0.0, 0.0, 0.0, 0.0], ok: false,
        raw_x: Vec::new(), pts_t: Vec::new(), fit_center: (0.0, 0.0), height: 1.0 }
}
