//! Post-processing: options, logging, brightness rescale, crop-to-center,
//! image_process (CLAHE + protus + PNG/FITS output). Port of solex_util.image_process
//! and Solex_recon.single_image_process helpers.

use crate::clahe::clahe;
use crate::num::*;
use crate::pngio;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

#[derive(Clone)]
pub struct Options {
    pub shift: Vec<i64>,
    pub shift_requested: Vec<i64>,
    pub ellipse_fit_shift: i64,
    pub flag_display: bool,
    pub ratio_fixe: Option<f64>,
    pub slant_fix: Option<f64>, // degrees
    pub save_fit: bool,
    pub clahe_only: bool,
    pub protus_only: bool,
    pub disk_display: bool,
    pub delta_radius: i64,
    pub crop_width_square: bool,
    pub transversalium: bool,
    pub stubborn: bool,
    pub trans_strength: i64,
    pub img_rotate: i64,
    pub flip_x: bool,
    pub fixed_width: Option<i64>,
    pub output_dir: String,
    pub de_vignette: bool,
    pub nolog: bool,
    pub nthreads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            shift: vec![0],
            shift_requested: vec![0],
            ellipse_fit_shift: 10,
            flag_display: false,
            ratio_fixe: None,
            slant_fix: None,
            save_fit: false,
            clahe_only: false,
            protus_only: false,
            disk_display: true,
            delta_radius: 0,
            crop_width_square: false,
            transversalium: true,
            stubborn: false,
            trans_strength: 301,
            img_rotate: 0,
            flip_x: false,
            fixed_width: None,
            output_dir: String::new(),
            de_vignette: false,
            nolog: false,
            nthreads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        }
    }
}

/// solex_util.output_path
pub fn output_path(path: &str, opts: &Options) -> String {
    if opts.output_dir.trim().is_empty() {
        return path.to_string();
    }
    let base = Path::new(path).file_name().unwrap();
    Path::new(&opts.output_dir).join(base).to_string_lossy().into_owned()
}

fn now_str() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{} (unix)", secs)
}

pub fn clearlog(logpath: &str, opts: &Options) {
    let p = output_path(logpath, opts);
    let _ = std::fs::write(&p, format!("start time: {}\n", now_str()));
}

pub fn logme(logpath: &str, opts: &Options, s: &str) {
    if opts.nolog { return; }
    let p = output_path(logpath, opts);
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&p) {
        let _ = writeln!(f, "{}", s);
    }
}

pub fn write_complete(logpath: &str, opts: &Options) {
    let p = output_path(logpath, opts);
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&p) {
        let _ = writeln!(f, "end time: {}", now_str());
    }
}

/// rescale_brightness(img, lo, hi): 65535*(v-lo)/(hi-lo) clamped, truncate to u16.
pub fn rescale(frame: &Img, lo: f64, hi: f64) -> Img {
    // Python asserts sat >= hi > lo; with degenerate bounds it crashes.
    // Graceful equivalent: emit zeros.
    if !(hi > lo) {
        return Img::new(frame.h, frame.w);
    }
    let scale = 65535.0 / (hi - lo);
    let mut o = Img::new(frame.h, frame.w);
    for i in 0..frame.d.len() {
        let v = (frame.d[i] as f64 - lo) * scale;
        o.d[i] = if v <= 0.0 { 0 } else if v >= 65535.0 { 65535 } else { v as u16 };
    }
    o
}

/// cv2.circle(thickness=-1) equivalent: per-row dx = floor(sqrt(r^2-dy^2)).
fn fill_circle(img: &mut Img, x0: i64, y0: i64, r: i64, val: u16) {
    for dy in -r..=r {
        let y = y0 + dy;
        if y < 0 || y >= img.h as i64 { continue; }
        let dx = ((r * r - dy * dy) as f64).sqrt().floor() as i64;
        let xa = (x0 - dx).max(0);
        let xb = (x0 + dx).min(img.w as i64 - 1);
        for x in xa..=xb {
            img.s(y as usize, x as usize, val);
        }
    }
}

/// get_stable_circle_center (Solex_recon.single_image_process)
pub fn get_stable_circle_center(img: &Img, cercle0: (f64, f64, f64)) -> (i64, i64) {
    if !(cercle0.0 == -1.0 && cercle0.1 == -1.0 && cercle0.2 == -1.0) {
        return (cercle0.0.trunc() as i64, cercle0.1.trunc() as i64);
    }
    let mn = img.min() as f64;
    let mx = img.max() as f64;
    if mx > mn {
        let range = mx - mn;
        let mut colsum = vec![0.0f64; img.w];
        let mut rowsum = vec![0.0f64; img.h];
        let mut total = 0.0f64;
        for r in 0..img.h {
            let mut rs = 0.0;
            for c in 0..img.w {
                let v = (img.g(r, c) as f64 - mn) / range;
                rs += v;
                colsum[c] += v;
            }
            rowsum[r] = rs;
            total += rs;
        }
        if total > 0.0 && total.is_finite() {
            let cx: f64 = (0..img.w).map(|c| c as f64 * colsum[c]).sum::<f64>() / total;
            let cy: f64 = (0..img.h).map(|r| r as f64 * rowsum[r]).sum::<f64>() / total;
            if cx.is_finite() && cy.is_finite() {
                return (round_half_even(cx) as i64, round_half_even(cy) as i64);
            }
        }
    }
    (img.w as i64 / 2, img.h as i64 / 2)
}

/// center_image_around_point: translate onto a bg-filled canvas (no crop).
pub fn center_image_around_point(img: &Img, center: (i64, i64), size: (i64, i64)) -> Img {
    let (tw, th) = (size.0 as usize, size.1 as usize);
    let h = img.h;
    let w = img.w;
    // bg = mean(img[0:10,0:10]) truncated to u16
    let rh = h.min(10);
    let rw = w.min(10);
    let mut s = 0u64;
    for r in 0..rh {
        for c in 0..rw {
            s += img.g(r, c) as u64;
        }
    }
    let bg = (s as f64 / (rh * rw) as f64) as u16;
    let mut o = Img { h: th, w: tw, d: vec![bg; th * tw] };
    let new_cx = (tw / 2) as i64;
    let new_cy = (th / 2) as i64;
    let dx = new_cx - center.0;
    let dy = new_cy - center.1;
    let sx0 = (-dx).max(0);
    let sx1 = (w as i64).min(tw as i64 - dx);
    let sy0 = (-dy).max(0);
    let sy1 = (h as i64).min(th as i64 - dy);
    let dxs = dx.max(0);
    let dys = dy.max(0);
    if sx1 > sx0 && sy1 > sy0 {
        for r in sy0..sy1 {
            for c in sx0..sx1 {
                let v = img.g(r as usize, c as usize);
                o.s((dys + (r - sy0)) as usize, (dxs + (c - sx0)) as usize, v);
            }
        }
    }
    o
}

/// Images produced by `image_process` (kept in memory for the GUI preview).
pub struct Processed {
    pub clahe: Img,
    pub high_contrast: Img,
    pub protus: Img,
}

/// solex_util.image_process: CLAHE + brightness stretch + protus disk + save.
pub fn image_process(frame: &Img, cercle: (f64, f64, f64), opts: &Options, basefich: &str, logbase: &str) -> Processed {
    let _ = logbase;
    let cl1 = clahe(frame, 0.8);
    let bright = frame.max() as f64;
    let cl1_f: Vec<f64> = cl1.d.iter().map(|&v| v as f64).collect();
    let dark_clahe = percentile(&cl1_f, 10.0);
    let bright_clahe = cl1.max() as f64;

    let frame_raw = frame.clone();
    let frame_hc = rescale(frame, bright * 0.25, bright);
    let mut frame_protus = rescale(frame, 0.0, bright * 0.18);
    let cc = rescale(&cl1, dark_clahe, bright_clahe);

    if !(cercle.0 == -1.0 && cercle.1 == -1.0 && cercle.2 == -1.0) && opts.disk_display {
        let x0 = cercle.0.trunc() as i64;
        let y0 = cercle.1.trunc() as i64;
        let r = cercle.2.trunc() as i64 + opts.delta_radius;
        if r > 0 {
            fill_circle(&mut frame_protus, x0, y0, r, 80);
        }
    }

    let k = {
        let kk = opts.img_rotate.div_euclid(90);
        kk.rem_euclid(4) as usize
    };
    let (frame_raw, frame_hc, frame_protus, cc) = if k != 0 {
        (frame_raw.rot90(k), frame_hc.rot90(k), frame_protus.rot90(k), cc.rot90(k))
    } else {
        (frame_raw, frame_hc, frame_protus, cc)
    };

    if !opts.nolog {
        if opts.clahe_only || !opts.protus_only {
            println!("saving image to:{}", basefich.to_string() + "_clahe.png");
            let _ = pngio::write_png16(&output_path(&(basefich.to_string() + "_clahe.png"), opts), &cc);
        }
        if opts.protus_only || !opts.clahe_only {
            let _ = pngio::write_png16(&output_path(&(basefich.to_string() + "_protus.png"), opts), &frame_protus);
        }
        if !opts.clahe_only && !opts.protus_only {
            let _ = pngio::write_png16(&output_path(&(basefich.to_string() + "_uncontrasted.png"), opts), &frame_raw);
            let _ = pngio::write_png16(&output_path(&(basefich.to_string() + "_high_contrast.png"), opts), &frame_hc);
        }
    }

    if opts.save_fit {
        let _ = pngio::write_fits16(&output_path(&(basefich.to_string() + "_clahe.fits"), opts), &cl1);
    }
    Processed { clahe: cc, high_contrast: frame_hc, protus: frame_protus }
}
