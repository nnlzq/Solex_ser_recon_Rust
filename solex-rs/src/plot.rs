//! Minimal raster plotting for the `save_fit` diagnostic PNGs
//! (spectral line fit, ellipse fit, transversalium correction).
//! Zero dependencies: draws into an RGB canvas and writes an 8-bit PNG.
//! Axes/titles are omitted on purpose — these are quick inspection images.

use crate::num::Img;
use crate::pngio;
use std::io;

pub const BLACK: [u8; 3] = [0, 0, 0];
pub const RED: [u8; 3] = [255, 0, 0];
pub const BLUE: [u8; 3] = [0, 0, 255];

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    px: Vec<u8>,
}

impl Canvas {
    pub fn new(w: usize, h: usize) -> Canvas {
        Canvas { w, h, px: vec![255u8; w * h * 3] }
    }

    #[inline]
    pub fn set(&mut self, x: i64, y: i64, c: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 { return; }
        let i = (y as usize * self.w + x as usize) * 3;
        self.px[i] = c[0];
        self.px[i + 1] = c[1];
        self.px[i + 2] = c[2];
    }

    /// Bresenham line between integer pixel coords.
    pub fn line(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, c: [u8; 3]) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let (mut x, mut y) = (x0, y0);
        loop {
            self.set(x, y, c);
            if x == x1 && y == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x += sx; }
            if e2 <= dx { err += dx; y += sy; }
        }
    }

    pub fn polyline(&mut self, pts: &[(f64, f64)], c: [u8; 3]) {
        for w2 in pts.windows(2) {
            let (x0, y0) = (w2[0].0.round() as i64, w2[0].1.round() as i64);
            let (x1, y1) = (w2[1].0.round() as i64, w2[1].1.round() as i64);
            self.line(x0, y0, x1, y1, c);
        }
    }

    pub fn rect(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, c: [u8; 3]) {
        self.line(x0, y0, x1, y0, c);
        self.line(x0, y1, x1, y1, c);
        self.line(x0, y0, x0, y1, c);
        self.line(x1, y0, x1, y1, c);
    }

    /// Filled disc marker ('o').
    pub fn dot(&mut self, cx: f64, cy: f64, r: i64, c: [u8; 3]) {
        let (x, y) = (cx.round() as i64, cy.round() as i64);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r { self.set(x + dx, y + dy, c); }
            }
        }
    }

    /// 'x' marker.
    pub fn cross(&mut self, cx: f64, cy: f64, half: i64, c: [u8; 3]) {
        let (x, y) = (cx.round() as i64, cy.round() as i64);
        self.line(x - half, y - half, x + half, y + half, c);
        self.line(x - half, y + half, x + half, y - half, c);
    }

    /// Nearest-neighbour grayscale render of an image into a rectangle,
    /// using min/max contrast normalisation (matplotlib imshow default).
    pub fn gray(&mut self, r: Rect, img: &Img) {
        let rows = r.rows();
        let cols = r.cols();
        let (lo, hi) = (img.min() as f64, img.max() as f64);
        let span = if hi > lo { hi - lo } else { 1.0 };
        let sx: Vec<usize> = (0..cols).map(|px| (px * img.w / cols).min(img.w - 1)).collect();
        let sy: Vec<usize> = (0..rows).map(|py| (py * img.h / rows).min(img.h - 1)).collect();
        for (py, &iy) in sy.iter().enumerate() {
            for (px, &ix) in sx.iter().enumerate() {
                let v = ((img.g(iy, ix) as f64 - lo) / span * 255.0).round().clamp(0.0, 255.0) as u8;
                self.set(r.x0 + px as i64, r.y0 + py as i64, [v, v, v]);
            }
        }
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        pngio::write_png_rgb8(path, self.w, self.h, &self.px)
    }
}

#[derive(Clone, Copy)]
pub struct Rect {
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
}

impl Rect {
    pub fn new(x0: i64, y0: i64, x1: i64, y1: i64) -> Rect {
        Rect { x0, y0, x1, y1 }
    }
    pub fn rows(&self) -> usize { (self.y1 - self.y0).max(1) as usize }
    pub fn cols(&self) -> usize { (self.x1 - self.x0).max(1) as usize }
}

/// Linear map of a data range onto a pixel range.
fn map_range(v: f64, d: (f64, f64), p: (f64, f64)) -> f64 {
    if (d.1 - d.0).abs() < f64::EPSILON { return (p.0 + p.1) / 2.0; }
    p.0 + (v - d.0) / (d.1 - d.0) * (p.1 - p.0)
}

fn with_pad(d: (f64, f64)) -> (f64, f64) {
    if d.1 > d.0 { (d.0 - 0.02 * (d.1 - d.0), d.1 + 0.02 * (d.1 - d.0)) } else { (d.0 - 0.5, d.1 + 0.5) }
}

/// Centre a data-aspect-preserving rect inside `r` (matplotlib aspect='equal').
fn fit_rect(r: Rect, dw: f64, dh: f64) -> Rect {
    if dw <= 0.0 || dh <= 0.0 { return r; }
    let pw = r.cols() as f64;
    let ph = r.rows() as f64;
    let target = dh / dw;
    let (nw, nh) = if target * pw <= ph { (pw, target * pw) } else { (ph / target, ph) };
    let x0 = r.x0 + ((pw - nw) / 2.0) as i64;
    let y0 = r.y0 + ((ph - nh) / 2.0) as i64;
    Rect::new(x0, y0, x0 + nw as i64, y0 + nh as i64)
}

/// `_transversalium_correction.png`: per-row correction factor.
pub fn transversalium(path: &str, c: &[f64]) -> io::Result<()> {
    let (wd, ht) = (900usize, 600usize);
    let mut cv = Canvas::new(wd, ht);
    let area = Rect::new(70, 30, wd as i64 - 30, ht as i64 - 60);
    cv.rect(area.x0, area.y0, area.x1, area.y1, BLACK);
    let n = c.len();
    let lo = c.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let vr = with_pad((lo, hi));
    let pts: Vec<(f64, f64)> = c.iter().enumerate().map(|(i, &v)| {
        let px = map_range(i as f64, (0.0, (n.max(2) - 1) as f64), (area.x0 as f64, area.x1 as f64));
        let py = map_range(v, vr, (area.y1 as f64, area.y0 as f64));
        (px, py)
    }).collect();
    cv.polyline(&pts, BLUE);
    cv.save(path)
}

/// `_spectral_line_data.png`: mean spectrum image, detected line minima
/// (red x) and the polynomial fit (blue).
pub fn spectral_line(path: &str, mean: &Img, sharp: &[usize], mask: &[bool],
                     y1: usize, y2: usize, curve: &[f64]) -> io::Result<()> {
    let (wd, ht) = (1000usize, 760usize);
    let mut cv = Canvas::new(wd, ht);
    let area0 = Rect::new(60, 30, wd as i64 - 30, ht as i64 - 40);
    let ih = mean.h as f64;
    let iw = mean.w as f64;
    // matplotlib: ax.set_aspect(0.1) -> a y-unit is 0.1 the size of an x-unit
    let area = fit_rect(area0, iw, ih * 0.1);
    cv.gray(area, mean);
    // imshow(origin='upper'): x in [-.5, w-.5], y in [ih-.5, -.5]
    let to_px = |x: f64, y: f64| -> (f64, f64) {
        let px = map_range(x, (-0.5, iw - 0.5), (area.x0 as f64, area.x1 as f64));
        let py = map_range(y, (ih - 0.5, -0.5), (area.y0 as f64, area.y1 as f64));
        (px, py)
    };
    // detected minima, subsampled like the Python plot: [::s] after masking
    let step = ((y2.saturating_sub(y1)) / 20 + 1).max(1);
    let mut kept = 0usize;
    for y in y1..y2.min(sharp.len()) {
        if mask.get(y).copied().unwrap_or(false) {
            if kept % step == 0 {
                let (px, py) = to_px(sharp[y] as f64, y as f64);
                cv.cross(px, py, 3, RED);
            }
            kept += 1;
        }
    }
    // polynomial fit line
    let fit: Vec<(f64, f64)> = curve.iter().enumerate()
        .map(|(i, &v)| to_px(v, i as f64)).collect();
    cv.polyline(&fit, BLUE);
    cv.save(path)
}

pub struct EllipseGeom {
    pub center: (f64, f64), // (row, col) of the fit
    pub height: f64,
    pub phi: f64,
    pub ratio: f64,
}

/// 100-point outline of the fitted ellipse, in (row, col) data coords,
/// using the same parametric form as `LsqEllipse.return_fit`:
///   x = cx + w*cos(t)*cos(phi) - h*sin(t)*sin(phi)
///   y = cy + w*cos(t)*sin(phi) + h*sin(t)*cos(phi)
fn ellipse_outline(g: &EllipseGeom) -> Vec<(f64, f64)> {
    let width = g.ratio * g.height;
    let (sp, cp) = (g.phi.sin(), g.phi.cos());
    let n = 100usize;
    (0..n).map(|k| {
        let t = 2.0 * std::f64::consts::PI * k as f64 / (n - 1) as f64;
        let (st, ct) = (t.sin(), t.cos());
        let x = g.center.0 + width * ct * cp - g.height * st * sp;
        let y = g.center.1 + width * ct * sp + g.height * st * cp;
        (x, y)
    }).collect()
}

/// `_ellipse_fit.png`: 2x2 panel of uncorrected image, detected edges,
/// geometrically corrected image with borders, filtered edges + ellipse.
pub fn ellipse_fit(path: &str, disk: &Img, raw_x: &[(f64, f64)], pts_t: &[(f64, f64)],
                   geom: &EllipseGeom, fix: &Img, borders: &[f64; 4]) -> io::Result<()> {
    let (wd, ht) = (1200usize, 920usize);
    let mut cv = Canvas::new(wd, ht);
    let pad = 25i64;
    let pw = (wd as i64 - 3 * pad) / 2;
    let ph = (ht as i64 - 3 * pad) / 2;
    let tl0 = Rect::new(pad, pad, pad + pw, pad + ph);
    let tr0 = Rect::new(2 * pad + pw, pad, 2 * pad + 2 * pw, pad + ph);
    let bl0 = Rect::new(pad, 2 * pad + ph, pad + pw, 2 * pad + 2 * ph);
    let br0 = Rect::new(2 * pad + pw, 2 * pad + ph, 2 * pad + 2 * pw, 2 * pad + 2 * ph);
    let iw = disk.w as f64;
    let ih = disk.h as f64;
    // the three image panels keep square pixels (matplotlib aspect='equal')
    let tl = fit_rect(tl0, iw, ih);
    let tr = fit_rect(tr0, iw, ih);
    let bl = fit_rect(bl0, fix.w as f64, fix.h as f64);

    cv.gray(tl, disk);
    cv.gray(tr, disk);
    cv.gray(bl, fix);

    // top-right: raw edge pixels
    let to_px = |r: Rect, x: f64, y: f64| -> (f64, f64) {
        (map_range(x, (-0.5, iw - 0.5), (r.x0 as f64, r.x1 as f64)),
         map_range(y, (-0.5, ih - 0.5), (r.y0 as f64, r.y1 as f64)))
    };
    for &(row, col) in raw_x {
        let (px, py) = to_px(tr, col, row);
        cv.dot(px, py, 1, RED);
    }
    // bottom-left: crop borders (horizontal at rows, vertical at cols)
    let (bx0, by0, bx1, by1) = (borders[0], borders[1], borders[2], borders[3]);
    for &yy in &[by0, by1] {
        let py = map_range(yy, (-0.5, ih - 0.5), (bl.y0 as f64, bl.y1 as f64)) as i64;
        cv.line(bl.x0, py, bl.x1, py, BLACK);
    }
    for &xx in &[bx0, bx1] {
        let px = map_range(xx, (-0.5, iw - 0.5), (bl.x0 as f64, bl.x1 as f64)) as i64;
        cv.line(px, bl.y0, px, bl.y1, BLACK);
    }
    // bottom-right: filtered edges + ellipse outline, auto-scaled
    let outline = ellipse_outline(geom);
    let mut xmin = f64::INFINITY; let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY; let mut ymax = f64::NEG_INFINITY;
    for &(r, c) in pts_t.iter().chain(outline.iter()) {
        xmin = xmin.min(c); xmax = xmax.max(c);
        ymin = ymin.min(r); ymax = ymax.max(r);
    }
    if xmin.is_finite() && ymin.is_finite() {
        let xr = with_pad((xmin, xmax));
        let yr = with_pad((ymin, ymax));
        let br = fit_rect(br0, xr.1 - xr.0, yr.1 - yr.0);
        let mp = |r: f64, c: f64| -> (f64, f64) {
            (map_range(c, xr, (br.x0 as f64, br.x1 as f64)),
             map_range(r, yr, (br.y0 as f64, br.y1 as f64)))
        };
        let line: Vec<(f64, f64)> = outline.iter().map(|&(r, c)| mp(r, c)).collect();
        cv.polyline(&line, BLUE);
        for &(r, c) in pts_t {
            let (px, py) = mp(r, c);
            cv.dot(px, py, 1, RED);
        }
    }
    cv.save(path)
}
