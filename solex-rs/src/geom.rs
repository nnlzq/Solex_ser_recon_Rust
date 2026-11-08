//! Geometry correction: ellipse->circle transform and image warping
//! (ellipse_to_circle.get_correction_matrix / correct_image).

use crate::num::*;

pub struct Mat2 { pub m: [[f64; 2]; 2] }

impl Mat2 {
    pub fn det(&self) -> f64 { self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0] }
    pub fn inv(&self) -> Mat2 {
        let d = self.det();
        Mat2 {
            m: [
                [self.m[1][1] / d, -self.m[0][1] / d],
                [-self.m[1][0] / d, self.m[0][0] / d],
            ],
        }
    }
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.m[0][0] * x + self.m[0][1] * y, self.m[1][0] * x + self.m[1][1] * y)
    }
}

/// rot(x) = [[cos, sin], [-sin, cos]]
fn rot(x: f64) -> Mat2 {
    Mat2 { m: [[x.cos(), x.sin()], [-x.sin(), x.cos()]] }
}

fn mul(a: &Mat2, b: &Mat2) -> Mat2 {
    Mat2 {
        m: [
            [a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0], a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1]],
            [a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0], a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1]],
        ],
    }
}

pub fn get_correction_matrix(phi: f64, r: f64) -> (Mat2, f64) {
    let d = Mat2 { m: [[r, 0.0], [0.0, 1.0]] };
    let stretch = mul(&rot(phi), &mul(&d, &rot(-phi)));
    let theta = (stretch.m[1][0] / stretch.m[0][0]).atan();
    let unrot = rot(theta);
    let mut corr = mul(&unrot, &stretch);
    corr.m[1][0] = 0.0;
    let s = corr.m[1][1];
    corr.m[0][0] /= s;
    corr.m[0][1] /= s;
    corr.m[1][0] /= s;
    corr.m[1][1] /= s;
    (corr.inv(), theta)
}

pub struct Mat3 { pub m: [[f64; 3]; 3] }

impl Mat3 {
    pub fn from_mat2(a: &Mat2) -> Mat3 {
        Mat3 { m: [[a.m[0][0], a.m[0][1], 0.0], [a.m[1][0], a.m[1][1], 0.0], [0.0, 0.0, 1.0]] }
    }
    pub fn inv(&self) -> Mat3 {
        let m = &self.m;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let cof = [
            [m[1][1] * m[2][2] - m[1][2] * m[2][1], -(m[1][0] * m[2][2] - m[1][2] * m[2][0]), m[1][0] * m[2][1] - m[1][1] * m[2][0]],
            [-(m[0][1] * m[2][2] - m[0][2] * m[2][1]), m[0][0] * m[2][2] - m[0][2] * m[2][0], -(m[0][0] * m[2][1] - m[0][1] * m[2][0])],
            [m[0][1] * m[1][2] - m[0][2] * m[1][1], -(m[0][0] * m[1][2] - m[0][2] * m[1][0]), m[0][0] * m[1][1] - m[0][1] * m[1][0]],
        ];
        let mut inv = [[0.0; 3]; 3];
        for i in 0..3 { for j in 0..3 { inv[i][j] = cof[j][i] / det; } }
        Mat3 { m: inv }
    }
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let w = self.m[2][0] * x + self.m[2][1] * y + self.m[2][2];
        (
            (self.m[0][0] * x + self.m[0][1] * y + self.m[0][2]) / w,
            (self.m[1][0] * x + self.m[1][1] * y + self.m[1][2]) / w,
        )
    }
}

/// correct_image: returns (u16 image, circle (x,y,r), mat3)
pub fn correct_image(image: &ImgF, phi: f64, ratio: f64, center: (f64, f64), height: f64) -> (Img, (f64, f64, f64), Mat3) {
    let (mat, _theta) = get_correction_matrix(phi, ratio);
    let mut mat3 = Mat3::from_mat2(&mat);
    let corners = [(0.0, 0.0), (0.0, image.h as f64), (image.w as f64, 0.0), (image.w as f64, image.h as f64)];
    let minv = mat3.inv();
    let new_corners: Vec<(f64, f64)> = corners.iter().map(|(x, y)| minv.apply(*x, *y)).collect();
    let min_x = new_corners.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = new_corners.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = new_corners.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = new_corners.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let new_h = max_y - min_y;
    let new_w = max_x - min_x;
    mat3.m[0][2] = mat.m[0][0] * min_x + mat.m[0][1] * min_y;
    mat3.m[1][2] = mat.m[1][0] * min_x + mat.m[1][1] * min_y;
    let out_h = new_h.ceil().max(1.0) as usize;
    let out_w = new_w.ceil().max(1.0) as usize;
    let border_val = image.g(0, 0);

    // cv2.warpPerspective(image, cv_M, (out_w,out_h), BORDER_CONSTANT):
    // OpenCV's M is the FORWARD src->dst map and internally samples src = M^-1 . dst.
    // Python passes cv_M = inv(mat3), so the effective sampling is src = mat3 . (x,y,1).
    if crate::edge::dbg_on() {
        eprintln!("DBG warp: min=({} {}) out=({} {}) mat3={:?} center=({} {}) height={}",
            min_x, min_y, out_w, out_h, mat3.m, center.0, center.1, height);
    }
    let mut out = ImgF::new(out_h, out_w);
    for y in 0..out_h {
        for x in 0..out_w {
            let (sx, sy) = mat3.apply(x as f64, y as f64);
            out.s(y, x, bilinear(image, sx, sy, border_val));
        }
    }
    // scale to 16-bit, clip
    let mut u = Img::new(out_h, out_w);
    for i in 0..out.d.len() {
        let v = (out.d[i] * 65536.0).clamp(0.0, 65535.0);
        u.d[i] = v as u16;
    }

    let invmat = mat.inv();
    let nc = invmat.apply(center.0, center.1);
    let new_center = (nc.0 - min_x, nc.1 - min_y);
    let new_radius = height * (ratio / mat.det()).abs().sqrt();
    (u, (new_center.0, new_center.1, new_radius), mat3)
}

fn bilinear(img: &ImgF, x: f64, y: f64, border: f64) -> f64 {
    if !x.is_finite() || !y.is_finite() { return border; }
    // cv2 INTER_LINEAR quantizes interpolation weights to 1/32 (INTER_BITS=5)
    let fx = x.floor();
    let fy = y.floor();
    let wx = ((x - fx) * 32.0).round() / 32.0;
    let wy = ((y - fy) * 32.0).round() / 32.0;
    let v00 = px(img, fx, fy, border);
    let v10 = px(img, fx + 1.0, fy, border);
    let v01 = px(img, fx, fy + 1.0, border);
    let v11 = px(img, fx + 1.0, fy + 1.0, border);
    let a = v00 + (v10 - v00) * wx;
    let b = v01 + (v11 - v01) * wx;
    a + (b - a) * wy
}

fn px(img: &ImgF, x: f64, y: f64, border: f64) -> f64 {
    if x < 0.0 || y < 0.0 || x >= img.w as f64 || y >= img.h as f64 { return border; }
    img.g(y as usize, x as usize)
}

/// borders: given thresholded points (row, col) and mat3, compute [minX,minY,maxX,maxY]
pub fn compute_borders(points: &[(f64, f64)], mat3: &Mat3) -> [f64; 4] {
    let inv = mat3.inv();
    let mut minx = f64::INFINITY;
    let mut miny = f64::INFINITY;
    let mut maxx = f64::NEG_INFINITY;
    let mut maxy = f64::NEG_INFINITY;
    for &(row, col) in points {
        // X_f3 = (x=col, y=row, 1)
        let (tx, ty) = inv.apply(col, row);
        if tx < minx { minx = tx; }
        if tx > maxx { maxx = tx; }
        if ty < miny { miny = ty; }
        if ty > maxy { maxy = ty; }
    }
    if !minx.is_finite() { return [0.0, 0.0, 0.0, 0.0]; }
    [minx, miny, maxx, maxy]
}
