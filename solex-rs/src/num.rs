//! Numeric helpers: images, filters, linear algebra, numpy/scipy/cv2-compatible ops.

#[derive(Clone)]
pub struct Img {
    pub h: usize,
    pub w: usize,
    pub d: Vec<u16>,
}

impl Img {
    pub fn new(h: usize, w: usize) -> Img {
        Img { h, w, d: vec![0u16; h * w] }
    }
    #[inline]
    pub fn g(&self, r: usize, c: usize) -> u16 {
        self.d[r * self.w + c]
    }
    #[inline]
    pub fn s(&mut self, r: usize, c: usize, v: u16) {
        self.d[r * self.w + c] = v;
    }
    #[inline]
    pub fn row(&self, r: usize) -> &[u16] {
        &self.d[r * self.w..(r + 1) * self.w]
    }
    pub fn max(&self) -> u16 {
        self.d.iter().copied().max().unwrap_or(0)
    }
    pub fn min(&self) -> u16 {
        self.d.iter().copied().min().unwrap_or(0)
    }
    pub fn to_f(&self) -> ImgF {
        ImgF { h: self.h, w: self.w, d: self.d.iter().map(|&v| v as f64).collect() }
    }
    /// np.rot90 CCW, k times
    pub fn rot90(&self, k: usize) -> Img {
        let k = k % 4;
        if k == 0 { return self.clone(); }
        let (h, w) = (self.h, self.w);
        let mut o = Img::new(if k % 2 == 1 { w } else { h }, if k % 2 == 1 { h } else { w });
        for r in 0..h {
            for c in 0..w {
                let (nr, nc) = match k {
                    1 => (w - 1 - c, r),
                    2 => (h - 1 - r, w - 1 - c),
                    _ => (c, h - 1 - r),
                };
                o.s(nr, nc, self.g(r, c));
            }
        }
        o
    }
    /// np.flip(axis=1)
    pub fn flip_x(&self) -> Img {
        let mut o = Img::new(self.h, self.w);
        for r in 0..self.h {
            for c in 0..self.w {
                o.s(r, self.w - 1 - c, self.g(r, c));
            }
        }
        o
    }
}

#[derive(Clone)]
pub struct ImgF {
    pub h: usize,
    pub w: usize,
    pub d: Vec<f64>,
}

impl ImgF {
    pub fn new(h: usize, w: usize) -> ImgF {
        ImgF { h, w, d: vec![0.0; h * w] }
    }
    #[inline]
    pub fn g(&self, r: usize, c: usize) -> f64 {
        self.d[r * self.w + c]
    }
    #[inline]
    pub fn s(&mut self, r: usize, c: usize, v: f64) {
        self.d[r * self.w + c] = v;
    }
    pub fn to_u16_trunc(&self) -> Img {
        // np.array(f, dtype=uint16): truncate toward zero, wrap on overflow
        let mut o = Img::new(self.h, self.w);
        for i in 0..self.d.len() {
            let v = self.d[i];
            o.d[i] = if v.is_nan() { 0 } else { v as u16 };
        }
        o
    }
}

/// cv2 BORDER_REFLECT_101 (whole-sample symmetric): (c b a | a b c d | c b a)
pub fn reflect101(i: i64, n: usize) -> usize {
    let n = n as i64;
    if n <= 1 { return 0; }
    let period = 2 * (n - 1);
    let mut i = i % period;
    if i < 0 { i += period; }
    if i < n { i as usize } else { (period - i) as usize }
}

/// scipy mode='reflect' (half-sample symmetric): (d c b a | a b c d | d c b a)
pub fn scipy_reflect(i: i64, n: usize) -> usize {
    let n = n as i64;
    if n <= 1 { return 0; }
    let period = 2 * n;
    let mut i = i % period;
    if i < 0 { i += period; }
    if i < n { i as usize } else { (period - 1 - i) as usize }
}

pub fn sort_f(v: &[f64]) -> Vec<f64> {
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s
}

/// np.median
pub fn median(v: &[f64]) -> f64 {
    percentile(v, 50.0)
}

/// np.percentile with linear interpolation
pub fn percentile(v: &[f64], q: f64) -> f64 {
    let s = sort_f(v);
    pctl_sorted(&s, q)
}

pub fn pctl_sorted(s: &[f64], q: f64) -> f64 {
    let n = s.len();
    if n == 0 { return f64::NAN; }
    if n == 1 { return s[0]; }
    let h = (n as f64 - 1.0) * (q / 100.0);
    let i0 = h.floor() as usize;
    let i1 = (i0 + 1).min(n - 1);
    s[i0] + (h - i0 as f64) * (s[i1] - s[i0])
}

/// np.round (banker's rounding, half to even)
pub fn round_half_even(x: f64) -> f64 {
    if !x.is_finite() { return x; }
    let f = x.floor();
    let d = x - f;
    if d > 0.5 { f + 1.0 } else if d < 0.5 { f } else {
        let fi = f as i64;
        if fi % 2 == 0 { f } else { f + 1.0 }
    }
}

pub fn std_pop(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
}

/// cv2.blur on u16 with ksize (kx, ky): window [i-k/2, i-k/2+k-1], reflect101, round-half-even
pub fn box_blur_u16(img: &Img, kx: usize, ky: usize) -> Img {
    let kx = kx.max(1);
    let ky = ky.max(1);
    if kx == 1 && ky == 1 { return img.clone(); }
    let tmp = box_blur_axis_u16(img, kx, 1);
    box_blur_axis_u16(&tmp, 1, ky)
}

fn box_blur_axis_u16(img: &Img, kx: usize, ky: usize) -> Img {
    let (h, w) = (img.h, img.w);
    let mut o = Img::new(h, w);
    let area = (kx * ky) as f64;
    if kx > 1 {
        let off = kx / 2;
        for r in 0..h {
            let ext: Vec<f64> = (0..(w + kx - 1)).map(|t| img.g(r, reflect101(t as i64 - off as i64, w)) as f64).collect();
            let mut pref = vec![0.0f64; ext.len() + 1];
            for i in 0..ext.len() { pref[i + 1] = pref[i] + ext[i]; }
            for c in 0..w {
                let s = pref[c + kx] - pref[c];
                o.s(r, c, round_half_even(s / area).max(0.0).min(65535.0) as u16);
            }
        }
    } else {
        let off = ky / 2;
        for c in 0..w {
            let ext: Vec<f64> = (0..(h + ky - 1)).map(|t| img.g(reflect101(t as i64 - off as i64, h), c) as f64).collect();
            let mut pref = vec![0.0f64; ext.len() + 1];
            for i in 0..ext.len() { pref[i + 1] = pref[i] + ext[i]; }
            for r in 0..h {
                let s = pref[r + ky] - pref[r];
                o.s(r, c, round_half_even(s / area).max(0.0).min(65535.0) as u16);
            }
        }
    }
    o
}

/// cv2.blur on f64
pub fn box_blur_f(img: &ImgF, kx: usize, ky: usize) -> ImgF {
    let kx = kx.max(1);
    let ky = ky.max(1);
    if kx == 1 && ky == 1 { return img.clone(); }
    let tmp = box_blur_axis_f(img, kx, 1);
    box_blur_axis_f(&tmp, 1, ky)
}

fn box_blur_axis_f(img: &ImgF, kx: usize, ky: usize) -> ImgF {
    let (h, w) = (img.h, img.w);
    let mut o = ImgF::new(h, w);
    let area = (kx * ky) as f64;
    if kx > 1 {
        let off = kx / 2;
        for r in 0..h {
            let ext: Vec<f64> = (0..(w + kx - 1)).map(|t| img.g(r, reflect101(t as i64 - off as i64, w))).collect();
            let mut pref = vec![0.0f64; ext.len() + 1];
            for i in 0..ext.len() { pref[i + 1] = pref[i] + ext[i]; }
            for c in 0..w {
                o.s(r, c, (pref[c + kx] - pref[c]) / area);
            }
        }
    } else {
        let off = ky / 2;
        for c in 0..w {
            let ext: Vec<f64> = (0..(h + ky - 1)).map(|t| img.g(reflect101(t as i64 - off as i64, h), c)).collect();
            let mut pref = vec![0.0f64; ext.len() + 1];
            for i in 0..ext.len() { pref[i + 1] = pref[i] + ext[i]; }
            for r in 0..h {
                o.s(r, c, (pref[r + ky] - pref[r]) / area);
            }
        }
    }
    o
}

/// scipy gaussian kernel, truncate=4.0.
/// Bit-exact with scipy.ndimage._filters._gaussian_kernel1d:
/// phi = exp((-0.5/sigma2) * x^2), normalized by numpy's pairwise sum.
pub fn gauss_kernel(sigma: f64) -> Vec<f64> {
    let r = (4.0 * sigma + 0.5) as usize;
    let mut k = vec![0.0f64; 2 * r + 1];
    let sigma2 = sigma * sigma;
    let c = -0.5 / sigma2;
    for i in 0..=2 * r {
        let x = i as f64 - r as f64;
        k[i] = (c * (x * x)).exp();
    }
    let s = pairwise_sum(&k);
    for v in k.iter_mut() { *v /= s; }
    k
}

/// numpy DOUBLE_pairwise_sum (numpy/_core/src/umath/loops_utils.h.src),
/// PW_BLOCKSIZE = 128. Reproduces np.sum ordering bit-exactly.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for i in 0..n { res += a[i]; }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        for j in 0..8 { r[j] = a[j]; }
        let mut i = 8usize;
        let stop = n - (n % 8);
        while i < stop {
            for j in 0..8 { r[j] += a[i + j]; }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n { res += a[i]; i += 1; }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// scipy.ndimage.gaussian_filter1d, mode='reflect' (half-sample).
/// Same scipy symmetric fast-path accumulation as gauss1d_const_axis.
pub fn gauss1d_reflect(v: &[f64], sigma: f64) -> Vec<f64> {
    let k = gauss_kernel(sigma);
    let r = k.len() / 2;
    let n = v.len();
    (0..n).map(|i| {
        let ii = i as i64;
        let mut s = v[scipy_reflect(ii, n)] * k[r];
        for j in (1..=r).rev() {
            let jj = j as i64;
            s += (v[scipy_reflect(ii - jj, n)] + v[scipy_reflect(ii + jj, n)]) * k[r + j];
        }
        s
    }).collect()
}

/// gaussian_filter1d mode='constant' cval=0 along one axis of 2D image.
/// Matches scipy NI_Correlate1D's symmetric fast path: the centre term is
/// accumulated first, then paired terms (in[i-j] + in[i+j]) * w[j] from the
/// outermost pair (j = r) inward to j = 1.
pub fn gauss1d_const_axis(img: &ImgF, sigma: f64, axis: usize) -> ImgF {
    let k = gauss_kernel(sigma);
    let r = k.len() / 2;
    let (h, w) = (img.h, img.w);
    let mut o = ImgF::new(h, w);
    if axis == 0 {
        for c in 0..w {
            for i in 0..h {
                let mut s = img.g(i, c) * k[r];
                for j in (1..=r).rev() {
                    let up = if i >= j { img.g(i - j, c) } else { 0.0 };
                    let dn = if i + j < h { img.g(i + j, c) } else { 0.0 };
                    s += (up + dn) * k[r + j];
                }
                o.s(i, c, s);
            }
        }
    } else {
        for rrow in 0..h {
            for i in 0..w {
                let mut s = img.g(rrow, i) * k[r];
                for j in (1..=r).rev() {
                    let up = if i >= j { img.g(rrow, i - j) } else { 0.0 };
                    let dn = if i + j < w { img.g(rrow, i + j) } else { 0.0 };
                    s += (up + dn) * k[r + j];
                }
                o.s(rrow, i, s);
            }
        }
    }
    o
}

/// scipy.ndimage.correlate1d with mode='reflect' (half-sample).
/// Reproduces NI_Correlate1D's three paths (symmetric / antisymmetric /
/// generic), including its accumulation order, so results are bit-exact.
pub fn correlate1d(img: &ImgF, weights: &[f64], axis: usize) -> ImgF {
    let n = weights.len();
    let size1 = n / 2;
    let size2 = n - size1 - 1;
    let (h, w) = (img.h, img.w);
    // scipy tests symmetry with an absolute tolerance of DBL_EPSILON.
    let mut sym: i8 = 0;
    if n % 2 == 1 {
        sym = 1;
        for ii in 1..=size1 {
            if (weights[size1 + ii] - weights[size1 - ii]).abs() > f64::EPSILON {
                sym = 0;
                break;
            }
        }
        if sym == 0 {
            sym = -1;
            for ii in 1..=size1 {
                if (weights[size1 + ii] + weights[size1 - ii]).abs() > f64::EPSILON {
                    sym = 0;
                    break;
                }
            }
        }
    }
    let mut o = ImgF::new(h, w);
    let dim = if axis == 0 { h } else { w };
    let at = |idx: i64, other: usize| -> f64 {
        let r = scipy_reflect(idx, dim) as usize;
        if axis == 0 { img.g(r, other) } else { img.g(other, r) }
    };
    for a in 0..h {
        for b in 0..w {
            let i: i64 = if axis == 0 { a as i64 } else { b as i64 };
            let other = if axis == 0 { b } else { a };
            let mut s = if sym != 0 {
                at(i, other) * weights[size1]
            } else {
                at(i + size2 as i64, other) * weights[n - 1]
            };
            if sym > 0 {
                for jj in (-(size1 as i64))..0 {
                    s += (at(i + jj, other) + at(i - jj, other))
                        * weights[(size1 as i64 + jj) as usize];
                }
            } else if sym < 0 {
                for jj in (-(size1 as i64))..0 {
                    s += (at(i + jj, other) - at(i - jj, other))
                        * weights[(size1 as i64 + jj) as usize];
                }
            } else {
                for jj in (-(size1 as i64))..size2 as i64 {
                    s += at(i + jj, other) * weights[(size1 as i64 + jj) as usize];
                }
            }
            o.s(a, b, s);
        }
    }
    o
}

/// scipy.ndimage.sobel(input, axis)
pub fn sobel(img: &ImgF, axis: usize) -> ImgF {
    let a = correlate1d(img, &[-1.0, 0.0, 1.0], axis);
    correlate1d(&a, &[1.0, 2.0, 1.0], 1 - axis)
}

/// cv2.filter2D, correlation, anchor center, BORDER_REFLECT_101, f64
pub fn filter2d(img: &ImgF, ker: &[Vec<f64>]) -> ImgF {
    let kh = ker.len();
    let kw = ker[0].len();
    let ay = kh / 2;
    let ax = kw / 2;
    let (h, w) = (img.h, img.w);
    let mut o = ImgF::new(h, w);
    for r in 0..h {
        for c in 0..w {
            let mut s = 0.0;
            for ky in 0..kh {
                let sr = reflect101(r as i64 + ky as i64 - ay as i64, h);
                for kx in 0..kw {
                    let sc = reflect101(c as i64 + kx as i64 - ax as i64, w);
                    s += ker[ky][kx] * img.g(sr, sc);
                }
            }
            o.s(r, c, s);
        }
    }
    o
}

/// np.polyfit(x, y, 3) -> ascending coefficients [c0, c1, c2, c3]
/// via Householder-style Gram-Schmidt least squares on [1, x, x^2, x^3].
pub fn polyfit3(xs: &[f64], ys: &[f64]) -> [f64; 4] {
    let n = xs.len();
    let mut cols: Vec<Vec<f64>> = (0..4).map(|d| (0..n).map(|i| xs[i].powi(d as i32)).collect()).collect();
    let mut b = ys.to_vec();
    // modified Gram-Schmidt
    let mut r = [[0.0f64; 4]; 4];
    let mut q: Vec<Vec<f64>> = Vec::new();
    for j in 0..4 {
        let mut v = cols[j].clone();
        for i in 0..j {
            let dot: f64 = q[i].iter().zip(v.iter()).map(|(a, x)| a * x).sum();
            r[i][j] = dot;
            for k in 0..n { v[k] -= dot * q[i][k]; }
        }
        let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
        r[j][j] = norm;
        if norm > 1e-12 { for k in 0..n { v[k] /= norm; } }
        q.push(v);
    }
    let mut rb = [0.0f64; 4];
    for j in 0..4 { rb[j] = q[j].iter().zip(b.iter()).map(|(a, x)| a * x).sum(); }
    let _ = &mut b;
    // back substitution R c = rb
    let mut c = [0.0f64; 4];
    for j in (0..4).rev() {
        let mut s = rb[j];
        for k in (j + 1)..4 { s -= r[j][k] * c[k]; }
        c[j] = if r[j][j].abs() > 1e-300 { s / r[j][j] } else { 0.0 };
    }
    let _ = &mut cols;
    c
}

/// evaluate ascending poly
pub fn polyval(c: &[f64], x: f64) -> f64 {
    let mut s = 0.0;
    for i in (0..c.len()).rev() { s = s * x + c[i]; }
    s
}

/// solve n x n dense system (Gaussian elimination, partial pivot)
pub fn solve_n(a: Vec<Vec<f64>>, b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut a = a;
    let mut b = b.to_vec();
    for c in 0..n {
        let mut piv = c;
        for r in (c + 1)..n { if a[r][c].abs() > a[piv][c].abs() { piv = r; } }
        if a[piv][c].abs() < 1e-14 { return None; }
        a.swap(c, piv);
        b.swap(c, piv);
        for r in (c + 1)..n {
            let f = a[r][c] / a[c][c];
            for j in c..n { a[r][j] -= f * a[c][j]; }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for j in (r + 1)..n { s -= a[r][j] * x[j]; }
        x[r] = s / a[r][r];
    }
    Some(x)
}

pub fn inv_n(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        out.push(solve_n(a.to_vec(), &e)?);
    }
    // transpose: solve_n returns column j of inverse
    let mut inv = vec![vec![0.0; n]; n];
    for j in 0..n { for i in 0..n { inv[i][j] = out[j][i]; } }
    Some(inv)
}

pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let k = b.len();
    (0..n).map(|i| (0..m).map(|j| (0..k).map(|t| a[i][t] * b[t][j]).sum()).collect()).collect()
}

pub fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| row.iter().zip(v.iter()).map(|(a, x)| a * x).sum()).collect()
}

/// scipy.signal.savgol_filter, polyorder=3, mode='interp'
pub fn savgol(v: &[f64], window: usize, poly: usize) -> Vec<f64> {
    let n = v.len();
    if n == 0 { return Vec::new(); }
    let mut window = window;
    if window % 2 == 0 { window = window.saturating_sub(1); }
    if window < poly + 1 || n <= 1 { return v.to_vec(); }
    if window > n { window = if n % 2 == 1 { n } else { n - 1 }; }
    if window < poly + 1 { return v.to_vec(); }
    let hw = window / 2;
    // interior convolution coefficients: c = A (A^T A)^-1 e0
    let a: Vec<Vec<f64>> = (0..window).map(|k| {
        let j = k as f64 - hw as f64;
        (0..=poly).map(|d| j.powi(d as i32)).collect()
    }).collect();
    let ata: Vec<Vec<f64>> = (0..=poly).map(|i| (0..=poly).map(|j| {
        (0..window).map(|k| a[k][i] * a[k][j]).sum()
    }).collect()).collect();
    let mut e0 = vec![0.0; poly + 1];
    e0[0] = 1.0;
    let z = solve_n(ata, &e0).unwrap_or_else(|| {
        let mut z = vec![0.0; poly + 1];
        z[0] = 1.0;
        z
    });
    let coeff: Vec<f64> = (0..window).map(|k| (0..=poly).map(|d| a[k][d] * z[d]).sum()).collect();
    let mut out = vec![0.0; n];
    for i in hw..(n - hw) {
        let mut s = 0.0;
        for (j, &c) in coeff.iter().enumerate() { s += c * v[i - hw + j]; }
        out[i] = s;
    }
    // edges: fit poly to first/last window samples, evaluate at i
    let fit_at = |xs: &[f64], ys: &[f64], xq: f64| -> f64 {
        let c = polyfit3(xs, ys);
        polyval(&c, xq)
    };
    for i in 0..hw.min(n) {
        let xs: Vec<f64> = (0..window).map(|k| k as f64).collect();
        let ys: Vec<f64> = (0..window).map(|k| v[k]).collect();
        out[i] = fit_at(&xs, &ys, i as f64);
    }
    for i in (n - hw).max(hw)..n {
        let xs: Vec<f64> = (0..window).map(|k| k as f64).collect();
        let ys: Vec<f64> = (0..window).map(|k| v[n - window + k]).collect();
        out[i] = fit_at(&xs, &ys, i as f64 - (n - window) as f64);
    }
    out
}

/// 8-connected labeling. labels start at 1. Returns (labels, count).
pub fn label8(mask: &[bool], h: usize, w: usize) -> (Vec<i32>, usize) {
    let mut lab = vec![0i32; h * w];
    let mut next = 1i32;
    let mut count = 0usize;
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            if mask[i] && lab[i] == 0 {
                let id = next;
                next += 1;
                count += 1;
                stack.push((r, c));
                lab[i] = id;
                while let Some((y, x)) = stack.pop() {
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            if dy == 0 && dx == 0 { continue; }
                            let ny = y as i64 + dy;
                            let nx = x as i64 + dx;
                            if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 { continue; }
                            let j = ny as usize * w + nx as usize;
                            if mask[j] && lab[j] == 0 {
                                lab[j] = id;
                                stack.push((ny as usize, nx as usize));
                            }
                        }
                    }
                }
            }
        }
    }
    (lab, count)
}

/// Monotone-chain convex hull. Returns indices of hull vertices in the input slice.
pub fn hull_indices(pts: &[(f64, f64)]) -> Vec<usize> {
    let n = pts.len();
    if n < 3 { return (0..n).collect(); }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        pts[a].0.partial_cmp(&pts[b].0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(pts[a].1.partial_cmp(&pts[b].1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let cross = |o: usize, a: usize, b: usize| -> f64 {
        (pts[a].0 - pts[o].0) * (pts[b].1 - pts[o].1) - (pts[a].1 - pts[o].1) * (pts[b].0 - pts[o].0)
    };
    let mut hull: Vec<usize> = Vec::with_capacity(2 * n);
    for &i in &idx {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], i) <= 0.0 {
            hull.pop();
        }
        hull.push(i);
    }
    let lower = hull.len();
    for &i in idx.iter().rev() {
        while hull.len() > lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], i) <= 0.0 {
            hull.pop();
        }
        hull.push(i);
    }
    hull.pop();
    hull.sort();
    hull.dedup();
    hull
}

/// Real eigen-decomposition helpers for 3x3 matrices (np.linalg.eig subset).
/// Returns eigenvectors (normalized) for real eigenvalues, in descending
/// eigenvalue order (LAPACK order is arbitrary; usually only one is valid).
pub fn eig3_vecs(m: [[f64; 3]; 3]) -> Vec<[f64; 3]> {
    // characteristic: λ^3 - tr λ^2 + s λ - d = 0
    let tr = m[0][0] + m[1][1] + m[2][2];
    let s = (m[0][0] * m[1][1] - m[0][1] * m[1][0])
        + (m[0][0] * m[2][2] - m[0][2] * m[2][0])
        + (m[1][1] * m[2][2] - m[1][2] * m[2][1]);
    let d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let roots = cubic_roots(-tr, s, -d);
    let mut out = Vec::new();
    for lam in roots {
        // null space of (M - lam I): cross product of two rows
        let a = [
            [m[0][0] - lam, m[0][1], m[0][2]],
            [m[1][0], m[1][1] - lam, m[1][2]],
            [m[2][0], m[2][1], m[2][2] - lam],
        ];
        let cands = [
            cross3(a[0], a[1]),
            cross3(a[0], a[2]),
            cross3(a[1], a[2]),
        ];
        for v in cands {
            let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if nv > 1e-9 {
                out.push([v[0] / nv, v[1] / nv, v[2] / nv]);
                break;
            }
        }
    }
    out
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// roots of x^3 + p x^2 + q x + r = 0 (all real ones)
pub fn cubic_roots(p: f64, q: f64, r: f64) -> Vec<f64> {
    // depressed: t^3 + a t + b, x = t - p/3
    let a = q - p * p / 3.0;
    let b = 2.0 * p * p * p / 27.0 - p * q / 3.0 + r;
    let shift = -p / 3.0;
    let disc = b * b / 4.0 + a * a * a / 27.0;
    if disc > 1e-12 {
        // one real root
        let sq = disc.sqrt();
        let u = (-b / 2.0 + sq).cbrt();
        let v = (-b / 2.0 - sq).cbrt();
        vec![u + v + shift]
    } else if disc.abs() <= 1e-12 {
        // repeated roots
        if b.abs() <= 1e-12 && a.abs() <= 1e-12 {
            vec![shift]
        } else {
            let u = (-b / 2.0).cbrt();
            vec![2.0 * u + shift, -u + shift]
        }
    } else {
        // three real roots
        let rho = (-a / 3.0).sqrt();
        let theta = (-b / (2.0 * rho * rho * rho)).clamp(-1.0, 1.0).acos();
        (0..3).map(|k| 2.0 * rho * ((theta / 3.0 + 2.0 * std::f64::consts::PI * k as f64 / 3.0).cos()) + shift).collect()
    }
}
