//! CLAHE for 16-bit images, matching cv::createCLAHE(clipLimit, (2,2)).

use crate::num::*;

pub fn clahe(frame: &Img, clip_limit: f64) -> Img {
    let (h, w) = (frame.h, frame.w);
    // OpenCV CLAHE_Impl::apply: if w%tilesX==0 && h%tilesY==0 -> NO padding;
    // else pad BOTH dims by tiles - (n % tiles) (an even dim gets padded by 2!).
    let (eh, ew) = if w % 2 == 0 && h % 2 == 0 {
        (h, w)
    } else {
        (h + (2 - h % 2), w + (2 - w % 2))
    };
    let tw = ew / 2;
    let th = eh / 2;
    let tile_area = tw * th;
    // OpenCV: static_cast<float>(histSize-1) / tileSizeTotal  (float division)
    let lut_scale = 65535.0f32 / tile_area as f32;
    let clip = ((clip_limit * tile_area as f64 / 65536.0) as i64).max(1);

    let ext = |r: usize, c: usize| -> usize {
        frame.g(reflect101(r as i64, h), reflect101(c as i64, w)) as usize
    };

    // build 4 LUTs (row-major tiles)
    let luts: Vec<Vec<u16>> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for t in 0..4 {
            let ext = &ext;
            hs.push(s.spawn(move || -> Vec<u16> {
                let tx = t % 2;
                let ty = t / 2;
                let mut hist = vec![0u32; 65536];
                for r in ty * th..(ty + 1) * th {
                    for c in tx * tw..(tx + 1) * tw {
                        hist[ext(r, c)] += 1;
                    }
                }
                let mut clipped: i64 = 0;
                for i in 0..65536 {
                    let d = hist[i] as i64 - clip;
                    if d > 0 { clipped += d; hist[i] = clip as u32; }
                }
                let batch = clipped / 65536;
                let mut residual = clipped - batch * 65536;
                if batch > 0 { for i in 0..65536 { hist[i] += batch as u32; } }
                if residual != 0 {
                    // OpenCV: residualStep computed ONCE from the initial residual
                    let step = (65536 / residual).max(1) as usize;
                    let mut i = 0usize;
                    while i < 65536 && residual > 0 {
                        hist[i] += 1;
                        i += step;
                        residual -= 1;
                    }
                }
                let mut lut = vec![0u16; 65536];
                let mut sum: i64 = 0;
                for j in 0..65536 {
                    sum += hist[j] as i64;
                    let v = round_half_even((sum as f32 * lut_scale) as f64);
                    lut[j] = v.clamp(0.0, 65535.0) as u16;
                }
                lut
            }));
        }
        hs.into_iter().map(|hh| hh.join().unwrap()).collect()
    });

    let mut out = Img::new(h, w);
    std::thread::scope(|s| {
        let nchunk = num_cpus().min(h).max(1);
        let chunk = h.div_ceil(nchunk);
        let mut hs = Vec::new();
        for (ci, rows) in out.d.chunks_mut(chunk * w).enumerate() {
            let r0 = ci * chunk;
            let r1 = (r0 + chunk).min(h);
            let luts = &luts;
            hs.push(s.spawn(move || {
                // replicate CLAHE_Interpolation_Body float32 arithmetic
                let inv_tw = 1.0f32 / tw as f32;
                let inv_th = 1.0f32 / th as f32;
                for r in r0..r1 {
                    let tyf = r as f32 * inv_th - 0.5f32;
                    let tyf_floor = tyf.floor();
                    let ya = tyf - tyf_floor;
                    let ty1 = (tyf_floor.max(0.0)) as usize;
                    let ty2 = ((tyf_floor + 1.0).min(1.0)) as usize;
                    for c in 0..w {
                        let v = frame.g(r, c) as usize;
                        let txf = c as f32 * inv_tw - 0.5f32;
                        let txf_floor = txf.floor();
                        let wgt = txf - txf_floor;
                        let tx1 = (txf_floor.max(0.0)) as usize;
                        let tx2 = ((txf_floor + 1.0).min(1.0)) as usize;
                        let l00 = luts[ty1 * 2 + tx1][v] as f32;
                        let l01 = luts[ty1 * 2 + tx2][v] as f32;
                        let l10 = luts[ty2 * 2 + tx1][v] as f32;
                        let l11 = luts[ty2 * 2 + tx2][v] as f32;
                        let top = l00 * (1.0 - wgt) + l01 * wgt;
                        let bot = l10 * (1.0 - wgt) + l11 * wgt;
                        let res = top * (1.0 - ya) + bot * ya;
                        rows[(r - r0) * w + c] = round_half_even(res as f64).clamp(0.0, 65535.0) as u16;
                    }
                }
            }));
        }
        for hh in hs { hh.join().unwrap(); }
    });
    out
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}
