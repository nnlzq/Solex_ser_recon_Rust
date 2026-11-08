//! Disk reconstruction from SER frames (read_video_improved).

use crate::ser::{self, SerInfo};
use std::io;

/// Reconstruct one disk image (ih x frame_count) for a given pixel shift.
/// Parallel over frame chunks (disjoint columns).
pub fn reconstruct(path: &str, info: &SerInfo, fit: &[(i64, f64, f64)], shift: i64, nthreads: usize) -> io::Result<Vec<u16>> {
    let ih = info.ih;
    let iw = info.iw;
    let nf = info.frame_count;
    // precompute column indices and weights
    let mut ind_l = vec![0i64; ih];
    let mut rw = vec![0.0f64; ih];
    for r in 0..ih {
        let mut l = fit[r].0 + shift;
        if l < 0 { l = 0; }
        if l > iw as i64 - 2 { l = iw as i64 - 2; }
        ind_l[r] = l;
        rw[r] = fit[r].1; // right weight = frac
    }
    let mut disk = vec![0u16; ih * nf];
    let nchunk = nthreads.min(nf).max(1);
    let chunk = nf.div_ceil(nchunk);
    let partials: Vec<(usize, usize, Vec<u16>)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for ci in 0..nchunk {
            let f0 = ci * chunk;
            let f1 = (f0 + chunk).min(nf);
            if f0 >= f1 { continue; }
            let ind_l = &ind_l;
            let rw = &rw;
            handles.push(s.spawn(move || {
                let bw = f1 - f0;
                let mut buf = vec![0u16; ih * bw];
                let mut file = ser::open(path).unwrap();
                for f in f0..f1 {
                    let frame = ser::read_frame(&mut file, info, f).unwrap();
                    for r in 0..ih {
                        let l = ind_l[r] as usize;
                        let lv = frame[r * iw + l] as f64;
                        let rv = frame[r * iw + l + 1] as f64;
                        let v = lv * (1.0 - rw[r]) + rv * rw[r];
                        buf[r * bw + (f - f0)] = v as u16; // numpy float->uint16 truncation
                    }
                }
                (f0, f1, buf)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (f0, f1, buf) in partials {
        let bw = f1 - f0;
        for r in 0..ih {
            disk[r * nf + f0..r * nf + f1].copy_from_slice(&buf[r * bw..(r + 1) * bw]);
        }
    }
    Ok(disk)
}
