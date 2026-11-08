//! Shared processing pipeline (solex_read / solex_process), used by both the
//! command line front end and the GUI.

use crate::edge;
use crate::fitline;
use crate::geom;
use crate::num::{Img, ImgF};
use crate::plot;
use crate::pngio;
use crate::procss::{self, Options};
use crate::recon;
use crate::ser;
use crate::transv;
use crate::vig;
use std::path::Path;

/// Result of the read pass: reconstructed disks for every requested shift.
pub struct ReadResult {
    pub disks: Vec<Img>,
    pub bounds: (usize, usize),
    /// mean spectrum image (for the GUI preview)
    pub mean: Img,
    /// polynomial line fit sampled over all rows (for the GUI preview)
    pub curve: Vec<f64>,
}

/// Contrast-stretched images kept in memory for the "Show graphics" preview.
#[derive(Clone)]
pub struct Preview {
    pub clahe: Img,
    pub high_contrast: Img,
    pub protus: Img,
}

/// solex_read: read SER, fit spectral line, reconstruct disks for each shift.
pub fn solex_read(file: &str, opts: &mut Options) -> std::io::Result<ReadResult> {
    let basefich0 = Path::new(file).with_extension("");
    let basefich0 = basefich0.to_string_lossy().into_owned();
    let log = basefich0.clone() + "_log.txt";
    procss::clearlog(&log, opts);
    procss::logme(&log, opts, &format!("Pixel shift : {:?}", opts.shift));
    opts.shift_requested = opts.shift.clone();
    // dedupe([ellipse_fit_shift, 0] + shift), order-preserving
    let mut new_shift = vec![opts.ellipse_fit_shift, 0i64];
    new_shift.extend(opts.shift.iter().copied());
    let mut seen: Vec<i64> = Vec::new();
    for s in new_shift {
        if !seen.contains(&s) { seen.push(s); }
    }
    opts.shift = seen;

    let info = ser::SerInfo::probe(file)?;
    procss::logme(&log, opts, &format!("Width, Height : {} {}", info.width, info.height));
    procss::logme(&log, opts, &format!("Number of frames : {}", info.frame_count));

    let (mean_img, max_img) = fitline::compute_mean_max(file, &info, opts.nthreads)?;
    if opts.save_fit {
        let _ = pngio::write_fits16(&procss::output_path(&(basefich0.clone() + "_mean.fits"), opts), &mean_img);
    }

    let lf = fitline::compute_mean_return_fit(file, &info, &mean_img, &max_img, opts.nthreads);
    procss::logme(&log, opts, &format!("Vertical limits y1, y2 : {} {}", lf.y1, lf.y2));
    let poly_str: Vec<String> = lf.poly.iter().map(|v| format!("{:.14}", v)).collect();
    procss::logme(&log, opts, &format!("Spectral line polynomial fit: [{}]", poly_str.join(" ")));
    if !opts.clahe_only && !opts.protus_only {
        let _ = plot::spectral_line(
            &procss::output_path(&(basefich0.clone() + "_spectral_line_data.png"), opts),
            &mean_img, &lf.sharp, &lf.mask, lf.y1, lf.y2, &lf.curve);
    }

    let mut disks: Vec<Img> = Vec::new();
    for &sh in &opts.shift {
        let d = recon::reconstruct(file, &info, &lf.fit, sh, opts.nthreads)?;
        let img = Img { h: info.ih, w: info.frame_count, d };
        disks.push(if opts.flip_x { img.flip_x() } else { img });
    }
    for (i, disk) in disks.iter().enumerate() {
        let requested = opts.shift_requested.contains(&opts.shift[i]);
        if opts.save_fit && requested {
            let basefich = format!("{}_shift={}", basefich0, opts.shift[i]);
            let _ = pngio::write_fits16(&procss::output_path(&(basefich + "_raw.fits"), opts), disk);
        }
    }
    Ok(ReadResult { disks, bounds: (lf.y1, lf.y2), mean: mean_img, curve: lf.curve })
}

/// solex_process: circularise, detransversalium, crop, contrast-adjust, save.
pub fn solex_process(file: &str, opts: &mut Options, disks: &[Img], bounds: (usize, usize)) -> Vec<Preview> {
    let basefich0 = Path::new(file).with_extension("");
    let basefich0 = basefich0.to_string_lossy().into_owned();
    let log = basefich0.clone() + "_log.txt";
    if opts.transversalium {
        procss::logme(&log, opts, &format!("Transversalium correction : {}", opts.trans_strength));
    } else {
        procss::logme(&log, opts, "Transversalium disabled");
    }
    procss::logme(&log, opts, &format!("Mirror X : {}", pybool(opts.flip_x)));
    procss::logme(&log, opts, &format!("Post-rotation : {} degrees", opts.img_rotate));
    procss::logme(&log, opts, &format!("Protus adjustment : {}", opts.delta_radius));
    procss::logme(&log, opts, &format!("de-vignette : {}", pybool(opts.de_vignette)));

    let mut borders = [0.0f64; 4];
    let mut cercle0 = (-1.0f64, -1.0f64, -1.0f64);
    let mut previews: Vec<Preview> = Vec::new();

    for i in 0..disks.len() {
        let requested = opts.shift_requested.contains(&opts.shift[i]);
        let basefich = format!("{}_shift={}", basefich0, opts.shift[i]);

        let frame: ImgF = if opts.ratio_fixe.is_none() && opts.slant_fix.is_none() {
            let res = edge::ellipse_to_circle(&disks[i]);
            if !opts.clahe_only && !opts.protus_only && res.ok {
                let g = plot::EllipseGeom {
                    center: res.fit_center, height: res.height, phi: res.phi, ratio: res.ratio,
                };
                let _ = plot::ellipse_fit(
                    &procss::output_path(&(basefich.clone() + "_ellipse_fit.png"), opts),
                    &disks[i], &res.raw_x, &res.pts_t, &g, &res.image, &res.borders);
            }
            // correct_image(print_log=True) logging
            procss::logme(&log, opts, &format!("Y/X ratio : {:.3}", res.ratio));
            procss::logme(&log, opts, &format!("Tilt angle : {:.3} degrees", res.phi.to_degrees()));
            let (mat, _theta) = geom::get_correction_matrix(res.phi, res.ratio);
            procss::logme(&log, opts, &format!(
                "Linear transform correction matrix : \n[[{:.14} {:.14}]\n [{:.14} {:.14}]]",
                mat.m[0][0], mat.m[0][1], mat.m[1][0], mat.m[1][1]));
            if res.circle.2 == -1.0 {
                procss::logme(&log, opts, "Disk position, radius : UNKNOWN");
            } else {
                procss::logme(&log, opts, &format!(
                    "Disk position, radius : [{:.14} {:.14}], {:.3}",
                    res.circle.0, res.circle.1, res.circle.2));
            }
            procss::logme(&log, opts, &format!("Unrotation : {:.3} degrees", res.theta.to_degrees()));
            opts.ratio_fixe = Some(res.ratio);
            opts.slant_fix = Some(res.phi.to_degrees());
            cercle0 = res.circle;
            borders = res.borders;
            res.image.to_f()
        } else {
            let ratio = opts.ratio_fixe.unwrap_or(1.0);
            let phi = opts.slant_fix.unwrap_or(0.0).to_radians();
            if !requested { continue; }
            let imgf = num_imgf(disks[i].clone());
            let (fix, _nc, _m3) = geom::correct_image(&imgf, phi, ratio, (-1.0, -1.0), -1.0);
            if i == 0 {
                procss::logme(&log, opts, &format!("Y/X ratio : {:.3}", ratio));
                procss::logme(&log, opts, &format!("Tilt angle : {:.3} degrees", phi.to_degrees()));
                let (mat, _theta) = geom::get_correction_matrix(phi, ratio);
                procss::logme(&log, opts, &format!(
                    "Linear transform correction matrix : \n[[{:.14} {:.14}]\n [{:.14} {:.14}]]",
                    mat.m[0][0], mat.m[0][1], mat.m[1][0], mat.m[1][1]));
                procss::logme(&log, opts, "Disk position, radius : UNKNOWN");
                procss::logme(&log, opts, &format!("Unrotation : {:.3} degrees", _theta.to_degrees()));
            }
            if opts.de_vignette {
                if !(cercle0.0 == -1.0 && cercle0.1 == -1.0 && cercle0.2 == -1.0) {
                    vig::remove_vignette(&fix, cercle0)
                } else {
                    eprintln!("WARNING: cannot de-vignette without ellipse fit");
                    fix.to_f()
                }
            } else {
                fix.to_f()
            }
        };

        if !requested { continue; }
        if let Some(p) = single_image_process(frame, opts, cercle0, &borders, &basefich, bounds) {
            previews.push(p);
        }
        procss::write_complete(&log, opts);
    }
    previews
}

fn num_imgf(img: Img) -> ImgF {
    ImgF { h: img.h, w: img.w, d: img.d.iter().map(|&v| v as f64 / 65536.0).collect() }
}

pub(crate) fn single_image_process(
    frame: ImgF,
    opts: &Options,
    cercle0: (f64, f64, f64),
    borders: &[f64; 4],
    basefich: &str,
    bounds: (usize, usize),
) -> Option<Preview> {
    if opts.save_fit {
        let _ = pngio::write_fits16(
            &procss::output_path(&(basefich.to_string() + "_circular.fits"), opts),
            &frame.to_u16_trunc());
    }

    let detrans: Img = if opts.transversalium {
        let (c, b) = if !(cercle0.0 == -1.0 && cercle0.1 == -1.0 && cercle0.2 == -1.0) {
            (cercle0, *borders)
        } else {
            ((0.0, 0.0, 99999.0),
             [0.0, bounds.0 as f64 + 20.0, frame.w as f64 - 1.0, bounds.1 as f64 - 20.0])
        };
        transv::correct_transversalium2(&frame, c, &b, opts.trans_strength, opts.stubborn,
            if !opts.clahe_only && !opts.protus_only {
                Some(procss::output_path(&(basefich.to_string() + "_transversalium_correction.png"), opts))
            } else { None })
    } else {
        frame.to_u16_trunc()
    };

    if opts.save_fit && opts.transversalium {
        let _ = pngio::write_fits16(
            &procss::output_path(&(basefich.to_string() + "_detransversaliumed.fits"), opts),
            &detrans);
    }

    let mut cercle = cercle0;
    let mut detrans = detrans;
    if opts.fixed_width.is_some() || opts.crop_width_square {
        let h = detrans.h as i64;
        let target_w = opts.fixed_width.unwrap_or(h);
        let target_h = if opts.crop_width_square { target_w } else { h };
        let (cx, cy) = procss::get_stable_circle_center(&detrans, cercle0);
        detrans = procss::center_image_around_point(&detrans, (cx, cy), (target_w, target_h));
        if !(cercle.0 == -1.0 && cercle.1 == -1.0 && cercle.2 == -1.0) {
            cercle = ((target_w / 2) as f64, (target_h / 2) as f64, cercle.2);
        }
    }

    let p = procss::image_process(&detrans, cercle, opts, basefich, basefich);
    Some(Preview { clahe: p.clahe, high_contrast: p.high_contrast, protus: p.protus })
}

pub fn pybool(b: bool) -> &'static str {
    if b { "True" } else { "False" }
}

/// Parse "a,b,c" or "x:y" or "x:y:w" pixel-shift specifications.
pub fn parse_shift_spec(s: &str) -> Result<Vec<i64>, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() == 1 {
        if s.trim().is_empty() { return Err("invalid shift input".into()); }
        let mut v = Vec::new();
        for x in s.split(',') {
            v.push(x.trim().parse::<i64>().map_err(|_| "invalid shift input".to_string())?);
        }
        Ok(v)
    } else if parts.len() == 2 {
        let a = parts[0].trim().parse::<i64>().map_err(|_| "invalid shift input".to_string())?;
        let b = parts[1].trim().parse::<i64>().map_err(|_| "invalid shift input".to_string())?;
        Ok((a..=b).collect())
    } else if parts.len() == 3 {
        let a = parts[0].trim().parse::<i64>().map_err(|_| "invalid shift input".to_string())?;
        let b = parts[1].trim().parse::<i64>().map_err(|_| "invalid shift input".to_string())?;
        let st = parts[2].trim().parse::<i64>().map_err(|_| "invalid shift input".to_string())?;
        if st == 0 { return Err("invalid shift input".into()); }
        let mut v = Vec::new();
        if st > 0 {
            let mut x = a;
            while x <= b { v.push(x); x += st; }
        } else {
            let mut x = a;
            while x >= b { v.push(x); x += st; }
        }
        Ok(v)
    } else {
        Err("invalid shift input".into())
    }
}
