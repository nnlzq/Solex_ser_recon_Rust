//! Build script: turns assets/sun_icon.png into a multi-size Windows .ico,
//! embeds it into the executable as a resource, and leaves a 256x256 PNG
//! in OUT_DIR for the runtime window/taskbar icon.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const SIZES: [usize; 7] = [16, 24, 32, 48, 64, 128, 256];

fn main() {
    println!("cargo:rerun-if-changed=assets/sun_icon.png");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let (w, h, master) = match read_png(Path::new("assets/sun_icon.png")) {
        Some(v) => v,
        None => return,
    };
    let master = key_out_background(master, w, h);

    let mut images = Vec::new();
    for &size in SIZES.iter() {
        let rgba = resize(&master, w, h, size);
        images.push((size, encode_png(size, size, &rgba)));
        if size == 256 {
            let mut f = fs::File::create(out_dir.join("icon_256.png")).expect("icon_256.png");
            f.write_all(&images.last().unwrap().1).expect("write icon_256.png");
        }
    }
    let ico = build_ico(&images);
    let ico_path = out_dir.join("sun_icon.ico");
    fs::write(&ico_path, &ico).expect("write ico");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let Some(rc) = find_rc() else {
        println!("cargo:warning=rc.exe not found; exe icon will not be embedded");
        return;
    };
    let rc_file = out_dir.join("app.rc");
    let ico_str = ico_path.display().to_string().replace('\\', "\\\\");
    fs::write(&rc_file, format!("1 ICON \"{}\"\n", ico_str)).expect("write rc");
    let res_file = out_dir.join("app.res");
    let rc_dir = rc.parent().unwrap();
    let path = format!(
        "{};{}",
        rc_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let status = Command::new(&rc)
        .env("PATH", path)
        .arg("/nologo")
        .arg(format!("/fo{}", res_file.display()))
        .arg(&rc_file)
        .status()
        .expect("run rc.exe");
    if !status.success() {
        println!("cargo:warning=rc.exe failed; exe icon will not be embedded");
        return;
    }
    println!("cargo:rustc-link-arg={}", res_file.display());
}

fn read_png(path: &Path) -> Option<(usize, usize, Vec<u8>)> {
    let file = fs::File::open(path).ok()?;
    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let mut rgba = Vec::with_capacity((info.width * info.height * 4) as usize);
    match info.color_type {
        png::ColorType::Rgba => rgba.extend_from_slice(&buf[..info.buffer_size()]),
        png::ColorType::Rgb => {
            for c in buf[..info.buffer_size()].chunks(3) {
                rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        _ => return None,
    }
    Some((info.width as usize, info.height as usize, rgba))
}

/// Makes the white page background transparent (flood fill from the border)
/// and fades anti-aliased white fringes so the sun sits cleanly on any
/// desktop colour.
fn key_out_background(mut rgba: Vec<u8>, w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    let mut seen = vec![false; n];
    let mut stack = Vec::new();
    let white = |d: &[u8]| d[0] >= 225 && d[1] >= 225 && d[2] >= 225;
    for i in 0..n {
        let (x, y) = (i % w, i / w);
        if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
            if white(&rgba[i * 4..i * 4 + 4]) {
                seen[i] = true;
                stack.push(i);
            }
        }
    }
    while let Some(i) = stack.pop() {
        rgba[i * 4 + 3] = 0;
        let (x, y) = (i % w, i / w);
        let push = |j: usize, seen: &mut Vec<bool>, stack: &mut Vec<usize>| {
            if !seen[j] && white(&rgba[j * 4..j * 4 + 4]) {
                seen[j] = true;
                stack.push(j);
            }
        };
        if x > 0 {
            push(i - 1, &mut seen, &mut stack);
        }
        if x + 1 < w {
            push(i + 1, &mut seen, &mut stack);
        }
        if y > 0 {
            push(i - w, &mut seen, &mut stack);
        }
        if y + 1 < h {
            push(i + w, &mut seen, &mut stack);
        }
    }
    for i in 0..n {
        if rgba[i * 4 + 3] == 0 {
            continue;
        }
        let min = rgba[i * 4].min(rgba[i * 4 + 1]).min(rgba[i * 4 + 2]);
        if min > 200 {
            let fade = (255 - min) as f64 / 55.0 * 255.0;
            rgba[i * 4 + 3] = fade.min(255.0) as u8;
        }
    }
    rgba
}

/// Alpha-aware box filter downscale (premultiplied, so edges stay clean).
fn resize(master: &[u8], mw: usize, mh: usize, size: usize) -> Vec<u8> {
    let mut out = vec![0u8; size * size * 4];
    for y in 0..size {
        let y0 = (y * mh / size).min(mh - 1);
        let y1 = (((y + 1) * mh / size).max(y0 + 1)).min(mh);
        for x in 0..size {
            let x0 = (x * mw / size).min(mw - 1);
            let x1 = (((x + 1) * mw / size).max(x0 + 1)).min(mw);
            let (mut r, mut g, mut b, mut a, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let i = (yy * mw + xx) * 4;
                    let al = master[i + 3] as f64 / 255.0;
                    r += master[i] as f64 * al;
                    g += master[i + 1] as f64 * al;
                    b += master[i + 2] as f64 * al;
                    a += al;
                    n += 1.0;
                }
            }
            let j = (y * size + x) * 4;
            if a > 0.0 {
                out[j] = (r / a) as u8;
                out[j + 1] = (g / a) as u8;
                out[j + 2] = (b / a) as u8;
                out[j + 3] = (a / n * 255.0) as u8;
            }
        }
    }
    out
}

fn encode_png(w: usize, h: usize, rgba: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut buf, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().expect("png header");
        writer.write_image_data(rgba).expect("png data");
    }
    buf
}

/// ICO container with PNG-compressed entries (supported by Windows Vista+).
fn build_ico(images: &[(usize, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(images.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * images.len();
    for (size, data) in images {
        let dim = if *size >= 256 { 0u8 } else { *size as u8 };
        out.push(dim);
        out.push(dim);
        out.push(0);
        out.push(0);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += data.len();
    }
    for (_, data) in images {
        out.extend_from_slice(data);
    }
    out
}

fn find_rc() -> Option<PathBuf> {
    let base = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut versions: Vec<PathBuf> = fs::read_dir(base)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir() && p.file_name().map_or(false, |n| n.to_string_lossy().starts_with("10."))
        })
        .collect();
    versions.sort();
    for v in versions.iter().rev() {
        for arch in ["x64", "x86"] {
            let p = v.join(arch).join("rc.exe");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}
