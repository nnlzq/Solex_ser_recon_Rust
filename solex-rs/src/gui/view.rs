//! GUI helpers: PNG decoding (flags) and grayscale image -> egui textures.

use crate::num::Img;
use egui::ColorImage;
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

static TEX_SEQ: AtomicUsize = AtomicUsize::new(0);

pub fn decode_png(bytes: &[u8]) -> Option<ColorImage> {
    let mut dec = png::Decoder::new(Cursor::new(bytes));
    // Expand palette images to RGB(A) and low bit-depth grayscale to 8-bit.
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let data = &buf[..info.buffer_size()];
    let (w, h) = (info.width as usize, info.height as usize);
    let mut rgba = vec![0u8; w * h * 4];
    match info.color_type {
        png::ColorType::Rgba => rgba.copy_from_slice(data),
        png::ColorType::Rgb => {
            for i in 0..w * h {
                rgba[i * 4..i * 4 + 3].copy_from_slice(&data[i * 3..i * 3 + 3]);
                rgba[i * 4 + 3] = 255;
            }
        }
        png::ColorType::Grayscale => {
            for i in 0..w * h {
                let v = data[i];
                rgba[i * 4] = v;
                rgba[i * 4 + 1] = v;
                rgba[i * 4 + 2] = v;
                rgba[i * 4 + 3] = 255;
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for i in 0..w * h {
                let v = data[i * 2];
                rgba[i * 4] = v;
                rgba[i * 4 + 1] = v;
                rgba[i * 4 + 2] = v;
                rgba[i * 4 + 3] = data[i * 2 + 1];
            }
        }
        png::ColorType::Indexed => return None,
    }
    Some(ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}

/// Nearest-neighbour grayscale conversion, min/max normalised, downscaled so
/// that neither dimension exceeds `maxdim`.
pub fn gray_color_image(img: &Img, maxdim: usize) -> ColorImage {
    let (oh, ow) = downscaled_size(img.h, img.w, maxdim);
    let lo = img.min() as f64;
    let hi = img.max() as f64;
    let span = if hi > lo { hi - lo } else { 1.0 };
    let mut rgba = vec![0u8; ow * oh * 4];
    for y in 0..oh {
        let sy = (y * img.h / oh).min(img.h - 1);
        for x in 0..ow {
            let sx = (x * img.w / ow).min(img.w - 1);
            let v = (((img.g(sy, sx) as f64 - lo) / span) * 255.0).round().clamp(0.0, 255.0) as u8;
            let i = (y * ow + x) * 4;
            rgba[i] = v;
            rgba[i + 1] = v;
            rgba[i + 2] = v;
            rgba[i + 3] = 255;
        }
    }
    ColorImage::from_rgba_unmultiplied([ow, oh], &rgba)
}

pub fn downscaled_size(h: usize, w: usize, maxdim: usize) -> (usize, usize) {
    let m = h.max(w);
    if m <= maxdim {
        return (h.max(1), w.max(1));
    }
    let f = maxdim as f64 / m as f64;
    (((h as f64 * f) as usize).max(1), ((w as f64 * f) as usize).max(1))
}

pub fn make_texture(ctx: &egui::Context, img: &Img, maxdim: usize) -> egui::TextureHandle {
    let n = TEX_SEQ.fetch_add(1, Ordering::Relaxed);
    ctx.load_texture(format!("img-{}", n), gray_color_image(img, maxdim), egui::TextureOptions::LINEAR)
}

pub fn make_texture_ci(ctx: &egui::Context, ci: ColorImage) -> egui::TextureHandle {
    let n = TEX_SEQ.fetch_add(1, Ordering::Relaxed);
    ctx.load_texture(format!("ci-{}", n), ci, egui::TextureOptions::LINEAR)
}

#[cfg(test)]
mod tests {
    use super::decode_png;

    #[test]
    fn embedded_flags_decode() {
        for lang in crate::gui::resources::langs() {
            assert!(decode_png(lang.flag).is_some(), "failed to decode flag for {}", lang.name);
        }
    }
}
