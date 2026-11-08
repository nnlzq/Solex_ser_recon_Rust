//! Zero-dependency writers: 16-bit grayscale PNG (stored deflate) and FITS.
//! PNG output is bit-exact in pixel values vs cv2.imwrite (compression differs).

use crate::num::Img;
use std::fs::File;
use std::io::{self, Write};

fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    for i in 0..256usize {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        t[i] = c;
    }
    t
}

fn crc32(data: &[u8]) -> u32 {
    let t = crc_table();
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = t[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut td = Vec::with_capacity(4 + data.len());
    td.extend_from_slice(tag);
    td.extend_from_slice(data);
    out.extend_from_slice(&td);
    out.extend_from_slice(&crc32(&td).to_be_bytes());
}

/// zlib stream using only stored (uncompressed) deflate blocks.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 16);
    out.push(0x78);
    out.push(0x01);
    let mut off = 0usize;
    loop {
        let n = (raw.len() - off).min(65535);
        let last = off + n >= raw.len();
        out.push(if last { 1 } else { 0 });
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(!(n as u16)).to_le_bytes());
        out.extend_from_slice(&raw[off..off + n]);
        off += n;
        if last { break; }
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

/// Write 16-bit grayscale PNG (color type 0, bit depth 16, filter 0 rows).
pub fn write_png16(path: &str, img: &Img) -> io::Result<()> {
    let row_bytes = 1 + img.w * 2;
    let mut raw = Vec::with_capacity(img.h * row_bytes);
    for r in 0..img.h {
        raw.push(0u8);
        let row = img.row(r);
        for &v in row {
            raw.extend_from_slice(&v.to_be_bytes());
        }
    }
    let idat = zlib_stored(&raw);
    let mut out: Vec<u8> = Vec::with_capacity(idat.len() + 64);
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(img.w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(img.h as u32).to_be_bytes());
    ihdr.push(16); // bit depth
    ihdr.push(0);  // color type: grayscale
    ihdr.push(0);  // compression
    ihdr.push(0);  // filter
    ihdr.push(0);  // interlace
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &idat);
    chunk(&mut out, b"IEND", &[]);
    let mut f = File::create(path)?;
    f.write_all(&out)
}

/// Write 8-bit RGB PNG (color type 2, bit depth 8, filter 0 rows).
pub fn write_png_rgb8(path: &str, w: usize, h: usize, rgb: &[u8]) -> io::Result<()> {
    let mut raw = Vec::with_capacity(h * (1 + w * 3));
    for r in 0..h {
        raw.push(0u8);
        raw.extend_from_slice(&rgb[r * w * 3..(r + 1) * w * 3]);
    }
    let idat = zlib_stored(&raw);
    let mut out: Vec<u8> = Vec::with_capacity(idat.len() + 64);
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.push(8);  // bit depth
    ihdr.push(2);  // color type: RGB
    ihdr.push(0);
    ihdr.push(0);
    ihdr.push(0);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &idat);
    chunk(&mut out, b"IEND", &[]);
    let mut f = File::create(path)?;
    f.write_all(&out)
}

/// Write FITS: BITPIX=16 with BZERO=32768 (astropy's uint16 convention),
/// big-endian int16 payload, header padded to a 2880-byte block.
pub fn write_fits16(path: &str, img: &Img) -> io::Result<()> {
    let mut hdr: Vec<u8> = Vec::new();
    let mut card = |s: String| {
        let mut c = s;
        while c.len() < 80 { c.push(' '); }
        hdr.extend_from_slice(c.as_bytes());
    };
    card("SIMPLE  = ".to_string() + &format!("{:>20}", "T"));
    card("BITPIX  = ".to_string() + &format!("{:>20}", 16));
    card("NAXIS   = ".to_string() + &format!("{:>20}", 2));
    card("NAXIS1  = ".to_string() + &format!("{:>20}", img.w as i64));
    card("NAXIS2  = ".to_string() + &format!("{:>20}", img.h as i64));
    card("BZERO   = ".to_string() + &format!("{:>20}", 32768));
    card("BSCALE  = ".to_string() + &format!("{:>20}", 1));
    card("BIN1    = ".to_string() + &format!("{:>20}", 1));
    card("BIN2    = ".to_string() + &format!("{:>20}", 1));
    card("EXPTIME = ".to_string() + &format!("{:>20}", 0));
    card("END".to_string());
    while hdr.len() % 2880 != 0 {
        hdr.extend_from_slice(b" ");
    }
    let mut f = File::create(path)?;
    f.write_all(&hdr)?;
    let mut buf = Vec::with_capacity(img.d.len() * 2);
    for &v in &img.d {
        // unsigned -> signed offset storage
        buf.extend_from_slice(&((v as i32 - 32768) as i16).to_be_bytes());
    }
    f.write_all(&buf)?;
    let rem = buf.len() % 2880;
    if rem != 0 {
        f.write_all(&vec![0u8; 2880 - rem])?;
    }
    Ok(())
}
