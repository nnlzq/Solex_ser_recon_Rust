//! SER file reader (video_reader.py equivalent).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub struct SerInfo {
    pub width: usize,
    pub height: usize,
    pub depth: usize, // bytes per pixel: 1 or 2
    pub frame_count: usize,
    pub flag_rotate: bool,
    pub ih: usize,
    pub iw: usize,
}

impl SerInfo {
    pub fn probe(path: &str) -> std::io::Result<SerInfo> {
        let mut f = File::open(path)?;
        // Header layout: 14 FileID, u32 LuID(14), u32 ColorID(18), u32 endian(22),
        // u32 Width(26), u32 Height(30), u32 PixelDepthPerPlane(34), u32 FrameCount(38)
        let mut hdr = [0u8; 42];
        f.read_exact(&mut hdr)?;
        let u32b = |o: usize| -> u32 {
            u32::from_le_bytes([hdr[o], hdr[o + 1], hdr[o + 2], hdr[o + 3]])
        };
        // Python's video_reader ignores the littleEndian field; we read LE unconditionally.
        let width = u32b(26) as usize;
        let height = u32b(30) as usize;
        let depth_bytes = (u32b(34) as usize + 7) / 8;
        let frame_count = u32b(38) as usize;
        if width == 0 || height == 0 || frame_count == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad SER header"));
        }
        let flag_rotate = width > height;
        let (ih, iw) = if flag_rotate { (width, height) } else { (height, width) };
        Ok(SerInfo { width, height, depth: depth_bytes, frame_count, flag_rotate, ih, iw })
    }
}

pub const DATA_OFFSET: u64 = 178;

/// Read one frame as u16 in rotated orientation (ih x iw).
pub fn read_frame(f: &mut File, info: &SerInfo, idx: usize) -> std::io::Result<Vec<u16>> {
    let (h, w) = (info.height, info.width);
    let n = h * w;
    let mut buf = vec![0u8; n * info.depth];
    f.seek(SeekFrom::Start(DATA_OFFSET + idx as u64 * (n * info.depth) as u64))?;
    f.read_exact(&mut buf)?;
    let mut raw = vec![0u16; n];
    if info.depth == 2 {
        for i in 0..n {
            raw[i] = u16::from_le_bytes([buf[2 * i], buf[2 * i + 1]]);
        }
    } else {
        for i in 0..n {
            raw[i] = (buf[i] as u16) * 256;
        }
    }
    if !info.flag_rotate {
        return Ok(raw);
    }
    // np.rot90 CCW: out[i][j] = raw[j][W-1-i], out shape (W, H)
    let mut out = vec![0u16; n];
    for i in 0..w {
        let src_col = w - 1 - i;
        for j in 0..h {
            out[i * h + j] = raw[j * w + src_col];
        }
    }
    Ok(out)
}

pub fn open(path: &str) -> std::io::Result<File> {
    File::open(path)
}
