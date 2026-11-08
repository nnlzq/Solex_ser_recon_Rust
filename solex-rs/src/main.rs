//! Solex SER solar spectrogram reconstruction - Rust port.
//! Command line front end (SHG_MAIN.py compatible) plus the native GUI.

mod clahe;
mod edge;
mod fitline;
mod geom;
mod gui;
mod num;
mod pipeline;
mod plot;
mod pngio;
mod procss;
mod recon;
mod ser;
mod transv;
mod vig;

use pipeline::{parse_shift_spec, solex_read, solex_process};
use procss::Options;

fn usage() -> String {
    let mut u = String::new();
    u += "solex-rs [-hwdxfcpstmr] [-o DIR] [file(s) to treat]\n";
    u += "'h' : 'Help', display help menu.\n";
    u += "'w' : 'a,b,c, ...'  produce images at a, b, c ... pixels.\n";
    u += "'w' : 'x:y:w'  produce images starting at x, finishing at y, every w pixels.\n";
    u += "'d' : 'flag_display', display all graphics (no-op in CLI build)\n";
    u += "'x' : 'ratio_fixe', disable ellipse fitting\n";
    u += "'f' : 'save_fit', save all fits files (False by default)\n";
    u += "'c' : 'clahe_only',  only final clahe image is saved (False by default)\n";
    u += "'p' : 'disk_display' turn off black disk with protuberance images (False by default)\n";
    u += "'s' : 'crop_square_width', crop the width to equal the height (False by default)\n";
    u += "'t' : 'disable transversalium', disable transversalium correction (False by default)\n";
    u += "'m' : 'mirror flip', mirror flip in x-direction (False by default)\n";
    u += "'r' : 'w'  crop width to a constant no. of pixels.\n";
    u += "'o' : 'DIR'  write outputs into DIR (extension).\n";
    u += "Run without arguments to open the graphical interface.\n";
    u
}

fn treat_flag_at_cli(args: &[String], idx: &mut usize, opts: &mut Options) -> bool {
    // returns false to exit (help)
    let argument = &args[*idx];
    let chars: Vec<char> = argument.chars().skip(1).collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        match c {
            'h' => { print!("{}", usage()); std::process::exit(0); }
            'w' => {
                let mut shift = String::new();
                i += 1;
                while i < chars.len() {
                    let ch = chars[i];
                    if ch.is_ascii_digit() || ch == ':' || ch == ',' || ch == '-' {
                        shift.push(ch);
                        i += 1;
                    } else {
                        i += 1;
                        break;
                    }
                }
                match parse_shift_spec(&shift) {
                    Ok(v) => opts.shift = v,
                    Err(e) => { eprintln!("{}", e); std::process::exit(1); }
                }
                continue;
            }
            't' => { opts.transversalium = false; i += 1; }
            'p' => { opts.disk_display = false; i += 1; }
            'x' => { opts.ratio_fixe = Some(1.0); i += 1; }
            'r' => {
                let mut fw = String::new();
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    fw.push(chars[i]);
                    i += 1;
                }
                opts.fixed_width = Some(fw.parse::<i64>().unwrap_or(0));
                continue;
            }
            'd' => { opts.flag_display = true; i += 1; }
            'f' => { opts.save_fit = true; i += 1; }
            'c' => { opts.clahe_only = true; i += 1; }
            's' => { opts.crop_width_square = true; i += 1; }
            'm' => { opts.flip_x = true; i += 1; }
            'o' => {
                // extension: -o DIR (DIR is the next argv token)
                if *idx + 1 < args.len() {
                    opts.output_dir = args[*idx + 1].clone();
                    *idx += 1;
                }
                i += 1;
            }
            _ => {
                eprintln!("ERROR !!! At least one argument is not accepted");
                eprint!("{}", usage());
                i += 1;
            }
        }
    }
    true
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        if let Err(e) = gui::run() {
            eprintln!("ERROR starting GUI: {}", e);
            std::process::exit(1);
        }
        return;
    }

    let mut opts = Options::default();
    let mut files: Vec<String> = Vec::new();
    let mut i = 1usize;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with('-') {
            treat_flag_at_cli(&args, &mut i, &mut opts);
        } else {
            let ext = a.rsplit('.').next().unwrap_or("").to_uppercase();
            if ext == "SER" || ext == "AVI" {
                files.push(a.clone());
            } else {
                println!("WARNING: {} was not a valid SER or AVI file name and was ignored. Remember to use \"-\" if you want to input a flag", a);
            }
        }
        i += 1;
    }
    if files.is_empty() {
        eprint!("{}", usage());
        std::process::exit(1);
    }
    println!("theses files are going to be processed : {:?}", files);

    let nthreads = opts.nthreads;
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for f in &files {
            let f = f.clone();
            let mut o = opts.clone();
            o.nthreads = nthreads;
            handles.push(s.spawn(move || {
                println!("file {} is processing", f);
                if f.to_ascii_lowercase().ends_with(".avi") {
                    eprintln!(
                        "ERROR processing {}: AVI is not supported by this build; please provide a SER file",
                        f
                    );
                    return;
                }
                let t0 = std::time::Instant::now();
                match solex_read(&f, &mut o) {
                    Ok(res) => {
                        let t_read = t0.elapsed();
                        let t1 = std::time::Instant::now();
                        solex_process(&f, &mut o, &res.disks, res.bounds);
                        if std::env::var("SOLEX_DEBUG").is_ok() {
                            eprintln!("DBG timing: read={:?} process={:?}", t_read, t1.elapsed());
                        }
                    }
                    Err(e) => {
                        eprintln!("ERROR processing {}: {}", f, e);
                    }
                }
            }));
        }
        for h in handles {
            let _ = h.join();
        }
    });
}
