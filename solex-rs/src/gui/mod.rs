//! Native GUI (egui/eframe) reproducing the FreeSimpleGUI front end of SHG_MAIN.py.

pub mod resources;
pub mod spectral;
pub mod view;

use crate::pipeline::{self, Preview};
use crate::procss::{self, Options};
use egui::Context;
use resources::Lang;
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    File,
    Folder,
}

struct PreviewState {
    file: String,
    mean: crate::num::Img,
    curve: Vec<f64>,
    pv: Preview,
    mean_tex: Option<egui::TextureHandle>,
    sun_tex: Option<[egui::TextureHandle; 3]>,
}

enum Msg {
    Current(String),
    Progress(usize, String),
    Preview(String, crate::num::Img, Vec<f64>, Preview, String),
    Done,
}

struct Job {
    rx: Receiver<Msg>,
    total: usize,
    par: usize,
    continuous: bool,
    #[allow(dead_code)]
    done: Arc<AtomicUsize>,
}

/// Labels pulled from the selected language pack (English fallback is built in).
struct L {
    title: String,
    files: String,
    choose_files: String,
    folder_mode: String,
    folder: String,
    file_mode: String,
    choose_input_folder: String,
    continuous: String,
    out_label: String,
    choose_out_folder: String,
    show_graphics: String,
    save_fits: String,
    clahe_only: String,
    protus_only: String,
    crop_square: String,
    fixed_width: String,
    mirror_x: String,
    rotate: String,
    transversalium: String,
    stubborn: String,
    de_vignette: String,
    trans_strength: String,
    ratio: String,
    tilt: String,
    pixel_offset: String,
    protus_adj: String,
    ellipse_shift: String,
    pixel_live: String,
    open_out_folder: String,
    no_file: String,
    no_folder: String,
    pixel_tip: String,
    protus_tip: String,
    ellipse_tip: String,
}

pub struct App {
    langs: Vec<Lang>,
    lang_idx: usize,
    flag_tex: Option<egui::TextureHandle>,
    cfg: Map<String, Value>,

    mode: Mode,
    file_text: String,
    folder_text: String,
    output_text: String,
    continuous: bool,
    show_graphics: bool,
    save_fit: bool,
    clahe_only: bool,
    protus_only: bool,
    crop_square: bool,
    fixed_width_text: String,
    mirror_x: bool,
    rotate: f64,
    transversalium: bool,
    stubborn: bool,
    de_vignette: bool,
    trans_strength: f64,
    ratio_text: String,
    tilt_text: String,
    pixel_text: String,
    protus_text: String,
    ellipse_text: String,

    status: String,
    popup: Option<String>,
    job: Option<Job>,
    current_file: String,
    progress_hidden: bool,
    auto_open: bool,
    auto_done: usize,
    auto_status: String,
    auto_last: String,
    auto_title: String,
    auto_tex: Option<egui::TextureHandle>,
    auto_last_scan: std::time::Instant,
    /// New file awaiting a stable size: (path, size, when first seen).
    auto_pending: Option<(String, u64, std::time::Instant)>,
    processed: HashSet<String>,
    auto_stop: bool,
    work_dir: String,

    preview: Option<PreviewState>,
    preview_mean_open: bool,
    preview_sun_open: bool,

    spectral: Option<spectral::Spectral>,
    sized: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> App {
        install_fonts(&cc.egui_ctx);
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        let cfg = resources::load_config();
        let langs = resources::langs();
        let lang_name = resources::cfg_str(&cfg, "language", "English");
        let lang_idx = langs.iter().position(|l| l.name == lang_name).unwrap_or(1);

        let fixed_width_text = match cfg.get("fixed_width") {
            Some(Value::Number(n)) => n.to_string(),
            _ => String::new(),
        };
        let rotate = cfg.get("img_rotate").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let trans_strength = cfg.get("trans_strength").and_then(|v| v.as_f64()).unwrap_or(301.0) / 100.0;
        let protus_text = cfg
            .get("delta_radius")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            .to_string();
        let ellipse_text = cfg
            .get("ellipse_fit_shift")
            .and_then(|v| v.as_i64())
            .unwrap_or(10)
            .to_string();
        let mode = if resources::cfg_str(&cfg, "selected_mode", "File input mode") == "Folder input mode" {
            Mode::Folder
        } else {
            Mode::File
        };
        let out_text = resources::cfg_str(&cfg, "output_dir", "");
        let folder_text = resources::cfg_str(&cfg, "input_dir", "");
        let work_dir = resources::cfg_str(&cfg, "workDir", "");

        let mut app = App {
            langs,
            lang_idx,
            flag_tex: None,
            cfg,
            mode,
            file_text: work_dir.clone(),
            folder_text,
            output_text: out_text,
            continuous: false,
            show_graphics: false,
            save_fit: false,
            clahe_only: false,
            protus_only: false,
            crop_square: false,
            fixed_width_text,
            mirror_x: false,
            rotate,
            transversalium: true,
            stubborn: false,
            de_vignette: false,
            trans_strength,
            ratio_text: String::new(),
            tilt_text: String::new(),
            pixel_text: "0".to_string(),
            protus_text,
            ellipse_text,
            status: "Ready.".to_string(),
            popup: None,
            job: None,
            current_file: String::new(),
            progress_hidden: false,
            auto_open: false,
            auto_done: 0,
            auto_status: "Looking for files ...".to_string(),
            auto_last: String::new(),
            auto_title: "Continuous processing mode".to_string(),
            auto_tex: None,
            auto_last_scan: std::time::Instant::now(),
            auto_pending: None,
            processed: HashSet::new(),
            auto_stop: false,
            work_dir,
            preview: None,
            preview_mean_open: false,
            preview_sun_open: false,
            spectral: None,
            sized: false,
        };
        app.load_cfg_values();
        app.load_flag(&cc.egui_ctx);
        app
    }

    fn load_cfg_values(&mut self) {
        let c = &self.cfg;
        self.continuous = resources::cfg_bool(c, "continuous_detect_mode", false);
        self.show_graphics = resources::cfg_bool(c, "flag_display", false);
        self.save_fit = resources::cfg_bool(c, "save_fit", false);
        self.clahe_only = resources::cfg_bool(c, "clahe_only", false);
        self.protus_only = resources::cfg_bool(c, "protus_only", false);
        self.crop_square = resources::cfg_bool(c, "crop_width_square", false);
        self.mirror_x = resources::cfg_bool(c, "flip_x", false);
        self.transversalium = resources::cfg_bool(c, "transversalium", true);
        self.stubborn = resources::cfg_bool(c, "stubborn_transversalium", false);
        self.de_vignette = resources::cfg_bool(c, "de-vignette", false);
    }

    fn load_flag(&mut self, ctx: &Context) {
        let png = self.langs[self.lang_idx].flag;
        self.flag_tex = view::decode_png(png).map(|ci| view::make_texture_ci(ctx, ci));
    }

    fn tr(&self, key: &str) -> String {
        self.langs[self.lang_idx]
            .dict
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or(key)
            .to_string()
    }

    fn labels(&self) -> L {
        L {
            title: self.tr("Solar disk reconstruction from SHG video files"),
            files: self.tr("File(s)"),
            choose_files: self.tr("Choose file(s)"),
            folder_mode: self.tr("Folder input mode"),
            folder: self.tr("Folder"),
            file_mode: self.tr("File input mode"),
            choose_input_folder: self.tr("Choose input folder"),
            continuous: self.tr("Continuous detect mode"),
            out_label: self.tr("Output Folder (blank for same as input):"),
            choose_out_folder: self.tr("Choose output folder"),
            show_graphics: self.tr("Show graphics"),
            save_fits: self.tr("Save fits files"),
            clahe_only: self.tr("Save clahe.png only"),
            protus_only: self.tr("Save protus.png only"),
            crop_square: self.tr("Crop square"),
            fixed_width: self.tr("Fixed image width (blank for none)"),
            mirror_x: self.tr("Mirror X"),
            rotate: self.tr("Rotate png images:"),
            transversalium: self.tr("Correct transversalium lines"),
            stubborn: self.tr("Stubborn transversalium"),
            de_vignette: self.tr("de-vignette"),
            trans_strength: self.tr("Transversalium correction strength (pixels x 100) :"),
            ratio: self.tr("Y/X ratio (blank for auto)"),
            tilt: self.tr("Tilt angle (blank for auto)"),
            pixel_offset: self.tr("Pixel offset"),
            protus_adj: self.tr("Protus adjustment"),
            ellipse_shift: self.tr("Ellipse fit shift [advanced]"),
            pixel_live: self.tr("Pixel offset live"),
            open_out_folder: self.tr("Open output folder"),
            no_file: self.tr("no_file_error"),
            no_folder: self.tr("no_folder_error"),
            pixel_tip: self.tr("pixel_offset_tooltip"),
            protus_tip: self.tr("protus_adjustment_tooltip"),
            ellipse_tip: self.tr("ellipse_fit_shift_tooltip"),
        }
    }

    /// UI_handler.interpret_UI_values: build Options from the form.
    fn build_options(&self, no_file: bool) -> Result<Options, String> {
        let mut o = Options::default();
        o.shift = pipeline::parse_shift_spec(&self.pixel_text)
            .map_err(|_| "invalid pixel offset value!".to_string())?;
        if o.shift.is_empty() {
            return Err("Error: pixel offset input lower bound greater than upper bound!".into());
        }
        o.flag_display = self.show_graphics;
        o.ratio_fixe = if self.ratio_text.trim().is_empty() {
            None
        } else {
            Some(self.ratio_text.trim().parse::<f64>().map_err(|_| "invalid Y/X ratio value")?)
        };
        o.slant_fix = if self.tilt_text.trim().is_empty() {
            None
        } else {
            Some(self.tilt_text.trim().parse::<f64>().map_err(|_| "invalid tilt angle value!")?)
        };
        o.fixed_width = if self.fixed_width_text.trim().is_empty() {
            None
        } else {
            Some(self.fixed_width_text.trim().parse::<i64>().map_err(|_| "invalid fixed width value!")?)
        };
        o.delta_radius = self
            .protus_text
            .trim()
            .parse::<i64>()
            .map_err(|_| "invalid protus_radius_adjustment".to_string())?;
        o.ellipse_fit_shift = if self.ellipse_text.trim().is_empty() {
            10
        } else {
            self.ellipse_text.trim().parse::<i64>().map_err(|_| "invalid ellipse_fit_shift!".to_string())?
        };
        o.save_fit = self.save_fit;
        o.clahe_only = self.clahe_only;
        o.protus_only = self.protus_only;
        o.crop_width_square = self.crop_square;
        o.transversalium = self.transversalium;
        o.stubborn = self.stubborn;
        o.trans_strength = (self.trans_strength * 100.0) as i64 + 1;
        o.flip_x = self.mirror_x;
        o.img_rotate = self.rotate as i64;
        o.disk_display = true;
        o.output_dir = self.output_text.trim().to_string();
        o.de_vignette = self.de_vignette;
        if !o.output_dir.is_empty() && !Path::new(&o.output_dir).is_dir() {
            return Err("ERROR opening output folder :".to_string() + &o.output_dir);
        }
        if !no_file {
            match self.mode {
                Mode::File => {
                    let files = self.file_list();
                    if files.is_empty() {
                        return Err("ERROR opening file :!".into());
                    }
                    for f in &files {
                        if std::fs::File::open(f).is_err() {
                            return Err("ERROR opening file :".to_string() + f + "!");
                        }
                    }
                }
                Mode::Folder => {
                    if self.folder_text.trim().is_empty() {
                        return Err("ERROR opening folder :!".into());
                    }
                    if !Path::new(&self.folder_text).is_dir() {
                        return Err("ERROR opening folder :".to_string() + &self.folder_text);
                    }
                }
            }
        }
        Ok(o)
    }

    fn file_list(&self) -> Vec<String> {
        self.file_text
            .split(';')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn start_files(&mut self, files: Vec<String>, opts: Options, ctx: &Context, continuous: bool) {
        let total = files.len();
        let done = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = channel();
        let done2 = done.clone();
        let ctx2 = ctx.clone();
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let par = (cpus / 2).max(1).min(total.max(1));
        std::thread::spawn(move || {
            let t_start = std::time::Instant::now();
            let nthreads = (cpus / par).max(1);
            println!(
                "number of files todo: {} ({} parallel worker(s), {} thread(s) each)",
                total, par, nthreads
            );
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..par {
                    let tx = tx.clone();
                    let files = &files;
                    let opts = &opts;
                    let next = &next;
                    let done = &done2;
                    let ctx2 = &ctx2;
                    s.spawn(move || loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= files.len() {
                            break;
                        }
                        let f = files[i].clone();
                        if f.to_ascii_lowercase().ends_with(".avi") {
                            eprintln!(
                                "ERROR processing {}: AVI is not supported by this build; please provide a SER file",
                                f
                            );
                            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                            let _ = tx.send(Msg::Progress(d, f.clone()));
                            ctx2.request_repaint();
                            continue;
                        }
                        let mut o = opts.clone();
                        o.nthreads = nthreads;
                        println!("file {} is processing", f);
                        let _ = tx.send(Msg::Current(f.clone()));
                        let t_file = std::time::Instant::now();
                        // A corrupt/partial file can make the pipeline panic
                        // (e.g. unwrap on a short read). Keep the failure local
                        // to this file so the remaining files still run.
                        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            pipeline::solex_read(&f, &mut o).map(|res| {
                                let pvs = pipeline::solex_process(&f, &mut o, &res.disks, res.bounds);
                                (res, pvs)
                            })
                        }));
                        match outcome {
                            Ok(Ok((res, pvs))) => {
                                println!(
                                    "  read ok: {} disk(s), mean {}x{}, bounds ({}, {})",
                                    res.disks.len(), res.mean.w, res.mean.h, res.bounds.0, res.bounds.1
                                );
                                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                                println!(
                                    "file {} done ({}/{}) in {:.1}s",
                                    f,
                                    d,
                                    total,
                                    t_file.elapsed().as_secs_f64()
                                );
                                let _ = tx.send(Msg::Progress(d, f.clone()));
                                if let Some(p) = pvs.into_iter().last() {
                                    let last_shift = o.shift_requested.last().copied().unwrap_or(0);
                                    let base = Path::new(&f).with_extension("");
                                    let png = procss::output_path(
                                        &format!("{}_shift={}_clahe.png", base.to_string_lossy(), last_shift),
                                        &o,
                                    );
                                    let _ = tx.send(Msg::Preview(f, res.mean, res.curve, p, png));
                                }
                            }
                            Ok(Err(e)) => {
                                eprintln!(
                                    "ERROR processing {} (after {:.1}s): {}",
                                    f,
                                    t_file.elapsed().as_secs_f64(),
                                    e
                                );
                                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                                let _ = tx.send(Msg::Progress(d, f.clone()));
                            }
                            Err(p) => {
                                eprintln!(
                                    "ERROR processing {} (after {:.1}s): {}",
                                    f,
                                    t_file.elapsed().as_secs_f64(),
                                    panic_text(p.as_ref())
                                );
                                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                                let _ = tx.send(Msg::Progress(d, f.clone()));
                            }
                        }
                        ctx2.request_repaint();
                    });
                }
            });
            println!(
                "\n----> Time cost ({}-Process Mode):  {:.3}s\n",
                par,
                t_start.elapsed().as_secs_f64()
            );
            let _ = tx.send(Msg::Done);
            ctx2.request_repaint();
        });
        self.job = Some(Job { rx, total, par, continuous, done });
        self.current_file.clear();
        self.progress_hidden = false;
        self.status = format!("Processing {} file(s)...", total);
    }

    fn poll_job(&mut self, ctx: &Context) {
        let mut finished = false;
        let mut msgs: Vec<Msg> = Vec::new();
        if let Some(job) = &self.job {
            while let Ok(m) = job.rx.try_recv() {
                msgs.push(m);
                if msgs.len() > 64 {
                    break;
                }
            }
        }
        for m in msgs {
            match m {
                Msg::Current(f) => {
                    self.current_file = f;
                }
                Msg::Progress(d, f) => {
                    let total = self.job.as_ref().map(|j| j.total).unwrap_or(d);
                    self.status = format!("Completed {}/{} tasks ({})", d, total, f);
                }
                Msg::Preview(file, mean, curve, pv, png) => {
                    if self.auto_open {
                        self.auto_tex = Some(view::make_texture(ctx, &pv.clahe, 600));
                        self.auto_last = png;
                    }
                    if self.show_graphics {
                        let mean_tex = Some(view::make_texture(ctx, &mean, 900));
                        let sun_tex = Some([
                            view::make_texture(ctx, &pv.clahe, 900),
                            view::make_texture(ctx, &pv.high_contrast, 900),
                            view::make_texture(ctx, &pv.protus, 900),
                        ]);
                        self.preview = Some(PreviewState { file, mean, curve, pv, mean_tex, sun_tex });
                        self.preview_mean_open = true;
                        self.preview_sun_open = true;
                    }
                }
                Msg::Done => finished = true,
            }
        }
        if finished {
            let total = self.job.as_ref().map(|j| j.total).unwrap_or(0);
            let cont_job = self.job.as_ref().map(|j| j.continuous).unwrap_or(false);
            self.status = format!("Done. {} file(s) processed.", total);
            self.job = None;
            if cont_job && !self.auto_stop {
                self.scan_continuous(ctx);
            } else {
                self.auto_stop = false;
                if cont_job {
                    self.auto_open = false;
                    self.auto_tex = None;
                    self.auto_pending = None;
                }
            }
        }
    }

    fn scan_continuous(&mut self, ctx: &Context) {
        const SETTLE: std::time::Duration = std::time::Duration::from_secs(2);

        // A freshly created file may still be written by the acquisition
        // software: remember its size, wait SETTLE, then only start it when
        // the size no longer changes.
        if let Some((f, size, seen)) = self.auto_pending.clone() {
            let now = match std::fs::metadata(&f).map(|m| m.len()).ok() {
                Some(n) => n,
                None => {
                    // File disappeared; look for another one.
                    self.auto_pending = None;
                    return;
                }
            };
            if now != size {
                // Still growing: restart the settle timer with the new size.
                self.auto_pending = Some((f, now, std::time::Instant::now()));
                self.waiting_status(ctx);
                return;
            }
            if seen.elapsed() < SETTLE {
                self.waiting_status(ctx);
                return;
            }
            self.auto_pending = None;
            self.start_continuous_file(f, ctx);
            return;
        }

        let dir = self.folder_text.clone();
        if let Some(f) = scan_folder(&dir, &self.processed).into_iter().next() {
            let size = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
            self.auto_pending = Some((f, size, std::time::Instant::now()));
            self.waiting_status(ctx);
        } else {
            self.status = "Looking for files ...".to_string();
            if self.auto_open {
                self.auto_status = "Looking for files ...".to_string();
            }
            ctx.request_repaint_after(std::time::Duration::from_secs(1));
        }
    }

    fn waiting_status(&mut self, ctx: &Context) {
        let key = "Waiting for the new file to finish writing ...";
        self.status = key.to_string();
        if self.auto_open {
            self.auto_status = key.to_string();
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(500));
    }

    fn start_continuous_file(&mut self, f: String, ctx: &Context) {
        self.processed.insert(f.clone());
        match self.build_options(false) {
            Ok(o) => {
                self.status = format!("About to process 1 file: {}", f);
                if self.auto_open {
                    self.auto_status = "About to process 1 file".to_string();
                    self.auto_done += 1;
                }
                self.start_files(vec![f], o, ctx, true);
            }
            Err(e) => self.popup = Some(format!("Error: {}", e)),
        }
    }

    fn save_config(&mut self) {
        let shift: Vec<i64> = pipeline::parse_shift_spec(&self.pixel_text).unwrap_or_else(|_| vec![0]);
        let fixed_width: Value = self
            .fixed_width_text
            .trim()
            .parse::<i64>()
            .map(|n| json!(n))
            .unwrap_or(Value::Null);
        let mut c = self.cfg.clone();
        c.insert("language".into(), json!(self.langs[self.lang_idx].name));
        c.insert("shift".into(), json!(shift));
        c.insert("flag_display".into(), json!(self.show_graphics));
        c.insert("ratio_fixe".into(), Value::Null);
        c.insert("slant_fix".into(), Value::Null);
        c.insert("save_fit".into(), json!(self.save_fit));
        c.insert("clahe_only".into(), json!(self.clahe_only));
        c.insert("protus_only".into(), json!(self.protus_only));
        c.insert("disk_display".into(), json!(true));
        c.insert("delta_radius".into(), json!(self.protus_text.trim().parse::<i64>().unwrap_or(0)));
        c.insert("crop_width_square".into(), json!(self.crop_square));
        c.insert("transversalium".into(), json!(self.transversalium));
        c.insert("stubborn_transversalium".into(), json!(self.stubborn));
        c.insert("trans_strength".into(), json!((self.trans_strength * 100.0) as i64 + 1));
        c.insert("img_rotate".into(), json!(self.rotate as i64));
        c.insert("flip_x".into(), json!(self.mirror_x));
        c.insert("fixed_width".into(), fixed_width);
        c.insert("output_dir".into(), json!(self.output_text.trim()));
        c.insert("workDir".into(), json!(self.work_dir));
        c.insert("input_dir".into(), json!(self.folder_text.trim()));
        c.insert(
            "selected_mode".into(),
            json!(if self.mode == Mode::File { "File input mode" } else { "Folder input mode" }),
        );
        c.insert("continuous_detect_mode".into(), json!(self.continuous));
        c.insert("ellipse_fit_shift".into(), json!(self.ellipse_text.trim().parse::<i64>().unwrap_or(10)));
        c.insert("de-vignette".into(), json!(self.de_vignette));
        if !c.contains_key("dispersion") {
            c.insert("dispersion".into(), json!(0.05));
        }
        if !c.contains_key("specDir") {
            c.insert("specDir".into(), json!(""));
        }
        if !c.contains_key("tempo") {
            c.insert("tempo".into(), json!(30000));
        }
        resources::save_config(&Value::Object(c));
    }

    fn ok_clicked(&mut self, ctx: &Context) {
        let files = if self.mode == Mode::File {
            match self.build_options(false) {
                Ok(_) => {}
                Err(e) => {
                    self.popup = Some(e);
                    return;
                }
            }
            self.file_list()
        } else {
            let folder = self.folder_text.trim().to_string();
            if folder.is_empty() {
                self.popup = Some(self.labels().no_folder);
                return;
            }
            if !Path::new(&folder).is_dir() {
                self.popup = Some("ERROR opening folder :".to_string() + &folder);
                return;
            }
            if self.continuous {
                self.processed.clear();
                self.auto_pending = None;
                self.auto_stop = false;
                self.auto_open = true;
                self.auto_done = 0;
                self.auto_status = "Looking for files ...".to_string();
                self.auto_last = String::new();
                self.auto_tex = None;
                self.auto_title = Path::new(&folder)
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| folder.clone());
                self.auto_last_scan = std::time::Instant::now();
                match self.build_options(false) {
                    Ok(_) => self.scan_continuous(ctx),
                    Err(e) => self.popup = Some(e),
                }
                self.save_config();
                return;
            }
            scan_folder_all(&folder, &self.processed)
        };
        if files.is_empty() {
            self.popup = Some(if self.mode == Mode::File {
                self.labels().no_file
            } else {
                "number of files todo: 0".to_string()
            });
            return;
        }
        let opts = match self.build_options(false) {
            Ok(o) => o,
            Err(e) => {
                self.popup = Some(e);
                return;
            }
        };
        if self.mode == Mode::File {
            if let Some(first) = files.first() {
                self.work_dir = Path::new(first)
                    .parent()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
            }
        }
        // Starting an explicit file/folder batch ends any continuous session.
        self.auto_open = false;
        self.auto_tex = None;
        self.auto_pending = None;
        self.save_config();
        self.start_files(files, opts, ctx, false);
    }

    fn open_output_folder(&mut self) {
        let mut p = self.output_text.trim().to_string();
        if p.is_empty() {
            p = if self.mode == Mode::File {
                self.work_dir.clone()
            } else {
                self.folder_text.trim().to_string()
            };
        }
        if p.is_empty() || !Path::new(&p).is_dir() {
            self.popup = Some(self.labels().no_folder);
            return;
        }
        let _ = std::process::Command::new("explorer").arg(&p).spawn();
    }

    fn ui_main(&mut self, ctx: &Context) {
        let l = self.labels();
        let running = self.job.is_some();
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&l.title).size(16.0).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let prev = self.lang_idx;
                        egui::ComboBox::from_id_source("lang_input")
                            .selected_text(self.langs[self.lang_idx].name.clone())
                            .width(110.0)
                            .show_ui(ui, |ui| {
                                for (i, lang) in self.langs.iter().enumerate() {
                                    ui.selectable_value(&mut self.lang_idx, i, &lang.name);
                                }
                            });
                        if self.lang_idx != prev {
                            self.load_flag(ctx);
                        }
                        if let Some(t) = &self.flag_tex {
                            let sz = t.size();
                            let w = 32.0_f32;
                            let h = (sz[1] as f32 * w / sz[0] as f32).max(1.0);
                            ui.add(egui::Image::new((t.id(), egui::vec2(w, h))));
                        }
                    });
                });
                ui.add_space(6.0);
                ui.separator();
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    if ui.selectable_label(self.mode == Mode::File, &l.file_mode).clicked() {
                        self.mode = Mode::File;
                    }
                    if ui.selectable_label(self.mode == Mode::Folder, &l.folder_mode).clicked() {
                        self.mode = Mode::Folder;
                    }
                });
                ui.add_space(4.0);
                match self.mode {
                    Mode::File => {
                        ui.horizontal(|ui| {
                            ui.add_sized([70.0, 20.0], egui::Label::new(&l.files));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(&l.choose_files).clicked() {
                                    if let Some(files) = rfd::FileDialog::new()
                                        .add_filter("Video Files (AVI or SER)", &["ser", "avi"])
                                        .pick_files()
                                    {
                                        self.file_text = files
                                            .iter()
                                            .map(|p| p.to_string_lossy().into_owned())
                                            .collect::<Vec<_>>()
                                            .join(";");
                                    }
                                }
                                ui.add_sized(
                                    [ui.available_width(), 22.0],
                                    egui::TextEdit::singleline(&mut self.file_text),
                                );
                            });
                        });
                    }
                    Mode::Folder => {
                        ui.horizontal(|ui| {
                            ui.add_sized([70.0, 20.0], egui::Label::new(&l.folder));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(&l.choose_input_folder).clicked() {
                                    if let Some(p) = rfd::FileDialog::new().pick_folder() {
                                        self.folder_text = p.to_string_lossy().into_owned();
                                    }
                                }
                                ui.add_sized(
                                    [ui.available_width(), 22.0],
                                    egui::TextEdit::singleline(&mut self.folder_text),
                                );
                            });
                        });
                        ui.checkbox(&mut self.continuous, &l.continuous);
                    }
                }
                ui.add_space(8.0);
                ui.label(&l.out_label);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(&l.choose_out_folder).clicked() {
                            if let Some(p) = rfd::FileDialog::new().pick_folder() {
                                self.output_text = p.to_string_lossy().into_owned();
                            }
                        }
                        ui.add_sized(
                            [ui.available_width(), 22.0],
                            egui::TextEdit::singleline(&mut self.output_text),
                        );
                    });
                });
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.show_graphics, &l.show_graphics);
                    ui.add_space(30.0);
                    ui.checkbox(&mut self.save_fit, &l.save_fits);
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.clahe_only, &l.clahe_only);
                    ui.add_space(30.0);
                    ui.checkbox(&mut self.protus_only, &l.protus_only);
                });
                ui.checkbox(&mut self.crop_square, &l.crop_square);
                ui.horizontal(|ui| {
                    ui.add_sized([250.0, 20.0], egui::Label::new(&l.fixed_width));
                    ui.add_sized([80.0, 22.0], egui::TextEdit::singleline(&mut self.fixed_width_text));
                });
                ui.checkbox(&mut self.mirror_x, &l.mirror_x);
                ui.label(&l.rotate);
                ui.add(
                    egui::Slider::new(&mut self.rotate, 0.0..=270.0)
                        .step_by(90.0)
                        .fixed_decimals(0),
                );
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.transversalium, &l.transversalium);
                    ui.add_space(20.0);
                    if self.transversalium {
                        ui.checkbox(&mut self.stubborn, &l.stubborn);
                    }
                    ui.add_space(20.0);
                    ui.checkbox(&mut self.de_vignette, &l.de_vignette);
                });
                if self.transversalium {
                    ui.label(&l.trans_strength);
                    ui.add(
                        egui::Slider::new(&mut self.trans_strength, 0.25..=7.0)
                            .step_by(0.25)
                            .fixed_decimals(2),
                    );
                }
                ui.horizontal(|ui| {
                    ui.add_sized([250.0, 20.0], egui::Label::new(&l.ratio));
                    ui.add_sized([80.0, 22.0], egui::TextEdit::singleline(&mut self.ratio_text));
                });
                ui.horizontal(|ui| {
                    ui.add_sized([250.0, 20.0], egui::Label::new(&l.tilt));
                    ui.add_sized([80.0, 22.0], egui::TextEdit::singleline(&mut self.tilt_text));
                });
                ui.horizontal(|ui| {
                    ui.add_sized([250.0, 20.0], egui::Label::new(&l.pixel_offset));
                    ui.add_sized([80.0, 22.0], egui::TextEdit::singleline(&mut self.pixel_text))
                        .on_hover_text(&l.pixel_tip);
                    ui.add_space(30.0);
                    if ui.button(&l.pixel_live).clicked() {
                        let env = spectral::Env {
                            output_dir: self.output_text.trim().to_string(),
                            ratio_fixe: self.ratio_text.trim().parse::<f64>().ok(),
                            slant_fix: self.tilt_text.trim().parse::<f64>().ok(),
                            ellipse_fit_shift: self.ellipse_text.trim().parse::<i64>().unwrap_or(10),
                            flip_x: self.mirror_x,
                            de_vignette: self.de_vignette,
                            dispersion: self
                                .cfg
                                .get("dispersion")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.05),
                            spec_dir: resources::cfg_str(&self.cfg, "specDir", ""),
                            transversalium: self.transversalium,
                            trans_strength: (self.trans_strength * 100.0) as i64 + 1,
                            stubborn: self.stubborn,
                            img_rotate: self.rotate as i64,
                            delta_radius: self.protus_text.trim().parse::<i64>().unwrap_or(0),
                            fixed_width: self.fixed_width_text.trim().parse::<i64>().ok(),
                            crop_square: self.crop_square,
                        };
                        self.spectral = Some(spectral::Spectral::new(
                            env,
                            self.lang_idx,
                            self.langs[self.lang_idx].dict.clone(),
                        ));
                    }
                });
                ui.horizontal(|ui| {
                    ui.add_sized([250.0, 20.0], egui::Label::new(&l.protus_adj));
                    ui.add_sized([80.0, 22.0], egui::TextEdit::singleline(&mut self.protus_text))
                        .on_hover_text(&l.protus_tip);
                });
                ui.horizontal(|ui| {
                    ui.add_sized([250.0, 20.0], egui::Label::new(&l.ellipse_shift));
                    ui.add_sized([80.0, 22.0], egui::TextEdit::singleline(&mut self.ellipse_text))
                        .on_hover_text(&l.ellipse_tip);
                });
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.add_enabled(!running, egui::Button::new(self.tr("OK"))).clicked() {
                        self.ok_clicked(ctx);
                    }
                    if ui.button(self.tr("Cancel")).clicked() {
                        self.job = None;
                        self.status = "Cancelled.".to_string();
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(&l.open_out_folder).clicked() {
                            self.open_output_folder();
                        }
                    });
                });
                ui.add_space(6.0);
                if running || self.continuous {
                    if let Some(job) = &self.job {
                        let done = job.done.load(Ordering::Relaxed);
                        let frac = if job.total > 0 { done as f32 / job.total as f32 } else { 0.0 };
                        ui.add(egui::ProgressBar::new(frac).text(format!("{}/{}", done, job.total)));
                    }
                    if self.continuous {
                        ui.horizontal(|ui| {
                            if ui.button(self.tr("Stop")).clicked() {
                                self.auto_stop = true;
                                if self.auto_open {
                                    self.auto_status = "WILL STOP AFTER PROCESSING CURRENT BATCH OF 1 FILE(S)".to_string();
                                }
                            }
                        });
                    }
                }
                ui.label(&self.status);
            });
        });
    }

    fn ui_popup(&mut self, ctx: &Context) {
        if let Some(msg) = self.popup.clone() {
            let mut open = true;
            egui::Window::new("SHG")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.label(msg);
                    ui.add_space(6.0);
                    if ui.button(self.tr("OK")).clicked() {
                        self.popup = None;
                    }
                });
            if !open {
                self.popup = None;
            }
        }
    }

    fn ui_preview(&mut self, ctx: &Context) {
        // "Show graphics" off: no preview windows (and unchecking closes the ones open).
        if !self.show_graphics {
            self.preview_mean_open = false;
            self.preview_sun_open = false;
            self.preview = None;
            return;
        }
        let Some(pv) = &self.preview else { return };
        if self.preview_mean_open {
            let mut close = false;
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("ser_mean_window"),
                egui::ViewportBuilder::default()
                    .with_title("Ser mean")
                    .with_inner_size([620.0, 700.0])
                    .with_min_inner_size([320.0, 260.0])
                    .with_resizable(true),
                |ctx, _class| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        ui.label(&pv.file);
                        if let Some(t) = &pv.mean_tex {
                            // Fill the whole window (matplotlib aspect='auto'): SER frames are
                            // a wide, flat spectral strip, so fitting the aspect would waste most
                            // of the window.
                            let avail = ui.available_size();
                            let size = egui::vec2(avail.x.max(20.0), avail.y.max(20.0));
                            let (rect, _resp) = ui.allocate_exact_size(size, egui::Sense::hover());
                            ui.painter().image(
                                t.id(),
                                rect,
                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                egui::Color32::WHITE,
                            );
                            // polynomial fit (blue)
                            let mut pts = Vec::with_capacity(pv.curve.len());
                            for (i, &c) in pv.curve.iter().enumerate() {
                                let x = rect.left()
                                    + ((c / (pv.mean.w as f64 - 1.0).max(1.0)) as f32) * rect.width();
                                let y = rect.top()
                                    + ((i as f64 / (pv.mean.h as f64 - 1.0).max(1.0)) as f32)
                                        * rect.height();
                                pts.push(egui::pos2(x, y));
                            }
                            ui.painter().add(egui::Shape::line(
                                pts,
                                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(80, 140, 255)),
                            ));
                        }
                    });
                    if ctx.input(|i| i.viewport().close_requested()) {
                        close = true;
                    }
                },
            );
            if close {
                self.preview_mean_open = false;
            }
        }
        if self.preview_sun_open {
            let mut close = false;
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("sun_images_window"),
                egui::ViewportBuilder::default()
                    .with_title("Sun images")
                    .with_inner_size([1040.0, 430.0])
                    .with_min_inner_size([320.0, 200.0])
                    .with_resizable(true),
                |ctx, _class| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        if let Some(tex) = &pv.sun_tex {
                            let names = ["clahe", "high contrast", "protus"];
                            let imgs = [&pv.pv.clahe, &pv.pv.high_contrast, &pv.pv.protus];
                            let avail = ui.available_size();
                            let cell_w = ((avail.x - 16.0) / 3.0).max(40.0);
                            ui.horizontal(|ui| {
                                for i in 0..3 {
                                    let (ih, iw) = (imgs[i].h as f32, imgs[i].w as f32);
                                    let scale = ((cell_w - 8.0) / iw)
                                        .min((avail.y - 24.0) / ih)
                                        .max(0.05);
                                    let size = egui::vec2(iw * scale, ih * scale);
                                    ui.vertical(|ui| {
                                        ui.label(names[i]);
                                        ui.add(egui::Image::new((tex[i].id(), size)));
                                    });
                                }
                            });
                        }
                    });
                    if ctx.input(|i| i.viewport().close_requested()) {
                        close = true;
                    }
                },
            );
            if close {
                self.preview_sun_open = false;
            }
        }
    }

    /// Stand-alone progress meter shown while a multi-file batch runs
    /// (mirrors the PySimpleGUI one_line_progress_meter of the Python version).
    fn ui_progress(&mut self, ctx: &Context) {
        let Some(job) = self.job.as_ref() else { return };
        if job.total <= 1 || self.progress_hidden {
            return;
        }
        let total = job.total;
        let par = job.par;
        let done = job.done.load(Ordering::Relaxed);
        let frac = done as f32 / total as f32;
        let current = self.current_file.clone();
        let title = format!("{}-{}", par, self.tr("Process Task Progress"));
        let completed = format!("{} {}/{} {}", self.tr("Completed"), done, total, self.tr("tasks"));
        let waiting = self.tr("Waiting for tasks...");
        let name = Path::new(&current)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| current.clone());
        let processing = if current.is_empty() {
            waiting
        } else {
            format!("{}: {}", self.tr("Processing"), name)
        };
        let mut close = false;
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("shg_process_progress"),
            egui::ViewportBuilder::default()
                .with_title(title)
                .with_inner_size([440.0, 120.0])
                .with_min_inner_size([340.0, 105.0])
                .with_always_on_top()
                .with_resizable(false),
            |ctx, _class| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(&completed).size(15.0));
                    ui.add(egui::ProgressBar::new(frac).text(format!("{}/{}", done, total)));
                    ui.add_space(4.0);
                    ui.label(&processing);
                });
                if ctx.input(|i| i.viewport().close_requested()) {
                    close = true;
                }
            },
        );
        if close {
            self.progress_hidden = true;
        }
    }

    /// Continuous-detect window (mirrors the Python "Auto processing" window):
    /// shows the processed count, the current status, the last processed image
    /// and its path, refreshed until the user presses Stop.
    fn ui_auto(&mut self, ctx: &Context) {
        if !self.auto_open {
            return;
        }
        let title = self.auto_title.clone();
        let header = self.tr("Auto processing of SHG video files");
        let stop_label = self.tr("Stop");
        let count_text = format!("{}: {}", self.tr("Number of files processed"), self.auto_done);
        let status = self.tr(&self.auto_status);
        let last = if self.auto_last.is_empty() {
            format!("{}: {}", self.tr("Last"), self.tr("None"))
        } else {
            format!("{}: {}", self.tr("Last"), self.auto_last)
        };
        let tex = self.auto_tex.clone();
        let mut stop = false;
        let mut close = false;
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("shg_auto_processing"),
            egui::ViewportBuilder::default()
                .with_title(title)
                .with_inner_size([640.0, 700.0])
                .with_min_inner_size([420.0, 420.0])
                .with_resizable(true),
            |ctx, _class| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(&header).size(15.0).strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add_enabled(!stop, egui::Button::new(&stop_label))
                                .clicked()
                            {
                                stop = true;
                            }
                        });
                    });
                    ui.horizontal(|ui| {
                        ui.label(&count_text);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(&status);
                        });
                    });
                    ui.separator();
                    let avail = ui.available_size();
                    if let Some(t) = &tex {
                        let (iw, ih) = (t.size()[0] as f32, t.size()[1] as f32);
                        let scale = (avail.x / iw).min((avail.y - 26.0) / ih).max(0.05);
                        ui.add(egui::Image::new((t.id(), egui::vec2(iw * scale, ih * scale))));
                    } else {
                        ui.add_space((avail.y - 26.0).max(0.0));
                    }
                    ui.label(&last);
                });
                if ctx.input(|i| i.viewport().close_requested()) {
                    close = true;
                }
            },
        );
        if stop {
            self.auto_stop = true;
            self.auto_status = "WILL STOP AFTER PROCESSING CURRENT BATCH OF 1 FILE(S)".to_string();
            if self.job.is_none() {
                self.auto_open = false;
                self.auto_tex = None;
                self.auto_pending = None;
            }
        }
        if close {
            self.auto_open = false;
            self.auto_stop = true;
            self.auto_tex = None;
            self.auto_pending = None;
        }
    }

    fn ui_spectral(&mut self, ctx: &Context) {
        let idx = self.lang_idx;
        let Some(sp) = &mut self.spectral else { return };
        if !sp.open {
            return;
        }
        sp.sync_lang(idx, &self.langs[idx].dict);
        let action = sp.ui(ctx);
        match action {
            spectral::Action::None => {}
            spectral::Action::Close(shift) => {
                if let Some(s) = shift {
                    self.pixel_text = s.to_string();
                }
                self.spectral = None;
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        if !self.sized {
            self.sized = true;
            // Shrink the default window to fit the monitor when display scaling
            // would otherwise push it past the screen (e.g. 200% DPI).
            if let Some(ms) = ctx.input(|i| i.viewport().monitor_size) {
                if ms.x > 0.0 && ms.y > 0.0 {
                    let w = 900.0_f32.min(ms.x * 0.94);
                    let h = 780.0_f32.min(ms.y * 0.92);
                    if w < 900.0 || h < 780.0 {
                        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(w, h)));
                    }
                }
            }
        }
        self.poll_job(ctx);
        if self.job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        if self.continuous
            && self.auto_open
            && self.mode == Mode::Folder
            && self.job.is_none()
            && !self.auto_stop
            && (self.auto_pending.is_some()
                || self.auto_last_scan.elapsed() >= std::time::Duration::from_secs(1))
        {
            self.auto_last_scan = std::time::Instant::now();
            self.scan_continuous(ctx);
        }
        self.ui_spectral(ctx);
        self.ui_main(ctx);
        self.ui_preview(ctx);
        self.ui_progress(ctx);
        self.ui_auto(ctx);
        self.ui_popup(ctx);
    }
}

fn install_fonts(ctx: &Context) {
    let mut fonts = egui::FontDefinitions::default();
    // Fallback chain, tried glyph-by-glyph: CJK -> Korean -> Latin/Cyrillic.
    let mut paths: Vec<&str> = Vec::new();
    for p in [
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\Deng.ttf",
        "C:\\Windows\\Fonts\\simkai.ttf",
        "C:\\Windows\\Fonts\\msyh.ttc",
    ] {
        if std::fs::metadata(p).is_ok() {
            paths.push(p);
            break;
        }
    }
    // Hangul (Korean).
    paths.push("C:\\Windows\\Fonts\\malgun.ttf");
    // Segoe UI covers Latin and full Cyrillic, including Ukrainian і/ї/є.
    paths.push("C:\\Windows\\Fonts\\segoeui.ttf");
    for (i, path) in paths.iter().enumerate() {
        let Ok(bytes) = std::fs::read(path) else { continue };
        let name = format!("system{i}");
        fonts.font_data.insert(name.clone(), egui::FontData::from_owned(bytes));
        for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            if let Some(f) = fonts.families.get_mut(&fam) {
                f.push(name.clone());
            }
        }
    }
    ctx.set_fonts(fonts);
}

fn scan_folder_all(dir: &str, processed: &HashSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            let ext = p.extension().map(|x| x.to_string_lossy().to_uppercase()).unwrap_or_default();
            if ext == "SER" {
                let s = p.to_string_lossy().into_owned();
                if !processed.contains(&s) {
                    out.push(s);
                }
            }
        }
    }
    out.sort();
    out
}

fn scan_folder(dir: &str, processed: &HashSet<String>) -> Vec<String> {
    scan_folder_all(dir, processed)
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown error".to_string()
    }
}

fn window_icon() -> Option<egui::IconData> {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/icon_256.png"));
    let mut reader = png::Decoder::new(&bytes[..]).read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.color_type != png::ColorType::Rgba {
        return None;
    }
    Some(egui::IconData {
        rgba: buf[..info.buffer_size()].to_vec(),
        width: info.width,
        height: info.height,
    })
}

pub fn run() -> Result<(), eframe::Error> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([900.0, 780.0])
        .with_min_inner_size([620.0, 420.0])
        .with_title("SHG Version 4.3.9 (Mod with Rust by lzq)");
    if let Some(icon) = window_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }
    let opts = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "SHG Version 4.3.9 (Mod with Rust by lzq)",
        opts,
        Box::new(|cc| Ok(Box::new(App::new(cc)) as Box<dyn eframe::App>)),
    )
}
